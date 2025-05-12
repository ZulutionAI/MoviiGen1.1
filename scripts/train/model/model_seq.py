# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.amp as amp
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from fastvideo.models.wan.modules.attention import flash_attention
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import (get_sequence_parallel_state,
                                             nccl_info)
from torch.nn import functional as F
from xfuser.core.distributed import (get_sequence_parallel_rank,
                                     get_sequence_parallel_world_size,
                                     get_sp_group)
from xfuser.core.long_ctx_attention import xFuserLongContextAttention

__all__ = ['WanModel']


def pad_freqs(original_tensor, target_len):
    seq_len, s1, s2 = original_tensor.shape
    pad_size = target_len - seq_len
    padding_tensor = torch.ones(
        pad_size,
        s1,
        s2,
        dtype=original_tensor.dtype,
        device=original_tensor.device)
    padded_tensor = torch.cat([original_tensor, padding_tensor], dim=0)
    return padded_tensor


@amp.autocast("cuda", enabled=False)
def rope_apply_dist(x, grid_sizes, freqs):
    """
    x:          [B, L, N, C].
    grid_sizes: [B, 3].
    freqs:      [M, C // 2].
    """
    s, n, c = x.size(1), x.size(2), x.size(3) // 2
    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :s].to(torch.float64).reshape(
            s, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        # sp_size = get_sequence_parallel_world_size()
        # sp_rank = get_sequence_parallel_rank()
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        freqs_i = pad_freqs(freqs_i, s * sp_size)
        s_per_rank = s
        freqs_i_rank = freqs_i[(sp_rank * s_per_rank):((sp_rank + 1) *
                                                       s_per_rank), :, :]
        x_i = torch.view_as_real(x_i * freqs_i_rank).flatten(2)
        x_i = torch.cat([x_i, x[i, s:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            # freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


def shrink_head(encoder_state, dim):
    local_heads = encoder_state.shape[dim] // nccl_info.sp_size
    return encoder_state.narrow(dim, nccl_info.rank_within_group * local_heads, local_heads)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        rank = dist.get_rank()
        # if rank == 0:
        #     print(f'x_in_attn: {x.sum()}')

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        # if rank == 0:
        #     print(f'q_in_attn: {q.sum()}')

        if get_sequence_parallel_state():
            q = rope_apply_dist(q, grid_sizes, freqs)
            k = rope_apply_dist(k, grid_sizes, freqs)
        else:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

        # if rank == 0:
        #     print(f'q_roped_attn: {q.sum()}')

        if get_sequence_parallel_state():
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)

        # if get_sequence_parallel_state():
        # # batch_size, seq_len, attn_heads, head_dim
        #     q = all_gather(q, dim=1).contiguous()
        #     k = all_gather(k, dim=1).contiguous()

        # q = rope_apply(q, grid_sizes, freqs)
        # k = rope_apply(k, grid_sizes, freqs)

        # if get_sequence_parallel_state():
        #     q = shrink_head(q, dim=2)
        #     k = shrink_head(k, dim=2)
        #     v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)

        x = flash_attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        if get_sequence_parallel_state():
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)

        # if rank == 0:
            # print(f'x_attned: {x.sum()}, {x.shape}')

        # output
        x = x.flatten(2)
        x = self.o(x)
        # if rank == 0:
        # print(f'x_attn_out: {x.sum()}, {x.shape}')
        # print('--------')
        return x

class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        # print(f'rank {dist.get_rank()}, getin cross attn {x.sum()}, ctx: {context.sum()}')

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        if get_sequence_parallel_state():
            # batch_size, seq_len, attn_heads, head_dim
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = shrink_head(k, dim=2)
            v = shrink_head(v, dim=2)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)
        if get_sequence_parallel_state():
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        img_x = flash_attention(q, k_img, v_img, k_lens=None)
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        x = x + img_x
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0], seq_lens, grid_sizes,
            freqs)
        with amp.autocast("cuda", dtype=torch.float32):
            x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast("cuda", dtype=torch.float32):
                x = x + y * e[5]
            return x

        # dist.barrier()
        # print(f'rank {dist.get_rank()}, before cross attn {x.sum()}, ctx: {context.sum()}')
        # dist.barrier()
        x = cross_attn_ffn(x, context, context_lens, e)
        # dist.barrier()
        # print(f'rank {dist.get_rank()}, after cross attn {x.sum()}')
        # dist.barrier()

        # dist.barrier()
        # if dist.get_rank() == 0:
        #     print(f'attn block out {x.sum()}')
        # dist.barrier()

        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        assert e.dtype == torch.float32
        with amp.autocast("cuda", dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ],
            dim=1,
        )

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        batch_context=None,
        context_mask=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            batch_context : [b,l,c]
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        rank = dist.get_rank()
        # if rank == 0:
        #     print('inside dit forward:')
        #     print('x: ', x[0][0][0][0][:10])
        #     # print('context', batch_context, context)
        #     # print('context: ', batch_context[0][0][:10])
        #     print('t: ', t)
        dist.barrier()

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None

        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        # if get_sequence_parallel_state():
        #     print(f"device{device} x {x[0].shape}")
        #     new_x = []
        #     for u in x:
        #         u = all_gather(u,dim=1).contiguous()
        #         print(f"device{device} u after all gather{u.shape}")
        #         u = self.patch_embedding(u.unsqueeze(0))
        #         one_seq_len = u.size(2) // nccl_info.sp_size
        #         new_x.append(u.narrow(2, nccl_info.rank_within_group * one_seq_len, one_seq_len))
        #     print(f"device{device} new_x {new_x[0].shape}")
        #     x = new_x
        # else:

        original_shapes = [u.shape[1:] for u in x]  # Store F, H, W
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # print(f'rank {rank} embeds x {x[0].shape}, {x[0].sum()}') #[1, 5120, 20, 30, 52] [B, C, F, H, W]
        dist.barrier()
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        # if get_sequence_parallel_state():
        #     org_grid_sizes = grid_sizes.clone()
        #     grid_sizes[:,0] *= nccl_info.sp_size
        x = [u.flatten(2).transpose(1, 2) for u in x]
        # print(f'rank {rank} flattened x {x[0].shape}, {x[0].sum()}')

        dist.barrier()
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)

        assert seq_lens.max() <= seq_len, print(f"max seq_lens {seq_lens.max()}")

        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
        ])

        # time embeddings
        with amp.autocast("cuda", dtype=torch.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = self.time_projection(e).unflatten(1, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        ############
        # context = batch_context
        ############
        # print(f'rank {rank} ctx {batch_context[0].shape}, {batch_context[0].sum()}')
        dist.barrier()
        context_lens = None
        if context is not None:
            print('Should not be here')
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
            dist.barrier()
        else:
            assert batch_context is not None
            context = self.text_embedding(batch_context)
        # print(f'rank {rank} embed ctx {context[0].shape}, {context.sum()}')

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
        )

        # i = 0
        # print(f'rank {rank}: attn x {x.shape}, {x.sum()}') #[1, 5120, 20, 30, 52] [B, C, F, H, W] -> [1, 31200, 5120] [B, HWF, C]
        dist.barrier()
        # for block in self.blocks:
        #     # print(f"device{device} block {i}")
        #     x = block(x, **kwargs)

        x = torch.chunk(x, nccl_info.sp_size, dim=1)[nccl_info.rank_within_group]

        # print(f'rank {rank}: chunked x {x.shape}, {x.sum()}')
        dist.barrier()
        for block in self.blocks:
            x = block(x, **kwargs)

            # if rank == 0:
            #     print(f'in loop x: {x.sum()}')
            # dist.barrier()
            # assert(False)
        dist.barrier()
        # Context Parallel

        ###
        # x_split = x.clone()
        # x_split = self.head(x_split, e)
        # grid_sizes_split = grid_sizes.clone()
        # grid_sizes_split[0] = grid_sizes_split[0] // nccl_info.sp_size
        # x_split = self.unpatchify(x_split, grid_sizes_split)
        # x_split = [u.float() for u in x_split]
        ###

        x = all_gather(x, dim=1).contiguous()
        # print(f'rank {rank}: gathered x {x.shape}, {x.sum()}')
        # dist.barrier()

        # head
        x = self.head(x, e)

        # if get_sequence_parallel_state():
        #     grid_sizes = org_grid_sizes

        # unpatchify
        #  x = self.unpatchify(x, grid_sizes, original_shapes=original_shapes)
        x = self.unpatchify(x, grid_sizes)

        # print(f'rank {rank}: out x {x[0].shape}, {x[0].sum()}')
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes, original_shapes=None):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for idx, (u, v) in enumerate(zip(x, grid_sizes.tolist())):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            if original_shapes is not None:
                original_H = original_shapes[idx][1]
                u = u[:, :, :original_H, :]
            out.append(u)

        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
