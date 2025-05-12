from typing import Any, Tuple

import torch
import torch.distributed as dist
from fastvideo.utils.communications import (all_gather, all_to_all,
                                            all_to_all_4D, broadcast)
from fastvideo.utils.parallel_states import nccl_info
from torch import Tensor


def prepare_sequence_parallel_data(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask):
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )

    def prepare(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask):
        # print(f'before all to all {encoder_hidden_states.shape}')
        hidden_states = all_to_all(hidden_states, scatter_dim=1, gather_dim=0)
        encoder_hidden_states = all_to_all(encoder_hidden_states, scatter_dim=1, gather_dim=0)
        attention_mask = all_to_all(attention_mask, scatter_dim=1, gather_dim=0)
        encoder_attention_mask = all_to_all(encoder_attention_mask, scatter_dim=1, gather_dim=0)
        dist.barrier()
        # print(f'after all to all {encoder_hidden_states.shape}')
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )

    sp_size = nccl_info.sp_size
    # assert frame % sp_size == 0, "frame should be a multiple of sp_size"
    # print(f'shapes 1 rank:{dist.get_rank()}:', hidden_states.shape, encoder_hidden_states.shape, attention_mask.shape, encoder_attention_mask.shape, '\n')
    # print(f'sum_latent 1 rank:{dist.get_rank()}:', hidden_states.sum(), '\n')

    (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
    ) = prepare(
        hidden_states.repeat(1, sp_size, 1, 1, 1),
        encoder_hidden_states.repeat(1, sp_size, 1),
        attention_mask.repeat(1, sp_size, 1, 1),
        encoder_attention_mask.repeat(1, sp_size),
    )

    # print(f'shapes 2 rank:{dist.get_rank()}:', hidden_states.shape, encoder_hidden_states.shape, attention_mask.shape, encoder_attention_mask.shape)
    # print(f'sum_latent 2 rank:{dist.get_rank()}:', hidden_states.sum())

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask


def sp_parallel_dataloader_wrapper(dataloader, device, train_batch_size, sp_size, train_sp_batch_size):
    while True:
        for data_item in dataloader:
            latents, cond, attn_mask, cond_mask = data_item
            latents = latents.to(device)
            cond = cond.to(device)
            attn_mask = attn_mask.to(device)
            cond_mask = cond_mask.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, cond, attn_mask, cond_mask
            else:
                print("latents shape: ", latents.shape, " sp size: ", sp_size)
                latents, cond, attn_mask, cond_mask = prepare_sequence_parallel_data(
                    latents, cond, attn_mask, cond_mask)
                print("after prepare, latents shape: ", latents.shape, " sp size: ", sp_size)
                assert (train_batch_size * sp_size >=
                        train_sp_batch_size), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size // train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    encoder_hidden_states = cond[st_idx:ed_idx]
                    attention_mask = attn_mask[st_idx:ed_idx]
                    encoder_attention_mask = cond_mask[st_idx:ed_idx]
                    yield (
                        latents[st_idx:ed_idx],
                        encoder_hidden_states,
                        attention_mask,
                        encoder_attention_mask,
                    )
