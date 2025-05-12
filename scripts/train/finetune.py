#!/bin/python3
# isort: skip_file
from scripts.dataset.latent_datasets import LatentDataset, latent_collate_function
from scripts.train.util.util import load_wan
from torch.utils.tensorboard import SummaryWriter
from scripts.train.util.math_util import cosine_optimal_transport
from scripts.train.model.model_seq import WanAttentionBlock as WanAttentionBlock_SEQ

from fastvideo.models.wan.configs import WAN_CONFIGS
from fastvideo.utils.parallel_states import initialize_sequence_parallel_state
from fastvideo.utils.fsdp_util import apply_fsdp_checkpointing, get_dit_fsdp_kwargs
from fastvideo.utils.communications import broadcast
from fastvideo.utils.checkpoint import resume_lora_optimizer, save_checkpoint, save_lora_checkpoint

from tqdm.auto import tqdm

from scripts.dataset.aspect_ratio_length_bucket_sampler import SPAwareAspectRatioLengthBucketDistributedSampler

from torch.utils.data import DataLoader
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from diffusers.utils import check_min_version
from diffusers.optimization import get_scheduler
from diffusers import FlowMatchEulerDiscreteScheduler
from accelerate.utils import set_seed
import torch.distributed as dist
import torch
from collections import deque
import time
import argparse
import math
import os
import sys
sys.path.insert(0, os.path.abspath("."))


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

SIGMA_MIN = 1e-5
DEBUG = False


def sp_aware_wrapper(dataloader, device, sp_size):
    """
    A simplified sequence parallel wrapper designed for SPAwareAspectRatioLengthBucketDistributedSampler

    Assumption: All GPUs within the same SP group have loaded identical video data (guaranteed by SPAware sampler)
    Purpose: Only responsible for splitting videos along the time dimension and distributing to different GPUs

    Args:
        dataloader: DataLoader using SPAwareAspectRatioLengthBucketDistributedSampler
        device: Current device
        sp_size: Sequence parallel group size
    """
    from fastvideo.utils.parallel_states import nccl_info

    # Get current GPU's rank within the SP group
    rank_in_group = nccl_info.rank_within_group

    while True:
        for data_item in dataloader:
            # Unpack data (still on CPU)
            latents, encoder_hidden_states, attention_mask, encoder_attention_mask = data_item

            # Get frame count (assuming latents shape is [B, C, T, H, W])
            frames = latents.shape[2]

            # For single-frame videos, just move to GPU and yield
            if frames == 1:
                yield (
                    latents.to(device),
                    encoder_hidden_states.to(device),
                    attention_mask.to(device),
                    encoder_attention_mask.to(device)
                )
                continue

            # Ensure frame count is divisible by SP group size
            assert frames % sp_size == 0, f"Frame count {frames} must be divisible by SP group size {sp_size}"

            # Calculate frames per GPU
            frames_per_gpu = frames // sp_size

            # Calculate frame range for current GPU
            start_idx = rank_in_group * frames_per_gpu
            end_idx = start_idx + frames_per_gpu

            # Split latents along time dimension while still on CPU
            latents_split = latents[:, :, start_idx:end_idx, :, :]

            # Adjust attention mask while still on CPU
            if attention_mask.ndim >= 3 and attention_mask.shape[2] == frames:
                # If mask has time dimension, split accordingly
                attention_mask_split = attention_mask[:, :, start_idx:end_idx, :]
            else:
                # Otherwise keep unchanged
                attention_mask_split = attention_mask

            # Now move the split data to GPU
            yield (
                latents_split.to(device),
                encoder_hidden_states.to(device),  # Text embeddings are small, move whole tensor
                attention_mask_split.to(device),
                encoder_attention_mask.to(device)  # Text mask is small, move whole tensor
            )


def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)


def reshard_fsdp(model):
    for m in FSDP.fsdp_modules(model):
        if m._has_params and m.sharding_strategy is not ShardingStrategy.NO_SHARD:
            torch.distributed.fsdp._runtime_utils._reshard(m, m._handle, True)


def get_sigmas(noise_scheduler, device, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item()
                    for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    generator,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(
            mean=logit_mean,
            std=logit_std,
            size=(batch_size, ),
            device="cpu",
            generator=generator,
        )
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2)**2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size, ), device="cpu", generator=generator)
    return u


def train_one_step(
    transformer,
    optimizer,
    lr_scheduler,
    loader,
    vae,
    noise_scheduler,
    noise_random_generator,
    gradient_accumulation_steps,
    sp_size,
    max_grad_norm,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    max_seq_len=32760,
    enable_timing=False,
):
    total_loss = 0.0
    optimizer.zero_grad()

    timing_stats = {}

    for _ in range(gradient_accumulation_steps):
        # Performance monitoring
        if enable_timing:
            start_load = time.time()

        # Get batch data
        latents, encoder_hidden_states, _, _ = next(loader)

        if enable_timing:
            torch.cuda.synchronize()
            load_time = time.time() - start_load
            timing_stats['data_loading'] = load_time
            start_prep = time.time()

        # Prepare inputs
        model_input = latents
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        if args.compute_ot:
            # compute OT pairings
            transport_cost, indices = cosine_optimal_transport(latents.reshape(bsz, -1), noise.reshape(bsz, -1))
            noise = noise[indices[1].view(-1)]

        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bsz,
            generator=noise_random_generator,
            logit_mean=logit_mean,
            logit_std=logit_std,
            mode_scale=mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

        # Only broadcast if necessary (optimization)
        if sp_size > 1:
            # Make sure that the timesteps and noise are the same across all SP processes
            broadcast(timesteps)
            broadcast(noise)

        # Add noise according to flow matching
        sigmas = get_sigmas(
            noise_scheduler,
            latents.device,
            timesteps,
            n_dim=latents.ndim,
            dtype=latents.dtype,
        )
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * model_input

        if enable_timing:
            torch.cuda.synchronize()
            prep_time = time.time() - start_prep
            timing_stats['prep'] = prep_time
            start_forward = time.time()

        # Forward pass with autocast for better performance
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = [noisy_model_input[i] for i in range(noisy_model_input.size(0))]
            model_preds = transformer(x, timesteps, None, max_seq_len,
                                      batch_context=encoder_hidden_states, context_mask=None)
            model_pred = model_preds[0]

        if enable_timing:
            torch.cuda.synchronize()
            forward_time = time.time() - start_forward
            timing_stats['forward'] = forward_time
            start_loss = time.time()

        # Calculate loss
        target = noise - latents
        # if target.shape[3] > model_pred.shape[2]:
        #     target = target[:, :, :, :model_pred.shape[2], :]

        loss = (torch.mean((model_pred.float() - target.float()) ** 2) / gradient_accumulation_steps)

        if enable_timing:
            torch.cuda.synchronize()
            loss_time = time.time() - start_loss
            timing_stats['loss'] = loss_time
            start_backward = time.time()

        # Backward pass
        if not DEBUG:
            loss.backward()

        if enable_timing:
            torch.cuda.synchronize()
            backward_time = time.time() - start_backward
            timing_stats['backward'] = backward_time

        # Collect loss statistics
        avg_loss = loss.detach().clone()
        dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        total_loss += avg_loss.item()

    # Gradient clipping
    if enable_timing:
        start_grad = time.time()

    grad_norm = transformer.clip_grad_norm_(max_grad_norm)

    if enable_timing:
        torch.cuda.synchronize()
        grad_clip_time = time.time() - start_grad
        timing_stats['grad_clip'] = grad_clip_time
        start_optim = time.time()

    # Optimizer step
    if not DEBUG:
        optimizer.step()
        lr_scheduler.step()

    if enable_timing:
        torch.cuda.synchronize()
        optim_time = time.time() - start_optim
        timing_stats['optim'] = optim_time

    # Only synchronize at the end of the step (optimization)
    dist.barrier()

    # Debug visualization (only on rank 0)
    if DEBUG and dist.get_rank() == 0:
        from wan.utils.utils import cache_video
        with torch.no_grad():
            for value, name in [(latents, "latents"), (model_pred, "model_pred"), (noisy_model_input-sigmas*model_pred, "onestep_denoised")]:
                save_video_path = f"{args.output_dir}/{name}_rank{dist.get_rank()}_timestep_{timesteps[0].item()}.mp4"
                videos = vae.decode([value.squeeze(0).float()])
                video = videos[0]
                cache_video(tensor=video[None], save_file=save_video_path,
                            fps=16, nrow=1, normalize=True, value_range=(-1, 1))

    dist.barrier()
    return total_loss, grad_norm.item(), timesteps.mean().item(), timing_stats


def main(args):
    # Enable TF32 for faster training
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Get distributed training info
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Setup device
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    dist.init_process_group("nccl", device_id=torch.device(f'cuda:{device}'))

    # Initialize SP
    assert world_size % args.sp_size == 0, f"World size ({world_size}) must be divisible by SP size ({args.sp_size})"
    initialize_sequence_parallel_state(args.sp_size)

    # Set seed for reproducibility (different per rank)
    if args.seed is not None:
        set_seed(args.seed + rank)
    noise_random_generator = None

    # Create output directories and summary writer
    if rank <= 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.logging_dir))
        os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    main_print(f"--> loading model from {args.ckpt_dir}")
    cfg = WAN_CONFIGS[args.task]

    transformer = load_wan(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        weight_path=None,
    )

    transformer.requires_grad_(True)
    torch.cuda.empty_cache()

    main_print(
        f"  Total training parameters = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e6} M"
    )

    # Initialize FSDP
    main_print(f"--> Initializing FSDP with sharding strategy: {args.fsdp_sharding_startegy}")
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(
        transformer,
        args.fsdp_sharding_startegy,
        args.use_lora,
        args.use_cpu_offload,
        args.master_weight_type,
        (WanAttentionBlock_SEQ,),
    )

    transformer = FSDP(transformer, **fsdp_kwargs)
    main_print("--> model loaded")

    # Setup VAE for debug visualization
    if DEBUG:
        from wan.modules.vae import WanVAE
        vae = WanVAE(vae_pth="/vepfs-zulution/models/Wan2.1-T2V-14B/Wan2.1_VAE.pth")
        autocast_type = torch.bfloat16
        vae.model = vae.model.to(device).to(autocast_type)
        vae.model.eval()
        vae.model.requires_grad_(False)
    else:
        vae = None

    # Apply gradient checkpointing if enabled
    if args.gradient_checkpointing:
        apply_fsdp_checkpointing(transformer, no_split_modules, args.selective_checkpointing)

    # Set model to train mode
    transformer.train()

    # Create noise scheduler
    if args.use_dynamic_shift:
        noise_scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=args.use_dynamic_shift)
    else:
        noise_scheduler = FlowMatchEulerDiscreteScheduler(shift=args.shift)

    # Create optimizer
    params_to_optimize = transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    # Load checkpoint if resuming
    init_steps = 0
    if args.resume_from_lora_checkpoint:
        transformer, optimizer, init_steps = resume_lora_optimizer(
            transformer, args.resume_from_lora_checkpoint, optimizer)
    main_print(f"optimizer: {optimizer}")

    # Create dataset
    train_dataset = LatentDataset(
        args.data_json_path,
        args.num_latent_t,
        cfg_rate=0.0,
        prompt_type=args.prompt_type,
    )

    sampler = SPAwareAspectRatioLengthBucketDistributedSampler(
        dataset_size=len(train_dataset),
        batch_size=args.train_batch_size,
        sp_size=args.sp_size,
        num_replicas=world_size,
        rank=rank,
        lengths=train_dataset.lengths,
        aspect_ratios=train_dataset.aspect_ratios if hasattr(train_dataset, 'aspect_ratios') else None,
        num_length_bins=8,
        drop_last=True,
        seed=args.seed if args.seed is not None else 42,
        verbose=(rank == 0),
    )

    # Create dataloader with the SP-aware sampler
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=sampler,  # Use batch_sampler instead of sampler + batch_size
        collate_fn=latent_collate_function,
        pin_memory=True,
        num_workers=args.dataloader_num_workers,
        persistent_workers=(args.dataloader_num_workers > 0),
    )

    for batch in train_dataloader:
        print("iter a  batch")

    # Calculate steps per epoch correctly
    # Effective parallel processes (accounting for SP groups)
    effective_processes = world_size // args.sp_size

    # Samples processed per step across all effective processes
    effective_batch_size = args.train_batch_size * effective_processes * args.gradient_accumulation_steps

    # Steps needed for one epoch (each unique sample)
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)

    # Validate and set training steps based on epochs or max_steps
    # 二选一：要么指定轮数，要么指定步数
    if args.max_train_steps is None and args.num_train_epochs is not None:
        # 如果只指定了轮数，计算步数
        args.max_train_steps = math.ceil(num_update_steps_per_epoch * args.num_train_epochs)
        main_print(f"设置 max_train_steps 为 {args.max_train_steps}，基于 {args.num_train_epochs} 轮训练")
    elif args.max_train_steps is not None and args.num_train_epochs is None:
        # 如果只指定了步数，计算轮数
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        main_print(f"设置 num_train_epochs 为 {args.num_train_epochs}，基于 {args.max_train_steps} 步训练")
    elif args.max_train_steps is None and args.num_train_epochs is None:
        # 如果都未指定，使用默认值：1轮
        args.num_train_epochs = 1
        args.max_train_steps = num_update_steps_per_epoch
        main_print(f"未指定轮数或步数。设置为1轮训练 ({args.max_train_steps} 步)")
    else:
        # 如果都指定了，使用步数为准，重新计算轮数
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        main_print(f"同时指定了轮数和步数。使用 {args.max_train_steps} 步作为标准 (约 {args.num_train_epochs} 轮)")

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * world_size,
        num_training_steps=args.max_train_steps * world_size,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        last_epoch=init_steps - 1,
    )

    # Log training info
    total_batch_size = effective_batch_size
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Effective parallel processes = {effective_processes}")
    main_print(f"  Total train batch size (w. parallelism, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(f"  Steps per epoch = {num_update_steps_per_epoch}")
    main_print(
        f"  Total training parameters per FSDP shard = {sum(p.numel() for p in transformer.parameters() if p.requires_grad) / 1e9} B")
    main_print(f"  Master weight dtype: {next(transformer.parameters()).dtype}")

    if args.sp_size > 1:
        main_print(f"  Using sequence parallelism with {args.sp_size} GPUs per group")
        main_print(f"  SP groups: {world_size // args.sp_size}")
        main_print(f"  Using SP aware wrapper for SPAwareAspectRatioLengthBucketDistributedSampler")

    dist.barrier()

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        assert NotImplementedError(
            "resume_from_checkpoint is not supported now.")
        # TODO

    # Create progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        disable=rank > 0,
    )

    # Use our custom SP aware wrapper for the dataloader when SP is enabled
    if args.sp_size > 1:
        loader = sp_aware_wrapper(
            train_dataloader,
            device,
            args.sp_size
        )
    else:
        # For single GPU or data parallel only, just use the raw dataloader
        loader = train_dataloader

    # For timing statistics
    step_times = deque(maxlen=100)
    enable_timing = args.profile_steps > 0

    # Calculate initial epoch for the sampler
    current_epoch = init_steps // num_update_steps_per_epoch

    # Set the initial epoch for the sampler
    if hasattr(sampler, 'set_epoch'):
        sampler.set_epoch(current_epoch)
        main_print(f"Setting initial epoch to {current_epoch}")

    # Skip steps for resuming training
    for i in range(init_steps):
        next(loader)

    # Main training loop
    for step in range(init_steps + 1, args.max_train_steps + 1):
        # Update epoch at epoch boundaries
        new_epoch = (step - 1) // num_update_steps_per_epoch
        if new_epoch > current_epoch:
            current_epoch = new_epoch
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(current_epoch)
                main_print(f"Starting epoch {current_epoch}")

        # Enable timing for designated profiling steps
        do_profile = enable_timing and (step <= args.profile_steps or step % 100 == 0)

        start_time = time.time()

        loss, grad_norm, avg_time_step, timing_stats = train_one_step(
            transformer,
            optimizer,
            lr_scheduler,
            loader,
            vae,
            noise_scheduler,
            noise_random_generator,
            args.gradient_accumulation_steps,
            args.sp_size,
            args.max_grad_norm,
            args.weighting_scheme,
            args.logit_mean,
            args.logit_std,
            args.mode_scale,
            args.max_seq_len,
            enable_timing=do_profile,
        )

        step_time = time.time() - start_time
        step_times.append(step_time)
        avg_step_time = sum(step_times) / len(step_times)

        # Log results
        if do_profile:
            profile_msg = " | ".join([f"{k}: {v:.2f}s" for k, v in timing_stats.items()])
            main_print(f"Profile step {step}: {profile_msg}\n")

        main_print(
            f"Step {step}/{args.max_train_steps} - Loss: {loss:.4f} - Step time: {step_time:.2f}s - Avg time: {avg_step_time:.2f}s"
        )

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss:.4f}",
            "step_time": f"{step_time:.2f}s",
            "grad_norm": f"{grad_norm:.3f}",
            "epoch": current_epoch,
            "avg_time": f"{avg_step_time:.2f}s",
        })
        progress_bar.update(1)

        # Log to tensorboard
        if rank <= 0:
            writer.add_scalar("loss", loss, step)
            writer.add_scalar("grad_norm", grad_norm, step)
            writer.add_scalar("epoch", current_epoch, step)
            writer.add_scalar("step_time", step_time, step)
            writer.add_scalar("avg_step_time", avg_step_time, step)
            bin_index = min(int(avg_time_step / 100), 10 - 1)
            writer.add_scalar(f"loss_bin_{bin_index}", loss, step)

            # Log timing stats if available
            if timing_stats:
                for k, v in timing_stats.items():
                    writer.add_scalar(f"time/{k}", v, step)

        # Save checkpoint at regular intervals
        if step % args.checkpointing_steps == 0:
            if args.use_lora:
                # Save LoRA weights
                save_lora_checkpoint(transformer, optimizer, rank, args.output_dir, step)
            else:
                # Save full checkpoint
                save_checkpoint(transformer, rank, args.output_dir, step)
            dist.barrier()

        # Clean memory periodically
        if step % 10 == 0:
            torch.cuda.empty_cache()


def get_args():
    parser = argparse.ArgumentParser()

    # wan arguments
    parser.add_argument(
        "--task",
        type=str,
        default="t2v-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The task to run.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.")
    parser.add_argument("--resume_from_weight", type=str, default=None)
    parser.add_argument("--max_seq_len", type=int,
                        default=32760)  # 32760 480p 75600 720p

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_height", type=int, default=480)
    parser.add_argument("--num_width", type=int, default=848)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")
    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")
    parser.add_argument("--group_ar", action="store_true")  # grouped by aspect ratio

    # validation & logs
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="A seed for reproducible training.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=("Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
              " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
              " training using `--resume_from_checkpoint`."),
    )
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--use_dynamic_shift", action="store_true", default=False)

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous checkpoint. Use a path saved by"
              ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
              ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    # optimizer & scheduler & Training
    # 互斥参数组：要么指定轮数，要么指定步数
    training_group = parser.add_mutually_exclusive_group()
    training_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="训练轮数。与 max_train_steps 互斥，只能指定其中一个。"
    )
    training_group.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="训练总步数。与 num_train_epochs 互斥，只能指定其中一个。"
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument("--max_grad_norm", default=1.0,
                        type=float, help="Max gradient norm.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
              " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1,
                        help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=False,
        help="Whether to use LoRA for finetuning.",
    )
    parser.add_argument("--lora_alpha", type=int, default=256,
                        help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_rank", type=int, default=128,
                        help="LoRA rank parameter. ")
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "uniform"],
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme.",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=('The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
              ' "constant", "constant_with_warmup"]'),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay to apply.")
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument("--compute_ot", action="store_true")
    parser.add_argument("--prompt_type", type=str, default="prompt_cap_base_path")

    # Performance profiling
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=5,
        help="要进行性能分析的初始步数。设置为0禁用分析。"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
