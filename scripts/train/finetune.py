#!/bin/python3
# isort: skip_file
import datetime

from scripts.dataset.latent_datasets import LatentDataset, latent_collate_function
from scripts.train.util.util import load_wan
from torch.utils.tensorboard import SummaryWriter
from scripts.train.util.math_util import cosine_optimal_transport
from scripts.train.model.model_seq import WanAttentionBlock as WanAttentionBlock_SEQ

from wan.configs import WAN_CONFIGS
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
import signal
sys.path.insert(0, os.path.abspath("."))


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0")

MAX_SEED_VAL = 2**32 - 1
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
    if int(os.environ.get("LOCAL_RANK", "0")) <= 0:
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
    """Compute sampling density for timesteps based on selected scheme"""
    if weighting_scheme == "logit_normal":
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

def setup_signal_handler(save_checkpoint_func):
    """Setup handlers for signals to allow graceful termination"""
    def signal_handler(sig, frame):
        main_print("Received termination signal, saving checkpoint...")
        if dist.get_rank() == 0:
            save_checkpoint_func(is_final=True)
        main_print("Checkpoint saved, exiting...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

def process_batch(
    transformer,
    batch,
    noise_scheduler,
    noise_random_generator,
    sp_size,
    max_grad_norm,
    weighting_scheme,
    logit_mean,
    logit_std,
    mode_scale,
    compute_ot=False,
    max_seq_len=32760,
    enable_timing=False,
    device=None,
):
    """Process a single batch and return loss and metrics"""
    timing_stats = {}

    if enable_timing:
        start_prep = time.time()

    # Extract batch components and move to device
    latents, encoder_hidden_states, latent_attn_mask, prompt_attention_mask = [
        x.to(device) if x is not None else None for x in batch
    ]

    # print("rank ", int(os.environ["RANK"]), " latent shapes: ", len(latents), ", ", latents[0].shape)

    # Prepare inputs
    model_input = latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    # Compute optimal transport if enabled
    if compute_ot:
        transport_cost, indices = cosine_optimal_transport(latents.reshape(bsz, -1), noise.reshape(bsz, -1))
        noise = noise[indices[1].view(-1)]

    # Sample timesteps
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

    # Synchronize across sequence parallel processes if needed
    if sp_size > 1:
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

    # Forward pass
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = [noisy_model_input[i] for i in range(noisy_model_input.size(0))]
        model_preds = transformer(x, timesteps, None, max_seq_len,
                                  batch_context=encoder_hidden_states, context_mask=prompt_attention_mask)
        model_pred = model_preds[0]

    if enable_timing:
        torch.cuda.synchronize()
        forward_time = time.time() - start_forward
        timing_stats['forward'] = forward_time
        start_loss = time.time()

    # Calculate loss
    target = noise - latents
    loss = torch.mean((model_pred.float() - target.float()) ** 2)

    if enable_timing:
        torch.cuda.synchronize()
        loss_time = time.time() - start_loss
        timing_stats['loss'] = loss_time

    # Return loss, timesteps, and timing information
    return loss, timesteps, sigmas, timing_stats


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

    init_timeout = datetime.timedelta(minutes=20)  # Increased timeout for large clusters
    main_print(f"Initializing process group with timeout {init_timeout}")
    dist.init_process_group("nccl", device_id=torch.device(f'cuda:{device}'), timeout=init_timeout)

    # Initialize sequence parallel
    assert world_size % args.sp_size == 0, f"World size ({world_size}) must be divisible by SP size ({args.sp_size})"
    initialize_sequence_parallel_state(args.sp_size)

    # Determine the base seed: Generate on rank 0 if None, then broadcast
    if args.seed is None:

        if rank == 0:
            seed_bytes = os.urandom(8)
            base_seed = int.from_bytes(seed_bytes, sys.byteorder) % MAX_SEED_VAL
            main_print(f"Generated random seed on rank 0: {base_seed}")

            args.seed = base_seed  # Store back for consistency
            # Prepare the seed tensor on the *correct GPU device* for rank 0
            seed_tensor = torch.tensor([base_seed], dtype=torch.long, device=device)  # Use the rank's GPU device

        else:
            # Other ranks prepare a placeholder tensor on *their correct GPU device*
            seed_tensor = torch.tensor([0], dtype=torch.long, device=device)  # Use the rank's GPU device

        # Broadcast the seed tensor (now on GPU) from rank 0 to all other ranks.
        # NCCL backend can handle GPU tensors.
        dist.broadcast(seed_tensor, src=0)
        # All ranks now have the same seed in seed_tensor on their respective GPUs.
        # Extract it back to CPU memory for general use.
        shared_base_seed = seed_tensor.item()
        # Ensure args.seed is consistent across all ranks
        args.seed = shared_base_seed

        # Optional: Print on all ranks to verify (or only rank 0)
        print(f"Rank {rank} received shared base seed: {shared_base_seed}")
    else:
        base_seed = args.seed
        main_print(f"Using manually provided seed: {base_seed}")

    set_seed(args.seed + rank)
    noise_random_generator = torch.Generator(device="cpu").manual_seed(args.seed + rank)

    # Create output directories and summary writer
    if rank <= 0:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.logging_dir))
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    # Load model
    main_print(f"--> Loading model from {args.ckpt_dir}")
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
    main_print("--> Model loaded")

    # Setup VAE for debug visualization
    if DEBUG:
        from wan.modules.vae import WanVAE
        vae = WanVAE(vae_pth="./MoviiGen1.1/Wan2.1_VAE.pth")
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
    main_print(f"Optimizer initialized, resuming from step {init_steps}")

    # Create dataset
    train_dataset = LatentDataset(
        args.data_json_path,
        args.num_latent_t,
        cfg_rate=0.0,
        seed=args.seed,
        prompt_type=args.prompt_type,
        resolution_mix=args.resolution_mix,
    )

    # Create sampler
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
        seed=args.seed,
        verbose=(rank == 0),
    )

    # Create dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        collate_fn=latent_collate_function,
        pin_memory=True,
        num_workers=args.dataloader_num_workers,
        persistent_workers=(args.dataloader_num_workers > 0),
    )

    # loader = sp_aware_wrapper(train_dataloader, device, args.sp_size)

    # Calculate steps per epoch and total training steps
    effective_processes = world_size // args.sp_size
    effective_batch_size = args.train_batch_size * effective_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataset) / effective_batch_size)

    # Setup training parameters (epochs vs steps)
    if args.max_train_steps is None and args.num_train_epochs is not None:
        args.max_train_steps = math.ceil(num_update_steps_per_epoch * args.num_train_epochs)
        main_print(f"Setting max_train_steps to {args.max_train_steps} based on {args.num_train_epochs} epochs")
    elif args.max_train_steps is not None and args.num_train_epochs is None:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        main_print(f"Setting num_train_epochs to {args.num_train_epochs} based on {args.max_train_steps} steps")
    elif args.max_train_steps is None and args.num_train_epochs is None:
        args.num_train_epochs = 1
        args.max_train_steps = num_update_steps_per_epoch
        main_print(f"No epochs or steps specified. Setting to 1 epoch ({args.max_train_steps} steps)")
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        main_print(f"Using specified {args.max_train_steps} steps (approximately {args.num_train_epochs} epochs)")

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
    main_print(f"  Num Epochs = {args.num_train_epochs}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Effective parallel processes = {effective_processes}")
    main_print(f"  Total train batch size (w. parallelism, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps = {args.max_train_steps}")
    main_print(f"  Steps per epoch = {num_update_steps_per_epoch}")

    if args.sp_size > 1:
        main_print(f"  Using sequence parallelism with {args.sp_size} GPUs per group")
        main_print(f"  SP groups: {world_size // args.sp_size}")

    # Function to save checkpoints
    def save_checkpoint_func(step=None, is_final=False):
        checkpoint_path = os.path.join(args.output_dir, "checkpoints")
        if is_final:
            suffix = "final"
        else:
            suffix = f"step_{step}"

        if args.use_lora:
            save_lora_checkpoint(transformer, optimizer, rank, checkpoint_path, suffix)
        else:
            save_checkpoint(transformer, rank, checkpoint_path, suffix)

        main_print(f"Checkpoint saved: {suffix}")

    # Setup signal handler
    if rank == 0:
        setup_signal_handler(save_checkpoint_func)

    # Create progress bar
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=init_steps,
        desc="Steps",
        disable=rank > 0,
    )

    # Training metrics
    step_times = deque(maxlen=100)
    enable_timing = args.profile_steps > 0

    # Calculate initial epoch for the sampler
    current_epoch = init_steps // num_update_steps_per_epoch
    current_step = init_steps

    # Main training loop
    while current_step < args.max_train_steps:
        train_dataset.set_epoch(current_epoch)
        # Set epoch for sampler
        sampler.set_epoch(current_epoch)
        main_print(f"Starting epoch {current_epoch}")

        # Track accumulated gradients
        accumulated_batches = 0

        # Loop through batches in the current epoch
        for batch in train_dataloader:

            # Skip steps that were already processed
            if current_step < init_steps:
                current_step += 1
                continue

            # Check if we should enable profiling for this step
            do_profile = enable_timing and (current_step <= args.profile_steps or current_step % 100 == 0)

            # Track timing
            batch_start_time = time.time()

            # Zero gradients at the beginning of accumulation cycle
            if accumulated_batches == 0:
                optimizer.zero_grad()

            try:
                # Process batch
                loss, timesteps, sigmas, timing_stats = process_batch(
                    transformer,
                    batch,
                    noise_scheduler,
                    noise_random_generator,
                    args.sp_size,
                    args.max_grad_norm,
                    args.weighting_scheme,
                    args.logit_mean,
                    args.logit_std,
                    args.mode_scale,
                    compute_ot=args.compute_ot,
                    max_seq_len=args.max_seq_len,
                    enable_timing=do_profile,
                    device=device,
                )

                # Scale loss for gradient accumulation
                scaled_loss = loss / args.gradient_accumulation_steps

                # Backward pass
                if not DEBUG:
                    scaled_loss.backward()

                # Average loss across processes for logging
                avg_loss = loss.detach().clone()
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

                # Increment accumulation counter
                accumulated_batches += 1

                # If we've accumulated enough gradients, update weights
                if accumulated_batches >= args.gradient_accumulation_steps:
                    if do_profile and timing_stats:
                        start_grad = time.time()

                    # Clip gradients
                    grad_norm = transformer.clip_grad_norm_(args.max_grad_norm)

                    if do_profile and timing_stats:
                        torch.cuda.synchronize()
                        grad_clip_time = time.time() - start_grad
                        timing_stats['grad_clip'] = grad_clip_time
                        start_optim = time.time()

                    # Optimizer step
                    if not DEBUG:
                        optimizer.step()
                        lr_scheduler.step()

                    if do_profile and timing_stats:
                        torch.cuda.synchronize()
                        optim_time = time.time() - start_optim
                        timing_stats['optim'] = optim_time

                    # Reset for next accumulation cycle
                    optimizer.zero_grad()
                    accumulated_batches = 0

                    # Track step timing
                    step_time = time.time() - batch_start_time
                    step_times.append(step_time)
                    avg_step_time = sum(step_times) / len(step_times)

                    # Log profiling data
                    if do_profile and timing_stats:
                        profile_msg = " | ".join([f"{k}: {v:.2f}s" for k, v in timing_stats.items()])
                        main_print(f"Profiling step {current_step}: {profile_msg}\n")

                    main_print(
                        f"Step {current_step}/{args.max_train_steps} - Loss: {loss:.4f} - Step time: {step_time:.2f}s - Avg time: {avg_step_time:.2f}s"
                    )

                    # Update progress
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss.item():.4f}",
                        "step_time": f"{step_time:.2f}s",
                        "grad_norm": f"{grad_norm.item():.3f}",
                        "epoch": current_epoch,
                        "avg_time": f"{avg_step_time:.2f}s",
                    })
                    progress_bar.update(1)

                    # Log to tensorboard
                    if rank <= 0:
                        writer.add_scalar("loss", avg_loss.item(), current_step)
                        writer.add_scalar("grad_norm", grad_norm.item(), current_step)
                        writer.add_scalar("epoch", current_epoch, current_step)
                        writer.add_scalar("step_time", step_time, current_step)
                        writer.add_scalar("avg_step_time", avg_step_time, current_step)

                        # Log timestep bin statistics
                        bin_index = min(int(timesteps.mean().item() / 100), 10 - 1)
                        writer.add_scalar(f"loss_bin_{bin_index}", avg_loss.item(), current_step)

                        # Log timing details
                        if timing_stats:
                            for k, v in timing_stats.items():
                                writer.add_scalar(f"time/{k}", v, current_step)

                    # Save checkpoint at regular intervals
                    if current_step != 0 and current_step % args.checkpointing_steps == 0:
                        save_checkpoint_func(current_step)
                        dist.barrier(device_ids=[local_rank])

                    # Clean memory periodically
                    if current_step % 10 == 0:
                        torch.cuda.empty_cache()

                    # Increment step counter
                    current_step += 1

                    # Check if we've reached max steps
                    if current_step >= args.max_train_steps:
                        break

            except Exception as e:
                main_print(f"Error processing batch: {str(e)}")
                # Reset accumulated gradients on error
                if accumulated_batches > 0:
                    optimizer.zero_grad()
                    accumulated_batches = 0
                # Continue with next batch

        # Increment epoch counter after processing all batches
        current_epoch += 1

    # Save final checkpoint
    if rank == 0:
        save_checkpoint_func(current_step, is_final=True)

    main_print("Training completed!")


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
    parser.add_argument("--max_seq_len", type=int, default=32760)  # 32760 480p 75600 720p

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
    parser.add_argument("--train_batch_size",
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
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
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
        help=("Save a checkpoint of the training state every X updates."),
    )
    parser.add_argument("--shift", type=float, default=1.0)
    parser.add_argument("--use_dynamic_shift", action="store_true", default=False)

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous checkpoint."),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=("Whether training should be resumed from a previous lora checkpoint."),
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=("TensorBoard log directory"),
    )

    # optimizer & scheduler & Training
    training_group = parser.add_mutually_exclusive_group()
    training_group.add_argument(
        "--num_train_epochs",
        type=int,
        default=None,
        help="Number of training epochs."
    )
    training_group.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps."
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
        help=("Whether or not to allow TF32 on Ampere GPUs"),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=("Whether to use mixed precision"),
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
    parser.add_argument("--prompt_type", type=str, default="prompt_embed_path")

    parser.add_argument("--resolution_mix", default=None, type=str)

    # Performance profiling
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=5,
        help="Number of initial steps to profile. Set to 0 to disable profiling."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
