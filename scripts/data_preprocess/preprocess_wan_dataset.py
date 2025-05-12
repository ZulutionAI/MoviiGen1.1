#!/usr/bin/env python3
"""
WAN Dataset Preprocessor v5 - Enhanced Version

Preprocesses video datasets for WAN model training, with support for:
- Distributed processing across multiple GPUs
- Aspect ratio binning for efficient batching
- Advanced error handling and recovery
- Progress tracking and checkpointing
- Merging newly processed samples with previously processed ones
"""

import argparse
import datetime
import json
import logging
import os
import random
import re
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import decord
    import numpy as np
    from decord import VideoReader, cpu, gpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

# Import custom modules
from torch.utils.data.distributed import DistributedSampler

from scripts.dataset.distributed_sampler import DistributedGroupSampler
from scripts.dataset.t2v_datasets import T2V_dataset
from wan.modules.clip import CLIPModel
from wan.modules.t5 import T5EncoderModel
# Assume you have these modules imported from your existing codebase
from wan.modules.vae import WanVAE


# Configure Python standard logging
def setup_logger():
    """Set up formatted logger handler"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Set format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()

# Configure decord to use torch bridge if available
if HAS_DECORD:
    decord.bridge.set_bridge('torch')


def sanitize_filename(filename):
    """Convert any filename to a safe string for file storage"""
    # Remove invalid characters
    safe_name = re.sub(r'[\\/*?:"<>|]', "", filename)
    # Replace spaces with underscores
    safe_name = safe_name.replace(' ', '_')
    # Limit length
    if len(safe_name) > 200:
        # Keep extension if present
        parts = safe_name.rsplit('.', 1)
        if len(parts) > 1:
            safe_name = parts[0][:190] + '...' + parts[1]
        else:
            safe_name = safe_name[:195] + '...'
    return safe_name


def setup_checkpointing(args):
    """Setup checkpointing directory and files"""
    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_file = os.path.join(checkpoint_dir, f"processed_videos_{args.run_id}.json")
    return checkpoint_dir, checkpoint_file


def load_processed_items(checkpoint_file):
    """Load set of already processed items from checkpoint file"""
    processed_items = set()
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                processed_items = set(data.get('processed_items', []))
                logger.info(f"Loaded {len(processed_items)} processed items from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint file: {e}")
    return processed_items


def save_processed_items(checkpoint_file, processed_items, stats=None):
    """Save set of processed items to checkpoint file"""
    try:
        data = {
            'processed_items': list(processed_items),
            'last_updated': time.time(),
            'stats': stats or {}
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"Failed to save checkpoint: {e}")


def collate_fn(batch):
    """
    Custom collate function to handle special processing for videos.

    Args:
        batch: List of data items from dataset

    Returns:
        Dictionary with batched data
    """
    path = []
    pixel_values = []
    text = []
    img = []
    metadata = []
    aspect_ratio_bins = []

    for b in batch:
        path.append(b["path"])
        pixel_values.append(b["pixel_values"])
        text.append(b["text"])
        metadata.append(b["metadata"])
        aspect_ratio_bins.append(b["metadata"].get("aspect_ratio_bin", 1))  # Default to bin 1
        if "img" in b:
            img.append(b["img"])

    return dict(
        path=path,
        pixel_values=torch.stack(pixel_values),
        text=text,
        img=img if len(img) > 0 else None,
        metadata=metadata,
        aspect_ratio_bins=aspect_ratio_bins,
    )


def collate_fn_naive(batch):
    """Simple collate function that does no special processing"""
    return batch


def create_temporal_sampler(args):
    """
    Create a temporal sampling function.

    Args:
        args: Command line arguments

    Returns:
        Function that samples frame indices from a video
    """
    def temporal_sample(length):
        """Uniformly sample seq_len indices from [0, length)."""
        if length >= args.num_frames:
            # Take a random subsequence from the video
            start_idx = random.randint(0, length - args.num_frames)
            return start_idx, start_idx + args.num_frames
        else:
            # Return the entire video
            return 0, length
    return temporal_sample


def handle_exception(e, video_name=None, rank=0):
    """
    Handle exceptions during processing.

    Args:
        e: Exception that was raised
        video_name: Name of the video being processed
        rank: Current process rank

    Returns:
        Formatted error message
    """
    error_type = type(e).__name__
    error_msg = str(e)

    # Format the error message
    if video_name:
        msg = f"Error processing {video_name} on rank {rank}: {error_type} - {error_msg}"
    else:
        msg = f"Error on rank {rank}: {error_type} - {error_msg}"

    # Log stack trace for debugging
    logger.error(f"{msg}\n{traceback.format_exc()}")
    return msg


def setup_signal_handler(processed_items, checkpoint_file):
    """
    Setup handlers for signals to allow graceful termination.

    Args:
        processed_items: Set of processed item names
        checkpoint_file: Path to checkpoint file
    """
    def signal_handler(sig, frame):
        logger.info("Received termination signal, saving checkpoint...")
        save_processed_items(checkpoint_file, processed_items)
        logger.info("Checkpoint saved, exiting...")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal


def get_run_id():
    """Generate a unique run ID based on timestamp"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    random_suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=6))
    return f"{timestamp}_{random_suffix}"


def main(args):
    """
    Main function to run the preprocessing pipeline.

    Args:
        args: Command line arguments
    """
    start_time = time.time()

    # Set up distributed processing
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    logger.info(f"world_size: {world_size}, local rank: {local_rank}, rank: {rank}")

    # Generate a unique run ID if not provided
    if not hasattr(args, 'run_id') or not args.run_id:
        args.run_id = get_run_id()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup checkpointing
    checkpoint_dir, checkpoint_file = setup_checkpointing(args)
    processed_items = load_processed_items(checkpoint_file)

    # Load previously processed samples from the filter.py output
    already_processed_json = os.path.join(args.output_dir, "videos2captions_processed.json")
    already_processed_data = []
    if os.path.exists(already_processed_json) and args.merge_with_processed:
        try:
            with open(already_processed_json, 'r') as f:
                already_processed_data = json.load(f)
            logger.info(
                f"Loaded {len(already_processed_data)} previously processed samples from {already_processed_json}")
        except Exception as e:
            logger.warning(f"Failed to load previously processed data: {e}")

    # Set up signal handlers for graceful termination
    if rank == 0:
        setup_signal_handler(processed_items, checkpoint_file)

    # Error tracking
    error_counts = defaultdict(int)
    processing_stats = {
        'successful': 0,
        'failed': 0,
        'skipped': 0
    }

    # Set random seed for reproducible crops if specified
    if args.random_seed is not None:
        random.seed(args.random_seed + rank)  # Different seed per process
        torch.manual_seed(args.random_seed + rank)
        np.random.seed(args.random_seed + rank)

    # Setup device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Initialize distributed process group if not already done
    if not dist.is_initialized():
        init_timeout = datetime.timedelta(minutes=30)  # Increased timeout for large clusters
        logger.info(f"Initializing process group with timeout {init_timeout}")
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=init_timeout
        )

    # Create dummy transform functions for the dataset
    def dummy_transform(x):
        return x

    # Create dataset with aspect ratio processing
    logger.info("Creating dataset...")
    temporal_sample = create_temporal_sampler(args)
    try:
        train_dataset = T2V_dataset(
            args,
            transform=dummy_transform,
            temporal_sample=temporal_sample,
            transform_topcrop=dummy_transform
        )
    except Exception as e:
        logger.error(f"Failed to create dataset: {str(e)}")
        logger.error(traceback.format_exc())
        dist.destroy_process_group()
        return

    # Use the distributed group sampler to group by aspect ratio
    sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    # sampler = DistributedGroupSampler(dataset=train_dataset, samples_per_gpu=args.train_batch_size, num_replicas=world_size, rank=rank, shuffle=True)

    # Create dataloader with the custom sampler
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn_naive,
        pin_memory=True,  # Improve transfer speed to GPU
        prefetch_factor=2,  # Prefetch 2 batches per worker
        persistent_workers=args.dataloader_num_workers > 0,  # Keep workers alive between batches
    )

    # Create output directories
    logger.info("Creating output directories...")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.include_video:
        os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)
    if args.include_prompt:
        os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    if args.dataset == 'i2v' and args.include_video:
        os.makedirs(os.path.join(args.output_dir, "y"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "clip_feature"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "metadata"), exist_ok=True)

    # Enable automatic mixed precision for better performance
    autocast_type = torch.bfloat16

    # Load models only if needed
    logger.info("Loading models...")
    vae = None
    text_encoder = None
    clip = None

    if args.include_video:
        vae = WanVAE(vae_pth=os.path.join(args.model_path, "Wan2.1_VAE.pth"))
        vae.model = vae.model.to(device).to(autocast_type)
        vae.model.eval()
        vae.model.requires_grad_(False)

    if args.include_prompt:
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=autocast_type,
            device=device,
            checkpoint_path=os.path.join(args.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(args.model_path, "google/umt5-xxl"),
            shard_fn=None,
        )

    if args.dataset == 'i2v' and args.include_video:
        clip = CLIPModel(
            dtype=autocast_type,
            device=device,
            checkpoint_path=os.path.join(args.model_path, 'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
            tokenizer_path=os.path.join(args.model_path, 'xlm-roberta-large'))

    # Create a JSON file to store processed video information
    json_data = []
    begin_time = time.time()

    # Progress tracking
    total_batches = len(train_dataloader)
    pbar = tqdm(total=total_batches, disable=local_rank != 0)
    last_checkpoint_time = time.time()
    checkpoint_interval = args.checkpoint_interval if hasattr(args, 'checkpoint_interval') else 300  # 5 minutes

    # Synchronize all processes before starting
    dist.barrier(device_ids=[local_rank])
    logger.info(f"Process {rank}/{world_size} starting batch processing")

    # Process each batch
    for batch_idx, data in enumerate(train_dataloader):
        data_time = time.time() - begin_time
        pbar.set_postfix(data_time=data_time)

        data = data[0]
        video_path = data["path"]
        video_name = os.path.basename(video_path)

        # Clean up the video name for storage
        video_suffix = video_name.split(".")[-1]
        video_name = video_name[: -len(video_suffix) - 1]
        video_name = sanitize_filename(video_name)  # Ensure the name is safe for storage

        # Skip already processed items unless overwrite is enabled
        if video_name in processed_items and not args.overwrite:
            logger.info(f"Skipping already processed {video_name}")
            processing_stats['skipped'] += 1
            begin_time = time.time()
            pbar.update(1)
            continue

        # Process the video
        logger.info(f"Processing {video_name}...")
        process_success = False

        try:
            with torch.inference_mode():
                with torch.autocast("cuda", dtype=autocast_type):
                    item = {}
                    processing_metadata = {}

                    # Process video if needed
                    if args.include_video:
                        tensor_path = os.path.join(args.output_dir, "latent", video_name + ".pt")

                        # Prepare input for VAE
                        input = torch.stack([data["pixel_values"]]).to(device)

                        # Get metadata from the frames
                        frame_metadata = data["metadata"]
                        aspect_ratio_bin = frame_metadata["aspect_ratio_bin"]

                        # Store metadata
                        processing_metadata["frames"] = frame_metadata
                        processing_metadata["aspect_ratio_bin"] = aspect_ratio_bin

                        # Log crop dimensions
                        crop_dimensions = frame_metadata.get("crop_dimensions", [args.crop_height, args.crop_width])
                        processing_metadata["crop_dimensions"] = crop_dimensions

                        # Encode frames to latents
                        latents = vae.encode(input)[0]
                        torch.save(latents.to(autocast_type), tensor_path)

                        # Store information for the dataset JSON
                        item["latent_path"] = os.path.join(video_name + ".pt")
                        item["length"] = latents.shape[1]
                        item["crop_dimensions"] = crop_dimensions
                        item["aspect_ratio_bin"] = aspect_ratio_bin

                    # Process text if needed
                    if args.include_prompt:
                        txt = data["text"]
                        text_embed = text_encoder(txt, device)[0]
                        tensor_path = os.path.join(args.output_dir, "prompt_embed", video_name + ".pt")
                        torch.save(text_embed.to(autocast_type), tensor_path)

                        item["prompt_embed_path"] = os.path.join(video_name + ".pt")
                        item["caption"] = data["text"]

                    # Process image-to-video data if needed
                    if args.dataset == 'i2v' and args.include_video and "img" in data and data["img"] is not None:
                        img = data["img"].to(device)
                        bin_value = aspect_ratio_bin

                        # Get correct crop dimensions for the reference image
                        if bin_value < len(train_dataset.bin_crop_sizes):
                            crop_height = train_dataset.bin_crop_sizes[bin_value]["height"]
                            crop_width = train_dataset.bin_crop_sizes[bin_value]["width"]
                        else:
                            crop_height = args.crop_height
                            crop_width = args.crop_width

                        # Use the proper dimensions for the current aspect ratio bin
                        h, lat_h = crop_height, crop_height // 8
                        w, lat_w = crop_width, crop_width // 8

                        # Create mask
                        msk = torch.ones(1, 81, lat_h, lat_w, device=device)
                        msk[:, 1:] = 0
                        msk = torch.concat([
                            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                        ], dim=1)
                        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                        msk = msk.transpose(1, 2)[0]

                        # Process the image
                        try:
                            y = vae.encode([
                                torch.concat([
                                    torch.nn.functional.interpolate(
                                        img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                                    torch.zeros(3, 80, h, w)
                                ], dim=1).to(device)
                            ])[0]
                            y = torch.concat([msk, y])
                            clip_context = clip.visual([img[:, None, :, :].to(device)])[0]

                            # Save tensors
                            tensor_path = os.path.join(args.output_dir, "y", video_name + ".pt")
                            torch.save(y.to(autocast_type), tensor_path)
                            item["y_path"] = os.path.join(video_name + ".pt")

                            tensor_path = os.path.join(args.output_dir, "clip_feature", video_name + ".pt")
                            torch.save(clip_context.to(autocast_type), tensor_path)
                            item["clip_feature_path"] = os.path.join(video_name + ".pt")
                        except Exception as e:
                            error_type = type(e).__name__
                            error_counts[error_type] += 1
                            logger.warning(f"Error during I2V processing for {video_name}: {error_type} - {str(e)}")
                            # Continue with other processing even if I2V fails

                # Save processing metadata
                metadata_path = os.path.join(args.output_dir, "metadata", video_name + ".json")
                with open(metadata_path, "w") as f:
                    json.dump(processing_metadata, f, indent=4)

                # Add to dataset JSON
                item["processing"] = {
                    "method": "aspect_ratio_crop",
                    "metadata_path": os.path.join("metadata", video_name + ".json")
                }
                item["original_path"] = data["path"]
                json_data.append(item)

                process_success = True
                processing_stats['successful'] += 1

        except Exception as e:
            error_message = handle_exception(e, video_name, rank)
            error_type = type(e).__name__
            error_counts[error_type] += 1
            processing_stats['failed'] += 1

            # Log error but continue processing
            if error_counts[error_type] <= 10:  # Limit logging for each error type
                logger.error(f"Error processing {video_name}: {error_message}")

        # Mark as processed regardless of success (to avoid repeated failures)
        if process_success or not args.retry_failed:
            processed_items.add(video_name)

        # Periodic checkpoint saving
        current_time = time.time()
        if current_time - last_checkpoint_time > checkpoint_interval and rank == 0:
            stats = {
                'progress': f"{batch_idx+1}/{total_batches}",
                'successful': processing_stats['successful'],
                'failed': processing_stats['failed'],
                'skipped': processing_stats['skipped'],
                'error_counts': {k: v for k, v in error_counts.items()},
                'elapsed_time': time.time() - start_time
            }
            save_processed_items(checkpoint_file, processed_items, stats)
            last_checkpoint_time = current_time
            logger.info(f"Checkpoint saved: {len(processed_items)} processed items")

        begin_time = time.time()
        pbar.update(1)

    # Synchronize processes before gathering results
    logger.info(f"Process {rank} completed its batch processing.")

    # Use async barrier with device_ids for better performance
    barrier_handle = dist.barrier(device_ids=[local_rank], async_op=True)

    # While waiting for other processes, save a local checkpoint
    if rank == 0:
        stats = {
            'progress': f"{total_batches}/{total_batches}",
            'successful': processing_stats['successful'],
            'failed': processing_stats['failed'],
            'skipped': processing_stats['skipped'],
            'error_counts': {k: v for k, v in error_counts.items()},
            'elapsed_time': time.time() - start_time
        }
        save_processed_items(checkpoint_file, processed_items, stats)
        logger.info(f"Waiting for other processes to complete...")

    # Wait for all processes
    barrier_handle.wait()
    logger.info(f"All processes synchronized, gathering results...")

    # Gather data from all processes
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    logger.info(
        f"Process {rank} gathered {len(local_data)} items, total {sum(len(data) for data in gathered_data if data)}")

    # Write combined JSON on rank 0
    if rank == 0:
        all_json_data = []
        for sublist in gathered_data:
            if sublist:  # Check if the data is not None
                all_json_data.extend(sublist)

        # Merge with previously processed data if requested
        if args.merge_with_processed:
            combined_json_data = already_processed_data + all_json_data
            logger.info(
                f"Combining {len(all_json_data)} newly processed samples with {len(already_processed_data)} previously processed samples")
        else:
            combined_json_data = all_json_data
            logger.info(f"Using only {len(all_json_data)} newly processed samples (merge not requested)")

        # Write the dataset JSON
        json_path = os.path.join(args.output_dir, f"videos2caption_{args.run_id}.json")
        with open(json_path, "w") as f:
            json.dump(combined_json_data, f, indent=4)

        # Update the processed JSON for filter.py integration
        if args.merge_with_processed:
            with open(already_processed_json, "w") as f:
                json.dump(combined_json_data, f, indent=4)
            logger.info(f"Updated {already_processed_json} with all processed samples")

        # Create a symlink to the latest version
        latest_link = os.path.join(args.output_dir, "videos2caption_latest.json")
        if os.path.exists(latest_link):
            try:
                os.remove(latest_link)
            except Exception as e:
                logger.warning(f"Failed to remove existing symlink: {e}")

        try:
            os.symlink(json_path, latest_link)
            logger.info(f"Created symlink from {json_path} to {latest_link}")
        except Exception as e:
            logger.warning(f"Failed to create symlink: {e}")
            # Fall back to copy on platforms where symlinks aren't supported
            try:
                import shutil
                shutil.copy2(json_path, latest_link)
                logger.info(f"Created copy from {json_path} to {latest_link}")
            except Exception as e2:
                logger.warning(f"Failed to create copy: {e2}")

        # Print final statistics
        total_time = time.time() - start_time
        logger.info(f"\nPreprocessing completed in {total_time:.2f} seconds")
        logger.info(f"Processed total of {len(combined_json_data)} videos")
        logger.info(
            f"This run - Success: {processing_stats['successful']}, Failed: {processing_stats['failed']}, Skipped: {processing_stats['skipped']}")
        if error_counts:
            logger.info("Error counts by type:")
            for error_type, count in error_counts.items():
                logger.info(f"  {error_type}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset & dataloader arguments
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=8,
        help="Number of subprocesses to use for data loading.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")

    # Video dimensions
    parser.add_argument("--max_height", type=int, default=576)
    parser.add_argument("--max_width", type=int, default=1024)

    # Default crop dimensions (used as fallback)
    parser.add_argument("--crop_height", type=int, default=480)
    parser.add_argument("--crop_width", type=int, default=848)

    # Aspect ratio grouping parameters
    parser.add_argument("--aspect_ratio_buckets", type=int, default=4,
                        help="Number of aspect ratio buckets to use for grouping")
    parser.add_argument("--drop_third_bin", action="store_true",
                        help="Whether to drop videos from the third bin (high aspect ratio)")

    # Video processing parameters
    parser.add_argument("--video_length_tolerance_range", type=int, default=5)
    parser.add_argument("--train_fps", type=int, default=16)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=512)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--random_crop", action="store_true", help="Enable random cropping of input frames")

    # Processing arguments
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed for reproducible crops")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing processed files")
    parser.add_argument("--retry_failed", action="store_true", help="Retry previously failed items")
    parser.add_argument("--include_video", action="store_true",
                        help="Whether to include video processing")
    parser.add_argument("--include_prompt", action="store_true",
                        help="Whether to include prompt processing")
    parser.add_argument("--dataset", default="t2v", choices=["t2v", "i2v"],
                        help="Dataset type: text-to-video or image-to-video")

    # Data merging arguments
    parser.add_argument("--merge_with_processed", action="store_true",
                        help="Merge newly processed samples with samples from videos2captions_processed.json")

    # Output paths
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="The output directory where the processed data will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Logging directory for tensorboard and other logs.",
    )

    # Checkpointing arguments
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=300,  # 5 minutes
        help="Interval in seconds between checkpoints",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default="",
        help="Unique identifier for this run (defaults to timestamp)",
    )

    # Text encoder & model parameters
    parser.add_argument("--text_encoder_name", type=str, default="/cv/models/umt5-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    args = parser.parse_args()
    main(args)
