"""
Text-to-Video Dataset Module (v4)
- Enhanced data loading and preprocessing for video datasets
- Supports aspect ratio binning and adaptive cropping
- Handles both video and image inputs
- Includes robust error handling and performance optimizations
"""

import json
import math
import os
import random
import traceback
from collections import Counter, defaultdict
from os.path import join as opj
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

# Video decoding libraries with fallback support
try:
    import decord
    from decord import VideoReader, cpu, gpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False

try:
    from torchcodec.decoders import VideoDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False


def main_print(msg, force=False, is_master=True):
    """Print message only from master process unless forced"""
    if is_master or force:
        print(msg)


def get_rank():
    """Get current process rank in distributed setting"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_master_process():
    """Check if current process is the master process"""
    return get_rank() == 0


# Simplified placeholder for DecordInit with better error handling
class DecordInit:
    def __init__(self, device_id=0):
        self.device_id = device_id
        if HAS_DECORD:
            decord.bridge.set_bridge('torch')

    def __call__(self, path):
        if not HAS_DECORD:
            raise ImportError("Decord library not available. Please install decord.")
        try:
            return VideoReader(path, ctx=cpu(self.device_id))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize video with Decord: {e}") from e


def filter_resolution(h, w, max_h_div_w_ratio=17/16, min_h_div_w_ratio=8/16):
    """
    Filter videos based on aspect ratio constraints.

    Args:
        h: Video height
        w: Video width
        max_h_div_w_ratio: Maximum allowed height/width ratio
        min_h_div_w_ratio: Minimum allowed height/width ratio

    Returns:
        bool: True if the video satisfies aspect ratio constraints
    """
    if h <= 0 or w <= 0:
        return False

    ratio = h / w
    return min_h_div_w_ratio <= ratio <= max_h_div_w_ratio


def resize_maintain_aspect_ratio_enhanced(
    video: torch.Tensor,
    target_size: int,
    resize_method: str = "long_edge",
    crop_size=None,
    random_crop: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Resize video tensor while maintaining aspect ratio, then optionally apply random cropping.

    Args:
        video: Video tensor with shape [T, C, H, W]
        target_size: Target size value (applied to long edge or short edge)
        resize_method: Resize method - "long_edge" or "short_edge"
        crop_size: Crop size (height, width), if None no cropping is performed
        random_crop: Whether to apply random cropping, False for center crop

    Returns:
        Processed video tensor with shape [T, C, H, W] or [T, C, crop_size[0], crop_size[1]]
        Metadata dictionary with processing information
    """
    if not isinstance(video, torch.Tensor):
        raise TypeError("Video must be a torch.Tensor type")

    if video.dim() != 4:
        raise ValueError(f"Video must be a 4D tensor [T, C, H, W], but got {video.dim()} dimensions")

    # Get video dimensions
    t, c, h, w = video.shape
    original_aspect_ratio = w / h

    # If cropping is needed, ensure resized dimensions are at least as large as crop size
    adjusted_target_size = target_size
    if crop_size:
        if resize_method == "long_edge":
            # When resizing based on long edge
            if w >= h:  # Width is long edge
                # Ensure height is large enough
                min_target_for_height = int(crop_size[0] * w / h)
                if min_target_for_height > target_size:
                    adjusted_target_size = min_target_for_height
            else:  # Height is long edge
                # Ensure width is large enough
                min_target_for_width = int(crop_size[1] * h / w)
                if min_target_for_width > target_size:
                    adjusted_target_size = min_target_for_width
        else:  # "short_edge"
            # When resizing based on short edge
            if w >= h:  # Height is short edge
                # Ensure height is large enough
                if crop_size[0] > target_size:
                    adjusted_target_size = crop_size[0]
            else:  # Width is short edge
                # Ensure width is large enough
                if crop_size[1] > target_size:
                    adjusted_target_size = crop_size[1]

    # Calculate new dimensions, maintaining original aspect ratio
    if resize_method == "long_edge":
        # Resize based on long edge
        if w >= h:
            # Width is the long edge
            new_width = adjusted_target_size
            new_height = int(adjusted_target_size / original_aspect_ratio)
        else:
            # Height is the long edge
            new_height = adjusted_target_size
            new_width = int(adjusted_target_size * original_aspect_ratio)
    else:  # "short_edge"
        # Resize based on short edge
        if w >= h:
            # Height is the short edge
            new_height = adjusted_target_size
            new_width = int(adjusted_target_size * original_aspect_ratio)
        else:
            # Width is the short edge
            new_width = adjusted_target_size
            new_height = int(adjusted_target_size / original_aspect_ratio)

    # Ensure dimensions are integers and at least 1
    new_height = max(1, int(new_height))
    new_width = max(1, int(new_width))

    # Final check - ensure new dimensions are not smaller than crop size
    if crop_size:
        if new_height < crop_size[0] or new_width < crop_size[1]:
            # Resize again to accommodate cropping
            scale_h = crop_size[0] / new_height if new_height < crop_size[0] else 1
            scale_w = crop_size[1] / new_width if new_width < crop_size[1] else 1
            scale = max(scale_h, scale_w)

            new_height = max(crop_size[0], int(new_height * scale))
            new_width = max(crop_size[1], int(new_width * scale))

    # Resize the video (with automatic mixed precision for better performance)
    with torch.amp.autocast("cuda", enabled=False):
        resized_video = torch.nn.functional.interpolate(
            video,  # [T, C, H, W]
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False
        )

    metadata = {
        "resized_size": (new_height, new_width),
        "original_size": (h, w),
        "original_aspect_ratio": original_aspect_ratio,
    }

    # Apply random or center crop (if requested)
    if crop_size:
        t, c, h, w = resized_video.shape

        # Double-check dimensions (safety check)
        if crop_size[0] > h or crop_size[1] > w:
            raise ValueError(f"Even after adjustment, crop size {crop_size} is still larger than resized video dimensions ({h}, {w})")

        if random_crop:
            # Random crop
            top = random.randint(0, h - crop_size[0])
            left = random.randint(0, w - crop_size[1])
        else:
            # Center crop
            top = (h - crop_size[0]) // 2
            left = (w - crop_size[1]) // 2

        # Apply crop
        resized_video = resized_video[:, :, top:top + crop_size[0], left:left + crop_size[1]]

        metadata.update({
            "crop_offset": (top, left),
            "crop_size": crop_size,
        })

    return resized_video, metadata


def check_video_integrity(video: torch.Tensor, min_mean_pixel_value: float = 5.0) -> bool:
    """
    Check if a video has valid content (not all black frames or corrupted).

    Args:
        video: Video tensor with shape [T, C, H, W]
        min_mean_pixel_value: Minimum average pixel value to consider valid

    Returns:
        bool: True if video appears valid
    """
    if video is None or video.numel() == 0:
        return False

    # Check if video has valid shape
    if video.dim() != 4:
        return False

    # Check if video has valid values (not all zeros or NANs)
    if torch.isnan(video).any() or torch.mean(video.float()) < min_mean_pixel_value:
        return False

    return True


class T2V_dataset(Dataset):
    """
    Text-to-Video dataset that handles loading and preprocessing of video data.
    Supports aspect ratio binning, adaptive cropping, and both video and image inputs.

    Features:
    - Aspect ratio grouping for more efficient batching
    - Adaptive crop sizes based on aspect ratio bins
    - Robust video loading with fallback mechanisms
    - Support for both video and image inputs
    - Configurable temporal sampling
    """

    def __init__(self, args, transform, temporal_sample, transform_topcrop):
        """
        Initialize the Text-to-Video dataset.

        Args:
            args: Configuration arguments
            transform: Transform function for video frames
            temporal_sample: Function to sample frames temporally
            transform_topcrop: Transform function for top-cropped images
        """

        self.is_master = is_master_process()

        self.args = args  # Save all args for reference
        self.data = args.data_merge_path
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.use_image_num = args.use_image_num

        self.transform = transform
        self.transform_topcrop = transform_topcrop
        self.temporal_sample = temporal_sample

        self.text_max_length = args.text_max_length
        self.cfg = args.cfg
        self.speed_factor = args.speed_factor

        self.random_crop = args.random_crop

        # Base dimensions
        self.max_height = args.max_height
        self.max_width = args.max_width

        # Default crop dimensions
        self.crop_height = args.crop_height
        self.crop_width = args.crop_width

        # Add aspect ratio grouping parameters
        self.aspect_ratio_buckets = getattr(args, 'aspect_ratio_buckets', 4)

        # Aspect ratio specific crop dimensions
        # Define crop sizes for different bins (9:21 and 9:16)
        self.bin_crop_sizes = [
            {"height": 832, "width": 1920},  # Bin 0: 9:21 ratio (0.429)
            {"height": 1072, "width": 1920},  # Bin 1: 9:16 ratio (0.5625)
            {"height": 960, "width": 1280},  # Bin 2: 3:4 ratio (0.75)
        ]

        # Filter settings
        self.drop_bins = getattr(args, 'drop_bins', [])
        self.drop_short_ratio = args.drop_short_ratio

        # Error tracking
        self.error_counts = defaultdict(int)
        self.max_errors = getattr(args, 'max_errors_per_type', 100)

        assert self.speed_factor >= 1, "Speed factor must be at least 1"

        # Initialize video decoders with fallback
        self.setup_video_decoders()

        self.video_length_tolerance_range = args.video_length_tolerance_range
        self.support_Chinese = True
        if hasattr(args, 'text_encoder_name') and "mt5" not in args.text_encoder_name:
            self.support_Chinese = False

        # Get and filter the initial cap_list
        self.cap_list = self.get_cap_list()
        assert len(self.cap_list) > 0, "No valid items found in the dataset"

        # Process frame indices
        self.cap_list, self.sample_num_frames = self.define_frame_index(self.cap_list)
        self.lengths = self.sample_num_frames

        # Calculate aspect ratios and filter out bin 2 if needed
        self.aspect_ratios, self.cap_list = self.calculate_aspect_ratios(self.cap_list)

        main_print(f"Final video dataset length: {len(self.cap_list)}", is_master=self.is_master)

    def setup_video_decoders(self):
        """Initialize video decoders with proper error handling and fallbacks"""
        # Primary decoder
        self.has_torchcodec = HAS_TORCHCODEC

        # Fallback decoder
        self.has_decord = HAS_DECORD
        if self.has_decord:
            self.v_decoder = DecordInit()

        if not (self.has_torchcodec or self.has_decord):
            raise ImportError("No video decoding libraries available. Please install either torchcodec or decord.")

    def calculate_aspect_ratios(self, cap_list):
        """
        Calculate aspect ratio group/bucket for each video and filter if needed.

        Args:
            cap_list: List of video items with resolution information

        Returns:
            aspect_ratios: Array of aspect ratio bin indices
            filtered_cap_list: Filtered list of videos
        """
        aspect_ratios = []
        filtered_cap_list = []

        min_ratio = 0.3  # Minimum h/w ratio to consider
        max_ratio = 0.9  # Maximum h/w ratio to consider

        # Create bins for aspect ratios
        ratio_bins = np.linspace(min_ratio, max_ratio, self.aspect_ratio_buckets)

        # Store actual aspect ratios for calculating statistics
        bin_values = [[] for _ in range(self.aspect_ratio_buckets)]

        # Track bin distributions for potential balancing
        bin_counts = [0] * self.aspect_ratio_buckets

        for item in cap_list:
            if 'resolution' in item and item['resolution'] is not None:
                height = item['resolution'].get('height', 0)
                width = item['resolution'].get('width', 0)

                if height > 0 and width > 0:
                    aspect_ratio = height / width
                    # Assign to nearest bucket
                    bucket_idx = np.digitize(aspect_ratio, ratio_bins) - 1
                    # Ensure it's within bounds
                    bucket_idx = max(0, min(bucket_idx, self.aspect_ratio_buckets - 1))

                    # Skip videos in bin 2 if drop_third_bin is True
                    if self.drop_third_bin and bucket_idx == 2:
                        continue

                    # Store actual aspect ratio for statistics
                    bin_values[bucket_idx].append(aspect_ratio)
                    bin_counts[bucket_idx] += 1

                    # Add processed bucket index to cap_list item
                    item['aspect_ratio_bin'] = int(bucket_idx)
                    filtered_cap_list.append(item)
                    aspect_ratios.append(int(bucket_idx))
                else:
                    continue  # Skip invalid dimensions
            else:
                continue  # Skip items without resolution

        # Print distribution and mean values for each bin
        counts = np.bincount(aspect_ratios)

        main_print(f"\nAspect ratio distribution across {len(counts)} buckets:", is_master=self.is_master)
        for i in range(len(counts)):
            if i < len(bin_values) and bin_values[i]:
                bin_mean = sum(bin_values[i]) / len(bin_values[i])
                bin_min = min(bin_values[i]) if bin_values[i] else 'N/A'
                bin_max = max(bin_values[i]) if bin_values[i] else 'N/A'
                if i < len(ratio_bins) - 1:
                    bin_range = f"{ratio_bins[i]:.2f}-{ratio_bins[i+1]:.2f}"
                else:
                    bin_range = f"≥{ratio_bins[-1]:.2f}"

                main_print(f"  Bin {i} ({bin_range}): {counts[i]} items, mean={bin_mean:.4f}, min={bin_min}, max={bin_max}",
                          is_master=self.is_master)

        if self.drop_third_bin:
            main_print(f"Dropped videos in bin 2 (aspect ratio > {ratio_bins[2]:.2f})", is_master=self.is_master)

        main_print(f"Total: {sum(counts)} items after aspect ratio filtering", is_master=self.is_master)

        return np.array(aspect_ratios, dtype=np.int64), filtered_cap_list

    def __len__(self):
        """Return the number of items in the dataset"""
        return len(self.cap_list)

    def __getitem__(self, idx):
        """
        Get data for a specific index.

        Args:
            idx: Item index

        Returns:
            Data dictionary with video/image and metadata
        """
        path = self.cap_list[idx]["path"]

        if path.endswith(".mp4"):
            return self.get_video(idx)
        else:
            return self.get_image(idx)

    def get_video(self, idx):
        """
        Load and process a video.

        Args:
            idx: Video index

        Returns:
            Dictionary with processed video and metadata
        """
        video_path = self.cap_list[idx]["path"]

        # Try alternative paths for the video
        clip_roots = [
            '/movii-data-gen/share/cutscene_not_border/',
            '/movii-data-gen/share/cutscene_rm_border/',
            '/cv/share/cutscene_not_border/',
            '/cv/share/cutscene_rm_border/',
            '/movii-data-gen/share/cutscene/',
            '/cv/share/cutscene/'
        ]

        original_path = video_path
        for clip_root in clip_roots:
            video_root = '/'.join(video_path.split('/')[:4]) + '/'
            video_path_new = video_path.replace(video_root, clip_root)
            if os.path.exists(video_path_new):
                video_path = video_path_new
                break

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {original_path}, tried alternatives including {video_path}")

        frame_indices = self.cap_list[idx]["sample_frame_index"]

        # Try primary decoder first, fall back to secondary if needed
        video = None
        decoder_used = "none"
        error_messages = []

        # Try torchcodec
        if self.has_torchcodec:
            try:
                torchvision_video = VideoDecoder(video_path, device="cpu")
                video_length = len(torchvision_video)

                # Validate indices
                frame_indices = [min(idx, video_length - 1) for idx in frame_indices]

                video = torchvision_video.get_frames_at(indices=frame_indices).data
                decoder_used = "torchcodec"
            except Exception as e:
                error_messages.append(f"TorchCodec error: {str(e)}")

        # Fall back to decord if needed
        if video is None and self.has_decord:
            try:
                decord_vr = self.v_decoder(video_path)
                video_length = len(decord_vr)

                # Validate indices
                frame_indices = [min(idx, video_length - 1) for idx in frame_indices]

                video = decord_vr.get_batch(frame_indices).permute(0, 3, 1, 2)
                decoder_used = "decord"
            except Exception as e:
                error_messages.append(f"Decord error: {str(e)}")

        # If both decoders failed, raise exception
        if video is None:
            raise RuntimeError(f"All video decoders failed for {video_path}: {'; '.join(error_messages)}")

        metadata = {
            "original_aspect_ratio": video.shape[2]/video.shape[3],
            "decoder_used": decoder_used,
            "num_frames": len(frame_indices),
        }

        # Determine crop dimensions based on aspect ratio bin
        aspect_bin = self.cap_list[idx].get('aspect_ratio_bin', 1)  # Default to bin 1 if not found
        if aspect_bin < len(self.bin_crop_sizes):
            crop_height = self.bin_crop_sizes[aspect_bin]["height"]
            crop_width = self.bin_crop_sizes[aspect_bin]["width"]
        else:
            # Default crop sizes
            crop_height = self.crop_height
            crop_width = self.crop_width

        metadata["aspect_ratio_bin"] = aspect_bin
        metadata["crop_dimensions"] = [crop_height, crop_width]

        # Resize and crop the video
        video, resize_metadata = resize_maintain_aspect_ratio_enhanced(
            video,
            target_size=self.max_width,
            resize_method="long_edge",
            crop_size=(crop_height, crop_width) if self.random_crop else None,
            random_crop=self.random_crop,
        )

        # Verify dimensions after processing
        if video.shape[2] != crop_height or video.shape[3] != crop_width:
            raise ValueError(
                f"Video dimensions mismatch: expected ({crop_height}, {crop_width}), got ({video.shape[2]}, {video.shape[3]})"
            )

        metadata.update(resize_metadata)

        # Rearrange and normalize video
        video = rearrange(video, "t c h w -> c t h w")
        video = video.to(torch.uint8)
        video = video.float() / 127.5 - 1.0  # Normalize to [-1, 1]

        # Get text caption
        text = self.cap_list[idx]["cap"]
        if not isinstance(text, list):
            text = [text]
        text = random.choice(text)

        return dict(
            pixel_values=video,
            text=text,
            img=video[:, 0, :, :],
            path=video_path,
            metadata=metadata,
        )

    def get_image(self, idx):
        """
        Load and process an image.

        Args:
            idx: Image index

        Returns:
            Dictionary with processed image and metadata
        """
        image_data = self.cap_list[idx]

        if not os.path.exists(image_data["path"]):
            raise FileNotFoundError(f"Image file not found: {image_data['path']}")

        try:
            image = Image.open(image_data["path"]).convert("RGB")  # [h, w, c]
            image = torch.from_numpy(np.array(image))  # [h, w, c]
            image = rearrange(image, "h w c -> c h w").unsqueeze(0)  #  [1 c h w]
        except Exception as e:
            raise IOError(f"Failed to load image {image_data['path']}: {str(e)}")

        # Apply appropriate transform
        image = (self.transform_topcrop(image) if "human_images" in image_data["path"] else self.transform(image)
                 )  #  [1 C H W] -> num_img [1 C H W]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]
        image = image.float() / 127.5 - 1.0  # Normalize to [-1, 1]

        # Process text
        caps = (image_data["cap"] if isinstance(image_data["cap"], list) else [image_data["cap"]])
        caps = [random.choice(caps)]
        text = caps

        # Apply classifier-free guidance randomly
        text = text[0] if random.random() > self.cfg else ""

        # Tokenize if tokenizer available
        if hasattr(self, 'tokenizer'):
            text_tokens_and_mask = self.tokenizer(
                text,
                max_length=self.text_max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            input_ids = text_tokens_and_mask["input_ids"]  # 1, l
            cond_mask = text_tokens_and_mask["attention_mask"]  # 1, l
            return dict(
                pixel_values=image,
                text=text,
                input_ids=input_ids,
                cond_mask=cond_mask,
                path=image_data["path"],
                metadata={"is_image": True},
            )
        else:
            return dict(
                pixel_values=image,
                text=text,
                path=image_data["path"],
                metadata={"is_image": True},
            )

    def define_frame_index(self, cap_list):
        """
        Process the dataset items and define frame indices for each video.

        Args:
            cap_list: List of items to process

        Returns:
            new_cap_list: Filtered and processed list
            sample_num_frames: List of sampled frame counts
        """
        new_cap_list = []
        sample_num_frames = []

        # Counters for filtering statistics
        counters = {
            "cnt_too_long": 0,
            "cnt_too_short": 0,
            "cnt_no_cap": 0,
            "cnt_no_resolution": 0,
            "cnt_resolution_mismatch": 0,
            "cnt_movie": 0,
            "cnt_img": 0,
            "cnt_no_fps_duration": 0
        }

        for i in cap_list:
            path = i["path"]
            cap = i.get("cap", None)

            # Skip if no caption
            if cap is None:
                counters["cnt_no_cap"] += 1
                continue

            if path.endswith(".mp4"):
                # Check for fps and duration
                duration = i.get("duration", None)
                fps = i.get("fps", None)
                if fps is None or duration is None:
                    counters["cnt_no_fps_duration"] += 1
                    continue

                # Check resolution requirements
                resolution = i.get("resolution", None)
                if resolution is None:
                    counters["cnt_no_resolution"] += 1
                    continue

                if (resolution.get("height", None) is None or resolution.get("width", None) is None):
                    counters["cnt_no_resolution"] += 1
                    continue

                height, width = i["resolution"]["height"], i["resolution"]["width"]
                aspect = self.max_height / self.max_width
                hw_aspect_thr = 1.5

                is_pick = filter_resolution(
                    height,
                    width,
                    max_h_div_w_ratio=hw_aspect_thr * aspect,
                    min_h_div_w_ratio=1 / hw_aspect_thr * aspect,
                )
                if not is_pick:
                    counters["cnt_resolution_mismatch"] += 1
                    continue

                # Calculate number of frames
                i["num_frames"] = math.ceil(fps * duration)

                # Filter by duration
                max_length = self.video_length_tolerance_range * (self.num_frames / self.train_fps * self.speed_factor)
                if i["num_frames"] / fps > max_length:
                    counters["cnt_too_long"] += 1
                    continue

                # Resample frame indices, handling high FPS videos
                frame_interval = fps / self.train_fps
                start_frame_idx = 0
                frame_indices = np.arange(start_frame_idx, i["num_frames"], frame_interval).astype(int)

                # Filter short videos based on probability
                if len(frame_indices) < self.num_frames and random.random() < self.drop_short_ratio:
                    counters["cnt_too_short"] += 1
                    continue

                # Temporal crop for long videos
                if len(frame_indices) > self.num_frames:
                    begin_index, end_index = self.temporal_sample(len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]

                # Store frame indices in the item
                i["sample_frame_index"] = frame_indices.tolist()
                i["sample_num_frames"] = len(i["sample_frame_index"])  # Used by group sampler
                sample_num_frames.append(i["sample_num_frames"])

                # Add to processed list
                new_cap_list.append(i)
                counters["cnt_movie"] += 1

            elif path.endswith(".jpg"):  # image
                counters["cnt_img"] += 1
                new_cap_list.append(i)
                i["sample_num_frames"] = 1
                sample_num_frames.append(i["sample_num_frames"])
            else:
                raise ValueError(
                    f"Unknown file extension {path.split('.')[-1]}, only support .mp4 for video and .jpg for image"
                )

        # Log filtering statistics
        main_print(
            f"no_cap: {counters['cnt_no_cap']}, too_long: {counters['cnt_too_long']}, too_short: {counters['cnt_too_short']}, "
            f"no_resolution: {counters['cnt_no_resolution']}, resolution_mismatch: {counters['cnt_resolution_mismatch']}, "
            f"no_fps_duration: {counters['cnt_no_fps_duration']}, "
            f"Counter(sample_num_frames): {Counter(sample_num_frames)}, cnt_movie: {counters['cnt_movie']}, cnt_img: {counters['cnt_img']}, "
            f"before filter: {len(cap_list)}, after filter: {len(new_cap_list)}",
            is_master=self.is_master
        )

        return new_cap_list, sample_num_frames

    def read_jsons(self, data):
        """
        Read JSON files specified in the data merge file.

        Args:
            data: Path to data merge file

        Returns:
            Combined list of items from all JSON files
        """
        cap_lists = []
        try:
            with open(data, "r") as f:
                folder_anno = [i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0]

            for folder, anno in folder_anno:
                try:
                    with open(anno, "r") as f:
                        sub_list = json.load(f)

                    # Ensure folder path is properly formatted
                    folder = folder.rstrip('/')

                    for i in range(len(sub_list)):
                        # Handle relative vs absolute paths
                        if os.path.isabs(sub_list[i]["path"]):
                            pass  # Keep absolute paths as-is
                        else:
                            sub_list[i]["path"] = opj(folder, sub_list[i]["path"])

                    cap_lists.extend(sub_list)
                except Exception as e:
                    main_print(f"Error loading annotation file {anno}: {str(e)}", force=True)
                    continue
        except Exception as e:
            raise RuntimeError(f"Failed to load data merge file {data}: {str(e)}")

        if not cap_lists:
            raise ValueError(f"No valid items found in data merge file: {data}")

        return cap_lists

    def get_cap_list(self):
        """Load and return the caption list from the data merge file"""
        return self.read_jsons(self.data)
