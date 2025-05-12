import random
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


def resize_maintain_aspect_ratio(
    video: torch.Tensor,  # Video tensor with shape [T, C, H, W]
    target_size: int,     # Single size value, applied to long or short edge
    resize_method: Literal["long_edge", "short_edge"] = "long_edge",
    crop_size: Optional[Tuple[int, int]] = None,  # Optional crop size (H, W)
    random_crop: bool = True  # Whether to use random crop
) -> torch.Tensor:
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
    """
    if not isinstance(video, torch.Tensor):
        raise TypeError("Video must be a torch.Tensor type")

    if video.dim() != 4:
        raise ValueError(f"Video must be a 4D tensor [T, C, H, W], but got {video.dim()} dimensions")

    # Get video dimensions
    t, c, h, w = video.shape
    original_aspect_ratio = w / h

    # Calculate new dimensions, maintaining original aspect ratio
    if resize_method == "long_edge":
        # Resize based on long edge
        if w >= h:
            # Width is the long edge
            new_width = target_size
            new_height = int(target_size / original_aspect_ratio)
        else:
            # Height is the long edge
            new_height = target_size
            new_width = int(target_size * original_aspect_ratio)
    else:  # "short_edge"
        # Resize based on short edge
        if w >= h:
            # Height is the short edge
            new_height = target_size
            new_width = int(target_size * original_aspect_ratio)
        else:
            # Width is the short edge
            new_width = target_size
            new_height = int(target_size / original_aspect_ratio)

    # Ensure dimensions are integers
    new_height = max(1, new_height)
    new_width = max(1, new_width)

    # Resize the video
    resized_video = F.interpolate(
        video,  # [T, C, H, W]
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False
    )

    # Apply random or center crop (if requested)
    if crop_size:
        t, c, h, w = resized_video.shape

        # Check if crop size is smaller than video dimensions
        if crop_size[0] > h or crop_size[1] > w:
            raise ValueError(f"Crop size {crop_size} is larger than resized video dimensions ({h}, {w})")

        if random_crop:
            # Random crop
            top = random.randint(0, h - crop_size[0])
            left = random.randint(0, w - crop_size[1])
        else:
            # Center crop
            top = (h - crop_size[0]) // 2
            left = (w - crop_size[1]) // 2

        # Apply crop
        resized_video= resized_video[:, :, top:top + crop_size[0], left:left + crop_size[1]]

    return resized_video


def resize_maintain_aspect_ratio_enhanced(
    video: torch.Tensor,  # Video tensor with shape [T, C, H, W]
    target_size: int,     # Single size value, applied to long or short edge
    resize_method: Literal["long_edge", "short_edge"] = "long_edge",
    crop_size: Optional[Tuple[int, int]] = None,  # Optional crop size (H, W)
    random_crop: bool = True  # Whether to use random crop
) -> torch.Tensor:
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
                min_target_for_height = int(crop_size[0] * original_aspect_ratio)
                if min_target_for_height > target_size:
                    adjusted_target_size = min_target_for_height
            else:  # Height is long edge
                # Ensure width is large enough
                min_target_for_width = int(crop_size[1] / original_aspect_ratio)
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

    if adjusted_target_size != target_size:
        print(f"Warning: To accommodate crop size {crop_size}_hw, target size adjusted from {target_size}_w to {adjusted_target_size}_w")

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

            print(f"Additional adjustment: New dimensions set to ({new_height}, {new_width}) to accommodate crop size {crop_size}")

    # Resize the video
    resized_video = F.interpolate(
        video,  # [T, C, H, W]
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False
    )

    metadata = {
        "resized_size": (new_height, new_width),
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
        })

    return resized_video, metadata
