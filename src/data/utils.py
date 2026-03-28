"""
Data utility functions for lane segmentation.

Provides helpers for mask encoding/decoding, colormap visualization,
and dataset statistics.
"""

import numpy as np
import torch
import cv2


# Lane class definitions
LANE_CLASSES = {
    0: "background",
    1: "dashed",
    2: "solid",
}

# Colormap for visualization (BGR format for OpenCV)
LANE_COLORMAP = {
    0: (0, 0, 0),        # Background: black
    1: (0, 255, 255),    # Dashed: yellow
    2: (0, 255, 0),      # Solid: green
}

# RGB colormap for matplotlib
LANE_COLORMAP_RGB = {
    0: (0, 0, 0),
    1: (255, 255, 0),
    2: (0, 255, 0),
}


def mask_to_color(
    mask: np.ndarray, colormap: dict = LANE_COLORMAP
) -> np.ndarray:
    """
    Convert a class-index mask to a colorized RGB/BGR image.

    Args:
        mask:     2D array (H, W) with class indices.
        colormap: Dict mapping class_id → (B, G, R) tuple.

    Returns:
        Color image (H, W, 3) as uint8.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in colormap.items():
        color_mask[mask == class_id] = color

    return color_mask


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a segmentation mask on the original image.

    Args:
        image: Original image (H, W, 3), BGR uint8.
        mask:  Class-index mask (H, W).
        alpha: Transparency of the overlay (0 = transparent, 1 = opaque).

    Returns:
        Blended image (H, W, 3) as uint8.
    """
    color_mask = mask_to_color(mask)

    # Only overlay non-background pixels
    overlay = image.copy()
    non_bg = mask > 0
    overlay[non_bg] = cv2.addWeighted(
        image[non_bg], 1 - alpha,
        color_mask[non_bg], alpha,
        0,
    )

    return overlay


def denormalize_image(
    image: torch.Tensor,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Reverse ImageNet normalization and convert tensor to numpy image.

    Args:
        image: Normalized image tensor (3, H, W) or (H, W, 3).
        mean:  Normalization mean.
        std:   Normalization std.

    Returns:
        Denormalized image as uint8 numpy array (H, W, 3), RGB.
    """
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.permute(1, 2, 0)  # CHW → HWC
        image = image.numpy()

    mean = np.array(mean)
    std = np.array(std)

    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)

    return image


def logits_to_mask(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert model output logits to a class-index mask.

    Args:
        logits: Model output (B, num_classes, H, W) or (num_classes, H, W).

    Returns:
        Class-index mask (B, H, W) or (H, W) as LongTensor.
    """
    return logits.argmax(dim=-3)  # works for both batched and unbatched
