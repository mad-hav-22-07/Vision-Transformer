"""
ROS2 utility functions for image conversion and processing.
"""

import numpy as np
import cv2


def resize_mask_to_original(
    mask: np.ndarray,
    original_size: tuple[int, int],
) -> np.ndarray:
    """
    Resize segmentation mask back to original image size.

    Uses nearest-neighbor interpolation to preserve class indices.

    Args:
        mask:          Prediction mask (H_model, W_model).
        original_size: (H_original, W_original).

    Returns:
        Resized mask (H_original, W_original).
    """
    return cv2.resize(
        mask,
        (original_size[1], original_size[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def extract_lane_info(mask: np.ndarray) -> dict:
    """
    Extract lane information from a segmentation mask.

    Divides the image into left and right halves and determines
    the dominant lane type in each region.

    Args:
        mask: Segmentation mask (H, W) with values {0, 1, 2}.

    Returns:
        Dict with lane information:
            {
                "left_lane_type": "dashed" | "solid" | "none",
                "right_lane_type": "dashed" | "solid" | "none",
                "left_lane_pixels": int,
                "right_lane_pixels": int,
                "total_lane_pixels": int,
            }
    """
    h, w = mask.shape
    mid_w = w // 2

    left_mask = mask[:, :mid_w]
    right_mask = mask[:, mid_w:]

    def classify_region(region_mask):
        dashed_count = np.sum(region_mask == 1)
        solid_count = np.sum(region_mask == 2)

        if dashed_count == 0 and solid_count == 0:
            return "none", 0
        elif dashed_count > solid_count:
            return "dashed", int(dashed_count + solid_count)
        else:
            return "solid", int(dashed_count + solid_count)

    left_type, left_pixels = classify_region(left_mask)
    right_type, right_pixels = classify_region(right_mask)

    return {
        "left_lane_type": left_type,
        "right_lane_type": right_type,
        "left_lane_pixels": left_pixels,
        "right_lane_pixels": right_pixels,
        "total_lane_pixels": int(np.sum(mask > 0)),
    }
