"""
Gaussian Heatmap Auxiliary Loss for Lane Segmentation.

Inspired by LDTR's auxiliary heatmap branch. Generates Gaussian heatmaps
from segmentation masks along lane pixels and supervises the model's
heatmap prediction head during training. This encourages the model to
learn lane structure and continuity.

The heatmap is discarded at inference time — it only helps training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HeatmapLoss(nn.Module):
    """
    Gaussian Heatmap Auxiliary Loss.

    Generates target heatmaps from segmentation masks by applying Gaussian
    blur to lane pixels, then computes MSE loss against the model's
    heatmap prediction.

    Args:
        sigma:      Gaussian kernel standard deviation.
        kernel_size: Gaussian kernel size (must be odd).
        reduction:  Loss reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        sigma: float = 3.0,
        kernel_size: int = 15,
        reduction: str = "mean",
    ):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.reduction = reduction

        # Pre-compute Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, sigma)
        # Register as buffer (non-trainable, moves with model device)
        self.register_buffer("gaussian_kernel", kernel)

    @staticmethod
    def _create_gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        kernel = g.unsqueeze(0) * g.unsqueeze(1)  # Outer product
        kernel = kernel / kernel.sum()
        # Shape: (1, 1, size, size) for conv2d
        return kernel.unsqueeze(0).unsqueeze(0)

    def generate_heatmap(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Generate Gaussian heatmaps from segmentation masks.

        Args:
            masks: Ground truth masks (B, H, W) with class indices.

        Returns:
            Heatmaps (B, 1, H, W) with Gaussian-blurred lane regions.
        """
        # Create binary lane mask (class > 0 = any lane)
        # Force float32 to avoid AMP half-precision issues with conv2d
        lane_mask = (masks > 0).float().unsqueeze(1)  # (B, 1, H, W)

        # Ensure kernel matches input device AND dtype
        kernel = self.gaussian_kernel.to(device=lane_mask.device, dtype=lane_mask.dtype)

        # Apply Gaussian blur
        padding = self.kernel_size // 2
        heatmap = F.conv2d(
            lane_mask,
            kernel,
            padding=padding,
        )

        # Normalize to [0, 1]
        B = heatmap.shape[0]
        for i in range(B):
            max_val = heatmap[i].max()
            if max_val > 0:
                heatmap[i] = heatmap[i] / max_val

        return heatmap

    def forward(
        self,
        pred_heatmap: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute heatmap loss.

        Args:
            pred_heatmap: Predicted heatmap (B, 1, H, W) from model.
            masks:        Ground truth masks (B, H, W).

        Returns:
            Scalar MSE loss.
        """
        # Generate target heatmap from masks
        target_heatmap = self.generate_heatmap(masks)

        # Resize prediction to match target if needed
        if pred_heatmap.shape[2:] != target_heatmap.shape[2:]:
            pred_heatmap = F.interpolate(
                pred_heatmap,
                size=target_heatmap.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # MSE loss
        loss = F.mse_loss(pred_heatmap, target_heatmap, reduction=self.reduction)

        return loss
