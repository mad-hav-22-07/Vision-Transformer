"""
Patch Embedding Layer for Vision Transformer.

Splits an input image into non-overlapping patches and linearly projects
each patch into a fixed-dimension embedding vector. Implemented efficiently
using a strided convolution rather than explicit reshaping.

Input:  (B, 3, H, W)
Output: (B, num_patches, embed_dim)  where num_patches = (H/P) * (W/P)
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Converts a batch of images into a sequence of patch embeddings.

    Uses nn.Conv2d with kernel_size=stride=patch_size as an efficient
    linear projection of flattened patches.

    Args:
        img_size:   Tuple (H, W) of the input image resolution.
        patch_size: Size of each square patch (default: 16).
        in_channels: Number of input channels (default: 3 for RGB).
        embed_dim:  Dimension of each patch embedding vector.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (360, 640),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Strided conv is equivalent to: flatten each patch → linear projection
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input image size ({H}x{W}) doesn't match expected "
            f"size ({self.img_size[0]}x{self.img_size[1]})"
        )

        # (B, C, H, W) → (B, embed_dim, H/P, W/P)
        x = self.projection(x)

        # (B, embed_dim, H/P, W/P) → (B, embed_dim, num_patches) → (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        return x
