"""
Positional Encoding for Vision Transformer.

Provides learnable positional embeddings that encode the 2D spatial
position of each patch in the image grid. Supports interpolation to
handle resolution changes at inference time (trained at 360x640 but
deployed at a different resolution).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    Learnable 2D positional embeddings.

    Each patch position gets a unique learnable embedding vector that is
    added element-wise to the patch embedding. This encodes spatial
    structure that the transformer otherwise cannot perceive.

    Args:
        num_patches: Total number of patches (grid_h * grid_w).
        embed_dim:   Dimension of each embedding vector.
        grid_size:   Tuple (grid_h, grid_w) for 2D interpolation support.
        dropout:     Dropout probability applied after adding positional info.
    """

    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        grid_size: tuple[int, int] = (22, 40),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.grid_size = grid_size

        # Learnable positional embeddings — initialized from truncated normal
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )
        self.dropout = nn.Dropout(p=dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize positional embeddings with truncated normal distribution."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def interpolate_pos_encoding(
        self, x: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings when input resolution differs
        from the training resolution.

        This enables deploying a model trained on 360x640 at any other
        resolution (e.g., 480x640, 720x1280) without retraining.

        Args:
            x: Patch embeddings (B, N, D).
            h: Number of patches along height for the current input.
            w: Number of patches along width for the current input.

        Returns:
            Interpolated positional embeddings matching (B, h*w, D).
        """
        num_patches = x.shape[1]

        if num_patches == self.num_patches:
            return self.pos_embed

        # Reshape from (1, N, D) to (1, D, grid_h, grid_w) for spatial interpolation
        pos_embed = self.pos_embed.reshape(
            1, self.grid_size[0], self.grid_size[1], self.embed_dim
        ).permute(0, 3, 1, 2)  # (1, D, grid_h, grid_w)

        # Bilinear interpolation to new grid size
        pos_embed = F.interpolate(
            pos_embed,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        # Reshape back to (1, h*w, D)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, self.embed_dim)

        return pos_embed

    def forward(
        self, x: torch.Tensor, grid_size: tuple[int, int] | None = None
    ) -> torch.Tensor:
        """
        Add positional encoding to patch embeddings.

        Args:
            x: Patch embeddings of shape (B, num_patches, embed_dim).
            grid_size: Optional (h, w) if input resolution differs from training.

        Returns:
            Patch embeddings with positional information added.
        """
        if grid_size is not None and (
            grid_size[0] != self.grid_size[0] or grid_size[1] != self.grid_size[1]
        ):
            pos = self.interpolate_pos_encoding(x, grid_size[0], grid_size[1])
        else:
            pos = self.pos_embed

        x = x + pos
        x = self.dropout(x)

        return x
