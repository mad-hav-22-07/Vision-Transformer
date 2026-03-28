"""
Vision Transformer Encoder with Multi-Scale Feature Extraction.

Stacks N TransformerBlocks to form the full encoder. Extracts intermediate
features at specified layers (skip connections) for the decoder to use,
enabling multi-scale feature fusion for fine-grained segmentation.
"""

import torch
import torch.nn as nn

from .patch_embed import PatchEmbedding
from .positional_encoding import PositionalEncoding
from .transformer_block import TransformerBlock


class ViTEncoder(nn.Module):
    """
    Vision Transformer Encoder.

    Processes input images through patch embedding → positional encoding →
    N transformer blocks, extracting features at multiple depths for
    the segmentation decoder.

    Args:
        img_size:           Input image resolution (H, W).
        patch_size:         Patch size in pixels.
        in_channels:        Number of input image channels.
        embed_dim:          Token embedding dimension.
        depth:              Number of transformer blocks.
        num_heads:          Number of attention heads per block.
        expansion_ratio:    FFN expansion factor.
        attn_drop:          Attention dropout rate.
        proj_drop:          Projection dropout rate.
        ffn_drop:           FFN dropout rate.
        drop_path_rate:     Max stochastic depth rate (linearly increases).
        skip_indices:       Layer indices to extract skip features from.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (360, 640),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        expansion_ratio: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.1,
        drop_path_rate: float = 0.1,
        skip_indices: tuple[int, ...] | None = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth

        # Default: extract features from layers 3, 6, 9 (0-indexed: 2, 5, 8)
        # The final layer output is always included
        if skip_indices is None:
            self.skip_indices = self._default_skip_indices(depth)
        else:
            self.skip_indices = skip_indices

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            num_patches=self.patch_embed.num_patches,
            embed_dim=embed_dim,
            grid_size=self.patch_embed.grid_size,
            dropout=ffn_drop,
        )

        # Linearly increasing stochastic depth rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                expansion_ratio=expansion_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                ffn_drop=ffn_drop,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _default_skip_indices(depth: int) -> tuple[int, ...]:
        """Compute evenly-spaced skip connection layer indices."""
        if depth <= 4:
            return tuple(range(depth))
        step = depth // 4
        return (step - 1, 2 * step - 1, 3 * step - 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass through the encoder.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Tuple of:
                - Final encoder output (B, num_patches, embed_dim).
                - List of skip features from intermediate layers,
                  each of shape (B, num_patches, embed_dim).
        """
        # Patch embedding: (B, C, H, W) → (B, N, D)
        x = self.patch_embed(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Collect skip features for decoder
        skip_features = []

        # Process through transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x)

            if i in self.skip_indices:
                skip_features.append(x)

        # Final norm
        x = self.norm(x)

        return x, skip_features
