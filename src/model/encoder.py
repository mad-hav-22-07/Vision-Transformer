"""
Multi-Scale Encoder for LDTR-inspired Lane Segmentation.

Replaces the single-scale ViT encoder with a hybrid CNN + Transformer
architecture that produces multi-scale feature maps, similar to LDTR's
backbone + encoder design:

    Stage 1: Conv blocks → features at 1/4 resolution
    Stage 2: Conv blocks → features at 1/8 resolution
    Stage 3: Conv blocks → features at 1/16 resolution
    Stage 4: Transformer self-attention on 1/16 features for global context

This gives the decoder both local detail (CNN stages) and global semantic
understanding (transformer stage), which is critical for detecting lanes
that span the entire image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import PositionalEncoding
from .attention import MultiHeadSelfAttention
from .transformer_block import TransformerBlock


class ConvBlock(nn.Module):
    """Two convolution layers with BatchNorm and GELU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
        # Residual connection
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if in_ch != out_ch or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class MultiScaleEncoder(nn.Module):
    """
    Multi-Scale Encoder with CNN backbone + Transformer top.

    Produces feature maps at 3 scales (1/4, 1/8, 1/16) for the
    deformable attention decoder to attend to.

    Args:
        in_channels:       Number of input image channels (3 for RGB).
        backbone_channels: Channel dimensions for each CNN stage.
        embed_dim:         Transformer embedding dimension (applied at 1/16 scale).
        num_heads:         Number of attention heads for transformer blocks.
        transformer_depth: Number of transformer blocks at the deepest scale.
        expansion_ratio:   FFN expansion ratio in transformer blocks.
        dropout:           Dropout rate.
        drop_path_rate:    Maximum stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: tuple[int, ...] = (64, 128, 256),
        embed_dim: int = 256,
        num_heads: int = 8,
        transformer_depth: int = 6,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_feature_levels = len(backbone_channels)

        # --- CNN Backbone Stages ---
        # Stage 1: stride-4 (two stride-2 convs) → 1/4 resolution
        self.stage1 = nn.Sequential(
            ConvBlock(in_channels, backbone_channels[0], stride=2),
            ConvBlock(backbone_channels[0], backbone_channels[0], stride=2),
        )

        # Stage 2: stride-2 → 1/8 resolution
        self.stage2 = nn.Sequential(
            ConvBlock(backbone_channels[0], backbone_channels[1], stride=2),
            ConvBlock(backbone_channels[1], backbone_channels[1], stride=1),
        )

        # Stage 3: stride-2 → 1/16 resolution
        self.stage3 = nn.Sequential(
            ConvBlock(backbone_channels[1], backbone_channels[2], stride=2),
            ConvBlock(backbone_channels[2], backbone_channels[2], stride=1),
        )

        # --- Feature Projection ---
        # Project each backbone stage output to embed_dim for the decoder
        self.level_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
            )
            for ch in backbone_channels
        ])

        # --- Transformer Encoder at 1/16 scale ---
        # Adds global self-attention on the deepest features
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, transformer_depth)]
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                expansion_ratio=expansion_ratio,
                ffn_drop=dropout,
                drop_path=dpr[i],
            )
            for i in range(transformer_depth)
        ])
        self.transformer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[tuple[int, int]]]:
        """
        Forward pass through multi-scale encoder.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Tuple of:
                - List of feature maps per level, each (B, H_l*W_l, embed_dim).
                - List of spatial shapes per level, each (H_l, W_l).
        """
        # CNN backbone stages
        f1 = self.stage1(x)   # (B, C1, H/4,  W/4)
        f2 = self.stage2(f1)  # (B, C2, H/8,  W/8)
        f3 = self.stage3(f2)  # (B, C3, H/16, W/16)

        backbone_features = [f1, f2, f3]

        # Project all levels to embed_dim and flatten to sequences
        feature_list = []
        spatial_shapes = []

        for lvl, (feat, proj) in enumerate(zip(backbone_features, self.level_projs)):
            B, _, H_l, W_l = feat.shape
            spatial_shapes.append((H_l, W_l))

            # Project channels
            feat_proj = proj(feat)  # (B, embed_dim, H_l, W_l)

            if lvl == len(backbone_features) - 1:
                # Apply transformer self-attention on deepest features
                tokens = feat_proj.flatten(2).transpose(1, 2)  # (B, N, D)
                for block in self.transformer_blocks:
                    tokens = block(tokens)
                tokens = self.transformer_norm(tokens)
                feature_list.append(tokens)
            else:
                # Just flatten for CNN-only levels
                tokens = feat_proj.flatten(2).transpose(1, 2)  # (B, N, D)
                feature_list.append(tokens)

        return feature_list, spatial_shapes
