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


from torchvision.models import resnet50, ResNet50_Weights

class MultiScaleEncoder(nn.Module):
    """
    Multi-Scale Encoder with pretrained ResNet-50 backbone + Transformer top.

    Produces feature maps at 3 scales (1/4, 1/8, 1/16) for the
    deformable attention decoder to attend to.

    Args:
        in_channels:       Number of input image channels (3 for RGB).
        backbone_channels: Channel dimensions for each CNN stage (256, 512, 1024 for ResNet-50).
        embed_dim:         Transformer embedding dimension (applied at 1/16 scale).
        num_heads:         Number of attention heads for transformer blocks.
        transformer_depth: Number of transformer blocks at the deepest scale.
        expansion_ratio:   FFN expansion ratio in transformer blocks.
        dropout:           Dropout rate.
        drop_path_rate:    Maximum stochastic depth rate.
        freeze_backbone:   If True, freezes the ResNet backbone weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        backbone_channels: tuple[int, ...] = (256, 512, 1024),
        embed_dim: int = 256,
        num_heads: int = 8,
        transformer_depth: int = 6,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_feature_levels = len(backbone_channels)

        # --- Base ResNet-50 Backbone ---
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Stage 1: stride-4 → 1/4 resolution (256 channels)
        self.stage1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool, # maxpool causes total stride of 4
            resnet.layer1
        )

        # Stage 2: stride-2 → 1/8 resolution (512 channels)
        self.stage2 = resnet.layer2

        # Stage 3: stride-2 → 1/16 resolution (1024 channels)
        self.stage3 = resnet.layer3

        if freeze_backbone:
            for param in self.stage1.parameters():
                param.requires_grad = False
            for param in self.stage2.parameters():
                param.requires_grad = False
            for param in self.stage3.parameters():
                param.requires_grad = False

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
        f1 = self.stage1(x)   # (B, 256,  H/4,  W/4)
        f2 = self.stage2(f1)  # (B, 512,  H/8,  W/8)
        f3 = self.stage3(f2)  # (B, 1024, H/16, W/16)

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

