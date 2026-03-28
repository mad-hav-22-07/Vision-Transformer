"""
ViT-LaneSeg — LDTR-inspired Lane Segmentation Model.

Complete model that assembles:
  - Multi-Scale Encoder (CNN backbone + Transformer) → Multi-level features
  - Transformer Decoder (deformable cross-attention + FPN) → Segmentation mask

Performs per-pixel semantic segmentation of lane markings into 3 classes:
  0: Background
  1: Dashed lane
  2: Solid lane

Architecture inspired by LDTR (Lane Detection Transformer) with:
  - Multi-scale feature pyramid from hybrid CNN-Transformer encoder
  - Deformable cross-attention for efficient feature aggregation
  - Learnable queries for global lane reasoning
  - Gaussian heatmap auxiliary supervision for lane structure
"""

import torch
import torch.nn as nn

from .encoder import MultiScaleEncoder
from .decoder import SegmentationDecoder


class ViTLaneSeg(nn.Module):
    """
    LDTR-inspired Vision Transformer for Lane Segmentation.

    End-to-end model: image → per-pixel lane class logits.

    Args:
        img_size:            Input image resolution (H, W).
        in_channels:         Number of input channels (3 for RGB).
        backbone_channels:   Channel dims for CNN backbone stages.
        embed_dim:           Transformer/decoder embedding dimension.
        num_heads:           Number of attention heads.
        num_classes:         Number of segmentation classes.
        transformer_depth:   Transformer blocks in encoder's deepest stage.
        num_decoder_layers:  Deformable decoder layers.
        num_queries:         Learnable queries for decoder.
        num_deform_points:   Sampling points per deformable attention head.
        fpn_channels:        FPN intermediate channels.
        expansion_ratio:     FFN expansion factor.
        dropout:             Global dropout rate.
        drop_path_rate:      Max stochastic depth rate.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (360, 640),
        in_channels: int = 3,
        backbone_channels: tuple[int, ...] = (64, 128, 256),
        embed_dim: int = 256,
        num_heads: int = 8,
        num_classes: int = 3,
        transformer_depth: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        num_deform_points: int = 4,
        fpn_channels: int = 128,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()

        self.img_size = img_size
        self.num_classes = num_classes

        # Multi-Scale Encoder
        self.encoder = MultiScaleEncoder(
            in_channels=in_channels,
            backbone_channels=backbone_channels,
            embed_dim=embed_dim,
            num_heads=num_heads,
            transformer_depth=transformer_depth,
            expansion_ratio=expansion_ratio,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )

        # Transformer Segmentation Decoder
        self.decoder = SegmentationDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_levels=len(backbone_channels),
            num_points=num_deform_points,
            fpn_channels=fpn_channels,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: image → segmentation logits.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            During training: tuple of (logits, heatmap)
                - logits:  (B, num_classes, H, W)
                - heatmap: (B, 1, H, W) auxiliary lane heatmap
            During inference: logits only (B, num_classes, H, W)
        """
        # Encode: extract multi-scale features
        feature_list, spatial_shapes = self.encoder(x)

        # Decode: deformable attention + FPN → segmentation mask
        logits, heatmap = self.decoder(
            feature_list,
            spatial_shapes,
            target_size=self.img_size,
        )

        if self.training:
            return logits, heatmap
        return logits

    def get_param_count(self) -> dict:
        """Returns parameter counts for each component."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params

        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "total": total_params,
            "total_M": f"{total_params / 1e6:.1f}M",
        }

    @classmethod
    def from_config(cls, config: dict) -> "ViTLaneSeg":
        """
        Create model from a configuration dictionary.

        Args:
            config: Dict with 'model' key containing architecture params.

        Returns:
            Initialized ViTLaneSeg model.
        """
        model_cfg = config.get("model", config)

        return cls(
            img_size=tuple(model_cfg.get("img_size", [360, 640])),
            in_channels=model_cfg.get("in_channels", 3),
            backbone_channels=tuple(model_cfg.get("backbone_channels", [64, 128, 256])),
            embed_dim=model_cfg.get("embed_dim", 256),
            num_heads=model_cfg.get("num_heads", 8),
            num_classes=model_cfg.get("num_classes", 3),
            transformer_depth=model_cfg.get("transformer_depth", 6),
            num_decoder_layers=model_cfg.get("num_decoder_layers", 6),
            num_queries=model_cfg.get("num_queries", 100),
            num_deform_points=model_cfg.get("num_deform_points", 4),
            fpn_channels=model_cfg.get("fpn_channels", 128),
            expansion_ratio=model_cfg.get("expansion_ratio", 4),
            dropout=model_cfg.get("dropout", 0.1),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
        )
