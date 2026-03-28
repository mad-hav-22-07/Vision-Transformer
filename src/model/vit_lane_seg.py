"""
ViT-LaneSeg — Vision Transformer for Lane Segmentation.

Complete model that assembles:
  - Patch Embedding → Positional Encoding → ViT Encoder (with skips) → Segmentation Decoder

Performs per-pixel semantic segmentation of lane markings into 3 classes:
  0: Background
  1: Dashed lane
  2: Solid lane

Designed for TensorRT-friendly export (no dynamic control flow, standard ops).
"""

import torch
import torch.nn as nn

from .encoder import ViTEncoder
from .decoder import SegmentationDecoder


class ViTLaneSeg(nn.Module):
    """
    Vision Transformer for Lane Segmentation.

    End-to-end model: image → per-pixel lane class logits.

    Args:
        img_size:        Input image resolution (H, W).
        patch_size:      Patch size in pixels.
        in_channels:     Number of input channels (3 for RGB).
        embed_dim:       Transformer embedding dimension.
        depth:           Number of transformer encoder blocks.
        num_heads:       Number of attention heads.
        num_classes:     Number of segmentation classes.
        expansion_ratio: FFN expansion factor.
        attn_drop:       Attention dropout rate.
        proj_drop:       Projection dropout rate.
        ffn_drop:        FFN and positional encoding dropout rate.
        drop_path_rate:  Maximum stochastic depth rate.
        decoder_channels: Tuple of channel dims for decoder stages.
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (360, 640),
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        num_classes: int = 3,
        expansion_ratio: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.1,
        drop_path_rate: float = 0.1,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Encoder
        self.encoder = ViTEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            expansion_ratio=expansion_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            drop_path_rate=drop_path_rate,
        )

        grid_size = self.encoder.patch_embed.grid_size
        num_skips = len(self.encoder.skip_indices)

        # Decoder
        self.decoder = SegmentationDecoder(
            embed_dim=embed_dim,
            num_classes=num_classes,
            grid_size=grid_size,
            decoder_channels=decoder_channels,
            num_skips=num_skips,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: image → segmentation logits.

        Args:
            x: Input images (B, 3, H, W).

        Returns:
            Segmentation logits (B, num_classes, H, W).
        """
        # Encode: extract multi-scale features
        encoder_output, skip_features = self.encoder(x)

        # Decode: upsample to full resolution segmentation map
        logits = self.decoder(
            encoder_output,
            skip_features,
            target_size=self.img_size,
        )

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
            patch_size=model_cfg.get("patch_size", 16),
            in_channels=model_cfg.get("in_channels", 3),
            embed_dim=model_cfg.get("embed_dim", 512),
            depth=model_cfg.get("depth", 12),
            num_heads=model_cfg.get("num_heads", 8),
            num_classes=model_cfg.get("num_classes", 3),
            expansion_ratio=model_cfg.get("expansion_ratio", 4),
            attn_drop=model_cfg.get("attn_drop", 0.0),
            proj_drop=model_cfg.get("proj_drop", 0.0),
            ffn_drop=model_cfg.get("dropout", 0.1),
            drop_path_rate=model_cfg.get("drop_path_rate", 0.1),
            decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 32])),
        )
