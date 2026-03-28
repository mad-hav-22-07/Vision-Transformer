"""
Transformer Segmentation Decoder for LDTR-inspired Lane Detection.

Replaces the UNet-style upsampling decoder with a DETR-inspired transformer
decoder that uses:
    1. Learnable query embeddings that interact with multi-scale encoder features
    2. Deformable cross-attention for efficient feature aggregation
    3. Self-attention between queries for inter-lane reasoning
    4. Pixel decoder (FPN) to convert features back to full-resolution mask
    5. Auxiliary Gaussian heatmap head for lane centerline supervision

Architecture:
    Encoder features (multi-scale) 
        ↓
    N × DeformableCrossAttentionLayer (query refinement)
        ↓
    Query features → Mask prediction via dot-product with pixel features
        ↓
    Pixel Decoder (FPN upsample) → Full-resolution segmentation logits
    + Auxiliary heatmap branch (training only)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deformable_attention import DeformableCrossAttentionLayer


class PixelDecoder(nn.Module):
    """
    Feature Pyramid Network (FPN) that upsamples multi-scale features
    back to high resolution for final segmentation.

    Takes the multi-scale encoder features and query-modulated features,
    progressively upsamples and fuses them to produce a high-resolution
    feature map.

    Args:
        embed_dim:    Input channel dimension from encoder.
        fpn_channels: Intermediate channel dimension in FPN.
        num_levels:   Number of encoder feature levels.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        fpn_channels: int = 128,
        num_levels: int = 3,
    ):
        super().__init__()

        # Lateral convolutions (1x1) to reduce channel dim
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, fpn_channels, 1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.GELU(),
            )
            for _ in range(num_levels)
        ])

        # Output convolutions after fusion (3x3)
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(fpn_channels),
                nn.GELU(),
            )
            for _ in range(num_levels)
        ])

        # Final upsample refinement: from 1/4 to full resolution
        self.final_conv = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
        )

    def forward(
        self,
        feature_list: list[torch.Tensor],
        spatial_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        FPN forward pass.

        Args:
            feature_list:   List of L feature maps (B, H_l*W_l, D).
            spatial_shapes: List of L spatial shapes (H_l, W_l).

        Returns:
            High-resolution feature map (B, fpn_channels, H/4, W/4).
        """
        # Reshape sequences back to spatial feature maps and apply lateral convs
        laterals = []
        for lvl, (feat, (H_l, W_l)) in enumerate(zip(feature_list, spatial_shapes)):
            B = feat.shape[0]
            # (B, H*W, D) → (B, D, H, W)
            feat_2d = feat.transpose(1, 2).reshape(B, -1, H_l, W_l)
            lateral = self.lateral_convs[lvl](feat_2d)
            laterals.append(lateral)

        # Top-down pathway: fuse from coarsest to finest
        for i in range(len(laterals) - 1, 0, -1):
            # Upsample coarser level to match finer level
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            laterals[i - 1] = laterals[i - 1] + upsampled

        # Apply output convolutions
        fpn_outputs = []
        for i, lateral in enumerate(laterals):
            fpn_outputs.append(self.output_convs[i](lateral))

        # Use finest level (1/4 resolution) as the output
        output = fpn_outputs[0]
        output = self.final_conv(output)

        return output


class SegmentationDecoder(nn.Module):
    """
    LDTR-inspired Transformer Segmentation Decoder.

    Uses learnable queries with deformable cross-attention to encoder
    features, followed by FPN pixel decoder and per-pixel classification.

    Args:
        embed_dim:          Embedding dimension.
        num_classes:        Number of segmentation classes.
        num_queries:        Number of learnable queries (affects capacity).
        num_decoder_layers: Number of deformable decoder layers.
        num_heads:          Attention heads.
        num_levels:         Number of feature map scales from encoder.
        num_points:         Deformable sampling points per head per level.
        fpn_channels:       FPN intermediate channels.
        dropout:            Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_classes: int = 3,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
        fpn_channels: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_levels = num_levels

        # Learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_pos = nn.Embedding(num_queries, embed_dim)

        # Learnable reference points per query per level
        self.reference_points = nn.Linear(embed_dim, num_levels * 2)

        # Stack of deformable decoder layers
        self.decoder_layers = nn.ModuleList([
            DeformableCrossAttentionLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_levels=num_levels,
                num_points=num_points,
                ffn_dim=embed_dim * 4,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Pixel decoder (FPN)
        self.pixel_decoder = PixelDecoder(
            embed_dim=embed_dim,
            fpn_channels=fpn_channels,
            num_levels=num_levels,
        )

        # Query-to-pixel attention: project queries to modulate pixel features
        self.query_to_pixel = nn.Sequential(
            nn.Linear(embed_dim, fpn_channels),
            nn.GELU(),
        )

        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
            nn.Conv2d(fpn_channels, num_classes, 1),
        )

        # Auxiliary heatmap head (for training, predicts lane centerlines)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels // 2),
            nn.GELU(),
            nn.Conv2d(fpn_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize reference points to uniform grid."""
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.constant_(self.reference_points.bias, 0.5)

    def _get_reference_points(
        self, query: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate reference points from queries.

        Args:
            query: (B, N_q, D)

        Returns:
            Reference points (B, N_q, num_levels, 2) in [0, 1].
        """
        B = query.shape[0]
        # Use positional embedding to generate reference coords
        ref = self.reference_points(self.query_pos.weight)  # (N_q, num_levels * 2)
        ref = ref.sigmoid()  # Clamp to [0, 1]
        ref = ref.view(self.num_queries, self.num_levels, 2)
        ref = ref.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N_q, num_levels, 2)
        return ref

    def forward(
        self,
        feature_list: list[torch.Tensor],
        spatial_shapes: list[tuple[int, int]],
        target_size: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            feature_list:   List of encoder features per level (B, N_l, D).
            spatial_shapes: List of (H_l, W_l) per level.
            target_size:    (H, W) to resize output to.

        Returns:
            Tuple of:
                - Segmentation logits (B, num_classes, H, W).
                - Heatmap prediction (B, 1, H, W) if training, else None.
        """
        B = feature_list[0].shape[0]

        # Initialize queries
        query = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        query = query + self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)

        # Reference points
        ref_points = self._get_reference_points(query)

        # Deformable decoder layers
        for layer in self.decoder_layers:
            query = layer(query, ref_points, feature_list, spatial_shapes)
        query = self.decoder_norm(query)

        # Pixel decoder: multi-scale features → high-res feature map
        pixel_features = self.pixel_decoder(feature_list, spatial_shapes)
        # pixel_features: (B, fpn_channels, H/4, W/4)

        # Query-pixel modulation: queries attend to pixel features
        query_projected = self.query_to_pixel(query)  # (B, N_q, fpn_channels)

        # Compute attention map: queries × pixel features
        B, C, H_p, W_p = pixel_features.shape
        pixel_flat = pixel_features.flatten(2)  # (B, C, H_p*W_p)

        # (B, N_q, C) × (B, C, H_p*W_p) → (B, N_q, H_p*W_p)
        attn_map = torch.bmm(query_projected, pixel_flat)
        attn_map = attn_map / (C ** 0.5)
        attn_map = F.softmax(attn_map, dim=1)  # Normalize over queries

        # Weighted sum of query features for each pixel
        # (B, N_q, C)^T × (B, N_q, H_p*W_p) → (B, C, H_p*W_p)
        modulated = torch.bmm(query_projected.transpose(1, 2), attn_map)
        modulated = modulated.view(B, C, H_p, W_p)

        # Combine pixel and query-modulated features
        enhanced = pixel_features + modulated

        # Segmentation head
        logits = self.seg_head(enhanced)

        # Upsample to target size
        if target_size is not None:
            logits = F.interpolate(
                logits, size=target_size, mode="bilinear", align_corners=False
            )

        # Heatmap auxiliary branch (training only)
        heatmap = None
        if self.training:
            heatmap = self.heatmap_head(enhanced)
            if target_size is not None:
                heatmap = F.interpolate(
                    heatmap, size=target_size, mode="bilinear", align_corners=False
                )

        return logits, heatmap
