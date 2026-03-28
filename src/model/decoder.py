"""
Segmentation Decoder for ViT-LaneSeg.

Progressive upsampling decoder that takes the transformer encoder's token
output and multi-scale skip features, reshapes them from sequence form
back to 2D spatial feature maps, and progressively upsamples to produce
a full-resolution segmentation mask.

Architecture:
    Reshape tokens → (B, D, H/16, W/16)
    Stage 1: Conv + Upsample 2× + skip  → (B, 256, H/8,  W/8)
    Stage 2: Conv + Upsample 2× + skip  → (B, 128, H/4,  W/4)
    Stage 3: Conv + Upsample 2× + skip  → (B, 64,  H/2,  W/2)
    Stage 4: Conv + Upsample 2×          → (B, 32,  H,    W)
    Head:    Conv 1×1                     → (B, C,   H,    W)   C = num_classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsampleBlock(nn.Module):
    """
    Single upsampling stage: fuse skip features, upsample 2×, refine with convolutions.

    Args:
        in_channels:   Number of input channels.
        skip_channels: Number of channels from the skip connection (0 = no skip).
        out_channels:  Number of output channels after this stage.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.has_skip = skip_channels > 0

        # If we have a skip connection, project it to match before concat
        if self.has_skip:
            self.skip_proj = nn.Sequential(
                nn.Conv2d(skip_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
            )
            conv_in = in_channels + out_channels
        else:
            conv_in = in_channels

        # Refinement convolutions after concatenation
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input feature map (B, C_in, H, W).
            skip: Optional skip feature map (B, C_skip, H*2, W*2) — at 2× resolution.

        Returns:
            Upsampled and refined feature map (B, C_out, H*2, W*2).
        """
        # Bilinear upsample 2×
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Fuse with skip connection
        if self.has_skip and skip is not None:
            skip = self.skip_proj(skip)
            # Handle size mismatches from non-power-of-2 dimensions
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)

        x = self.conv(x)
        return x


class SegmentationDecoder(nn.Module):
    """
    Progressive upsampling decoder with skip connections.

    Takes the final encoder tokens and multi-scale intermediate features,
    reshapes tokens to spatial form, then progressively upsamples through
    4 stages to produce a full-resolution segmentation mask.

    Args:
        embed_dim:    Transformer embedding dimension.
        num_classes:  Number of segmentation classes (3: bg, dashed, solid).
        grid_size:    Spatial grid size of encoder output (H/P, W/P).
        decoder_channels: Channel dimensions for each upsampling stage.
        num_skips:    Number of skip connections from encoder.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_classes: int = 3,
        grid_size: tuple[int, int] = (22, 40),
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
        num_skips: int = 3,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.num_classes = num_classes

        # Initial projection to reshape tokens → spatial feature map
        self.initial_proj = nn.Sequential(
            nn.Conv2d(embed_dim, decoder_channels[0], kernel_size=1, bias=False),
            nn.BatchNorm2d(decoder_channels[0]),
            nn.GELU(),
        )

        # Skip projection layers (transform encoder skip features to 2D spatial maps)
        self.skip_projections = nn.ModuleList()
        for i in range(num_skips):
            self.skip_projections.append(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
            )

        # Build upsampling stages
        # Stage 0: (256, H/16, W/16) + skip → (256, H/8, W/8)
        # Stage 1: (256, H/8,  W/8)  + skip → (128, H/4, W/4)
        # Stage 2: (128, H/4,  W/4)  + skip → (64,  H/2, W/2)
        # Stage 3: (64,  H/2,  W/2)          → (32,  H,   W)
        self.stages = nn.ModuleList()
        in_ch = decoder_channels[0]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = embed_dim if i < num_skips else 0
            self.stages.append(
                UpsampleBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                )
            )
            in_ch = out_ch

        # Final segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.GELU(),
            nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1),
        )

    def _tokens_to_spatial(
        self, tokens: torch.Tensor, grid_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Reshape token sequence to 2D spatial feature map.

        (B, N, D) → (B, D, grid_h, grid_w)
        """
        B, N, D = tokens.shape
        h, w = grid_size
        x = tokens.transpose(1, 2).reshape(B, D, h, w)
        return x

    def forward(
        self,
        encoder_output: torch.Tensor,
        skip_features: list[torch.Tensor],
        target_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: Final encoder tokens (B, N, D).
            skip_features:  List of intermediate encoder features, each (B, N, D).
                           Ordered from shallowest to deepest.
            target_size:    Optional (H, W) to resize output to. If None, uses
                           grid_size * patch_size * 16 (original image size).

        Returns:
            Segmentation logits (B, num_classes, H, W).
        """
        # Reshape encoder output to spatial: (B, N, D) → (B, D, H/16, W/16)
        x = self._tokens_to_spatial(encoder_output, self.grid_size)

        # Initial channel projection
        x = self.initial_proj(x)

        # Reshape skip features to spatial maps
        spatial_skips = []
        for i, skip in enumerate(skip_features):
            s = self._tokens_to_spatial(skip, self.grid_size)
            if i < len(self.skip_projections):
                s = self.skip_projections[i](s)
            spatial_skips.append(s)

        # Reverse skip features: decoder goes from deepest → shallowest
        spatial_skips = spatial_skips[::-1]

        # Progressive upsampling through stages
        for i, stage in enumerate(self.stages):
            skip = spatial_skips[i] if i < len(spatial_skips) else None
            x = stage(x, skip)

        # Segmentation head
        logits = self.head(x)

        # Resize to target size if needed
        if target_size is not None and logits.shape[2:] != target_size:
            logits = F.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

        return logits
