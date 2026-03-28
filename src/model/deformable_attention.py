"""
Multi-Scale Deformable Attention for LDTR-inspired Lane Segmentation.

Inspired by Deformable DETR and LDTR's Multi-Referenced Deformable Attention
(MRDA). Instead of attending to all spatial positions (O(N²)), each query
samples K learned offset positions per attention head across multi-scale
feature maps, making it efficient for high-resolution feature aggregation.

Key differences from standard attention:
    - Samples a fixed number of points (num_points) per query instead of all positions
    - Predicts spatial offsets relative to reference points
    - Operates across multiple feature map scales simultaneously
    - O(N × K × L) instead of O(N²) where K=points, L=levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleDeformableAttention(nn.Module):
    """
    Multi-Scale Deformable Attention Module.

    Each query predicts sampling offsets and attention weights for K points
    across L feature map levels, then aggregates the sampled features.

    Args:
        embed_dim:   Total embedding dimension.
        num_heads:   Number of attention heads.
        num_levels:  Number of feature map levels (scales).
        num_points:  Number of sampling points per head per level.
        dropout:     Dropout on output.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.head_dim = embed_dim // num_heads

        # Predict sampling offsets: 2D (x, y) offset per point per head per level
        self.sampling_offsets = nn.Linear(
            embed_dim, num_heads * num_levels * num_points * 2
        )

        # Predict attention weights for each sampled point
        self.attention_weights = nn.Linear(
            embed_dim, num_heads * num_levels * num_points
        )

        # Value projection
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize sampling offsets to spread around reference points."""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)

        # Initialize offsets as a grid pattern around the reference
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid_init = (
            grid_init / grid_init.abs().max(-1, keepdim=True)[0]
        )  # Normalize to [-1, 1]
        grid_init = grid_init.view(self.num_heads, 1, 1, 2).repeat(
            1, self.num_levels, self.num_points, 1
        )

        # Scale offsets by point index for diversity
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))

        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value_list: list[torch.Tensor],
        spatial_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            query:            Query features (B, N_q, D).
            reference_points: Normalized reference coords (B, N_q, num_levels, 2)
                             in range [0, 1].
            value_list:       List of L feature maps, each (B, H_l*W_l, D).
            spatial_shapes:   List of L tuples (H_l, W_l).

        Returns:
            Output features (B, N_q, D).
        """
        B, N_q, _ = query.shape

        # Project values for all levels
        value_projected = []
        for lvl, value in enumerate(value_list):
            v = self.value_proj(value)  # (B, H_l*W_l, D)
            H_l, W_l = spatial_shapes[lvl]
            # Reshape to (B, H_l, W_l, num_heads, head_dim)
            v = v.view(B, H_l, W_l, self.num_heads, self.head_dim)
            value_projected.append(v)

        # Predict sampling offsets: (B, N_q, num_heads * num_levels * num_points * 2)
        offsets = self.sampling_offsets(query)
        offsets = offsets.view(
            B, N_q, self.num_heads, self.num_levels, self.num_points, 2
        )

        # Predict attention weights: (B, N_q, num_heads * num_levels * num_points)
        attn_weights = self.attention_weights(query)
        attn_weights = attn_weights.view(
            B, N_q, self.num_heads, self.num_levels * self.num_points
        )
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.view(
            B, N_q, self.num_heads, self.num_levels, self.num_points
        )

        # Sample and aggregate features
        output = torch.zeros(
            B, N_q, self.num_heads, self.head_dim,
            device=query.device, dtype=query.dtype,
        )

        for lvl in range(self.num_levels):
            H_l, W_l = spatial_shapes[lvl]

            # Get sampling locations for this level
            # reference_points: (B, N_q, num_levels, 2) → (B, N_q, 1, 1, 2)
            ref = reference_points[:, :, lvl, :].unsqueeze(2).unsqueeze(3)

            # Normalize offsets by spatial size
            offset_normalizer = torch.tensor(
                [W_l, H_l], device=query.device, dtype=query.dtype
            )
            # offsets[:, :, :, lvl, :, :] → (B, N_q, num_heads, num_points, 2)
            lvl_offsets = offsets[:, :, :, lvl, :, :] / offset_normalizer

            # Sampling locations: ref + offset, convert to grid_sample format [-1, 1]
            sampling_locations = ref + lvl_offsets  # (B, N_q, num_heads, num_points, 2)
            sampling_grid = 2.0 * sampling_locations - 1.0  # [0,1] → [-1,1]

            # Value map: (B, H_l, W_l, num_heads, head_dim) → (B*num_heads, head_dim, H_l, W_l)
            val = value_projected[lvl]
            val = val.permute(0, 3, 4, 1, 2)  # (B, num_heads, head_dim, H_l, W_l)
            val = val.reshape(B * self.num_heads, self.head_dim, H_l, W_l)

            # Sampling grid: (B, N_q, num_heads, num_points, 2)
            #              → (B*num_heads, N_q, num_points, 2)
            grid = sampling_grid.permute(0, 2, 1, 3, 4)
            grid = grid.reshape(B * self.num_heads, N_q, self.num_points, 2)

            # Sample features using grid_sample
            sampled = F.grid_sample(
                val, grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )  # (B*num_heads, head_dim, N_q, num_points)

            # Reshape back: (B, num_heads, head_dim, N_q, num_points)
            sampled = sampled.view(B, self.num_heads, self.head_dim, N_q, self.num_points)
            # → (B, N_q, num_heads, num_points, head_dim)
            sampled = sampled.permute(0, 3, 1, 4, 2)

            # Weight and accumulate: attn_weights[:, :, :, lvl, :] → (B, N_q, num_heads, num_points, 1)
            w = attn_weights[:, :, :, lvl, :].unsqueeze(-1)
            output = output + (w * sampled).sum(dim=3)  # Sum over points

        # Reshape: (B, N_q, num_heads, head_dim) → (B, N_q, embed_dim)
        output = output.reshape(B, N_q, self.embed_dim)

        output = self.output_proj(output)
        output = self.dropout(output)

        return output


class DeformableCrossAttentionLayer(nn.Module):
    """
    Single decoder layer with deformable cross-attention + self-attention + FFN.

    Architecture (Pre-LN):
        x = x + SelfAttention(LN(x))
        x = x + DeformableCrossAttention(LN(x), encoder_features)
        x = x + FFN(LN(x))

    Args:
        embed_dim:      Embedding dimension.
        num_heads:      Number of attention heads.
        num_levels:     Number of encoder feature scales.
        num_points:     Deformable sampling points per head per level.
        ffn_dim:        FFN hidden dimension.
        dropout:        Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 3,
        num_points: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Deformable cross-attention to encoder features
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value_list: list[torch.Tensor],
        spatial_shapes: list[tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            query:            (B, N_q, D) query embeddings.
            reference_points: (B, N_q, num_levels, 2) normalized coords.
            value_list:       List of encoder features per level.
            spatial_shapes:   List of (H, W) per level.

        Returns:
            Updated queries (B, N_q, D).
        """
        # Self-attention
        q = self.norm1(query)
        q2 = self.self_attn(q, q, q)[0]
        query = query + self.dropout1(q2)

        # Deformable cross-attention
        q = self.norm2(query)
        q2 = self.cross_attn(q, reference_points, value_list, spatial_shapes)
        query = query + q2

        # FFN
        q = self.norm3(query)
        query = query + self.ffn(q)

        return query
