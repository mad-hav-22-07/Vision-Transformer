"""
Transformer Encoder Block for Vision Transformer.

A single transformer block consists of:
  1. Layer Norm + Multi-Head Self-Attention + Residual
  2. Layer Norm + Feed-Forward Network (MLP) + Residual

Uses Pre-LayerNorm architecture (norm before attention/FFN) which is
more stable during training compared to the original Post-LN design.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Two-layer MLP with GELU activation:
        Linear(D → 4D) → GELU → Dropout → Linear(4D → D) → Dropout

    Args:
        embed_dim:       Input/output dimension.
        expansion_ratio: FFN hidden dimension = embed_dim * expansion_ratio.
        dropout:         Dropout probability.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        expansion_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = embed_dim * expansion_ratio

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Single Transformer Encoder Block with Pre-LayerNorm.

    Architecture:
        x = x + MHSA(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Args:
        embed_dim:       Token embedding dimension.
        num_heads:       Number of attention heads.
        expansion_ratio: FFN expansion factor (default: 4).
        attn_drop:       Dropout on attention weights.
        proj_drop:       Dropout on attention output projection.
        ffn_drop:        Dropout inside FFN.
        drop_path:       Stochastic depth rate (0 = disabled).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        expansion_ratio: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.1,
        drop_path: float = 0.0,
    ):
        super().__init__()

        # Pre-norm before attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # Pre-norm before FFN
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardNetwork(
            embed_dim=embed_dim,
            expansion_ratio=expansion_ratio,
            dropout=ffn_drop,
        )

        # Stochastic depth for regularization (drops entire block output)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Token embeddings (B, N, D).

        Returns:
            Transformed token embeddings (B, N, D).
        """
        # Pre-norm + attention + residual
        x = x + self.drop_path(self.attn(self.norm1(x)))

        # Pre-norm + FFN + residual
        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x


class DropPath(nn.Module):
    """
    Stochastic Depth — randomly drops entire residual branch during training.

    This is a regularization technique that improves generalization by
    making the network robust to the removal of any single block.

    Args:
        drop_prob: Probability of dropping the path (0 = no drop).
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x

        keep_prob = 1 - self.drop_prob
        # Shape: (B, 1, 1, ...) — same drop decision for all tokens in a sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)

        # Scale output to maintain expected values
        output = x / keep_prob * random_tensor
        return output
