"""
Multi-Head Self-Attention (MHSA) for Vision Transformer.

Implements scaled dot-product attention across multiple heads in parallel.
Each head attends to a different subspace of the embedding, allowing the
model to capture diverse spatial relationships between patches.

Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
"""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.

    Projects input tokens into Q, K, V representations across multiple
    heads, computes scaled dot-product attention independently per head,
    then concatenates and projects back.

    Args:
        embed_dim:  Total embedding dimension (split across heads).
        num_heads:  Number of attention heads.
        attn_drop:  Dropout on attention weights.
        proj_drop:  Dropout on the output projection.
        qkv_bias:   Whether to include bias in Q/K/V projections.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d_k)

        # Single linear layer computes Q, K, V together for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tokens of shape (B, N, D) where
               B = batch size, N = number of patches, D = embed_dim.

        Returns:
            Output tokens of shape (B, N, D) after self-attention.
        """
        B, N, D = x.shape

        # Compute Q, K, V in one shot: (B, N, 3*D) → (B, N, 3, num_heads, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)

        # Rearrange to (3, B, num_heads, N, head_dim) for easy slicing
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        # (B, num_heads, N, head_dim) @ (B, num_heads, head_dim, N) → (B, num_heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) → (B, num_heads, N, head_dim)
        x = attn @ v

        # Concatenate heads: (B, num_heads, N, head_dim) → (B, N, D)
        x = x.transpose(1, 2).reshape(B, N, D)

        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
