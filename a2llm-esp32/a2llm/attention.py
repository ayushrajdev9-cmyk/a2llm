"""Causal multi-head self-attention.

Math (per head, for a single query position q with keys/values k, v):

    score(q, k) = (q . k) / sqrt(d_head)     # scaled dot-product
    weights    = softmax(score, causal_mask)
    output     = sum_i weights_i * v_i

The causal mask sets score(i, j) = -inf for j > i so token i can only
attend to tokens 0..i (it cannot see the future).

This is an explicit, educational implementation (no fused kernels), which
makes the mechanism easy to read, test, and port to the C / ESP32 engine.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_causal_mask(seq_len: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Upper-triangular -inf mask of shape (seq_len, seq_len).

    mask[i, j] == 0 for j <= i (allowed), -inf for j > i (blocked).
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with a learned causal mask.

    All heads share one projection for queries, keys, values and one output
    projection; heads are just reshaped slices of the same tensor:

        input  (B, T, C) -> qkv (B, T, 3C) -> split into q, k, v
        q, k, v: (B, H, T, d_head)
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # One matrix producing query, key, value projections at once.
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _mask(self, seq_len: int) -> torch.Tensor:
        return build_causal_mask(seq_len, device=self.qkv.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, H, T, d_head) so each head attends independently.
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: (B, H, T, T) attention weights.
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn = attn + self._mask(T)                    # block future positions
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))
