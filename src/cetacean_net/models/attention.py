"""Attention building blocks used by the CetaceanNet fusion models."""

from __future__ import annotations

import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """Residual multi-head self-attention followed by layer norm."""

    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x)
        return self.norm(attn_output + x)


class CrossAttentionBlock(nn.Module):
    """Residual cross-attention: ``query`` attends to a projected ``context``."""

    def __init__(self, query_dim: int, context_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=query_dim, num_heads=heads, batch_first=True)
        self.k_proj = nn.Linear(context_dim, query_dim)
        self.v_proj = nn.Linear(context_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        key = self.k_proj(context)
        value = self.v_proj(context)
        attn_output, _ = self.attn(query, key, value)
        return self.norm(attn_output + query)


class ModalFusionAttention(nn.Module):
    """Learned weighted sum over a stack of per-modality embeddings.

    Given a tensor of shape ``[B, M, dim]`` (``M`` modalities), a small gating
    network produces a scalar weight per modality; the modalities are then
    combined into a single ``[B, dim]`` embedding.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        weights = self.weight_net(embeddings).softmax(dim=1)
        return (weights * embeddings).sum(dim=1)
