"""Cross-attention language conditioning for the four-character checkpoint."""

from __future__ import annotations

import torch
from torch import nn


class ConditioningAdapter(nn.Module):
    def __init__(self, kind: str, hidden_dim: int, text_dim: int, latent_dim: int = 128, bottleneck: int = 64) -> None:
        super().__init__()
        if kind != "cross_attention":
            raise ValueError("this checkpoint requires cross_attention conditioning")
        if min(hidden_dim, text_dim, latent_dim, bottleneck) <= 0 or latent_dim % 4:
            raise ValueError("conditioning dimensions must be positive; latent width must be divisible by four")
        self.hidden_dim, self.text_dim, self.latent_dim = hidden_dim, text_dim, latent_dim
        self.text_projection = nn.Linear(text_dim, latent_dim)
        self.query_projection = nn.Linear(hidden_dim, latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, 4, dropout=0.0, batch_first=True)
        self.output = nn.Linear(latent_dim, hidden_dim)

    def forward(
        self, hidden: torch.Tensor, tokens: torch.Tensor, token_mask: torch.Tensor, condition_mask: torch.Tensor
    ) -> torch.Tensor:
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError("hidden must have shape [B,H]")
        if (
            tokens.ndim != 3
            or tokens.shape[0] != hidden.shape[0]
            or tokens.shape[-1] != self.text_dim
            or tokens.shape[1] < 1
        ):
            raise ValueError("tokens must have shape [B,L,D] with L positive")
        if token_mask.shape != tokens.shape[:-1] or token_mask.dtype != torch.bool:
            raise ValueError("token mask must be boolean with shape [B,L]")
        if condition_mask.shape != hidden.shape[:-1] or condition_mask.dtype != torch.bool:
            raise ValueError("condition mask must be boolean with shape [B]")
        if not (hidden.device == tokens.device == token_mask.device == condition_mask.device):
            raise ValueError("conditioning tensors must share a device")
        active = condition_mask & token_mask.any(dim=-1)
        valid = token_mask & active.unsqueeze(-1)
        clean = torch.where(valid.unsqueeze(-1), tokens, 0.0)
        flat_valid = valid.reshape(-1, valid.shape[-1]).clone()
        flat_valid[:, 0] |= ~active.reshape(-1)
        keys = self.text_projection(clean).reshape(-1, tokens.shape[-2], self.latent_dim)
        queries = self.query_projection(hidden).reshape(-1, 1, self.latent_dim)
        attended, _ = self.attention(queries, keys, keys, key_padding_mask=~flat_valid, need_weights=False)
        delta = self.output(attended[:, 0]).reshape(*hidden.shape[:-1], self.hidden_dim)
        return torch.where(active.unsqueeze(-1), hidden + delta, hidden)
