"""BERT-compatible text encoder for phrase routing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class EncoderConfig:
    vocab_size: int
    hidden: int
    layers: int
    heads: int
    intermediate: int
    max_positions: int = 512
    type_vocab: int = 2
    eps: float = 1e-12
    pooling: str = "mean"
    normalise: bool = False

    def __post_init__(self) -> None:
        if self.hidden % self.heads:
            raise ValueError(f"hidden {self.hidden} is not a multiple of heads {self.heads}")
        if self.pooling != "mean" or self.normalise:
            raise ValueError("unsupported text encoder; this router requires unnormalised mean pooling")


class _Embeddings(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden)
        self.position_embeddings = nn.Embedding(config.max_positions, config.hidden)
        self.token_type_embeddings = nn.Embedding(config.type_vocab, config.hidden)
        self.LayerNorm = nn.LayerNorm(config.hidden, eps=config.eps)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        positions = self.position_embeddings.weight[:length]
        segment = self.token_type_embeddings.weight[0]
        out: torch.Tensor = self.LayerNorm(self.word_embeddings(ids) + positions + segment)
        return out


class _SelfAttention(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.query = nn.Linear(config.hidden, config.hidden)
        self.key = nn.Linear(config.hidden, config.hidden)
        self.value = nn.Linear(config.hidden, config.hidden)

    def forward(self, hidden: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape

        def split(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, length, self.heads, width // self.heads).transpose(1, 2)

        mixed = F.scaled_dot_product_attention(
            split(self.query(hidden)),
            split(self.key(hidden)),
            split(self.value(hidden)),
            attn_mask=keep,
            dropout_p=0.0,
        )
        return mixed.transpose(1, 2).reshape(batch, length, width)


class _Residual(nn.Module):
    def __init__(self, inputs: int, config: EncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(inputs, config.hidden)
        self.LayerNorm = nn.LayerNorm(config.hidden, eps=config.eps)

    def forward(self, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.LayerNorm(self.dense(hidden) + residual)
        return out


class _Attention(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.self = _SelfAttention(config)
        self.output = _Residual(config.hidden, config)

    def forward(self, hidden: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.output(self.self(hidden, keep), hidden)
        return out


class _Intermediate(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden, config.intermediate)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.dense(hidden))


class _Layer(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.attention = _Attention(config)
        self.intermediate = _Intermediate(config)
        self.output = _Residual(config.intermediate, config)

    def forward(self, hidden: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        attended = self.attention(hidden, keep)
        out: torch.Tensor = self.output(self.intermediate(attended), attended)
        return out


class _Stack(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.layer = nn.ModuleList(_Layer(config) for _ in range(config.layers))


class BertEncoder(nn.Module):
    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = _Embeddings(config)
        self.encoder = _Stack(config)

    def hidden(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if ids.shape[1] > self.config.max_positions:
            raise ValueError(f"{ids.shape[1]} tokens exceed the encoder's {self.config.max_positions} positions")
        keep = mask.bool()[:, None, None, :]
        states: torch.Tensor = self.embeddings(ids)
        for layer in self.encoder.layer:
            states = layer(states, keep)
        return states

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        states = self.hidden(ids, mask)
        weights = mask.to(states.dtype)[..., None]
        return (states * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
