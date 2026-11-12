"""Typed dimensions and precision settings for the supported inference architecture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
_SUPPORTED = {
    "framework": "pytorch",
    "encoder_type": "slippi_ai_enhanced",
    "use_self_nana": True,
    "use_controller_rnn": False,
    "use_learned_character": True,
    "use_learned_action": True,
    "use_character_action_joint": True,
    "use_item_sum": True,
    "use_items": True,
    "hybrid_embed": False,
    "use_randall": True,
    "use_fod_platforms": True,
    "condition_on_player_name": False,
    "player_name_vocab_size": 0,
    "action_offset_frames": 1,
    "action_codec": "custom_v1",
    "backbone_type": "causal_transformer",
    "attention_type": "full",
    "attention_output_gate": "elementwise_sigmoid",
    "attention_gate_bias": False,
    "attention_gate_compute_dtype": "float32",
    "residual_mode": "full_attnres",
    "norm": "rmsnorm",
    "qk_norm": True,
    "activation": "swiglu",
    "position_encoding": "rope",
    "bias": False,
    "attention_dropout": 0.0,
    "residual_dropout": 0.0,
    "attention_implementation": "sdpa",
    "use_kv_cache": True,
    "controller_head_type": "slippi_ai_autoregressive",
    "softmax_dtype": "float32",
}
_METADATA = {
    "gradient_checkpointing",
    "prevalidated_inputs",
    "profile",
    "controller_rnn_cell",
    "slippi_ai_commit",
    "initializer_std",
}


@dataclass(frozen=True)
class ModelConfig:
    encoder_hidden_size: int = 128
    item_mlp_layers: int = 2
    context_length: int = 256
    d_model: int = 384
    n_layers: int = 5
    n_heads: int = 6
    n_kv_heads: int = 2
    head_dim: int = 64
    d_ff: int = 768
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    component_depth: int = 2
    controller_residual_size: int = 128
    parameter_dtype: str = "float32"
    compute_dtype: str = "float32"
    cache_dtype: str = "float32"

    def __post_init__(self) -> None:
        positive = (
            self.encoder_hidden_size,
            self.context_length,
            self.d_model,
            self.n_layers,
            self.n_heads,
            self.n_kv_heads,
            self.head_dim,
            self.d_ff,
            self.controller_residual_size,
            self.norm_eps,
            self.rope_theta,
        )
        if min(positive) <= 0 or self.item_mlp_layers < 0 or self.component_depth < 0:
            raise ValueError("model dimensions must be positive and layer depths non-negative")
        if self.d_model != self.n_heads * self.head_dim or self.n_heads % self.n_kv_heads or self.head_dim % 2:
            raise ValueError("incompatible attention dimensions")
        if any(dtype not in _DTYPES for dtype in (self.parameter_dtype, self.compute_dtype, self.cache_dtype)):
            raise ValueError("unsupported precision dtype")

    @classmethod
    def from_checkpoint(cls, raw: Mapping[str, Any]) -> ModelConfig:
        dimensions = {field.name for field in fields(cls)}
        unknown = set(raw) - dimensions - set(_SUPPORTED) - _METADATA
        mismatched = [name for name, value in _SUPPORTED.items() if raw.get(name, value) != value]
        if unknown or mismatched:
            raise ValueError(f"unsupported inference architecture settings: {sorted(unknown | set(mismatched))}")
        return cls(**{key: value for key, value in raw.items() if key in dimensions})

    @property
    def torch_parameter_dtype(self) -> torch.dtype:
        return _DTYPES[self.parameter_dtype]

    @property
    def torch_compute_dtype(self) -> torch.dtype:
        return _DTYPES[self.compute_dtype]

    @property
    def torch_cache_dtype(self) -> torch.dtype:
        return _DTYPES[self.cache_dtype]
