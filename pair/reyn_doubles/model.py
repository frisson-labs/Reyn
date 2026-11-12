"""Policy encoder, causal Transformer, action heads and autoregressive cache."""

from __future__ import annotations

import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from reyn_doubles.codec import ControllerLabels, CustomV1Codec
from reyn_doubles.config import ModelConfig


@dataclass
class KVCache:
    keys: torch.Tensor
    values: torch.Tensor
    valid_length: torch.Tensor
    write_position: torch.Tensor
    next_position: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.keys.shape[1])

    @property
    def capacity(self) -> int:
        return int(self.keys.shape[2])

    @property
    def n_layers(self) -> int:
        return int(self.keys.shape[0])

    def reset_slots(self, reset_mask: torch.Tensor) -> None:
        reset = reset_mask.to(device=self.keys.device, dtype=torch.bool)
        if reset.shape != (self.batch_size,):
            raise ValueError(f"reset_mask must have shape {(self.batch_size,)}")
        if not bool(reset.any()):
            return
        self.keys[:, reset] = 0
        self.values[:, reset] = 0
        self.valid_length[reset] = 0
        self.write_position[reset] = 0
        self.next_position[reset] = 0


def _chronological_cache_layer(
    cache: KVCache, layer_index: int, lengths: torch.Tensor, next_write: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 <= layer_index < cache.n_layers:
        raise IndexError("cache layer index out of range")
    batch = cache.batch_size
    capacity = cache.capacity
    keys = cache.keys.new_zeros((batch, capacity, cache.keys.shape[3], cache.keys.shape[4]))
    values = cache.values.new_zeros(keys.shape)
    valid = torch.zeros((batch, capacity), dtype=torch.bool, device=cache.keys.device)
    for batch_index in range(batch):
        length = int(lengths[batch_index].item())
        if length == 0:
            continue
        start = (int(next_write[batch_index].item()) - length) % capacity
        indices = (torch.arange(length, device=cache.keys.device) + start) % capacity
        keys[batch_index, :length] = cache.keys[layer_index, batch_index, indices]
        values[batch_index, :length] = cache.values[layer_index, batch_index, indices]
        valid[batch_index, :length] = True
    return (keys, values, valid)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-06, n_heads: int | None = None):
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm dimension must be positive")
        shape = (dim,) if n_heads is None else (n_heads, dim)
        self.weight = nn.Parameter(torch.ones(shape, dtype=torch.float32))
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        variance = inputs.float().square().mean(dim=-1, keepdim=True)
        normalized = (inputs.float() * torch.rsqrt(variance + self.eps)).to(inputs.dtype)
        weight = self.weight.to(inputs.dtype)
        if weight.ndim == 2:
            if inputs.ndim < 3 or inputs.shape[1] != weight.shape[0]:
                raise ValueError("head-wise RMSNorm expects shape [B, H, ..., Dh]")
            shape = (1, weight.shape[0], *[1] * (inputs.ndim - 3), weight.shape[1])
            weight = weight.view(shape)
        return normalized * weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim <= 0 or head_dim % 2:
            raise ValueError("RoPE head_dim must be a positive even number")
        inv_freq = 1.0 / theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.inv_freq: torch.Tensor
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(inputs: torch.Tensor) -> torch.Tensor:
        first, second = inputs.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    def apply_rotary(
        self, queries: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(queries.shape[0], -1)
        if positions.shape != (queries.shape[0], queries.shape[2]):
            raise ValueError("RoPE positions must have shape [B, T] or [T]")
        frequencies = positions.float().unsqueeze(-1) * self.inv_freq.float()
        embedding = torch.cat((frequencies, frequencies), dim=-1).unsqueeze(1)
        cosine = embedding.cos().to(queries.dtype)
        sine = embedding.sin().to(queries.dtype)
        queries = queries * cosine + self._rotate_half(queries) * sine
        keys = keys * cosine + self._rotate_half(keys) * sine
        return (queries, keys)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary(queries, keys, positions)


def _autocast_context(device: torch.device, dtype: torch.dtype) -> contextlib.AbstractContextManager[Any]:
    if dtype == torch.bfloat16 and device.type in ("cpu", "cuda"):
        return torch.autocast(device_type=device.type, dtype=dtype)
    if dtype == torch.float16 and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def _controller_labels(codec: CustomV1Codec, value: ControllerLabels) -> ControllerLabels:
    if value.buttons.shape != value.main_stick.shape:
        raise ValueError("controller label component shapes must match")
    for name, labels, size in zip(codec.component_order, value.values(), codec.vocabulary_sizes, strict=True):
        if labels.is_floating_point() or labels.is_complex():
            raise TypeError(f"{name} labels must have integer dtype")
        if bool(((labels < 0) | (labels >= size)).any()):
            raise ValueError(f"{name} labels must be in [0, {size})")
    return ControllerLabels(value.buttons.long(), value.main_stick.long())


class SlippiEncoder(nn.Module):
    ACTION_SIZE = 399
    CHARACTER_SIZE = 33
    STAGE_SIZE = 64
    JUMPS_SIZE = 7
    ITEM_COUNT = 15
    ITEM_TYPE_INPUT_SIZE = 236 + 1
    ITEM_TYPE_SIZE = ITEM_TYPE_INPUT_SIZE + 1
    ITEM_STATE_INPUT_SIZE = 12
    ITEM_STATE_SIZE = ITEM_STATE_INPUT_SIZE + 1
    ITEM_SIZE = 1 + ITEM_TYPE_SIZE + ITEM_STATE_SIZE + 2

    def __init__(self, config: ModelConfig, codec: CustomV1Codec | None = None):
        super().__init__()
        self.config = config
        self.codec = codec or CustomV1Codec()
        hidden = config.encoder_hidden_size
        self.character_embedding = nn.Embedding(self.CHARACTER_SIZE, hidden)
        self.action_embedding = nn.Embedding(self.ACTION_SIZE, hidden)
        self.character_action_embedding = nn.Embedding(self.CHARACTER_SIZE * self.ACTION_SIZE, hidden)
        item_layers: list[nn.Module] = []
        input_size = self.ITEM_SIZE
        for layer_index in range(config.item_mlp_layers):
            if layer_index:
                item_layers.append(nn.ReLU())
            linear = nn.Linear(input_size, hidden, bias=True)
            item_layers.append(linear)
            input_size = hidden
        self.item_mlp = nn.Sequential(*item_layers) if item_layers else nn.Identity()
        self.item_output_size = input_size
        self._output_dim = self._calculate_output_dim()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def _calculate_output_dim(self) -> int:
        action_size = self.config.encoder_hidden_size
        character_size = self.config.encoder_hidden_size
        player_core = 14 + action_size + character_size
        nana = player_core + 1
        players = player_core + nana + player_core + nana
        output = players + self.STAGE_SIZE
        output += 2
        output += 2
        output += self.item_output_size
        output += sum(self.codec.vocabulary_sizes)
        return output

    @staticmethod
    def _scaled_float(value: torch.Tensor, scale: float) -> torch.Tensor:
        return (value.float() * scale).clamp(-10.0, 10.0).unsqueeze(-1)

    def _embed_player_or_nana(self, player: Any, *, nana: bool) -> torch.Tensor:
        percent = player["percent"].long()
        percent = torch.remainder(percent, 1 << 16)
        facing_raw = player["facing"]
        facing = torch.where(
            facing_raw.bool() if facing_raw.dtype == torch.bool else facing_raw > 0, 1.0, -1.0
        )
        x = player["x"]
        y = player["y"]
        action = player["action"].long().clamp(0, self.ACTION_SIZE - 1)
        invulnerable = player["invulnerable"].bool()
        character = player["character"].long()
        if bool(((character < 0) | (character >= self.CHARACTER_SIZE)).any()):
            raise ValueError("character index is outside the upstream [0, 33) vocabulary")
        jumps = player["jumps_left"].long()
        if bool(((jumps < 0) | (jumps >= self.JUMPS_SIZE)).any()):
            raise ValueError("jumps_left is outside the upstream [0, 7) vocabulary")
        shield = player["shield_strength"]
        on_ground = player["on_ground"].bool()
        action_embed = self.action_embedding(action)
        joint_index = character * self.ACTION_SIZE + action
        action_embed = action_embed + self.character_action_embedding(joint_index)
        character_embed = self.character_embedding(character)
        parts = [
            self._scaled_float(percent, 0.01),
            facing.unsqueeze(-1),
            self._scaled_float(x, 0.05),
            self._scaled_float(y, 0.05),
            action_embed,
            invulnerable.float().unsqueeze(-1),
            character_embed,
            F.one_hot(jumps, self.JUMPS_SIZE).float(),
            self._scaled_float(shield, 0.01),
            on_ground.float().unsqueeze(-1),
        ]
        if nana:
            exists = player["exists"].bool()
            parts.append(exists.float().unsqueeze(-1))
        return torch.cat([part.float() for part in parts], dim=-1)

    def _embed_player(self, player: Mapping[str, Any]) -> torch.Tensor:
        return torch.cat(
            (
                self._embed_player_or_nana(player, nana=False),
                self._embed_player_or_nana(player["nana"], nana=True),
            ),
            dim=-1,
        )

    def _item_fields(
        self, items: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (items["exists"].bool(), items["type"].long(), items["state"].long(), items["x"], items["y"])

    def _embed_items(self, game: Any) -> torch.Tensor:
        exists, item_type, item_state, x, y = self._item_fields(game["items"])
        type_valid = (item_type >= 0) & (item_type < self.ITEM_TYPE_INPUT_SIZE)
        state_valid = (item_state >= 0) & (item_state < self.ITEM_STATE_INPUT_SIZE)
        item_type = torch.where(type_valid, item_type, self.ITEM_TYPE_INPUT_SIZE)
        item_state = torch.where(state_valid, item_state, self.ITEM_STATE_INPUT_SIZE)
        embedded = torch.cat(
            (
                exists.float().unsqueeze(-1),
                F.one_hot(item_type, self.ITEM_TYPE_SIZE).float(),
                F.one_hot(item_state, self.ITEM_STATE_SIZE).float(),
                self._scaled_float(x, 0.05),
                self._scaled_float(y, 0.05),
            ),
            dim=-1,
        )
        if embedded.shape[-2:] != (self.ITEM_COUNT, self.ITEM_SIZE):
            raise RuntimeError("internal item representation shape mismatch")
        transformed = self.item_mlp(embedded)
        transformed = torch.where(exists.unsqueeze(-1), transformed, 0.0)
        return transformed.sum(dim=-2).float()

    def _controller_embedding(self, labels: ControllerLabels) -> torch.Tensor:
        return torch.cat(
            (
                F.one_hot(labels.buttons.long(), self.codec.vocabulary_sizes[0]),
                F.one_hot(labels.main_stick.long(), self.codec.vocabulary_sizes[1]),
            ),
            dim=-1,
        ).float()

    def forward(self, game_state_t: Mapping[str, Any], controller_t: ControllerLabels) -> torch.Tensor:
        labels = _controller_labels(self.codec, controller_t)
        stage = game_state_t["stage"].long()
        if bool(((stage < 0) | (stage >= self.STAGE_SIZE)).any()):
            raise ValueError("stage is invalid for the 64-way vocabulary")
        randall, fod = (game_state_t["randall"], game_state_t["fod_platforms"])
        parts = [
            self._embed_player(game_state_t["p0"]),
            self._embed_player(game_state_t["p1"]),
            F.one_hot(stage, self.STAGE_SIZE).float(),
            self._scaled_float(randall["x"], 0.05),
            self._scaled_float(randall["y"], 0.05),
            self._scaled_float(fod["left"], 0.05),
            self._scaled_float(fod["right"], 0.05),
            self._embed_items(game_state_t),
            self._controller_embedding(labels),
        ]
        result = torch.cat([part.float() for part in parts], dim=-1)
        if result.shape[-1] != self.output_dim:
            raise RuntimeError(f"encoder produced {result.shape[-1]} features, expected {self.output_dim}")
        return result


class GroupedQueryCausalAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int = 0):
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.groups = config.n_heads // config.n_kv_heads
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.output_projection = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)
        self.attention_gate = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.norm_eps, n_heads=config.n_heads)
        self.k_norm = RMSNorm(config.head_dim, config.norm_eps, n_heads=config.n_kv_heads)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta)

    def _project(
        self, inputs: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, time, _ = inputs.shape
        queries = self.q_proj(inputs).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        keys = self.k_proj(inputs).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        values = self.v_proj(inputs).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        queries = self.q_norm(queries)
        keys = self.k_norm(keys)
        queries, keys = self.rope(queries, keys, positions)
        return (queries, keys, values)

    def _sdpa(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        dropout = 0.0
        if self.groups == 1:
            return F.scaled_dot_product_attention(
                queries, keys, values, attn_mask=attention_mask, dropout_p=dropout, is_causal=is_causal
            )
        try:
            return F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=attention_mask,
                dropout_p=dropout,
                is_causal=is_causal,
                enable_gqa=True,
            )
        except (TypeError, RuntimeError) as error:
            message = str(error).lower()
            if not any(word in message for word in ("gqa", "head", "enable_gqa")):
                raise
        keys = keys.repeat_interleave(self.groups, dim=1)
        values = values.repeat_interleave(self.groups, dim=1)
        return F.scaled_dot_product_attention(
            queries, keys, values, attn_mask=attention_mask, dropout_p=dropout, is_causal=is_causal
        )

    def _apply_output_gate(self, head_output: torch.Tensor, normalized_inputs: torch.Tensor) -> torch.Tensor:
        batch, _, time, _ = head_output.shape
        gate = torch.sigmoid(self.attention_gate(normalized_inputs).float())
        gate = gate.view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        gated = head_output.float() * gate
        flattened = gated.transpose(1, 2).reshape(batch, time, -1).to(normalized_inputs.dtype)
        return self.output_projection(flattened)

    def forward_cached(
        self,
        normalized_inputs: torch.Tensor,
        positions: torch.Tensor,
        cache: KVCache,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_inputs.shape[1] != 1:
            raise ValueError("cached attention accepts exactly one frame")
        queries, keys, values = self._project(normalized_inputs, positions)
        valid = valid_mask.to(device=cache.keys.device, dtype=torch.bool)
        for batch_index in range(cache.batch_size):
            if not bool(valid[batch_index]):
                continue
            write = int(cache.write_position[batch_index].item())
            cache.keys[self.layer_index, batch_index, write] = (
                keys[batch_index, :, 0].detach().to(cache.keys.dtype)
            )
            cache.values[self.layer_index, batch_index, write] = (
                values[batch_index, :, 0].detach().to(cache.values.dtype)
            )
        lengths = torch.minimum(
            cache.valid_length + valid.long(), torch.full_like(cache.valid_length, cache.capacity)
        )
        next_write = torch.where(
            valid, torch.remainder(cache.write_position + 1, cache.capacity), cache.write_position
        )
        chronological_k, chronological_v, key_valid = _chronological_cache_layer(
            cache, self.layer_index, lengths, next_write
        )
        chronological_k = chronological_k.permute(0, 2, 1, 3).to(queries.dtype)
        chronological_v = chronological_v.permute(0, 2, 1, 3).to(queries.dtype)
        mask = key_valid[:, None, None, :] & valid[:, None, None, None]
        head_output = self._sdpa(
            queries, chronological_k, chronological_v, attention_mask=mask, is_causal=False
        )
        head_output = torch.where(valid[:, None, None, None], head_output, 0.0)
        return self._apply_output_gate(head_output, normalized_inputs)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_projection = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up_projection = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down_projection = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down_projection(F.silu(self.gate_projection(inputs)) * self.up_projection(inputs))


class FullAttentionResidual(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-06):
        super().__init__()
        self.routing_norm = RMSNorm(d_model, eps)
        self.depth_query = nn.Parameter(torch.zeros(d_model, dtype=torch.float32))

    def forward(self, values: Sequence[torch.Tensor]) -> torch.Tensor:
        if not values:
            raise ValueError("FullAttentionResidual requires at least one source")
        stacked = torch.stack(tuple(values), dim=0)
        normalized_sources = self.routing_norm(stacked.float())
        scores = torch.einsum("d,sbtd->sbt", self.depth_query.float(), normalized_sources)
        weights = torch.softmax(scores.float(), dim=0)
        mixed = torch.einsum("sbt,sbtd->btd", weights, stacked.float())
        return mixed.to(stacked.dtype)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_index: int):
        super().__init__()
        self.config = config
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = GroupedQueryCausalAttention(config, layer_index)
        self.feed_forward = SwiGLU(config)
        self.attention_residual: FullAttentionResidual | None = FullAttentionResidual(
            config.d_model, config.norm_eps
        )
        self.ffn_residual: FullAttentionResidual | None = FullAttentionResidual(
            config.d_model, config.norm_eps
        )


class CausalTransformer(nn.Module):
    def __init__(self, config: ModelConfig, input_dim: int):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.config = config
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, config.d_model, bias=False)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_index) for layer_index in range(config.n_layers)]
        )
        self.output_residual = FullAttentionResidual(config.d_model, config.norm_eps)
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)

    def init_cache(
        self, batch_size: int, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> KVCache:
        if batch_size <= 0:
            raise ValueError("cache batch_size must be positive")
        parameter = self.input_projection.weight
        cache_device = parameter.device if device is None else torch.device(device)
        cache_dtype = self.config.torch_cache_dtype if dtype is None else dtype
        shape = (
            self.config.n_layers,
            batch_size,
            self.config.context_length,
            self.config.n_kv_heads,
            self.config.head_dim,
        )
        return KVCache(
            keys=torch.zeros(shape, device=cache_device, dtype=cache_dtype),
            values=torch.zeros(shape, device=cache_device, dtype=cache_dtype),
            valid_length=torch.zeros(batch_size, device=cache_device, dtype=torch.long),
            write_position=torch.zeros(batch_size, device=cache_device, dtype=torch.long),
            next_position=torch.zeros(batch_size, device=cache_device, dtype=torch.long),
        )

    def _validate_cache(self, cache: KVCache, batch_size: int, device: torch.device) -> None:
        expected = (
            self.config.n_layers,
            batch_size,
            self.config.context_length,
            self.config.n_kv_heads,
            self.config.head_dim,
        )
        if tuple(cache.keys.shape) != expected or tuple(cache.values.shape) != expected:
            raise ValueError(f"cache K/V shapes must both equal {expected}")
        if cache.keys.device != device or cache.values.device != device:
            raise ValueError("cache and encoded inputs must be on the same device")
        if not cache.keys.is_floating_point() or not cache.values.is_floating_point():
            raise TypeError("cache keys and values must have floating dtype")
        for name in ("valid_length", "write_position", "next_position"):
            tensor = cast(torch.Tensor, getattr(cache, name))
            if tensor.shape != (batch_size,) or tensor.device != device:
                raise ValueError(f"cache {name} must have shape [B] on the input device")
            if tensor.dtype != torch.long:
                raise TypeError(f"cache {name} must have torch.long dtype")
        if bool(((cache.valid_length < 0) | (cache.valid_length > cache.capacity)).any()):
            raise ValueError("cache valid_length is outside [0, capacity]")
        if bool(((cache.write_position < 0) | (cache.write_position >= cache.capacity)).any()):
            raise ValueError("cache write_position is outside [0, capacity)")
        if bool((cache.next_position < 0).any()):
            raise ValueError("cache next_position must be non-negative")

    @staticmethod
    def _ffn_transform(layer: TransformerBlock, inputs: torch.Tensor) -> torch.Tensor:
        return layer.feed_forward(layer.ffn_norm(inputs))

    def _cached_frame(
        self, encoded_frame: torch.Tensor, cache: KVCache, reset_mask: torch.Tensor, valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, KVCache]:
        cache.reset_slots(reset_mask)
        positions = cache.next_position[:, None]
        projected = self.input_projection(encoded_frame).unsqueeze(1)
        valid = valid_mask.to(device=encoded_frame.device, dtype=torch.bool)
        values: list[torch.Tensor] = [projected]
        for layer in cast(Sequence[TransformerBlock], self.layers):
            if layer.attention_residual is None or layer.ffn_residual is None:
                raise RuntimeError("full_attnres layer is missing its depth reads")
            attention_input = layer.attention_residual(values)
            attention_output = layer.attention.forward_cached(
                layer.attention_norm(attention_input), positions, cache, valid
            )
            values.append(attention_output)
            ffn_input = layer.ffn_residual(values)
            values.append(self._ffn_transform(layer, ffn_input))
        if self.output_residual is None:
            raise RuntimeError("full_attnres backbone is missing its final depth read")
        hidden = self.output_residual(values)
        hidden = self.final_norm(hidden).squeeze(1)
        hidden = torch.where(valid.unsqueeze(-1), hidden, 0.0)
        cache.valid_length.copy_(
            torch.minimum(
                cache.valid_length + valid.long(), torch.full_like(cache.valid_length, cache.capacity)
            )
        )
        cache.write_position.copy_(
            torch.where(
                valid, torch.remainder(cache.write_position + 1, cache.capacity), cache.write_position
            )
        )
        cache.next_position.add_(valid.long())
        return (hidden, cache)

    def step(
        self, encoded_frame: torch.Tensor, cache: KVCache, reset_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, KVCache]:
        if encoded_frame.ndim != 2 or encoded_frame.shape[-1] != self.input_dim:
            raise ValueError(f"encoded_frame must have shape [B, {self.input_dim}]")
        batch = encoded_frame.shape[0]
        self._validate_cache(cache, batch, encoded_frame.device)
        if self.training and torch.is_grad_enabled():
            raise RuntimeError("the rolling KV cache is inference-only; call eval() or no_grad()")
        reset = (
            torch.zeros(batch, dtype=torch.bool, device=encoded_frame.device)
            if reset_mask is None
            else reset_mask.to(device=encoded_frame.device, dtype=torch.bool)
        )
        if reset.shape != (batch,):
            raise ValueError("reset_mask must have shape [B]")
        valid = torch.ones(batch, dtype=torch.bool, device=encoded_frame.device)
        with _autocast_context(encoded_frame.device, self.config.torch_compute_dtype):
            return self._cached_frame(encoded_frame, cache, reset, valid)


class _AutoregressiveComponent(nn.Module):
    def __init__(self, vocabulary_size: int, residual_size: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        input_size = residual_size + vocabulary_size
        features = [residual_size] * depth + [vocabulary_size]
        for index, output_size in enumerate(features):
            if index:
                layers.append(nn.ReLU())
            linear = nn.Linear(input_size, output_size, bias=True)
            layers.append(linear)
            input_size = output_size
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Linear(vocabulary_size, residual_size, bias=True)
        self.vocabulary_size = vocabulary_size

    def logits(self, residual: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        previous_embedding = F.one_hot(previous.long(), self.vocabulary_size).to(residual.dtype)
        return self.encoder(torch.cat((residual, previous_embedding), dim=-1))

    def update(self, residual: torch.Tensor, component: torch.Tensor) -> torch.Tensor:
        embedding = F.one_hot(component.long(), self.vocabulary_size).to(residual.dtype)
        return residual + self.decoder(embedding)


@dataclass
class ControllerHeadOutput:
    logits: dict[str, torch.Tensor]
    labels: ControllerLabels


class AutoregressiveControllerHead(nn.Module):
    def __init__(self, config: ModelConfig, codec: CustomV1Codec | None = None):
        super().__init__()
        self.config = config
        self.codec = codec or CustomV1Codec()
        residual_size = config.controller_residual_size
        self.to_residual = nn.Linear(config.d_model, residual_size, bias=True)
        self.components = nn.ModuleDict(
            {
                name: _AutoregressiveComponent(vocabulary_size, residual_size, config.component_depth)
                for name, vocabulary_size in zip(
                    self.codec.component_order, self.codec.vocabulary_sizes, strict=True
                )
            }
        )

    @staticmethod
    def _component_values(labels: ControllerLabels) -> dict[str, torch.Tensor]:
        return {"buttons": labels.buttons, "main_stick": labels.main_stick}

    @staticmethod
    def _labels(values: Mapping[str, torch.Tensor]) -> ControllerLabels:
        return ControllerLabels(buttons=values["buttons"], main_stick=values["main_stick"])

    def sample(
        self,
        hidden: torch.Tensor,
        previous_controller: Any,
        *,
        temperature: float = 1.0,
        generator: torch.Generator | None = None,
        button_mask: torch.Tensor | None = None,
    ) -> ControllerHeadOutput:
        previous = _controller_labels(self.codec, previous_controller)
        if hidden.ndim < 2 or hidden.shape[-1] != self.config.d_model:
            raise ValueError(f"hidden must have shape [..., {self.config.d_model}]")
        prefix = hidden.shape[:-1]
        if previous.buttons.shape != prefix or previous.main_stick.shape != prefix:
            raise ValueError(f"previous controller labels must have shape {tuple(prefix)}")
        if previous.buttons.device != hidden.device or previous.main_stick.device != hidden.device:
            raise ValueError("previous controller labels and hidden state must share a device")
        if button_mask is not None:
            expected = (*prefix, self.codec.vocabulary_sizes[0])
            if (
                button_mask.shape != expected
                or button_mask.dtype != torch.bool
                or button_mask.device != hidden.device
            ):
                raise ValueError(f"button mask must be bool {expected} on the hidden-state device")
            if not bool(button_mask.any(dim=-1).all()):
                raise ValueError("button mask needs at least one allowed token per row")
        with _autocast_context(hidden.device, self.config.torch_compute_dtype):
            residual = self.to_residual(hidden)
            previous_values = self._component_values(previous)
            logits: dict[str, torch.Tensor] = {}
            selected: dict[str, torch.Tensor] = {}
            for name in self.codec.component_order:
                component = cast(_AutoregressiveComponent, self.components[name])
                component_logits = component.logits(residual, previous_values[name])
                if name == "buttons" and button_mask is not None:
                    component_logits = component_logits.masked_fill(~button_mask, -torch.inf)
                logits[name] = component_logits
                probabilities = torch.softmax(component_logits.float() / temperature, dim=-1)
                flat = probabilities.reshape(-1, probabilities.shape[-1])
                choice = torch.multinomial(flat, 1, generator=generator).reshape(component_logits.shape[:-1])
                selected[name] = choice
                residual = component.update(residual, choice)
        return ControllerHeadOutput(logits, self._labels(selected))


class MeleePolicy(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.codec = CustomV1Codec()
        self.encoder = SlippiEncoder(config, self.codec)
        self.backbone = CausalTransformer(config, input_dim=self.encoder.output_dim)
        self.controller_head = AutoregressiveControllerHead(config, self.codec)
        if config.torch_parameter_dtype != torch.float32:
            self.to(dtype=config.torch_parameter_dtype)
