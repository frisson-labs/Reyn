"""Tensor-only, two-seat inference for Wombo Combo."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from reyn.codec import ControllerLabels
from reyn.config import ModelConfig
from reyn.inference import Action
from reyn.model import KVCache, MeleePolicy, _autocast_context
from reyn.observation import FRAME_SPEC, neutral_labels, prepare_input

_FILES = {
    "base.pt": "a3d3830bb0ed431970c0884ae9a8d9fea935050e6f97b0e7761cd246fee4a7c3",
    "adapter.pt": "14bbb0a1651390a1df053dd3e651f5b0597463bd6cc6df6eb022cd53948c782b",
    "feature_schema.json": "aa8b96d5bf70ef9761ec5e3d3cc389bf3fb8ec53b9dfb9f6ed88da4bdbde7736",
}
_SPEC = {
    "hidden_dim": 384,
    "width": 256,
    "features": 186,
    "categorical": {"partner.action": [39, 399], "last_move.mine": [175, 27], "last_move.partner": [176, 27]},
    "embed": 16,
    "gain": 1.0,
    "tables": {"partner.action": [399, 128]},
}


class _FeatureEncoder(nn.Module):
    table_0: torch.Tensor
    float_columns: torch.Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("table_0", torch.zeros(399, 128))
        self.register_buffer(
            "float_columns", torch.tensor([i for i in range(186) if i not in (39, 175, 176)]), persistent=False
        )
        self.learned = nn.ModuleList([nn.Embedding(27, 16), nn.Embedding(27, 16)])

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        action = features[..., 39].round().long().clamp(0, 398)
        mine = features[..., 175].round().long().clamp(0, 26)
        partner = features[..., 176].round().long().clamp(0, 26)
        return torch.cat(
            (
                features.index_select(-1, self.float_columns),
                F.embedding(action, self.table_0).to(features.dtype),
                self.learned[0](mine).to(features.dtype),
                self.learned[1](partner).to(features.dtype),
            ),
            dim=-1,
        )


class _PartnerAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FeatureEncoder()
        self.down_hidden = nn.Linear(384, 256)
        self.down_features = nn.Linear(343, 256, bias=False)
        self.mid = nn.Linear(256, 256)
        self.up = nn.Linear(256, 384)

    def forward(self, hidden: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        normed = F.layer_norm(hidden, (384,))
        inputs = self.encoder(features).to(hidden.dtype)
        z = F.gelu(self.down_hidden(normed) + self.down_features(inputs))
        output: torch.Tensor = self.up(F.gelu(self.mid(z)))
        return output


def _load(path: Path, expected_format: str) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or value.get("format") != expected_format:
        raise ValueError(f"Unsupported combo checkpoint: {path.name}")
    return value


class Combo:
    """Two independent attacker contexts sharing the frozen Wombo Combo weights.

    Each input row contains that attacker's base observation and its partner/scoring features.
    This specialized policy has no language-phrase interface.
    """

    trunk: MeleePolicy
    adapter: _PartnerAdapter
    cache: KVCache
    previous: ControllerLabels
    feature_schema: dict[str, Any]
    _device: torch.device
    _generator: torch.Generator
    _seed: int
    _first: torch.Tensor

    def __init__(self) -> None:
        raise TypeError("Use Combo.load(model_dir)")

    @classmethod
    def load(cls, model_dir: str | Path, *, device: str | torch.device = "cpu", seed: int = 0) -> Combo:
        """Load the hash-pinned Wombo Combo inference package, with frozen parameters."""
        root = Path(model_dir)
        metadata = json.loads((root / "config.json").read_text())
        if metadata.get("format") != "reyn.combo.v1":
            raise ValueError("Unsupported combo inference package")
        for name, expected in _FILES.items():
            with (root / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected or metadata["files"][name]["sha256"] != expected:
                raise ValueError(f"Checksum mismatch for combo inference asset: {name}")
        base = _load(root / "base.pt", "melee_policy.bc_checkpoint.v1")
        adapter = _load(root / "adapter.pt", "melee_rl.combo.partner_adapter.v1")
        if set(base) != {"format", "model_config", "codec", "state_dict"}:
            raise ValueError("Expected an inference-only base checkpoint")
        if set(adapter) != {"format", "spec", "state"} or adapter["spec"] != _SPEC:
            raise ValueError("Unsupported combo adapter specification")
        if base["codec"] != {"name": "custom_v1", "vocab_sizes": {"buttons": 728, "main_stick": 85}}:
            raise ValueError("Unsupported combo action codec")
        result = object.__new__(cls)
        result._device = torch.device(device)
        result.trunk = MeleePolicy(ModelConfig.from_checkpoint(base["model_config"]))
        result.trunk.load_state_dict(base["state_dict"], strict=True)
        result.adapter = _PartnerAdapter()
        result.adapter.load_state_dict(adapter["state"], strict=True)
        for module in (result.trunk, result.adapter):
            module.eval().requires_grad_(False).to(result.device)
        result.feature_schema = json.loads((root / "feature_schema.json").read_text())
        result.reset(seed=seed)
        return result

    @property
    def device(self) -> torch.device:
        return self._device

    def reset(self, *, seed: int | None = None) -> None:
        """Clear both row histories and restart the shared sampling stream."""
        if seed is not None:
            self._seed = int(seed)
        self._generator = torch.Generator(device=self.device).manual_seed(self._seed)
        self.cache = self.trunk.backbone.init_cache(2, device=self.device)
        self.previous = neutral_labels((2,), device=self.device)
        self._first = torch.ones(2, dtype=torch.bool, device=self.device)

    def example_input(self) -> tuple[dict[str, Any], torch.Tensor]:
        """Synthetic two-row tensors illustrating the input contract, not meaningful gameplay inputs."""
        dtypes = {"bool": torch.bool, "int": torch.int64, "float": torch.float32}

        def build(spec: Mapping[str, Any], path: str = "") -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, kind in spec.items():
                name = f"{path}.{key}" if path else key
                if isinstance(kind, Mapping):
                    result[key] = build(kind, name)
                else:
                    shape = (2, 15) if name.startswith("items.") else (2,)
                    value = 0.5 if ".main_stick." in name or ".c_stick." in name else 0
                    result[key] = torch.full(shape, value, dtype=dtypes[kind], device=self.device)
            return result

        frame = build(FRAME_SPEC)
        frame["p0"]["character"].fill_(1)
        frame["p1"]["character"].fill_(1)
        frame["stage"].fill_(25)
        features = torch.zeros((2, 186), dtype=torch.float32, device=self.device)
        features[:, 33] = 1  # Partner facing right.
        features[1, 185] = 1  # The second attacker's seat.
        return frame, features

    @torch.inference_mode()
    def predict(
        self,
        frame: Mapping[str, Any],
        features: torch.Tensor,
        *,
        reset: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> Action:
        """Predict both actions; reset rows clear only their own cache/previous labels, without reseeding."""
        if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if features.shape != (2, 186):
            raise ValueError("Combo features must have shape [2,186]")
        features = features.to(device=self.device, dtype=torch.float32)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Combo features must be finite")
        mask = self._first.clone()
        if reset is not None:
            if reset.shape != (2,):
                raise ValueError("reset must have shape [2]")
            mask |= reset.to(device=self.device, dtype=torch.bool)
        previous = ControllerLabels(
            buttons=self.previous.buttons.masked_fill(mask, 0),
            main_stick=self.previous.main_stick.masked_fill(mask, 0),
        )
        tree = prepare_input(frame, self.device, batch_size=2)
        with _autocast_context(self.device, self.trunk.config.torch_compute_dtype):
            encoded = self.trunk.encoder(tree, previous)
            hidden, _ = self.trunk.backbone.step(encoded, self.cache, mask)
            conditioned = hidden + self.adapter(hidden, features).to(hidden.dtype)
            output = self.trunk.controller_head.sample(
                conditioned, previous, temperature=temperature, generator=self._generator
            )
        self.previous = output.labels
        self._first.zero_()
        action = self.trunk.codec.decode(output.labels).as_packed_tensor()
        return Action(action, output.labels, {key: value.float() for key, value in output.logits.items()})
