"""Load model weights and predict actions conditioned on language and tensor inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from reyn.codec import ControllerLabels
from reyn.conditioning import ConditioningAdapter
from reyn.config import ModelConfig
from reyn.language.context import CHARACTERS
from reyn.language.features import CommandFeatures, load_command_features, with_fallback_rows
from reyn.language.router import Router
from reyn.language.stages import by_model_id
from reyn.language.steering import Reply, Steering
from reyn.model import KVCache, MeleePolicy, _autocast_context
from reyn.observation import FRAME_SPEC, neutral_labels, prepare_input
from reyn.state_adapter import StateAdapter, state_features
from reyn.state_adapter import from_payload as state_from_payload
from reyn.taunt import TauntHead, draw
from reyn.taunt import from_payload as taunt_from_payload

ASSET_FILES = ("rein_policy.pt", "adapter.pt", "features_fallback.pt", "router.pt")


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(path: Path, expected_format: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != expected_format:
        raise ValueError(f"{path.name}: unsupported checkpoint format")
    return payload


def _verify(root: Path, files: tuple[str, ...] = ASSET_FILES) -> None:
    for name in files:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Missing inference asset: {root / name}")
    checksum = root / "checksums.sha256"
    if not checksum.is_file():
        raise ValueError("Mandatory inference asset checksums missing")
    if checksum.exists():
        entries = {}
        for line in checksum.read_text().splitlines():
            if not line.strip():
                continue
            value, name = line.split(maxsplit=1)
            entries[name.lstrip("*")] = value
        for name in files:
            if name not in entries or _digest(root / name) != entries[name]:
                raise ValueError(f"Checksum mismatch or missing checksum for {name}")


@dataclass(frozen=True)
class Action:
    """Predicted action tensor, categorical component labels and component logits."""

    action: torch.Tensor
    labels: ControllerLabels
    logits: dict[str, torch.Tensor]


class Reyn:
    """One inference stream with held language conditioning and rolling model history."""

    trunk: MeleePolicy
    attachment: ConditioningAdapter
    features: CommandFeatures
    router: Router
    state_adapter: StateAdapter
    taunt: TauntHead
    modules: tuple[nn.Module, ...]
    cache: KVCache
    previous: ControllerLabels
    _device: torch.device
    _steering: Steering
    _characters: dict[str, int]
    _seed: int
    _first: bool
    _generator: torch.Generator
    _taunt_generator: torch.Generator

    def __init__(self) -> None:
        raise TypeError("Use Reyn.load(model_dir)")

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        character: str = "fox",
        seed: int = 0,
    ) -> Reyn:
        """Load the Reyn inference package without downloading or executing code."""
        root = Path(model_dir)
        _verify(root)
        base = _load(root / "rein_policy.pt", "melee_rl.rein_policy.v1")
        adapter = _load(root / "adapter.pt", "melee_rl.language_adapter.v1")
        if adapter.get("arm") != "rein":
            raise ValueError("This runtime supports released inference adapters only")
        expected = ((adapter.get("policy") or {}).get("rein") or {}).get("sha256")
        if not expected or _digest(root / "rein_policy.pt") != expected:
            raise ValueError("Steering weights do not match the base policy")
        config = {key: value for key, value in base["model_config"].items() if key != "prevalidated_inputs"}
        obj = object.__new__(cls)
        obj._characters = dict(CHARACTERS)
        obj.trunk = MeleePolicy(ModelConfig.from_checkpoint(config))
        obj.trunk.load_state_dict(base["policy"], strict=True)
        obj.trunk.to(device).eval().requires_grad_(False)
        obj._device = next(obj.trunk.parameters()).device
        obj.attachment = ConditioningAdapter(**adapter["spec"])
        obj.attachment.load_state_dict(adapter["attachment"], strict=True)
        table = adapter["features"]
        obj.features = CommandFeatures(
            texts=tuple(table["texts"]),
            tokens=table["tokens"],
            mask=table["mask"],
            encoder=table["encoder"],
        )
        router = Router.load(root / "router.pt")
        missing = [row for row in router.spec.rows() if row not in obj.features.texts]
        obj.features = with_fallback_rows(
            obj.features, load_command_features(root / "features_fallback.pt"), missing
        ).to(obj.device)
        if router.missing_rows(obj.features.texts):
            raise ValueError("The router and command representations are incompatible")
        obj.router = router
        obj.state_adapter = state_from_payload(adapter.get("state_adapter"))
        obj.taunt = taunt_from_payload(adapter.get("taunt"))
        obj.modules = (obj.trunk, obj.attachment, obj.state_adapter, obj.taunt, obj.router.model)
        for module in obj.modules:
            if module is not obj.router.model:
                module.to(obj.device)
            module.eval().requires_grad_(False)
        obj._steering = Steering(obj, obj.router, character=character)
        obj._seed = int(seed)
        obj.reset(character=character, seed=seed)
        return obj

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def current(self) -> int:
        return self._steering.current

    @property
    def character(self) -> str:
        return self._steering.character

    def command_ids(self, texts: list[str | None]) -> torch.Tensor:
        return torch.tensor([-1 if text is None else self.features.index(text) for text in texts], device=self.device)

    def reset(self, *, character: str | None = None, seed: int | None = None) -> None:
        """Clear language conditioning and cached autoregressive history."""
        who = self.character if character is None else character.lower()
        if who not in self._characters:
            raise ValueError(f"Unsupported character {who!r}; choose {tuple(self._characters)}")
        self._steering.character = who
        self.clear()
        self.cache = self.trunk.backbone.init_cache(1, device=self.device, dtype=self.trunk.config.torch_cache_dtype)
        self.previous = neutral_labels((1,), device=self.device)
        self._first = True
        if seed is not None:
            self._seed = int(seed)
        self._generator = torch.Generator(device=self.device).manual_seed(self._seed)
        self._taunt_generator = torch.Generator().manual_seed(self._seed)

    def set_context(self, character: str | int) -> None:
        """Reset history for a character context while retaining the same loaded weights."""
        if type(character) is int:
            character = next((key for key, value in self._characters.items() if value == character), "")
        if not isinstance(character, str):
            raise ValueError("A supported character key or model ID is required")
        self.reset(character=character)

    def say(self, text: str, *, stage: str | None = None) -> Reply:
        """Apply a request; a declined request keeps the previous command."""
        return self._steering.say(text, stage=stage)

    def clear(self) -> Reply:
        """Cancel deterministically, without invoking the language router."""
        changed = self.current != -1
        self._steering.current, self._steering.holding = -1, None
        return Reply("clear", None, "Cancel", changed, 1.0, "", asked="clear")

    def example_input(self) -> dict[str, Any]:
        """Synthetic model-input tensors for a minimal inference example."""
        dtypes = {"bool": torch.bool, "int": torch.int64, "float": torch.float32}

        def build(spec: Mapping[str, Any], path: str = "") -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, kind in spec.items():
                name = f"{path}.{key}" if path else key
                if isinstance(kind, Mapping):
                    result[key] = build(kind, name)
                else:
                    shape = (1, 15) if name.startswith("items.") else (1,)
                    value = 0.5 if ".main_stick." in name or ".c_stick." in name else 0
                    result[key] = torch.full(shape, value, dtype=dtypes[kind], device=self.device)
            return result

        result = build(FRAME_SPEC)
        context = self._characters[self.character]
        result["p0"]["character"].fill_(context)
        result["p1"]["character"].fill_(context)
        result["stage"].fill_(25)
        return result

    @torch.inference_mode()
    def predict(
        self,
        frame: Mapping[str, Any],
        *,
        phrase: str | None = None,
        temperature: float = 1.0,
    ) -> Action:
        """Predict one action from model tensors, optionally setting a new language phrase."""
        if not torch.isfinite(torch.tensor(temperature)) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        tree = prepare_input(frame, self.device)
        model_id = int(tree["p0"]["character"][0])
        context = next((key for key, value in self._characters.items() if value == model_id), None)
        if context is None:
            raise ValueError("Input context is not supported by this model release")
        if context != self.character:
            self.reset(character=context)
        if phrase is not None:
            try:
                stage = by_model_id(int(tree["stage"][0])).name
            except KeyError:
                stage = None
            self.say(phrase, stage=stage)
        command = torch.tensor([self.current], dtype=torch.long, device=self.device)
        reset = torch.tensor([self._first], dtype=torch.bool, device=self.device)
        facts = state_features(tree)
        with _autocast_context(self.device, self.trunk.config.torch_compute_dtype):
            encoded = self.trunk.encoder(tree, self.previous)
            hidden, _ = self.trunk.backbone.step(encoded, self.cache, reset)
            index = command.clamp(min=0)
            features = self.features
            conditioned = self.attachment(hidden, features.tokens[index], features.mask[index], command >= 0)
            conditioned = conditioned + self.state_adapter(hidden, command, facts)
            output = self.trunk.controller_head.sample(
                conditioned, self.previous, temperature=temperature, generator=self._generator
            )
        controller = self.trunk.codec.decode(output.labels).as_packed_tensor()
        if bool(self.taunt.active(command).any()):
            row = torch.cat((hidden, facts.to(hidden.dtype)), dim=-1)
            logits = self.taunt(row, command)
            pressed = draw(logits, self.taunt.active(command), self._taunt_generator).to(self.device)
            if bool(pressed.any()):
                controller = controller.clone()
                controller[pressed] = 0
                controller[pressed, :4] = 0.5
                controller[pressed, 12] = 1
        self._first = False
        self.previous = self.trunk.codec.encode(controller)
        return Action(controller, output.labels, {name: value.float() for name, value in output.logits.items()})
