"""Frozen Fox doubles prediction with independent fighter and enemy-view histories."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from reyn_doubles.adapter import PartnerAdapter
from reyn_doubles.codec import ControllerLabels
from reyn_doubles.commands import CommandState
from reyn_doubles.conditioning import ConditioningAdapter
from reyn_doubles.config import ModelConfig
from reyn_doubles.features import build_features
from reyn_doubles.model import KVCache, MeleePolicy, _autocast_context
from reyn_doubles.observation import FRAME_SPEC, neutral_labels, prepare_input
from reyn_doubles.schema import Roster, WorldFrame
from reyn_doubles.state_adapter import StateAdapter, state_features
from reyn_doubles.state_adapter import from_payload as state_from_payload
from reyn_doubles.targets import TargetSelector
from reyn_doubles.taunt import TauntHead, draw
from reyn_doubles.taunt import from_payload as taunt_from_payload

ASSETS = ("policy.pt", "steering.pt", "config.json")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path) -> None:
    """Require a checksum for every mandatory inference asset before deserialization."""
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        raise ValueError("Missing mandatory checksums.sha256")
    checksums = {}
    for line in manifest.read_text().splitlines():
        value, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        if name in checksums or "/" in name or "\\" in name:
            raise ValueError("Invalid or duplicate checksum entry")
        checksums[name] = value
    for name in ASSETS:
        if name not in checksums or not (root / name).is_file() or digest(root / name) != checksums[name]:
            raise ValueError(f"Missing asset or checksum mismatch: {name}")


def tree_map(function: Any, tree: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: tree_map(function, value) if isinstance(value, Mapping) else function(value)
        for key, value in tree.items()
    }


@dataclass(frozen=True)
class ViewStep:
    hidden: torch.Tensor
    labels: ControllerLabels
    logits: dict[str, torch.Tensor]
    log_prob: torch.Tensor


@dataclass(frozen=True)
class Action:
    """Two predicted controllers, ordered by the controlled team's physical ports."""

    action: torch.Tensor
    labels: ControllerLabels
    logits: dict[str, torch.Tensor]
    targets: tuple[int, int]
    commands: tuple[str | None, str | None]
    start: torch.Tensor


class Doubles:
    """Two Fox teammates sharing weights, with four independent rolling view caches."""

    trunk: MeleePolicy
    adapter: PartnerAdapter
    attachment: ConditioningAdapter
    state_adapter: StateAdapter | None
    taunt: TauntHead | None
    modules: tuple[torch.nn.Module, ...]
    _device: torch.device
    texts: tuple[str, ...]
    tokens: torch.Tensor
    mask: torch.Tensor
    menu: tuple[dict[str, Any], ...]
    _commands: CommandState
    _allowed_rows: frozenset[int]
    _safe_buttons: torch.Tensor | None

    def __init__(self) -> None:
        raise TypeError("Use Doubles.load(model_dir)")

    @classmethod
    def load(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        team: int = 0,
        seed: int = 0,
    ) -> Doubles:
        root = Path(model_dir)
        verify(root)
        config = json.loads((root / "config.json").read_text())
        if config.get("format") != "reyn.doubles.v1" or config.get("characters") != ["fox"]:
            raise ValueError("Unsupported doubles inference package")
        base = torch.load(root / "policy.pt", map_location="cpu", weights_only=True)
        steering = torch.load(root / "steering.pt", map_location="cpu", weights_only=True)
        if (
            base.get("format") != "reyn.doubles.policy.v1"
            or steering.get("format") != "reyn.doubles.steering.v1"
        ):
            raise ValueError("Unsupported inference weight format")
        if base["source"] != config["source"] or steering["source"] != config["source"]:
            raise ValueError("Doubles and language weights belong to different exports")
        obj = object.__new__(cls)
        obj.trunk = MeleePolicy(ModelConfig.from_checkpoint(base["model_config"]))
        obj.trunk.load_state_dict(base["trunk"], strict=True)
        payload = base["doubles"]
        if payload["format"] != "melee_rl.combo.partner_adapter.v1":
            raise ValueError("Unsupported doubles residual history format")
        spec = dict(payload["spec"])
        spec["categorical"] = {name: tuple(value) for name, value in spec["categorical"].items()}
        spec["tables"] = {name: torch.zeros(*shape) for name, shape in spec["tables"].items()}
        obj.adapter = PartnerAdapter(**spec)
        obj.adapter.load_state_dict(payload["state"], strict=True)
        obj.attachment = ConditioningAdapter(**steering["spec"])
        obj.attachment.load_state_dict(steering["attachment"], strict=True)
        obj.state_adapter = (
            state_from_payload(steering["state_adapter"]) if steering.get("state_adapter") else None
        )
        obj.taunt = taunt_from_payload(steering["taunt"]) if steering.get("taunt") else None
        obj._device = torch.device(device)
        obj.modules = tuple(
            x for x in (obj.trunk, obj.adapter, obj.attachment, obj.state_adapter, obj.taunt) if x is not None
        )
        for module in obj.modules:
            module.to(obj.device).eval().requires_grad_(False)
        table = steering["features"]
        obj.texts = tuple(table["texts"])
        obj.tokens = table["tokens"].to(obj.device)
        obj.mask = table["mask"].to(obj.device)
        if (
            obj.tokens.ndim != 3
            or obj.mask.shape != obj.tokens.shape[:2]
            or len(obj.texts) != obj.tokens.shape[0]
        ):
            raise ValueError("Invalid cached language table")
        obj.menu = tuple(config["commands"])
        phrases = [row["conditioning"] for row in obj.menu]
        if any(text not in obj.texts for text in phrases):
            raise ValueError("A supported phrase has no cached conditioning")
        obj._commands = CommandState(phrases)
        obj._allowed_rows = frozenset(obj.texts.index(text) for text in phrases)
        obj._friendly_fire_mask = False
        obj._friendly_fire_radius = 30.0
        obj._safe_buttons = None
        obj._seed = int(seed)
        obj.reset(team=team, seed=seed)
        return obj

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def current(self) -> tuple[str | None, str | None]:
        return self._commands.current

    @property
    def friendly_fire_mask_enabled(self) -> bool:
        return self._friendly_fire_mask

    def set_friendly_fire_mask(self, enabled: bool, *, radius: float = 30.0) -> None:
        """Optionally block attack inputs near a teammate; this is a proximity heuristic."""
        if type(enabled) is not bool:
            raise TypeError("enabled must be a bool")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        if enabled and self._safe_buttons is None:
            labels = torch.arange(self.trunk.codec.vocabulary_sizes[0], device=self.device)
            decoded = self.trunk.codec.decode(ControllerLabels(labels, torch.zeros_like(labels)))
            attack = decoded.buttons.A | decoded.buttons.B | decoded.buttons.Z
            attack |= (decoded.c_stick.x - 0.5).abs() > 1e-6
            attack |= (decoded.c_stick.y - 0.5).abs() > 1e-6
            self._safe_buttons = ~attack
        self._friendly_fire_radius = float(radius)
        self._friendly_fire_mask = enabled

    def _button_mask(self, world: WorldFrame) -> torch.Tensor | None:
        if not self._friendly_fire_mask:
            return None
        ally = self.roster.perspectives()[0, self.seats, 1]
        offset = world.position[0, self.seats] - world.position[0, ally]
        nearby = np.linalg.norm(offset, axis=-1) <= self._friendly_fire_radius
        hittable = world.actor_valid[0, ally] & ~world.respawning[0, ally] & ~world.invulnerable[0, ally]
        risk = torch.as_tensor(nearby & hittable, device=self.device).repeat_interleave(2)
        if self._safe_buttons is None:
            raise RuntimeError("Friendly-fire mask has not been initialized")
        return ~risk[:, None] | self._safe_buttons[None, :]

    def say(self, phrase: str, *, recipient: str = "both") -> tuple[str | None, str | None]:
        """Hold one exact tested language instruction on both Foxes or one Fox."""
        return self._commands.say(phrase, recipient=recipient)

    def clear(self, *, recipient: str = "both") -> tuple[str | None, str | None]:
        """Cancel language conditioning for the recipient; preserve all view histories."""
        return self._commands.clear(recipient=recipient)

    def init_cache(self, rows: int = 4) -> KVCache:
        return self.trunk.backbone.init_cache(
            rows, device=self.device, dtype=self.trunk.config.torch_cache_dtype
        )

    def reset(self, *, team: int | None = None, seed: int | None = None) -> None:
        """Start a fresh sequence, clearing commands and all four view histories."""
        team = getattr(self, "team", 0) if team is None else team
        if type(team) is not int or team not in (0, 1):
            raise ValueError("team must be 0 or 1")
        self.team = team
        self.roster = Roster.fox(1)
        self.seats = np.array([2 * team, 2 * team + 1], dtype=np.int64)
        self.view_rows = (2 * self.seats[:, None] + np.arange(2)[None, :]).reshape(-1)
        self.selector = TargetSelector(self.roster)
        self.cache = self.init_cache()
        self.previous = neutral_labels((4,), device=self.device)
        self._absent = np.zeros((1, 4), dtype=np.int64)
        self._share_pending = np.zeros((1, 4), dtype=np.bool_)
        self.clear()
        self._first = True
        if seed is not None:
            self._seed = int(seed)
        self._generator = torch.Generator(device=self.device).manual_seed(self._seed)
        self._taunt_generator = torch.Generator().manual_seed(self._seed)

    def command_ids(self) -> torch.Tensor:
        return torch.tensor(
            [-1 if text is None else self.texts.index(text) for text in self.current], device=self.device
        )

    def condition(
        self, hidden: torch.Tensor, features: torch.Tensor, command: torch.Tensor, tree: Mapping[str, Any]
    ) -> torch.Tensor:
        """Compose the two trained residuals on the original unconditioned hidden state."""
        doubles = hidden + self.adapter(hidden, features).to(hidden.dtype)
        index = command.clamp(min=0)
        language = self.attachment(hidden, self.tokens[index], self.mask[index], command >= 0)
        if self.state_adapter is not None:
            language = language + self.state_adapter(hidden, command, state_features(tree)).to(language.dtype)
        return doubles + (language - hidden).to(doubles.dtype)

    @torch.inference_mode()
    def step_views(
        self,
        tree: Mapping[str, Any],
        features: torch.Tensor,
        previous: ControllerLabels,
        cache: KVCache,
        *,
        command: torch.Tensor,
        reset: torch.Tensor | None = None,
        generator: torch.Generator,
        temperature: float = 1.0,
        button_mask: torch.Tensor | None = None,
    ) -> ViewStep:
        """Low-level tensor step: sample both enemy views, keeping each view's cache."""
        if features.shape != (cache.batch_size, 94) or command.shape != (cache.batch_size,):
            raise ValueError("Expected one 94-feature row and command per cached enemy view")
        if any(int(row) not in self._allowed_rows and int(row) != -1 for row in command.tolist()):
            raise ValueError("Only supported cached command bindings or OFF (-1) are allowed")
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        with _autocast_context(self.device, self.trunk.config.torch_compute_dtype):
            encoded = self.trunk.encoder(tree, previous)
            hidden, _ = self.trunk.backbone.step(encoded, cache, reset)
            conditioned = self.condition(hidden, features, command, tree)
            output = self.trunk.controller_head.sample(
                conditioned, previous, temperature=temperature, generator=generator, button_mask=button_mask
            )
        log_prob = torch.zeros_like(command, dtype=torch.float32)
        for name, labels in zip(self.trunk.codec.component_order, output.labels.values(), strict=True):
            log_prob += (
                torch.log_softmax(output.logits[name].float(), -1).gather(-1, labels[..., None]).squeeze(-1)
            )
        return ViewStep(
            hidden, output.labels, {name: value.float() for name, value in output.logits.items()}, log_prob
        )

    def _check_views(self, world: WorldFrame, tree: Mapping[str, Any]) -> None:
        if not bool((tree["stage"] == int(world.stage[0])).all()):
            raise ValueError("The four-player world and view tensors must have the same stage")
        enemies = self.roster.perspectives()[0, :, 2:]
        for seat in range(4):
            for view in range(2):
                row = 2 * seat + view
                for name, fighter in (("p0", seat), ("p1", int(enemies[seat, view]))):
                    player = tree[name]
                    if int(player["character"][row]) != 1:
                        raise ValueError("This experimental release supports Fox only")
                    if bool(world.present[0, fighter]):
                        if int(player["action"][row]) != int(world.action[0, fighter]):
                            raise ValueError("View action state does not match its physical fighter")
                        for key, index in (("x", 0), ("y", 1)):
                            if not np.isclose(
                                float(player[key][row]), float(world.position[0, fighter, index]), atol=1e-5
                            ):
                                raise ValueError(
                                    "Enemy-view tensor ordering does not match the four-player world"
                                )

    @torch.inference_mode()
    def predict(self, world: WorldFrame, views: Mapping[str, Any], *, temperature: float = 1.0) -> Action:
        """Predict the controlled Fox pair from a four-player state and eight stable view rows."""
        world.check_roster(self.roster)
        tree = prepare_input(views, self.device, batch_size=8)
        self._check_views(world, tree)
        self._absent = np.where(~world.present, self._absent + 1, 0)
        self._share_pending[world.stocks > 0] = False
        allies = self.roster.perspectives()[:, :, 1]
        ally_stocks = world.stocks[np.arange(1)[:, None], allies]
        absent = ~world.present & (self._absent >= 60)
        shares = absent & (world.stocks == 0) & (ally_stocks >= 2) & ~self._share_pending
        self._share_pending |= shares
        targets, _ = self.selector.update(world)
        encoded_features = build_features(world, targets)
        features = np.concatenate(
            [encoded_features.continuous, encoded_features.actions, encoded_features.characters], -1
        )
        feature_tensor = torch.as_tensor(
            features[0, self.seats], dtype=torch.float32, device=self.device
        ).repeat_interleave(2, 0)
        rows = torch.as_tensor(self.view_rows, device=self.device)
        mine = tree_map(lambda value: value.index_select(0, rows), tree)
        command = self.command_ids().repeat_interleave(2)
        reset = torch.full((4,), self._first, dtype=torch.bool, device=self.device)
        step = self.step_views(
            mine,
            feature_tensor,
            self.previous,
            self.cache,
            command=command,
            reset=reset,
            generator=self._generator,
            temperature=temperature,
            button_mask=self._button_mask(world),
        )
        enemies = self.roster.perspectives()[0, self.seats, 2:]
        chosen = torch.as_tensor(
            2 * np.arange(2) + (targets[0, self.seats] == enemies[:, 1]), device=self.device
        )
        labels = ControllerLabels(step.labels.buttons[chosen], step.labels.main_stick[chosen])
        action = self.trunk.codec.decode(labels).as_packed_tensor()
        start = torch.as_tensor(shares[0, self.seats], device=self.device)
        valid = torch.as_tensor(world.actor_valid[0, self.seats], device=self.device) & ~start
        pressed = torch.zeros(2, dtype=torch.bool, device=self.device)
        if self.taunt is not None and self.taunt.enables(command.tolist()):
            facts = state_features(mine)
            logits = self.taunt(torch.cat([step.hidden, facts.to(step.hidden.dtype)], -1), command)
            pressed = draw(logits, self.taunt.active(command), self._taunt_generator).to(self.device)[chosen]
            action[pressed] = 0
            action[pressed, :4] = 0.5
            action[pressed, 12] = 1
        action[~valid] = 0
        action[~valid, :4] = 0.5
        neutral = ~valid | pressed
        labels = ControllerLabels(
            torch.where(neutral, 0, labels.buttons), torch.where(neutral, 0, labels.main_stick)
        )
        self.previous = ControllerLabels(
            labels.buttons.repeat_interleave(2), labels.main_stick.repeat_interleave(2)
        )
        self._first = False
        return Action(
            action,
            labels,
            {name: value[chosen] for name, value in step.logits.items()},
            (int(targets[0, self.seats[0]]), int(targets[0, self.seats[1]])),
            self.current,
            start,
        )

    def example_input(self) -> tuple[WorldFrame, dict[str, Any]]:
        """Synthetic tensor contract example; no live game connection is included."""
        world = WorldFrame.blank(self.roster)
        world.position[0, :, 0] = [-30, -10, 10, 30]
        dtypes = {"bool": torch.bool, "int": torch.int64, "float": torch.float32}

        def build(spec: Mapping[str, Any], path: str = "") -> dict[str, Any]:
            return {
                key: build(kind, f"{path}.{key}")
                if isinstance(kind, Mapping)
                else torch.zeros((8, 15) if path == ".items" else (8,), dtype=dtypes[kind])
                for key, kind in spec.items()
            }

        tree = build(FRAME_SPEC)
        enemies = self.roster.perspectives()[0, :, 2:]
        tree["stage"].fill_(25)
        for seat in range(4):
            for view in range(2):
                row = 2 * seat + view
                for name, fighter in (("p0", seat), ("p1", int(enemies[seat, view]))):
                    tree[name]["character"][row] = 1
                    tree[name]["x"][row] = float(world.position[0, fighter, 0])
                    tree[name]["on_ground"][row] = True
                    tree[name]["jumps_left"][row] = 2
                    tree[name]["action"][row] = 14
                    tree[name]["shield_strength"][row] = 60
        return world, tree
