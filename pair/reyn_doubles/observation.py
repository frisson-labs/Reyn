"""Canonical input tensor schema and batch-one validation for model inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

import torch

from reyn_doubles.codec import BUTTON_ORDER, ControllerLabels

LeafKind = Literal["bool", "int", "float"]
PLAYER_FIELDS: Final[tuple[tuple[str, LeafKind], ...]] = (
    ("percent", "int"),
    ("facing", "bool"),
    ("x", "float"),
    ("y", "float"),
    ("action", "int"),
    ("invulnerable", "bool"),
    ("character", "int"),
    ("jumps_left", "int"),
    ("shield_strength", "float"),
    ("on_ground", "bool"),
)
NANA_FIELDS: Final[tuple[tuple[str, LeafKind], ...]] = (("exists", "bool"), *PLAYER_FIELDS)
ITEM_FIELDS: Final[tuple[tuple[str, LeafKind], ...]] = (
    ("exists", "bool"),
    ("type", "int"),
    ("state", "int"),
    ("x", "float"),
    ("y", "float"),
)
_STICK_SPEC: Final[dict[str, Any]] = {"x": "float", "y": "float"}
_CONTROLLER_SPEC: Final[dict[str, Any]] = {
    "main_stick": _STICK_SPEC,
    "c_stick": _STICK_SPEC,
    "shoulder": "float",
    "buttons": dict.fromkeys(BUTTON_ORDER, "bool"),
}
_PLAYER_SPEC: Final[dict[str, Any]] = {
    **dict(PLAYER_FIELDS),
    "controller": _CONTROLLER_SPEC,
    "nana": dict(NANA_FIELDS),
}
FRAME_SPEC: Final[dict[str, Any]] = {
    "p0": _PLAYER_SPEC,
    "p1": _PLAYER_SPEC,
    "stage": "int",
    "randall": _STICK_SPEC,
    "fod_platforms": {"left": "float", "right": "float"},
    "items": dict(ITEM_FIELDS),
}


def prepare_input(
    value: Mapping[str, Any], device: torch.device | str = "cpu", *, batch_size: int = 1
) -> dict[str, Any]:
    """Convert the observation tree to tensors with an explicit batch size (one by default)."""
    tree = value
    dtypes = {"bool": torch.bool, "int": torch.int64, "float": torch.float32}

    def convert(data: Mapping[str, Any], spec: Mapping[str, Any], path: str = "") -> dict[str, Any]:
        if not isinstance(data, Mapping) or set(data) != set(spec):
            raise ValueError(f"Observation {path or '<root>'} fields must be {sorted(spec)}")
        result: dict[str, Any] = {}
        for key, kind in spec.items():
            name = f"{path}.{key}" if path else key
            if isinstance(kind, Mapping):
                result[key] = convert(data[key], kind, name)
                continue
            tensor = torch.as_tensor(data[key], dtype=dtypes[kind], device=device)
            expected = (batch_size, 15) if name.startswith("items.") else (batch_size,)
            if batch_size == 1 and tensor.shape == expected[1:]:
                tensor = tensor.unsqueeze(0)
            if tuple(tensor.shape) != expected:
                raise ValueError(f"Observation {name} must have shape {expected}, got {tuple(tensor.shape)}")
            result[key] = tensor
        return result

    return convert(tree, FRAME_SPEC)


def neutral_labels(
    batch_shape: Sequence[int] = (), *, device: torch.device | str | None = None
) -> ControllerLabels:
    shape = torch.Size(batch_shape)
    return ControllerLabels(
        buttons=torch.zeros(shape, dtype=torch.long, device=device),
        main_stick=torch.zeros(shape, dtype=torch.long, device=device),
    )
