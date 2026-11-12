"""Encode and decode the model's categorical action representation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import torch
from torch import nn

Tensor = torch.Tensor
BUTTON_ORDER: Final[tuple[str, ...]] = ("A", "B", "X", "Y", "Z", "L", "R", "D_UP")
COMPONENT_ORDER: Final[tuple[str, ...]] = ("buttons", "main_stick")
PACKED_CONTROLLER_ORDER: Final[tuple[str, ...]] = (
    "main_x",
    "main_y",
    "c_x",
    "c_y",
    "shoulder",
    *BUTTON_ORDER,
)
_LIGHT_SHOULDER_THRESHOLD: Final = 0.3
_FULL_SHOULDER_THRESHOLD: Final = 0.9
_SHOULDER_VALUES: Final[tuple[float, ...]] = (0.0, 0.35, 1.0)
_MIN_NONZERO_RADIUS: Final = 23
_MAX_RADIUS: Final = 80
_RAW_AXIS_SPACING: Final = 160
_RAW_AXIS_RADIUS: Final = 80


@dataclass(frozen=True)
class StickState:
    x: Tensor
    y: Tensor


@dataclass(frozen=True)
class ButtonState:
    A: Tensor
    B: Tensor
    X: Tensor
    Y: Tensor
    Z: Tensor
    L: Tensor
    R: Tensor
    D_UP: Tensor

    def values(self) -> tuple[Tensor, ...]:
        return tuple(getattr(self, name) for name in BUTTON_ORDER)


@dataclass(frozen=True)
class ControllerState:
    main_stick: StickState
    c_stick: StickState
    shoulder: Tensor
    buttons: ButtonState

    def as_packed_tensor(self, *, dtype: torch.dtype = torch.float32) -> Tensor:
        values = (
            self.main_stick.x,
            self.main_stick.y,
            self.c_stick.x,
            self.c_stick.y,
            self.shoulder,
            *self.buttons.values(),
        )
        return torch.stack(tuple(value.to(dtype=dtype) for value in values), dim=-1)


@dataclass(frozen=True)
class ControllerLabels:
    buttons: Tensor
    main_stick: Tensor

    def values(self) -> tuple[Tensor, Tensor]:
        return (self.buttons, self.main_stick)

    def as_tensor(self) -> Tensor:
        return torch.stack(self.values(), dim=-1)


def _coerce_controller(value: Tensor) -> ControllerState:
    if value.ndim < 1 or value.shape[-1] != 13:
        raise ValueError("action tensor must have shape [...,13]")
    return ControllerState(
        main_stick=StickState(value[..., 0].float(), value[..., 1].float()),
        c_stick=StickState(value[..., 2].float(), value[..., 3].float()),
        shoulder=value[..., 4].float(),
        buttons=ButtonState(
            **{name: value[..., 5 + index].bool() for index, name in enumerate(BUTTON_ORDER)}
        ),
    )


class _PolarStickCodec(nn.Module):
    radius_table: Tensor
    count_table: Tensor
    offset_table: Tensor
    label_to_radius: Tensor
    label_to_inner: Tensor
    angle_table: Tensor

    def __init__(self, angle_buckets_by_nonzero_radius: Sequence[int]) -> None:
        super().__init__()
        angle_buckets = tuple(int(value) for value in angle_buckets_by_nonzero_radius)
        if not angle_buckets or any(value <= 0 for value in angle_buckets):
            raise ValueError("each nonzero radius must have a positive angle-bucket count")
        self.angle_buckets_by_nonzero_radius = angle_buckets
        counts = (1, *angle_buckets)
        offsets: list[int] = []
        offset = 0
        for count in counts:
            offsets.append(offset)
            offset += count
        self.num_labels = offset
        radius_values = torch.exp(
            torch.linspace(
                math.log(_MIN_NONZERO_RADIUS), math.log(_MAX_RADIUS), len(angle_buckets), dtype=torch.float64
            )
        )
        radius_table = torch.cat((torch.zeros(1, dtype=torch.float64), radius_values)).float()
        count_table = torch.tensor(counts, dtype=torch.long)
        offset_table = torch.tensor(offsets, dtype=torch.long)
        label_to_radius = torch.repeat_interleave(torch.arange(len(counts), dtype=torch.long), count_table)
        label_to_inner = torch.cat(tuple(torch.arange(count, dtype=torch.long) for count in counts))
        angle_table = torch.cat(
            tuple(
                torch.arange(count, dtype=torch.float64) / count * (2.0 * math.pi) - math.pi
                for count in counts
            )
        ).float()
        self.register_buffer("radius_table", radius_table, persistent=False)
        self.register_buffer("count_table", count_table, persistent=False)
        self.register_buffer("offset_table", offset_table, persistent=False)
        self.register_buffer("label_to_radius", label_to_radius, persistent=False)
        self.register_buffer("label_to_inner", label_to_inner, persistent=False)
        self.register_buffer("angle_table", angle_table, persistent=False)

    @staticmethod
    def _at(buffer: Tensor, reference: Tensor) -> Tensor:
        return buffer if buffer.device == reference.device else buffer.to(reference.device)

    def bucket(self, stick: StickState) -> Tensor:
        x = stick.x.float()
        y = stick.y.float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError("stick coordinates must be finite")
        raw_x = torch.round(x * _RAW_AXIS_SPACING - _RAW_AXIS_RADIUS).long()
        raw_y = torch.round(y * _RAW_AXIS_SPACING - _RAW_AXIS_RADIUS).long()
        radius = torch.sqrt((raw_x.square() + raw_y.square()).float())
        is_origin = radius <= _MIN_NONZERO_RADIUS - 1
        min_log_radius = math.log(_MIN_NONZERO_RADIUS)
        max_log_radius = math.log(_MAX_RADIUS)
        normalized_radius = (torch.log(radius + 0.001) - min_log_radius) / (max_log_radius - min_log_radius)
        nonzero_radius = torch.round(
            normalized_radius * (len(self.angle_buckets_by_nonzero_radius) - 1)
        ).long()
        radius_bucket = torch.where(is_origin, torch.zeros_like(nonzero_radius), nonzero_radius + 1)
        invalid = (radius_bucket < 0) | (radius_bucket >= self.count_table.numel())
        if invalid.any():
            bad_radius = radius[invalid].max().item()
            raise ValueError(
                "stick lies outside custom_v1's legal radial domain; "
                f"largest invalid raw radius was {bad_radius:g}"
            )
        counts = self._at(self.count_table, radius_bucket)[radius_bucket]
        angle = torch.atan2(raw_y.float(), raw_x.float())
        normalized_angle = (angle + math.pi) / (2.0 * math.pi)
        angle_bucket = torch.remainder(torch.round(normalized_angle * counts).long(), counts)
        offsets = self._at(self.offset_table, radius_bucket)
        return offsets[radius_bucket] + angle_bucket

    def decode(self, labels: Tensor) -> StickState:
        labels = _integer_labels(labels, self.num_labels, "stick")
        label_to_radius = self._at(self.label_to_radius, labels)
        radius_table = self._at(self.radius_table, labels)
        angle_table = self._at(self.angle_table, labels)
        radius_bucket = label_to_radius[labels]
        raw_radius = radius_table[radius_bucket]
        angle = angle_table[labels]
        raw_x = raw_radius * torch.cos(angle)
        raw_y = raw_radius * torch.sin(angle)
        return StickState(
            x=((raw_x + _RAW_AXIS_RADIUS) / _RAW_AXIS_SPACING).float(),
            y=((raw_y + _RAW_AXIS_RADIUS) / _RAW_AXIS_SPACING).float(),
        )


def _integer_labels(labels: Any, vocab_size: int, component: str) -> Tensor:
    tensor = torch.as_tensor(labels)
    if tensor.is_floating_point() or tensor.is_complex():
        if tensor.is_floating_point() and torch.equal(tensor, tensor.round()):
            tensor = tensor.long()
        else:
            raise TypeError(f"{component} labels must be integer-valued")
    else:
        tensor = tensor.long()
    if ((tensor < 0) | (tensor >= vocab_size)).any():
        minimum = tensor.min().item() if tensor.numel() else None
        maximum = tensor.max().item() if tensor.numel() else None
        raise ValueError(f"{component} labels must be in [0, {vocab_size}), got range [{minimum}, {maximum}]")
    return tensor


class CustomV1Codec(nn.Module):
    component_order: Final[tuple[str, ...]] = COMPONENT_ORDER
    button_order: Final[tuple[str, ...]] = BUTTON_ORDER

    def __init__(
        self,
        *,
        c_stick_angle_buckets: Sequence[int] = (4, 8),
        main_stick_angle_buckets: Sequence[int] = (4, 16, 64),
    ) -> None:
        super().__init__()
        self.c_stick = _PolarStickCodec(c_stick_angle_buckets)
        self.main_stick = _PolarStickCodec(main_stick_angle_buckets)
        self._button_axis_sizes = (2, 2, 2, 7, self.c_stick.num_labels)
        button_vocab = math.prod(self._button_axis_sizes)
        self._vocab_sizes = MappingProxyType(
            {"buttons": button_vocab, "main_stick": self.main_stick.num_labels}
        )

    @property
    def vocabulary_sizes(self) -> tuple[int, int]:
        return (self._vocab_sizes["buttons"], self._vocab_sizes["main_stick"])

    def encode(self, controller: Tensor) -> ControllerLabels:
        state = _coerce_controller(controller)
        c_stick = self.c_stick.bucket(state.c_stick)
        shoulder_bucket = torch.zeros_like(state.shoulder, dtype=torch.long)
        shoulder_bucket = torch.where(
            state.shoulder > _LIGHT_SHOULDER_THRESHOLD, torch.ones_like(shoulder_bucket), shoulder_bucket
        )
        shoulder_bucket = torch.where(
            state.shoulder > _FULL_SHOULDER_THRESHOLD, torch.full_like(shoulder_bucket, 2), shoulder_bucket
        )
        buttons = state.buttons
        xy = buttons.X | buttons.Y
        lr = buttons.L | buttons.R
        z_a_shoulder = buttons.A.long() * 3 + shoulder_bucket
        z_a_shoulder = torch.where(buttons.Z, torch.full_like(z_a_shoulder, 6), z_a_shoulder)
        label = buttons.B.long()
        for size, component in zip(
            self._button_axis_sizes[1:], (xy.long(), lr.long(), z_a_shoulder, c_stick), strict=True
        ):
            label = label * size + component
        main_stick = self.main_stick.bucket(state.main_stick)
        return ControllerLabels(buttons=label.long(), main_stick=main_stick.long())

    def _coerce_labels(self, labels: ControllerLabels) -> ControllerLabels:
        buttons = _integer_labels(labels.buttons, self._vocab_sizes["buttons"], "buttons")
        main_stick = _integer_labels(labels.main_stick, self._vocab_sizes["main_stick"], "main_stick")
        if buttons.shape != main_stick.shape or buttons.device != main_stick.device:
            raise ValueError("action labels must have matching shapes and devices")
        return ControllerLabels(buttons, main_stick)

    def decode(self, labels: ControllerLabels) -> ControllerState:
        labels = self._coerce_labels(labels)
        button_label = labels.buttons
        components: list[Tensor] = []
        quotient = button_label
        for size in reversed(self._button_axis_sizes):
            quotient, component = (
                torch.div(quotient, size, rounding_mode="floor"),
                torch.remainder(quotient, size),
            )
            components.append(component)
        components.reverse()
        b, xy, lr, z_a_shoulder, c_stick_label = components
        z = z_a_shoulder == 6
        a = torch.where(z, torch.ones_like(z), torch.div(z_a_shoulder, 3, rounding_mode="floor").bool())
        shoulder_bucket = torch.where(z, torch.ones_like(z_a_shoulder), torch.remainder(z_a_shoulder, 3))
        shoulder_values = torch.tensor(_SHOULDER_VALUES, dtype=torch.float32, device=button_label.device)
        return ControllerState(
            main_stick=self.main_stick.decode(labels.main_stick),
            c_stick=self.c_stick.decode(c_stick_label),
            shoulder=shoulder_values[shoulder_bucket],
            buttons=ButtonState(
                A=a.bool(),
                B=b.bool(),
                X=torch.zeros_like(xy, dtype=torch.bool),
                Y=xy.bool(),
                Z=z.bool(),
                L=lr.bool(),
                R=torch.zeros_like(lr, dtype=torch.bool),
                D_UP=torch.zeros_like(b, dtype=torch.bool),
            ),
        )
