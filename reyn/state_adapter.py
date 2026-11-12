"""State-dependent residual conditioning for the four-character checkpoint."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

FEATURES: Final[int] = 16
_UNIT: Final[float] = 100.0
_CLIP: Final[float] = 4.0
FEATURE_GAIN: Final[float] = 3.0


def state_features(tree: Mapping[str, Any]) -> torch.Tensor:
    me, them = (tree["p0"], tree["p1"])
    x, y = (me["x"].float() / _UNIT, me["y"].float() / _UNIT)
    his_x, his_y = (them["x"].float() / _UNIT, them["y"].float() / _UNIT)
    dx, dy = (his_x - x, his_y - y)
    facing = me["facing"].float() * 2.0 - 1.0
    his_facing = them["facing"].float() * 2.0 - 1.0
    features = torch.stack(
        (
            x,
            y,
            his_x,
            his_y,
            dx,
            dy,
            dx.abs(),
            x.abs(),
            facing,
            his_facing,
            facing * torch.tanh(dx * 20.0),
            me["percent"].float() / _UNIT,
            them["percent"].float() / _UNIT,
            me["on_ground"].float(),
            them["on_ground"].float(),
            me["jumps_left"].float() / 2.0,
        ),
        dim=-1,
    )
    return features.clamp(-_CLIP, _CLIP)


class StateAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        rows: int,
        *,
        width: int = 256,
        embed: int = 64,
        features: int = 16,
        depth: int = 0,
        stage: int = 0,
    ) -> None:
        super().__init__()
        if depth or stage or features != FEATURES:
            raise ValueError("unsupported state adapter; requires 16 state features and no depth/stage branch")
        if min(hidden_dim, rows, width, embed) < 1:
            raise ValueError("state adapter dimensions must be positive")
        self.hidden_dim, self.rows = hidden_dim, rows
        self.row = nn.Embedding(rows, embed)
        self.down_hidden = nn.Linear(hidden_dim, width)
        self.down_command = nn.Linear(embed + FEATURES, width, bias=False)
        self.mid = nn.Linear(width, width)
        self.up = nn.Linear(width, hidden_dim)

    def forward(self, hidden: torch.Tensor, command: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != FEATURES:
            raise ValueError("state adapter requires 16 state features")
        active = (command >= 0) & (command < self.rows)
        index = command.clamp(min=0, max=self.rows - 1)
        parts = [self.row(index).to(hidden.dtype), features.to(hidden.dtype) * FEATURE_GAIN]
        normed = F.layer_norm(hidden, (self.hidden_dim,))
        inner = self.down_hidden(normed) + self.down_command(torch.cat(parts, dim=-1))
        z = F.gelu(inner)
        delta = self.up(F.gelu(self.mid(z)))
        return torch.where(active.unsqueeze(-1), delta, torch.zeros_like(delta))


def from_payload(payload: dict[str, Any] | None) -> StateAdapter:
    if not payload:
        raise ValueError("checkpoint is missing its state adapter")
    module = StateAdapter(**{key: int(value) for key, value in payload["spec"].items()})
    module.load_state_dict(payload["state"], strict=True)
    return module
