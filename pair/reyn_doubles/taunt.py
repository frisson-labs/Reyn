"""Auxiliary action head and its inference-time sampling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Final

import torch
import torch.nn.functional as F
from torch import nn

LOGIT_LIMIT: Final[float] = 15.0
FACT_GAIN: Final[float] = 3.0


class TauntHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        rows: int,
        *,
        facts: int = 16,
        width: int = 64,
        embed: int = 16,
    ) -> None:
        super().__init__()
        if facts != 16:
            raise ValueError("unsupported auxiliary head; requires 16 state features")
        if min(hidden_dim, rows, width, embed) < 1:
            raise ValueError("auxiliary head dimensions must be positive")
        self.hidden_dim, self.rows, self.facts = (int(hidden_dim), int(rows), int(facts))
        self.row = nn.Embedding(rows, embed)
        self.down_hidden = nn.Linear(hidden_dim, width)
        self.down_row = nn.Linear(embed, width, bias=False)
        self.down_facts = nn.Linear(self.facts, width, bias=False)
        self.out = nn.Linear(width, 1)
        self.enabled: torch.Tensor
        self.register_buffer("enabled", torch.zeros(rows, dtype=torch.bool))
        self._enabled_rows: frozenset[int] = frozenset()

    def _refresh(self) -> None:
        self._enabled_rows = frozenset(int(row) for row in torch.nonzero(self.enabled).flatten().tolist())

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._refresh()
        return result

    def enables(self, commands: Iterable[int]) -> bool:
        live = self._enabled_rows
        return bool(live) and any(int(command) in live for command in commands)

    def active(self, command: torch.Tensor) -> torch.Tensor:
        inside = (command >= 0) & (command < self.rows)
        return inside & self.enabled[command.clamp(min=0, max=self.rows - 1)]

    def forward(self, row: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        if row.shape[-1] < self.hidden_dim + self.facts:
            raise ValueError(
                f"the taunt head reads {self.hidden_dim} hidden and {self.facts} fact columns; "
                f"the row has {int(row.shape[-1])}"
            )
        hidden = row[..., : self.hidden_dim].float()
        index = command.clamp(min=0, max=self.rows - 1)
        inner = self.down_hidden(F.layer_norm(hidden, (self.hidden_dim,))) + self.down_row(self.row(index))
        facts = row[..., self.hidden_dim : self.hidden_dim + self.facts].float()
        inner = inner + self.down_facts(facts * FACT_GAIN)
        logit: torch.Tensor = self.out(F.gelu(inner)).squeeze(-1)
        return logit.clamp(-LOGIT_LIMIT, LOGIT_LIMIT)


def draw(logit: torch.Tensor, active: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    host = logit.detach().float().cpu()
    uniform = torch.rand(host.shape, generator=generator)
    return (uniform < torch.sigmoid(host)) & active.cpu()


def from_payload(payload: Mapping[str, Any] | None) -> TauntHead:
    if not payload:
        raise ValueError("checkpoint is missing its auxiliary action head")
    spec = payload["spec"]
    head = TauntHead(
        int(spec["hidden_dim"]),
        int(spec["rows"]),
        facts=int(spec.get("facts", 0)),
        width=int(spec["width"]),
        embed=int(spec["embed"]),
    )
    head.load_state_dict(payload["state"], strict=True)
    return head
