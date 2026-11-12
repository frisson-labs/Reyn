"""Doubles residual inference layers, with the trained categorical feature tables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

Categorical = Mapping[str, tuple[int, int]]


class FeatureEncoder(nn.Module):
    """A feature row as network input: the float columns times ``gain``, then one table row per id column.

    An id column holds an exact small integer in ``[0, cardinality)`` (out-of-range values are clamped, so a
    captured CUDA graph can never index outside a table).  A column named in ``tables`` reads that frozen
    table (a buffer: saved with the module, never trained); every other id column gets a learned embedding of
    width ``embed``."""

    def __init__(
        self,
        features: int,
        *,
        categorical: Categorical | None = None,
        embed: int = 16,
        gain: float = 1.0,
        tables: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if features < 0 or embed < 1:
            raise ValueError("features must be >= 0 and embed >= 1")
        self.features = int(features)
        self.embed = int(embed)
        self.gain = float(gain)
        self.categorical: dict[str, tuple[int, int]] = {}
        taken: set[int] = set()
        for name, (column, cardinality) in (categorical or {}).items():
            column, cardinality = int(column), int(cardinality)
            if not 0 <= column < self.features or column in taken:
                raise ValueError(
                    f"id column {name!r}: column {column} is outside the {self.features} features "
                    "or used twice"
                )
            if cardinality < 1:
                raise ValueError(f"id column {name!r}: cardinality must be >= 1, got {cardinality}")
            taken.add(column)
            self.categorical[str(name)] = (column, cardinality)
        unknown = sorted(set(tables or {}) - set(self.categorical))
        if unknown:
            raise ValueError(f"a frozen table needs an id column of its name; none for {unknown}")
        floats = [column for column in range(self.features) if column not in taken]
        self.float_columns: torch.Tensor
        self.register_buffer("float_columns", torch.tensor(floats, dtype=torch.long), persistent=False)
        self.learned = nn.ModuleList()
        self._sources: list[
            tuple[int, int, str, int]
        ] = []  # (column, cardinality, "table" | "learned", index)
        self._learned_index: dict[str, int] = {}
        self.table_shapes: dict[str, tuple[int, int]] = {}
        out = len(floats)
        for position, (name, (column, cardinality)) in enumerate(self.categorical.items()):
            table = (tables or {}).get(name)
            if table is None:
                self._learned_index[name] = len(self.learned)
                self._sources.append((column, cardinality, "learned", len(self.learned)))
                self.learned.append(nn.Embedding(cardinality, self.embed))
                out += self.embed
                continue
            if table.dim() != 2 or table.shape[0] < cardinality:
                raise ValueError(
                    f"the frozen table of {name!r} must be [rows >= {cardinality}, width], "
                    f"got {tuple(table.shape)}"
                )
            self.register_buffer(f"table_{position}", table.detach().clone().float())
            self.table_shapes[name] = (int(table.shape[0]), int(table.shape[1]))
            self._sources.append((column, cardinality, "table", position))
            out += int(table.shape[1])
        self.out_dim = out

    def embedding(self, name: str) -> nn.Embedding:
        """The learned table of id column ``name`` (``KeyError`` for a column that reads a frozen table)."""

        return cast(nn.Embedding, self.learned[self._learned_index[name]])

    def spec(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "categorical": {name: [column, size] for name, (column, size) in self.categorical.items()},
            "embed": self.embed,
            "gain": self.gain,
            "tables": {name: list(shape) for name, shape in self.table_shapes.items()},
        }

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``[..., features] -> [..., out_dim]`` (sync-free: index ops and gathers only)."""

        if features.shape[-1] != self.features:
            raise ValueError(f"this encoder reads {self.features} features a row, got {features.shape[-1]}")
        if not self._sources:
            return features if self.gain == 1.0 else features * self.gain
        parts = [features.index_select(-1, self.float_columns) * self.gain]
        for column, cardinality, kind, index in self._sources:
            ids = features[..., column].round().long().clamp(0, cardinality - 1)
            if kind == "learned":
                parts.append(self.learned[index](ids).to(features.dtype))
            else:
                table = cast(torch.Tensor, getattr(self, f"table_{index}"))
                parts.append(F.embedding(ids, table).to(features.dtype))
        return torch.cat(parts, dim=-1)


class PartnerAdapter(nn.Module):
    """``delta [..., hidden_dim]`` for a hidden state and its feature row; zero until ``up`` trains."""

    def __init__(
        self,
        hidden_dim: int,
        features: int,
        *,
        width: int = 256,
        categorical: Categorical | None = None,
        embed: int = 16,
        gain: float = 1.0,
        tables: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or width < 1:
            raise ValueError("hidden_dim and width must be positive")
        self.hidden_dim, self.width = int(hidden_dim), int(width)
        self.encoder = FeatureEncoder(
            features, categorical=categorical, embed=embed, gain=gain, tables=tables
        )
        self.down_hidden = nn.Linear(self.hidden_dim, self.width)
        self.down_features = nn.Linear(self.encoder.out_dim, self.width, bias=False)
        self.mid = nn.Linear(self.width, self.width)
        self.up = nn.Linear(self.width, self.hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    @property
    def features(self) -> int:
        return self.encoder.features

    def spec(self) -> dict[str, Any]:
        return {"hidden_dim": self.hidden_dim, "width": self.width, **self.encoder.spec()}

    def forward(self, hidden: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"this adapter reads hidden states of {self.hidden_dim}, got {hidden.shape[-1]}")
        normed = F.layer_norm(hidden, (self.hidden_dim,))
        # the ids are read from the features as they come (float32): a half-precision copy could not hold them
        inputs = self.encoder(features).to(hidden.dtype)
        z = F.gelu(self.down_hidden(normed) + self.down_features(inputs))
        delta: torch.Tensor = self.up(F.gelu(self.mid(z)))
        return delta
