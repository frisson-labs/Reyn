"""Load and combine learned command representations."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

FEATURES_FORMAT: Final[str] = "melee_rl.command_features.v1"


@dataclass(frozen=True)
class CommandFeatures:
    texts: tuple[str, ...]
    tokens: torch.Tensor
    mask: torch.Tensor
    encoder: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = len(self.texts)
        if len(set(self.texts)) != count or not all(isinstance(text, str) and text for text in self.texts):
            raise ValueError("command texts must be distinct non-empty strings")
        if self.tokens.ndim != 3:
            raise ValueError("tokens must have shape [N,L,D]")
        width = self.tokens.shape[2]
        if self.tokens.ndim != 3 or self.tokens.shape[0] != count or self.tokens.shape[2] != width:
            raise ValueError(f"tokens must have shape [{count}, L, {width}], got {tuple(self.tokens.shape)}")
        if self.mask.dtype != torch.bool or tuple(self.mask.shape) != tuple(self.tokens.shape[:2]):
            raise ValueError(f"mask must be bool with shape {tuple(self.tokens.shape[:2])}")
        if count and (not bool(self.mask.any(dim=1).all())):
            raise ValueError("every command needs at least one valid token")
        if not bool(torch.isfinite(self.tokens).all()):
            raise ValueError("command features must be finite")

    @property
    def feature_dim(self) -> int:
        return int(self.tokens.shape[2])

    def index(self, text: str) -> int:
        try:
            return self.texts.index(text)
        except ValueError:
            raise KeyError(f"no features for command text {text!r}") from None

    def select(self, texts: Iterable[str]) -> CommandFeatures:
        rows = [self.index(text) for text in texts]
        order = torch.tensor(rows, dtype=torch.long)
        return CommandFeatures(
            texts=tuple(self.texts[row] for row in rows),
            tokens=self.tokens[order],
            mask=self.mask[order],
            encoder=self.encoder,
        )

    def to(self, device: torch.device | str) -> CommandFeatures:
        return dataclasses.replace(self, tokens=self.tokens.to(device), mask=self.mask.to(device))


ENCODER_IDENTITY: Final[tuple[str, ...]] = ("name", "model_id", "revision", "weights", "weights_sha256")


def _padded(features: CommandFeatures, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    extra = width - int(features.tokens.shape[1])
    if extra <= 0:
        return (features.tokens, features.mask)
    rows, _, dim = features.tokens.shape
    tokens = torch.cat([features.tokens, features.tokens.new_zeros(rows, extra, dim)], dim=1)
    mask = torch.cat([features.mask, features.mask.new_zeros(rows, extra)], dim=1)
    return (tokens, mask)


def with_fallback_rows(table: CommandFeatures, base: CommandFeatures, texts: Iterable[str]) -> CommandFeatures:
    missing = [text for text in dict.fromkeys(texts) if text not in table.texts]
    if not missing:
        return table
    if base.feature_dim != table.feature_dim:
        raise ValueError(f"feature dim {base.feature_dim} of the fallback table is not {table.feature_dim}")
    mismatched = [key for key in ENCODER_IDENTITY if table.encoder.get(key) != base.encoder.get(key)]
    if mismatched:
        raise ValueError(f"the fallback table's encoder differs from the adapter's in {mismatched}")
    extra = base.select(missing).to(table.tokens.device)
    width = max(int(table.tokens.shape[1]), int(extra.tokens.shape[1]))
    trained_tokens, trained_mask = _padded(table, width)
    extra_tokens, extra_mask = _padded(extra, width)
    return CommandFeatures(
        texts=(*table.texts, *extra.texts),
        tokens=torch.cat([trained_tokens, extra_tokens.to(trained_tokens.dtype)], dim=0),
        mask=torch.cat([trained_mask, extra_mask], dim=0),
        encoder=dict(table.encoder),
    )


def load_command_features(path: str | Path) -> CommandFeatures:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != FEATURES_FORMAT:
        raise ValueError(f"{path}: not a {FEATURES_FORMAT} file")
    return CommandFeatures(
        texts=tuple(payload["texts"]),
        tokens=payload["tokens"],
        mask=payload["mask"],
        encoder=json.loads(payload["encoder"]),
    )
