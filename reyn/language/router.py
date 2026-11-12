"""Route English phrases to compatible learned command representations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from torch import nn

from reyn.language.encoder import BertEncoder, EncoderConfig
from reyn.language.spec import ABSTAIN, CLEAR, Spec, spec_from_text
from reyn.language.text import dedup_key
from reyn.language.tokenizer import WordPiece

ROUTER_FORMAT: Final[str] = "melee_rl.router.v1"
NAMES: Final[Mapping[str, str]] = {CLEAR: "Command cleared", ABSTAIN: "Not understood"}


@dataclass(frozen=True)
class Rule:
    threshold: float = 0.5
    margin: float = 0.0
    temperature: float = 1.0
    veto: bool = True
    opposite: float = 1.0

    def __post_init__(self) -> None:
        if self.veto:
            raise ValueError("unsupported polarity-veto router; the released router has veto disabled")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.opposite <= 1.0:
            raise ValueError("opposite must be a probability")


@dataclass(frozen=True)
class Route:
    outcome: str
    row: str | None
    name: str
    score: float
    forced: str
    stage: str
    reason: str


class RouterModel(nn.Module):
    def __init__(self, config: EncoderConfig, outcomes: int) -> None:
        super().__init__()
        self.encoder = BertEncoder(config)
        self.head = nn.Linear(config.hidden, outcomes)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pooled = self.encoder(ids, mask)
        return self.head(pooled)


@dataclass(frozen=True)
class Pick:
    forced: str
    score: float
    margin: float
    opposite: float = 0.0


_FLIPS: dict[str, torch.Tensor] = {}


def flip_matrix(spec: Spec) -> torch.Tensor:
    known = _FLIPS.get(spec.sha256)
    if known is None:
        outcomes = list(spec.outcomes)
        known = torch.zeros(len(outcomes), len(outcomes))
        for key in spec.keys:
            for other in spec.wrong(key):
                known[outcomes.index(key), outcomes.index(other)] = 1.0
        _FLIPS[spec.sha256] = known
    return known


def picks_from_logits(
    logits: torch.Tensor, outcomes: Sequence[str], *, temperature: float = 1.0, spec: Spec
) -> list[Pick]:
    full = torch.softmax(logits.detach().float().cpu() / temperature, dim=-1)
    probs = full.clone()
    probs[:, list(outcomes).index(ABSTAIN)] = -1.0
    top = torch.topk(probs, 2, dim=-1)
    against = (full * flip_matrix(spec)[top.indices[:, 0]]).sum(dim=-1).tolist()
    return [
        Pick(outcomes[indices[0]], float(values[0]), float(values[0] - values[1]), float(other))
        for values, indices, other in zip(top.values.tolist(), top.indices.tolist(), against, strict=True)
    ]


def decide(pick: Pick, rule: Rule) -> tuple[str, str]:
    if pick.score < rule.threshold:
        return (ABSTAIN, "low_score")
    if pick.margin < rule.margin:
        return (ABSTAIN, "margin")
    if pick.opposite > rule.opposite:
        return (ABSTAIN, "opposite")
    return (pick.forced, "")


class Router:
    def __init__(
        self,
        *,
        spec: Spec,
        model: nn.Module,
        tokenizer: WordPiece,
        exact: Mapping[str, str],
        rule: Rule,
        max_length: int = 64,
    ) -> None:
        self.spec = spec
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.exact = dict(exact)
        self.rule = rule
        self.max_length = max_length
        self._outcomes = spec.outcomes
        self._labels = {label.key: label for label in spec.labels}

    def logits(self, texts: Sequence[str], *, batch: int = 64) -> torch.Tensor:
        device = next(self.model.parameters()).device
        parts = []
        for start in range(0, len(texts), batch):
            ids, mask = self.tokenizer.batch(texts[start : start + batch], max_length=self.max_length)
            with torch.inference_mode():
                logits = self.model(
                    torch.tensor(ids, dtype=torch.long, device=device),
                    torch.tensor(mask, dtype=torch.long, device=device),
                )
            parts.append(logits.float().cpu())
        return torch.cat(parts) if parts else torch.zeros(0, len(self._outcomes))

    def picks(self, texts: Sequence[str], *, batch: int = 64) -> list[Pick]:
        if not texts:
            return []
        logits = self.logits(texts, batch=batch)
        return picks_from_logits(logits, self._outcomes, temperature=self.rule.temperature, spec=self.spec)

    def _route(self, outcome: str, pick: Pick, stage: str, reason: str = "") -> Route:
        label = self._labels.get(outcome)
        return Route(
            outcome=outcome,
            row=None if label is None else label.row,
            name=NAMES[outcome] if label is None else label.name,
            score=pick.score,
            forced=pick.forced,
            stage=stage,
            reason=reason,
        )

    def route_many(self, texts: Sequence[str]) -> list[Route]:
        routes: list[Route | None] = [None] * len(texts)
        pending = []
        for index, text in enumerate(texts):
            known = self.exact.get(dedup_key(text))
            if known is not None:
                routes[index] = self._route(known, Pick(known, 1.0, 1.0), "exact")
            else:
                pending.append(index)
        for index, pick in zip(pending, self.picks([texts[index] for index in pending]), strict=True):
            outcome, reason = decide(pick, self.rule)
            routes[index] = self._route(outcome, pick, "model", reason)
        return [route for route in routes if route is not None]

    def route(self, text: str) -> Route:
        return self.route_many([text])[0]

    def missing_rows(self, table_texts: Iterable[str]) -> list[str]:
        held = set(table_texts)
        cards = () if self.spec.characters is None else self.spec.characters.cards
        return [
            *(label.key for label in self.spec.labels if label.row not in held),
            *(card.key for card in cards if card.row not in held),
        ]

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> Router:
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != ROUTER_FORMAT:
            raise ValueError(f"{path}: not a router file ({ROUTER_FORMAT})")
        spec = spec_from_text(payload["spec"], source=str(path))
        model = RouterModel(EncoderConfig(**payload["encoder"]), len(spec.outcomes))
        model.load_state_dict(payload["state"], strict=True)
        return cls(
            spec=spec,
            model=model.to(device),
            tokenizer=WordPiece(payload["vocab"]),
            exact=payload["exact"],
            rule=Rule(**payload["rule"]),
            max_length=int(payload["max_length"]),
        )
