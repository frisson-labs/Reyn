"""Stage IDs and platform capabilities used only for language resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class Stage:
    name: str
    model_id: int
    platforms: frozenset[str]

    @property
    def has_top(self) -> bool:
        return "top" in self.platforms

    @property
    def has_sides(self) -> bool:
        return {"left", "right"} <= self.platforms

    def offers(self, need: str) -> bool:
        available = {
            "FD": True,
            "any": True,
            "platforms": bool(self.platforms),
            "top": self.has_top,
            "sides": self.has_sides,
            "three": self.has_top and self.has_sides,
        }
        if need not in available:
            raise ValueError(f"unknown stage need {need!r}")
        return available[need]


_SIDES = frozenset({"left", "right"})
_THREE = _SIDES | {"top"}
STAGES: Final[dict[str, Stage]] = {
    "FD": Stage("FD", 25, frozenset()),
    "BF": Stage("BF", 24, _THREE),
    "YS": Stage("YS", 6, _THREE),
    "DL": Stage("DL", 26, _THREE),
    "FoD": Stage("FoD", 8, _THREE),
    "PS": Stage("PS", 18, _SIDES),
}
_BY_ID = {stage.model_id: stage for stage in STAGES.values()}


def by_model_id(stage_id: int) -> Stage:
    return _BY_ID[int(stage_id)]
