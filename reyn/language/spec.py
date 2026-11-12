"""Typed command catalogue and compatibility checks for router weights."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

SPEC_FORMAT: Final[str] = "melee_rl.router_spec.v1"
CLEAR: Final[str] = "clear"
ABSTAIN: Final[str] = "abstain"


@dataclass(frozen=True)
class Label:
    key: str
    name: str
    row: str
    trained_texts: tuple[str, ...]
    negation: str
    opposites: tuple[str, ...]
    equivalents: tuple[str, ...]


@dataclass(frozen=True)
class Card:
    key: str
    name: str
    row: str
    character: str


@dataclass(frozen=True)
class Move:
    label: str
    has: tuple[str, ...]
    says: str
    slot: str = ""
    by_slot: bool = False
    gives: tuple[tuple[str, str], ...] = ()
    names: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Slot:
    key: str
    name: str
    words: tuple[str, ...]


@dataclass(frozen=True)
class Characters:
    default: str
    cast: tuple[tuple[str, str, int], ...]
    cards: tuple[Card, ...]
    moves: tuple[Move, ...]
    slots: tuple[Slot, ...]
    unnamed: tuple[tuple[str, str, str], ...] = ()

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple((key for key, _, _ in self.cast))

    def name(self, character: str) -> str:
        for key, name, _ in self.cast:
            if key == character:
                return name
        raise KeyError(f"no character {character!r}")

    def card(self, key: str) -> Card | None:
        return next((card for card in self.cards if card.key == key), None)

    def move(self, label: str) -> Move | None:
        return next((move for move in self.moves if move.label == label), None)


@dataclass(frozen=True)
class Spec:
    labels: tuple[Label, ...]
    sha256: str
    characters: Characters

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(label.key for label in self.labels)

    @property
    def outcomes(self) -> tuple[str, ...]:
        return (*self.keys, CLEAR, ABSTAIN)

    def label(self, key: str) -> Label:
        for label in self.labels:
            if label.key == key:
                return label
        raise KeyError(f"no label {key!r}")

    def accepted(self, outcome: str) -> frozenset[str]:
        if outcome in (CLEAR, ABSTAIN):
            return frozenset({outcome})
        return frozenset({outcome, *self.label(outcome).equivalents})

    def wrong(self, key: str) -> frozenset[str]:
        own = self.label(key)
        flips = set(own.opposites)
        for other in own.equivalents:
            flips.update(self.label(other).opposites)
        return frozenset(flips) - self.accepted(key)

    def rows(self) -> tuple[str, ...]:
        cards = self.characters.cards
        return (*(label.row for label in self.labels), *(card.row for card in cards))


def _label(raw: Mapping[str, Any]) -> Label:
    return Label(
        key=str(raw["key"]),
        name=str(raw["name"]),
        row=str(raw["row"]),
        trained_texts=tuple(str(text) for text in raw["trained_texts"]),
        negation=str(raw["negation"]),
        opposites=tuple(str(key) for key in raw.get("opposites", ())),
        equivalents=tuple(str(key) for key in raw.get("equivalents", ())),
    )


def _check(labels: tuple[Label, ...]) -> None:
    keys = [label.key for label in labels]
    for key in keys:
        if keys.count(key) > 1:
            raise ValueError(f"label {key!r} is listed twice")
        if key in (CLEAR, ABSTAIN):
            raise ValueError(f"{key!r} is an outcome, not a label")
    known = set(keys)
    by_key = {label.key: label for label in labels}
    rows: dict[str, str] = {}
    for label in labels:
        if label.row not in label.trained_texts:
            raise ValueError(f"{label.key}: its row {label.row!r} is not one of its trained wordings")
        if label.row in rows:
            raise ValueError(f"{label.key} and {rows[label.row]} share the row {label.row!r}")
        rows[label.row] = label.key
        if label.negation not in known | {CLEAR, ABSTAIN} or label.negation == label.key:
            raise ValueError(f"{label.key}: negation {label.negation!r} is not a label, clear or abstain")
        for other in (*label.opposites, *label.equivalents):
            if other not in known or other == label.key:
                raise ValueError(f"{label.key}: unknown reference {other!r}")
        for other in label.equivalents:
            if label.key not in by_key[other].equivalents:
                raise ValueError(f"{label.key}: equivalent {other!r} does not list it back")
        for other in label.opposites:
            if label.key not in by_key[other].opposites:
                raise ValueError(f"{label.key}: opposite {other!r} does not list it back")
        if set(label.opposites) & {label.key, *label.equivalents}:
            raise ValueError(f"{label.key}: an equivalent is also listed as an opposite")
        if label.negation in known and label.negation not in label.opposites:
            raise ValueError(f"{label.key}: its negation {label.negation!r} must be one of its opposites")


def _characters(raw: Mapping[str, Any]) -> Characters:
    return Characters(
        default=str(raw["default"]),
        cast=tuple((str(item["key"]), str(item["name"]), int(item["id"])) for item in raw["cast"]),
        cards=tuple(
            Card(str(item["key"]), str(item["name"]), str(item["row"]), str(item["character"]))
            for item in raw.get("cards", ())
        ),
        moves=tuple(
            Move(
                label=str(item["label"]),
                has=tuple(str(who) for who in item["has"]),
                says=str(item["says"]),
                slot=str(item.get("slot", "")),
                by_slot=bool(item.get("by_slot", False)),
                gives=tuple(((str(who), str(key)) for who, key in dict(item.get("gives", {})).items())),
                names=tuple(((str(words), str(said)) for words, said in item.get("names", ()))),
            )
            for item in raw.get("moves", ())
        ),
        slots=tuple(
            Slot(str(item["key"]), str(item["name"]), tuple(str(words) for words in item["words"]))
            for item in raw.get("slots", ())
        ),
        unnamed=tuple(
            (str(item["character"]), str(item["slot"]), str(item["move"])) for item in raw.get("unnamed", ())
        ),
    )


def _check_characters(labels: tuple[Label, ...], table: Characters) -> None:
    cast = set(table.keys)
    if len(cast) != len(table.cast) or table.default not in cast:
        raise ValueError("characters: the cast's keys must be distinct and hold the default character")
    known = {label.key for label in labels}
    rows = {label.row for label in labels}
    slots = {slot.key for slot in table.slots}
    cards: set[str] = set()
    for card in table.cards:
        if card.key in known | cards | {CLEAR, ABSTAIN} or card.row in rows or card.character not in cast:
            raise ValueError(f"card {card.key!r}: a key and a row of its own, and a character of the cast")
        cards.add(card.key)
        rows.add(card.row)
    seen: set[str] = set()
    for move in table.moves:
        if move.label not in known or move.label in seen:
            raise ValueError(f"move {move.label!r}: one entry a label")
        seen.add(move.label)
        if not move.has or set(move.has) - cast:
            raise ValueError(f"move {move.label!r}: has {move.has} must name characters of the cast")
        if move.slot and move.slot not in slots:
            raise ValueError(f"move {move.label!r}: unknown slot {move.slot!r}")
        if move.by_slot and (not move.slot):
            raise ValueError(f"move {move.label!r}: a label that answers for a slot names the slot")
        for who, key in move.gives:
            if who not in move.has or key not in known | cards:
                raise ValueError(f"move {move.label!r}: gives {who!r} -> {key!r} is not a command of theirs")
    for slot in slots:
        if sum(move.by_slot and move.slot == slot for move in table.moves) != 1:
            raise ValueError(f"slot {slot!r}: exactly one label answers for it")
    for who, slot, _ in table.unnamed:
        if who not in cast or slot not in slots:
            raise ValueError(f"unnamed special {who!r} / {slot!r}: unknown character or slot")


def spec_from_text(text: str, *, source: str = "the specification") -> Spec:
    raw = json.loads(text)
    if raw.get("format") != SPEC_FORMAT or not raw.get("characters"):
        raise ValueError(f"{source}: unsupported router specification")
    labels = tuple(_label(item) for item in raw["labels"])
    _check(labels)
    characters = _characters(raw["characters"])
    _check_characters(labels, characters)
    return Spec(labels=labels, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), characters=characters)
