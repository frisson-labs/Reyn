"""CPU world contract, independent of libmelee, the simulator and Ali's policy.

Arrays are match-major. The four source-player slots NEVER change when a fighter
disappears or players cross. Physical ports are 1..4; team IDs are normalized 0/1.
Backends must translate their IDs, presence and hitstun flags explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

Ints = npt.NDArray[np.int64]
Floats = npt.NDArray[np.float32]
Bools = npt.NDArray[np.bool_]
SCHEMA = "melee_rl.doubles.world.v1"


@dataclass(frozen=True)
class Roster:
    ports: Ints
    teams: Ints
    characters: Ints

    def __post_init__(self) -> None:
        shape = self.ports.shape
        if len(shape) != 2 or shape[0] < 1 or shape[1] != 4:
            raise ValueError("roster arrays must be [E, 4] with E positive")
        for name in ("ports", "teams", "characters"):
            values = getattr(self, name)
            if values.shape != shape or not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"roster {name} must be integer {shape}")
        if not np.all(np.sort(self.ports, axis=1) == [1, 2, 3, 4]):
            raise ValueError("roster ports must be a permutation of 1,2,3,4")
        if not np.all(np.sort(self.teams, axis=1) == [0, 0, 1, 1]):
            raise ValueError("roster must have two players on each team, normalized IDs 0/1")
        if np.any((self.characters < 0) | (self.characters >= 32)):
            raise ValueError("roster characters must use internal character IDs")
        # Metadata is owned here, not by a simulator's mutable observation buffer.
        for name in ("ports", "teams", "characters"):
            value = np.array(getattr(self, name), dtype=np.int64, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @classmethod
    def fox(cls, envs: int, teams: tuple[int, int, int, int] = (0, 0, 1, 1)) -> Roster:
        return cls(
            np.tile([1, 2, 3, 4], (envs, 1)),
            np.tile(teams, (envs, 1)),
            np.ones((envs, 4), dtype=np.int64),
        )

    @property
    def envs(self) -> int:
        return int(self.ports.shape[0])

    def matches(self, other: Roster) -> bool:
        return self is other or self._content() == other._content()

    def _content(self) -> tuple[tuple[int, ...], bytes, bytes, bytes]:
        """The roster's numbers as bytes, worked out once (the arrays are read-only int64; 24 Oct 2026: the
        live frame compared rosters about six times)."""

        cached = self.__dict__.get("_content_key")
        if cached is None:
            cached = (self.ports.shape, self.ports.tobytes(), self.teams.tobytes(), self.characters.tobytes())
            object.__setattr__(self, "_content_key", cached)
        return cached

    def perspectives(self) -> Ints:
        """[E, actor, role] source-slot indices: self, ally, enemy A, enemy B.

        Enemies are sorted by physical port, never by proximity or presence.  The roster is immutable, so the
        table is worked out once (23 Oct 2026, play path (a): it was recomputed about nine times a live frame)
        and each caller gets its own copy.
        """
        cached = self.__dict__.get("_perspectives")
        if cached is None:
            cached = self._perspectives_of(self)
            object.__setattr__(self, "_perspectives", cached)
        return cached.copy()

    @staticmethod
    def _perspectives_of(roster: Roster) -> Ints:
        self = roster
        order = np.argsort(self.ports, axis=1)
        result = np.empty((self.envs, 4, 4), dtype=np.int64)
        for seat in range(4):
            same = self.teams == self.teams[:, seat, None]
            same[:, seat] = False
            result[:, seat, 0] = seat
            result[:, seat, 1] = np.argmax(same, axis=1)
            enemy = self.teams != self.teams[:, seat, None]
            sorted_enemy = np.take_along_axis(enemy, order, axis=1)
            result[:, seat, 2:] = order[sorted_enemy].reshape(self.envs, 2)
        return result

    def team_sum(self, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if values.shape != self.teams.shape:
            raise ValueError("team values must be [E, 4]")
        return np.stack([np.where(self.teams == t, values, 0).sum(axis=1) for t in (0, 1)], axis=1)


_BLANK_TEMPLATES: dict[type, dict[int, Any]] = {}
"""24 Oct 2026: ``WorldFrame.blank``'s validated templates, per class and match count."""


@dataclass(frozen=True)
class WorldFrame:
    roster: Roster
    frame: Ints
    remaining_frames: Ints
    stage: Ints
    ledge_x: Floats
    position: Floats
    velocity: Floats
    percent: Floats
    stocks: Ints
    action: Ints
    action_frame: Floats
    shield: Floats
    jumps: Ints
    facing: Bools
    grounded: Bools
    present: Bools
    invulnerable: Bools
    respawning: Bools
    hitlag: Floats
    hitstun: Floats
    in_hitstun: Bools

    def __post_init__(self) -> None:
        e = self.roster.envs
        integers = {"frame", "remaining_frames", "stage", "stocks", "action", "jumps"}
        booleans = {"facing", "grounded", "present", "invulnerable", "respawning", "in_hitstun"}
        for item in fields(self):
            name = item.name
            if name == "roster":
                continue
            value = getattr(self, name)
            shape: tuple[int, ...] = (
                (e,) if name in {"frame", "remaining_frames", "stage", "ledge_x"} else (e, 4)
            )
            if name == "position":
                shape += (2,)
            elif name == "velocity":
                shape += (5,)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"world {name} must have shape {shape}")
            if value.dtype.kind not in "biu" and not np.isfinite(value).all():  # integers are always finite
                raise ValueError(f"world {name} contains nonfinite values")
            if name in integers and not np.issubdtype(value.dtype, np.integer):
                raise ValueError(f"world {name} must be integer")
            if name in booleans and value.dtype != np.bool_:
                raise ValueError(f"world {name} must be bool")
        if np.any(self.stocks < 0) or np.any(self.remaining_frames < 0):
            raise ValueError("stock and remaining frame counts cannot be negative")

    @classmethod
    def blank(cls, roster: Roster) -> WorldFrame:
        """Synthetic four-stock Fox/FD fixture; backends must supply observed fields.  24 Oct 2026 (the live
        frame builds one each frame): copies of a validated template, the constants not validated again."""

        templates = _BLANK_TEMPLATES.setdefault(cls, {})
        template = templates.get(roster.envs)
        if template is None:
            template = templates[roster.envs] = cls._blank(roster)
        world = object.__new__(cls)
        object.__setattr__(world, "roster", roster)
        for item in fields(cls)[1:]:
            object.__setattr__(world, item.name, getattr(template, item.name).copy())
        return world

    @classmethod
    def _blank(cls, roster: Roster) -> WorldFrame:
        e = roster.envs
        f = lambda value: np.full((e, 4), value, dtype=np.float32)  # noqa: E731
        i = lambda value: np.full((e, 4), value, dtype=np.int64)  # noqa: E731
        b = lambda value: np.full((e, 4), value, dtype=np.bool_)  # noqa: E731
        return cls(
            roster,
            np.zeros(e, dtype=np.int64),
            np.full(e, 8 * 60 * 60, dtype=np.int64),
            np.full(e, 25, dtype=np.int64),
            np.full(e, 85.5656967163086, dtype=np.float32),
            np.zeros((e, 4, 2), dtype=np.float32),
            np.zeros((e, 4, 5), dtype=np.float32),
            f(0),
            i(4),
            i(14),
            f(0),
            f(60),
            i(2),
            b(True),
            b(True),
            b(True),
            b(False),
            b(False),
            f(0),
            f(0),
            b(False),
        )

    @property
    def actor_valid(self) -> Bools:
        return self.present & (self.stocks > 0) & (self.frame[:, None] >= 0)

    def check_roster(self, roster: Roster) -> None:
        if not self.roster.matches(roster):
            raise ValueError("roster changed inside an environment")
