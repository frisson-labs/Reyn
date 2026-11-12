"""Deterministic target identities with distance hysteresis; no episode resets on switches."""

from __future__ import annotations

import numpy as np

from reyn_doubles.schema import Bools, Ints, Roster, WorldFrame


def baseline_targets(world: WorldFrame, targets: Ints, learner_team: Ints, *, mode: str) -> Ints:
    """Pair the unadapted opponents by physical port, switching only at final-stock loss."""
    if mode not in ("nearest", "paired_survivor"):
        raise ValueError("unknown opponent targeting mode")
    if targets.shape != (world.roster.envs, 4) or learner_team.shape != (world.roster.envs,):
        raise ValueError("target and team shapes must match the roster")
    if mode == "nearest":
        return targets.copy()
    roster = world.roster
    enemies = roster.perspectives()[:, :, 2:]
    # Each team's lower-port player sees the other team's lower-port player.
    same_team = roster.teams[:, :, None] == roster.teams[:, None, :]
    lower_port = roster.ports[:, None, :] < roster.ports[:, :, None]
    rank = np.sum(same_team & lower_port, axis=-1)
    assigned = np.take_along_axis(enemies, rank[..., None], axis=-1)[..., 0]
    other = np.take_along_axis(enemies, (1 - rank)[..., None], axis=-1)[..., 0]
    rows = np.arange(roster.envs)[:, None]
    fallback = (world.stocks[rows, assigned] == 0) & (world.stocks[rows, other] > 0)
    paired = np.where(fallback, other, assigned)
    return np.where(roster.teams != learner_team[:, None], paired, targets)


class TargetSelector:
    def __init__(self, roster: Roster, *, margin: float = 20.0, hold_frames: int = 12) -> None:
        if not np.isfinite(margin) or margin < 0 or hold_frames < 1:
            raise ValueError("target hysteresis needs nonnegative margin and positive frame count")
        self.roster = roster
        self.enemies = roster.perspectives()[:, :, 2:]
        self.targets = self.enemies[:, :, 0].copy()
        self.count = np.zeros_like(self.targets)
        self.last_frame = np.zeros(roster.envs, dtype=np.int64)
        self.started = np.zeros(roster.envs, dtype=np.bool_)
        self.margin = margin
        self.hold_frames = hold_frames
        self._grid = np.ix_(np.arange(roster.envs), np.arange(4))
        """24 Oct 2026: ``[E,1]`` and ``[1,4]`` index grids, so a per-fighter pick is one fancy index (the
        live frame's ``take_along_axis`` calls cost about 10 us each)."""

    def update(self, world: WorldFrame, reset: Bools | None = None) -> tuple[Ints, Bools]:
        world.check_roster(self.roster)
        reset = np.zeros(self.roster.envs, dtype=np.bool_) if reset is None else reset
        if reset.shape != (self.roster.envs,) or reset.dtype != np.bool_:
            raise ValueError("reset must be bool [E]")
        initial = reset | ~self.started
        if np.any(~initial & (world.frame != self.last_frame + 1)):
            raise ValueError("target selector needs consecutive frames or an explicit reset")
        env = np.arange(self.roster.envs)[:, None, None]
        positions = world.position[env, self.enemies]
        distances = np.linalg.norm(positions - world.position[:, :, None, :], axis=-1)
        available = world.present[env, self.enemies] & (world.stocks[env, self.enemies] > 0)
        distances = np.where(available, distances, np.inf)
        nearest_pos = np.argmin(distances, axis=-1)
        env_index, seat_index = self._grid
        nearest = self.enemies[env_index, seat_index, nearest_pos]
        current_pos = (self.enemies[..., 1] == self.targets).astype(np.int64)
        current_distance = distances[env_index, seat_index, current_pos]
        nearest_distance = distances[env_index, seat_index, nearest_pos]
        better = (nearest != self.targets) & (nearest_distance + self.margin < current_distance)
        self.count = np.where(better, self.count + 1, 0)
        self.count[initial] = 0
        immediate = ~np.isfinite(current_distance) & np.isfinite(nearest_distance)
        switch = immediate | (better & (self.count >= self.hold_frames)) | initial[:, None]
        old = self.targets.copy()
        self.targets = np.where(switch & np.isfinite(nearest_distance), nearest, self.targets)
        self.count[switch] = 0
        self.last_frame = world.frame.copy()
        self.started[:] = True
        return self.targets.copy(), (old != self.targets) | initial[:, None]
