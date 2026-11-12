"""Role-ordered extra information; categorical IDs stay separate from continuous inputs.

Projectiles are not represented in v1 yet. This explicit version is suitable for
contract probes, not a claim that the planned full actor observation is complete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reyn_doubles.schema import Bools, Floats, Ints, WorldFrame

SCHEMA = "melee_rl.doubles.features.v1"
ENTITY_COLUMNS = (
    "dx",
    "dy",
    "air_x",
    "ground_x",
    "self_y",
    "attack_x",
    "attack_y",
    "percent",
    "stocks",
    "action_frame",
    "shield",
    "jumps",
    "facing",
    "grounded",
    "present",
    "invulnerable",
    "respawning",
    "hitlag",
    "hitstun",
    "selected",
)
GLOBAL_COLUMNS = ("own_stocks", "enemy_stocks", "time_left", "ledge_x", "self_x", "self_y")
FEATURE_SIZE = 4 * len(ENTITY_COLUMNS) + len(GLOBAL_COLUMNS)


@dataclass(frozen=True)
class FeatureBatch:
    continuous: Floats
    actions: Ints
    characters: Ints
    actor_valid: Bools


def build_features(world: WorldFrame, selected: Ints) -> FeatureBatch:
    roster = world.roster
    e = roster.envs
    roles = roster.perspectives()
    if selected.shape != (e, 4) or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("selected enemy indices must be integer [E,4]")
    if not np.all((roles[:, :, 2:] == selected[:, :, None]).any(axis=-1)):
        raise ValueError("selected slot must be an enemy, never an ally")
    env = np.arange(e)[:, None, None]
    present = world.present[env, roles]
    # Sanitize absent entity state before computing offsets, including absent self.
    positions = np.where(world.present[..., None], world.position, 0)
    relative = (positions[env, roles] - positions[:, :, None, :]) / 100.0
    velocity = world.velocity[env, roles] / 5.0
    scalar = np.stack(
        [
            world.percent / 100,
            world.stocks / 4,
            world.action_frame / 60,
            world.shield / 60,
            world.jumps / 2,
            world.facing,
            world.grounded,
            world.present,
            world.invulnerable,
            world.respawning,
            world.hitlag / 20,
            np.where(world.in_hitstun, world.hitstun, 0) / 60,
        ],
        axis=-1,
    )[env, roles]
    entity = np.concatenate([relative, velocity, scalar, (roles == selected[..., None])[..., None]], axis=-1)
    # Keep stock/presence/selection metadata for an absent fighter, but no stale motion.
    metadata = {"stocks", "present", "selected"}
    for column, name in enumerate(ENTITY_COLUMNS):
        if name not in metadata:
            entity[..., column] *= present
    totals = roster.team_sum(world.stocks)
    rows = np.arange(e)[:, None]
    team = roster.teams
    global_features = np.stack(
        [
            totals[rows, team] / 8,
            totals[rows, 1 - team] / 8,
            np.broadcast_to(world.remaining_frames[:, None] / 28800, (e, 4)),
            np.broadcast_to(world.ledge_x[:, None] / 100, (e, 4)),
            positions[..., 0] / 100,
            positions[..., 1] / 100,
        ],
        axis=-1,
    )
    continuous = np.concatenate([entity.reshape(e, 4, -1), global_features], axis=-1)
    return FeatureBatch(
        np.clip(continuous, -10, 10).astype(np.float32),
        np.where(present, np.clip(world.action[env, roles], 0, 398), 0).astype(np.int64),
        roster.characters[env, roles],
        world.actor_valid,
    )
