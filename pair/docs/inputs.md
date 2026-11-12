# Tensor inputs

`Doubles.predict(world, views)` returns one prediction for each of two Fox teammates. It does not include a live-game connection.

`world` is a `reyn_doubles.schema.WorldFrame`. Every array has one match row. Fighter slots remain physical ports 1, 2, 3, and 4 in that order throughout the sequence, including after elimination. Team IDs are `[0, 0, 1, 1]`, and internal character IDs are all `1` (Fox). Other roster layouts are rejected. `team=0` controls ports 1 and 2; `team=1` controls ports 3 and 4.

| Field | Shape | Meaning |
|---|---|---|
| `frame`, `remaining_frames`, `stage`, `ledge_x` | `[1]` | Frame index, remaining frame count, internal stage ID, stage edge X |
| `position` | `[1, 4, 2]` | World X/Y per physical fighter |
| `velocity` | `[1, 4, 5]` | Self air X, self ground X, self Y, attack X, attack Y |
| `percent`, `stocks`, `action`, `action_frame`, `shield`, `jumps` | `[1, 4]` | Per-fighter state values |
| `facing`, `grounded`, `present`, `invulnerable`, `respawning` | `[1, 4]` | Boolean state flags |
| `hitlag`, `hitstun`, `in_hitstun` | `[1, 4]` | Hitlag frames, hitstun frames, hitstun validity |

Integer and boolean fields are validated by the schema. Supply observed values for every live fighter; the synthetic fixture only illustrates shapes. Consecutive calls must advance `world.frame` by exactly one. Call `reset()` before starting a new sequence.

`views` is a nested tensor dictionary with the exact fields in [`FRAME_SPEC`](../reyn_doubles/observation.py). Scalar leaves have shape `[8]`; item leaves have shape `[8, 15]`. Each physical fighter has two persistent views, one per enemy, ordered by enemy port:

| Rows | Self (`p0`) | Enemy (`p1`) |
|---|---|---|
| 0, 1 | Port 1 | Port 3, Port 4 |
| 2, 3 | Port 2 | Port 3, Port 4 |
| 4, 5 | Port 3 | Port 1, Port 2 |
| 6, 7 | Port 4 | Port 1, Port 2 |

The frame tree contains `p0` and `p1` player values, controller values, optional Nana fields, stage values, Randall position, Fountain platform heights, and 15 item slots. The schema file names and types are authoritative. The runtime checks stage, Fox identity, action state, and player positions against `world` to catch incompatible view bindings.

The model advances both enemy-view histories for each controlled Fox before selecting its predicted action. Enemy selection uses the original nearest-target rule, with a 20-unit margin held for 12 frames and an immediate switch when the selected opponent becomes unavailable. Both views receive the controller label actually selected for that Fox on the previous frame. Switching commands or targets does not reset histories; `reset()` does.

`prediction.action` is `[2, 13]`, ordered Fox 1 then Fox 2. The columns are main-stick X/Y, C-stick X/Y, shoulder, A, B, X, Y, Z, L, R, and D-pad up. Stick values use `[0, 1]`, with `0.5` centered. `prediction.labels` and `prediction.logits` expose the categorical action representation. `prediction.targets` gives the selected physical fighter slots, indexed 0 through 3. `prediction.commands` reports the held phrases.

`prediction.start` is a separate two-boolean stock-share request. It preserves the original one-pulse helper behavior after 60 frames of eligible absence. It is not a learned controller-head output.

The optional `set_friendly_fire_mask(True, radius=30.0)` setting applies in `predict`. If a present, living, non-respawning and non-invulnerable teammate is within that Euclidean X/Y distance, attack inputs are masked before sampling. The returned button logits include the mask, the stick is conditioned on the sampled button token, and both histories receive the executed labels. `set_friendly_fire_mask(False)` restores unfiltered sampling. The setting is off by default and survives `reset()`.

This proximity heuristic does not predict hitboxes or stop ongoing attacks and projectiles. It does not change game rules.

The low-level `step_views` method exposes the model's cached view step for tensor integrations. It accepts an explicit boolean `button_mask` with shape `[4, 728]`; the friendly-fire toggle is handled by `predict`, which has access to the world positions. `step_views` does not maintain previous-controller history for the caller; the higher-level `predict` method does.
