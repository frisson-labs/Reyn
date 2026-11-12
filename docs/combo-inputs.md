# Wombo Combo inputs

`Combo.predict(frames, features)` predicts both attackers together. Both rows share weights and retain independent caches and previous-controller labels. Row 0 is attacker 0; row 1 is attacker 1. Within each row, `p0` is that attacker and `p1` is the combo target. **Ordinary two-player observations alone are insufficient:** the policy also reads partner and scoring-state features.

## Base observation tensors

All leaves include a leading dimension of 2. Integer leaves use `torch.int64`, boolean leaves `torch.bool` and continuous leaves `torch.float32`. The canonical nested structure is the same as the language model's, with two rows:

| Path | Fields | Shape |
|---|---|---|
| `p0`, `p1` | `percent:int`, `facing:bool`, `x:float`, `y:float`, `action:int`, `invulnerable:bool`, `character:int`, `jumps_left:int`, `shield_strength:float`, `on_ground:bool` | `[2]` each |
| `p0.controller`, `p1.controller` | `main_stick.x/y:float`, `c_stick.x/y:float`, `shoulder:float`; `buttons.A/B/X/Y/Z/L/R/D_UP:bool` | `[2]` each |
| `p0.nana`, `p1.nana` | `exists:bool` and all 10 player fields above | `[2]` each |
| `stage` | Stage identifier | `[2]` |
| `randall` | `x:float`, `y:float` | `[2]` each |
| `fod_platforms` | `left:float`, `right:float` | `[2]` each |
| `items` | `exists:bool`, `type:int`, `state:int`, `x:float`, `y:float` | `[2,15]` each |

Base observations use unscaled model-schema values: positions in game units, percent as integer percent, and controller axes/shoulder in [0,1]. The demonstrated scope is Fox: internal character identifier 1, Final Destination: stage identifier 25. Both rows include the same target, from their own attacker's perspective. The companion-exists leaf is false for this model's Fox setup; keep all required leaves present.

## Partner/scoring features

`features` is finite `float32[2,186]`. Pass it **already scaled**: divide each raw continuous value by the group's divisor, then clip to its bounds. Bits are 0/1; signs are −1/+1. Categorical columns contain exact integer-valued floats, without normalization. Vectors use the order in Meaning below. Per-move vectors use the exact ordered move list after the table. NaN is invalid. The runtime preserves the checkpoint's categorical embedding lookup and clamping; it does not reconstruct these features from the base observation.

| Columns (zero-based) | Feature | Kind | Divisor | Bounds | Meaning |
|---|---|---|---:|---|---|
| 0 | `target.hitstun` | value | 60 | [0,3] | frames of hitstun left |
| 1 | `target.hitlag` | value | 20 | [0,3] | frames of hitlag left |
| 2–6 | `target.velocity` | value | 5 | [-3,3] | own air x, own ground x, own y, knockback x, knockback y (units a frame) |
| 7 | `target.percent` | value | 100 | [0,10] | true percent (the base model's leaf stops at 300) |
| 8 | `target.action_frame` | value | 60 | [-1,5] | frames into his action state |
| 9 | `target.stunned` | bit | 1 | [0,1] | stunned: the engine's hitstun flag |
| 10 | `target.held` | bit | 1 | [0,1] | held: in a grab or a throw |
| 11 | `target.locked` | bit | 1 | [0,1] | locked: he cannot act on the next frame |
| 12–13 | `target.offset` | value | 50 | [-5,5] | (target - me) x, y |
| 14–15 | `position.me` | value | 100 | [-3,3] | x, y |
| 16–17 | `position.target` | value | 100 | [-3,3] | x, y |
| 18–19 | `position.partner` | value | 100 | [-3,3] | x, y |
| 20–22 | `position.edge` | value | 50 | [-5,5] | ledge - \|x\| of me, target, partner; negative: beyond it |
| 23 | `position.target_off_stage` | bit | 1 | [0,1] | the target is beyond the ledge |
| 24–25 | `partner.offset_me` | value | 50 | [-5,5] | (partner - me) x, y |
| 26–27 | `partner.offset_target` | value | 50 | [-5,5] | (target - partner) x, y |
| 28–32 | `partner.velocity` | value | 5 | [-3,3] | own air x, own ground x, own y, knockback x, knockback y (units a frame) |
| 33 | `partner.facing` | sign | 1 | [-1,1] | +1 he looks right, -1 left |
| 34 | `partner.on_ground` | bit | 1 | [0,1] | |
| 35 | `partner.jumps` | value | 2 | [0,3] | air jumps left |
| 36 | `partner.hitlag` | value | 20 | [0,3] | frames of hitlag left |
| 37 | `partner.action_frame` | value | 60 | [-1,5] | frames into his action state |
| 38 | `partner.percent` | value | 100 | [0,10] | percent |
| 39 | `partner.action` | category | 1 | [0,398] | his action state id (clamped like the base encoder's) |
| 40 | `combo.open` | bit | 1 | [0,1] | a true combo is open |
| 41 | `combo.counted` | value | 10 | [0,10] | counted moves |
| 42 | `combo.moves` | value | 10 | [0,10] | linked moves, counted or not |
| 43 | `combo.hits` | value | 20 | [0,10] | hits |
| 44 | `combo.since_hit` | value | 60 | [0,5] | frames since the last hit |
| 45 | `combo.since_counted` | value | 60 | [0,5] | frames since the last counted move (credit ends at 1) |
| 46–47 | `combo.hit_last` | bit | 1 | [0,1] | the last hit was mine, the partner's |
| 48–49 | `combo.counted_last` | bit | 1 | [0,1] | the last counted move was mine, the partner's |
| 50–74 | `memory.mine` | bit | 1 | [0,1] | the move is among my last distinct moves; one column per move of the ordered moves list |
| 75–99 | `uses.mine` | value | 1 | [0,1] | how often it can still count for me / the cap; one column per move of the ordered moves list |
| 100–124 | `memory.partner` | bit | 1 | [0,1] | the move is among the partner's last distinct moves; one column per move of the ordered moves list |
| 125–149 | `uses.partner` | value | 1 | [0,1] | how often it can still count for him / the cap; one column per move of the ordered moves list |
| 150–174 | `pairs.shown` | bit | 1 | [0,1] | landed now, the move would follow the move it already followed in this combo; one column per move of the ordered moves list |
| 175 | `last_move.mine` | category | 1 | [0,26] | 0 none, 1 + index in the ordered moves list, last: another move |
| 176 | `last_move.partner` | category | 1 | [0,26] | 0 none, 1 + index in the ordered moves list, last: another move |
| 177 | `hold.active` | bit | 1 | [0,1] | the target is in a grab (the grab cap's window) |
| 178–179 | `hold.grabber` | bit | 1 | [0,1] | I hold him, the partner holds him |
| 180 | `hold.pummel_spent` | bit | 1 | [0,1] | this hold's one counted pummel is used |
| 181 | `hold.assist_spent` | bit | 1 | [0,1] | this hold's one counted hit by the grabber's partner is used |
| 182 | `mode.locked` | bit | 1 | [0,1] | the target's percent is pinned |
| 183 | `mode.team_attack` | bit | 1 | [0,1] | the attackers can hit each other |
| 184 | `mode.escaper` | bit | 1 | [0,1] | the target is the scripted escaper, not the dummy |
| 185 | `seat` | bit | 1 | [0,1] | 0 for seat 0, 1 for seat 1 |

### Ordered move vectors

| Index | Move identifier | Meaning |
|---:|---:|---|
| 0 | 0 | unknown move |
| 1 | 2 | jab |
| 2 | 6 | dash attack |
| 3 | 7 | forward tilt |
| 4 | 8 | up tilt |
| 5 | 9 | down tilt |
| 6 | 10 | forward smash |
| 7 | 11 | up smash |
| 8 | 12 | down smash |
| 9 | 13 | neutral air |
| 10 | 14 | forward air |
| 11 | 15 | back air |
| 12 | 16 | up air |
| 13 | 17 | down air |
| 14 | 18 | laser |
| 15 | 19 | illusion |
| 16 | 20 | fire fox |
| 17 | 21 | shine |
| 18 | 50 | get-up attack |
| 19 | 52 | pummel |
| 20 | 53 | forward throw |
| 21 | 54 | back throw |
| 22 | 55 | up throw |
| 23 | 56 | down throw |
| 24 | 61 | ledge attack |

Last-move categories are 0 for none, 1 + the move index above, or 26 for a move outside the list. Partner action (column 39) holds an action-state identifier in [0,398]. Last-move categories are columns 175 and 176 in [0,26]. Seat column 185 is 0 for the first row and 1 for the second. All scoring-state facts must describe the current shared combo consistently from each row's perspective. They include whether a combo is open, last hitter/counted attacker, distinct-move memory, remaining scoring uses, pair history and grab-credit state. These facts are inputs to inference, not outputs computed by the model.

The machine-readable [feature_schema.json](https://huggingface.co/frisson-labs/Wombo-Combo/blob/main/feature_schema.json) contains all 186 ordered column names, the groups and the exact base tree. `model.feature_schema` exposes that same hash-verified object.

## Predictions and history

`prediction.action` has shape `[2,13]`, in this order: `main_x,main_y,c_x,c_y,shoulder,A,B,X,Y,Z,L,R,D_UP`. Axes/shoulder are normalized; button entries are0/1. `prediction.labels.as_tensor()` is `int64[2,2]` in `buttons,main_stick` order. `prediction.logits` holds float32 `buttons:[2,728]` and `main_stick:[2,85]`, with autoregressive conditioning on the sampled prefix.

`predict` advances both characters' histories by one input frame and keeps the previous categorical labels internally. `reset=torch.tensor([True,False])` clears the first row's cache and previous labels before that prediction. The second row's history and the sampling stream stay intact. Ending a combo doesn't reset either character's history. `model.reset(seed=0)` clears both histories and restarts the shared generator. Repeating the same tensors and seed on the same backend reproduces the sampling. Results may differ across devices and PyTorch versions. The rolling cache keeps 256 frames per row.

`example_input()` returns synthetic tensors to illustrate structure. Replace every value with consistent model inputs for meaningful predictions. This policy uses the pinned Wombo Combo base/adapter and has no language-routing API.
