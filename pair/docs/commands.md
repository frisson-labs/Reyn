# Language commands

Reyn-Pair accepts the **102 language instructions** below. Use the exact wording, including capitalization and punctuation. Each instruction is connected to conditioning stored in this model. Commands describe what to try; responses depend on the situation.

```python
model.say("Attempt a grab.", recipient="fox1")
model.say("Go to the top platform.", recipient="fox2")
# Both Foxes now have different active instructions.
model.clear(recipient="fox1")  # Fox 2 keeps its platform instruction.
```

Choose `"both"`, `"fox1"`, or `"fox2"`. Each Fox holds its own instruction until you replace or clear it. Updating one Fox leaves the other unchanged. `clear()` cancels both instructions; `reset()` also clears prediction history. Fox1 is the lower-port teammate and Fox2 is the higher-port teammate.

## Play style

| Behavior | Exact instruction |
|---|---|
| Aggressive play | `Aim to give me very little breathing room when your attacks can reach.` |
| Defensive play | `Play defensive.` |
| Safe play | `Play safe.` |
| Stock protection | `Don't die.` |
| Grab avoidance | `Don't get grabbed.` |
| Slower pace | `After each attack, leave a half-second pause measured only while normal control is available.` |
| Faster pace | `Increase your action pace from the way you were playing a moment ago.` |
| Varied timing | `Change up how long you wait before attacking again.` |
| Air play | `Favor airborne play whenever you have control outside a recovery.` |
| Ground play | `Favor grounded play whenever you have control outside a recovery.` |

## Movement

| Behavior | Exact instruction |
|---|---|
| Jump | `Jump from the ground.` |
| Double jump | `Jump again in the air.` |
| Double jump held back | `Keep your air jumps unused until you are recovering.` |
| Air dodge | `Air-dodge.` |
| Dash dance | `Dash dance.` |
| Fastfall | `Fastfall.` |
| L-cancel | `L-cancel your aerials.` |
| Wavedash | `Complete a wavedash from a ground jump.` |

## Position and spacing

| Behavior | Exact instruction |
|---|---|
| Left side: move there | `Go to the left side.` |
| Left side: stay there | `When you have control, favor playing on the left side.` |
| Right side: move there | `Go to the right side.` |
| Right side: stay there | `When you have control, favor playing on the right side.` |
| Centre of the stage | `Keep to the center of the stage.` |
| Close spacing | `Favor a small horizontal gap from your opponent while you have control.` |
| Far spacing | `Favor wide horizontal spacing from your opponent while you have control.` |
| Approach | `During the time you can steer horizontally, make movement toward your opponent your usual choice.` |
| Holding position | `Favor holding your horizontal position whenever you can steer horizontally.` |
| Above the opponent | `During the time you have control, make a position above your opponent your usual choice.` |
| Below the opponent | `During the time you have control, make a position below your opponent your usual choice.` |

## Platforms

These instructions were tested on Battlefield. Use a stage with the required platforms.

| Behavior | Exact instruction |
|---|---|
| Left platform | `Go to the left platform.` |
| Right platform | `Go to the right platform.` |
| Top platform | `Go to the top platform.` |
| On the platforms | `Stay on the platforms.` |
| Main floor | `Stay on the main floor.` |
| Follow onto the platform | `Follow him onto the platform.` |
| Off his platform | `Don't share his platform.` |
| Under his platform | `Stay under his platform.` |
| Platform above him | `Get on the platform above him.` |

## Ledges and recovery

| Behavior | Exact instruction |
|---|---|
| Near the ledge | `Wait by the ledge.` |
| Left ledge | `Wait by the left ledge.` |
| Right ledge | `Wait by the right ledge.` |
| Ledge hold | `Take the ledge.` |
| Ledgedash | `Ledgedash.` |
| Edgeguard from the stage | `Attempt your edgeguards from onstage positions.` |
| Low recovery | `Aim for the ledge with a recovery from below.` |

## Attacks

| Behavior | Exact instruction |
|---|---|
| Jab | `Use your jab.` |
| Up tilt | `Use up tilt.` |
| Down tilt | `Use down tilt.` |
| Tilts | `Use tilts.` |
| Forward smash | `Use forward smash.` |
| Down smash | `Use down smash.` |
| Up-smash | `Up-smash him.` |
| Smash attack | `Go for a smash attack.` |
| Up-air | `Choose up air for this attack.` |
| Back-air | `Back-air him.` |
| Down-air | `Drill him.` |
| Neutral-air | `Nair him.` |
| Dash attack | `Use dash attack.` |
| Aerials as the main attack | `Choose aerial attacks for most of your offensive commitments.` |
| Normals as the main attack | `Choose normal attacks for most of your offensive commitments.` |
| Specials as the main attack | `Choose special attacks for most of your offensive commitments.` |

## Grabs, throws and combos

| Behavior | Exact instruction |
|---|---|
| Grab | `Attempt a grab.` |
| Throw | `Grab and throw your opponent.` |
| Down-throw | `Down-throw him.` |
| Throw off the stage | `Get your opponent offstage with a throw.` |
| Grab into up-air | `Connect a grab, throw them, and continue the combo into an up air.` |
| Combo | `Combo your opponent with two or more separate attacks.` |
| Three-hit combo | `Do a three-hit combo.` |
| Combo extension | `Continue looking for combo follow-ups until I break free.` |
| Up-air as the combo ender | `Finish your combo with an up-air.` |
| Up-smash as the kill move | `Kill with up-smash.` |
| Kill now | `Cause your opponent to lose a stock.` |
| Damage without the kill | `Aim for the highest damage you can put on me while keeping my stock intact.` |

## Defense and punishes

| Behavior | Exact instruction |
|---|---|
| Shield | `Put your shield up.` |
| No shielding on a worn shield | `Avoid shielding any time your shield health is below fifty percent.` |
| Spot dodge | `Choose a spot dodge.` |
| Roll backward | `Choose the backward roll.` |
| Roll forward | `Choose the forward roll.` |
| Rolls and dodges | `Roll and dodge.` |
| Whiff punish | `Punish his whiffs.` |
| Landing punish | `Punish his landing.` |
| Tech-chase | `Tech-chase him.` |
| Out-of-shield punish | `After blocking my attack, land a grab directly out of shield.` |

## Fox techniques

| Behavior | Exact instruction |
|---|---|
| Laser shot | `Begin a laser attack.` |
| Lasers as the main attack | `Choose laser activations for most of your offensive commitments.` |
| Shine | `Activate your reflector.` |
| Projectile reflection | `Connect a reflection on a projectile.` |
| Shine-spike | `Shine-spike him.` |
| Waveshine | `As Fox, combo a grounded shine into up smash by wavedashing out of the shine, making no other attack in between.` |
| Fire Fox | `Hit him with Fire Fox.` |
| Fox Illusion | `Fox Illusion.` |
| Laser camping | `Keep your distance and shoot lasers.` |
| Multishine | `Multishine.` |

## Restrictions

| Behavior | Exact instruction |
|---|---|
| No jumping | `Do not jump at all.` |
| No attacking | `Don't start any attacks.` |
| No special moves | `Avoid all specials, even when recovering.` |

## Extra moves

| Behavior | Exact instruction |
|---|---|
| Taunt | `Taunt him.` |
| Platform laps | `Do platform laps.` |
| Ledge stall | `Stall on the ledge.` |
| Wavedash back and forth | `Wavedash back and forth.` |
| Tea-bag from a distance | `Tea-bag him from a distance.` |

## Special moves

| Behavior | Exact instruction |
|---|---|
| Up special | `Begin the Fire Fox move.` |

## Availability

Powershield is excluded because this checkpoint has no cached instruction for it. Free-form paraphrases are not routed by this inference package. The list describes accepted inputs, not a guarantee that every instruction will be carried out consistently.
