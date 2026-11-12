# Reyn command reference

Use language steering to shape play style, movement, position, attacks, defense and special moves. Start with the example instructions below and send one instruction at a time.

Use moves that fit your character and destinations that exist on the stage. Fox techniques such as Fire Fox and waveshining are character-specific; platform requests need a stage with the requested platform. The general-command results on the model card cover Fox, Falco, Marth and Mario. The special-move tests cover all 26 characters.

## Browse commands

- [Play style](#play-style)
- [Movement](#movement)
- [Position and spacing](#position-and-spacing)
- [Platforms](#platforms)
- [Ledges and recovery](#ledges-and-recovery)
- [Attacks](#attacks)
- [Grabs, throws and combos](#grabs-throws-and-combos)
- [Defense and punishes](#defense-and-punishes)
- [Fox and Falco techniques](#fox-and-falco-techniques)
- [Sword and fireball techniques](#sword-and-fireball-techniques)
- [Restrictions](#restrictions)
- [Extra moves](#extra-moves)
- [Character-specific special moves](#character-specific-special-moves)
- [Change or clear a command](#change-or-clear-a-command)

## Play style

| Command | Example instruction |
| :--- | :--- |
| Aggressive play | `Be more aggressive.` |
| Defensive play | `Play defensive.` |
| Safe play | `Play safe.` |
| Stock protection | `Don't die.` |
| Grab avoidance | `Don't get grabbed.` |
| Slower pace | `Play slower.` |
| Faster pace | `Play faster.` |
| Varied timing | `Change up how long you wait before attacking again.` |
| Air play | `Favor airborne play whenever you have control outside a recovery.` |
| Ground play | `Favor grounded play whenever you have control outside a recovery.` |

## Movement

| Command | Example instruction |
| :--- | :--- |
| Jump | `Jump from the ground.` |
| Double jump | `Jump again in the air.` |
| Double jump held back | `Keep your air jumps unused until you are recovering.` |
| Air dodge | `Air-dodge.` |
| Dash dance | `Dash dance.` |
| Fastfall | `Fastfall.` |
| L-cancel | `L-cancel your aerials.` |
| Wavedash | `Complete a wavedash from a ground jump.` |

## Position and spacing

| Command | Example instruction |
| :--- | :--- |
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

| Command | Example instruction |
| :--- | :--- |
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

| Command | Example instruction |
| :--- | :--- |
| Near the ledge | `Wait by the ledge.` |
| Left ledge | `Wait by the left ledge.` |
| Right ledge | `Wait by the right ledge.` |
| Ledge hold | `Take the ledge.` |
| Ledgedash | `Ledgedash.` |
| Edgeguard from the stage | `Attempt your edgeguards from onstage positions.` |
| Low recovery | `Aim for the ledge with a recovery from below.` |

## Attacks

| Command | Example instruction |
| :--- | :--- |
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

| Command | Example instruction |
| :--- | :--- |
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

| Command | Example instruction |
| :--- | :--- |
| Shield | `Put your shield up.` |
| Powershield | `Block an incoming attack with a powershield.` |
| No shielding on a worn shield | `Avoid shielding any time your shield health is below fifty percent.` |
| Spot dodge | `Choose a spot dodge.` |
| Roll backward | `Choose the backward roll.` |
| Roll forward | `Choose the forward roll.` |
| Rolls and dodges | `Roll and dodge.` |
| Whiff punish | `Punish his whiffs.` |
| Landing punish | `Punish his landing.` |
| Tech-chase | `Tech-chase him.` |
| Out-of-shield punish | `After blocking my attack, land a grab directly out of shield.` |

## Fox and Falco techniques

| Command | Example instruction |
| :--- | :--- |
| Laser shot | `Begin a laser attack.` |
| Lasers as the main attack | `Choose laser activations for most of your offensive commitments.` |
| Shine | `Activate your reflector.` |
| Projectile reflection | `Connect a reflection on a projectile.` |
| Shine-spike | `Shine-spike him.` |
| Waveshine | `Waveshine him, please.` |
| Fire Fox | `Hit him with Fire Fox.` |
| Fox Illusion | `Fox Illusion.` |
| Laser camping | `Keep your distance and shoot lasers.` |
| Multishine | `Multishine.` |

## Sword and fireball techniques

| Command | Example instruction |
| :--- | :--- |
| Shield Breaker | `Hit him with Shield Breaker.` |
| Dancing Blade | `Hit him with Dancing Blade.` |
| Dolphin Slash | `Hit him with Dolphin Slash.` |
| Tipper | `Hit him with the tip of the sword.` |
| Fireballs | `Hit him with fireballs.` |
| Cape | `Hit him with the cape.` |
| Jump punch | `Hit him with the jump punch.` |
| Tornado | `Hit him with the tornado.` |

## Restrictions

| Command | Example instruction |
| :--- | :--- |
| No jumping | `Do not jump at all.` |
| No attacking | `Don't start any attacks.` |
| No special moves | `Avoid all specials, even when recovering.` |

## Extra moves

| Command | Example instruction |
| :--- | :--- |
| Taunt | `Taunt him.` |
| Platform laps | `Do platform laps.` |
| Ledge stall | `Stall on the ledge.` |
| Wavedash back and forth | `Wavedash back and forth.` |
| Tea-bag from a distance | `Tea-bag him from a distance.` |

## Character-specific special moves

Use these instructions to steer all four special moves of each character. Where two characters share a move name, keep the character cue shown in the example: use `mario fireball` or `luigi fireball`, for example. Selecting a character does not rewrite the phrase for the router.

The [special-move CSV](../commands.csv) contains these 104 moves with a second phrase and a flag for required character cues.

### Bowser

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `fire breath` |
| Side special | `koopa klaw` |
| Up special | `fortress` |
| Down special | `bowser bomb` |

### Donkey Kong

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `giant punch` |
| Side special | `headbutt` |
| Up special | `spinning kong` |
| Down special | `hand slap` |

### Dr. Mario

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `megavitamin` |
| Side special | `super sheet` |
| Up special | `doc jump punch` |
| Down special | `dr. tornado` |

### Falco

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `laser` |
| Side special | `phantasm` |
| Up special | `fire bird` |
| Down special | `shine` |

### Captain Falcon

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `falcon punch` |
| Side special | `raptor boost` |
| Up special | `falcon dive` |
| Down special | `falcon kick` |

### Fox

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `laser` |
| Side special | `illusion` |
| Up special | `fire fox` |
| Down special | `shine` |

### Mr. Game & Watch

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `chef` |
| Side special | `judge` |
| Up special | `game watch fire` |
| Down special | `oil panic` |

### Ganondorf

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `warlock punch` |
| Side special | `gerudo dragon` |
| Up special | `dark dive` |
| Down special | `wizard's foot` |

### Ice Climbers

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `ice shot` |
| Side special | `squall hammer` |
| Up special | `belay` |
| Down special | `blizzard` |

### Jigglypuff

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `rollout` |
| Side special | `pound` |
| Up special | `sing` |
| Down special | `rest` |

### Kirby

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `swallow` |
| Side special | `hammer` |
| Up special | `final cutter` |
| Down special | `stone` |

### Link

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `bow` |
| Side special | `boomerang` |
| Up special | `spin attack` |
| Down special | `bomb` |

### Luigi

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `luigi fireball` |
| Side special | `green missile` |
| Up special | `luigi jump punch` |
| Down special | `luigi cyclone` |

### Mario

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `mario fireball` |
| Side special | `cape` |
| Up special | `mario jump punch` |
| Down special | `tornado` |

### Marth

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `shield breaker` |
| Side special | `dancing blade` |
| Up special | `dolphin slash` |
| Down special | `counter` |

### Mewtwo

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `shadow ball` |
| Side special | `confusion` |
| Up special | `teleport` |
| Down special | `disable` |

### Ness

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `pk flash` |
| Side special | `pk fire` |
| Up special | `pk thunder` |
| Down special | `psi magnet` |

### Peach

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `toad` |
| Side special | `peach bomber` |
| Up special | `parasol` |
| Down special | `turnip` |

### Pichu

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `thunder jolt` |
| Side special | `skull bash` |
| Up special | `agility` |
| Down special | `thunder` |

### Pikachu

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `thunder jolt` |
| Side special | `skull bash` |
| Up special | `quick attack` |
| Down special | `thunder` |

### Roy

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `flare blade` |
| Side special | `double-edge` |
| Up special | `blazer` |
| Down special | `counter` |

### Samus

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `charge shot` |
| Side special | `missile` |
| Up special | `screw attack` |
| Down special | `bomb` |

### Sheik

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `needle storm` |
| Side special | `chain` |
| Up special | `vanish` |
| Down special | `transform` |

### Yoshi

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `egg lay` |
| Side special | `egg roll` |
| Up special | `egg throw` |
| Down special | `yoshi bomb` |

### Young Link

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `fire bow` |
| Side special | `boomerang` |
| Up special | `spin attack` |
| Down special | `bomb` |

### Zelda

| Move | Example instruction |
| :--- | :--- |
| Neutral special | `nayru's love` |
| Side special | `din's fire` |
| Up special | `farore's wind` |
| Down special | `transform` |

## Change or clear a command

An accepted command stays active until you replace or clear it. A reply with `outcome == "abstain"` leaves the previous command active.

```python
reply = model.say("Be more aggressive.")
reply = model.say("Go to the top platform.")
model.clear()
```

Use `model.clear()` to cancel a command directly, or `model.reset()` to clear both the command and prediction history. You can also ask to cancel with a phrase such as `Cancel that last command.`; check the reply to see whether the router accepted it.

The listed instructions are examples of supported meanings. The router reads your text as written. The model card describes measured execution results; recognizing a request and carrying it out are separate steps.
