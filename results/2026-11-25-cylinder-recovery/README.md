# Cylinder pyramid with knock-overs (2026-11-25)

Build the six-cylinder 3-2-1 pyramid while someone knocks one or two cylinders over, and rebuild
it. Sim only (Franka Panda in MuJoCo). Nothing here ran on hardware.

## Task (`tasks/franka_stack_cylinder_pyramid_knocked.json`)
- The cylinder pyramid (2 in x 3.3 in cylinders, 100 g assumed) with `knockdown`: between steps
  400 and 1400, 1-2 cylinders (never the one in the gripper) are knocked over, with everything
  resting on them. Each is laid on its side within 8 cm of where it stood. 5000 steps (250 s).
- A border around the work area: at the foot of each 6 cm wall, a 12 deg ramp, 10 cm long.
  - Toe line: back x = 0.30 m; sides y = +-0.51 m; front on tangents to a 0.70 m circle about the
    base, at 0 and +-30 deg. That is the arm's reach for a grasp low on the table.
  - A rolling cylinder runs up the ramp or hits the wall, rolls back, and rests at the ramp's foot,
    where open fingers still fit beside it.
  - What did not work: a bare wall left cylinders pinned against it, with no room for a finger. A
    1.2 cm lip stopped slow cylinders, but 1 m/s ones jumped it.
  - A real setup should have a ramped border like this one, not a bare wall.

## Skill: `StackWithRecovery` 0.1.0 + `UprightCylinder` 0.1.0 (`skills/recover_stack.py`)
StackWithRecovery re-plans every time its current action ends:
1. Stand up a lying cylinder (UprightCylinder), if the open hand fits beside it.
   - If no lying cylinder has room, move a standing neighbour (nothing on top) out of the way,
     to a free spot clear of the lying cylinder's hand patch.
2. Otherwise fill the first spot that can be filled now.
   - The object must stand upright; the spots below it must be filled.
   - Anything standing in the spot is cleared first.
   - An object out of attempts (6) is skipped; the rest is still built.
3. If a placer's supports get knocked out from under it mid-carry, it is redirected to a free spot.

A spot counts as filled on the mean of the last 10 observed positions, with 2 mm of slack. A
cylinder nothing stands on yet must sit 1.5 mm inside its spot, or it is re-placed: once something
stands on it, it can no longer be fixed.

UprightCylinder stands a lying cylinder up with gravity:
- It grips from above, across the body, 1 cm from one end: an end where the open hand fits.
- It lifts to 10 cm, then loosens the grip under feedback until the cylinder swings down to hang
  vertical. It loosens a little each step, tightens if the cylinder slides, and grips firmly once
  it is vertical.
- Hanging vertical, the bottom is ~2.6 cm above the table, so if it slides in the fingers, the
  table catches it standing.
- It sets the cylinder down where it was picked, opens and rises.
- A retry after a drop grips the other end and loosens more gently.

## Results (true sim state)

| Seeds | Rebuilt |
| --- | --- |
| 0-49 (all knocked) | **49/50** (seed 47: grip slipped; 50/50 before main's domino sim changes moved the knock-over later in the step) |
| 100-149 (held out, all knocked) | **49/50** (seed 107: out of time) |
| UprightCylinder alone: 4 x 6 positions x 4 lying directions | 96/96 |

### How it got there (seeds 0-49)

| Change | Rebuilt |
| --- | --- |
| First version (upright by a 24 cm swing and a carry to a staging spot; no border) | 19 |
| Walls at the table's edge (none fall off now; cylinders pinned against walls) | 21 |
| Ramps at the walls' feet; cylinders roll with rolling friction | 20 |
| The lift timed out (its ramp crawled under load); 700-step timeout | 38 |
| Pick the end the hand fits at; move neighbours out of the way; widen the free-spot search | 40 |
| Lift and carry on an advancing reference | 43 |
| Sides of the border from +-0.40 to +-0.51 m | 46 |
| Never "place" a lying cylinder; keep building when one is out of attempts; low swing, no carry | 48 |
| Averaged positions, 2 mm slack; a cylinder nothing stands on must be 1.5 mm inside its spot | **50** |

## Sim changes (T1)
- Knock-over treats fixed objects (the border) as obstacles: it used to crash on them.
- After a knock-over the env refreshes its cached robot contacts. Without that, a knock-over while
  the hand touched something crashed the next observation (stale contact index); main had this
  after merging knockdown with the domino disturbances.
- Cylinders use contact dimension 6, so the rolling friction they already specified (0.0001 m)
  applies. At dimension 4 nothing slowed a rolling cylinder: once nudged, it rolled on until it
  hit something or left the table. The regression baselines (tubes, rack, cylinder to bin,
  cylinder pyramid) still pass.

## Findings
- **Turning the wrist first, then gripping and rotating, is impossible with this hand.** To pinch a
  lying cylinder's end with the hand level, the fingertip centre must be at the cylinder's axis,
  2.5 cm up. The hand body reaches 3.2 cm below that point, into the table.
- **Gravity swing: a fixed grip force works only in a narrow band.** That band (a command of
  0.46-0.47) moves with friction and mass. Loosening under feedback works across the
  randomization.
- **The hand blocks more than the fingers.** The Franka hand body is +-10.4 cm wide along the
  opening direction, and its underside is only 3.7 cm above the fingertip centre. At a low grasp
  it is below the tops of standing 8.4 cm cylinders, so a lying cylinder next to the pyramid often
  has no room to be gripped.
- **Gripping 6 mm above the axis to raise the hand broke the swing** (1 of 24 cases at x 0.64).
- **The slide in the grip.** At friction 0.55x and mass 1.5x, the cylinder slides in the grip during
  the swing until it hangs by its very end. Carried like that, it drops at any acceleration:
  hence the low swing and no carry.
- **Rolling it away by pressing on top moves it only ~1 cm per stroke.** The hand moves twice as far
  as the cylinder's centre and runs off its top line.

## Not done / untested
- Nothing on hardware. The 100 g mass, the friction ranges and the border are assumptions.
- The knock-over lays cylinders down at rest. Real knocked cylinders fall and roll; with the border
  and rolling friction that is plausible in sim, but this task does not exercise it.
- No regression baseline yet for the knocked task (an episode is up to 5000 steps).

Check scripts (not in git): `runs/checks/recover_ev.py` (evaluation),
`runs/checks/show_recovery.py` (video).
