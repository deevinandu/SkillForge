# Six-block pyramid (2026-11-25)

Stack six 4 cm cubes into a 3-2-1 pyramid: three in a row on the table, two on top straddling
the gaps, one on top. Sim only (Franka Panda in MuJoCo). Nothing here ran on hardware.

## Task (`tasks/franka_stack_block_pyramid.json`)
- Six 4 x 4 x 4 cm cubes, 50 g, coloured blue, black, orange (bottom row), white, yellow
  (middle) and red (top). They start in a 2 x 3 grid on the table, with up to 1.5 cm of xy
  jitter and 0.3 rad of yaw.
- The bottom row is 8 mm apart along y. Each upper block sits centred over the gap below it, so
  it rests 16 mm onto each of its two supports.
- Success: `all_of` six `object_in_region` conditions, one tight spot per block (8 mm in x and
  y, 5 mm in z, around the block's centre at its stack height). A block on the table next to
  the pyramid, or one not seated on its supports, does not count.
- Randomization: friction 0.5-1.2x, mass 0.7-1.5x, gain 0.8-1.2x, observation noise 0-2 mm,
  latency 0-2 steps. 2400 steps (120 s).

## Skill (`skills/place_in_order.py`)
- **`PlaceInOrder` 0.1.0:** one placer per condition, in the task's order (bottom row first).
  The fingers close across the row (along x), so they never reach into the neighbouring
  blocks.
- **`SpotPlace`:** a PickPlace that aims the block's centre at the spot's centre, with a 2 mm
  arrival tolerance and the in-hand offset averaged over the last 8 observations.
- PickPlace gained an optional `closing_axis`; it is off by default, so library behaviour is
  unchanged.

## What it took (seeds 0-99)

| Change | Pyramid |
| --- | --- |
| PickPlace per block, aimed at the spot centre, 4 mm gaps | 81 |
| Arrival tolerance 6 mm -> 2 mm | 85 |
| In-hand offset from the last 8 samples instead of 40 | 92 |
| Gaps 4 mm -> 8 mm | **99** |

What each fix addressed:
- **Tolerance.** PickPlace's 6 mm "arrived" tolerance released blocks short of their spot, on
  the side they came from.
- **In-hand offset.** With the fingers closing across the row, only friction holds the block
  along the row. It slides 4-15 mm in the fingers on the carry, and a 40-sample average lagged
  that slide by about 5 mm. The last 8 samples track it: block_1 in seed 8 went from 4.7 mm off
  to 0.1 mm.
- **Gaps.** With 4 mm gaps, a few millimetres of error made a middle block clip its
  neighbour's edge on the way down and push it about 10 mm (traced in seed 8).

## Results

| Seeds | Success |
| --- | --- |
| 0-99, the task's ranges | **99/100** (1 wrong placement) |
| 0-49, stress x2 | 46/50 (collision 2, wrong placement 1, slip 1) |

A full pyramid takes about 640-740 steps (32-37 s of sim time). A regression baseline (99/100)
is committed.

## Not done
- Blocks touching, as in a real stacked-cube shelf: this needs better placement accuracy, or
  pushing a block against its neighbour.
- Blocks still slide in the grip along the row. A smoother carry, or a grasp along the carry
  direction with a re-grasp before placing, would remove the slide itself.
- Recovery when the pyramid is knocked over.
