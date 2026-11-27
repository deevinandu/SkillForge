# Cylinder pyramid (2026-11-25)

Stand six cylinders up into a 3-2-1 pyramid, the same layout as the block pyramid. Sim only
(Franka Panda in MuJoCo). Nothing here ran on hardware.

## Task (`tasks/franka_stack_cylinder_pyramid.json`)
- Six cylinders, 2 in (50.8 mm) diameter and 3.3 in (83.8 mm) tall, the real ones' size. The
  mass (100 g) is an assumption, not a measurement.
- The bottom row is 8 mm apart. Each upper cylinder stands centred over the gap below it, on a
  21 mm wide strip of each support's top.
- Success: six tight spots (8 mm in x and y, 6 mm in z). Randomization as in the block pyramid.

## Skill: `PlaceInOrder` 0.2.0
Upright cylinders are now placed with TubeIntoRack, the tube-insertion placer. It does a
speed-ramped carry, aligns above the spot, then descends slowly with corrections and releases
narrow. Blocks keep SpotPlace. TubeIntoRack now aligns above the higher of the rack top and the
spot's seated height, so it can place on top of other objects (unchanged for the tube tasks).

## What it took (seeds 0-99)

| Change | Pyramid |
| --- | --- |
| Block placer (SpotPlace), as in the block pyramid | 22 |
| Carry height from the object's height (the held cylinder swept through its neighbours) | 29 |
| TubeIntoRack placer (ramped carry, aligned slow descent, narrow release) | 75 |
| Speed ramp 0.12 -> 0.08 m/s^2 | 96 |
| Speed ramp 0.048 m/s^2 | **99** |

What each fix addressed:
- **Carry height.** PickPlace carries 10 cm above the object's rest height, but a cylinder held
  near its top hangs 6.4 cm below the fingers, so it swept through the 8.4 cm neighbours.
- **Placer.** Fast, jerky carries made cylinders slide in the grip, land off target and shove
  their neighbours (up to 30 mm, traced).
- **Speed ramp.** The remaining slips came under low friction. Grip force is ample (about 14 N
  a finger for a 1.3 N cylinder), but a flat pad touches a cylinder along one line, and the
  cylinder swings about it under acceleration until it pops out. A gentler ramp keeps the swing
  small.

## Results

| Task | Seeds 0-99 | Stress x2 (0-49) |
| --- | --- | --- |
| Cylinder pyramid | **99/100** (1 slip) | 42/50 (timeout 6, slip 1, wrong placement 1) |
| Block pyramid (unchanged) | 99/100 | 46/50 |

The slow ramp makes a cylinder pyramid take longer; under stress x2 most failures are timeouts.
Regression baselines for both pyramids are committed.
