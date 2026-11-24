# Tube into rack (2026-11-23)

First labware task: pick up a test tube standing on the table and stand it in the middle slot of
a three-slot rack. Sim only (Franka Panda in MuJoCo). Nothing here ran on hardware.

## Task (`tasks/franka_tube_into_rack.json`)
- Tube: 16 mm x 100 mm cylinder, 20 g, standing upright at (0.5, -0.15), xy jitter up to 3 cm.
- Rack: six fixed boxes (two long bars, four separators), 5 cm tall, 20 mm square holes at a
  25 mm pitch, so about 2 mm of clearance either side of the tube.
- Success: the tube's centre within 4 mm of the slot centre in x and y, and within 4 mm of the
  seated height (5 cm) in z. Judged on the true sim state.
- Randomization: friction 0.7-1.3x, mass 0.8-1.5x, gain 0.9-1.1x, observation noise 0-2 mm,
  latency 0-2 steps.

The first version of the region was 1 cm tall in z. That let a tube lying across the rack top
over the slot count as a success: its centre sits at 5.8 cm. The region is now 4 mm, and a test
checks that the lying-on-top height is outside it.

## Skill (`skills/tube_rack.py`, `TubeIntoRack` 0.1.0)
A PickPlace subclass:
- **Grasp:** top-down only, 3 cm below the tube's top (as deep as the hand allows).
- **Carry:** rate-limited to 0.24 m/s.
- **Align:** over the slot, with the tube-to-hand offset averaged over the carry, until the
  x/y error has stayed under 1.5 mm for 6 steps.
- **Insert:** at 4 cm/s, correcting x/y on the way down. If the tube slides more than 6 mm up in
  the fingers (a jam on the rack edge), it backs off and realigns, up to 3 times.

## Results (seeds 0-99)

| Skill | Task's own ranges | Stress x2 |
| --- | --- | --- |
| `TubeIntoRack` 0.1.0 | **99/100** (1 tube slipped out on the carry, fell over, and could not be re-grasped) | **78/100** (timeout 18, slip 4; 85 jam retries) |
| `PickPlace` 0.3.0 | 5/100 (IK failure 47, slip 27, wrong placement 17, collision 4) | not run |

Under the task's own ranges, 60 traced seeds had no jams at all. Under stress x2 the observation
noise reaches 4 mm, more than the clearance, so the tube hits the rack edge and the jam retries
run out of time. That is the obvious target for the agent loop.

## Not done
- A fallen tube is not recovered: the skill only grasps an upright tube.
- One slot, one tube. Filling or sorting a rack (several tubes, the `all_of` success
  condition) comes next.
- Real tube and rack dimensions should replace these numbers once the labware is measured.
