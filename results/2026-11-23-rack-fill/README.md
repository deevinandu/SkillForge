# Rack fill: three tubes into three slots (2026-11-23)

Second labware task, and the first that uses the `all_of` success condition. Sim only (Franka
Panda in MuJoCo). Nothing here ran on hardware.

## Task (`tasks/franka_fill_tube_rack.json`)
- Three 16 x 100 mm tubes stand on the table about 12 cm apart: `tube_a`, `tube_b`, `tube_c`,
  each with up to 2 cm of xy jitter.
- The rack is the same as in `franka_tube_into_rack`: 20 mm holes at a 25 mm pitch, about 2 mm
  of clearance a side.
- Success: `all_of` three `object_in_region` conditions, tube_a in the left slot, tube_b in the
  middle, tube_c on the right, all at the same step. Each slot region is 4 mm in x, y and z, so
  a tube lying on the rack does not count.
- 1200 steps (60 s). The other randomization is as in the single-tube task.

## Skills
- **`FillRack` 0.1.0** (`skills/rack_fill.py`): runs one `TubeIntoRack` per condition, in order.
  The fingers close across the rack row and open only 12 mm wider than the tube on release, so
  they do not hit tubes already in the neighbouring slots. It stops at the first tube that fails.
- **`TubeIntoRack` 0.2.0** (`skills/tube_rack.py`), changed for this task:
  - A short checked lift, then a vertical rise high enough to clear standing tubes, then the
    carry.
  - Every carry segment follows a reference moving at a capped speed with a gentle speed ramp
    (0.12 m/s^2).
  - Options for the finger closing axis and a narrow release.

## What it took (seeds 0-99, `FillRack`)

| Change | Rack fill |
| --- | --- |
| First version (tube_into_rack 0.1 behaviour, crosswise grasp), seeds 0-49 | 30/50; 19/50 with a first, measured-EE speed ramp |
| Carry height from the tube's bottom; checked lift + ramped rise | 69 (all failures timeouts) |
| Ramp follows its own reference, not the lagging measured EE | 86 |
| Speed ramp 0.32 -> 0.12 m/s^2 | **99** |

Two findings along the way:
- **The slips came from the lift, not the grip direction.** With the other tubes present, the
  old safe height made the arm lift 5 cm higher at full speed. The same scene with that extra
  lift went from 30/30 to 13/30. The grip direction alone made no difference (40/40 for each
  axis on the single-tube task).
- **The old limiter stepped from the measured EE, which lags its target.** The arm therefore
  moved far slower than commanded: about 170 steps for a 40 cm carry, so three tubes did not
  fit in 60 s.

## Results

| Task | Skill | Seeds 0-99 | Stress x2 (seeds 0-49) |
| --- | --- | --- | --- |
| Rack fill | FillRack 0.1.0 | **99/100** (1 collision) | 40/50 (timeout 9, collision 1) |
| Tube into rack | TubeIntoRack 0.2.0 | **100/100** (0.1.0: 99) | 48/50 (0.1.0: 78/100 on seeds 0-99) |

A full rack fill takes about 690 steps (35 s sim time). Under stress x2 most failures are
timeouts: jam retries when the noise exceeds the clearance, on top of three slow carries.

## Not done
- A fallen tube is still not recovered.
- Any-tube-to-any-slot assignment: the task fixes which tube goes where.
- Sorting by cap colour, which needs perception in the loop.
- Real tube and rack dimensions.
