# Sort tubes by cap colour, using the wrist camera (2026-11-24)

Third labware task. The robot must LOOK to know which tube is which colour. Sim only (Franka
Panda in MuJoCo); nothing here ran on hardware.

## Task (`tasks/franka_sort_tubes_by_color.json`)
- Four 16 x 100 mm tubes, two red and two blue. `shuffle_colors` permutes the colours among them
  every seed, so the names and the task file do not say which tube is which colour.
- Two grey two-slot racks, red on the left, blue on the right; same slot size as before (about
  2 mm of clearance a side).
- Success: `all_of` two `color_in_region` conditions (every tube that is red THIS episode is in
  the red rack, every blue one in the blue rack), judged on the true state and the drawn colours.
- The scene asks for the wrist camera in `obs_cameras`, so it arrives in `Observation.images`.
  1600 steps (80 s).

## Skill (`skills/sort_tubes.py`, `SortTubes` 0.1.0)
1. **Look.** For each tube, the wrist camera goes 12 cm above the tube's top. The skill reads the
   mean colour of a 9 x 9 pixel window at the image centre over 3 frames, then classifies it by
   chromaticity against reference colours: red and blue as measured in the sim, plus the table
   as a "nothing here" class. The tubes render washed out under the overhead light (red reads
   pink), so the nominal palette colours did not work: every read was rejected until the
   references were calibrated. On hardware this is a colour calibration under the cell's light.
2. **Sort.** Each tube goes into the next free slot of its colour's rack, through `TubeIntoRack`,
   with the fingers closing across the rack row and a narrow release (as in `FillRack`).

## Results (seeds 0-99)

| Skill | Success |
| --- | --- |
| `SortTubes` 0.1.0 (camera) | **97/100** |
| Same skill, but trusting the task file's colours instead of the camera | 13/100 |

The blind version succeeded exactly when the shuffle happened to leave the file's colours right
(13/100; 1 in 6 is expected for two reds and two blues).

The three camera failures, traced:

| Seed | Colours read right | Failure |
| --- | --- | --- |
| 18 | 4/4 | a tube slipped on the carry |
| 66 | 3/4 | one tube unreadable (no class close enough), so skipped |
| 85 | 3/4 | one tube unreadable, so skipped |

No tube was ever read as the wrong colour; the misses were "could not tell", which the skill
reports and skips. In 4 traced seeds all 16 reads were right.

## Not done
- Re-looking at an unreadable tube from a second pose or with a wider window.
- Colour reads under visual randomization (lighting, colour jitter: T6's `VisualRandomization`).
- More colours or more tubes than slots.
- The real cell: camera calibration, colour calibration and real tube caps.
- A regression baseline: CI renders on the CPU, so 100 camera episodes would add many minutes
  per run. `tests/test_sort_tubes.py` runs two seeds end to end instead.
