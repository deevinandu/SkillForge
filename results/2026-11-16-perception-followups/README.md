# Follow-ups to the perception robustness sweep (T6, sim only)

2026-11-16. Cube hard task, skill `results/2026-11-04-claude-code/hard/pick_place_0.2.2.py`,
held-out seeds 5000-5099 (100 per cell). Nothing here ran on hardware.

## Why high actuator gain did better under perception

The sweep in `results/2026-11-15-perception-robustness/` had gain 2x at 74 with the env's own
poses and 93 / 89 with perception. The guess was that the env's per-step pose noise shakes an
over-driven arm. Tested in plain sim (env's own poses), pinning gain and observation noise:

| Observation noise | Gain 1x | Gain 2x |
| --- | --- | --- |
| none | 99 | 87 |
| 5 mm | 98 | 84 |
| 15 mm | 89 | 46 |
| task's range (0-15 mm) | 98 | 74 |

The guess holds. Noise and gain interact: 15 mm of pose noise costs 10 points at nominal gain
and 41 at double gain (38 of the 54 failures are grasp slips). With no pose noise, gain 2x
gives 87, close to the 93 and 89 seen under perception, which does not see that noise. So the
perception rows were not more robust to gain; they were spared the noise, as with the stress
result. `gain_noise_interaction.json` has the reports.

What it means for a real arm: a stiff arm is far less tolerant of jittery pose estimates than
a soft one. T6's estimates are steady while the object is visible (under 1 mm) but jump by
1-3 cm in the hand, so this matters most during the carry.

## Sweeps no longer pile up render contexts

`python -m skillforge.perception robustness` now starts fresh worker processes for every
point (`cli.FreshPool`) and prints each point as it finishes. Before, every worker kept 8 envs
with their GL contexts alive (T4's pool cache): 11.5 of 12 GB of GPU memory on 12 workers,
and the `dr-color` sweep took 1 h 44 min against 30 min for `depth`. The sweep time after
the change has not been measured yet.
