# Robustness sweeps on the fixed sim (2026-11-21)

T4 robustness sweeps after the arm-armature fix (#22, armature 0.5). Sim only. Seeds 5000-5099
(held out from the gate's 0-99), **100 per point**. Each axis is pinned to one value at a time,
and the other axes keep the task's ranges. At 90% success a 95% interval is about 83-95%, so
differences under about 7 points are noise.

```bash
MUJOCO_GL=egl python -m skillforge.perception robustness <task> [--skill <path>[:Class]] \
  --seeds 5000-5100 --workers 12 --envs <true|depth|dr-color> [--no-depth-noise] --out <json>
```

## Hard cube with perception, before -> after the fix

Skill: agent-written `results/2026-11-04-claude-code/hard/pick_place_0.2.2.py`. Before is the
2026-11-15 sweep (`results/2026-11-15-perception-robustness/`), after is this one. The
`true`-pose rerun is in `results/2026-11-21-armature-sweep/`.

`depth` (overview + wrist depth cameras, sim masks, clean depth, perfect calibration):

| Axis | Points |
| --- | --- |
| xy jitter (m) | 0: 100->100, .02: 100->100, .04: 98->100, .06: 95->100, .08: 96->100, .1: 91->100 |
| yaw jitter (rad) | 0: 96->100, .31: 98->99, .63: 97->100, .94: 94->100, 1.26: 98->100, 1.57: 98->100 |
| friction scale | .25: 69->87, .5: 96->100, .75: 97->100, 1: 95->100, 1.5: 97->100, 2: 97->100 |
| mass scale | .5: 93->100, 1: 96->100, 1.5: 96->100, 2: 80->93, 3: 33->60, 4: 3->31 |
| gain scale | .25: 36->40, .5: 92->97, .75: 99->100, 1: 90->100, 1.5: 90->89, 2: 93->97 |
| obs noise (m) | every point 97->100 |
| latency (steps) | 0-6: 95-97 -> 100 |
| stress x2 | 60->69 |
| depth noise (std at 1 m) | 0: 97->100, 1.5 mm: 97->100, 3 mm: 94->100, 6 mm: 89->98, 12 mm: 96->99 |

`dr-color` (colour segmentation, 3 mm depth noise, visual randomization including calibration error):

| Axis | Points |
| --- | --- |
| xy jitter (m) | 0: 98->97, .02: 93->98, .04: 96->98, .06: 90->98, .08: 90->100, .1: 87->95 |
| yaw jitter (rad) | 0: 91->98, .31: 93->99, .63: 93->99, .94: 91->99, 1.26: 90->97, 1.57: 89->100 |
| friction scale | .25: 68->84, .5: 89->100, .75: 86->99, 1: 92->99, 1.5: 94->99, 2: 90->100 |
| mass scale | .5: 91->99, 1: 91->99, 1.5: 93->99, 2: 81->90, 3: 32->58, 4: 6->30 |
| gain scale | .25: 36->41, .5: 92->95, .75: 99->99, 1: 84->99, 1.5: 84->90, 2: 89->92 |
| obs noise (m) | every point 89->99 |
| latency (steps) | 0-6: 89-91 -> 97-99 |
| stress x2 | 55->69 |

The 2026-11-15 run put the hardest perception setting about 10 points below `depth` on most
axes (89 vs 97). Most of that gap was the artifact: it is now 1-3 points (99 vs 100). The
remaining perception failures are grasp misses (6 per 100 at most) and a few wrong placements.
This run took 1 h 31 min on 12 workers (1 h 44 min on 2026-11-15). The `IndexError` seen in
the 2026-11-15 timing check did not recur, so it is still not traced.

What's left with clean depth: heavy objects (3x mass 60, 4x 31: grasp slips) and the weakest
servo gain (0.25x, 40). Those are real limits of the grip and the arm, not the artifact.

## First sweeps of the other tasks (true poses)

Library skills: `PickPlace` 0.2.0 for far bin, ball and cylinder; `PickFromClutter` 0.1.0; `PushToRegion` 0.2.0.

| Axis | Far bin | Ball | Cylinder | Clutter | Push |
| --- | --- | --- | --- | --- | --- |
| xy jitter 0 / .1 m | 78 / 80 | 97 / 93 | 94 / 88 | 100 / 98 | 100 / 98 |
| yaw jitter 0 / 1.57 | 79 / 80 | 95 / 96 | 88 / 92 | 100 / 100 | 100 / 100 |
| friction .25 / 1 / 2 | 66 / 82 / 81 | 84 / 95 / 98 | 93 / 95 / 92 | 100 / 100 / 100 | 100 / 100 / 99 |
| mass 1 / 3 / 4 | 80 / 40 / 13 | 95 / 95 / 93 | 87 / 81 / 51 | 100 / 100 / 100 | 99 / 100 / 97 |
| gain .25 / 1 / 2 | 32 / 78 / 51 | 96 / 95 / 67 | 71 / 87 / 54 | 100 / 100 / 99 | 93 / 100 / 100 |
| obs noise 0 / 8 / 10 mm | 100 / 100 / 76 | 100 / 97 / 89 | 99 / 92 / 76 | 100 / 99 / 92 | 100 / 98 / 99 |
| latency 0 / 6 steps | 83 / 77 | 95 / 93 | 91 / 87 | 100 / 100 | 97 / 100 |
| stress x2 | 20 | 38 | 34 | 100 | 100 |

What the sweeps show:
- **Far bin is limited by observation noise.** With noise pinned at 8 mm or less it succeeds
  100/100. The task samples noise up to 15 mm, so roughly the top third of episodes fail: 77-80
  on every other axis. Claude Code's far-bin round (`results/2026-11-21-fixed-sim-forge/`) blamed
  low friction and missed this. The agent-written hard-cube skill already handles 15 mm noise on
  the hard cube, so it is the obvious seed for the next far-bin round.
- **Ball and cylinder** also fall off above 8 mm of noise and at 2x gain (67 and 54). The
  cylinder also drops at 4x mass (51). These numbers are for the library `PickPlace`, not the
  cylinder skill accepted in #24, which has not been swept.
- **Clutter and push** are near 100 on every axis, including stress x2, because their task
  ranges are narrow (clutter noise tops out at 5 mm). These tasks are now too easy for the loop.

Run time on 12 workers: about 1-1.5 min per true-pose sweep, 25.5 min for `depth`.
