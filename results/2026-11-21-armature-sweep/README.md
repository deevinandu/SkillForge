# Robustness sweep before and after the arm armature fix (2026-11-21)

Same skill (agent-written hard-cube `pick_place` 0.2.2), same task (`franka_pick_cube_to_bin_hard`),
same seeds (5000-5099, 100 per point), true poses, sim only. Before: the 2026-11-15 sweep
(`results/2026-11-15-perception-robustness/`, armature 0.1). After: branch `track/t1-armature`
(arm armature floored at 0.5, which stops the servos bang-banging between their force limits).

```bash
MUJOCO_GL=egl python -m skillforge.perception robustness franka_pick_cube_to_bin_hard \
  --skill results/2026-11-04-claude-code/hard/pick_place_0.2.2.py --seeds 5000-5100 \
  --envs true --workers 12 --out runs/armature/hard_true.json
```

Successes out of 100, before -> after:

| Axis | Points |
| --- | --- |
| xy jitter (m) | 0: 99->100, 0.02: 98->100, 0.04: 98->100, 0.06: 99->100, 0.08: 95->100, 0.1: 96->99 |
| yaw jitter (rad) | 0: 98->100, 0.31: 99->100, 0.63: 98->100, 0.94: 93->99, 1.26: 96->100, 1.57: 94->100 |
| friction scale | 0.25: 71->87, 0.5: 97->100, 0.75: 97->100, 1: 97->100, 1.5: 98->100, 2: 97->100 |
| mass scale | 0.5: 97->100, 1: 96->100, 1.5: 93->100, 2: 81->94, 3: 29->61, 4: 6->31 |
| actuator gain scale | 0.25: 36->44, 0.5: 94->97, 0.75: 99->100, 1: 98->100, 1.5: 87->98, 2: 74->98 |
| obs noise std (m) | 0: 99->99, 0.002: 97->100, ..., 0.01: 95->100 |
| action latency (steps) | 0: 97->100, ..., 6: 95->100 |
| stress x2 | 37 -> 47 (grasp_slip 41 -> 32) |

What this changes:
- The background 2-5% failure rate at every point was the artifact: nominal points are now ~100%.
- Breaking points move: mass 3x 29 -> 61, friction 0.25x 71 -> 87. Gain 2x (74 -> 98) is no longer a
  weakness, so the 2026-11-16 "gain x observation noise" explanation in
  `results/2026-11-16-perception-followups/` was measured on the artifact and needs re-checking.
- Stress x2 is still hard (47%), and still mostly slips: those look real.
- Every earlier robustness number in `results/` was measured with the artifact.
