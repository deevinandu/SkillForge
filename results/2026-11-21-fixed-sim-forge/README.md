# Forge runs on the fixed sim (2026-11-21)

Two runs of the agent loop after the arm-armature fix (#22, armature 0.5). Sim only. Both used
the headless Claude Code improver on the Pro subscription through the CLI's own login (no API
key, extra usage disabled, so no money spent). One improver call each.

```bash
MUJOCO_GL=egl skillforge forge tasks/<task>.json --seed-skill <seed> \
  --rounds 2 --rollouts 100 --improver claude-code --model claude-sonnet-5-5 \
  --budget-usd 3 --per-call-budget-usd 3 --gate paired --library runs/library_<x> --out runs/forge/<x>
```

## Cylinder: accepted

Seed: the agent-written `results/2026-11-05-claude-code-repeats/cylinder_r1/pick_place_0.2.1.py`.
It scored 98/100 before the fix and 88/100 after: it had been tuned against the servo artifact.

| | Gate seeds 0-99 | Held-out seeds 5000-5199 |
| --- | --- | --- |
| seed 0.1.0 | 88/100 (ik_failure 7, collision 5) | 177/200 (ik_failure 13, grasp_slip 6, collision 4) |
| rewrite 0.1.1 | **100/100** | **198/200** (ik_failure 1, grasp_slip 1) |

Paired gate: +12/-0, p = 0.0002, accepted. The held-out run was done afterwards by hand.

What changed (`cylinder/pick_place_0.1.1.py`, `cylinder/RATIONALE.md`): only the carry limits.
Horizontal speed 0.25 -> 0.18 m/s, vertical 0.1 -> 0.07 m/s, acceleration 0.5 -> 0.3 m/s^2.
The improver's diagnosis: most failures started as a slip of a heavy, low-friction can under
high gain. The re-grasp after the drop then ran into IK failures or finger-table collisions.
Improver cost figure from the CLI: $0.13 (2,028 output tokens, about 1 min).

## Far bin: rejected (no change)

Seed: library `PickPlace` 0.2.0, 77/100 on gate seeds (collision 8, wrong_placement 8,
timeout 5, grasp_slip 2). The improver tried capping the carry speed (0.02 and 0.05 m per step,
transport and lift). None beat the seed on its 30 dev seeds, so it reverted its changes. An
unused helper was left in the file. The rewrite scored the same 77/100 on the same seeds
(+0/-0) and was rejected. Its notes (`far_bin/RATIONALE.md`) put the failures at low
friction (0.33-0.42 scale) and at finger-table collisions during re-grasps. Neither is fixed.
Improver cost figure: $0.20 (4,958 output tokens).

## Not done
- Far bin with more rounds or a different seed skill.
- Neither rewrite was tested under stress or swept.
