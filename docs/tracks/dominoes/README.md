# Dominoes: topple a chain, recover from disturbances, transfer across arms

The project task for Team Echo's autonomy lab: a chain of dominoes the robot sets up, keeps in
line while someone (the env) bumps, nudges or knocks dominoes, and finally topples, with the same
agent-built skill library running on a Franka Panda and a Kinova Gen3.

What exists (branches, stacked in this order, each with a HANDOVER entry):

| Branch | Track | What |
| --- | --- | --- |
| `contracts/dominoes` | contracts | `object_toppled` success leaf, `SceneSpec.disturbances`, `unrecovered_disturbance` tag |
| `track/t1-dominoes` | T1 | seeded mid-episode disturbances in `MujocoEnv` and `StubEnv`, `toppled:` events |
| `track/t2-dominoes` | T2 | three domino tasks, `ToppleChain` (survey, repair, push, nudge), kinematic test world, `scripts/` |
| `track/t4-disturbance-eval` | T4 | failure tag ranking, `disturbance_scale` sweep axis |
| `docs/domino-briefs` | docs | these briefs |

**Status (MuJoCo, 2026-11-25, 20 seeds):** `franka_topple_domino_chain` 20/20,
`franka_topple_domino_chain_knock` 20/20, `franka_setup_domino_chain` 13/20 with the old staging
row (picks from x >= 0.56 slip); the staging row was then moved to where picks were 10/10, and
that layout still needs its MuJoCo run. All T1 domino tests pass on MuJoCo.

Try it: `python scripts/try_dominoes.py --view` (one episode with the skill's log; `--task`,
`--seed`, `--no-disturb`, `--gif`, `--seeds N`). Where picks fail: `python scripts/diag_pick.py`.

Tasks: `franka_topple_domino_chain`, `franka_topple_domino_chain_knock`, `franka_setup_domino_chain`,
`franka_build_domino_chain_scattered` (random starts: standing, face down, on the side, any yaw),
`franka_rebuild_domino_chain` (dominoes taken out and laid down beside the lane).
Standing a fallen domino up: `StandUp` in `skills/dominoes.py`; check it with
`python scripts/diag_standup.py`.

Briefs: [T1](T1-sim.md) · [T2](T2-skills.md) · [T3](T3-agents.md) · [T4](T4-eval.md) ·
[T5](T5-learning.md) · [T6](T6-perception.md) · [T7](T7-real.md)
