# SkillForge

**CI/CD for robot behaviors.** Describe a task; a team of AI agents writes a robot skill,
stress-tests it under domain randomization, fixes what breaks, turns it into training data
for a learned policy, and (later) validates it on a real arm.

> Status (2026-11-16): Phases 0-2 are done **in simulation** (MuJoCo Franka, agent-written skills,
> eval and regression harness, a learned policy at parity with its teacher). Phase 3 is in
> progress: camera-based pose estimation is measured in sim, and the real-robot env, safety
> limits, session runner and system ID are built and tested against fakes and a simulated twin.
> **Nothing has run on a real arm yet**, and the VLM success judge has never made a real API
> call. `HANDOVER.md` has the measurements and what each one does and does not show.

## This fork's research question
Team Echo (CMU Robot Autonomy) uses SkillForge to study a **skill library shared across robot
arms with scoped repairs** (general vs robot-specific fixes), building on ASPIRE
(arXiv 2607.00272). See [`docs/PROJECT.md`](docs/PROJECT.md) and the brief
[`docs/tracks/scoped-library.md`](docs/tracks/scoped-library.md). Cross-robot work has not
started: the sim supports only the Franka Panda.

## The loop
```
TaskSpec ──▶ skill (agent-written) ──▶ randomized rollouts ──▶ EvalReport
                ▲                                                  │
                └──────────── agent reads failures, rewrites ◀─────┘
hardened skill ──▶ demos ──▶ diffusion policy / ACT ──▶ real robot ──▶ failures feed the twin
```

## Quickstart
```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,sim]"
.venv/bin/pytest -q
```

## Tracks
T1 sim · T2 skills · T3 agents · T4 eval/CI · T5 learning · T6 perception · T7 real robot · T8 scale and research.
See `docs/ROADMAP.md` and `docs/tracks/`.

## Built with agents
This project is built by directing teams of Claude Code agents in parallel tracks against
frozen interface contracts. `CLAUDE.md` and `HANDOVER.md` are the agents' shared context,
committed on purpose: how the agents are directed is part of the work.
