# SkillForge

**CI/CD for robot behaviors.** Describe a task; a team of AI agents writes a robot skill,
stress-tests it under domain randomization, fixes what breaks, turns it into training data
for a learned policy, and (later) validates it on a real arm.

> Status: Phase 0. Contracts and a stub pipeline run end to end. Nothing below is on hardware yet.

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
