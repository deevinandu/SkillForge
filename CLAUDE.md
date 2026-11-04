# SkillForge: rules for every agent session

SkillForge is CI/CD for robot behaviors: a task goes in; agents write a skill, harden it under
domain randomization, turn it into training data and a learned policy, then deploy it on a real
robot. Real failures feed back into the twin and the tests. Roadmap: docs/ROADMAP.md.

## Read first, write last
1. Read this file, `HANDOVER.md`, and your track brief in `docs/tracks/`.
2. Do the work on your track branch (`track/tN-name`).
3. Before you stop, append a progress entry to `HANDOVER.md`: date, track, what was done, what's next, gotchas.

## Layout and ownership
| Path | Owner | Notes |
| --- | --- | --- |
| `src/skillforge/contracts/` | nobody (frozen) | changes only by a PR that touches contracts alone |
| `src/skillforge/sim/` | T1 | every env implements `contracts.SimEnv` |
| `src/skillforge/skills/` | T2 | every skill implements `contracts.Skill` |
| `src/skillforge/agents/` | T3 | every improver implements `contracts.SkillImprover` |
| `src/skillforge/eval/` | T4 | harness, robustness sweeps, regression, dashboard |
| `src/skillforge/learning/` | T5 | |
| `src/skillforge/perception/` | T6 | |
| `src/skillforge/real/` | T7 | |
| `src/skillforge/forge.py` | shared spine | change its body, not its signature |
| `tasks/*.json` | anyone | a `TaskSpec` each; add, don't edit others' |

Stay inside your track's directories. If you need a contract change, write it up under
"Contract requests" in HANDOVER.md and work around it; don't edit `contracts/` from a track branch.

## Hard rules
- `pytest -q` and `ruff check src tests` pass before every commit. Add tests with every feature.
- Success is judged on TRUE sim state, never on noisy observations.
- Randomness comes only from the env seed. Same seed means same episode.
- Units: metres, kilograms, seconds, radians. Quaternions are (w, x, y, z), MuJoCo order.
- No secrets in the repo. LLM keys come from the environment (`ANTHROPIC_API_KEY`).
- Large files (meshes, videos, datasets, checkpoints) stay out of git: `runs/`, `assets/cache/` are ignored.
- Be honest in reports and README: say what is measured in sim, what on hardware, what is untested.

## Commands
```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,sim]"
.venv/bin/pytest -q
.venv/bin/ruff check src tests
```
Headless rendering: `MUJOCO_GL=egl` (GPU) or `MUJOCO_GL=osmesa` (CPU-only cloud box).
