# T4 · Eval and CI (Phase 1)

**Goal:** make "how good is this skill, and how does it break" fast, rigorous and visible.

## Deliverables
1. `eval/parallel.py`: `evaluate_parallel(make_env, skill_source, task, seeds, workers)` over a process pool (skills cross the process boundary as `SkillSource`). Same result as serial `evaluate` for the same seeds (test it).
2. `eval/robustness.py`: sweep one randomization axis at a time (e.g. obs noise 0→1 cm in 6 steps, 50 seeds each) and return a success-vs-value curve per axis; plus a "stress" preset that widens every range 2×. Output a `RobustnessReport` model (in `eval/`, not contracts).
3. `eval/regression.py`: for each `tasks/*.json` × each accepted skill in the library, a stored baseline; `pytest -m regression` fails when a skill drops below its baseline's CI lower bound.
4. `eval/dashboard.py`: render a single static HTML file (no server) from forge `history.json` + robustness reports: success rate per round with CI bars, failure-tag breakdown per round, robustness curves, the rationale text per version, embedded GIFs when episodes have video. Use inline SVG or a CDN chart lib.
5. `eval/video.py`: record selected episodes (first failure per tag, one success) via `env.render` to GIF with imageio; skip cleanly when rendering is unavailable.
6. Better failure tags in `harness._classify` using T1's events (`grasp_slip`, `collision`, `ik_failure`).
7. Tests for each; the dashboard test renders from a fixture history.

## Notes
- Develop on `StubEnv`; switch the examples to the Franka task when T1 merges.
- Don't change contracts; request changes in HANDOVER.md.
