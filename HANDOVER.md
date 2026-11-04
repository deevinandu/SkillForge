# HANDOVER: cross-session progress log

Every session reads this first and appends an entry last. Newest first.
Entry format: `- YYYY-MM-DD · track · done · next · gotchas`.

## Status
- Phase 0 (contracts, stubs, gate): **done** 2026-11-04.
- Phase 1 (T1 sim, T2 skills, T3 agents, T4 eval): open, run in parallel.

## Contract requests
(none yet)

## Log
- 2026-11-04 · Phase 0 · Repo, contracts (`specs.py` TaskSpec/SceneSpec/RandomizationSpec; `runtime.py` Observation/Action/SimEnv/Skill/Episode/EvalReport; `agents.py` SkillSource/SkillImprover), `StubEnv` (kinematic point gripper), scripted closed-loop `PickPlace`, `evaluate()` harness with Wilson CI and failure tags, `NullImprover`, `forge()` loop, `tasks/stub_pick_cube_to_bin.json`. 13 tests pass. Gate met: stub pipeline runs end to end (20/20 on the stub task). · Next: Phase 1 tracks, briefs in docs/tracks/. · Gotchas: the stub is too easy (100% under its randomization); real difficulty arrives with T1's physics. `load_skill` runs agent code in-process; T3 must sandbox it.
