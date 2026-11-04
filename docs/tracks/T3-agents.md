# T3 · Agent Orchestrator (Phase 1)

**Goal:** a `SkillImprover` that reads an `EvalReport` and rewrites the skill until it passes, plus a CLI that runs the forge loop.

## Deliverables
1. `agents/llm_improver.py`: `ClaudeImprover(model="claude-opus-5-5")` implementing `SkillImprover`.
   - Prompt carries: TaskSpec JSON, current `SkillSource.code`, `EvalReport.summary()`, the top failure cases with their randomization values, the Skill/Observation/Action contract source, and the primitive library API (T2).
   - Output: one complete Python file + a short rationale, parsed from a fenced block. Validate by compiling and `load_skill` in a sandbox before returning; on error, retry with the traceback (max 2).
   - Roles: **planner** (which failure to fix first and why), **coder** (writes the file), **critic** (reads the new report vs the old, accepts or reverts). Start as three prompts in one class; split later.
   - Backend: Anthropic Messages API via `ANTHROPIC_API_KEY`. Add a `FakeImprover` for tests that applies scripted edits.
2. `agents/sandbox.py`: run a skill's rollouts in a subprocess with a timeout and no network, returning the `EvalReport` as JSON. `forge()` must be able to use it.
3. `agents/library.py`: a versioned skill store: `runs/library/<task_id>/<version>/{skill.py, report.json, rationale.md}`, keep-the-best policy (never accept a version whose success rate is lower than the best's 95% CI lower bound), lineage (parent_version).
4. `skillforge/cli.py` + `[project.scripts] skillforge = ...`: `skillforge forge tasks/X.json --seed-skill pick_place --rounds 8 --rollouts 100 --improver claude|fake|null`, printing one line per round and writing a `runs/forge/<timestamp>/history.json` that T4's dashboard reads.
5. Tests with `FakeImprover`: the loop accepts an improvement, rejects a regression, survives a skill that doesn't compile, and respects the sandbox timeout.

## Notes
- Cost guard: log input/output tokens per round to history.json; a `--budget-usd` flag stops the loop.
- Don't change contracts; request changes in HANDOVER.md.
