# Scoped lessons: the skill library, rebuilt on ASPIRE

Design for brief item 5 of [`tracks/scoped-library.md`](tracks/scoped-library.md). Background:
[`PROJECT.md`](PROJECT.md). Code: `src/skillforge/agents/lessons.py` (store, retrieval,
rendering, grades), `agents/scope.py` (structural label, scope decision), `agents/curate.py`
(verifier and cross-robot check), wired into `agents/loop.py` and the improvers.

## What was wrong
`agents/library.py` `SkillLibrary` stores **versions of whole skill files, per task**. What an
improver learned lived in `ClaudeCodeImprover.notes`, in memory, for one run, the last ten lines,
and never reached another task, another robot, or a later run. Operator labels and notes from real
trials never reached it either. So nothing learned was reused, and there was nothing to scope.

## ASPIRE, as its code does it (NVlabs/ASPIRE, `.claude/skills/iterative-debugging/`)
- **Library = text lessons**, grouped by topic (`grasp.md`, `transport.md`, ...). An entry is
  *trigger -> code -> evidence*: the symptom or scene condition, a 5-20 line working snippet,
  and provenance (task, seeds it flipped, held-out result, date). Thresholds go in one-line tables.
- **Route in**: the actor (one coding agent per task) writes `skill_report.json` with
  `proposed_new` / `proposed_edits`. It never edits the library.
- **Verifier**: a fresh agent re-solves the *same* task from the library plus the proposal alone,
  denied the proposer's program; 5 seeds, 2 attempts, pass at >= 4/5. Proposals are graded
  `sound` / `defective` / `untested`. Passing shows the text *transmits*, not that it transfers.
- **Coordinator** promotes verified proposals only, and repairs entries first: every actor grades
  each section it opened `useful` / `misleading` / `wrong` / `not-applicable`. `wrong` is fixed
  at once, `misleading` gets a narrower trigger or a runnable precondition.
- **Library hash** in every report; a final **rerun sweep** re-solves low scorers that ran on an
  older library. A rerun that scores worse is read as a harmful entry.
- **Not in ASPIRE**: entries are checked only on the task they came from, and nothing says
  which robot an entry is for (its code carries Panda offsets implicitly). No negative-transfer
  measurement.

## What we build (ASPIRE + scope)
| Piece | ASPIRE | Ours |
| --- | --- | --- |
| Entry | trigger, code, evidence | `Lesson`: title, topic, `failure_signature` (our failure tags), trigger, precondition, strategy, code sketch, `params` (the numbers it sets), provenance (robot, task, skill version and parent, the diff, the paired gate result) **and a scope** |
| Proposal | actor's `skill_report.json` | the improver writes `LESSONS.json` next to `RATIONALE.md`: grades of the lessons it was shown, and a proposal for its change with **its own scope label and reason** |
| Admission | blind verifier, same task | `curate.verify_lesson`: a fresh improver gets the parent skill and the lesson text only and must reproduce a **paired** gain on seeds the proposer never saw |
| Scope | none | three labels, all recorded: **agent** (the proposer's), **structural** (`scope.structural_scope`: a change that only moves numbers is parameter-only), **checking** (`curate.check_scope`: the lesson applied to each other robot's best skill, paired gate on that robot's seeds; any significant loss makes it robot-specific) |
| Retrieval | the whole library, as files | `LessonStore.for_robot(robot, policy)`; policies are the experiment's conditions: `scoped` (general + this robot's), `share_all` (every promoted lesson), `none` (from scratch) |
| Grades | per section opened | `Grade` on the lesson, with robot and task, so a lesson that is `useful` on one arm and `misleading` on another is visible as a scope problem |
| Library hash | per report | `LessonStore.sha()` in every round of `history.json` |

Status flow: `proposed -> verified | refuted -> promoted -> retired`. Only `promoted` lessons
are ever shown to an improver. A lesson's decided scope is `general` or `robot:<name>`.

### Deciding scope (the experiment variable)
`scope.decide_scope(lesson, method)` with `method` one of PROJECT.md's three ways:
- `structure`: parameter-only -> robot-specific, otherwise general;
- `agent`: the proposer's label;
- `checking` (default): robot-specific unless checked on at least one other robot with no
  significant loss there; with no other robot checked yet it stays robot-specific (safe default).
The scope-label error (agent vs checking) is `scope.label_confusion(lessons)`.

### Robot-specific numbers
Until `RobotSpec.params` exists (a contracts request in HANDOVER.md), a robot-specific lesson
carries its numbers in `params` (e.g. `{"upright_accel": 0.00012}`), and the rendered lesson
says which robot they were measured on. Once the contract lands, the same `params` become
`task.robot.params` entries.

## Not done yet
- The experiment runner (brief item 6) that runs the four conditions and the share-back step.
- Real-trial lessons: operator labels and notes from `runs/real/*/real.json` as proposals with
  scope `robot:<lab arm>` (the lab Franka's wrist lag is one).
- Edits to promoted lessons (ASPIRE's `proposed_edits`): today a repair is a new lesson that
  retires the old one.
