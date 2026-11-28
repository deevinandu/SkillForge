"""The coordinator's side of the scoped lesson library (docs/LESSONS.md).

- `record_round`: after an accepted repair, turn the improver's LESSONS.json into a proposed
  lesson (with provenance, the diff and the structural label) and file its grades of the lessons
  it was shown.
- `verify_lesson`: ASPIRE's verifier. A FRESH improver gets the parent skill and the library plus
  the proposal, never the proposer's code, and must reproduce a paired gain over the parent on
  seeds the proposer never saw, within `attempts` tries.
- `check_scope`: the lesson applied to another robot's current skill (a fresh improver told to
  apply that lesson and nothing else), scored paired on that robot's seeds.
- `admit`: decide the scope (`scope.decide_scope`) and promote a verified lesson.

Improvers take the lesson text through two optional attributes, read with getattr like the
critic hook: `lessons_text` (shown in the prompt) and `directive` (an extra instruction). The
LESSONS.json the improver wrote is read back from its `lesson_report` attribute.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from skillforge.contracts import SimEnv, SkillImprover, SkillSource, TaskSpec

from .lessons import Grade, Lesson, LessonReport, LessonStore, Provenance, Scope, render
from .library import DEFAULT_ALPHA, PairedComparison, passes_paired_gate
from .sandbox import evaluate_source
from .scope import check_from_outcomes, decide_scope, structural_scope

DEFAULT_VERIFY_SEEDS: tuple[int, ...] = tuple(range(1000, 1050))  # never gate (0..n) or held-out (100+) seeds

APPLY_ONE = (
    "Apply lesson {id} ('{title}') to this skill, adapted to this robot and task, and make no other "
    "change. If it cannot apply here, leave the skill's behaviour unchanged and say why in RATIONALE.md."
)
REPRODUCE = (
    "Lesson {id} ('{title}') is a candidate for the shared library. Use the lessons above, and that one "
    "in particular, to improve this skill. You have not seen the code it came from; follow the text."
)


def unified_diff(parent: SkillSource, cand: SkillSource, context: int = 2) -> str:
    return "".join(difflib.unified_diff(
        parent.code.splitlines(keepends=True), cand.code.splitlines(keepends=True),
        fromfile=f"{parent.name}@{parent.version}", tofile=f"{cand.name}@{cand.version}", n=context,
    ))


def record_round(
    store: LessonStore, report: LessonReport | None, *, task: TaskSpec, robot: str,
    parent: SkillSource, cand: SkillSource, accepted: bool, cmp: PairedComparison | None,
    success_before: float | None, success_after: float | None, library_sha: str,
) -> Lesson | None:
    """File the grades in `report`, and, if the round was accepted, its proposal as a lesson.
    Returns the proposed lesson, if any. Grades are kept for rejected rounds too: a lesson that
    misled a failed attempt is exactly what the coordinator must hear about."""
    if report is None:
        return None
    known = {les.id for les in store.all()}
    for g in report.consulted:
        if g.lesson in known:
            store.add_grade(g.lesson, Grade(robot=robot, task_id=task.id, verdict=g.verdict,
                                            evidence=g.evidence, library_sha=library_sha))
    p = report.proposal
    if not accepted or p is None:
        return None
    agent_scope = None
    try:
        agent_scope = Scope.parse(p.scope) if p.scope else None
    except ValueError:
        pass
    les = Lesson(
        id=store.new_id(p.title), title=p.title, topic=p.topic, failure_signature=p.failure_signature,
        trigger=p.trigger, precondition=p.precondition, strategy=p.strategy, code=p.code, params=p.params,
        source=Provenance(
            robot=robot, task_id=task.id, skill=cand.name, parent_version=parent.version, version=cand.version,
            gate=cmp.as_dict() if cmp else None, success_before=success_before, success_after=success_after,
            diff=unified_diff(parent, cand),
        ),
        agent_scope=agent_scope, agent_reason=p.scope_reason,
        structural_scope=structural_scope(parent.code, cand.code),  # type: ignore[arg-type]
    )
    return store.propose(les)


@dataclass
class Attempt:
    version: str
    success_rate: float
    paired: dict | None
    passed: bool
    error: str = ""


def _run_improver(
    improver: SkillImprover, task: TaskSpec, skill: SkillSource, report, lessons_text: str, directive: str,
) -> SkillSource:
    if hasattr(improver, "lessons_text"):
        improver.lessons_text = lessons_text
    if hasattr(improver, "directive"):
        improver.directive = directive
    return improver.propose(task, skill, report)


def verify_lesson(
    store: LessonStore, lesson_id: str, *, task: TaskSpec, parent: SkillSource,
    make_env: Callable[[TaskSpec], SimEnv] | str, improver_factory: Callable[[], SkillImprover],
    seeds: Sequence[int] = DEFAULT_VERIFY_SEEDS, attempts: int = 2, alpha: float = DEFAULT_ALPHA,
    timeout_s: float = 600.0,
) -> Lesson:
    """Blind re-derivation of a proposed lesson's gain. Verified = a paired gain over `parent`
    (the proposer's parent version) on `seeds` within `attempts` improver calls."""
    les = store.get(lesson_id)
    seeds = list(seeds)
    base = evaluate_source(task, parent, make_env, seeds, timeout_s=timeout_s, max_failures=len(seeds))
    text = render([*store.for_robot(les.source.robot, "scoped"), les], robot=les.source.robot)
    improver = improver_factory()  # fresh: no notes, no workspace from the proposer
    tries: list[Attempt] = []
    for _ in range(attempts):
        try:
            cand = _run_improver(improver, task, parent, base.report, text, REPRODUCE.format(id=les.id, title=les.title))
        except Exception as e:  # an improver that produced nothing is a failed attempt
            tries.append(Attempt(version="", success_rate=0.0, paired=None, passed=False, error=f"{type(e).__name__}: {e}"))
            continue
        res = evaluate_source(task, cand, make_env, seeds, timeout_s=timeout_s, max_failures=len(seeds))
        ok, cmp = passes_paired_gate(base.outcomes, res.outcomes, alpha)
        tries.append(Attempt(cand.version, res.report.success_rate, cmp.as_dict() if cmp else None, ok, res.error))
        if ok:
            break
    verified = any(t.passed for t in tries)
    record = {
        "seeds": [seeds[0], seeds[-1], len(seeds)], "attempts_used": len(tries), "alpha": alpha,
        "base_version": parent.version, "base_success_rate": base.report.success_rate,
        "attempts": [t.__dict__ for t in tries], "verdict": "verified" if verified else "refuted",
    }
    return store.set_verification(lesson_id, verified, record)


@dataclass
class Target:
    """Another robot's current skill for a task: where a lesson is checked."""

    robot: str
    task: TaskSpec
    skill: SkillSource


def check_scope(
    store: LessonStore, lesson_id: str, targets: Sequence[Target], *,
    make_env: Callable[[TaskSpec], SimEnv] | str, improver_factory: Callable[[], SkillImprover],
    seeds: Sequence[int] = DEFAULT_VERIFY_SEEDS, alpha: float = DEFAULT_ALPHA, timeout_s: float = 600.0,
) -> Lesson:
    """Apply the lesson to each target robot's skill and score it paired against that skill."""
    les = store.get(lesson_id)
    seeds = list(seeds)
    for t in targets:
        base = evaluate_source(t.task, t.skill, make_env, seeds, timeout_s=timeout_s, max_failures=len(seeds))
        try:
            applied = _run_improver(improver_factory(), t.task, t.skill, base.report, render([les], robot=t.robot),
                                    APPLY_ONE.format(id=les.id, title=les.title))
        except Exception as e:
            store.add_scope_check(lesson_id, check_from_outcomes(
                t.robot, t.task.id, t.skill.version, None, base.outcomes, None, alpha, note=f"{type(e).__name__}: {e}"))
            continue
        res = evaluate_source(t.task, applied, make_env, seeds, timeout_s=timeout_s, max_failures=len(seeds))
        store.add_scope_check(lesson_id, check_from_outcomes(
            t.robot, t.task.id, t.skill.version, applied.version, base.outcomes, res.outcomes, alpha, note=res.error))
    return store.get(lesson_id)


def admit(store: LessonStore, lesson_id: str, method: str = "checking") -> Lesson:
    """Decide the scope of a verified lesson and promote it. A refuted lesson is left as it is."""
    les = store.get(lesson_id)
    if les.status != "verified":
        return les
    store.set_scope(lesson_id, decide_scope(les, method), method)
    return store.promote(lesson_id)
