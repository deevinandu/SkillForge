"""The full forge loop: sandboxed eval, keep-the-best gate, optional critic, library, budget.

Gate (`gate=`): every round is scored on the SAME gate seeds 0..rollouts-1, so a candidate is
compared with the best version seed by seed.
- 'paired' (default): accept only if the candidate solves more gate seeds than the best AND a
  one-sided exact McNemar test on the discordant seeds gives p < `alpha` (default 0.1). See
  `library.passes_paired_gate`.
- 'ci' (legacy): accept unless the candidate's rate is below the best's 95% CI lower bound. It
  let no-gain versions through (hard task 2026-11-04, round 1: +11/-7 seeds, p 0.24, accepted).
Paired counts are recorded for every evaluated round under either rule.

`skillforge.forge.forge()` delegates here; the CLI calls it directly to get the extras.

history.json schema (read by T4's dashboard), written after every round:
{
  "schema": "skillforge.forge_history/1",
  "task_id", "started_at", "meta": {...cli args, improver, model},
  "stopped": "target" | "rounds" | "budget" | null (still running),
  "best_version", "best_success_rate",
  "usage_total": {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens, usd, calls},
  "gate": {"rule": "paired" | "ci", "alpha"},
  "rounds": [{
    "round", "version", "parent_version", "rationale", "accepted", "reason", "critic",
    "sandbox_stage", "sandbox_error", "eval_wall_s",
    "report": EvalReport,
    "success_seeds": [gate seeds this version solved] | null (not evaluated this round),
    "paired": {"vs_version", "n", "both", "best_only", "cand_only", "neither", "p_value"} | null
              (candidate vs the best at the time; null for the seed and unevaluated rounds),
    "usage": {...same keys as usage_total, this round only}, "usd_total"
  }]
}
"gate" and the per-round "success_seeds"/"paired" were added after schema /1 shipped; readers
must treat them as optional.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from skillforge.contracts import EvalReport, SimEnv, SkillImprover, SkillSource, TaskSpec

from .budget import BudgetExceeded
from .curate import record_round
from .lessons import LessonStore, render
from .library import (
    DEFAULT_ALPHA,
    GATE_RULES,
    PairedComparison,
    SkillLibrary,
    bump_version,
    passes_gate,
    passes_paired_gate,
)
from .sandbox import evaluate_source

_ZERO = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "usd": 0.0,
    "calls": 0,
}


class RoundRecord(BaseModel):
    round: int
    source: SkillSource
    report: EvalReport
    accepted: bool
    reason: str
    critic: str = ""
    sandbox_stage: str = ""
    sandbox_error: str = ""
    eval_wall_s: float = 0.0
    usage: dict[str, float] = Field(default_factory=lambda: dict(_ZERO))
    usd_total: float = 0.0
    outcomes: dict[int, bool] | None = None  # gate seed -> success; None if not evaluated this round
    paired: dict | None = None  # PairedComparison.as_dict() + vs_version, vs the best at the time
    lessons_sha: str | None = None  # the promoted lesson library the improver was shown (None: no store)
    lessons_shown: list[str] = Field(default_factory=list)  # their ids
    lesson_proposed: str | None = None  # id of the lesson this round proposed


class LoopResult(BaseModel):
    task_id: str
    rounds: list[RoundRecord]
    best: RoundRecord
    stopped: str


def _usage(improver: Any) -> dict[str, float]:
    u = getattr(improver, "usage", None)
    return u.snapshot() if u is not None else dict(_ZERO)


def _delta(a: dict, b: dict) -> dict:
    return {k: b.get(k, 0) - a.get(k, 0) for k in _ZERO}


def run_loop(
    task: TaskSpec,
    seed_skill: SkillSource,
    improver: SkillImprover,
    make_env: Callable[[TaskSpec], SimEnv] | str,
    rounds: int = 5,
    rollouts: int = 50,
    target: float = 0.9,
    timeout_s: float = 600.0,
    library: SkillLibrary | None = None,
    history_path: str | Path | None = None,
    on_round: Callable[[RoundRecord], None] | None = None,
    meta: dict | None = None,
    gate: str = "paired",
    alpha: float = DEFAULT_ALPHA,
    lessons: LessonStore | None = None,
    robot: str | None = None,
    lesson_policy: str = "scoped",
) -> LoopResult:
    """`rounds` counts evaluations, seed included. Every round uses the SAME seeds.

    `gate` is 'paired' (default) or 'ci' (legacy); `alpha` is the paired gate's significance level.

    `lessons` (docs/LESSONS.md): before each proposal the improver is shown the promoted lessons
    for `robot` (default `task.scene.robot.name`) under `lesson_policy` ('scoped', 'share_all',
    'none'), through its optional `lessons_text` attribute. Afterwards its `lesson_report`
    (LESSONS.json) is filed: grades of the lessons it saw, and, for an accepted round, a proposed
    lesson. Proposals still need `curate.verify_lesson` and `curate.admit` to enter the library.
    """
    if gate not in GATE_RULES:
        raise ValueError(f"gate must be one of {GATE_RULES}, got {gate!r}")
    seeds = list(range(rollouts))
    robot = robot or task.scene.robot.name
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    used: set[str] = set()
    history: list[RoundRecord] = []

    def fresh(v: str) -> str:
        while v in used or (library is not None and library.has(task.id, v)):
            v = bump_version(v)
        used.add(v)
        return v

    def write(stopped: str | None) -> None:
        if history_path is None:
            return
        best = _best(history)
        doc = {
            "schema": "skillforge.forge_history/1",
            "task_id": task.id,
            "started_at": started,
            "meta": meta or {},
            "stopped": stopped,
            "best_version": best.source.version if best else None,
            "best_success_rate": best.report.success_rate if best else None,
            "usage_total": _usage(improver),
            "gate": {"rule": gate, "alpha": alpha},
            "rounds": [
                {
                    "round": h.round,
                    "version": h.source.version,
                    "parent_version": h.source.parent_version,
                    "rationale": h.source.rationale,
                    "accepted": h.accepted,
                    "reason": h.reason,
                    "critic": h.critic,
                    "sandbox_stage": h.sandbox_stage,
                    "sandbox_error": h.sandbox_error,
                    "eval_wall_s": h.eval_wall_s,
                    "report": json.loads(h.report.model_dump_json()),
                    "success_seeds": None if h.outcomes is None else sorted(s for s, ok in h.outcomes.items() if ok),
                    "paired": h.paired,
                    "usage": h.usage,
                    "usd_total": h.usd_total,
                    "lessons_sha": h.lessons_sha,
                    "lessons_shown": h.lessons_shown,
                    "lesson_proposed": h.lesson_proposed,
                }
                for h in history
            ],
        }
        p = Path(history_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=2))
        tmp.replace(p)

    def finish(rec: RoundRecord, u0: dict, store: bool = True) -> None:
        u1 = _usage(improver)
        rec.usage, rec.usd_total = _delta(u0, u1), float(u1.get("usd", 0.0))
        history.append(rec)
        if store and library is not None:
            library.add(rec.source, rec.report, rec.accepted, note=rec.reason)
        tell = getattr(improver, "record_outcome", None)  # optional: lets an improver learn the verdict
        if store and rec.round > 0 and callable(tell):
            try:
                tell(rec.source.version, rec.accepted, rec.reason)
            except Exception:
                pass
        write(None)
        if on_round:
            on_round(rec)

    def evaluate(src: SkillSource):
        return evaluate_source(task, src, make_env, seeds, timeout_s=timeout_s)

    # -- round 0: the seed skill. It is the baseline, accepted by definition.
    u0 = _usage(improver)
    seed = seed_skill.model_copy(update={"version": fresh(seed_skill.version)})
    res = evaluate(seed)
    finish(
        RoundRecord(
            round=0, source=seed, report=res.report, accepted=True, reason="seed",
            sandbox_stage=res.stage, sandbox_error=res.error, eval_wall_s=res.wall_s, outcomes=res.outcomes,
        ),
        u0,
    )
    current, cur_report = seed, res.report
    best = history[0]
    stopped = "target" if cur_report.success_rate >= target else "rounds"
    review = getattr(improver, "review", None)

    for r in range(1, rounds):
        if stopped == "target":
            break
        u0 = _usage(improver)
        shown, sha = _show_lessons(improver, lessons, robot, lesson_policy)
        try:
            cand = improver.propose(task, current, cur_report)
        except BudgetExceeded as e:
            stopped = "budget"
            _note(history, f"stopped before round {r}: {e}")
            break
        except Exception as e:  # an improver failure costs a round, not the run
            rec = RoundRecord(
                round=r, source=current, report=cur_report, accepted=False,
                reason=f"improver error: {type(e).__name__}: {e}"[:2000], lessons_sha=sha, lessons_shown=shown,
            )
            finish(rec, u0, store=False)
            continue

        if cand.code == current.code:
            finish(RoundRecord(round=r, source=current, report=cur_report, accepted=False, reason="unchanged",
                               lessons_sha=sha, lessons_shown=shown), u0, store=False)
            continue

        cand = cand.model_copy(update={"version": fresh(cand.version), "parent_version": current.version})
        res = evaluate(cand)
        rec = RoundRecord(
            round=r, source=cand, report=res.report, accepted=False, reason="",
            sandbox_stage=res.stage, sandbox_error=res.error, eval_wall_s=res.wall_s, outcomes=res.outcomes,
        )
        paired_ok, cmp = passes_paired_gate(best.outcomes, res.outcomes, alpha) if best.outcomes else (False, None)
        if cmp is not None:
            rec.paired = {"vs_version": best.source.version, **cmp.as_dict()}
        if not res.ok:
            rec.reason = f"sandbox {res.stage}"
        elif gate == "ci":
            if not passes_gate(best.report, res.report):
                lo = best.report.ci95[0]
                rec.reason = f"regression: {res.report.success_rate:.0%} < best {best.source.version} CI lower bound {lo:.0%}"
            else:
                rec.accepted, rec.reason = True, "passes keep-the-best gate (ci)"
        elif cmp is None:
            raise RuntimeError("paired gate needs per-seed outcomes for the best version")
        elif not paired_ok:
            rec.reason = _paired_reject(cmp, best.source.version, alpha)
        else:
            rec.accepted, rec.reason = True, f"passes paired gate vs {best.source.version}: {cmp.short()} < {alpha:g}"
        rec.lessons_sha, rec.lessons_shown = sha, shown
        if rec.accepted:
            if review is not None:
                try:
                    ok, why = review(task, current, cur_report, cand, res.report)
                    rec.critic = why
                    if not ok:
                        rec.accepted, rec.reason = False, "critic reverted"
                except BudgetExceeded as e:
                    rec.critic = f"critic skipped: {e}"
                    stopped = "budget"
                except Exception as e:
                    rec.critic = f"critic failed, deferring to gate: {type(e).__name__}: {e}"[:2000]
        if lessons is not None:
            rec.lesson_proposed = _file_lessons(
                lessons, improver, task=task, robot=robot, parent=current, cand=cand, rec=rec,
                cmp=cmp, before=cur_report.success_rate, sha=sha,
            )
        finish(rec, u0)

        if rec.accepted:
            current, cur_report = cand, res.report
            if res.report.success_rate > best.report.success_rate:
                best = rec
            if res.report.success_rate >= target:
                stopped = "target"
        if stopped == "budget":
            break

    write(stopped)
    return LoopResult(task_id=task.id, rounds=history, best=best, stopped=stopped)


def _show_lessons(improver: Any, store: LessonStore | None, robot: str, policy: str) -> tuple[list[str], str | None]:
    if store is None:
        return [], None
    shown = store.for_robot(robot, policy)  # type: ignore[arg-type]
    if hasattr(improver, "lessons_text"):
        improver.lessons_text = render(shown, robot=robot)
    if hasattr(improver, "lesson_report"):
        improver.lesson_report = None  # never file last round's report again
    return [les.id for les in shown], store.sha(shown)


def _file_lessons(store: LessonStore, improver: Any, *, task, robot, parent, cand, rec, cmp, before, sha) -> str | None:
    report = getattr(improver, "lesson_report", None)
    if report is None:
        return None
    try:
        les = record_round(
            store, report, task=task, robot=robot, parent=parent, cand=cand, accepted=rec.accepted, cmp=cmp,
            success_before=before, success_after=rec.report.success_rate, library_sha=sha or "",
        )
    except Exception as e:  # a bad report costs the lesson, not the run
        _note([rec], f"lesson report not filed: {type(e).__name__}: {e}"[:500])
        return None
    return les.id if les else None


def _paired_reject(cmp: PairedComparison, vs: str, alpha: float) -> str:
    if cmp.cand_only < cmp.best_only:
        return f"regression: {cmp.short()} seeds vs best {vs}"
    return f"no significant gain vs {vs}: {cmp.short()} (alpha {alpha:g})"


def _best(history: list[RoundRecord]) -> RoundRecord | None:
    acc = [h for h in history if h.accepted]
    return max(acc, key=lambda h: h.report.success_rate) if acc else None  # max keeps the first on ties


def _note(history: list[RoundRecord], msg: str) -> None:
    if history:
        last = history[-1]
        last.critic = (last.critic + "\n" if last.critic else "") + msg
