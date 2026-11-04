"""The full forge loop: sandboxed eval, keep-the-best gate, optional critic, library, budget.

`skillforge.forge.forge()` delegates here; the CLI calls it directly to get the extras.

history.json schema (read by T4's dashboard), written after every round:
{
  "schema": "skillforge.forge_history/1",
  "task_id", "started_at", "meta": {...cli args, improver, model},
  "stopped": "target" | "rounds" | "budget" | null (still running),
  "best_version", "best_success_rate",
  "usage_total": {input_tokens, output_tokens, cache_creation_input_tokens, cache_read_input_tokens, usd, calls},
  "rounds": [{
    "round", "version", "parent_version", "rationale", "accepted", "reason", "critic",
    "sandbox_stage", "sandbox_error", "eval_wall_s",
    "report": EvalReport,
    "usage": {...same keys as usage_total, this round only}, "usd_total"
  }]
}
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
from .library import SkillLibrary, bump_version, passes_gate
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
) -> LoopResult:
    """`rounds` counts evaluations, seed included. Every round uses the SAME seeds."""
    seeds = list(range(rollouts))
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
                    "usage": h.usage,
                    "usd_total": h.usd_total,
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
            sandbox_stage=res.stage, sandbox_error=res.error, eval_wall_s=res.wall_s,
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
        try:
            cand = improver.propose(task, current, cur_report)
        except BudgetExceeded as e:
            stopped = "budget"
            _note(history, f"stopped before round {r}: {e}")
            break
        except Exception as e:  # an improver failure costs a round, not the run
            rec = RoundRecord(
                round=r, source=current, report=cur_report, accepted=False,
                reason=f"improver error: {type(e).__name__}: {e}"[:2000],
            )
            finish(rec, u0, store=False)
            continue

        if cand.code == current.code:
            finish(RoundRecord(round=r, source=current, report=cur_report, accepted=False, reason="unchanged"), u0, store=False)
            continue

        cand = cand.model_copy(update={"version": fresh(cand.version), "parent_version": current.version})
        res = evaluate(cand)
        rec = RoundRecord(
            round=r, source=cand, report=res.report, accepted=False, reason="",
            sandbox_stage=res.stage, sandbox_error=res.error, eval_wall_s=res.wall_s,
        )
        if not res.ok:
            rec.reason = f"sandbox {res.stage}"
        elif not passes_gate(best.report, res.report):
            lo = best.report.ci95[0]
            rec.reason = f"regression: {res.report.success_rate:.0%} < best {best.source.version} CI lower bound {lo:.0%}"
        else:
            rec.accepted, rec.reason = True, "passes keep-the-best gate"
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


def _best(history: list[RoundRecord]) -> RoundRecord | None:
    acc = [h for h in history if h.accepted]
    return max(acc, key=lambda h: h.report.success_rate) if acc else None  # max keeps the first on ties


def _note(history: list[RoundRecord], msg: str) -> None:
    if history:
        last = history[-1]
        last.critic = (last.critic + "\n" if last.critic else "") + msg
