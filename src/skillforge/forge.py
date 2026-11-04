"""The forge loop: evaluate, hand the report to an improver, repeat until the bar is met.

This is the spine every track plugs into: sim (T1) runs it, skills (T2) seed it, agents (T3)
improve it, eval (T4) judges it. Phase 1 replaces each stub, not this loop's shape.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from skillforge.contracts import EvalReport, SimEnv, SkillImprover, SkillSource, TaskSpec
from skillforge.eval import evaluate
from skillforge.skills import load_skill


class ForgeRound(BaseModel):
    round: int
    source: SkillSource
    report: EvalReport


def forge(
    task: TaskSpec,
    seed_skill: SkillSource,
    improver: SkillImprover,
    make_env: Callable[[TaskSpec], SimEnv],
    rounds: int = 5,
    rollouts: int = 50,
    target: float = 0.9,
) -> list[ForgeRound]:
    """Every round evaluates on the SAME seeds, so a change in success rate is the skill, not luck."""
    history: list[ForgeRound] = []
    src = seed_skill
    seeds = range(rollouts)
    for r in range(rounds):
        report, _ = evaluate(make_env, load_skill(src), task, seeds)
        history.append(ForgeRound(round=r, source=src, report=report))
        if report.success_rate >= target:
            break
        src = improver.propose(task, src, report)
    return history
