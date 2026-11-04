"""The forge loop: evaluate, hand the report to an improver, repeat until the bar is met.

This is the spine every track plugs into: sim (T1) runs it, skills (T2) seed it, agents (T3)
improve it, eval (T4) judges it. Phase 1 replaces each stub, not this loop's shape.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from skillforge.contracts import EvalReport, SimEnv, SkillImprover, SkillSource, TaskSpec


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
    """Every round evaluates on the SAME seeds, so a change in success rate is the skill, not luck.

    Skill code runs only in a subprocess sandbox (T3), never in this process, so `make_env` must
    be importable by reference (a module-level function or class). A candidate is kept only if
    it passes the keep-the-best gate; rejected candidates still appear in the history.
    For the library, history.json, critic and budget, use `skillforge.agents.run_loop`.
    """
    from skillforge.agents.loop import run_loop  # lazy: agents imports eval/sim, not forge

    result = run_loop(task, seed_skill, improver, make_env, rounds=rounds, rollouts=rollouts, target=target)
    return [ForgeRound(round=h.round, source=h.source, report=h.report) for h in result.rounds]
