"""T4: run a skill over many randomized seeds and summarize how it fails."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timezone

from skillforge.contracts import (
    Episode,
    EvalReport,
    FailureCase,
    FailureTag,
    SimEnv,
    Skill,
    TaskSpec,
)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _classify(events: list[str], terminated: bool, timed_out: bool) -> FailureTag:
    if terminated:
        return "object_dropped"
    if any(e.startswith("collision") for e in events):
        return "collision"
    if "grasp_miss" in events and not any(e.startswith("grasp:") for e in events):
        return "grasp_miss"
    if any(e.startswith("grasp:") for e in events):
        return "wrong_placement" if timed_out else "other"
    return "timeout" if timed_out else "other"


def run_episode(env: SimEnv, skill: Skill, task: TaskSpec, seed: int) -> Episode:
    t0 = time.perf_counter()
    obs = env.reset(seed)
    skill.reset(task)
    events: list[str] = []
    success = terminated = False
    n = 0
    failure: FailureTag | None = None
    detail = ""
    try:
        for n in range(1, task.max_steps + 1):  # noqa: B007 (n is read after the loop)
            obs, info = env.step(skill.act(obs))
            events += info.events
            if info.success:
                success = True
                break
            if info.terminated:
                terminated = True
                break
    except Exception as e:  # a crashing skill is a failure mode, not a harness crash
        failure, detail = "exception", f"{type(e).__name__}: {e}"
    if not success and failure is None:
        failure = _classify(events, terminated, timed_out=n >= task.max_steps)
        detail = f"after {n} steps; events: {events[-6:]}"
    return Episode(
        task_id=task.id,
        skill=skill.name,
        skill_version=skill.version,
        seed=seed,
        randomization=env.randomization_sample(),
        n_steps=n,
        success=success,
        failure=failure,
        failure_detail=detail,
        events=events,
        duration_s=time.perf_counter() - t0,
    )


def evaluate(
    make_env: Callable[[TaskSpec], SimEnv],
    skill: Skill,
    task: TaskSpec,
    seeds: Iterable[int],
    max_failures: int = 20,
) -> tuple[EvalReport, list[Episode]]:
    env = make_env(task)
    try:
        eps = [run_episode(env, skill, task, s) for s in seeds]
    finally:
        env.close()
    k = sum(e.success for e in eps)
    fails = [e for e in eps if not e.success]
    report = EvalReport(
        task_id=task.id,
        skill=skill.name,
        skill_version=skill.version,
        n_rollouts=len(eps),
        n_success=k,
        success_rate=k / len(eps) if eps else 0.0,
        ci95=wilson(k, len(eps)),
        failure_counts=dict(Counter(e.failure for e in fails)),
        failures=[
            FailureCase(seed=e.seed, tag=e.failure, detail=e.failure_detail, randomization=e.randomization)
            for e in fails[:max_failures]
        ],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return report, eps
