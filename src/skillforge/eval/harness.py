"""T4: run a skill over many randomized seeds and summarize how it fails."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
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

# Called after reset (step 0) and after every env.step with (env, step_index). Used by
# video capture; it must not touch env state that affects the rollout.
StepHook = Callable[[SimEnv, int], None]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _first(events: list[str], pred: Callable[[str], bool]) -> str | None:
    return next((e for e in events if pred(e)), None)


def _classify(events: list[str], terminated: bool, timed_out: bool) -> tuple[FailureTag, str]:
    """Name the most likely root cause of a failed episode from its event stream.

    Precedence runs from cause to symptom: a controller that could not reach its target
    (ik_failure) explains a collision, a collision explains a slip, a slip explains a drop.
    Event names follow StepInfo's convention: 'ik_failure[:...]', 'collision[:a-b]',
    'grasp_slip[:obj]', 'grasp:<obj>', 'grasp_miss', 'release:<obj>'.
    A sim artifact ('sim_unstable', 'object_ejected[:obj]': the physics, not the skill, is at
    fault) outranks everything, so an improver is never told its skill caused it.
    Returns (tag, the event that decided it or '').
    """
    if e := _first(events, lambda e: e == "sim_unstable" or e == "object_ejected" or e.startswith("object_ejected:")):
        return "sim_artifact", e
    if e := _first(events, lambda e: e == "ik_failure" or e.startswith("ik_failure:")):
        return "ik_failure", e
    if e := _first(events, lambda e: e == "collision" or e.startswith("collision:")):
        return "collision", e
    if e := _first(events, lambda e: e == "grasp_slip" or e.startswith("grasp_slip:")):
        return "grasp_slip", e
    if terminated:
        return "object_dropped", ""
    grasped = any(e.startswith("grasp:") for e in events)
    if not grasped:
        if "grasp_miss" in events:
            return "grasp_miss", "grasp_miss"
        return ("timeout", "") if timed_out else ("other", "")
    last_grasp = max(i for i, e in enumerate(events) if e.startswith("grasp:"))
    released = any(e.startswith("release:") for e in events[last_grasp:])
    if released:  # let go somewhere that is not the goal
        return "wrong_placement", events[last_grasp]
    return ("timeout", "still holding") if timed_out else ("other", "")


def run_episode(
    env: SimEnv, skill: Skill, task: TaskSpec, seed: int, on_step: StepHook | None = None
) -> Episode:
    t0 = time.perf_counter()
    obs = env.reset(seed)
    skill.reset(task)
    if on_step is not None:
        on_step(env, 0)
    events: list[str] = []
    success = terminated = False
    n = 0
    failure: FailureTag | None = None
    detail = ""
    try:
        for n in range(1, task.max_steps + 1):  # noqa: B007 (n is read after the loop)
            obs, info = env.step(skill.act(obs))
            events += info.events
            if on_step is not None:
                on_step(env, n)
            if info.success:
                success = True
                break
            if info.terminated:
                terminated = True
                break
    except Exception as e:  # a crashing skill is a failure mode, not a harness crash
        failure, detail = "exception", f"{type(e).__name__}: {e}"
    if not success and failure is None:
        failure, why = _classify(events, terminated, timed_out=n >= task.max_steps)
        detail = f"after {n} steps; {f'cause: {why}; ' if why else ''}events: {events[-6:]}"
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


def summarize(
    task: TaskSpec, skill_name: str, skill_version: str, eps: Sequence[Episode], max_failures: int = 20
) -> EvalReport:
    """Build an EvalReport from episodes. Serial and parallel eval both go through here."""
    k = sum(e.success for e in eps)
    fails = [e for e in eps if not e.success]
    return EvalReport(
        task_id=task.id,
        skill=skill_name,
        skill_version=skill_version,
        n_rollouts=len(eps),
        n_success=k,
        success_rate=k / len(eps) if eps else 0.0,
        ci95=wilson(k, len(eps)),
        failure_counts=dict(Counter(e.failure for e in fails)),
        failures=[
            FailureCase(
                seed=e.seed, tag=e.failure, detail=e.failure_detail, randomization=e.randomization, video=e.video
            )
            for e in fails[:max_failures]
        ],
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
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
    return summarize(task, skill.name, skill.version, eps, max_failures), eps
