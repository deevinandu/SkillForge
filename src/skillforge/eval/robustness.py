"""T4: robustness sweeps. How does success fall off as one randomization axis grows?

`sweep` varies one `RandomizationSpec` field at a time, pinning it to each value in turn
(`Range.fixed(v)`), and evaluates the skill on the same seeds at every point, so the curve
reflects the axis and not seed luck. `stress_spec` widens every range by a factor around
its nominal value (the "stress" preset, 2x by default).

The models here (`RobustnessReport`, `AxisCurve`, `CurvePoint`) are T4's, not contracts.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from skillforge.contracts import EvalReport, RandomizationSpec, Range, SkillSource, TaskSpec

from .parallel import EvalPool, MakeEnv

AXES: tuple[str, ...] = tuple(RandomizationSpec.model_fields)

# Value of each axis that means "no perturbation".
NOMINAL: dict[str, float] = {
    "object_xy_jitter": 0.0,
    "object_yaw_jitter": 0.0,
    "friction_scale": 1.0,
    "mass_scale": 1.0,
    "actuator_gain_scale": 1.0,
    "obs_noise_std": 0.0,
    "action_latency_steps": 0.0,
}
# Physical lower limits: jitters/noise/latency can't go negative, scales must stay positive.
FLOOR: dict[str, float] = {
    "object_xy_jitter": 0.0,
    "object_yaw_jitter": 0.0,
    "friction_scale": 0.05,
    "mass_scale": 0.05,
    "actuator_gain_scale": 0.05,
    "obs_noise_std": 0.0,
    "action_latency_steps": 0.0,
}
# Default sweep grids (6 points each). Units: metres, radians, unitless scales, steps.
DEFAULT_GRID: dict[str, tuple[float, ...]] = {
    "object_xy_jitter": tuple(np.linspace(0.0, 0.10, 6).round(6)),
    "object_yaw_jitter": tuple(np.linspace(0.0, np.pi / 2, 6).round(6)),
    "friction_scale": (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    "mass_scale": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0),
    "actuator_gain_scale": (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    "obs_noise_std": tuple(np.linspace(0.0, 0.01, 6).round(6)),  # 0 -> 1 cm
    "action_latency_steps": (0.0, 1.0, 2.0, 3.0, 4.0, 6.0),
}

Others = Literal["task", "nominal"]


class CurvePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float
    n: int
    n_success: int
    success_rate: float
    ci95: tuple[float, float]
    failure_counts: dict[str, int]


class AxisCurve(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis: str
    others: Others  # "task": other axes keep the task's ranges; "nominal": other axes fixed at nominal
    points: list[CurvePoint]

    def breaking_point(self, threshold: float = 0.8) -> float | None:
        """Smallest perturbation (distance from nominal) whose success rate drops below threshold."""
        nom = NOMINAL.get(self.axis, 0.0)
        bad = [p.value for p in self.points if p.success_rate < threshold]
        return min(bad, key=lambda v: abs(v - nom)) if bad else None


class RobustnessReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    skill: str
    skill_version: str
    seeds: list[int]
    curves: list[AxisCurve]
    stress: EvalReport | None = None
    stress_factor: float | None = None
    created_at: str

    def curve(self, axis: str) -> AxisCurve:
        for c in self.curves:
            if c.axis == axis:
                return c
        raise KeyError(axis)

    def summary(self, threshold: float = 0.8) -> str:
        lines = [f"{self.skill}@{self.skill_version} on {self.task_id}, {len(self.seeds)} seeds/point"]
        for c in self.curves:
            pts = ", ".join(f"{p.value:g}:{p.success_rate:.0%}" for p in c.points)
            bp = c.breaking_point(threshold)
            lines.append(f"  {c.axis}: {pts}" + (f"  (< {threshold:.0%} at {bp:g})" if bp is not None else ""))
        if self.stress is not None:
            lines.append(f"  stress x{self.stress_factor:g}: {self.stress.summary()}")
        return "\n".join(lines)


# ------------------------------------------------------------------ spec helpers


def with_randomization(task: TaskSpec, rnd: RandomizationSpec) -> TaskSpec:
    """A copy of `task` under different randomization, re-validated."""
    d = task.model_dump(mode="json")
    d["scene"]["randomization"] = rnd.model_dump(mode="json")
    return TaskSpec.model_validate(d)


def pin_axis(task: TaskSpec, axis: str, value: float, others: Others = "task") -> TaskSpec:
    if axis not in AXES:
        raise KeyError(f"unknown randomization axis '{axis}'; have {list(AXES)}")
    base = task.scene.randomization if others == "task" else RandomizationSpec()
    rnd = base.model_copy(update={axis: Range.fixed(float(value))})
    return with_randomization(task, RandomizationSpec.model_validate(rnd.model_dump()))


def stress_spec(rnd: RandomizationSpec, factor: float = 2.0) -> RandomizationSpec:
    """Widen every range `factor`x away from its nominal value, clamped to physical limits.

    e.g. xy jitter 0..5 cm -> 0..10 cm; friction 0.6..1.2 -> 0.2..1.4; noise 1..4 mm -> 1..8 mm.

    A range that is fixed at nominal (no randomization) stays fixed: there is nothing to widen.
    """
    if factor < 1.0:
        raise ValueError("stress factor must be >= 1")
    out = {}
    for axis in AXES:
        r: Range = getattr(rnd, axis)
        nom = NOMINAL[axis]
        # push each end that lies away from nominal `factor`x further out; an end on the nominal
        # side stays put, so the stressed range always contains the original one
        lo = nom - factor * (nom - r.lo) if r.lo < nom else r.lo
        hi = nom + factor * (r.hi - nom) if r.hi > nom else r.hi
        lo = max(lo, FLOOR[axis])
        out[axis] = Range(lo=lo, hi=max(hi, lo))
    return RandomizationSpec(**out)


# ------------------------------------------------------------------ sweeps


def _point(report: EvalReport, value: float) -> CurvePoint:
    return CurvePoint(
        value=float(value),
        n=report.n_rollouts,
        n_success=report.n_success,
        success_rate=report.success_rate,
        ci95=report.ci95,
        failure_counts=report.failure_counts,
    )


def sweep_axis(
    pool: EvalPool,
    make_env: MakeEnv,
    src: SkillSource,
    task: TaskSpec,
    axis: str,
    values: Iterable[float],
    seeds: Sequence[int],
    others: Others = "task",
) -> AxisCurve:
    pts = []
    for v in values:
        report, _ = pool.evaluate(make_env, src, pin_axis(task, axis, v, others), seeds)
        pts.append(_point(report, v))
    return AxisCurve(axis=axis, others=others, points=pts)


def sweep(
    make_env: MakeEnv,
    src: SkillSource,
    task: TaskSpec,
    axes: dict[str, Sequence[float]] | Sequence[str] | None = None,
    seeds: Iterable[int] = range(50),
    workers: int | None = None,
    others: Others = "task",
    stress_factor: float | None = 2.0,
    pool: EvalPool | None = None,
) -> RobustnessReport:
    """Sweep each axis in `axes` (names use DEFAULT_GRID; a dict gives explicit values).

    `stress_factor=None` skips the stress eval. Pass `pool` to reuse an open EvalPool.
    """
    seeds = [int(s) for s in seeds]
    if axes is None:
        grid = dict(DEFAULT_GRID)
    elif isinstance(axes, dict):
        grid = {a: tuple(v) for a, v in axes.items()}
    else:
        grid = {a: DEFAULT_GRID[a] for a in axes}
    own = pool is None
    pool = pool or EvalPool(workers)
    if own:
        pool.__enter__()
    try:
        curves = [sweep_axis(pool, make_env, src, task, a, vals, seeds, others) for a, vals in grid.items()]
        stress = None
        if stress_factor is not None:
            stressed = with_randomization(task, stress_spec(task.scene.randomization, stress_factor))
            stress, _ = pool.evaluate(make_env, src, stressed, seeds)
    finally:
        if own:
            pool.close()
    return RobustnessReport(
        task_id=task.id,
        skill=src.name,
        skill_version=src.version,
        seeds=seeds,
        curves=curves,
        stress=stress,
        stress_factor=stress_factor,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
