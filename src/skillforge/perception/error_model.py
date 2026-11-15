"""Measure an estimator against true sim poses while a skill runs, and turn the result into
the `obs_noise_std` range that best describes it.

The numbers come from sim: rendered images, the sim's own segmentation unless the estimator
segments by colour, and whatever depth noise the frame source was given. They are an estimate
of real-camera error only as far as those match a real camera.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict

from skillforge.contracts import Range, Skill, TaskSpec

from .env import PerceivedEnv
from .estimators import quat_yaw, wrap


class ObjectError(BaseModel):
    """Position error is |estimate given to the skill - true position|, in metres."""

    model_config = ConfigDict(extra="forbid")

    object: str
    n_steps: int
    seen_rate: float  # share of steps with a confident model fit (not coarse, carried or held)
    pos_p50: float
    pos_p95: float
    pos_max: float
    seen_pos_p50: float | None  # steps with a confident model fit
    seen_pos_p95: float | None
    unseen_pos_p50: float | None  # steps on a coarse, carried or held estimate
    unseen_pos_p95: float | None
    bias: tuple[float, float, float]  # mean signed error, world frame
    yaw_p50: float | None  # radians, modulo the shape's symmetry; boxes seen upright only
    yaw_p95: float | None
    equivalent_noise_std: float  # isotropic Gaussian std with the same RMS position error


class PerceptionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    skill: str
    skill_version: str
    n_episodes: int
    n_success: int
    objects: list[ObjectError]
    obs_noise_std: Range  # 5th..95th percentile of the per-episode equivalent std, all objects

    def summary(self) -> str:
        lines = [
            f"{self.task_id}: {self.n_episodes} episodes ({self.n_success} succeeded), "
            f"equivalent obs_noise_std {self.obs_noise_std.lo * 1e3:.1f}-{self.obs_noise_std.hi * 1e3:.1f} mm"
        ]
        mm = lambda v: "n/a" if v is None else f"{v * 1e3:.1f}"
        for o in self.objects:
            yaw = "n/a" if o.yaw_p50 is None else f"{np.degrees(o.yaw_p50):.1f}/{np.degrees(o.yaw_p95):.1f} deg"
            lines.append(
                f"  {o.object}: pos p50/p95/max {mm(o.pos_p50)}/{mm(o.pos_p95)}/{mm(o.pos_max)} mm; "
                f"fitted {o.seen_rate:.0%} of steps (p50/p95 {mm(o.seen_pos_p50)}/{mm(o.seen_pos_p95)} mm), "
                f"other steps p50/p95 {mm(o.unseen_pos_p50)}/{mm(o.unseen_pos_p95)} mm; yaw p50/p95 {yaw}"
            )
        return "\n".join(lines)


def _yaw_error(task: TaskSpec, name: str, est_quat: np.ndarray, true_quat: np.ndarray) -> float | None:
    o = task.scene.object(name)
    if o.shape != "box":
        return None
    w, x, y, z = true_quat
    if 1 - 2 * (x * x + y * y) < 0.99:  # the box's z axis is not vertical: yaw is not defined
        return None
    square = abs(o.size[0] - o.size[1]) < 0.1 * max(o.size[0], o.size[1])
    return abs(wrap(quat_yaw(est_quat) - quat_yaw(true_quat), np.pi / 2 if square else np.pi))


def measure(
    make_env: Callable[[TaskSpec], PerceivedEnv],
    skill: Skill,
    task: TaskSpec,
    seeds: Iterable[int],
) -> PerceptionReport:
    """Roll `skill` out in a PerceivedEnv and score every step's object estimates against truth."""
    env = make_env(task)
    if env.true_poses() is None:
        env.close()
        raise ValueError("measure() needs an inner env that exposes true poses (MujocoEnv)")
    names = list(env.true_poses())
    err: dict[str, list[np.ndarray]] = {n: [] for n in names}
    seen: dict[str, list[bool]] = {n: [] for n in names}
    yaw: dict[str, list[float]] = {n: [] for n in names}
    per_episode_std: list[float] = []
    n_ep = n_ok = 0

    def score(obs) -> list[np.ndarray]:
        truth = env.true_poses()
        out = []
        for n in names:
            e = obs.objects[n].pos - truth[n].pos
            err[n].append(e)
            est = env.estimates.get(n)
            s = bool(est is not None and est.found and est.confidence >= env.min_confidence)
            seen[n].append(s)
            if s:
                y = _yaw_error(task, n, obs.objects[n].quat, truth[n].quat)
                if y is not None:
                    yaw[n].append(y)
            out.append(e)
        return out

    try:
        for seed in seeds:
            obs = env.reset(seed)
            skill.reset(task)
            ep = score(obs)
            ok = False
            for _ in range(task.max_steps):
                obs, info = env.step(skill.act(obs))
                ep += score(obs)
                if info.success or info.terminated:
                    ok = info.success
                    break
            n_ep += 1
            n_ok += ok
            per_episode_std.append(float(np.sqrt(np.mean(np.square(ep)))))
    finally:
        env.close()

    pct = lambda a, q: float(np.percentile(a, q)) if len(a) else None
    objects = []
    for n in names:
        e = np.array(err[n])
        d = np.linalg.norm(e, axis=1)
        s = np.array(seen[n])
        objects.append(
            ObjectError(
                object=n,
                n_steps=len(d),
                seen_rate=float(s.mean()),
                pos_p50=pct(d, 50),
                pos_p95=pct(d, 95),
                pos_max=float(d.max()),
                seen_pos_p50=pct(d[s], 50),
                seen_pos_p95=pct(d[s], 95),
                unseen_pos_p50=pct(d[~s], 50),
                unseen_pos_p95=pct(d[~s], 95),
                bias=tuple(float(v) for v in e.mean(axis=0)),
                yaw_p50=pct(yaw[n], 50),
                yaw_p95=pct(yaw[n], 95),
                equivalent_noise_std=float(np.sqrt(np.mean(np.square(e)))),
            )
        )
    lo, hi = np.percentile(per_episode_std, [5, 95])
    return PerceptionReport(
        task_id=task.id,
        skill=skill.name,
        skill_version=skill.version,
        n_episodes=n_ep,
        n_success=n_ok,
        objects=objects,
        obs_noise_std=Range(lo=float(lo), hi=float(hi)),
    )
