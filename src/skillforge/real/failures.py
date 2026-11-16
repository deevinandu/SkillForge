"""T7: close the loop. Every failed real trial becomes a sim test case.

`pin_failure` writes a task variant with the object at the pose it was MEASURED at in that
trial and no pose jitter. Friction, mass, gain and noise keep the task's ranges (the trial did
not measure them); latency takes the calibrated range when a `Calibration` is given.
`close_loop` does that for every failed trial of a session, saves each as a new file under a
tasks directory (never overwriting), and makes a regression baseline in T4's format by running
the skill on the pinned task in sim.

A pinned case the skill passes in sim is as useful as one it fails: it is a failure the twin
does not reproduce, which is a measured piece of the sim2real gap. `reproduced` says which.

The contract cannot pin a signed offset (a jitter range draws a sign), so the measured pose is
written as the object's nominal pose. See the request in HANDOVER.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skillforge.contracts import Range, SkillSource, TaskSpec
from skillforge.eval.regression import Baseline, make_baseline, save_baseline

from .core import _quat_mul
from .sysid import Calibration


def pin_failure(task: TaskSpec, trial: dict, calibration: Calibration | None = None) -> TaskSpec:
    """`trial` is one entry of a session's `real.json` `trials`."""
    d = task.model_dump(mode="json")
    measured = trial["measured"]
    for o in d["scene"]["objects"]:
        name = o["name"]
        if f"{name}.placed_dx" not in measured:
            continue  # fixed, or never seen: leave it at nominal
        o["pose"]["pos"][0] += measured[f"{name}.placed_dx"]
        o["pose"]["pos"][1] += measured[f"{name}.placed_dy"]
        yaw = measured.get(f"{name}.placed_dyaw", 0.0)
        qz = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        o["pose"]["quat"] = [float(v) for v in _quat_mul(qz, np.asarray(o["pose"]["quat"], dtype=float))]
    rnd = d["scene"]["randomization"]
    rnd["object_xy_jitter"] = Range.fixed(0.0).model_dump()
    rnd["object_yaw_jitter"] = Range.fixed(0.0).model_dump()
    if calibration is not None and calibration.latency_steps:
        rnd["action_latency_steps"] = {"lo": float(min(calibration.latency_steps)), "hi": float(max(calibration.latency_steps))}
    d["id"] = f"{task.id}_real_seed{trial['seed']}"
    return TaskSpec.model_validate(d)


def write_task(task: TaskSpec, directory: str | Path) -> Path:
    """Save as `<directory>/<task.id>.json`. Refuses to overwrite: add, don't edit."""
    p = Path(directory) / f"{task.id}.json"
    if p.exists():
        raise FileExistsError(f"{p} exists; pinned tasks are never overwritten")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(task.model_dump(mode="json"), indent=2) + "\n")
    return p


@dataclass
class PinnedCase:
    seed: int
    label_failure: str | None
    task: TaskSpec
    task_path: Path
    baseline: Baseline
    baseline_path: Path

    @property
    def reproduced(self) -> bool:
        """Does the skill also fail on the pinned task in sim, at least half the time?"""
        return self.baseline.success_rate < 0.5


def close_loop(
    session: str | Path,
    task: TaskSpec,
    src: SkillSource,
    make_env: Callable,
    tasks_dir: str | Path,
    baselines_dir: str | Path,
    *,
    seeds=range(20),
    calibration: Calibration | None = None,
    pool=None,
) -> list[PinnedCase]:
    """One pinned task and one baseline per failed trial. `seeds` vary what the trial did not
    measure (friction, mass, gain, noise, latency within its range)."""
    side = json.loads((Path(session) / "real.json").read_text())
    if side["task_id"] != task.id:
        raise ValueError(f"session is for task {side['task_id']}, not {task.id}")
    cases = []
    for trial in side["trials"]:
        if trial["label_success"]:
            continue
        pinned = pin_failure(task, trial, calibration)
        path = write_task(pinned, tasks_dir)
        base = make_baseline(make_env, src, pinned, seeds, pool=pool)
        cases.append(PinnedCase(trial["seed"], trial["label_failure"], pinned, path, base, save_baseline(base, Path(baselines_dir))))
    return cases
