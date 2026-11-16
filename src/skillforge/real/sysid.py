"""T7: system identification from session logs. Fits what the sim can randomize.

From the commanded and measured joint trajectories that `runner` writes per trial
(`joint.start`, `joint.command`, `joint.measured`, `gripper.command`, `gripper.width`):

- action latency, in control steps, and a tracking gain, fitted together. The model is first
  order: each step the joints close a fraction `g` of the distance to the command sent `L` steps
  ago, `q[t] = q[t-1] + g * (cmd[t-L] - q[t-1])`. `L` is the integer that leaves the smallest
  residual; `g` is the least-squares fit at that `L`.
- gripper close time and the settled width while closed (the measured width of what it holds).

`g` is not the sim's `actuator_gain_scale`. `gain_curve` runs the same fit on a simulated twin
at known scales, and `equivalent_gain_scale` reads the scale that gives the fitted `g` off it.
Measured on the Franka twin this does NOT work for the gain: the curve is flat (see
`gain_is_identifiable`), so the gain scale is left alone. Latency is recovered exactly.

`propose_randomization` turns a calibration into a `RandomizationSpec` for a NEW matched task
file: centres from the fit, ranges from its spread. It never edits an existing task.

Validated on synthetic logs and on dry runs against the twin's known latency. Not yet run on
logs from a real arm, where the first-order model is an approximation to be checked.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from skillforge.contracts import RandomizationSpec, Range, TaskSpec

MIN_MOTION = 1e-3  # rad (or m for the stub): an episode whose commands move less than this says nothing


class TrackingFit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    latency_steps: int
    gain: float  # fraction of the remaining distance closed per control step, at that latency
    residual: float  # RMS of the model's one-step prediction error, in joint units
    n: int  # samples used


def fit_tracking(start: np.ndarray, command: np.ndarray, measured: np.ndarray, max_latency: int = 10) -> TrackingFit | None:
    """Latency and gain for one trial. None if the arm barely moved."""
    command, measured = np.asarray(command, dtype=float), np.asarray(measured, dtype=float)
    T = len(command)
    if T < max_latency + 5 or np.ptp(command, axis=0).max() < MIN_MOTION:
        return None
    prev = np.vstack([np.asarray(start, dtype=float)[None], measured[:-1]])  # q[t-1]
    dq = measured - prev
    best = None
    for lag in range(max_latency + 1):
        # before the first delayed command arrives the arm holds its start: nothing to fit there
        err = command[: T - lag] - prev[lag:]
        y = dq[lag:]
        denom = float(np.sum(err * err))
        if denom < 1e-12:
            continue
        g = float(np.sum(err * y) / denom)
        res = float(np.sqrt(np.mean((y - g * err) ** 2)))
        if best is None or res < best.residual:
            best = TrackingFit(latency_steps=lag, gain=g, residual=res, n=int(err.size))
    return best


class GripperFit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    close_time_s: float  # from the close command to the fingers settling
    closed_width: float  # metres, settled opening while closed: the object's width if it holds one


def fit_gripper(command: np.ndarray, width: np.ndarray, dt: float, settle: float = 2e-4, hold: int = 3) -> GripperFit | None:
    """The first close in the trial. None if the gripper never closed or never settled."""
    command, width = np.asarray(command, dtype=float), np.asarray(width, dtype=float)
    closed = command > 0.5
    starts = np.flatnonzero(closed & ~np.concatenate([[False], closed[:-1]]))
    if len(starts) == 0:
        return None
    t0 = int(starts[0])
    opens = np.flatnonzero(~closed[t0:])
    end = t0 + int(opens[0]) if len(opens) else len(command)
    still = np.abs(np.diff(width[t0:end])) < settle
    for i in range(1, len(still) - hold + 2):  # skip step 0: the fingers may not have started moving
        if still[i : i + hold].all() and width[t0 + i] < width[t0] - settle:
            return GripperFit(close_time_s=i * dt, closed_width=float(np.median(width[t0 + i : end])))
    return None


class Calibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # the session directory
    mode: str  # "dry_run" or "hardware", from the session
    task_id: str
    n_trials: int
    latency_steps: list[int]  # one per trial that could be fitted
    gain: list[float]
    residual: list[float]
    close_time_s: list[float]
    closed_width: list[float]
    created_at: str

    def summary(self) -> str:
        def ms(v):
            return f"{np.mean(v):.3g} +- {np.std(v):.2g} (n={len(v)})" if len(v) else "no data"

        return (f"{self.task_id} [{self.mode}], {self.n_trials} trials\n"
                f"  latency steps: {ms(self.latency_steps)}\n  gain per step: {ms(self.gain)}\n"
                f"  gripper close time s: {ms(self.close_time_s)}\n  closed width m: {ms(self.closed_width)}")


def fit_session(session: str | Path, dt: float | None = None, max_latency: int = 10) -> Calibration:
    """Fit every trial of a session written by `runner.run_session`."""
    session = Path(session)
    side = json.loads((session / "real.json").read_text())
    lat, gain, res, close, width = [], [], [], [], []
    for trial in side["trials"]:
        if not trial["arrays"]:
            continue
        with np.load(session / trial["arrays"]) as z:
            step = dt if dt is not None else float(np.median(np.diff(z["timestamp"]))) if len(z["timestamp"]) > 1 else 0.05
            fit = fit_tracking(z["joint.start"], z["joint.command"], z["joint.measured"], max_latency)
            grip = fit_gripper(z["gripper.command"], z["gripper.width"], step)
        if fit is not None:
            lat.append(fit.latency_steps)
            gain.append(fit.gain)
            res.append(fit.residual)
        if grip is not None:
            close.append(grip.close_time_s)
            width.append(grip.closed_width)
    return Calibration(
        source=str(session), mode=side["mode"], task_id=side["task_id"], n_trials=len(side["trials"]),
        latency_steps=lat, gain=gain, residual=res, close_time_s=close, closed_width=width,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------- gain: fit -> sim scale


def gain_curve(task: TaskSpec, scales: Sequence[float] = (0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0), seeds: Sequence[int] = (0, 1, 2),
               skill=None) -> list[tuple[float, float]]:
    """(actuator_gain_scale, fitted gain per step) on a simulated twin with no latency. Needs MuJoCo."""
    from skillforge.skills import PickPlace

    from .fakes import make_twin_env

    out = []
    for k in scales:
        rnd = task.scene.randomization.model_copy(update={
            "actuator_gain_scale": Range.fixed(float(k)), "action_latency_steps": Range.fixed(0.0)})
        pinned = task.model_copy(update={"scene": task.scene.model_copy(update={"randomization": rnd})})
        env = make_twin_env(pinned)
        fits = []
        for seed in seeds:
            sk = skill or PickPlace()
            obs = env.reset(seed)
            sk.reset(pinned)
            start, cmd, meas = obs.qpos.copy(), [], []
            for _ in range(pinned.max_steps):
                obs, info = env.step(sk.act(obs))
                cmd.append(env.command[0])
                meas.append(obs.qpos.copy())
                if info.success or info.terminated:
                    break
            fit = fit_tracking(start, np.array(cmd), np.array(meas), max_latency=0)
            if fit is not None:
                fits.append(fit.gain)
        env.close()
        out.append((float(k), float(np.mean(fits))))
    return out


def equivalent_gain_scale(gain: float, curve: Sequence[tuple[float, float]]) -> float:
    """The actuator_gain_scale whose simulated arm tracks with this gain. Clamped to the curve's ends."""
    pts = sorted(curve, key=lambda p: p[1])
    return float(np.interp(gain, [g for _, g in pts], [k for k, _ in pts]))


def gain_is_identifiable(curve: Sequence[tuple[float, float]], min_span: float = 0.05) -> bool:
    """Does the fitted gain change enough, and in one direction, across the curve's scales?

    Measured on the Franka twin it does not: the env scales kp and kv together, so the tracking
    lag stays the same (fitted gain 0.397-0.407 for scales 0.4-2.0). Pick-and-place logs cannot
    tell those arms apart. A hold test does not either (the model is gravity compensated: zero
    droop at every scale). The scale only shows under load, so it would take a loaded test.
    """
    g = np.array([g for _, g in sorted(curve)])
    d = np.diff(g)
    return bool(np.ptp(g) >= min_span and (np.all(d > 0) or np.all(d < 0)))


def propose_randomization(task: TaskSpec, cal: Calibration, curve: Sequence[tuple[float, float]] | None = None,
                          spread: float = 2.0) -> RandomizationSpec:
    """The task's randomization with latency and gain re-centred on the fit.

    Latency: the range of fitted values. Gain (needs a `curve` that passes
    `gain_is_identifiable`, else the task's range is kept): mean +- `spread` standard
    deviations of the equivalent scale. Every other axis keeps the task's range: these logs
    say nothing about friction, mass or observation noise.
    """
    update = {}
    if cal.latency_steps:
        update["action_latency_steps"] = Range(lo=float(min(cal.latency_steps)), hi=float(max(cal.latency_steps)))
    if cal.gain and curve and gain_is_identifiable(curve):
        k = np.array([equivalent_gain_scale(g, curve) for g in cal.gain])
        update["actuator_gain_scale"] = Range(lo=float(max(k.mean() - spread * k.std(), 0.05)), hi=float(k.mean() + spread * k.std()))
    return RandomizationSpec.model_validate(task.scene.randomization.model_copy(update=update).model_dump())
