"""`PerceivedEnv`: a SimEnv wrapper whose observations carry estimated object poses.

The wrapped env still owns physics and success. This wrapper only replaces
`Observation.objects` with what an estimator makes of the camera frames, so a skill is
evaluated on what it would actually see. `StepInfo.success` and `terminated` pass through
untouched: success is judged on true sim state, never on estimates.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from skillforge.contracts import Action, ObjectState, Observation, SimEnv, StepInfo, TaskSpec

from .estimators import DepthEstimator, Estimate, OracleEstimator, PoseEstimator
from .frames import CameraFrame, FrameSource, SimFrameSource, overview_camera


class PerceivedEnv:
    """Wrap `inner` so skills see estimated object poses.

    cameras: names passed to the frame source each capture; their clouds are fused.
    estimate_hz: run the estimator at this rate (None = every control step). Between captures
        the last estimate is held, as a slow real pipeline would.
    in_hand: while the gripper is closed on something next to the object's last estimate and the
        cameras cannot fit the object well (the fingers hide it), carry the estimate with the
        end effector. This uses only the observation (measured gripper width, ee pose). The
        carry assumes a rigid grasp, so an object sliding in the fingers is tracked with error.

    Each step an object's pose comes from, in order: a confident model fit; the in-hand carry; a
    coarse estimate if it is more than `jump` from the last pose (the object moved); else the last
    pose. An object with no pixels at all keeps its last pose (its nominal pose from the task
    before the first sighting) and the events gain `perception_miss:<obj>`, once per streak.
    """

    def __init__(
        self,
        inner: SimEnv,
        estimator: PoseEstimator,
        *,
        source: FrameSource | None = None,
        cameras: tuple[str, ...] = ("overview", "wrist"),
        estimate_hz: float | None = None,
        in_hand: bool = True,
        hold_radius: float = 0.05,  # metres: an estimate this close to the tcp can be "in hand"
        min_confidence: float | None = None,  # below this an estimate is coarse; default: the estimator's
        jump: float = 0.03,  # metres: a coarse estimate this far from the last one replaces it
    ):
        self.inner = inner
        self.task: TaskSpec = inner.task
        self.estimator = estimator
        self.cameras = tuple(cameras)
        self.in_hand = in_hand
        self.hold_radius = hold_radius
        self.min_confidence = getattr(estimator, "min_confidence", 0.0) if min_confidence is None else min_confidence
        self.jump = jump
        if source is None and estimator.needs_frames:
            bx, by = self.task.scene.robot.base_pose.pos[:2]
            source = SimFrameSource(inner, virtual=(overview_camera((bx, by)),))
        self.source = source
        hz = self.task.control_hz if estimate_hz is None else estimate_hz
        self._every = max(1, round(self.task.control_hz / hz))
        self._movable = [o for o in self.task.scene.objects if not o.fixed]
        self._last: dict[str, ObjectState] = {}
        self._estimates: dict[str, Estimate] = {}
        self._missing: set[str] = set()
        self._prev_ee: np.ndarray | None = None
        self._k = 0

    # -- contract ----------------------------------------------------------

    def reset(self, seed: int) -> Observation:
        obs = self.inner.reset(seed)
        for target in (self.source, self.estimator):
            if hasattr(target, "reseed"):
                target.reseed(seed)
        self.estimator.reset(self.task)
        self._last = {
            o.name: ObjectState(pos=np.array(o.pose.pos, dtype=float), quat=np.array(o.pose.quat, dtype=float))
            for o in self._movable
        }
        self._estimates = {}
        self._missing = set()
        self._prev_ee = None
        self._k = 0
        obs, _ = self._perceive(obs)
        return obs

    def step(self, action: Action) -> tuple[Observation, StepInfo]:
        obs, info = self.inner.step(action)
        obs, events = self._perceive(obs)
        return obs, StepInfo(success=info.success, terminated=info.terminated, events=info.events + events)

    def randomization_sample(self) -> dict[str, float]:
        return self.inner.randomization_sample()

    def render(self, camera: str | None = None) -> np.ndarray | None:
        return self.inner.render(camera)

    def close(self) -> None:
        if hasattr(self.source, "close"):
            self.source.close()
        self.inner.close()

    # -- perception --------------------------------------------------------

    @property
    def estimates(self) -> dict[str, Estimate]:
        """The estimator's raw output at the last capture (before hold / in-hand logic)."""
        return self._estimates

    def true_poses(self) -> dict[str, ObjectState] | None:
        """Ground truth for scoring an estimator. None if the inner env is not a MuJoCo env."""
        m, d = getattr(self.inner, "model", None), getattr(self.inner, "data", None)
        if m is None or d is None:
            return None
        out = {}
        for o in self._movable:
            b = m.body(o.name).id
            out[o.name] = ObjectState(pos=d.xpos[b].copy(), quat=d.xquat[b].copy())
        return out

    def _capture(self) -> list[CameraFrame]:
        if not self.estimator.needs_frames or self.source is None:
            return []
        frames = (self.source.capture(c) for c in self.cameras)
        return [f for f in frames if f is not None]

    def _perceive(self, obs: Observation) -> tuple[Observation, list[str]]:
        events: list[str] = []
        fresh = self._k % self._every == 0
        self._k += 1
        if fresh:
            self._estimates = self.estimator.estimate(self._capture(), obs)
        ee = np.asarray(obs.ee_pos, dtype=float)
        max_w = self.task.scene.robot.max_gripper_width
        gripping = 0.004 < obs.gripper_width < 0.92 * max_w
        objects = dict(obs.objects)
        for o in self._movable:
            last = self._last[o.name]
            est = self._estimates.get(o.name)
            seen = fresh and est is not None and est.found
            held = (
                self.in_hand
                and gripping
                and self._prev_ee is not None
                and np.linalg.norm(last.pos - self._prev_ee) < self.hold_radius
            )
            confident = seen and est.confidence >= self.min_confidence
            if confident:
                state = ObjectState(pos=est.pos.copy(), quat=est.quat.copy())
            elif held:  # the fingers are the usual reason a held object is half hidden
                state = ObjectState(pos=last.pos + (ee - self._prev_ee), quat=last.quat.copy())
            elif seen and np.linalg.norm(est.pos - last.pos) > self.jump:
                # a coarse estimate far from the last one: the object moved (fell over, rolled)
                state = ObjectState(pos=est.pos.copy(), quat=est.quat.copy())
            else:
                # not seen, or a coarse estimate that agrees with the last one: the object is
                # where it was and something (often the open fingers) is in the way
                state = last
                if fresh and not seen and o.name not in self._missing:  # once per streak
                    events.append(f"perception_miss:{o.name}")
            if seen:
                self._missing.discard(o.name)
            elif fresh:
                self._missing.add(o.name)
            self._last[o.name] = state
            objects[o.name] = ObjectState(pos=state.pos.copy(), quat=state.quat.copy())
        self._prev_ee = ee
        return replace(obs, objects=objects), events


# Module-level factories: picklable, so they work with `evaluate_parallel` and the sandbox.


def make_oracle_env(task: TaskSpec) -> PerceivedEnv:
    from skillforge.sim import make_env

    return PerceivedEnv(make_env(task), OracleEstimator())


def make_depth_env(task: TaskSpec) -> PerceivedEnv:
    """MuJoCo env seen through the overview and wrist depth cameras, sim masks for segmentation."""
    from skillforge.sim import make_env

    return PerceivedEnv(make_env(task), DepthEstimator())


NOISY_DEPTH_STD = 0.003  # metres at 1 m, growing with z^2: the order of a consumer stereo depth camera


def make_noisy_depth_env(task: TaskSpec) -> PerceivedEnv:
    """`make_depth_env` with per-pixel depth noise. The noise is independent per pixel, so it
    mostly averages out over an object's pixels; a real camera's correlated error (bias, warp)
    is not modelled."""
    from skillforge.sim import make_env

    inner = make_env(task)
    bx, by = task.scene.robot.base_pose.pos[:2]
    source = SimFrameSource(inner, virtual=(overview_camera((bx, by)),), depth_noise=NOISY_DEPTH_STD)
    return PerceivedEnv(inner, DepthEstimator(), source=source)
