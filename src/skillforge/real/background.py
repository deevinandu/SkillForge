"""T7: run a slow pose estimator in a background thread, so a control step never waits for it.

On the arm (2026-11-27, third pyramid trial) the can estimator took ~270 ms inside every 50 ms
control step. Steps ran at ~4 Hz, RealEnv's watchdog fired on each, and the commanded tool point
went back and forth by ~6 mm a step: the arm jerked. `BackgroundEstimator` wraps the estimator:

- `needs_frames` is False, so `PerceivedEnv` does not capture in the step; this thread captures;
- after `reset(task)`, the FIRST `estimate()` runs synchronously (a trial starts from a real,
  current measurement); the second starts the thread, and from then on calls return the newest
  finished result at once;
- an estimate is one capture-plus-estimate old when it is handed over. For an object that was
  in the hand when its frame was captured (within `hold_radius` of the tool point then), the
  estimate is shifted by how far the hand has moved since, so a carried object does not appear
  to lag behind the fingers. Objects on the table are not shifted.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from skillforge.contracts import Observation, TaskSpec
from skillforge.perception import Estimate
from skillforge.perception.estimators import yaw_quat


class BackgroundEstimator:
    needs_frames = False

    def __init__(self, estimator, frames, camera: str, hold_radius: float = 0.05, decimate: int = 1):
        """decimate: in the background thread, use every `decimate`-th pixel (2: a quarter of the
        points, ~3x faster). The first, synchronous estimate after a reset always uses full frames."""
        self.inner, self.frames, self.camera, self.hold_radius = estimator, frames, camera, hold_radius
        self.decimate = max(1, int(decimate))
        self.min_confidence = getattr(estimator, "min_confidence", 0.0)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest: dict[str, Estimate] | None = None
        self._ee_at_capture: np.ndarray | None = None
        self._ee_now: np.ndarray | None = None
        self._obs_now: Observation | None = None
        self.n_results = 0
        self.last_duration = 0.0
        self._calls = 0

    # -- PoseEstimator -------------------------------------------------------

    def reset(self, task: TaskSpec) -> None:
        self.close()
        self.inner.reset(task)
        with self._lock:
            self._latest, self._ee_at_capture = None, None
        self.n_results = 0
        self._calls = 0

    def estimate(self, frames, obs: Observation | None = None) -> dict[str, Estimate]:
        if obs is not None:
            with self._lock:
                self._ee_now = np.asarray(obs.ee_pos, dtype=float).copy()
                self._obs_now = obs
        self._calls += 1
        if self._calls == 1:
            self._once()  # first call after reset: a direct, current measurement
        elif self._thread is None:
            # From the second call on, a trial is running: estimate in the background. (A placement
            # check resets before each single measurement, so it never starts the thread and never
            # competes for the camera with the placement preview.)
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        with self._lock:
            latest, ee0, ee1 = self._latest, self._ee_at_capture, self._ee_now
        if latest is None:
            return {}
        if ee0 is None or ee1 is None:
            return latest
        moved = ee1 - ee0
        out = {}
        for n, e in latest.items():
            if e.found and np.linalg.norm(e.pos - ee0) < self.hold_radius:
                out[n] = Estimate(e.pos + moved, e.quat, e.found, e.confidence, e.n_points,
                                  None if e.xy is None else e.xy + moved[:2], e.xy_confidence)
            else:
                out[n] = e
        return out

    # -- the thread --------------------------------------------------------------

    def _once(self, decimate: int = 1) -> None:
        with self._lock:
            ee = None if self._ee_now is None else self._ee_now.copy()
            obs = self._obs_now  # the robot as of this frame: the estimator needs the hand and gripper
        t0 = time.monotonic()
        frame = self.frames.capture(self.camera)
        if frame is None:
            return
        if decimate > 1:
            frame = _decimated(frame, decimate)
        result = self.inner.estimate([frame], obs)
        self.last_duration = time.monotonic() - t0
        with self._lock:
            self._latest, self._ee_at_capture = result, ee
        self.n_results += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._once(self.decimate)
            except Exception as e:  # keep the last good estimate; say why
                print(f"(background estimator: {type(e).__name__}: {e})")
                self._stop.wait(0.2)

    def close(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None


def _decimated(frame, k: int):
    """Every k-th pixel of a CameraFrame, with intrinsics to match (pixel centres preserved)."""
    from skillforge.perception import CameraFrame

    K = np.array(frame.K, dtype=float, copy=True)
    K[0, 0] /= k
    K[1, 1] /= k
    K[0, 2] = (K[0, 2] + 0.5) / k - 0.5
    K[1, 2] = (K[1, 2] + 0.5) / k - 0.5
    return CameraFrame(rgb=frame.rgb[::k, ::k], depth=np.ascontiguousarray(frame.depth[::k, ::k]), K=K,
                       T_world_cam=frame.T_world_cam, t=frame.t, name=frame.name)


class MeasureOnce:
    """Measure the scene once per trial (the first `estimate()` after `reset`), then carry it by
    bookkeeping: a can in the hand moves with the hand, a released can stays where the hand left
    it, the rest stay put. For a task where only the robot moves objects this is what a perfect
    observer would report, and what the skill was evaluated against in sim (true poses).

    Why not live estimates: on the arm and twin (2026-11-27) the hand hides the held can and its
    neighbours; the estimator reported a lifted can still near the table, ghosts took names, and
    the skill let go mid-lift and mid-carry. Cost: the skill cannot see a real slip or a knocked
    object mid-trial; the operator's label records it.

    "In the hand" is decided here from the finger opening, for the estimator's known can size:
    closed to about a can's width (between `empty` and the diameter plus `slack`). PerceivedEnv's
    generic rule (anything under 92% of the maximum opening) kept a released can attached to the
    hand after the skill's partial release (6.3 cm in sim), and on the arm the full opening
    (7.5 cm) clears that rule's 7.36 cm by only 1.4 mm. All objects are reported every step,
    confident, so PerceivedEnv passes them through.
    """

    needs_frames = False

    def __init__(self, estimator, frames, camera: str, hold_radius: float = 0.05, empty: float = 0.004,
                 slack: float = 0.005):
        self.inner, self.frames, self.camera = estimator, frames, camera
        self.hold_radius, self.empty, self.slack = hold_radius, empty, slack
        self.min_confidence = 0.0
        self.pos: dict[str, np.ndarray] = {}
        self.held: str | None = None
        self.offset = np.zeros(3)
        self.measured: dict[str, Estimate] = {}

    def reset(self, task: TaskSpec) -> None:
        self.inner.reset(task)
        self.pos, self.held, self.measured = {}, None, {}

    def _holding(self, obs: Observation) -> bool:
        d = 2 * float(getattr(self.inner, "radius", 0.0) or 0.0)
        return d > 0 and self.empty < float(obs.gripper_width) < d + self.slack

    def estimate(self, frames, obs: Observation | None = None) -> dict[str, Estimate]:
        if not self.measured:
            frame = self.frames.capture(self.camera)
            self.measured = {} if frame is None else self.inner.estimate([frame], obs)
            self.pos = {n: e.pos.copy() for n, e in self.measured.items() if e.found}
        if obs is not None and self.pos:
            ee = np.asarray(obs.ee_pos, dtype=float)
            if self._holding(obs):
                if self.held is None:  # just closed on something: the nearest can within reach
                    n, p = min(self.pos.items(), key=lambda kv: float(np.linalg.norm(kv[1] - ee)))
                    if np.linalg.norm(p - ee) < self.hold_radius:
                        self.held, self.offset = n, p - ee
                if self.held is not None:
                    self.pos[self.held] = ee + self.offset
            else:
                self.held = None  # released (or closed on nothing): it stays where the hand left it
        return {n: Estimate(p.copy(), yaw_quat(0.0), True, confidence=1.0, n_points=0) for n, p in self.pos.items()}

    def close(self) -> None:
        pass
