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


class BackgroundEstimator:
    needs_frames = False

    def __init__(self, estimator, frames, camera: str, hold_radius: float = 0.05):
        self.inner, self.frames, self.camera, self.hold_radius = estimator, frames, camera, hold_radius
        self.min_confidence = getattr(estimator, "min_confidence", 0.0)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest: dict[str, Estimate] | None = None
        self._ee_at_capture: np.ndarray | None = None
        self._ee_now: np.ndarray | None = None
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

    def _once(self) -> None:
        with self._lock:
            ee = None if self._ee_now is None else self._ee_now.copy()
        t0 = time.monotonic()
        frame = self.frames.capture(self.camera)
        if frame is None:
            return
        result = self.inner.estimate([frame])
        self.last_duration = time.monotonic() - t0
        with self._lock:
            self._latest, self._ee_at_capture = result, ee
        self.n_results += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._once()
            except Exception as e:  # keep the last good estimate; say why
                print(f"(background estimator: {type(e).__name__}: {e})")
                self._stop.wait(0.2)

    def close(self) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=5)
            self._thread = None
