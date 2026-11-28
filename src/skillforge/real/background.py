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


class TablePerception:
    """Bookkeeping for precision, the camera for change: what the lab uses for the can tasks.

    - At the start of a trial the table is measured (synchronously).
    - Bookkeeping then carries the scene precisely: a can in the hand (fingers closed to about a
      can's width) moves with the hand; a released can stays where the hand left it; the rest stay.
    - Whenever the camera's view is clear (fingers open, the tool point above `clear_z`), the table
      is measured again, in a background thread (`sync=False`) at most every `period` s, at once
      after a release or a slip. A can the camera finds more than `moved` from its record has
      really moved (it fell, or was knocked): its record jumps to the camera's. Smaller differences
      are camera noise next to a precise record and are ignored: a placement is judged against a
      +-8 mm spot, and the camera's error on a half-hidden stacked can reaches ~1 cm.
    - A slip: the fingers collapse (below `empty`) while holding a can. Its record is marked
      dropped and a measurement is due as soon as the view clears; then the skill's own recovery
      ("re-grasp from wherever the object ended up") has the real position.
    A measurement whose frame was captured before a grasp, release or slip is discarded.

    Why not live estimates: on the arm and twin (2026-11-27) the hand hides the can it holds; the
    estimator reported a lifted can as not rising and the skill let go mid-lift (trial 5).
    """

    needs_frames = False

    def __init__(self, estimator, frames, camera: str, *, hold_radius: float = 0.05, empty: float = 0.004,
                 slack: float = 0.005, clear_z: float = 0.20, moved: float = 0.025, period: float = 1.0,
                 sync: bool = False, remeasure: bool = True, clock=time.monotonic):
        self.inner, self.frames, self.camera = estimator, frames, camera
        self.hold_radius, self.empty, self.slack = hold_radius, empty, slack
        self.clear_z, self.moved, self.period, self.sync, self.remeasure = clear_z, moved, period, sync, remeasure
        self.clock = clock
        self.min_confidence = 0.0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.events: list[str] = []
        self._reset_state()

    def _reset_state(self) -> None:
        self.pos: dict[str, np.ndarray] = {}
        self.held: str | None = None
        self.dropped: set[str] = set()
        self.offset = np.zeros(3)
        self.version = 0  # bumps on every grasp, release and slip
        self.due = False
        self.last_measure = -1e9
        self.n_measures = 0
        self._pending: tuple[int, list] | None = None  # (version at capture, detections)
        self._candidate: dict[str, np.ndarray] = {}  # name -> a far detection seen once (confirm next time)

    def reset(self, task: TaskSpec) -> None:
        self.close()
        self.inner.reset(task)
        self._reset_state()

    # -- the camera ------------------------------------------------------------

    def _detect(self, obs):
        frame = self.frames.capture(self.camera)
        if frame is None:
            return None
        dets = self.inner.detect([frame])
        return self._supported(dets)

    def _supported(self, dets) -> list:
        """Detections that could be cans: on the table (layer 0), or resting on something one layer
        down (detected now, or on record). The arm in view makes floating 'cans' in upper layers
        (twin, 2026-11-27: a record jumped to one at 17 cm with nothing under it)."""
        r = float(getattr(self.inner, "radius", 0.0) or 0.0)
        H = 2 * float(getattr(self.inner, "half_height", 0.0) or 0.0)
        reach = 2 * r * 1.25  # a can on a 3-2-1 pyramid straddles two below it: centres ~1.15 diameters apart
        # support: what the camera sees now, and records of cans that are where we think they are
        # (not the one in the hand, not one that slipped: its record is where the hand was)
        below = [np.r_[d.xy, d.z] for d in dets] + [p for n, p in self.pos.items() if n not in self.dropped and n != self.held]
        out = []
        for d in dets:
            p = np.array([d.xy[0], d.xy[1], d.z])
            if d.layer == 0 or any(abs(q[2] - (p[2] - H)) < 0.4 * H and np.linalg.norm(q[:2] - p[:2]) < reach for q in below):
                out.append(p)
        return out

    def _initial(self, obs) -> None:
        frame = self.frames.capture(self.camera)
        if frame is None:
            return
        est = self.inner.estimate([frame], obs)  # names from the task's start poses
        self.pos = {n: e.pos.copy() for n, e in est.items() if e.found}
        self.last_measure = self.clock()
        self.n_measures += 1

    def _measure_async(self, version: int, obs) -> None:
        try:
            dets = self._detect(obs)
        except Exception as e:  # keep the record; say why
            print(f"(table perception: {type(e).__name__}: {e})")
            dets = None
        with self._lock:
            if dets is not None:
                self._pending = (version, dets)

    def _apply(self, dets: list) -> None:
        """Change detection: move a record only if the camera finds that can far from it."""
        names = [n for n in self.pos if n != self.held]
        free = list(range(len(dets)))
        pairs = sorted((float(np.linalg.norm(dets[k] - self.pos[n])), n, k) for n in names for k in free)
        matched: dict[str, int] = {}
        for d, n, k in pairs:  # 1) each record to the nearest detection within `moved`: it has not moved
            if n in matched or k not in free or d > self.moved:
                continue
            matched[n] = k
            free.remove(k)
        # 2) the rest to what is left, nearest first. A missing detection usually means hidden (by the
        # arm), not moved: only a can that slipped moves at once; any other needs the same new
        # detection in two measurements in a row (twin, 2026-11-27: a hidden can's record jumped to a
        # ghost for one measurement).
        moved_names = [n for n in names if n not in matched]
        pairs = sorted((float(np.linalg.norm(dets[k] - self.pos[n])), n, k) for n in moved_names for k in free)
        candidates: dict[str, np.ndarray] = {}
        for _d, n, k in pairs:
            if n in matched or k not in free:
                continue
            matched[n] = k
            free.remove(k)
            prev = self._candidate.get(n)
            if n in self.dropped or (prev is not None and np.linalg.norm(prev - dets[k]) < 0.02):
                self.pos[n] = dets[k].copy()
                self.dropped.discard(n)
                self.events.append(f"perception:moved:{n}")
            else:
                candidates[n] = dets[k].copy()
        self._candidate = candidates

    # -- PoseEstimator ---------------------------------------------------------------

    def _holding(self, obs) -> bool:
        d = 2 * float(getattr(self.inner, "radius", 0.0) or 0.0)
        return d > 0 and self.empty < float(obs.gripper_width) < d + self.slack

    def estimate(self, frames, obs: Observation | None = None) -> dict[str, Estimate]:
        if not self.pos:
            self._initial(obs)
        if obs is not None and self.pos:
            ee = np.asarray(obs.ee_pos, dtype=float)
            holding = self._holding(obs)
            if holding and self.held is None:  # just closed on something: the nearest can within reach
                n, p = min(self.pos.items(), key=lambda kv: float(np.linalg.norm(kv[1] - ee)))
                if np.linalg.norm(p - ee) < self.hold_radius:
                    self.held, self.offset = n, p - ee
                    self.version += 1
            elif not holding and self.held is not None:
                if float(obs.gripper_width) <= self.empty:  # collapsed: it slipped out
                    self.dropped.add(self.held)
                    self.events.append(f"perception:slip:{self.held}")
                self.held = None
                self.version += 1
                self.due = True
            if self.held is not None:
                self.pos[self.held] = ee + self.offset
            # apply a finished measurement, unless the scene changed since its frame
            with self._lock:
                pending, self._pending = self._pending, None
            if pending is not None and pending[0] == self.version and self.held is None:
                self._apply(pending[1])
            # start one when the view is clear
            clear = self.held is None and ee[2] > self.clear_z
            now = self.clock()
            if self.remeasure and clear and (self.due or now - self.last_measure > self.period):
                if self.sync:
                    dets = self._detect(obs)  # already filtered to supported detections
                    if dets is not None:
                        self._apply(dets)
                    self.last_measure, self.due = now, False
                    self.n_measures += 1
                elif self._thread is None or not self._thread.is_alive():
                    self._thread = threading.Thread(target=self._measure_async, args=(self.version, obs), daemon=True)
                    self._thread.start()
                    self.last_measure, self.due = now, False
                    self.n_measures += 1
        return {n: Estimate(p.copy(), yaw_quat(0.0), True, confidence=1.0, n_points=0) for n, p in self.pos.items()}

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


class MeasureOnce(TablePerception):
    """`TablePerception` that never re-measures (the table is measured at the start only)."""

    def __init__(self, estimator, frames, camera: str, **kw):
        super().__init__(estimator, frames, camera, remeasure=False, **kw)
