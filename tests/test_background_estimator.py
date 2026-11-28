"""T7: BackgroundEstimator (the can estimator off the control loop's critical path)."""

import time

import numpy as np

from skillforge.contracts import Observation
from skillforge.perception import Estimate
from skillforge.perception.estimators import yaw_quat
from skillforge.real.background import BackgroundEstimator


class SlowEstimator:
    """Takes `delay` s; reports a table object, and one object wherever `held_at()` says."""

    needs_frames = True

    def __init__(self, delay, held_at):
        self.delay, self.held_at, self.calls, self.resets = delay, held_at, 0, 0

    def reset(self, task):
        self.resets += 1

    def estimate(self, frames, obs=None):
        time.sleep(self.delay)
        self.calls += 1
        return {"table": Estimate(np.array([0.5, 0.0, 0.04]), yaw_quat(0.0), True, confidence=0.9),
                "held": Estimate(np.array(self.held_at()), yaw_quat(0.0), True, confidence=0.9)}


class Frames:
    def __init__(self):
        self.n = 0

    def capture(self, name):
        self.n += 1
        return object()


def obs_at(ee):
    return Observation(t=0.0, qpos=np.zeros(7), qvel=np.zeros(7), ee_pos=np.array(ee, float), ee_quat=np.array([0, 1, 0, 0.0]),
                       gripper_width=0.05)


def test_steps_do_not_wait_after_the_first_call():
    inner = SlowEstimator(0.2, lambda: [0.4, 0.0, 0.2])
    b = BackgroundEstimator(inner, Frames(), "fixed")
    b.reset(None)
    t0 = time.monotonic()
    b.estimate([], obs_at([0.4, 0.0, 0.2]))  # first: synchronous
    assert time.monotonic() - t0 >= 0.2
    worst = 0.0
    for _ in range(20):
        t0 = time.monotonic()
        out = b.estimate([], obs_at([0.4, 0.0, 0.2]))
        worst = max(worst, time.monotonic() - t0)
        time.sleep(0.05)
    assert worst < 0.02 and out["table"].found and b.n_results >= 3
    b.close()


def test_placement_checks_never_start_the_thread():
    """Reset then one call, repeatedly (the free-placement check): only direct measurements."""
    frames = Frames()
    b = BackgroundEstimator(SlowEstimator(0.01, lambda: [0.4, 0.0, 0.2]), frames, "fixed")
    for _ in range(3):
        b.reset(None)
        b.estimate([], None)
    time.sleep(0.1)
    assert b._thread is None and frames.n == 3


def test_a_held_object_follows_the_hand_between_estimates():
    ee = {"p": np.array([0.4, 0.0, 0.2])}
    b = BackgroundEstimator(SlowEstimator(0.3, lambda: ee["p"].copy()), Frames(), "fixed")
    b.reset(None)
    b.estimate([], obs_at(ee["p"]))
    b.estimate([], obs_at(ee["p"]))  # starts the thread
    time.sleep(0.05)
    ee["p"] = ee["p"] + [0.0, 0.0, 0.05]  # the hand lifted 5 cm since the last capture
    out = b.estimate([], obs_at(ee["p"]))
    assert np.linalg.norm(out["held"].pos - ee["p"]) < 0.01  # shifted with the hand
    assert np.allclose(out["table"].pos, [0.5, 0.0, 0.04])  # the table object is not
    b.close()


def test_decimated_frames_keep_geometry():
    """Every 2nd pixel with matching intrinsics back-projects to the same world points."""
    from skillforge.perception import CameraFrame, backproject
    from skillforge.real.background import _decimated

    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]
    T[:3, 3] = [0.5, 0.0, 1.0]
    f = CameraFrame(rgb=np.zeros((720, 1280, 3), np.uint8), depth=np.full((720, 1280), 1.0, np.float32), K=K, T_world_cam=T)
    full = backproject(f)
    half = backproject(_decimated(f, 2))
    assert len(half) == len(full) // 4
    np.testing.assert_allclose(half.mean(axis=0), full.mean(axis=0), atol=1e-3)
    np.testing.assert_allclose(half.min(axis=0), full.min(axis=0), atol=2e-3)


# ---------------------------------------------------------------- MeasureOnce

class Cans:
    """Two cans of radius 2.54 cm on the table; counts calls."""

    needs_frames = True
    radius = 0.0254

    def __init__(self):
        self.calls = 0

    def reset(self, task):
        pass

    def estimate(self, frames, obs=None):
        self.calls += 1
        return {"a": Estimate(np.array([0.40, 0.00, 0.042]), yaw_quat(0.0), True, confidence=0.8),
                "b": Estimate(np.array([0.50, 0.00, 0.042]), yaw_quat(0.0), True, confidence=0.8)}


def obs_w(ee, width):
    o = obs_at(ee)
    o.gripper_width = width
    return o


def test_measure_once_measures_once_per_trial():
    from skillforge.real.background import MeasureOnce

    inner, frames = Cans(), Frames()
    m = MeasureOnce(inner, frames, "fixed")
    m.reset(None)
    for _ in range(5):
        out = m.estimate([], obs_w([0.4, 0.0, 0.3], 0.08))
    assert inner.calls == 1 and frames.n == 1 and out["a"].found and out["a"].confidence == 1.0
    m.reset(None)
    m.estimate([], obs_w([0.4, 0.0, 0.3], 0.08))
    assert inner.calls == 2  # a new trial measures again


def test_measure_once_carries_the_held_can_and_leaves_it_where_released():
    """Lab 2026-11-27: the sim skill's partial release (6.3 cm) and the arm's full opening (7.5 cm)
    must both count as released; closed on a can (4.9 cm) as held; closed on nothing (0.1 cm) not."""
    from skillforge.real.background import MeasureOnce

    m = MeasureOnce(Cans(), Frames(), "fixed")
    m.reset(None)
    m.estimate([], obs_w([0.40, 0.0, 0.30], 0.075))
    m.estimate([], obs_w([0.40, 0.0, 0.056], 0.075))  # down over can a, open
    out = m.estimate([], obs_w([0.40, 0.0, 0.056], 0.049))  # closed on it
    assert np.allclose(out["a"].pos, [0.40, 0.0, 0.042])
    out = m.estimate([], obs_w([0.45, 0.10, 0.30], 0.049))  # carried
    assert np.allclose(out["a"].pos, [0.45, 0.10, 0.286]) and np.allclose(out["b"].pos, [0.50, 0.0, 0.042])
    out = m.estimate([], obs_w([0.45, 0.10, 0.056], 0.049))  # put down
    out = m.estimate([], obs_w([0.45, 0.10, 0.056], 0.063))  # partial release (sim)
    out = m.estimate([], obs_w([0.45, 0.10, 0.30], 0.063))  # hand leaves, fingers still part-open
    assert np.allclose(out["a"].pos, [0.45, 0.10, 0.042])  # stayed where it was put
    out = m.estimate([], obs_w([0.50, 0.0, 0.056], 0.001))  # closed on nothing over can b
    out = m.estimate([], obs_w([0.50, 0.0, 0.30], 0.001))
    assert np.allclose(out["b"].pos, [0.50, 0.0, 0.042])  # not picked up by an empty grip
