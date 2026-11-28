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


# ---------------------------------------------------------------- TablePerception

class ScriptedCans:
    """detect() returns whatever `scene` holds: a list of (x, y, z, layer)."""

    needs_frames = True
    radius, half_height = 0.0254, 0.0419

    def __init__(self, scene):
        self.scene = scene
        self.last_detections = []

    def reset(self, task):
        pass

    def detect(self, frames):
        from skillforge.real.cans import Detection

        self.last_detections = [Detection(np.array([x, y]), z, layer, 100.0, 500, 0.8) for x, y, z, layer in self.scene]
        return self.last_detections

    def estimate(self, frames, obs=None):
        self.detect(frames)
        names = ["a", "b"]
        return {n: Estimate(np.array(d[:3]), yaw_quat(0.0), True, confidence=0.8) for n, d in zip(names, self.scene, strict=False)}


def table(scene):
    from skillforge.real.background import TablePerception

    tick = iter(range(10**6))  # each call is a second later: every clear view measures
    t = TablePerception(ScriptedCans(scene), Frames(), "fixed", sync=True, period=0.5, clock=lambda: float(next(tick)))
    t.reset(None)
    return t


HIGH_OPEN = ([0.45, 0.0, 0.35], 0.075)  # the view is clear: fingers open, hand high


def test_a_slipped_can_is_found_where_it_landed():
    scene = [(0.40, 0.0, 0.042, 0), (0.50, 0.0, 0.042, 0)]
    t = table(scene)
    t.estimate([], obs_w([0.40, 0.0, 0.30], 0.075))
    t.estimate([], obs_w([0.40, 0.0, 0.056], 0.049))  # holding a
    t.estimate([], obs_w([0.40, 0.0, 0.20], 0.049))  # lifted
    scene[0] = (0.30, -0.20, 0.042, 0)  # it slips out and lands here
    t.estimate([], obs_w([0.40, 0.0, 0.20], 0.001))  # fingers collapse
    assert "perception:slip:a" in t.events
    out = t.estimate([], obs_w(*HIGH_OPEN))  # first clear view
    assert np.allclose(out["a"].pos, [0.30, -0.20, 0.042]) and "perception:moved:a" in t.events


def test_a_hidden_can_keeps_its_record_and_a_ghost_does_not_take_it():
    scene = [(0.40, 0.0, 0.042, 0), (0.50, 0.0, 0.042, 0)]
    t = table(scene)
    t.estimate([], obs_w(*HIGH_OPEN))
    scene[:] = [(0.50, 0.0, 0.042, 0), (0.42, 0.04, 0.042, 0)]  # a hidden; something 4.5 cm off, once
    out = t.estimate([], obs_w(*HIGH_OPEN))
    assert np.allclose(out["a"].pos, [0.40, 0.0, 0.042])  # one sighting is not enough
    scene[:] = [(0.50, 0.0, 0.042, 0), (0.40, 0.0, 0.042, 0)]  # back in view
    out = t.estimate([], obs_w(*HIGH_OPEN))
    assert np.allclose(out["a"].pos, [0.40, 0.0, 0.042])


def test_a_knocked_can_moves_after_two_consistent_measurements():
    scene = [(0.40, 0.0, 0.042, 0), (0.50, 0.0, 0.042, 0)]
    t = table(scene)
    t.estimate([], obs_w(*HIGH_OPEN))
    scene[0] = (0.40, -0.12, 0.042, 0)  # knocked 12 cm
    out = t.estimate([], obs_w(*HIGH_OPEN))
    assert np.allclose(out["a"].pos, [0.40, 0.0, 0.042])
    out = t.estimate([], obs_w(*HIGH_OPEN))
    assert np.allclose(out["a"].pos, [0.40, -0.12, 0.042]) and "perception:moved:a" in t.events


def test_floating_detections_are_not_cans():
    scene = [(0.40, 0.0, 0.042, 0), (0.50, 0.0, 0.042, 0)]
    t = table(scene)
    t.estimate([], obs_w(*HIGH_OPEN))
    scene[0] = (0.40, -0.12, 0.042, 0)  # a moves away for real ...
    scene.append((0.42, 0.30, 0.21, 2))  # ... and the arm makes a 'can' at 21 cm with nothing under it
    t.estimate([], obs_w(*HIGH_OPEN))
    out = t.estimate([], obs_w(*HIGH_OPEN))
    assert np.allclose(out["a"].pos, [0.40, -0.12, 0.042])  # the real one, never the floating one
    assert all(p[2] < 0.1 for p in (e.pos for e in out.values()))


def test_a_measurement_from_before_a_grasp_is_not_applied():
    from skillforge.real.background import TablePerception

    t = TablePerception(ScriptedCans([(0.40, 0.0, 0.042, 0), (0.50, 0.0, 0.042, 0)]), Frames(), "fixed")
    t.reset(None)
    t.estimate([], obs_w(*HIGH_OPEN))
    t._pending = (t.version, [np.array([0.30, -0.3, 0.042]), np.array([0.5, 0.0, 0.042])])  # captured ...
    t.version += 1  # ... then a grasp or release happened
    t.estimate([], obs_w([0.45, 0.0, 0.10], 0.075))  # (view not clear: no new measurement)
    assert np.allclose(t.pos["a"], [0.40, 0.0, 0.042])
