"""T7: IdenticalCylinderEstimator on rendered frames of the cylinder pyramid task.

The camera sits where the lab's real camera is (its 2026-11-27 calibration, rounded), with
stereo-like depth noise, and no segmentation masks are used: the estimator must find the
cans from depth alone, as it does on the real camera. Needs MuJoCo with rendering.
"""

import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from skillforge.contracts import TaskSpec  # noqa: E402
from skillforge.perception import SimFrameSource, VirtualCamera  # noqa: E402
from skillforge.real.cans import IdenticalCylinderEstimator  # noqa: E402
from skillforge.sim import make_env  # noqa: E402

TASK = Path(__file__).parent.parent / "tasks" / "franka_stack_cylinder_pyramid.json"
# The lab camera (calibration 2026-11-27): at (0.549, 0.397, 0.530), optical axis toward the table.
CAM_POS = np.array([0.549, 0.397, 0.530])
CAM_FWD = np.array([-0.00984, -0.59354, -0.80475])


def lab_camera() -> VirtualCamera:
    """MuJoCo orbit parameters for a camera at CAM_POS looking along CAM_FWD."""
    t = -CAM_POS[2] / CAM_FWD[2]  # where the optical axis meets the table
    lookat = CAM_POS + t * CAM_FWD
    az = np.degrees(np.arctan2(CAM_FWD[1], CAM_FWD[0]))
    el = np.degrees(np.arcsin(CAM_FWD[2]))
    return VirtualCamera("lab", tuple(lookat), float(t), float(az), float(el))


@pytest.fixture(scope="module")
def scene():
    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    env = make_env(task)
    env.reset(0)
    if env.render() is None:
        env.close()
        pytest.skip("no GL context for rendering")
    src = SimFrameSource(env, virtual=(lab_camera(),), size=(480, 640), depth_noise=0.002)  # the sim framebuffer caps width at 640
    yield task, env, src
    src.close()
    env.close()


def set_pose(env, name, pos):
    m, d = env.model, env.data
    adr = m.joint(f"{name}_free").qposadr[0]
    d.qpos[adr:adr + 3] = pos
    d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]
    d.qvel[m.joint(f"{name}_free").dofadr[0]:m.joint(f"{name}_free").dofadr[0] + 6] = 0
    mujoco.mj_forward(m, d)


def true_pos(env, name):
    return env.data.xpos[env.model.body(name).id].copy()


def spot(task, name):
    return next(np.array(r.center, float) for r in task.scene.regions if r.name == f"spot_{name}")


def match(est, truth):
    """For each true can, the xy and z distance (m) to the nearest found estimate in 3D
    (in a pyramid the bottom-middle and top cans share xy, so xy alone would pair them)."""
    found = [e.pos for e in est.values() if e.found]
    out = []
    for p in truth:
        d = [np.linalg.norm(f - p) for f in found]
        k = int(np.argmin(d))
        out.append((np.linalg.norm(found[k][:2] - p[:2]), abs(found[k][2] - p[2])))
    return np.array(out)


def test_six_separate_cans_on_the_table(scene):
    task, env, src = scene
    env.reset(0)
    for i, (x, y) in enumerate([(0.38, 0.0), (0.38, 0.12), (0.5, -0.06), (0.5, 0.08), (0.62, 0.0), (0.62, 0.14)]):
        set_pose(env, f"cyl_{i + 1}", [x, y, 0.0419])
    est = IdenticalCylinderEstimator()
    est.reset(task)
    out = est.estimate([src.capture("lab")])
    assert sum(e.found for e in out.values()) == 6 and len(est.last_detections) == 6
    err = match(out, [true_pos(env, f"cyl_{i}") for i in range(1, 7)])
    assert err[:, 0].max() < 0.006 and err[:, 1].max() < 0.008, err


def facing_camera(task, name):
    """The task's pyramid spot turned 90 deg about the pyramid's centre, so its rows run along x."""
    s = spot(task, name)
    c = np.array([0.55, 0.12])
    d = s[:2] - c
    return np.array([c[0] + d[1], c[1] - d[0], s[2]])


def test_the_task_pyramid_is_seen_end_on_by_the_lab_camera(scene):
    """The task builds its rows along y at x = 0.55; the lab camera sits at x = 0.55 looking along
    -y, so each row is seen end-on and three cans are hidden. Why the lab task turns it 90 deg."""
    task, env, src = scene
    env.reset(0)
    for i in range(1, 7):
        set_pose(env, f"cyl_{i}", spot(task, f"cyl_{i}"))
    f = src.capture("lab")
    hidden = [n for n, m in f.masks.items() if m.sum() < 100]
    assert len(hidden) >= 3, {n: int(m.sum()) for n, m in f.masks.items()}


def test_the_built_pyramid_is_found_layer_by_layer(scene):
    """3-2-1 with 8 mm gaps, rows facing the camera: the bottom cans nearly touch, the upper ones straddle them."""
    task, env, src = scene
    env.reset(0)
    for i in range(1, 7):
        set_pose(env, f"cyl_{i}", facing_camera(task, f"cyl_{i}"))
    est = IdenticalCylinderEstimator()
    est.reset(task)
    out = est.estimate([src.capture("lab")])
    layers = sorted(d.layer for d in est.last_detections)
    assert layers == [0, 0, 0, 1, 1, 2], layers
    err = match(out, [true_pos(env, f"cyl_{i}") for i in range(1, 7)])
    assert err[:, 0].max() < 0.008 and err[:, 1].max() < 0.01, err


def test_names_follow_the_cans_when_one_moves(scene):
    task, env, src = scene
    env.reset(0)
    start = {f"cyl_{i + 1}": [0.4 + 0.11 * (i % 3), -0.05 + 0.13 * (i // 3), 0.0419] for i in range(6)}
    for n, p in start.items():
        set_pose(env, n, p)
    est = IdenticalCylinderEstimator()
    est.reset(task)
    first = est.estimate([src.capture("lab")])
    # whichever name the estimator gave the can at start["cyl_1"], it must keep following that can
    mover = min(first, key=lambda n: np.linalg.norm(first[n].pos[:2] - np.array(start["cyl_1"][:2])))
    others = {n: first[n].pos.copy() for n in first if n != mover}
    set_pose(env, "cyl_1", [0.42, -0.05 + 0.03, 0.0419])  # a small move, as a carry step would make
    second = est.estimate([src.capture("lab")])
    assert np.linalg.norm(second[mover].pos[:2] - [0.42, -0.02]) < 0.008
    for n, p in others.items():
        assert np.linalg.norm(second[n].pos[:2] - p[:2]) < 0.005, n


def test_an_empty_table_finds_nothing(scene):
    task, env, src = scene
    env.reset(0)
    for i in range(1, 7):
        set_pose(env, f"cyl_{i}", [2.0 + i, 2.0, 0.0419])  # far outside the workspace
    est = IdenticalCylinderEstimator()
    est.reset(task)
    out = est.estimate([src.capture("lab")])
    assert not any(e.found for e in out.values()) and est.last_detections == []


def test_mixed_sizes_are_refused():
    task = TaskSpec.model_validate(json.loads(TASK.read_text()))
    data = task.model_dump()
    data["scene"]["objects"][0]["size"] = [0.03, 0.0419, 0.0]
    with pytest.raises(ValueError, match="identical"):
        IdenticalCylinderEstimator().reset(TaskSpec.model_validate(data))
