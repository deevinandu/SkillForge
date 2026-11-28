"""T7: free placement (objects anywhere in a start area; measured, checked, predicted in sim)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from skillforge.contracts import TaskSpec
from skillforge.perception import Estimate
from skillforge.perception.estimators import yaw_quat
from skillforge.real.core import PlacementAborted
from skillforge.real.free_placement import (
    LAB_START_AREA,
    FreePlacementOperator,
    Prediction,
    check_layout,
    pinned_task,
)

TASK = Path(__file__).parent.parent / "tasks" / "franka_stack_can_pyramid_lab.json"
GOOD = {f"cyl_{i + 1}": (x, y) for i, (x, y) in enumerate(
    [(0.34, -0.06), (0.46, -0.07), (0.58, -0.05), (0.36, -0.20), (0.48, -0.22), (0.60, -0.19)])}


def task():
    return TaskSpec.model_validate(json.loads(TASK.read_text()))


def est(xy, conf=0.9):
    return {n: Estimate(np.array([x, y, 0.0419]), yaw_quat(0.0), True, confidence=conf) for n, (x, y) in xy.items()}


def test_a_spread_layout_in_the_area_is_fine():
    assert check_layout(task(), est(GOOD)) == []


def test_layout_problems_are_named():
    t = task()
    close = dict(GOOD, cyl_2=(0.38, -0.06))  # 4 cm centre to centre from cyl_1: touching
    msgs = check_layout(t, est(close))
    assert any("cyl_1 and cyl_2" in m and "fingers" in m for m in msgs)
    out = dict(GOOD, cyl_3=(0.70, -0.05))
    assert any("cyl_3" in m and "outside the start area" in m for m in check_layout(t, est(out)))
    missing = {k: v for k, v in GOOD.items() if k != "cyl_6"}
    assert any("sees 5 of 6" in m for m in check_layout(t, est(missing)))
    assert any("not clearly seen" in m for m in check_layout(t, est(GOOD, conf=0.2)))


def test_the_start_area_keeps_clear_of_the_pyramid_and_the_walls():
    t = task()
    for r in t.scene.regions:  # every pyramid spot is outside the start area, by more than a can
        c = np.array(r.center)
        assert c[1] - 0.0254 > LAB_START_AREA.y[1] + 0.03
    assert LAB_START_AREA.x[1] <= 0.62 and min(abs(LAB_START_AREA.y[0]), abs(LAB_START_AREA.y[1])) < 0.47 - 0.1


def test_pinned_task_starts_at_the_measured_layout():
    t = task()
    p = pinned_task(t, {n: np.array([x, y, 0.05]) for n, (x, y) in GOOD.items()})
    for o in p.scene.objects:
        assert o.pose.pos[:2] == pytest.approx(GOOD[o.name], abs=1e-4)
        assert o.pose.pos[2] == pytest.approx(0.0419)  # resting on the table, as in the task
    r = p.scene.randomization
    assert r.object_xy_jitter.hi == pytest.approx(0.003)
    assert r.mass_scale == t.scene.randomization.mass_scale  # physics still randomized
    assert p.success == t.success


def scripted(answers):
    it = iter(answers)
    return lambda prompt: next(it)


def test_operator_flow_fix_then_predict_then_run(tmp_path):
    t = task()
    layouts = [est(dict(GOOD, cyl_2=(0.38, -0.06))), est(GOOD)]  # first try: two cans touching
    said, predicted = [], []

    def predict(pinned):
        predicted.append(pinned)
        return Prediction(48, 50, 0.87, 0.99, {"dropped": 2})

    op = FreePlacementOperator(t, predict=predict, out=tmp_path, ask=scripted(["", "", ""]), say=said.append)
    op.place(7, {}, measure=lambda: layouts.pop(0))
    assert any("cyl_1 and cyl_2" in s for s in said)  # told what to fix
    assert len(predicted) == 1 and predicted[0].scene.object("cyl_2").pose.pos[0] == pytest.approx(0.46)
    assert any("48/50" in s for s in said)
    rec = json.loads((tmp_path / "layout_7.json").read_text())
    assert rec["prediction"]["n_success"] == 48 and set(rec["positions"]) == set(GOOD)
    TaskSpec.model_validate(rec["pinned_task"])  # replayable


def test_operator_abort():
    op = FreePlacementOperator(task(), ask=scripted(["a"]), say=lambda s: None)
    with pytest.raises(PlacementAborted):
        op.place(0, {}, measure=lambda: est(GOOD))


def test_sim_prediction_on_a_layout():
    pytest.importorskip("mujoco")
    from skillforge.eval.regression import source_of
    from skillforge.real.free_placement import sim_prediction
    from skillforge.skills import PlaceInOrder

    pred = sim_prediction(pinned_task(task(), {n: np.array([x, y, 0.0]) for n, (x, y) in GOOD.items()}),
                          source_of(PlaceInOrder), n=4)
    assert pred.n == 4 and 0 <= pred.lo <= pred.hi <= 1 and "for this layout" in pred.line()


def test_lab_env_with_free_placement(tmp_path):
    """Stand-ins for the hardware only (fake bridge, sim camera, a 'person' placing cans in sim)."""
    pytest.importorskip("mujoco")
    sys.path.insert(0, str(Path(__file__).parent))
    import subprocess

    from test_cans import set_pose
    from test_lab_env import ROOT, SimCamera

    from skillforge.real import FrankaBridgeRobotIO
    from skillforge.real.lab import make_lab_env
    from skillforge.sim import make_env

    t = task()
    scene = make_env(t)
    scene.reset(0)
    if scene.render() is None:
        scene.close()
        pytest.skip("no GL context")
    srv = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "frankapy_bridge_server.py"), "--fake", "--port", "0"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    port = int(srv.stdout.readline().split(":")[1].split()[0])
    cam = SimCamera(scene)

    def person_places(prompt):
        for n, (x, y) in GOOD.items():
            set_pose(scene, n, [x, y, 0.0419])
        return ""

    try:
        env = make_lab_env(t, extrinsics="unused", robot=FrankaBridgeRobotIO(port=port), frames=cam, picture=None,
                           free_placement=True, predict=lambda p: Prediction(5, 5, 0.57, 1.0, {}), layout_dir=tmp_path)
        env._arm.operator.ask = person_places
        env._arm.operator.say = lambda s: None
        obs = env.reset(0)
        seen = sorted((round(float(s.pos[0]), 2), round(float(s.pos[1]), 2)) for s in obs.objects.values())
        want = sorted((round(x, 2), round(y, 2)) for x, y in GOOD.values())
        assert np.allclose(seen, want, atol=0.011)
        assert (tmp_path / "layout_0.json").exists()
        env.close()
    finally:
        cam.src.close()
        scene.close()
        srv.terminate()
        srv.wait(5)
