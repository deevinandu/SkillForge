import json

import numpy as np
import pytest
from pydantic import ValidationError

from skillforge.contracts import Action, Range, SceneSpec, SimEnv, Skill, TaskSpec
from skillforge.sim import StubEnv
from skillforge.skills import PickPlace


def test_task_roundtrips_through_json(stub_task):
    again = TaskSpec.model_validate_json(stub_task.model_dump_json())
    assert again == stub_task


def test_success_must_reference_real_object(stub_task):
    d = json.loads(stub_task.model_dump_json())
    d["success"]["object"] = "ghost"
    with pytest.raises(ValidationError, match="ghost"):
        TaskSpec.model_validate(d)


def test_unknown_fields_rejected(stub_task):
    d = json.loads(stub_task.model_dump_json())
    d["scene"]["robot"]["colour"] = "blue"
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(d)


def test_duplicate_names_rejected(stub_task):
    d = json.loads(stub_task.model_dump_json())["scene"]
    d["regions"][0]["name"] = "cube"
    with pytest.raises(ValidationError, match="duplicate"):
        SceneSpec.model_validate(d)


def test_range_ordered():
    with pytest.raises(ValidationError):
        Range(lo=1.0, hi=0.0)


def test_action_validates():
    with pytest.raises(ValueError):
        Action(mode="ee")
    with pytest.raises(ValueError):
        Action(ee_pos=np.zeros(3), gripper=2.0)


def test_stub_env_and_skill_satisfy_protocols(stub_task):
    assert isinstance(StubEnv(stub_task), SimEnv)
    assert isinstance(PickPlace(), Skill)


def _two_cubes(stub_task) -> dict:
    d = json.loads(stub_task.model_dump_json())
    cube = d["scene"]["objects"][0]
    d["scene"]["objects"].append({**cube, "name": "cube2"})
    return d


def test_all_of_and_object_on_object_roundtrip(stub_task):
    from skillforge.contracts import AllOf, ObjectOnObject, success_conditions, success_objects

    d = _two_cubes(stub_task)
    d["success"] = {
        "kind": "all_of",
        "conditions": [
            {"kind": "object_in_region", "object": "cube", "region": "bin"},
            {"kind": "object_on_object", "object": "cube2", "support": "cube"},
        ],
    }
    t = TaskSpec.model_validate(d)
    assert isinstance(t.success, AllOf) and isinstance(t.success.conditions[1], ObjectOnObject)
    assert TaskSpec.model_validate_json(t.model_dump_json()) == t
    assert [c.kind for c in success_conditions(t.success)] == ["object_in_region", "object_on_object"]
    assert success_objects(t.success) == ("cube", "cube2")
    assert success_conditions(stub_task.success) == (stub_task.success,)
    assert success_objects(stub_task.success) == ("cube",)


@pytest.mark.parametrize(
    "success, missing",
    [
        ({"kind": "object_on_object", "object": "cube2", "support": "ghost"}, "ghost"),
        ({"kind": "all_of", "conditions": [{"kind": "object_lifted", "object": "ghost"}]}, "ghost"),
        ({"kind": "all_of", "conditions": [{"kind": "object_in_region", "object": "cube", "region": "nowhere"}]}, "nowhere"),
    ],
)
def test_compound_success_must_reference_real_names(stub_task, success, missing):
    d = _two_cubes(stub_task)
    d["success"] = success
    with pytest.raises(ValidationError, match=missing):
        TaskSpec.model_validate(d)


@pytest.mark.parametrize(
    "success",
    [
        {"kind": "object_on_object", "object": "cube", "support": "cube"},  # onto itself
        {"kind": "all_of", "conditions": []},  # empty
        {"kind": "all_of", "conditions": [{"kind": "all_of", "conditions": [{"kind": "object_lifted", "object": "cube"}]}]},  # nested
        {"kind": "object_on_object", "object": "cube2", "support": "cube", "z_tol": 0},
    ],
)
def test_bad_compound_success_rejected(stub_task, success):
    d = _two_cubes(stub_task)
    d["success"] = success
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(d)


def _coloured(stub_task, colors=("red", "blue")) -> dict:
    d = json.loads(stub_task.model_dump_json())
    cube = d["scene"]["objects"][0]
    d["scene"]["objects"] = [{**cube, "name": f"cube{i}", "color": c} for i, c in enumerate(colors)]
    d["success"] = {"kind": "color_in_region", "color": colors[0], "region": "bin"}
    return d


def test_color_sorting_fields_roundtrip(stub_task):
    from skillforge.contracts import ColorInRegion, success_conditions, success_objects

    d = _coloured(stub_task)
    d["scene"]["randomization"]["shuffle_colors"] = True
    d["scene"]["obs_cameras"] = ["wrist"]
    t = TaskSpec.model_validate(d)
    assert TaskSpec.model_validate_json(t.model_dump_json()) == t
    assert [o.color for o in t.scene.objects] == ["red", "blue"]
    assert t.scene.randomization.shuffle_colors and t.scene.obs_cameras == ("wrist",)
    assert isinstance(t.success, ColorInRegion) and success_conditions(t.success) == (t.success,)
    assert success_objects(t.success) == ()  # names no object: which ones depends on the episode


def test_color_fields_default_off(stub_task):
    assert stub_task.scene.objects[0].color is None
    assert not stub_task.scene.randomization.shuffle_colors and stub_task.scene.obs_cameras == ()


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: d["success"].update(region="nowhere"), "nowhere"),
        (lambda d: d["success"].update(color="green"), "green"),  # no object is green
        (lambda d: d["success"].update(color="mauve"), "color"),  # not a colour name
        (lambda d: d["scene"]["objects"][0].update(color="teal"), "color"),
    ],
)
def test_bad_color_sorting_rejected(stub_task, change, match):
    d = _coloured(stub_task)
    change(d)
    with pytest.raises(ValidationError, match=match):
        TaskSpec.model_validate(d)


def test_knockdown_roundtrip_and_validation(stub_task):
    from skillforge.contracts import Knockdown

    d = json.loads(stub_task.model_dump_json())
    assert stub_task.scene.randomization.knockdown is None
    d["scene"]["randomization"]["knockdown"] = {"at_step": {"lo": 100, "hi": 200}, "count": {"lo": 1, "hi": 3}}
    t = TaskSpec.model_validate(d)
    k = t.scene.randomization.knockdown
    assert isinstance(k, Knockdown) and k.radius == 0.08 and k.count.hi == 3
    assert TaskSpec.model_validate_json(t.model_dump_json()) == t
    d["scene"]["randomization"]["knockdown"] = {"at_step": {"lo": -5, "hi": 10}}
    with pytest.raises(ValidationError, match=">= 0"):
        TaskSpec.model_validate(d)


# ---------------------------------------------------------------- dominoes: toppled + disturbances


def _dominoes(stub_task, n=3) -> dict:
    d = json.loads(stub_task.model_dump_json())
    dom = {"name": "d", "shape": "box", "size": [0.005, 0.02, 0.04], "mass": 0.03}
    d["scene"]["objects"] = [{**dom, "name": f"d{i}", "pose": {"pos": [0.4 + 0.05 * i, 0.0, 0.04]}} for i in range(n)]
    d["scene"]["regions"] = [{"name": "lane", "center": [0.45, 0, 0.03], "half_extents": [0.15, 0.05, 0.04]}]
    d["success"] = {
        "kind": "all_of",
        "conditions": [
            {"kind": "object_toppled", "object": f"d{i}", "direction": [1.0, 0.0], "region": "lane"} for i in range(n)
        ],
    }
    return d


def test_toppled_and_disturbances_roundtrip(stub_task):
    from skillforge.contracts import Disturbance, ObjectToppled, success_conditions, success_objects

    d = _dominoes(stub_task)
    d["scene"]["disturbances"] = [
        {"kind": "pose", "objects": ["d1"], "at_step": {"lo": 10, "hi": 40}, "xy": {"lo": 0.01, "hi": 0.03},
         "yaw": {"lo": 0.1, "hi": 0.5}, "before_event": "toppled:"},
        {"kind": "knock", "after_event": "release:", "target": "event_object", "prob": 0.5,
         "ang_vel": {"lo": 2, "hi": 4}, "knock_dir": [1, 0]},
    ]
    t = TaskSpec.model_validate(d)
    assert TaskSpec.model_validate_json(t.model_dump_json()) == t
    leaves = success_conditions(t.success)
    assert all(isinstance(c, ObjectToppled) for c in leaves) and leaves[0].min_tilt == pytest.approx(0.6)
    assert success_objects(t.success) == ("d0", "d1", "d2")
    a, b = t.scene.disturbances
    assert isinstance(a, Disturbance) and a.count == 1 and a.prob == 1.0 and a.target == "random"
    assert b.kind == "knock" and b.knock_dir == (1.0, 0.0)


def test_disturbances_default_off(stub_task):
    assert stub_task.scene.disturbances == ()


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda d: d["success"]["conditions"][0].update(region="nowhere"), "nowhere"),
        (lambda d: d["success"]["conditions"][0].update(object="ghost"), "ghost"),
        (lambda d: d["success"]["conditions"][0].update(direction=[0, 0]), "direction"),
        (lambda d: d["success"]["conditions"][0].update(min_tilt=0), "min_tilt"),
        (lambda d: d["scene"].update(disturbances=[{"kind": "pose", "objects": ["ghost"], "xy": {"lo": 0, "hi": 0.01}}]), "ghost"),
        (lambda d: d["scene"].update(disturbances=[{"kind": "pose"}]), "non-zero xy or yaw"),
        (lambda d: d["scene"].update(disturbances=[{"kind": "knock"}]), "ang_vel"),
        (lambda d: d["scene"].update(disturbances=[{"kind": "pose", "yaw": {"lo": 0, "hi": 1}, "target": "event_object"}]), "after_event"),
        (lambda d: d["scene"].update(disturbances=[{"kind": "pose", "yaw": {"lo": -1, "hi": 1}}]), ">= 0"),
        (lambda d: d["scene"].update(disturbances=[{"kind": "pose", "yaw": {"lo": 0, "hi": 1}, "prob": 1.5}]), "prob"),
    ],
)
def test_bad_dominoes_rejected(stub_task, change, match):
    d = _dominoes(stub_task)
    change(d)
    with pytest.raises(ValidationError, match=match):
        TaskSpec.model_validate(d)
