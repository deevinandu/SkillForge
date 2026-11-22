"""T1: the success predicate for compound specs (object_on_object, all_of), on TRUE positions."""

import json

import numpy as np
import pytest

from skillforge.contracts import TaskSpec
from skillforge.sim import is_success
from skillforge.sim.success import footprint_radius


def _task(stub_task, success: dict) -> TaskSpec:
    d = json.loads(stub_task.model_dump_json())
    cube = d["scene"]["objects"][0]
    d["scene"]["objects"].append({**cube, "name": "top"})
    d["success"] = success
    return TaskSpec.model_validate(d)


def _stack(t: TaskSpec, dx: float = 0.0, dz: float = 0.0) -> dict[str, np.ndarray]:
    h = float(t.scene.object("cube").size[2])
    base = np.array([0.5, 0.0, h])
    return {"cube": base, "top": base + [dx, 0.0, 2 * h + dz]}


ON = {"kind": "object_on_object", "object": "top", "support": "cube"}


def test_object_on_object(stub_task):
    t = _task(stub_task, ON)
    r = footprint_radius(t.scene.object("cube"))
    start = _stack(t)
    assert is_success(t, _stack(t), start)
    assert is_success(t, _stack(t, dx=0.9 * r, dz=0.008), start)
    assert not is_success(t, _stack(t, dx=1.1 * r), start)  # off the edge
    assert not is_success(t, _stack(t, dz=0.03), start)  # hovering (e.g. still in the gripper)
    assert not is_success(t, _stack(t, dz=-2 * float(t.scene.object("cube").size[2])), start)  # beside it
    m = _task(stub_task, {**ON, "xy_margin": 0.5 * r})
    assert not is_success(m, _stack(m, dx=0.6 * r), start)


def test_all_of_needs_every_leaf(stub_task):
    bin_ = stub_task.scene.region("bin")
    t = _task(
        stub_task,
        {"kind": "all_of", "conditions": [ON, {"kind": "object_in_region", "object": "cube", "region": "bin"}]},
    )
    on = _stack(t)
    assert not is_success(t, on, on)  # stacked, but not in the bin
    c = np.asarray(bin_.center, dtype=float)
    h = float(t.scene.object("cube").size[2])
    in_bin = {"cube": np.array([c[0], c[1], c[2]]), "top": np.array([c[0], c[1], c[2] + 2 * h])}
    assert is_success(t, in_bin, on)
    in_bin["top"] = in_bin["top"] + [0.2, 0, 0]
    assert not is_success(t, in_bin, on)


def test_leaf_specs_unchanged(stub_task):
    p = {"cube": np.asarray(stub_task.scene.region("bin").center, dtype=float)}
    assert is_success(stub_task, p, p)
    with pytest.raises(KeyError):
        is_success(_task(stub_task, ON), p, p)  # the env must report every object the spec names
