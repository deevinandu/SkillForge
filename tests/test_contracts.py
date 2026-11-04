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
