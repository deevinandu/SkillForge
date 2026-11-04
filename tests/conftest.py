import json
from pathlib import Path

import pytest

from skillforge.contracts import TaskSpec

TASKS = Path(__file__).parent.parent / "tasks"


@pytest.fixture
def stub_task() -> TaskSpec:
    return TaskSpec.model_validate(json.loads((TASKS / "stub_pick_cube_to_bin.json").read_text()))
