"""The regression suite: every stored baseline vs the current library skill. `pytest -m regression`.

Baselines live in tests/regression/baselines/. Regenerate after an intended change with
`python -m skillforge.eval regression update` and commit the diff with the reason.
"""

from pathlib import Path

import pytest

from skillforge.eval.regression import check, library_sources, load_baselines, load_tasks
from skillforge.sim import make_env

ROOT = Path(__file__).parent.parent
BASELINES = load_baselines(ROOT / "tests" / "regression" / "baselines")
TASKS = load_tasks(ROOT / "tasks")
LIB = library_sources()


@pytest.mark.regression
@pytest.mark.parametrize("baseline", BASELINES, ids=[f"{b.task_id}__{b.skill}" for b in BASELINES])
def test_no_regression(baseline):
    if baseline.task_id not in TASKS or baseline.skill not in LIB:
        pytest.skip("task or skill no longer present")
    try:
        r = check(make_env, LIB[baseline.skill], TASKS[baseline.task_id], baseline)
    except NotImplementedError as e:  # sim backend not merged on this branch
        pytest.skip(str(e))
    assert r.passed, r.summary() + f"; newly failing seeds: {r.newly_failing[:20]}"


def test_suite_has_baselines():
    assert BASELINES, "no regression baselines stored"
