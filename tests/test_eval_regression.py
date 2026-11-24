"""T4: regression baselines, checks and storage (the suite itself is test_regression_suite.py)."""

from eval_helpers import noisy, pick_place_source

from skillforge.eval.regression import (
    Baseline,
    check,
    compare,
    library_sources,
    load_baselines,
    load_tasks,
    make_baseline,
    save_baseline,
    update_baselines,
)
from skillforge.sim import StubEnv


def test_same_skill_passes_its_own_baseline(stub_task):
    task = noisy(stub_task)
    b = make_baseline(StubEnv, pick_place_source(), task, range(40))
    assert b.threshold == b.ci95[0] < b.success_rate < 1.0
    r = check(StubEnv, pick_place_source(), task, b)
    assert r.passed and not r.stale
    assert r.candidate_rate == b.success_rate and r.newly_failing == [] == r.newly_passing


def test_worse_skill_fails_and_names_the_seeds(stub_task):
    b = make_baseline(StubEnv, pick_place_source(), stub_task, range(40))
    worse = pick_place_source("0.2.0", tol=0.0005)  # can't "arrive" under obs noise
    r = check(StubEnv, worse, stub_task, b)
    assert not r.passed and r.candidate_rate < b.threshold
    assert r.newly_failing and set(r.newly_failing) <= set(b.passed_seeds)
    assert "below threshold" in r.reason and "FAIL" in r.summary()


def test_min_success_floor_and_stale_task(stub_task):
    b = make_baseline(StubEnv, pick_place_source(), stub_task, range(20), min_success=0.99)
    assert b.threshold == 0.99
    r = check(StubEnv, pick_place_source(), noisy(stub_task, 0.03), b)
    assert r.stale and not r.passed and "re-baseline" in r.reason


def test_compare_two_versions(stub_task):
    r = compare(StubEnv, pick_place_source("0.2.0", tol=0.0005), pick_place_source(), stub_task, range(20))
    assert r.baseline_version == "0.1.0" and r.candidate_version == "0.2.0" and not r.passed


def test_storage_and_discovery(tmp_path, stub_task):
    from skillforge.sim import make_env

    lib = library_sources()
    assert "pick_place" in lib and lib["pick_place"].entrypoint == "PickPlace"
    tasks = load_tasks(__import__("pathlib").Path(__file__).parent.parent / "tasks")
    assert stub_task.id in tasks
    assert {"push_to_region", "pick_from_clutter"} <= set(lib)
    one = {"pick_place": lib["pick_place"]}  # every exported skill would be baselined; check one
    paths = update_baselines(make_env, {stub_task.id: stub_task}, one, tmp_path, seeds=range(10), workers=1)
    assert [p.name for p in paths] == [f"{stub_task.id}__pick_place.json"]
    (b,) = load_baselines(tmp_path)
    assert isinstance(b, Baseline) and b.seeds == list(range(10))
    save_baseline(b, tmp_path)  # overwrite is fine
    assert len(load_baselines(tmp_path)) == 1


def test_update_skips_skills_that_reject_the_task(tmp_path, stub_task):
    from skillforge.contracts import ObjectLifted
    from skillforge.sim import make_env

    obj = stub_task.success.object
    lift = stub_task.model_copy(update={"id": "stub_lift", "success": ObjectLifted(object=obj)})
    one = {"pick_place": library_sources()["pick_place"]}  # PickPlace raises ValueError on a lift task
    tasks = {lift.id: lift, stub_task.id: stub_task}
    paths = update_baselines(make_env, tasks, one, tmp_path, seeds=range(5), workers=1)
    assert [p.name for p in paths] == [f"{stub_task.id}__pick_place.json"]


def test_task_hash_ignores_default_valued_fields(stub_task):
    """A field added to the contract with a default must not make existing baselines stale."""
    from skillforge.eval.regression import _content, task_hash

    c = _content(stub_task)
    assert c["success"]["kind"] == "object_in_region"  # union tags are kept
    rebuilt = type(stub_task).model_validate(c)
    assert rebuilt == stub_task and task_hash(rebuilt) == task_hash(stub_task)
    changed = stub_task.model_copy(update={"max_steps": stub_task.max_steps + 1})
    assert task_hash(changed) != task_hash(stub_task)
