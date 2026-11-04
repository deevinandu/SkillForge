"""Phase 0 gate: the stub pipeline runs end to end."""

import inspect

from skillforge import skills
from skillforge.agents import NullImprover
from skillforge.contracts import SkillSource
from skillforge.eval import evaluate, wilson
from skillforge.forge import forge
from skillforge.sim import StubEnv, make_env
from skillforge.skills import PickPlace, load_skill


def test_scripted_skill_solves_stub_task_without_randomization(stub_task):
    task = stub_task.model_copy(update={"scene": stub_task.scene.model_copy(update={"randomization": type(stub_task.scene.randomization)()})})
    report, _ = evaluate(StubEnv, PickPlace(), task, range(5))
    assert report.success_rate == 1.0, report.summary()


def test_eval_is_deterministic_per_seed(stub_task):
    a, _ = evaluate(StubEnv, PickPlace(), stub_task, range(20))
    b, _ = evaluate(StubEnv, PickPlace(), stub_task, range(20))
    assert a.n_success == b.n_success and a.failure_counts == b.failure_counts


def test_randomization_is_recorded(stub_task):
    _, eps = evaluate(StubEnv, PickPlace(), stub_task, range(3))
    assert all("object_xy_jitter" in e.randomization for e in eps)
    assert len({e.randomization["object_xy_jitter"] for e in eps}) == 3


def test_crashing_skill_is_a_failure_not_a_crash(stub_task):
    class Boom(PickPlace):
        def act(self, obs):
            raise RuntimeError("kaboom")

    report, _ = evaluate(StubEnv, Boom(), stub_task, range(3))
    assert report.failure_counts == {"exception": 3}


def test_forge_loop_runs_with_source_skill(stub_task):
    src = SkillSource(
        name="pick_place",
        version="0.1.0",
        code=inspect.getsource(skills.pick_place),
        entrypoint="PickPlace",
    )
    assert load_skill(src).name == "pick_place"
    hist = forge(stub_task, src, NullImprover(), make_env, rounds=2, rollouts=20, target=1.01)
    assert len(hist) == 2
    assert hist[0].report.n_rollouts == 20
    print(hist[-1].report.summary())


def test_wilson_bounds():
    lo, hi = wilson(45, 50)
    assert 0.78 < lo < 0.9 < hi < 0.97
