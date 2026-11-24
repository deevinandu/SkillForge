"""PickPlace v0.2 on the stub and on the fault-injecting stub fixture (tests/perturb.py)."""

import inspect

import numpy as np
import pytest
from perturb import PerturbedStubEnv

from skillforge.contracts import Range, Skill, SkillSource
from skillforge.eval import evaluate
from skillforge.eval.harness import run_episode
from skillforge.sim import StubEnv
from skillforge.skills import PickPlace, PickPlaceV01, load_skill
from skillforge.skills import pick_place as pick_place_module
from skillforge.skills.grasp import quat_to_mat


def rollout(env, skill, task, seed):
    ep = run_episode(env, skill, task, seed)
    return ep, list(skill.ctx.log)


def test_v02_implements_skill_and_loads_from_source():
    assert isinstance(PickPlace(), Skill)
    src = SkillSource(name="pick_place", version="0.3.0", code=inspect.getsource(pick_place_module), entrypoint="PickPlace")
    sk = load_skill(src)
    assert sk.version == "0.3.0"


def test_v02_solves_randomized_stub(stub_task):
    report, _ = evaluate(StubEnv, PickPlace(), stub_task, range(50))
    assert report.success_rate == 1.0, report.summary()


def test_v01_still_available_and_unchanged(stub_task):
    assert PickPlaceV01.version == "0.1.0"
    report, _ = evaluate(StubEnv, PickPlaceV01(), stub_task, range(20))
    assert report.success_rate == 1.0, report.summary()


def test_v02_deterministic_per_seed(stub_task):
    mk = PerturbedStubEnv.factory(p_miss=0.3, p_slip=0.3, extra_noise=0.003)
    a, ea = evaluate(mk, PickPlace(), stub_task, range(15))
    b, eb = evaluate(mk, PickPlace(), stub_task, range(15))
    assert [e.success for e in ea] == [e.success for e in eb]
    assert [e.n_steps for e in ea] == [e.n_steps for e in eb]


def test_fixture_is_seeded_and_leaves_stub_draws_alone(stub_task):
    plain, faulty = StubEnv(stub_task), PerturbedStubEnv(stub_task, p_slip=1.0)
    for s in range(5):
        np.testing.assert_allclose(plain.reset(s).objects["cube"].pos, faulty.reset(s).objects["cube"].pos)
        assert plain.randomization_sample() == faulty.randomization_sample()


def test_injected_slip_triggers_regrasp(stub_task):
    """Every episode gets exactly one slip mid-carry; v0.2 re-grasps and succeeds, v0.1 does not."""
    st = PickPlace.settle_steps  # slip after the grip settles and the lift starts: mid-carry
    env = PerturbedStubEnv(stub_task, p_slip=1.0, max_slips=1, slip_after=(st + 4, st + 8))
    for seed in range(10):
        ep, log = rollout(env, PickPlace(), stub_task, seed)
        assert env.n_slips == 1
        assert ep.success, (seed, ep.failure_detail, log)
        assert any(":retry:grasp_slip" in e for e in log), log
        assert ep.events.count("grasp:cube") == 2
    v01, _ = evaluate(PerturbedStubEnv.factory(p_slip=1.0, max_slips=1, slip_after=(8, 12)), PickPlaceV01(), stub_task, range(10))
    assert v01.success_rate == 0.0


@pytest.mark.parametrize("honest", [False, True])
def test_slip_detection_signal(stub_task, honest):
    """With an honest width the width collapse fires; without it, object tracking does."""
    st = PickPlace.settle_steps
    env = PerturbedStubEnv(stub_task, p_slip=1.0, max_slips=1, slip_after=(st + 6, st + 6), honest_width=honest)
    ep, log = rollout(env, PickPlace(), stub_task, 0)
    assert ep.success
    slip = next(e for e in log if ":retry:grasp_slip" in e)
    assert ("width_collapsed" in slip) == honest, slip


def test_missed_grasp_retried_once_with_next_grasp(stub_task):
    env = PerturbedStubEnv(stub_task, p_miss=1.0, max_misses=1)
    ep, log = rollout(env, PickPlace(), stub_task, 0)
    assert ep.success, log
    assert sum(":retry:grasp_miss" in e for e in log) == 1
    sk = PickPlace()
    run_episode(env, sk, stub_task, 0)
    assert sk.ctx.bb["rank"] == 1  # second attempt used the next-ranked grasp

    env2 = PerturbedStubEnv(stub_task, p_miss=1.0, max_misses=2)  # two misses exceed the budget
    ep2, log2 = rollout(env2, PickPlace(), stub_task, 0)
    assert not ep2.success and ep2.failure == "grasp_miss"
    assert any(":failed:grasp_miss" in e for e in log2)


def test_grasp_yaw_follows_observed_object_yaw(stub_task):
    rnd = stub_task.scene.randomization.model_copy(update={"object_yaw_jitter": Range(lo=-0.7, hi=0.7)})
    task = stub_task.model_copy(update={"scene": stub_task.scene.model_copy(update={"randomization": rnd})})
    env = PerturbedStubEnv(task, report_yaw=True)
    for seed in range(5):
        obs = env.reset(seed)
        sk = PickPlace()
        sk.reset(task)
        act = sk.act(obs)
        yaw = env.randomization_sample()["object_yaw_jitter"]
        closing = quat_to_mat(act.ee_quat)[:, 1]
        obj_axes = quat_to_mat(obs.objects["cube"].quat)[:, :2]
        assert max(abs(closing @ obj_axes[:, 0]), abs(closing @ obj_axes[:, 1])) == pytest.approx(1.0, abs=1e-9)
        assert abs(yaw) < 0.7 + 1e-9
        assert run_episode(env, PickPlace(), task, seed).success


def test_v02_beats_v01_under_faults(stub_task):
    """Measured on the fixture, not physics: misses and slips at 30% each plus 3 mm extra noise."""
    mk = PerturbedStubEnv.factory(p_miss=0.3, p_slip=0.3, extra_noise=0.003)
    v01, _ = evaluate(mk, PickPlaceV01(), stub_task, range(100))
    v02, _ = evaluate(mk, PickPlace(), stub_task, range(100))
    print(v01.summary())
    print(v02.summary())
    assert v02.success_rate >= 0.8
    assert v02.success_rate > v01.success_rate + 0.2


def test_noise_alone_does_not_hurt_v02(stub_task):
    report, _ = evaluate(PerturbedStubEnv.factory(extra_noise=0.004), PickPlace(), stub_task, range(50))
    assert report.success_rate == 1.0, report.summary()
