import time

import pytest
from _skills import BROKEN, GOOD, HANG, LEAK_CHECK, NETWORK, NOOP, PARTIAL, src

from skillforge.agents.sandbox import SandboxError, check_source, env_ref, evaluate_source
from skillforge.eval import evaluate
from skillforge.sim import StubEnv, make_env
from skillforge.skills import load_skill


def test_sandbox_report_matches_in_process_eval(stub_task):
    s = src(PARTIAL)
    res = evaluate_source(stub_task, s, make_env, range(20), timeout_s=60)
    ref, _ = evaluate(StubEnv, load_skill(s), stub_task, range(20))  # trusted test code, in-process is fine
    assert res.ok and res.stage == "eval"
    assert (res.report.n_success, res.report.failure_counts) == (ref.n_success, ref.failure_counts)
    assert 0 < res.report.n_success < 20  # PARTIAL really is partial
    assert res.report.skill_version == "0.1.0"  # identity comes from SkillSource, not the class attr


def test_check_accepts_good_and_reports_broken(stub_task):
    assert check_source(stub_task, src(GOOD), make_env).ok
    bad = check_source(stub_task, src(BROKEN), make_env)
    assert not bad.ok and bad.stage == "load" and "SyntaxError" in bad.error


def test_broken_skill_becomes_an_all_exception_report(stub_task):
    res = evaluate_source(stub_task, src(BROKEN), make_env, range(5))
    assert not res.ok
    assert res.report.n_rollouts == 5 and res.report.failure_counts == {"exception": 5}
    assert "SyntaxError" in res.report.failures[0].detail


def test_timeout_kills_a_hanging_skill(stub_task):
    t0 = time.perf_counter()
    res = evaluate_source(stub_task, src(HANG), make_env, range(3), timeout_s=3)
    assert time.perf_counter() - t0 < 15
    assert res.stage == "timeout" and not res.ok
    assert res.report.success_rate == 0.0 and "timeout" in res.report.failures[0].detail


def test_network_is_denied(stub_task):
    res = check_source(stub_task, src(NETWORK), make_env)
    assert not res.ok and "network access is disabled" in res.error


def test_secrets_do_not_reach_the_sandbox(stub_task, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    res = evaluate_source(stub_task, src(LEAK_CHECK), make_env, range(2))
    assert res.ok and res.report.failure_counts.get("exception", 0) == 0, res.report.failures


def test_noop_skill_fails_without_crashing(stub_task):
    res = evaluate_source(stub_task, src(NOOP), make_env, range(3))
    assert res.ok and res.report.n_success == 0 and "exception" not in res.report.failure_counts


def test_env_ref_requires_importable_factory():
    assert env_ref(make_env) == "skillforge.sim:make_env"
    assert env_ref(StubEnv) == "skillforge.sim.stub_env:StubEnv"
    with pytest.raises(SandboxError):
        env_ref(lambda task: StubEnv(task))
