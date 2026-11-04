import json

from _skills import BROKEN, GOOD, HANG, NOOP, PARTIAL, src

from skillforge.agents import BudgetExceeded, FakeImprover, SkillLibrary, run_loop
from skillforge.agents.fake import DEFAULT_SCRIPT
from skillforge.forge import forge
from skillforge.sim import make_env


def test_loop_accepts_improvement_rejects_regression_survives_broken(stub_task, tmp_path):
    lib = SkillLibrary(tmp_path / "lib")
    hist = tmp_path / "history.json"
    res = run_loop(
        stub_task, src(PARTIAL), FakeImprover([GOOD, NOOP, BROKEN]), make_env,
        rounds=4, rollouts=20, target=1.01, library=lib, history_path=hist,
    )
    r = res.rounds
    assert [x.round for x in r] == [0, 1, 2, 3]
    assert 0 < r[0].report.success_rate < 1
    assert r[1].accepted and r[1].report.success_rate == 1.0  # improvement
    assert not r[2].accepted and r[2].reason.startswith("regression")  # 0% < best's CI lower bound
    assert not r[3].accepted and r[3].reason == "sandbox load"  # did not compile, loop went on
    assert r[3].report.failure_counts == {"exception": 20}
    assert res.best.source.version == "0.1.1" and res.stopped == "rounds"
    # regressions are reverted: the next candidate's parent is still the best, not the regression
    assert r[3].source.parent_version == "0.1.1"

    assert lib.versions(stub_task.id) == ["0.1.0", "0.1.1", "0.1.2", "0.1.3"]
    assert lib.lineage(stub_task.id, "0.1.3") == ["0.1.3", "0.1.1", "0.1.0"]
    assert lib.best(stub_task.id)[0].version == "0.1.1"

    doc = json.loads(hist.read_text())
    assert doc["schema"] == "skillforge.forge_history/1" and doc["stopped"] == "rounds"
    assert doc["best_version"] == "0.1.1" and len(doc["rounds"]) == 4
    assert doc["rounds"][1]["usage"]["usd"] == 0 and doc["rounds"][1]["parent_version"] == "0.1.0"


def test_loop_stops_at_target(stub_task):
    res = run_loop(stub_task, src(PARTIAL), FakeImprover([GOOD, NOOP]), make_env, rounds=5, rollouts=10, target=0.9)
    assert len(res.rounds) == 2 and res.stopped == "target"


def test_loop_respects_sandbox_timeout(stub_task):
    res = run_loop(
        stub_task, src(PARTIAL), FakeImprover([HANG, GOOD]), make_env,
        rounds=3, rollouts=5, target=1.01, timeout_s=3,
    )
    assert res.rounds[1].sandbox_stage == "timeout" and not res.rounds[1].accepted
    assert res.rounds[2].accepted  # the loop carried on after the timeout


def test_critic_can_revert(stub_task):
    class Picky(FakeImprover):
        def review(self, task, parent, parent_report, cand, cand_report):
            return False, "looks like overfitting"

    res = run_loop(stub_task, src(PARTIAL), Picky([GOOD]), make_env, rounds=2, rollouts=10, target=1.01)
    assert not res.rounds[1].accepted and res.rounds[1].reason == "critic reverted"
    assert res.rounds[1].critic == "looks like overfitting"
    assert res.best.round == 0


def test_budget_exceeded_stops_the_loop(stub_task, tmp_path):
    class Broke(FakeImprover):
        def propose(self, task, current, report):
            raise BudgetExceeded("over budget")

    res = run_loop(
        stub_task, src(PARTIAL), Broke(), make_env, rounds=5, rollouts=5, target=1.01,
        history_path=tmp_path / "h.json",
    )
    assert res.stopped == "budget" and len(res.rounds) == 1
    assert json.loads((tmp_path / "h.json").read_text())["stopped"] == "budget"


def test_improver_crash_costs_a_round_not_the_run(stub_task):
    def boom(code):
        raise RuntimeError("model said no")

    res = run_loop(stub_task, src(PARTIAL), FakeImprover([boom, GOOD]), make_env, rounds=3, rollouts=10, target=1.01)
    assert "improver error" in res.rounds[1].reason and res.rounds[2].accepted


def test_forge_spine_is_sandboxed_and_gated(stub_task):
    hist = forge(stub_task, src(PARTIAL), FakeImprover([NOOP, GOOD]), make_env, rounds=3, rollouts=10, target=1.01)
    assert [h.source.version for h in hist] == ["0.1.0", "0.1.1", "0.1.2"]
    assert hist[1].report.success_rate == 0.0 and hist[2].report.success_rate == 1.0
    assert hist[2].source.parent_version == "0.1.0"  # NOOP was rejected, so GOOD grew from the seed


def test_default_fake_script_applies_to_pick_place():
    import inspect

    from skillforge.skills import pick_place

    code = inspect.getsource(pick_place)
    for old, _new in DEFAULT_SCRIPT:
        assert old in code
