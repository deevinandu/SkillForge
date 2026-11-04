"""ClaudeCodeImprover, offline: a scripted runner stands in for the `claude` CLI."""

import json
import subprocess
from pathlib import Path

import pytest
from eval_helpers import noisy

from skillforge.agents import BudgetExceeded, ImproverError
from skillforge.agents.claude_code import DEV_SEED_BASE, ClaudeCodeImprover, parse_cli_json
from skillforge.agents.loop import run_loop
from skillforge.cli import load_seed_skill
from skillforge.sim.stub_env import StubEnv


def _result(cost=0.25, session="sess-1", subtype="success", turns=7):
    doc = {"type": "result", "subtype": subtype, "is_error": subtype != "success", "session_id": session,
           "total_cost_usd": cost, "num_turns": turns, "duration_ms": 1234, "result": "done",
           "usage": {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 30}}  # fmt: skip
    return json.dumps(doc)


class Script:
    """Each call applies the next edit to skill.py in the workspace and returns a CLI result."""

    def __init__(self, *edits, run_dev_eval=False, cost=0.25):
        self.edits = list(edits)
        self.cmds: list[list[str]] = []
        self.run_dev_eval = run_dev_eval
        self.cost = cost
        self.dev_out = ""

    def __call__(self, cmd, cwd: Path, timeout_s):
        self.cmds.append(cmd)
        edit = self.edits.pop(0) if self.edits else None
        skill = cwd / "skill.py"
        if edit is not None:
            skill.write_text(edit(skill.read_text()))
            (cwd / "RATIONALE.md").write_text(f"edit {len(self.cmds)}")
        if self.run_dev_eval:
            p = subprocess.run(["./dev_eval", "4"], cwd=cwd, capture_output=True, text=True, timeout=300)
            self.dev_out = p.stdout + p.stderr
        return subprocess.CompletedProcess(cmd, 0, stdout=_result(cost=self.cost), stderr="")


def _tighter(code: str) -> str:
    return code.replace("tol = 0.008", "tol = 0.004")


def _broken(code: str) -> str:
    return code + "\nthis is not python\n"


def _imp(runner, tmp_path, **kw):
    return ClaudeCodeImprover(runner=runner, workdir=tmp_path / "ws", make_env="skillforge.sim:make_env", **kw)


def test_parse_cli_json_success_and_error():
    r = parse_cli_json("log line\n" + _result(cost=1.5, session="abc"))
    assert r.ok and r.session_id == "abc" and r.cost_usd == 1.5 and r.num_turns == 7
    bad = parse_cli_json(_result(subtype="error_max_budget_usd"))
    assert not bad.ok and bad.error == "error_max_budget_usd"
    assert not parse_cli_json("not json").ok


def test_propose_returns_checked_candidate_and_records_cost(tmp_path, stub_task):
    seed = load_seed_skill("pick_place_v01")
    runner = Script(_tighter, run_dev_eval=True, cost=0.4)
    imp = _imp(runner, tmp_path, budget_usd=3.0)
    from skillforge.agents.sandbox import evaluate_source

    rep = evaluate_source(stub_task, seed, "skillforge.sim:make_env", range(4)).report
    cand = imp.propose(stub_task, seed, rep)
    assert cand.version == "0.1.1" and cand.parent_version == "0.1.0"
    assert "tol = 0.004" in cand.code and cand.rationale == "edit 1"
    assert imp.usage.usd == pytest.approx(0.4) and imp.usage.calls[0].cache_read_input_tokens == 30
    cmd = runner.cmds[0]
    assert cmd[:2] == ["claude", "-p"] and "--max-budget-usd" in cmd and "dontAsk" in cmd
    assert cmd[cmd.index("--max-budget-usd") + 1] == "3.00"  # capped by what is left of the run budget
    ws = Path(imp.calls[0]["workspace"])
    assert {"skill.py", "REPORT.md", "CONTEXT.md", "task.json", "dev_eval"} <= {p.name for p in ws.iterdir()}
    # dev_eval ran in the sandbox on DEV seeds, disjoint from the gate seeds 0..n-1
    assert "4/4" in runner.dev_out or "success" in runner.dev_out.lower(), runner.dev_out
    assert json.loads((ws / "dev_eval.json").read_text())["seed_base"] == DEV_SEED_BASE


def test_broken_edit_is_fixed_by_resuming_the_session(tmp_path, stub_task):
    seed = load_seed_skill("pick_place_v01")
    runner = Script(_broken, lambda c: _tighter(c.replace("\nthis is not python\n", "\n")))
    imp = _imp(runner, tmp_path)
    cand = imp.propose(stub_task, seed, _report(stub_task, seed))
    assert "tol = 0.004" in cand.code
    assert "--resume" in runner.cmds[1] and runner.cmds[1][runner.cmds[1].index("--resume") + 1] == "sess-1"
    assert "did not pass validation" in runner.cmds[1][2]


def test_no_change_raises_after_one_retry(tmp_path, stub_task):
    seed = load_seed_skill("pick_place_v01")
    runner = Script(None, None)
    imp = _imp(runner, tmp_path)
    with pytest.raises(ImproverError, match="did not change"):
        imp.propose(stub_task, seed, _report(stub_task, seed))
    assert len(runner.cmds) == 2


def test_budget_refuses_before_calling(tmp_path, stub_task):
    seed = load_seed_skill("pick_place_v01")
    runner = Script(_tighter)
    imp = _imp(runner, tmp_path, budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        imp.propose(stub_task, seed, _report(stub_task, seed))
    assert runner.cmds == []


def test_loop_with_claude_code_improver(tmp_path, stub_task):
    """End to end on a noisy stub task: sandboxed gate, library, budget accounting."""
    task = noisy(stub_task, 0.012)
    seed = load_seed_skill("pick_place_v01")
    imp = _imp(Script(_tighter, _broken), tmp_path, budget_usd=10.0, max_fix_attempts=0)
    from skillforge.agents import SkillLibrary

    res = run_loop(task, seed, imp, make_env="skillforge.sim:make_env", rounds=3, rollouts=12,
                   library=SkillLibrary(tmp_path / "lib"), history_path=tmp_path / "h.json", target=1.01)  # fmt: skip
    hist = res.rounds
    assert hist[1].source.version == "0.1.1" and hist[1].source.rationale == "edit 1"
    assert len(hist) == 3 and not hist[2].accepted and "claude-code produced no valid skill" in hist[2].reason
    assert (tmp_path / "h.json").exists() and imp.usage.usd == pytest.approx(0.5)


def _report(task, seed):
    from skillforge.eval.harness import evaluate
    from skillforge.skills.pick_place_v01 import PickPlaceV01

    return evaluate(StubEnv, PickPlaceV01(), task, range(3))[0]
