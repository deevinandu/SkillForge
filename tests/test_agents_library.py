import pytest
from _skills import GOOD, NOOP, src

from skillforge.agents.library import SkillLibrary, bump_version, passes_gate
from skillforge.agents.sandbox import failure_report
from skillforge.contracts import EvalReport
from skillforge.eval import wilson


def report(k: int, n: int, task_id: str = "stub_pick_cube_to_bin", version: str = "v") -> EvalReport:
    return EvalReport(
        task_id=task_id, skill="t", skill_version=version, n_rollouts=n, n_success=k,
        success_rate=k / n, ci95=wilson(k, n), failure_counts={}, failures=[], created_at="2026-11-04T00:00:00+00:00",
    )


def test_gate_uses_best_ci_lower_bound():
    best = report(45, 50)  # 90%, CI lower bound ~78.6%
    assert passes_gate(None, report(0, 50))
    assert passes_gate(best, report(41, 50))  # 82%: within noise, allowed
    assert not passes_gate(best, report(38, 50))  # 76%: below the lower bound


def test_bump_version():
    assert bump_version("0.1.0") == "0.1.1"
    assert bump_version("0.1.9") == "0.1.10"
    assert bump_version("dev") == "dev.1"


def test_library_roundtrip_lineage_and_best(tmp_path, stub_task):
    lib = SkillLibrary(tmp_path)
    a = src(NOOP, "0.1.0")
    b = src(GOOD, "0.1.1").model_copy(update={"parent_version": "0.1.0", "rationale": "fixed it"})
    c = src(NOOP, "0.1.2").model_copy(update={"parent_version": "0.1.1"})
    lib.add(a, report(2, 10, version="0.1.0"), accepted=True)
    lib.add(b, report(9, 10, version="0.1.1"), accepted=True)
    lib.add(c, failure_report(stub_task, c, [0, 1], "boom"), accepted=False)

    d = tmp_path / "stub_pick_cube_to_bin" / "0.1.1"
    assert {p.name for p in d.iterdir()} == {"skill.py", "report.json", "rationale.md", "meta.json"}
    assert lib.versions("stub_pick_cube_to_bin") == ["0.1.0", "0.1.1", "0.1.2"]
    assert lib.lineage("stub_pick_cube_to_bin", "0.1.2") == ["0.1.2", "0.1.1", "0.1.0"]
    best_src, best_rep = lib.best("stub_pick_cube_to_bin")
    assert best_src == b and best_rep.n_success == 9
    assert lib.unique_version("stub_pick_cube_to_bin", "0.1.1") == "0.1.3"
    with pytest.raises(FileExistsError):
        lib.add(a, report(2, 10), accepted=True)


def test_library_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        SkillLibrary(tmp_path).add(src(NOOP, "../evil"), report(0, 1), accepted=False)
