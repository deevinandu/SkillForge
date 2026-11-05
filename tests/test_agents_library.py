import json
from pathlib import Path

import pytest
from _skills import GOOD, NOOP, src

from skillforge.agents.library import (
    SkillLibrary,
    bump_version,
    mcnemar_one_sided,
    outcomes_from_report,
    paired_compare,
    passes_gate,
    passes_paired_gate,
)
from skillforge.agents.sandbox import failure_report
from skillforge.contracts import EvalReport, FailureCase
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


def test_mcnemar_one_sided_exact():
    assert mcnemar_one_sided(0, 0) == 1.0
    assert mcnemar_one_sided(0, 4) == pytest.approx(1 / 16)
    assert mcnemar_one_sided(1, 7) == pytest.approx(9 / 256)
    assert mcnemar_one_sided(7, 11) == pytest.approx(0.2403, abs=1e-4)
    assert mcnemar_one_sided(5, 5) > 0.5  # a tie is no evidence


def test_paired_gate_counts_and_rule():
    best = {s: s < 60 for s in range(100)}  # 60/100
    cand = {s: s < 56 or 60 <= s < 68 for s in range(100)}  # 64/100: loses 4, wins 8
    cmp = paired_compare(best, cand)
    assert (cmp.both, cmp.best_only, cmp.cand_only, cmp.neither) == (56, 4, 8, 32)
    assert not passes_paired_gate(best, cand, alpha=0.1)[0]  # +8/-4, p ~0.19
    better = {s: s < 65 for s in range(100)}  # +5/-0, p 1/32
    assert passes_paired_gate(best, better, alpha=0.1)[0]
    assert not passes_paired_gate(best, better, alpha=0.01)[0]
    assert passes_paired_gate(None, cand) == (True, None)
    with pytest.raises(ValueError, match="same seeds"):
        paired_compare(best, {s: True for s in range(50)})


def test_outcomes_from_report_needs_complete_failure_list():
    fails = [FailureCase(seed=s, tag="timeout", detail="", randomization={}) for s in (1, 3)]
    rep = report(2, 4).model_copy(update={"failures": fails})
    assert outcomes_from_report(rep, range(4)) == {0: True, 1: False, 2: True, 3: False}
    with pytest.raises(ValueError, match="max_failures"):
        outcomes_from_report(report(2, 4), range(4))  # the failure sample is empty


def test_paired_gate_on_the_2026_11_04_hard_run():
    """Per-seed outcomes of that run's versions, re-evaluated on gate seeds 0-99 (current physics,
    same 69/73/95 totals as the run). The legacy gate accepted 0.2.1; the paired gate rejects it
    and accepts 0.2.2."""
    doc = json.loads((Path(__file__).parent / "fixtures" / "hard_2026_11_04_gate_seeds.json").read_text())
    out = {v: {s: s not in set(fail) for s in range(100)} for v, fail in doc["failing_seeds"].items()}
    assert [sum(o.values()) for o in out.values()] == [69, 73, 95]
    best = report(69, 100)
    assert passes_gate(best, report(73, 100))  # what the legacy rule did
    ok1, c1 = passes_paired_gate(out["0.2.0"], out["0.2.1"])
    assert not ok1 and (c1.best_only, c1.cand_only) == (7, 11) and c1.p_value > 0.2
    ok2, c2 = passes_paired_gate(out["0.2.0"], out["0.2.2"])
    assert ok2 and (c2.best_only, c2.cand_only) == (2, 28) and c2.p_value < 1e-5
