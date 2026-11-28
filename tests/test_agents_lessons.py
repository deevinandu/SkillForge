"""Scoped lessons (docs/LESSONS.md): store, scope labels, curation, and the loop filing lessons."""

import json

import pytest
from _skills import GOOD, NOOP, PARTIAL, src

from skillforge.agents import FakeImprover, run_loop
from skillforge.agents.curate import Target, admit, check_scope, record_round, verify_lesson
from skillforge.agents.lessons import (
    Grade,
    Lesson,
    LessonReport,
    LessonStore,
    Provenance,
    Scope,
    parse_lesson_report,
    render,
)
from skillforge.agents.scope import (
    changed_numbers,
    check_from_outcomes,
    decide_scope,
    label_confusion,
    structural_scope,
)
from skillforge.sim import make_env


def _lesson(store: LessonStore, robot="franka_panda", title="Slow the carry", **kw) -> Lesson:
    les = Lesson(
        id=store.new_id(title), title=title, topic="transport", failure_signature=["grasp_slip"],
        trigger="a cylinder slides in the grip on the carry", strategy="ramp the carry speed up gently",
        params={"upright_accel": 0.00012},
        source=Provenance(robot=robot, task_id="t", skill="place_in_order", parent_version="0.1.0", version="0.1.1"),
        **kw,
    )
    return store.propose(les)


# ------------------------------------------------------------------ store


def test_scope_parse_and_applies():
    assert Scope.parse("general").applies_to("anything")
    s = Scope.parse("robot:kinova_gen3")
    assert s.robot == "kinova_gen3" and s.applies_to("kinova_gen3") and not s.applies_to("franka_panda")
    assert s.label() == "robot:kinova_gen3"
    with pytest.raises(ValueError):
        Scope.parse("robot:")


def test_only_verified_lessons_with_a_scope_are_promoted_and_shown(tmp_path):
    store = LessonStore(tmp_path)
    les = _lesson(store)
    assert store.for_robot("franka_panda", "share_all") == []  # proposed: never shown
    with pytest.raises(ValueError, match="only verified"):
        store.promote(les.id)
    empty_sha = store.sha()
    store.set_verification(les.id, True, {"verdict": "verified"})
    store.set_scope(les.id, Scope(kind="robot", robot="franka_panda"), "checking")
    store.promote(les.id)
    assert store.sha() != empty_sha
    assert [x.id for x in store.for_robot("franka_panda")] == [les.id]
    assert store.for_robot("kinova_gen3", "scoped") == []  # robot-specific: not another arm's
    assert [x.id for x in store.for_robot("kinova_gen3", "share_all")] == [les.id]
    assert store.for_robot("franka_panda", "none") == []
    with pytest.raises(ValueError):
        store.for_robot("franka_panda", "everything")  # type: ignore[arg-type]
    assert any("promoted" in h for h in store.get(les.id).history)


def test_refuted_lessons_stay_out_and_bad_grades_flag_repairs(tmp_path):
    store = LessonStore(tmp_path)
    a, b = _lesson(store, title="a"), _lesson(store, title="b")
    store.set_verification(a.id, False, {"verdict": "refuted"})
    assert admit(store, a.id).status == "refuted"
    store.set_verification(b.id, True, {})
    store.set_scope(b.id, Scope(kind="general"), "agent")
    store.promote(b.id)
    sha = store.sha()
    store.add_grade(b.id, Grade(robot="kinova_gen3", task_id="t", verdict="misleading", evidence="seeds 3, 7 slipped"))
    assert [x.id for x in store.needs_repair()] == [b.id]
    assert store.sha() == sha  # grades do not change what a reader is taught


def test_a_repair_retires_the_lesson_it_replaces(tmp_path):
    store = LessonStore(tmp_path)
    old = _lesson(store, title="old")
    store.set_verification(old.id, True, {})
    admit(store, old.id, "structure")
    new = _lesson(store, title="new", replaces=old.id)
    store.set_verification(new.id, True, {})
    admit(store, new.id, "structure")
    assert store.get(old.id).status == "retired"
    assert [x.id for x in store.promoted()] == [new.id]


def test_render_has_trigger_numbers_scope_and_evidence(tmp_path):
    store = LessonStore(tmp_path)
    les = _lesson(store, code="self.carry_accel = 0.00012")
    les.scope = Scope(kind="robot", robot="franka_panda")
    text = render([les], robot="kinova_gen3")
    for s in ("## transport", "[robot:franka_panda (not your robot)]", "When: a cylinder slides",
              "upright_accel = 0.00012", "measured on franka_panda", "```python"):
        assert s in text, s
    assert render([]) == ""


def test_parse_lesson_report_tolerates_a_code_fence():
    doc = {"consulted": [{"lesson": "L001-x", "verdict": "useful", "evidence": "seed 4"}],
           "proposal": {"title": "t", "trigger": "x", "strategy": "y", "scope": "general"}}
    rep = parse_lesson_report("Here it is:\n```json\n" + json.dumps(doc) + "\n```")
    assert rep.consulted[0].verdict == "useful" and rep.proposal.scope == "general"
    with pytest.raises(ValueError):
        parse_lesson_report("no json here")


# ------------------------------------------------------------------ scope


def test_structural_scope_sees_number_only_changes():
    a = 'class S:\n    """doc"""\n    version = "0.1.0"\n    accel = 0.0003\n    def f(self):\n        return -2 * self.accel\n'
    b = a.replace("0.0003", "0.00012").replace("-2", "-3").replace('"0.1.0"', '"0.1.1"').replace('"""doc"""', '"""new doc"""')
    c = a.replace("return -2 * self.accel", "return max(0.0, -2 * self.accel)")
    assert structural_scope(a, b) == "parameter-only"
    assert structural_scope(a, c) == "code"
    assert structural_scope(a, "def (") == "code"
    assert changed_numbers(a, b) == {"accel": (0.0003, 0.00012)}


def test_check_verdicts_and_scope_methods(tmp_path):
    base = {s: s < 10 for s in range(20)}  # base solves seeds 0-9
    gain = {s: s < 16 for s in range(20)}  # +6/-0
    loss = {s: s < 4 for s in range(20)}  # -6
    assert check_from_outcomes("kinova", "t", "b", "a", base, gain).verdict == "helps"
    assert check_from_outcomes("kinova", "t", "b", "a", base, loss).verdict == "hurts"
    assert check_from_outcomes("kinova", "t", "b", "a", base, base).verdict == "neutral"
    assert check_from_outcomes("kinova", "t", "b", None, base, None).verdict == "not_applied"

    store = LessonStore(tmp_path)
    les = _lesson(store, agent_scope=Scope(kind="general"), structural_scope="parameter-only")
    assert decide_scope(les, "agent").kind == "general"
    assert decide_scope(les, "structure") == Scope(kind="robot", robot="franka_panda")
    assert decide_scope(les, "checking").kind == "robot"  # not checked on any other robot: stays put
    les.scope_checks.append(check_from_outcomes("kinova", "t", "b", "a", base, gain))
    assert decide_scope(les, "checking").kind == "general"
    les.scope_checks.append(check_from_outcomes("panda_locked", "t", "b", "a", base, loss))
    assert decide_scope(les, "checking").kind == "robot"  # one arm lost: robot-specific
    assert label_confusion([les]) == {"agent=general,checked=robot": 1}
    with pytest.raises(ValueError):
        decide_scope(les, "vibes")


# ------------------------------------------------------------------ the loop and curation


class LessonImprover(FakeImprover):
    """A FakeImprover that, like ClaudeCodeImprover, takes lessons and writes a lesson report."""

    def __init__(self, script, report: dict | None = None):
        super().__init__(script)
        self.lessons_text = ""
        self.directive = ""
        self.lesson_report = None
        self.seen: list[str] = []
        self._report = report

    def propose(self, task, current, report):
        self.seen.append(self.lessons_text + self.directive)
        cand = super().propose(task, current, report)
        if self._report is not None:
            self.lesson_report = LessonReport.model_validate(self._report)
        return cand


PROPOSAL = {
    "title": "Do not give up on the far side", "topic": "grasp", "failure_signature": ["timeout"],
    "trigger": "the skill holds still when the cube starts at negative y", "strategy": "always attempt the pick",
    "code": "return super().act(obs)", "scope": "general", "scope_reason": "any arm can reach both sides",
}


def test_loop_shows_lessons_files_grades_and_proposes(stub_task, tmp_path):
    store = LessonStore(tmp_path / "lessons")
    robot = stub_task.scene.robot.name
    shown = _lesson(store, robot=robot, title="shown")
    store.set_verification(shown.id, True, {})
    admit(store, shown.id, "agent")  # agent_scope None -> robot-specific to this robot: shown here
    report = {"consulted": [{"lesson": shown.id, "verdict": "not-applicable"}, {"lesson": "L999-gone", "verdict": "useful"}],
              "proposal": PROPOSAL}
    imp = LessonImprover([GOOD], report)
    hist = tmp_path / "history.json"
    res = run_loop(stub_task, src(PARTIAL), imp, make_env, rounds=2, rollouts=20, target=1.01,
                   lessons=store, history_path=hist)
    r1 = res.rounds[1]
    assert r1.accepted and shown.title in imp.seen[0]
    assert r1.lessons_shown == [shown.id] and r1.lessons_sha == store.sha([store.get(shown.id)])
    les = store.get(r1.lesson_proposed)
    assert les.status == "proposed" and les.agent_scope == Scope(kind="general")
    assert les.structural_scope == "code" and les.source.robot == robot
    assert les.source.parent_version == "0.1.0" and les.source.gate["cand_only"] > 0
    assert "+from skillforge.skills.pick_place_v01" in les.source.diff or "@@" in les.source.diff
    assert [g.verdict for g in store.get(shown.id).grades] == ["not-applicable"]  # unknown ids are dropped
    doc = json.loads(hist.read_text())
    assert doc["rounds"][1]["lesson_proposed"] == les.id and doc["rounds"][1]["lessons_shown"] == [shown.id]


def test_rejected_round_files_grades_but_proposes_nothing(stub_task, tmp_path):
    store = LessonStore(tmp_path)
    les = _lesson(store, robot=stub_task.scene.robot.name)
    store.set_verification(les.id, True, {})
    admit(store, les.id, "agent")
    imp = LessonImprover([NOOP], {"consulted": [{"lesson": les.id, "verdict": "misleading", "evidence": "all seeds hung"}],
                                  "proposal": PROPOSAL})
    res = run_loop(stub_task, src(PARTIAL), imp, make_env, rounds=2, rollouts=10, target=1.01, lessons=store)
    assert not res.rounds[1].accepted and res.rounds[1].lesson_proposed is None
    assert [x.id for x in store.needs_repair()] == [les.id]


def test_loop_without_a_store_is_unchanged(stub_task):
    res = run_loop(stub_task, src(PARTIAL), LessonImprover([GOOD], {"proposal": PROPOSAL}), make_env,
                   rounds=2, rollouts=10, target=1.01)
    assert res.rounds[1].lessons_sha is None and res.rounds[1].lesson_proposed is None


def _proposed(store, stub_task):
    from skillforge.agents.library import paired_compare

    rep = LessonReport.model_validate({"proposal": PROPOSAL})
    cmp = paired_compare({0: False, 1: False}, {0: True, 1: True})
    return record_round(store, rep, task=stub_task, robot="franka_panda", parent=src(PARTIAL),
                        cand=src(GOOD, "0.1.1"), accepted=True, cmp=cmp, success_before=0.5,
                        success_after=1.0, library_sha="")


def test_verifier_reproduces_or_refutes(stub_task, tmp_path):
    store = LessonStore(tmp_path)
    seeds = range(1000, 1020)
    les = _proposed(store, stub_task)
    imps: list[LessonImprover] = []

    def good():
        imps.append(LessonImprover([GOOD]))
        return imps[-1]

    out = verify_lesson(store, les.id, task=stub_task, parent=src(PARTIAL), make_env=make_env,
                        improver_factory=good, seeds=seeds)
    assert out.status == "verified" and out.verification["attempts_used"] == 1
    assert les.title in imps[0].seen[0] and "not seen the code" in imps[0].seen[0]

    les2 = _proposed(store, stub_task)
    out2 = verify_lesson(store, les2.id, task=stub_task, parent=src(PARTIAL), make_env=make_env,
                         improver_factory=lambda: LessonImprover([NOOP, NOOP]), seeds=seeds)
    assert out2.status == "refuted" and out2.verification["attempts_used"] == 2


def test_check_scope_then_admit(stub_task, tmp_path):
    store = LessonStore(tmp_path)
    les = _proposed(store, stub_task)
    store.set_verification(les.id, True, {})
    targets = [Target("panda_locked", stub_task, src(PARTIAL)), Target("kinova_gen3", stub_task, src(GOOD))]
    scripts = iter([[GOOD], [NOOP]])  # helps the first arm, breaks the second
    out = check_scope(store, les.id, targets, make_env=make_env,
                      improver_factory=lambda: LessonImprover(next(scripts)), seeds=range(1000, 1020))
    assert [(c.robot, c.verdict) for c in out.scope_checks] == [("panda_locked", "helps"), ("kinova_gen3", "hurts")]
    promoted = admit(store, les.id, "checking")
    assert promoted.status == "promoted" and promoted.scope == Scope(kind="robot", robot="franka_panda")
    assert label_confusion(store.all()) == {"agent=general,checked=robot": 1}


def test_cli_lessons_list_show_admit(stub_task, tmp_path, capsys):
    from skillforge.cli import main

    store = LessonStore(tmp_path)
    les = _proposed(store, stub_task)
    assert main(["lessons", "--dir", str(tmp_path), "admit", les.id]) == 1  # not verified yet
    store.set_verification(les.id, True, {})
    assert main(["lessons", "--dir", str(tmp_path), "admit", les.id, "--method", "agent"]) == 0
    assert main(["lessons", "--dir", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "promoted" in out and "scope general" in out
    assert main(["lessons", "--dir", str(tmp_path), "show", les.id]) == 0
    assert json.loads(capsys.readouterr().out)["id"] == les.id


def test_cli_forge_files_lessons(stub_task, tmp_path):
    from pathlib import Path

    from skillforge.cli import main

    task_file = Path(__file__).resolve().parents[1] / "tasks" / "stub_pick_cube_to_bin.json"
    rc = main(["forge", str(task_file), "--seed-skill", "pick_place_v01", "--improver", "fake", "--rounds", "2",
               "--rollouts", "5", "--target", "1.01", "--library", str(tmp_path / "lib"), "--out", str(tmp_path / "out"),
               "--lessons", str(tmp_path / "lessons"), "--lesson-policy", "scoped"])
    assert rc == 0
    hist = json.loads(next((tmp_path / "out").glob("*/history.json")).read_text())
    assert hist["meta"]["lessons"] == str(tmp_path / "lessons") and hist["rounds"][-1]["lessons_sha"]
