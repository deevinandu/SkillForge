"""Scoped lessons: what a repair taught, as text an improver can reuse, with the robots it is for.

Built on ASPIRE's skill library (text entries: trigger -> code -> evidence, admitted only after a
fresh verifier reproduces the fix, graded by every later reader), plus a scope: `general` (a
strategy that should help every arm) or `robot:<name>` (numbers or moves that suit one arm).
Design and the mapping to ASPIRE: docs/LESSONS.md.

Layout: `<root>/<lesson id>.json`, one file per lesson, every status kept (refuted and retired
lessons too, so the record shows the dead ends). Only `promoted` lessons are ever shown to an
improver, filtered by `for_robot(robot, policy)`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

LessonStatus = Literal["proposed", "verified", "refuted", "promoted", "retired"]
GradeVerdict = Literal["useful", "misleading", "wrong", "not-applicable"]
CheckVerdict = Literal["helps", "neutral", "hurts", "not_applied"]
Policy = Literal["scoped", "share_all", "none"]
POLICIES: tuple[str, ...] = ("scoped", "share_all", "none")
TOPICS: tuple[str, ...] = ("grasp", "transport", "place", "recover", "perception", "control", "other")

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Scope(BaseModel):
    """`general`, or `robot` with the robot's name (`RobotSpec.name`)."""

    kind: Literal["general", "robot"]
    robot: str | None = None

    def label(self) -> str:
        return "general" if self.kind == "general" else f"robot:{self.robot}"

    @classmethod
    def parse(cls, s: str) -> Scope:
        s = s.strip()
        if s == "general":
            return cls(kind="general")
        if s.startswith("robot:") and len(s) > 6:
            return cls(kind="robot", robot=s[6:])
        raise ValueError(f"scope must be 'general' or 'robot:<name>', got {s!r}")

    def applies_to(self, robot: str) -> bool:
        return self.kind == "general" or self.robot == robot


class Provenance(BaseModel):
    """Where a lesson came from: the accepted repair it describes."""

    robot: str
    task_id: str
    skill: str
    parent_version: str | None
    version: str
    gate: dict | None = None  # PairedComparison.as_dict() of the repair vs its parent
    success_before: float | None = None
    success_after: float | None = None
    heldout: dict | None = None
    diff: str = ""
    created_at: str = Field(default_factory=_now)


class ScopeCheck(BaseModel):
    """The lesson applied to another robot's skill, scored paired against that skill."""

    robot: str
    task_id: str
    base_version: str
    applied_version: str | None
    n: int = 0
    both: int = 0
    best_only: int = 0  # seeds only the base solves: the lesson cost these
    cand_only: int = 0  # seeds only the lesson's version solves
    neither: int = 0
    p_gain: float = 1.0  # one-sided McNemar for "the lesson helps"
    p_loss: float = 1.0  # one-sided McNemar for "the lesson hurts"
    verdict: CheckVerdict
    note: str = ""
    at: str = Field(default_factory=_now)


class Grade(BaseModel):
    """A later reader's verdict on a lesson it was shown (ASPIRE's skill report)."""

    robot: str
    task_id: str
    verdict: GradeVerdict
    evidence: str = ""
    library_sha: str = ""
    at: str = Field(default_factory=_now)


class Lesson(BaseModel):
    id: str
    title: str
    topic: str = "other"
    failure_signature: list[str] = Field(default_factory=list)  # failure tags / event names it fixes
    trigger: str  # the symptom or scene condition that calls for it
    precondition: str = ""  # a check to run first, when the trigger alone is too broad
    strategy: str  # what to do, in words a stranger can follow
    code: str = ""  # a 5-20 line sketch from the working skill
    params: dict[str, float] = Field(default_factory=dict)  # the numbers it sets, measured on source.robot
    source: Provenance
    agent_scope: Scope | None = None  # the proposer's label
    agent_reason: str = ""
    structural_scope: Literal["parameter-only", "code"] | None = None
    scope: Scope | None = None  # the decided scope; None until decided
    scope_method: str = ""
    scope_checks: list[ScopeCheck] = Field(default_factory=list)
    status: LessonStatus = "proposed"
    verification: dict | None = None
    grades: list[Grade] = Field(default_factory=list)
    replaces: str | None = None  # id of a lesson this one repairs (that one is retired)
    history: list[str] = Field(default_factory=list)  # "<time> <what happened>"

    def log(self, what: str) -> None:
        self.history.append(f"{_now()} {what}")

    def bad_grades(self) -> list[Grade]:
        return [g for g in self.grades if g.verdict in ("misleading", "wrong")]


class LessonStore:
    def __init__(self, root: str | Path = "runs/lessons"):
        self.root = Path(root)

    # ------------------------------------------------------------------ storage

    def _path(self, lesson_id: str) -> Path:
        if not _SAFE.match(lesson_id):
            raise ValueError(f"unsafe lesson id {lesson_id!r}")
        return self.root / f"{lesson_id}.json"

    def save(self, lesson: Lesson) -> Lesson:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self._path(lesson.id).with_suffix(".json.tmp")
        tmp.write_text(lesson.model_dump_json(indent=2))
        tmp.replace(self._path(lesson.id))
        return lesson

    def get(self, lesson_id: str) -> Lesson:
        return Lesson.model_validate_json(self._path(lesson_id).read_text())

    def all(self) -> list[Lesson]:
        if not self.root.is_dir():
            return []
        return [Lesson.model_validate_json(p.read_text()) for p in sorted(self.root.glob("*.json"))]

    def new_id(self, title: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40] or "lesson"
        n = len(list(self.root.glob("*.json"))) if self.root.is_dir() else 0
        i = n + 1
        while (self.root / f"L{i:03d}-{slug}.json").exists():
            i += 1
        return f"L{i:03d}-{slug}"

    # ------------------------------------------------------------------ lifecycle

    def propose(self, lesson: Lesson) -> Lesson:
        if lesson.status != "proposed":
            raise ValueError("a new lesson starts as 'proposed'")
        if self._path(lesson.id).exists():
            raise FileExistsError(f"lesson {lesson.id} exists")
        lesson.log(f"proposed from {lesson.source.robot}/{lesson.source.task_id} {lesson.source.version}")
        return self.save(lesson)

    def set_verification(self, lesson_id: str, verified: bool, record: dict) -> Lesson:
        les = self.get(lesson_id)
        if les.status != "proposed":
            raise ValueError(f"{lesson_id} is {les.status}, not proposed")
        les.status = "verified" if verified else "refuted"
        les.verification = record
        les.log(les.status)
        return self.save(les)

    def add_scope_check(self, lesson_id: str, check: ScopeCheck) -> Lesson:
        les = self.get(lesson_id)
        les.scope_checks.append(check)
        les.log(f"checked on {check.robot}/{check.task_id}: {check.verdict}")
        return self.save(les)

    def set_scope(self, lesson_id: str, scope: Scope, method: str) -> Lesson:
        les = self.get(lesson_id)
        if les.scope is None or les.scope != scope:
            les.log(f"scope {scope.label()} by {method}")
        les.scope, les.scope_method = scope, method
        return self.save(les)

    def promote(self, lesson_id: str) -> Lesson:
        """Only a verified lesson with a decided scope enters the library."""
        les = self.get(lesson_id)
        if les.status != "verified":
            raise ValueError(f"{lesson_id} is {les.status}; only verified lessons are promoted")
        if les.scope is None:
            raise ValueError(f"{lesson_id} has no decided scope")
        les.status = "promoted"
        les.log("promoted")
        if les.replaces:
            self.retire(les.replaces, f"replaced by {les.id}")
        return self.save(les)

    def retire(self, lesson_id: str, why: str) -> Lesson:
        les = self.get(lesson_id)
        les.status = "retired"
        les.log(f"retired: {why}")
        return self.save(les)

    def add_grade(self, lesson_id: str, grade: Grade) -> Lesson:
        les = self.get(lesson_id)
        les.grades.append(grade)
        return self.save(les)

    # ------------------------------------------------------------------ retrieval

    def promoted(self) -> list[Lesson]:
        return [les for les in self.all() if les.status == "promoted"]

    def for_robot(self, robot: str, policy: Policy = "scoped") -> list[Lesson]:
        """The lessons an improver working on `robot` is shown, under one experiment condition:
        - `scoped`: general lessons and this robot's own;
        - `share_all`: every promoted lesson, whatever its scope;
        - `none`: nothing (from scratch)."""
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
        if policy == "none":
            return []
        lessons = self.promoted()
        if policy == "share_all":
            return lessons
        return [les for les in lessons if les.scope is not None and les.scope.applies_to(robot)]

    def needs_repair(self) -> list[Lesson]:
        """Promoted lessons a reader graded `wrong` or `misleading` (ASPIRE: repairs before additions)."""
        return [les for les in self.promoted() if les.bad_grades()]

    def sha(self, lessons: Iterable[Lesson] | None = None) -> str:
        """Content hash of the promoted library (or of `lessons`), recorded with every round."""
        h = hashlib.sha256()
        for les in sorted(self.promoted() if lessons is None else lessons, key=lambda x: x.id):
            h.update(json.dumps(_teaching(les), sort_keys=True).encode())
        return h.hexdigest()[:12]


def _teaching(les: Lesson) -> dict:
    """The part of a lesson that a reader sees (grades and checks change; the teaching does not)."""
    return {
        "id": les.id, "title": les.title, "topic": les.topic, "trigger": les.trigger,
        "precondition": les.precondition, "strategy": les.strategy, "code": les.code,
        "params": les.params, "scope": les.scope.label() if les.scope else None,
    }


def render(lessons: Iterable[Lesson], robot: str | None = None) -> str:
    """Markdown for an improver's prompt, grouped by topic: trigger -> strategy -> code -> evidence."""
    lessons = list(lessons)
    if not lessons:
        return ""
    out = ["# Lessons from earlier repairs", "",
           "Each lesson was verified: a fresh agent reproduced its gain from the text alone. Scope says",
           "which robots it is for. Grade every lesson you read in LESSONS.json (see the instructions).", ""]
    for topic in sorted({les.topic for les in lessons}, key=lambda t: (TOPICS.index(t) if t in TOPICS else 99, t)):
        out += [f"## {topic}", ""]
        for les in sorted((x for x in lessons if x.topic == topic), key=lambda x: x.id):
            scope = les.scope.label() if les.scope else "undecided"
            mark = "" if robot is None or les.scope is None or les.scope.applies_to(robot) else " (not your robot)"
            out.append(f"### {les.id}: {les.title}  [{scope}{mark}]")
            if les.failure_signature:
                out.append(f"- Fixes: {', '.join(les.failure_signature)}")
            out.append(f"- When: {les.trigger}")
            if les.precondition:
                out.append(f"- First check: {les.precondition}")
            out.append(f"- Do: {les.strategy}")
            if les.params:
                vals = ", ".join(f"{k} = {v:g}" for k, v in sorted(les.params.items()))
                out.append(f"- Numbers (measured on {les.source.robot}): {vals}")
            s = les.source
            ev = f"- Evidence: {s.robot}/{s.task_id} {s.parent_version} -> {s.version}"
            if s.success_before is not None and s.success_after is not None:
                ev += f", {s.success_before:.0%} -> {s.success_after:.0%}"
            if s.gate:
                ev += f" (paired +{s.gate.get('cand_only')}/-{s.gate.get('best_only')}, p={s.gate.get('p_value', 1):.3g})"
            out.append(ev)
            for c in les.scope_checks:
                out.append(f"- Checked on {c.robot}/{c.task_id}: {c.verdict} (+{c.cand_only}/-{c.best_only})")
            useful = sum(g.verdict == "useful" for g in les.grades)
            bad = len(les.bad_grades())
            if useful or bad:
                out.append(f"- Readers: {useful} useful, {bad} misleading or wrong")
            if les.code.strip():
                out += ["", "```python", les.code.strip("\n"), "```"]
            out.append("")
    return "\n".join(out).rstrip() + "\n"


# ------------------------------------------------------------------ the improver's report

LESSON_REPORT_FILE = "LESSONS.json"

LESSON_REPORT_INSTRUCTIONS = """\
Also write LESSONS.json (it is read by a program; keep to this shape):
{{
  "consulted": [{{"lesson": "<id>", "verdict": "useful|misleading|wrong|not-applicable", "evidence": "<seeds, numbers>"}}],
  "proposal": {{
    "title": "<short name>", "topic": "{topics}",
    "failure_signature": ["<failure tags it fixes>"],
    "trigger": "<the symptom or scene condition that calls for it>",
    "precondition": "<a check to run first, or empty>",
    "strategy": "<what to do, so that someone who never saw your code could do it>",
    "code": "<the 5-20 lines from skill.py that do it>",
    "params": {{"<name>": <number it sets>}},
    "scope": "general|robot:{robot}",
    "scope_reason": "<why: would this help any arm, or does it depend on this arm's kinematics, gripper or controller?>"
  }}
}}
Grade every lesson you were shown (not-applicable is fine). Set "proposal" to null if your change
teaches nothing reusable. `general` means it should help a different arm and gripper too.
"""


class ConsultedGrade(BaseModel):
    lesson: str
    verdict: GradeVerdict
    evidence: str = ""


class LessonProposal(BaseModel):
    title: str
    topic: str = "other"
    failure_signature: list[str] = Field(default_factory=list)
    trigger: str
    precondition: str = ""
    strategy: str
    code: str = ""
    params: dict[str, float] = Field(default_factory=dict)
    scope: str = ""
    scope_reason: str = ""


class LessonReport(BaseModel):
    consulted: list[ConsultedGrade] = Field(default_factory=list)
    proposal: LessonProposal | None = None


def parse_lesson_report(text: str) -> LessonReport:
    """Tolerant of a JSON object inside other text (a model may wrap it in a code fence)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the lesson report")
    return LessonReport.model_validate(json.loads(m.group(0)))
