"""Deciding a lesson's scope: general (should help every arm) or robot-specific.

PROJECT.md names three ways, and the experiment compares them, so all three are kept:
- `structure`: a repair that only moves numbers (`structural_scope` -> "parameter-only") is a
  tuning for the arm it was measured on; any other change is general;
- `agent`: the proposer's own label;
- `checking`: the lesson applied to other robots' skills and scored paired there
  (`agents/curate.check_scope`). Any significant loss makes it robot-specific; it becomes general
  only once at least one other robot was checked and none lost.
"""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Iterable, Mapping

from .lessons import Lesson, Scope, ScopeCheck
from .library import DEFAULT_ALPHA, mcnemar_one_sided, paired_compare

METHODS: tuple[str, ...] = ("checking", "structure", "agent")


# ------------------------------------------------------------------ structural


class _Numbers(ast.NodeTransformer):
    """Replace every numeric literal (and a unary minus on one) with a placeholder."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Name(id="__NUM__", ctx=ast.Load()), node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand, ast.Name) and node.operand.id == "__NUM__":
            return node.operand
        return node


def _shape(code: str) -> str:
    tree = ast.parse(code)
    # docstrings, version strings and comments are not behaviour
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and isinstance(
            getattr(body[0], "value", None), ast.Constant
        ) and isinstance(body[0].value.value, str):
            body.pop(0)
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "version" for t in node.targets):
            node.value = ast.Constant(value="")
    return ast.dump(_Numbers().visit(tree), annotate_fields=False)


def structural_scope(parent_code: str, candidate_code: str) -> str:
    """'parameter-only' if the candidate differs from its parent only in numeric literals
    (besides docstrings and the version string), else 'code'. Unparseable code counts as 'code'."""
    try:
        return "parameter-only" if _shape(parent_code) == _shape(candidate_code) else "code"
    except SyntaxError:
        return "code"


def changed_numbers(parent_code: str, candidate_code: str) -> dict[str, tuple[float, float]]:
    """Class or module constants (`name = <number>`) whose value changed: name -> (old, new)."""

    def consts(code: str) -> dict[str, float]:
        out: dict[str, float] = {}
        for node in ast.walk(ast.parse(code)):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    v = ast.literal_eval(node.value)
                except (ValueError, SyntaxError, TypeError):
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    out[node.targets[0].id] = float(v)
        return out

    try:
        a, b = consts(parent_code), consts(candidate_code)
    except SyntaxError:
        return {}
    return {k: (a[k], b[k]) for k in a.keys() & b.keys() if a[k] != b[k]}


# ------------------------------------------------------------------ checking


def check_from_outcomes(
    robot: str, task_id: str, base_version: str, applied_version: str | None,
    base: Mapping[int, bool], applied: Mapping[int, bool] | None, alpha: float = DEFAULT_ALPHA, note: str = "",
) -> ScopeCheck:
    """A ScopeCheck from per-seed outcomes of the base skill and the skill with the lesson applied
    (same seeds). `applied=None` means the lesson could not be applied to that robot's skill."""
    if applied is None:
        return ScopeCheck(robot=robot, task_id=task_id, base_version=base_version, applied_version=None,
                          verdict="not_applied", note=note)
    c = paired_compare(base, applied)
    p_gain = mcnemar_one_sided(c.best_only, c.cand_only)
    p_loss = mcnemar_one_sided(c.cand_only, c.best_only)
    if c.best_only > c.cand_only and p_loss < alpha:
        verdict = "hurts"
    elif c.cand_only > c.best_only and p_gain < alpha:
        verdict = "helps"
    else:
        verdict = "neutral"
    return ScopeCheck(robot=robot, task_id=task_id, base_version=base_version, applied_version=applied_version,
                      n=c.n, both=c.both, best_only=c.best_only, cand_only=c.cand_only, neither=c.neither,
                      p_gain=p_gain, p_loss=p_loss, verdict=verdict, note=note)


def decide_scope(lesson: Lesson, method: str = "checking") -> Scope:
    """The lesson's scope under one of METHODS. Robot-specific means the source robot's."""
    own = Scope(kind="robot", robot=lesson.source.robot)
    if method == "structure":
        return own if lesson.structural_scope == "parameter-only" else Scope(kind="general")
    if method == "agent":
        return lesson.agent_scope or own
    if method == "checking":
        others = [c for c in lesson.scope_checks if c.robot != lesson.source.robot and c.verdict != "not_applied"]
        if not others or any(c.verdict == "hurts" for c in others):
            return own
        return Scope(kind="general")
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def label_confusion(lessons: Iterable[Lesson]) -> dict[str, int]:
    """Scope-label error: the proposer's label against what checking found, over lessons that were
    checked on at least one other robot. Keys 'agent=<a>,checked=<c>' with a, c in general/robot."""
    out: Counter[str] = Counter()
    for les in lessons:
        if les.agent_scope is None:
            continue
        if not [c for c in les.scope_checks if c.robot != les.source.robot and c.verdict != "not_applied"]:
            continue
        found = decide_scope(les, "checking")
        out[f"agent={les.agent_scope.kind},checked={found.kind}"] += 1
    return dict(out)


def checks_by_robot(checks: Iterable[ScopeCheck]) -> dict[str, list[ScopeCheck]]:
    out: dict[str, list[ScopeCheck]] = {}
    for c in checks:
        out.setdefault(c.robot, []).append(c)
    return out
