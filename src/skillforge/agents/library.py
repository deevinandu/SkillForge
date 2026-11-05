"""Versioned skill store with lineage and the keep-the-best acceptance gates.

Gates: 'paired' (default; `passes_paired_gate`, a one-sided exact McNemar test on the gate seeds
where the candidate and the best disagree) and 'ci' (legacy; `passes_gate`).

Layout: `<root>/<task_id>/<version>/{skill.py, report.json, rationale.md, meta.json}`.
Every evaluated version is stored, accepted or not, so the lineage includes the dead ends.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from math import comb
from pathlib import Path

from skillforge.contracts import EvalReport, SkillSource

_SAFE = re.compile(r"^[A-Za-z0-9_.+\-]+$")


GATE_RULES = ("paired", "ci")
DEFAULT_ALPHA = 0.1


def passes_gate(best: EvalReport | None, candidate: EvalReport) -> bool:
    """Legacy rule ('ci'): reject only a version whose success rate is below the best's 95% CI
    lower bound. It accepts versions that are no better than the best (within noise)."""
    return best is None or candidate.success_rate >= best.ci95[0]


@dataclass(frozen=True)
class PairedComparison:
    """Per-seed comparison of a candidate against the best, both scored on the SAME seeds."""

    n: int
    both: int  # seeds both solve
    best_only: int  # seeds only the best solves (candidate regressed there)
    cand_only: int  # seeds only the candidate solves
    neither: int
    p_value: float  # one-sided exact McNemar: P(X >= cand_only), X ~ Binom(best_only + cand_only, 1/2)

    def as_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        return f"+{self.cand_only}/-{self.best_only} p={self.p_value:.3g}"


def mcnemar_one_sided(best_only: int, cand_only: int) -> float:
    """Exact one-sided McNemar p-value for 'the candidate is better'.

    Under H0 (both versions are equally good), each discordant seed is equally likely to favour
    either one, so cand_only ~ Binomial(best_only + cand_only, 1/2). Concordant seeds carry no
    information about the difference and are ignored. No discordant seeds gives p = 1.
    """
    n = best_only + cand_only
    if n == 0:
        return 1.0
    return min(1.0, sum(comb(n, k) for k in range(cand_only, n + 1)) / 2**n)


def paired_compare(best: Mapping[int, bool], cand: Mapping[int, bool]) -> PairedComparison:
    """Seed-by-seed counts. Both maps must cover the same seeds (the loop's gate seeds)."""
    if set(best) != set(cand):
        raise ValueError(f"paired comparison needs the same seeds: {len(best)} vs {len(cand)} seeds, sets differ")
    both = sum(1 for s in best if best[s] and cand[s])
    bo = sum(1 for s in best if best[s] and not cand[s])
    co = sum(1 for s in best if cand[s] and not best[s])
    return PairedComparison(len(best), both, bo, co, len(best) - both - bo - co, mcnemar_one_sided(bo, co))


def passes_paired_gate(
    best: Mapping[int, bool] | None, cand: Mapping[int, bool], alpha: float = DEFAULT_ALPHA
) -> tuple[bool, PairedComparison | None]:
    """Default rule ('paired'): accept only if the candidate solves MORE gate seeds than the best
    AND the one-sided exact McNemar test on the discordant seeds gives p < alpha.

    With 100 seeds and alpha 0.1 this needs, e.g., at least 4 net wins with no losses (+4/-0,
    p 0.0625), +7/-1 (p 0.035) or +11/-4 (p 0.059); +11/-7 (p 0.24) is rejected.
    """
    if best is None:
        return True, None
    cmp = paired_compare(best, cand)
    return (cmp.cand_only > cmp.best_only and cmp.p_value < alpha), cmp


def outcomes_from_report(report: EvalReport, seeds: Sequence[int]) -> dict[int, bool]:
    """Per-seed success recovered from a report, if its failure list is complete (it is a sample
    capped at `max_failures`, so this only works when every failing seed is listed)."""
    failing = {f.seed for f in report.failures}
    if len(failing) != report.n_rollouts - report.n_success or report.n_rollouts != len(seeds):
        raise ValueError(
            f"report lists {len(failing)} failing seeds but has {report.n_rollouts - report.n_success} failures; "
            "re-evaluate with max_failures >= n_rollouts"
        )
    return {int(s): s not in failing for s in seeds}


def bump_version(v: str) -> str:
    """'0.1.0' -> '0.1.1'; anything not ending in a number gets '.1' appended."""
    m = re.match(r"^(.*?)(\d+)$", v)
    return f"{m.group(1)}{int(m.group(2)) + 1}" if m else f"{v}.1"


class SkillLibrary:
    def __init__(self, root: str | Path = "runs/library"):
        self.root = Path(root)

    def _dir(self, task_id: str, version: str) -> Path:
        if not _SAFE.match(version) or not _SAFE.match(task_id):
            raise ValueError(f"unsafe task id or version for a path: {task_id!r} / {version!r}")
        return self.root / task_id / version

    def versions(self, task_id: str) -> list[str]:
        d = self.root / task_id
        if not d.is_dir():
            return []
        return sorted(p.name for p in d.iterdir() if (p / "meta.json").exists())

    def has(self, task_id: str, version: str) -> bool:
        return (self._dir(task_id, version) / "meta.json").exists()

    def unique_version(self, task_id: str, proposed: str) -> str:
        v = proposed if _SAFE.match(proposed) else re.sub(r"[^A-Za-z0-9_.+\-]", "_", proposed)
        while self.has(task_id, v):
            v = bump_version(v)
        return v

    def add(self, src: SkillSource, report: EvalReport, accepted: bool, note: str = "") -> Path:
        d = self._dir(report.task_id, src.version)
        if d.exists():
            raise FileExistsError(f"{d} already exists; versions are immutable")
        seq = len(self.versions(report.task_id))
        d.mkdir(parents=True)
        (d / "skill.py").write_text(src.code)
        (d / "report.json").write_text(report.model_dump_json(indent=2))
        (d / "rationale.md").write_text((src.rationale or "(no rationale)").rstrip() + "\n")
        meta = {
            "name": src.name,
            "version": src.version,
            "entrypoint": src.entrypoint,
            "parent_version": src.parent_version,
            "accepted": accepted,
            "seq": seq,
            "note": note,
            "success_rate": report.success_rate,
            "ci95": list(report.ci95),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        return d

    def meta(self, task_id: str, version: str) -> dict:
        return json.loads((self._dir(task_id, version) / "meta.json").read_text())

    def load(self, task_id: str, version: str) -> tuple[SkillSource, EvalReport]:
        d = self._dir(task_id, version)
        m = self.meta(task_id, version)
        src = SkillSource(
            name=m["name"],
            version=m["version"],
            code=(d / "skill.py").read_text(),
            entrypoint=m["entrypoint"],
            rationale=(d / "rationale.md").read_text().strip(),
            parent_version=m["parent_version"],
        )
        return src, EvalReport.model_validate_json((d / "report.json").read_text())

    def best(self, task_id: str) -> tuple[SkillSource, EvalReport] | None:
        """Highest success rate among accepted versions (ties: the earliest-added one)."""
        acc = [m for v in self.versions(task_id) if (m := self.meta(task_id, v))["accepted"]]
        if not acc:
            return None
        top = max(acc, key=lambda m: (m["success_rate"], -m.get("seq", 0)))
        return self.load(task_id, top["version"])

    def lineage(self, task_id: str, version: str) -> list[str]:
        """[version, parent, grandparent, ...] as far as the library knows."""
        out: list[str] = []
        v: str | None = version
        while v and self.has(task_id, v) and v not in out:
            out.append(v)
            v = self.meta(task_id, v)["parent_version"]
        return out
