"""Versioned skill store with lineage and a keep-the-best acceptance gate.

Layout: `<root>/<task_id>/<version>/{skill.py, report.json, rationale.md, meta.json}`.
Every evaluated version is stored, accepted or not, so the lineage includes the dead ends.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from skillforge.contracts import EvalReport, SkillSource

_SAFE = re.compile(r"^[A-Za-z0-9_.+\-]+$")


def passes_gate(best: EvalReport | None, candidate: EvalReport) -> bool:
    """Never accept a version whose success rate is below the best's 95% CI lower bound."""
    return best is None or candidate.success_rate >= best.ci95[0]


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
