"""Subprocess entry point for running untrusted skill code. Never import this in the parent.

Protocol: `python -I -m skillforge.agents._sandbox_worker <request.json> <result.json>`.
The request names a mode ('check' or 'eval'); the result is written to a file (not stdout,
which the skill may print to). Exit code 0 means the result file holds a valid answer.
"""

from __future__ import annotations

import importlib
import json
import socket
import sys
import traceback


def _no_network() -> None:
    """Best-effort: make socket creation and DNS fail in this process. Not OS-level isolation."""

    def deny(*_a, **_k):
        raise PermissionError("network access is disabled in the skill sandbox")

    class _Socket(socket.socket):
        def __init__(self, *a, **k):
            deny()

    socket.socket = _Socket  # type: ignore[misc]
    socket.create_connection = deny  # type: ignore[assignment]
    socket.getaddrinfo = deny  # type: ignore[assignment]
    socket.socketpair = deny  # type: ignore[assignment]


def _limits(max_memory_bytes: int | None) -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if max_memory_bytes:
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
    except (ImportError, ValueError, OSError):
        pass  # not on POSIX, or the limit is not allowed here; the wall-clock timeout still applies


def _resolve(ref: str):
    mod, _, attr = ref.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def main(req_path: str, out_path: str) -> int:
    req = json.loads(open(req_path).read())
    _limits(req.get("max_memory_bytes"))

    # Import trusted code first, then shut the network, then touch agent code.
    from skillforge.contracts import SkillSource, TaskSpec
    from skillforge.eval import evaluate
    from skillforge.skills import load_skill

    task = TaskSpec.model_validate(req["task"])
    src = SkillSource.model_validate(req["source"])
    make_env = _resolve(req["env_ref"])
    _no_network()

    def write(obj: dict) -> None:
        with open(out_path, "w") as f:
            json.dump(obj, f)

    try:
        skill = load_skill(src)
        # SkillSource is the source of truth for identity, so reports and the library agree.
        skill.name, skill.version = src.name, src.version
    except BaseException:
        write({"ok": False, "stage": "load", "error": traceback.format_exc(limit=8)})
        return 0

    if req["mode"] == "check":
        try:
            env = make_env(task)
            try:
                obs = env.reset(0)
                skill.reset(task)
                for _ in range(int(req.get("smoke_steps", 5))):
                    obs, _info = env.step(skill.act(obs))
            finally:
                env.close()
        except BaseException:
            write({"ok": False, "stage": "smoke", "error": traceback.format_exc(limit=8)})
            return 0
        write({"ok": True, "stage": "check"})
        return 0

    report, eps = evaluate(make_env, skill, task, req["seeds"], max_failures=int(req.get("max_failures", 20)))
    outcomes = [[int(e.seed), bool(e.success)] for e in eps]  # per seed, for the paired gate
    write({"ok": True, "stage": "eval", "report": json.loads(report.model_dump_json()), "outcomes": outcomes})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
