"""T5 dataset engine: roll a teacher skill out over many seeds and record what it saw and did.

On-disk format (`skillforge.dataset/1`), LeRobot-like in spirit (episode index + per-episode
arrays + meta), but numpy-only so it needs no extra dependency::

    runs/datasets/<name>/
      meta/info.json        format, task (full TaskSpec + sha), teacher, fps, feature schema
                            (dtype, shape, column names), totals, the config of every
                            collection pass that wrote into this dataset
      meta/episodes.jsonl   one line per episode: episode_index, seed, length, success,
                            failure, source ("teacher", "dart", "dagger:r1", ...), the env's
                            randomization sample, the file path, last events
      meta/stats.json       per-column mean/std/min/max of observation.state and action
      data/chunk-000/episode_000000.npz
                            observation.state  (T, D) float32  state BEFORE step t's action
                            action             (T, 8)  float32 the teacher's (expert) action:
                                                       ee_pos xyz, ee_quat wxyz, gripper
                            action.executed    (T, 8)  float32 what the env actually received
                                                       (differs under DART noise / DAgger)
                            label_valid        (T,)    bool    False where the teacher's label
                                                       is not trustworthy (see below)
                            timestamp          (T,)    float32 seconds since reset
                            frame_index        (T,)    int32

Determinism: an episode is a pure function of (task, teacher, seed, collection config,
learner checkpoint). The env draws its randomization from `seed`; DART noise and DAgger's
expert/learner mixing draw from `numpy.random.default_rng([seed, 7])` and `[seed, 8]`.

**Teacher relabeling (DAgger) and internal state.** PickPlace v0.2 is a phase machine with a
blackboard (filtered object pose, chosen grasp, retry budgets, hold monitor), so "the teacher's
action at state s" is not a function of s alone. We run the teacher *in the shadow* of the
executed trajectory: it receives every observation of the rollout, keeps its own phase state,
and its output is recorded as the label while the learner's action is executed. The label is
therefore the teacher's action given the learner's *history*, which is well-defined and
deterministic but path-dependent: if the learner skips a phase (e.g. closes early) the teacher
may still be waiting in the previous phase, or its hold monitor may declare a miss and replan.
Once the teacher's tree has failed outright (budgets exhausted, so it only parks in place), its
labels say nothing useful; `label_valid` is False from that step on and training skips them.

Measured on the Franka task (see HANDOVER, 2026-11-04 T5): this desync is the common case, not
an edge case. v0.2's phases end when the ee is within 6 mm of a waypoint; a cloned policy cuts
corners (blends hover -> descend), so the shadow teacher never sees its waypoint reached and
keeps labelling "go to hover" while the learner is already carrying the cube to the bin.
Training on those labels took the learner from 296/300 to 5/300 in one round. Hence the
**sync gate** (`sync_tol`, `sync_patience`): once teacher and learner disagree (gripper command
differs, or the clipped position targets differ by more than `sync_tol`) for more than
`sync_patience` consecutive steps, the teacher is considered out of phase and every label from
the start of that disagreement on is marked invalid. What is left is corrective signal near the
learner's own path, which is all a phase-machine expert can give without a state-only oracle.
"""

from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import platform
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from skillforge.contracts import Action, SimEnv, Skill, TaskSpec

from .features import ACTION_NAMES, action_vector, state_names, state_vector

FORMAT = "skillforge.dataset/1"
DEFAULT_TEACHER = "skillforge.skills.pick_place:PickPlace"
DEFAULT_ENV = "skillforge.sim:make_env"


def _is_path_spec(spec: str) -> bool:
    head = spec.rpartition(":")[0] if spec.rpartition(":")[2].isidentifier() else spec
    return head.endswith(".py") or "/" in head or os.sep in head


def _module_from_file(path: Path):
    """Import a single-file module from `path` under a stable, path-derived name.

    The module is registered in `sys.modules` so `inspect.getsource` (used by
    `eval.regression.source_of` to ship the skill to worker processes) and pickling work.
    """
    import hashlib
    import importlib.util
    import sys

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"teacher file not found: {path}")
    stem = "".join(c if c.isalnum() else "_" for c in path.stem)
    name = f"_skillforge_file_{stem}_{hashlib.sha256(str(path).encode()).hexdigest()[:8]}"
    mod = sys.modules.get(name)
    if mod is not None and getattr(mod, "__file__", None) == str(path):
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


def _only_skill_class(mod) -> Any:
    found = [
        v for v in vars(mod).values()
        if isinstance(v, type) and v.__module__ == mod.__name__
        and all(hasattr(v, k) for k in ("name", "version", "reset", "act"))
    ]
    if len(found) != 1:
        names = ", ".join(sorted(c.__name__ for c in found)) or "none"
        raise ValueError(f"{mod.__file__}: expected exactly one Skill class, found {names}; use 'path.py:Class'")
    return found[0]


def load_object(spec: str) -> Any:
    """Resolve a teacher/env spec to the object it names.

    Accepted forms: 'package.module:attr' (importable module), or 'path/to/file.py[:attr]'
    (a single-file module loaded from disk, e.g. an agent-written skill under results/). Without
    ':attr' a file must define exactly one Skill-like class (name, version, reset, act).
    """
    if _is_path_spec(spec):
        file, sep, attr = spec.rpartition(":")
        if not sep or not attr.isidentifier():
            file, attr = spec, ""
        mod = _module_from_file(Path(file))
        return getattr(mod, attr) if attr else _only_skill_class(mod)
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"expected 'module:attr' or 'path.py[:attr]', got '{spec}'")
    return getattr(importlib.import_module(mod), attr)


def resolve_spec(spec: str) -> str:
    """Make a file spec absolute (so worker processes and dataset meta do not depend on cwd)."""
    if not _is_path_spec(spec):
        return spec
    file, sep, attr = spec.rpartition(":")
    if not sep or not attr.isidentifier():
        return str(Path(spec).resolve())
    return f"{Path(file).resolve()}:{attr}"


def task_sha(task: TaskSpec) -> str:
    from skillforge.eval.regression import task_hash

    return task_hash(task)


# ---------------------------------------------------------------- collection config


@dataclass
class DartNoise:
    """DART-style noise on the EXECUTED action; labels stay the clean teacher action.

    Noise comes in bursts: the episode is cut into `segment`-step segments and each one is
    perturbed with probability `p`, by a constant xyz offset ~ N(0, pos_std) per axis. Constant
    offsets within a segment push the arm off the teacher's path (white noise per step mostly
    averages out in the env's speed-limited reference), and quiet segments let the teacher
    reach its phase targets, so it is never blocked by noise. The gripper is not perturbed.
    """

    pos_std: float = 0.015
    p: float = 0.3
    segment: int = 8


@dataclass
class CollectConfig:
    source: str = "teacher"  # label for episodes.jsonl: teacher | dart | dagger:r<k>
    teacher: str = DEFAULT_TEACHER
    make_env: str = DEFAULT_ENV
    dart: DartNoise | None = None
    learner_checkpoint: str | None = None  # DAgger: execute this policy, the teacher labels
    beta: float = 0.0  # DAgger: per-step probability of executing the teacher's action instead
    success_only: bool = False  # drop episodes that did not succeed
    sync_tol: float | None = 0.03  # DAgger sync gate, metres (None: keep every label)
    sync_patience: int = 3  # DAgger: consecutive disagreeing steps before the teacher counts as out of phase
    pos_clip: float = 0.05  # metres; targets are compared after this clip (as the policy sees them)
    learner_overrides: dict = field(default_factory=dict)  # e.g. {"ensemble_m": 0.1}
    # DAgger labeler: "shadow" runs the teacher alongside the learner, unsynced (use the sync
    # gate); "phase_sync" wraps it in expert.PhaseSyncExpert, which re-syncs its phase to the
    # executed state every step (set sync_tol None with it: its labels stay usable off-path)
    relabel: str = "shadow"

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> CollectConfig:
        d = dict(d)
        if d.get("dart") is not None:
            d["dart"] = DartNoise(**d["dart"])
        return cls(**d)


# ---------------------------------------------------------------- one episode


def _teacher_failed(teacher: Skill) -> bool:
    """True once a primitives-based teacher's tree has failed (it only parks after that)."""
    runner = getattr(teacher, "runner", None)
    status = getattr(runner, "status", None)
    return bool(getattr(status, "failed", False))


def _clip(d: np.ndarray, c: float) -> np.ndarray:
    n = float(np.linalg.norm(d))
    return d if n <= c else d * (c / n)


def _agree(a: Action, b: Action, ee: np.ndarray, cfg: CollectConfig) -> bool:
    """Teacher and learner command the same thing, as the policy would see it (clipped targets)."""
    if round(a.gripper) != round(b.gripper):
        return False
    da = _clip(np.asarray(a.ee_pos) - ee, cfg.pos_clip)
    db = _clip(np.asarray(b.ee_pos) - ee, cfg.pos_clip)
    return float(np.linalg.norm(da - db)) <= cfg.sync_tol


def record_episode(
    env: SimEnv,
    task: TaskSpec,
    teacher: Skill,
    seed: int,
    cfg: CollectConfig,
    learner: Skill | None = None,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Roll out one seed. Returns (episode record, arrays). Mirrors eval.harness.run_episode."""
    from skillforge.eval.harness import _classify

    t0 = time.perf_counter()
    obs = env.reset(seed)
    teacher.reset(task)
    if learner is not None:
        learner.reset(task)
    noise_rng = np.random.default_rng([seed, 7])
    mix_rng = np.random.default_rng([seed, 8])
    states, labels, executed, valid, ts = [], [], [], [], []
    events: list[str] = []
    offset = np.zeros(3)
    desynced, run_start = False, None
    success = terminated = False
    failure, detail = None, ""
    n = 0
    try:
        for n in range(1, task.max_steps + 1):  # noqa: B007 (n is read after the loop)
            s = state_vector(obs, task)
            a_t = teacher.act(obs)
            ok = not _teacher_failed(teacher)
            if learner is not None:
                a_l = learner.act(obs)  # always queried, so its history/ensemble stays consistent
                a_x = a_t if (cfg.beta > 0 and mix_rng.random() < cfg.beta) else a_l
                if cfg.sync_tol is not None and not desynced:
                    if _agree(a_t, a_l, obs.ee_pos, cfg):
                        run_start = None
                    else:
                        run_start = len(valid) if run_start is None else run_start
                        if len(valid) - run_start + 1 > cfg.sync_patience:
                            desynced = True
                            valid[run_start:] = [False] * (len(valid) - run_start)
                ok = ok and not desynced
            else:
                a_x = a_t
            if cfg.dart is not None:
                if (n - 1) % cfg.dart.segment == 0:
                    on = noise_rng.random() < cfg.dart.p
                    offset = noise_rng.normal(0.0, cfg.dart.pos_std, 3) if on else np.zeros(3)
                if np.any(offset):
                    a_x = Action(mode="ee", ee_pos=np.asarray(a_x.ee_pos) + offset, ee_quat=a_x.ee_quat, gripper=a_x.gripper)
            note = getattr(teacher, "note_executed", None)
            if note is not None:
                note(a_x)
            states.append(s)
            labels.append(action_vector(a_t))
            executed.append(action_vector(a_x))
            valid.append(ok)
            ts.append(obs.t)
            obs, info = env.step(a_x)
            events += info.events
            if info.success:
                success = True
                break
            if info.terminated:
                terminated = True
                break
    except Exception as e:  # record what we have; the episode is a failure
        failure, detail = "exception", f"{type(e).__name__}: {e}"
    if not success and failure is None:
        failure, why = _classify(events, terminated, timed_out=n >= task.max_steps)
        detail = why
    arrays = {
        "observation.state": np.asarray(states, np.float32).reshape(len(states), -1),
        "action": np.asarray(labels, np.float32).reshape(len(labels), -1),
        "action.executed": np.asarray(executed, np.float32).reshape(len(executed), -1),
        "label_valid": np.asarray(valid, bool),
        "timestamp": np.asarray(ts, np.float32),
        "frame_index": np.arange(len(states), dtype=np.int32),
    }
    rec = {
        "seed": int(seed),
        "length": len(states),
        "success": bool(success),
        "failure": failure,
        "failure_detail": detail,
        "source": cfg.source,
        "label_valid_frac": round(float(np.mean(valid)), 4) if valid else 0.0,
        "phase_jumps": len(getattr(teacher, "jumps", [])),
        "randomization": env.randomization_sample(),
        "events_tail": events[-8:],
        "duration_s": round(time.perf_counter() - t0, 4),
    }
    return rec, arrays


# ---------------------------------------------------------------- parallel collection

_WORKER: dict[str, Any] = {}


def _worker_setup(task_json: str, cfg_json: str) -> tuple[SimEnv, Skill, Skill | None, TaskSpec, CollectConfig]:
    key = task_json + cfg_json
    if _WORKER.get("key") != key:
        if "env" in _WORKER:
            _WORKER["env"].close()
        task = TaskSpec.model_validate_json(task_json)
        cfg = CollectConfig.from_json(json.loads(cfg_json))
        env = load_object(cfg.make_env)(task)
        if cfg.relabel == "phase_sync":
            from .expert import PhaseSyncExpert

            teacher = PhaseSyncExpert(load_object(cfg.teacher))
        elif cfg.relabel == "shadow":
            teacher = load_object(cfg.teacher)()
        else:
            raise ValueError(f"unknown relabel mode '{cfg.relabel}'")
        learner = None
        if cfg.learner_checkpoint:
            from .skill import LearnedSkill

            learner = LearnedSkill(cfg.learner_checkpoint, **cfg.learner_overrides)
        _WORKER.update(key=key, env=env, teacher=teacher, learner=learner, task=task, cfg=cfg)
    w = _WORKER
    return w["env"], w["teacher"], w["learner"], w["task"], w["cfg"]


def _release_worker() -> None:
    env = _WORKER.get("env")
    if env is not None:
        env.close()
    _WORKER.clear()


def _collect_chunk(task_json: str, cfg_json: str, seeds: list[int]) -> list[tuple[dict, dict]]:
    env, teacher, learner, task, cfg = _worker_setup(task_json, cfg_json)
    return [record_episode(env, task, teacher, s, cfg, learner) for s in seeds]


def collect(
    task: TaskSpec,
    seeds: Iterable[int],
    cfg: CollectConfig,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> list[tuple[dict, dict]]:
    """Episodes for `seeds`, in seed order, identical for any `workers`."""
    seeds = [int(s) for s in seeds]
    tj, cj = task.model_dump_json(), json.dumps(cfg.to_json(), sort_keys=True)
    if workers <= 1:
        out = []
        for i, s in enumerate(seeds):
            out += _collect_chunk(tj, cj, [s])
            if progress:
                progress(i + 1, len(seeds))
        _release_worker()
        return out
    size = max(1, min(16, len(seeds) // (workers * 4) or 1))
    parts = [seeds[i : i + size] for i in range(0, len(seeds), size)]
    out: list = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futs = [pool.submit(_collect_chunk, tj, cj, p) for p in parts]
        for f in futs:  # submission order = seed order
            out += f.result()
            if progress:
                progress(len(out), len(seeds))
    return out


# ---------------------------------------------------------------- storage


def _stats(x: np.ndarray) -> dict[str, list[float]]:
    return {
        "mean": x.mean(0).tolist(),
        "std": x.std(0).tolist(),
        "min": x.min(0).tolist(),
        "max": x.max(0).tolist(),
    }


class Dataset:
    """A dataset directory. `Dataset.create` starts one, `Dataset(path)` opens one."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta" / "info.json").read_text())
        if self.info.get("format") != FORMAT:
            raise ValueError(f"{self.root}: not a {FORMAT} dataset")
        p = self.root / "meta" / "episodes.jsonl"
        self.episodes = [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []

    @property
    def task(self) -> TaskSpec:
        return TaskSpec.model_validate(self.info["task"])

    @property
    def state_names(self) -> list[str]:
        return self.info["features"]["observation.state"]["names"]

    @classmethod
    def create(cls, root: str | Path, task: TaskSpec, teacher: str, n_q: int, overwrite: bool = False) -> Dataset:
        root = Path(root)
        if (root / "meta" / "info.json").exists() and not overwrite:
            raise FileExistsError(f"{root} already holds a dataset (append with Dataset(path).append)")
        (root / "meta").mkdir(parents=True, exist_ok=True)
        (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (root / "meta" / "episodes.jsonl").write_text("")
        t = load_object(teacher)()
        names = state_names(task, n_q)
        info = {
            "format": FORMAT,
            "task_id": task.id,
            "task_sha": task_sha(task),
            "task": task.model_dump(mode="json"),
            "teacher": {"entry": teacher, "name": t.name, "version": t.version},
            "fps": task.control_hz,
            "features": {
                "observation.state": {"dtype": "float32", "shape": [len(names)], "names": names},
                "action": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": list(ACTION_NAMES)},
                "action.executed": {"dtype": "float32", "shape": [len(ACTION_NAMES)], "names": list(ACTION_NAMES)},
                "label_valid": {"dtype": "bool", "shape": [1]},
                "timestamp": {"dtype": "float32", "shape": [1]},
                "frame_index": {"dtype": "int32", "shape": [1]},
            },
            "total_episodes": 0,
            "total_frames": 0,
            "collections": [],
            "software": {"python": platform.python_version(), "numpy": np.__version__},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            import mujoco

            info["software"]["mujoco"] = mujoco.__version__
        except ImportError:
            pass
        (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
        return cls(root)

    def episode_path(self, i: int) -> Path:
        return self.root / "data" / f"chunk-{i // 1000:03d}" / f"episode_{i:06d}.npz"

    def append(self, eps: Sequence[tuple[dict, dict]], cfg: CollectConfig, seeds: Sequence[int]) -> int:
        """Write episodes (skipping failures if cfg.success_only). Returns how many were kept."""
        kept = 0
        start = len(self.episodes)
        with (self.root / "meta" / "episodes.jsonl").open("a") as f:
            for rec, arr in eps:
                if cfg.success_only and not rec["success"]:
                    continue
                if rec["length"] == 0:
                    continue
                i = start + kept
                p = self.episode_path(i)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(p, **arr)
                rec = {"episode_index": i, **rec, "file": str(p.relative_to(self.root))}
                f.write(json.dumps(rec) + "\n")
                self.episodes.append(rec)
                kept += 1
        seeds = list(seeds)
        self.info["collections"].append(
            {
                "config": cfg.to_json(),
                "n_seeds": len(seeds),
                "seed_range": [min(seeds), max(seeds)] if seeds else None,
                "n_rolled": len(eps),
                "n_success": int(sum(r["success"] for r, _ in eps)),
                "n_kept": kept,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        self.info["total_episodes"] = len(self.episodes)
        self.info["total_frames"] = int(sum(e["length"] for e in self.episodes))
        (self.root / "meta" / "info.json").write_text(json.dumps(self.info, indent=2))
        self.write_stats()
        return kept

    def load(self, i: int) -> dict[str, np.ndarray]:
        with np.load(self.root / self.episodes[i]["file"]) as z:
            return {k: z[k] for k in z.files}

    def write_stats(self) -> None:
        if not self.episodes:
            return
        arrs = [self.load(i) for i in range(len(self.episodes))]
        st = {k: _stats(np.concatenate([a[k] for a in arrs])) for k in ("observation.state", "action")}
        (self.root / "meta" / "stats.json").write_text(json.dumps(st))


def collect_to(
    root: str | Path,
    task: TaskSpec,
    seeds: Sequence[int],
    cfg: CollectConfig,
    workers: int = 1,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[Dataset, dict]:
    """Roll out and write into `root` (created if missing, appended to if it exists)."""
    t0 = time.perf_counter()
    eps = collect(task, seeds, cfg, workers=workers, progress=progress)
    root = Path(root)
    if (root / "meta" / "info.json").exists() and not overwrite:
        ds = Dataset(root)
        if ds.info["task_sha"] != task_sha(task):
            raise ValueError(f"{root} was collected on a different task (sha {ds.info['task_sha']})")
    else:
        n_q = int(next((len(a["observation.state"][0]) for _, a in eps if len(a["observation.state"])), 0))
        n_q = _n_q_from_dim(task, n_q)
        ds = Dataset.create(root, task, cfg.teacher, n_q, overwrite=overwrite)
    kept = ds.append(eps, cfg, seeds)
    summary = {
        "rolled": len(eps),
        "success": int(sum(r["success"] for r, _ in eps)),
        "kept": kept,
        "frames_rolled": int(sum(r["length"] for r, _ in eps)),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return ds, summary


def _n_q_from_dim(task: TaskSpec, dim: int) -> int:
    """Recover the arm's joint count from a state vector's length (2 n_q + fixed part)."""
    fixed = len(state_names(task, 0))
    n2 = dim - fixed
    if n2 < 0 or n2 % 2:
        raise ValueError(f"state length {dim} does not fit the task layout")
    return n2 // 2


def default_dataset_dir(name: str) -> Path:
    base = os.environ.get("SKILLFORGE_RUNS")
    root = Path(base) if base else Path(__file__).resolve().parents[3] / "runs"
    return root / "datasets" / name
