"""Behavior cloning on CPU: an MLP over a short observation history that predicts an action chunk.

ACT-style chunking without the transformer/CVAE: the network maps the last `n_hist` feature
frames to the next `horizon` actions, each as (target position relative to the CURRENT ee, 6D
orientation, gripper). At execution `LearnedSkill` re-queries every step and averages all
predictions that cover the current step (temporal ensembling). Loss is L1 on normalized
targets, masked past the episode end and wherever the teacher's label is invalid.

Normalization statistics (feature and action mean/std) live in the checkpoint, so a checkpoint
is self-contained: `load_policy(path)` is all a consumer needs.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import Dataset
from .features import (
    FEATURE_DIM,
    POLICY_ACTION_DIM,
    Layout,
    chunk_targets,
    encode_actions,
    history_index,
    policy_features,
)

CKPT_FORMAT = "skillforge.bc_policy/1"
STD_FLOOR = 1e-2


@dataclass
class TrainConfig:
    n_hist: int = 2
    horizon: int = 10
    hidden: tuple[int, ...] = (512, 512, 512)
    pos_clip: float = 0.05  # metres; see features.encode_actions
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    val_frac: float = 0.1
    seed: int = 0
    threads: int = 4
    max_minutes: float | None = None  # stop early when the wall clock runs out
    sources: list[str] | None = None  # train only on episodes whose source is listed (None: all)
    log_every: int = 1
    extra: dict = field(default_factory=dict)


class MLPPolicy(nn.Module):
    def __init__(self, in_dim: int, out_shape: tuple[int, int], hidden: Sequence[int]):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()]
            d = h
        layers.append(nn.Linear(d, out_shape[0] * out_shape[1]))
        self.net = nn.Sequential(*layers)
        self.out_shape = out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1, *self.out_shape)


# ---------------------------------------------------------------- data


def build_arrays(
    datasets: Sequence[Dataset], cfg: TrainConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stack every episode into (X, Y, mask, episode_id). X: (N, n_hist*F), Y: (N, H, 10)."""
    names = datasets[0].state_names
    obj = datasets[0].task.success.object
    lay = Layout(names, obj)
    xs, ys, ms, eid = [], [], [], []
    k = 0
    for ds in datasets:
        if ds.state_names != names:
            raise ValueError(f"{ds.root}: state layout differs from {datasets[0].root}")
        for i, rec in enumerate(ds.episodes):
            if cfg.sources is not None and not any(rec["source"].startswith(s) for s in cfg.sources):
                continue
            a = ds.load(i)
            st, lab, valid = a["observation.state"].astype(np.float64), a["action"].astype(np.float64), a["label_valid"]
            T = len(st)
            feat = policy_features(st, lay)
            enc = encode_actions(lab, st, lay, cfg.pos_clip)
            y, m = chunk_targets(enc, st, lay, cfg.horizon)
            idx = np.minimum(np.arange(T)[:, None] + np.arange(cfg.horizon)[None, :], T - 1)
            m = m & valid[idx] & valid[:, None]
            keep = m[:, 0]
            if not keep.any():
                continue
            x = feat[history_index(T, cfg.n_hist)].reshape(T, -1)
            xs.append(x[keep])
            ys.append(y[keep])
            ms.append(m[keep])
            eid.append(np.full(int(keep.sum()), k))
            k += 1
    if not xs:
        raise ValueError("no usable frames in the datasets")
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ms), np.concatenate(eid), names


def _norm(x: np.ndarray, axis) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=axis).astype(np.float32), np.maximum(x.std(axis=axis), STD_FLOOR).astype(np.float32)


# ---------------------------------------------------------------- training


def train(
    datasets: Sequence[Dataset],
    out: str | Path,
    cfg: TrainConfig | None = None,
    log=print,
) -> dict:
    """Train a policy and write `out` (a .pt checkpoint) plus `<out>.json` (training summary)."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = np.random.default_rng(cfg.seed)
    t0 = time.perf_counter()
    X, Y, M, E, names = build_arrays(datasets, cfg)
    n_ep = int(E.max()) + 1
    val_eps = set(rng.permutation(n_ep)[: int(round(cfg.val_frac * n_ep))].tolist()) if cfg.val_frac > 0 else set()
    is_val = np.isin(E, list(val_eps))
    x_mu, x_sd = _norm(X[~is_val], 0)
    flat = Y[~is_val][M[~is_val]]  # (n, 10) every valid chunk entry
    y_mu, y_sd = _norm(flat, 0)

    def to_t(a, dt=torch.float32):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=dt)

    Xn, Yn = to_t((X - x_mu) / x_sd), to_t((Y - y_mu) / y_sd)
    Mt = to_t(M, torch.float32)
    tr = np.flatnonzero(~is_val)
    va = np.flatnonzero(is_val)
    model = MLPPolicy(X.shape[1], (cfg.horizon, POLICY_ACTION_DIM), cfg.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(tr) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.epochs * steps_per_epoch, pct_start=0.1)

    def loss_on(idx: np.ndarray) -> torch.Tensor:
        pred = model(Xn[idx])
        err = (pred - Yn[idx]).abs().mean(-1)  # (B, H)
        m = Mt[idx]
        return (err * m).sum() / m.sum().clamp(min=1.0)

    history = []
    epochs_run = 0
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for b in range(steps_per_epoch):
            idx = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size]
            loss = loss_on(idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach())
        epochs_run = ep + 1
        val = None
        if len(va):
            model.eval()
            with torch.no_grad():
                val = float(np.mean([float(loss_on(va[i : i + 4096])) for i in range(0, len(va), 4096)]))
        history.append({"epoch": epochs_run, "train_l1": tot / steps_per_epoch, "val_l1": val})
        if log and (epochs_run % cfg.log_every == 0 or epochs_run == cfg.epochs):
            log(f"epoch {epochs_run}/{cfg.epochs} train_l1={tot / steps_per_epoch:.4f}" + (f" val_l1={val:.4f}" if val is not None else ""))
        if cfg.max_minutes is not None and time.perf_counter() - t0 > 60 * cfg.max_minutes:
            log and log(f"stopping: {cfg.max_minutes} min wall-clock budget used")
            break
    seconds = time.perf_counter() - t0

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds0 = datasets[0]
    meta = {
        "format": CKPT_FORMAT,
        "config": {**asdict(cfg), "hidden": list(cfg.hidden)},
        "in_dim": int(X.shape[1]),
        "feature_dim": FEATURE_DIM,
        "state_names": names,
        "object": ds0.task.success.object,
        "task_id": ds0.task.id,
        "task_sha": ds0.info["task_sha"],
        "teacher": ds0.info["teacher"],
        "datasets": [str(d.root) for d in datasets],
        "n_frames": int(len(X)),
        "n_episodes": n_ep,
        "epochs_run": epochs_run,
        "train_seconds": round(seconds, 1),
        "history": history,
    }
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    norm = {"x_mu": x_mu, "x_sd": x_sd, "y_mu": y_mu, "y_sd": y_sd}
    torch.save({"meta": meta, "state_dict": state, "norm": {k: torch.as_tensor(v) for k, v in norm.items()}}, out)
    meta["checkpoint_sha"] = file_sha(out)
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


def file_sha(p: str | Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- inference


class LoadedPolicy:
    """A checkpoint ready for inference on CPU: `predict(x_hist) -> (horizon, 10)` in real units."""

    def __init__(self, path: str | Path, threads: int = 1):
        torch.set_num_threads(threads)
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        if ck["meta"].get("format") != CKPT_FORMAT:
            raise ValueError(f"{path}: not a {CKPT_FORMAT} checkpoint")
        self.meta = ck["meta"]
        c = self.meta["config"]
        self.n_hist, self.horizon, self.pos_clip = int(c["n_hist"]), int(c["horizon"]), float(c["pos_clip"])
        self.model = MLPPolicy(self.meta["in_dim"], (self.horizon, POLICY_ACTION_DIM), c["hidden"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        n = {k: v.numpy() for k, v in ck["norm"].items()}
        self.x_mu, self.x_sd, self.y_mu, self.y_sd = n["x_mu"], n["x_sd"], n["y_mu"], n["y_sd"]
        self.sha = file_sha(path)
        self.path = str(Path(path).resolve())

    def predict(self, x: np.ndarray) -> np.ndarray:
        xn = (np.asarray(x, np.float32).reshape(1, -1) - self.x_mu) / self.x_sd
        with torch.no_grad():
            y = self.model(torch.from_numpy(xn.astype(np.float32)))[0].numpy()
        return y * self.y_sd + self.y_mu
