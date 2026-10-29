#!/usr/bin/env python3
"""Shared utilities for the budget-matched ensemble study.

Everything is deterministic given (replicate_seed, condition, member).
All metrics are computed in float64 for numerical stability.
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

DATA_PATH = os.path.join(ROOT, "results", "data", "datasets.npz")
DATA_PATH_V2 = os.path.join(ROOT, "results", "data", "datasets_v2.npz")
SPLIT_SEED = 20260922  # fixed split seed, documented (same as v1)


def load_splits_for(id_name: str):
    """Generalized splits for v2 extension experiments.

    id_name in {'fmnist', 'mnist'} selects the in-distribution dataset
    (54k train / 6k val, same split seed as v1).  OOD probes are
    standardized with the *in-distribution* train statistics (same
    policy as v1).  Returns dict with train/val/test arrays plus
    'oods': {probe_name: {'x','y'}}.
    """
    key = f"load_splits_for:{id_name}"
    if key in _CACHE:
        return _CACHE[key]
    d = np.load(DATA_PATH_V2)
    ix = d[f"{id_name}_train_x"].astype(np.float32) / 255.0
    iy = d[f"{id_name}_train_y"].astype(np.int64)
    itx = d[f"{id_name}_test_x"].astype(np.float32) / 255.0
    ity = d[f"{id_name}_test_y"].astype(np.int64)

    rng = np.random.default_rng(SPLIT_SEED)
    idx = rng.permutation(60000)
    tr, va = idx[:54000], idx[54000:]
    mu, sd = ix[tr].mean(), ix[tr].std()

    oods = {}
    if id_name == "fmnist":
        probes = ["mnist", "kmnist"]
    elif id_name == "mnist":
        probes = ["fmnist", "kmnist"]
    else:
        raise ValueError(id_name)
    for p in probes:
        px = d[f"{p}_test_x"].astype(np.float32) / 255.0
        py = d[f"{p}_test_y"].astype(np.int64)
        oods[p] = dict(x=(px - mu) / sd, y=py)

    splits = dict(
        train_x=(ix[tr] - mu) / sd,
        train_y=iy[tr],
        val_x=(ix[va] - mu) / sd,
        val_y=iy[va],
        test_x=(itx - mu) / sd,
        test_y=ity,
        oods=oods,
        mu=mu,
        sd=sd,
    )
    _CACHE[key] = splits
    return splits

# ----------------------------------------------------------------------------- data

_CACHE = {}


def load_splits():
    """Fashion-MNIST 54k/6k/10k split + MNIST 10k OOD probe, standardized."""
    if "splits" in _CACHE:
        return _CACHE["splits"]
    d = np.load(DATA_PATH)
    fx = d["fmnist_train_x"].astype(np.float32) / 255.0
    fy = d["fmnist_train_y"].astype(np.int64)
    ftest_x = d["fmnist_test_x"].astype(np.float32) / 255.0
    ftest_y = d["fmnist_test_y"].astype(np.int64)
    mx = d["mnist_test_x"].astype(np.float32) / 255.0
    my = d["mnist_test_y"].astype(np.int64)

    rng = np.random.default_rng(20260922)  # fixed split seed, documented
    idx = rng.permutation(60000)
    tr, va = idx[:54000], idx[54000:]
    mu, sd = fx[tr].mean(), fx[tr].std()
    splits = dict(
        train_x=(fx[tr] - mu) / sd,
        train_y=fy[tr],
        val_x=(fx[va] - mu) / sd,
        val_y=fy[va],
        test_x=(ftest_x - mu) / sd,
        test_y=ftest_y,
        ood_x=(mx - mu) / sd,
        ood_y=my,
        mu=mu,
        sd=sd,
    )
    _CACHE["splits"] = splits
    return splits


# ----------------------------------------------------------------------------- model

class MLP(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)  # deterministic init per run
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----------------------------------------------------------------------------- training

@dataclass
class TrainConfig:
    epochs: int                 # total training epochs for this run
    cycles: int = 1             # number of cosine cycles (1 = plain cosine)
    lr_max: float = 0.1
    lr_min: float = 0.0
    momentum: float = 0.9
    weight_decay: float = 1e-4
    batch_size: int = 256
    seed: int = 0               # run seed (init + data order)
    snapshot: bool = False      # collect state_dict copies at cycle ends


def lr_at(step: int, cfg: TrainConfig, steps_per_epoch: int) -> float:
    """Per-step cosine schedule with `cycles` restarts over the whole run."""
    total_steps = cfg.epochs * steps_per_epoch
    cycle_steps = total_steps / cfg.cycles
    pos = (step % cycle_steps) / cycle_steps
    return cfg.lr_min + 0.5 * (cfg.lr_max - cfg.lr_min) * (1 + math.cos(math.pi * pos))


def train_run(cfg: TrainConfig, splits=None):
    """Train one MLP; return (state_dict list, per-epoch history).

    If cfg.snapshot, deep-copies of the model are stored after the last step
    of each cycle (epochs epochs/cycles, 2*epochs/cycles, ...).
    `splits` may be supplied explicitly (v2 multi-dataset runs); it defaults
    to the v1 Fashion-MNIST splits, preserving bit-level v1 behaviour.
    """
    s = load_splits() if splits is None else splits
    X = torch.from_numpy(s["train_x"])
    Y = torch.from_numpy(s["train_y"])
    n = X.shape[0]
    spe = math.ceil(n / cfg.batch_size)

    model = MLP(cfg.seed)
    opt = torch.optim.SGD(
        model.parameters(), lr=cfg.lr_max,
        momentum=cfg.momentum, weight_decay=cfg.weight_decay,
    )
    g = torch.Generator().manual_seed(cfg.seed + 1)  # data order stream

    snapshots, history = [], []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        tl, seen = 0.0, 0
        for i in range(spe):
            bidx = perm[i * cfg.batch_size:(i + 1) * cfg.batch_size]
            step = epoch * spe + i
            lr = lr_at(step, cfg, spe)
            for grp in opt.param_groups:
                grp["lr"] = lr
            opt.zero_grad()
            out = model(X[bidx])
            loss = F.cross_entropy(out, Y[bidx])
            loss.backward()
            opt.step()
            tl += loss.item() * len(bidx)
            seen += len(bidx)
        # monitoring: validation only (never used for selection)
        model.eval()
        with torch.no_grad():
            vlog = model(torch.from_numpy(s["val_x"]))
            vnll = F.cross_entropy(vlog, torch.from_numpy(s["val_y"])).item()
        history.append(dict(epoch=epoch + 1, train_loss=tl / seen,
                           val_nll=vnll, lr=lr_at((epoch + 1) * spe - 1, cfg, spe)))
        if cfg.snapshot and (epoch + 1) % (cfg.epochs // cfg.cycles) == 0:
            snapshots.append({k: v.detach().clone() for k, v in model.state_dict().items()})
    model_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model_sd, snapshots, history


# ----------------------------------------------------------------------------- metrics

def logits_of(sd, X):
    m = MLP(0)
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(X)).numpy()


def probs_of(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return e / e.sum(axis=1, keepdims=True)


def ens_probs(list_of_probs):
    return np.mean(np.stack(list_of_probs, axis=0), axis=0)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def accuracy(p, y):
    return float(np.mean(p.argmax(axis=1) == y))


def brier(p, y):
    onehot = np.eye(p.shape[1])[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def ece(p, y, bins=15):
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(conf, edges[1:-1], right=False)
    e, tot = 0.0, len(y)
    for b in range(bins):
        m = idx == b
        if m.any():
            e += (m.sum() / tot) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def ood_auroc(p_in, p_out, score="msp"):
    if score == "msp":
        s_in = p_in.max(axis=1)
        s_out = p_out.max(axis=1)
    elif score == "entropy":
        # negative-entropy score: s = sum(p*log p) <= 0, HIGHER = more in-distribution-like.
        # (Do not write -np.sum(...): that is +entropy and flips the orientation.)
        s_in = np.sum(p_in * np.log(np.clip(p_in, 1e-12, None)), axis=1)
        s_out = np.sum(p_out * np.log(np.clip(p_out, 1e-12, None)), axis=1)
    labels = np.concatenate([np.ones(len(s_in)), np.zeros(len(s_out))])
    scores = np.concatenate([s_in, s_out])
    return float(roc_auc_score(labels, scores))


def pairwise_jsd(probs_list):
    """Mean pairwise Jensen-Shannon divergence (nats) over test samples."""
    k = len(probs_list)
    vals = []
    for i in range(k):
        for j in range(i + 1, k):
            P, Q = probs_list[i], probs_list[j]
            M = 0.5 * (P + Q)
            def kl(A, B):
                return np.sum(A * np.log(np.clip(A, 1e-12, None) / np.clip(B, 1e-12, None)), axis=1)
            vals.append((0.5 * kl(P, M) + 0.5 * kl(Q, M)).mean())
    return float(np.mean(vals))


def swa_state_dicts(sds):
    """Uniform weight average of a list of state_dicts."""
    out = {}
    for k in sds[0]:
        out[k] = torch.mean(torch.stack([sd[k].float() for sd in sds]), dim=0)
    return out


# --------------------------------------------------------------------- temperature scaling

def _val_nll_at_T(z, y, logT):
    """Validation NLL of softmax(z / T) with T = exp(logT); z is (n, C) float64."""
    T = math.exp(logT)
    zz = z / T
    zz = zz - zz.max(axis=1, keepdims=True)
    e = np.exp(zz)
    p = e / e.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-300, None))))


def fit_temperature(z, y, lo=0.02, hi=50.0):
    """Fit a single temperature by minimizing validation NLL of softmax(z/T).

    Deterministic grid + golden refinement on log T.  z may be raw logits
    (single models; canonical Guo et al. scaling) or log of predictive
    probabilities (ensembles; the standard extension to averaged predictions,
    since a weight-averaged ensemble has no raw logits of its own).
    Returns (T_hat, val_nll_at_hat).
    """
    z = np.asarray(z, dtype=np.float64)
    lo_log, hi_log = math.log(lo), math.log(hi)
    grid = np.linspace(lo_log, hi_log, 400)
    vals = [_val_nll_at_T(z, y, lt) for lt in grid]
    i = int(np.argmin(vals))
    a = grid[max(i - 1, 0)]
    b = grid[min(i + 1, len(grid) - 1)]
    # golden-section refinement in [a, b]
    gr = (math.sqrt(5) - 1) / 2
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = _val_nll_at_T(z, y, c), _val_nll_at_T(z, y, d)
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _val_nll_at_T(z, y, c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _val_nll_at_T(z, y, d)
        if b - a < 1e-10:
            break
    logT = (a + b) / 2
    return float(math.exp(logT)), _val_nll_at_T(z, y, logT)


def apply_temperature(z, T):
    """Predictive probabilities after temperature scaling: softmax(z / T)."""
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_probs_as_logits(p, eps=1e-12):
    """Effective logit vector of a predictive distribution: log(clip(p))."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    return np.log(p)


def evaluate_predictor(p_test, s, p_ood=None):
    """All test-set metrics for one predictive distribution."""
    r = dict(
        nll=nll(p_test, s["test_y"]),
        acc=accuracy(p_test, s["test_y"]),
        brier=brier(p_test, s["test_y"]),
        ece15=ece(p_test, s["test_y"], bins=15),
        ece10=ece(p_test, s["test_y"], bins=10),
    )
    if p_ood is not None:
        r["auroc_msp"] = ood_auroc(p_test, p_ood, "msp")
        r["auroc_ent"] = ood_auroc(p_test, p_ood, "entropy")
    return r
