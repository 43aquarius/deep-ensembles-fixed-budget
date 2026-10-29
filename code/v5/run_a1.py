#!/usr/bin/env python3
"""A1 ablation driver (v5, Plan B): Single with validation-based model selection.

Pre-registered spec (Doc 3, Table 7-1, A1):
  RQ5: does the model-selection rule (final weights vs validation early
  stopping) change the "single worsens with budget" fact and the allocation
  advice?
  Grid: FMNIST B in {30, 60, 120} x 10 seeds (base = 10000 + 100s).
  Rule: retrain the Single trajectory exactly as run_experiments_v2.py does
  (same seed/init, same data order, same cosine LR), track per-epoch val NLL,
  evaluate the test set at the validation-optimal epoch (global argmin).

Bit-identity guarantee: the training loop below is copied verbatim from
common.train_run (single-run path, cycles=1). Additionally, the final-epoch
weights are evaluated too, so every record carries a built-in consistency
check against the archived S1 single metrics (must match bit-exactly).

Outputs (per budget): results/v5/experiment_v5_a1_fmnist_B{30,60,120}.jsonl
  Fields: seed, config, versions, histories.single (val trajectory),
  e_star, val_nll_star, single_es {...}, single_final {...}, wallclock_s.

Usage: python run_a1.py --budget B [--seeds a:b] --out PATH
       [--time-budget SEC]  (stop before a new seed; exit 42 = resumable)
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    TrainConfig, load_splits_for, logits_of, probs_of,
    ece, accuracy, nll, MLP, lr_at,
    fit_temperature, apply_temperature, log_probs_as_logits,
)
from run_experiments_v2 import reliability  # identical binning helper

CODE = os.path.dirname(os.path.abspath(__file__))


def train_single_with_val_best(cfg, splits):
    """Verbatim copy of common.train_run (cycles=1 path) + best-val tracking.

    Returns (final_sd, best_sd, e_star, history).
    Trajectory is bit-identical to train_run(cfg, splits) by construction.
    """
    s = splits
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

    history = []
    best_val, best_sd, e_star = float("inf"), None, 1
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
        # monitoring: validation only -- used ONLY by the A1 selection rule
        model.eval()
        with torch.no_grad():
            vlog = model(torch.from_numpy(s["val_x"]))
            vnll = F.cross_entropy(vlog, torch.from_numpy(s["val_y"])).item()
        history.append(dict(epoch=epoch + 1, train_loss=tl / seen,
                           val_nll=vnll, lr=lr_at((epoch + 1) * spe - 1, cfg, spe)))
        if vnll < best_val:
            best_val = vnll
            e_star = epoch + 1
            best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return model_sd, best_sd, e_star, history


def eval_single(sd, splits):
    """Same metric suite as run_experiments_v2.eval_cond for a single model."""
    from common import brier as brier_fn, ood_auroc as ood_auroc_fn
    Xtest, ytest = splits["test_x"], splits["test_y"]
    Xval, yval = splits["val_x"], splits["val_y"]
    oods = splits["oods"]

    p_test = probs_of(logits_of(sd, Xtest))
    p_val = probs_of(logits_of(sd, Xval))

    r = dict(
        nll=nll(p_test, ytest),
        acc=accuracy(p_test, ytest),
        brier=brier_fn(p_test, ytest),
        ece15=ece(p_test, ytest, bins=15),
        ece10=ece(p_test, ytest, bins=10),
    )
    for pname, od in oods.items():
        p_ood = probs_of(logits_of(sd, od["x"]))
        r[f"auroc_msp_{pname}"] = ood_auroc_fn(p_test, p_ood, "msp")
        r[f"auroc_ent_{pname}"] = ood_auroc_fn(p_test, p_ood, "entropy")
    r["reliability15"] = reliability(p_test, ytest, 15)

    z_val = log_probs_as_logits(p_val)
    T_hat, val_nll_ts = fit_temperature(z_val, yval)
    z_test = log_probs_as_logits(p_test)
    p_ts_test = apply_temperature(z_test, T_hat)
    ts = dict(T=T_hat, val_nll_ts=val_nll_ts)
    ts.update(dict(
        nll=nll(p_ts_test, ytest),
        acc=accuracy(p_ts_test, ytest),
        brier=brier_fn(p_ts_test, ytest),
        ece15=ece(p_ts_test, ytest, bins=15),
        ece10=ece(p_ts_test, ytest, bins=10),
    ))
    for pname, od in oods.items():
        p_ts_ood = apply_temperature(
            log_probs_as_logits(probs_of(logits_of(sd, od["x"]))), T_hat)
        ts[f"auroc_msp_{pname}"] = ood_auroc_fn(p_ts_test, p_ts_ood, "msp")
        ts[f"auroc_ent_{pname}"] = ood_auroc_fn(p_ts_test, p_ts_ood, "entropy")
    r["ts"] = ts
    return r


def run_seed(s, splits, B):
    base = 10000 + 100 * s
    t0 = time.time()
    cfg = TrainConfig(epochs=B, cycles=1, seed=base + 0)

    final_sd, best_sd, e_star, history = train_single_with_val_best(cfg, splits)

    res = dict(
        seed=s, base=base,
        config=dict(id="fmnist", budget=B,
                    ablation="a1_early_stop_single",
                    selection="val_argmin_earliest_tie"),
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        best_epoch=e_star,
        best_val_nll=history[e_star - 1]["val_nll"],
        final_val_nll=history[-1]["val_nll"],
        history=history,
        es=eval_single(best_sd, splits),
        final=eval_single(final_sd, splits),
        wallclock_s=time.time() - t0,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seeds", default="0:10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=None)
    args = ap.parse_args()

    a, b = (int(x) for x in args.seeds.split(":"))
    splits = load_splits_for("fmnist")

    done = set()
    try:
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["seed"])
                    except json.JSONDecodeError:
                        pass  # truncated tail -> ignore
    except FileNotFoundError:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t_start = time.time()
    all_done = True
    for s in range(a, b):
        if s in done:
            print(f"seed {s}: already done, skipping", flush=True)
            continue
        if args.time_budget is not None and time.time() - t_start > args.time_budget:
            print(f"time budget reached before seed {s}; stopping (resumable)",
                  flush=True)
            all_done = False
            break
        print(f"=== a1 fmnist B={args.budget} seed {s} ===", flush=True)
        r = run_seed(s, splits, args.budget)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"seed {s} done in {r['wallclock_s']:.0f}s | "
              f"e*={r['best_epoch']}/{args.budget} | "
              f"ES NLL={r['es']['nll']:.4f} "
              f"(TS {r['es']['ts']['nll']:.4f}) | "
              f"final NLL={r['final']['nll']:.4f} "
              f"(TS {r['final']['ts']['nll']:.4f})",
              flush=True)
    raise SystemExit(0 if all_done else 42)


if __name__ == "__main__":
    main()
