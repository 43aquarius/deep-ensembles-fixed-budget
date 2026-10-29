#!/usr/bin/env python3
"""v2 extension experiments: budget-matched comparison, generalized.

Extends the pre-registered v1 design along four axes, holding the v1
training procedure fixed:

  --id {fmnist,mnist}   in-distribution dataset (v1: fmnist)
  --budget B            total epochs (v1: 60)
  --k K                 ensemble size / cycle count (v1: 3)
  --seeds a:b           replicate seed range [a, b)

Per seed s, base = 10000 + 100*s (same convention as v1):
  - Single        : B epochs, one cosine cycle, seed base+0
  - Cyclic run    : B epochs, K cosine cycles, seed base+0, snapshots at
                    B/K, 2B/K, ..., B  (snapshot 1 bit-identical to a fresh
                    B/K-epoch run with seed base+0)
  - DeepEns-K     : {snapshot 1} + fresh runs seeds base+11..base+1K, B/K epochs
  - Snapshot-K    : the K cycle-end snapshots
  - SWA-K         : uniform weight average of the same K snapshots

New in v2 (not in v1):
  - two OOD probes per setting (fmnist-ID: mnist + kmnist; mnist-ID: fmnist + kmnist)
  - temperature scaling fitted per condition on the validation split
    (raw-logit TS for single models; log-mean-prob TS for ensembles),
    reported alongside the raw (uncalibrated) metrics
  - member-level metrics for every ensemble size

Usage:
  python run_experiments_v2.py --id fmnist --budget 60 --k 3 \
      --seeds 0:10 --out results/v2/experiment_v2_fmnist_B60_k3.jsonl
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    TrainConfig, train_run, load_splits_for, logits_of, probs_of, ens_probs,
    evaluate_predictor, pairwise_jsd, swa_state_dicts, ece, accuracy, nll,
    fit_temperature, apply_temperature, log_probs_as_logits,
)

N_CLASSES = 10


def reliability(p, y, bins=15):
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.digitize(conf, edges[1:-1], right=False)
    acc, cnf, cnt = [], [], []
    for b in range(bins):
        m = idx == b
        acc.append(float(correct[m].mean()) if m.any() else None)
        cnf.append(float(conf[m].mean()) if m.any() else None)
        cnt.append(int(m.sum()))
    return dict(bin_acc=acc, bin_conf=cnf, bin_count=cnt)


def run_seed(s, splits, B, K, id_name):
    base = 10000 + 100 * s
    t0 = time.time()
    member_epochs = B // K

    # --- training (2B - member_epochs epochs; + member_epochs extra on seed 0)
    sd_single, _, hist_single = train_run(
        TrainConfig(epochs=B, cycles=1, seed=base + 0), splits)
    sd_cyc, snaps, hist_cyc = train_run(
        TrainConfig(epochs=B, cycles=K, seed=base + 0, snapshot=True), splits)
    assert len(snaps) == K

    member_sds = [snaps[0]]  # member 1 == snapshot 1 (bit-identity verified in v1)
    for i in range(1, K):
        m_sd, _, _ = train_run(
            TrainConfig(epochs=member_epochs, cycles=1, seed=base + 10 + i), splits)
        member_sds.append(m_sd)

    shared_check = None
    if s == 0:
        fresh_sd, _, _ = train_run(
            TrainConfig(epochs=member_epochs, cycles=1, seed=base + 0), splits)
        shared_check = all(torch.equal(fresh_sd[k], snaps[0][k]) for k in fresh_sd)

    swa_sd = swa_state_dicts(snaps)

    # --- evaluation arrays
    Xtest, ytest = splits["test_x"], splits["test_y"]
    Xval, yval = splits["val_x"], splits["val_y"]
    oods = splits["oods"]  # {name: {x, y}}

    res = dict(
        seed=s, base=base,
        config=dict(id=id_name, budget=B, k=K, ood_probes=list(oods.keys())),
        versions=dict(torch=torch.__version__, numpy=np.__version__),
        shared_member_bit_identical=shared_check,
        histories=dict(single=hist_single, cyclic=hist_cyc),
    )

    from common import brier as brier_fn, ood_auroc as ood_auroc_fn

    def eval_cond(name, p_test, p_val, ood_p_by_probe, members_p=None):
        """Raw + temperature-scaled metrics for one condition."""
        r = dict(
            nll=nll(p_test, ytest),
            acc=accuracy(p_test, ytest),
            brier=brier_fn(p_test, ytest),
            ece15=ece(p_test, ytest, bins=15),
            ece10=ece(p_test, ytest, bins=10),
        )
        for pname, p_ood in ood_p_by_probe.items():
            r[f"auroc_msp_{pname}"] = ood_auroc_fn(p_test, p_ood, "msp")
            r[f"auroc_ent_{pname}"] = ood_auroc_fn(p_test, p_ood, "entropy")
        r["reliability15"] = reliability(p_test, ytest, 15)

        # TS: fit on validation effective logits, apply to test and OOD
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
        for pname, p_ood in ood_p_by_probe.items():
            p_ts_ood = apply_temperature(log_probs_as_logits(p_ood), T_hat)
            ts[f"auroc_msp_{pname}"] = ood_auroc_fn(p_ts_test, p_ts_ood, "msp")
            ts[f"auroc_ent_{pname}"] = ood_auroc_fn(p_ts_test, p_ts_ood, "entropy")
        r["ts"] = ts

        if members_p is not None:
            r["member_mean_nll"] = float(np.mean([nll(p, ytest) for p in members_p]))
            r["member_mean_acc"] = float(np.mean([accuracy(p, ytest) for p in members_p]))
            r["disagreement_jsd"] = pairwise_jsd(members_p)
        return r

    # --- per-condition evaluation
    def probs_on(sd, X):
        return probs_of(logits_of(sd, X))

    # Single
    p_s_test = probs_on(sd_single, Xtest)
    p_s_val = probs_on(sd_single, Xval)
    # DeepEns members
    dm_p_test = [probs_on(sd, Xtest) for sd in member_sds]
    dm_p_val = [probs_on(sd, Xval) for sd in member_sds]
    # Snapshot members
    sn_p_test = [probs_on(sd, Xtest) for sd in snaps]
    sn_p_val = [probs_on(sd, Xval) for sd in snaps]
    # SWA
    p_w_test = probs_on(swa_sd, Xtest)
    p_w_val = probs_on(swa_sd, Xval)

    res["single"] = eval_cond(
        "single", p_s_test, p_s_val,
        {pname: probs_on(sd_single, od["x"]) for pname, od in oods.items()})
    res["deepens"] = eval_cond(
        "deepens", ens_probs(dm_p_test), ens_probs(dm_p_val),
        {pname: ens_probs([probs_on(sd, od["x"]) for sd in member_sds])
         for pname, od in oods.items()},
        members_p=dm_p_test)
    res["snapshot"] = eval_cond(
        "snapshot", ens_probs(sn_p_test), ens_probs(sn_p_val),
        {pname: ens_probs([probs_on(sd, od["x"]) for sd in snaps])
         for pname, od in oods.items()},
        members_p=sn_p_test)
    res["swa"] = eval_cond(
        "swa", p_w_test, p_w_val,
        {pname: probs_on(swa_sd, od["x"]) for pname, od in oods.items()})

    res["wallclock_s"] = time.time() - t0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", choices=["fmnist", "mnist"], default="fmnist")
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seeds", default="0:10",
                    help="inclusive-exclusive range, e.g. 0:10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--time-budget", type=float, default=None,
                    help="seconds; stop before starting a new seed once "
                         "exceeded (resumable). Exit 42 = unfinished, 0 = done.")
    args = ap.parse_args()

    a, b = (int(x) for x in args.seeds.split(":"))
    splits = load_splits_for(args.id)

    done = set()
    try:
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["seed"])
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
        print(f"=== {args.id} B={args.budget} k={args.k} seed {s} ===", flush=True)
        r = run_seed(s, splits, args.budget, args.k, args.id)
        with open(args.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(f"seed {s} done in {r['wallclock_s']:.0f}s | "
              f"single NLL={r['single']['nll']:.4f} "
              f"(TS {r['single']['ts']['nll']:.4f}) | "
              f"deepens NLL={r['deepens']['nll']:.4f} "
              f"(TS {r['deepens']['ts']['nll']:.4f}) | "
              f"snapshot NLL={r['snapshot']['nll']:.4f} | "
              f"swa NLL={r['swa']['nll']:.4f} | "
              f"shared_ok={r['shared_member_bit_identical']}",
              flush=True)
    raise SystemExit(0 if all_done else 42)


if __name__ == "__main__":
    main()
