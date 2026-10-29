#!/usr/bin/env python3
"""A1 data-integrity checks (Doc 3 Table 7-1, A1 selection-rule ablation).

Cells: results/a1/experiment_v5_a1_fmnist_B{30,60,120}.jsonl (10 seeds each).

Checks:
  I1  each file has exactly seeds 0..9, one row each
  I2  versions match the S1/S2/S3 grid stamp (torch 2.14.1+cpu recorded as
      torch version string, numpy 2.1.3)
  I3  config fields correct (fmnist, right budget, a1 ablation tag, tie rule)
  I4  FINAL-epoch raw metrics bit-identical to the S1 'single' record of the
      same (B, seed)  [30 anchors]  -- proves the copied training loop and
      evaluation recipe reproduce the main-grid trajectory exactly
  I5  final TS metrics bit-identical to S1 as well
  I6  best_epoch in [1, B]; history has B entries; best_val_nll equals the
      minimum of history val_nll and corresponds to epoch best_epoch
  I7  ES metrics present and finite (nll/acc/brier/ece15/ece10/ts)
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import json
import math
import os

A1 = os.path.join(ROOT, "results", "a1")
S1 = os.path.join(ROOT, "results", "v5")

BUDGETS = [30, 60, 120]
fail = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fail.append(name)


# expected versions: read from an S1 file (grid stamp)
with open(os.path.join(S1, "experiment_v5_fmnist_B60_k2.jsonl")) as f:
    s1_row0 = json.loads(f.readline())
exp_torch = s1_row0["versions"]["torch"]
exp_numpy = s1_row0["versions"]["numpy"]

s1 = {}
for B in BUDGETS:
    with open(os.path.join(S1, f"experiment_v5_fmnist_B{B}_k2.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            s1[(B, r["seed"])] = r

total_anchors = 0
for B in BUDGETS:
    path = os.path.join(A1, f"experiment_v5_a1_fmnist_B{B}.jsonl")
    rows = [json.loads(l) for l in open(path) if l.strip()]

    # I1 seeds
    check(f"I1 B{B} seeds 0..9",
          sorted(r["seed"] for r in rows) == list(range(10)))

    # I2 versions
    ok = all(r["versions"] == dict(torch=exp_torch, numpy=exp_numpy)
             for r in rows)
    check(f"I2 B{B} versions = {exp_torch} / numpy {exp_numpy}", ok)

    # I3 config
    ok = all(r["config"] == dict(
        id="fmnist", budget=B, ablation="a1_early_stop_single",
        selection="val_argmin_earliest_tie") for r in rows)
    check(f"I3 B{B} config fields", ok)

    # I4/I5 bit-level final vs S1 single
    raw_metrics = ["nll", "acc", "brier", "ece15", "ece10"]
    ts_metrics = ["nll", "acc", "brier", "ece15", "ece10", "T",
                  "val_nll_ts"]
    bad_raw = bad_ts = 0
    for r in rows:
        ref = s1[(B, r["seed"])]["single"]
        for m in raw_metrics:
            if r["final"][m] != ref[m]:
                bad_raw += 1
        for m in ts_metrics:
            if r["final"]["ts"][m] != ref["ts"][m]:
                bad_ts += 1
        total_anchors += 1
    check(f"I4 B{B} final raw bit-identical to S1 single (10 seeds x 5)",
          bad_raw == 0, f"mismatches={bad_raw}")
    check(f"I5 B{B} final TS bit-identical (10 seeds x 7)",
          bad_ts == 0, f"mismatches={bad_ts}")

    # I6 best_epoch / history consistency
    ok = True
    for r in rows:
        h = r["history"]
        if not (1 <= r["best_epoch"] <= B) or len(h) != B:
            ok = False
        vals = [e["val_nll"] for e in h]
        if r["best_val_nll"] != min(vals):
            ok = False
        if h[r["best_epoch"] - 1]["val_nll"] != r["best_val_nll"]:
            ok = False
        first_min = 1 + min(range(B), key=lambda i: vals[i])
        if r["best_epoch"] != first_min:  # earliest-epoch tie rule
            ok = False
    check(f"I6 B{B} best_epoch/history consistency + tie rule", ok)

    # I7 ES finite
    ok = all(math.isfinite(r["es"][m]) and math.isfinite(r["es"]["ts"]["nll"])
             for r in rows for m in raw_metrics)
    check(f"I7 B{B} ES metrics finite", ok)

print()
print(f"anchors verified (final vs S1): {total_anchors} seeds x 12 metrics")
print("ALL A1 INTEGRITY CHECKS " + ("PASS" if not fail else f"FAIL: {fail}"))
raise SystemExit(0 if not fail else 1)
