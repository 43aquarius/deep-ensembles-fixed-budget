#!/usr/bin/env python3
"""A1/A2 analysis: RQ5 (selection-rule ablation) + H6 (inference-cost
benchmark), per Doc 3 Sections 1 (RQ5) / Table 3-1 (H6, descriptive).

A1 (ablation, exploratory relative to the pre-registered hypotheses -- no
pre-registered H or delta applies to it; reported as paired contrasts with
exact sign-flip permutation p-values and 95% t-CIs):
  T1  ES vs final within each B (10 paired seeds)
  T2  ES NLL across budgets (30->60->120, paired)  [RQ5 core: does the
      'single gets worse with budget' fact survive early stopping?]
  T3  final NLL across budgets (control, must reproduce S1's worsening)
  T4  allocation gap DE(k2) - Single under both selection rules:
      does early stopping change the allocation advice?
  T5  best_epoch distribution per budget

A2 (H6, descriptive): latency vs k (batch 1/256), params, storage, RSS.

Outputs:
  results/a1/analysis_v5_a1a2.json
  results/a1/summary_v5_a1.csv
  results/a1/summary_v5_a2_latency.csv
  results/a1/summary_v5_a2_cost.csv
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import csv
import json
import os

import numpy as np
from scipy import stats

BASE = os.path.join(ROOT, "results")
A1DIR = os.path.join(BASE, "a1")
A2FILE = os.path.join(BASE, "a2", "benchmark_inference.json")
S1 = os.path.join(BASE, "v5")
BUDGETS = [30, 60, 120]


def perm_test_paired(d):
    """Exact two-sided sign-flip permutation test on paired diffs
    (same implementation as analyze_s3.py)."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    obs = abs(d.mean())
    idx = np.arange(1 << n)
    signs = (((idx[:, None] >> np.arange(n)) & 1) * 2.0) - 1.0
    stat = np.abs((signs * d[None, :]).mean(axis=1))
    return float((stat >= obs - 1e-15).mean())


def ci95(d):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    m = d.mean()
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    return [float(m - h), float(m + h)]


def msp(a):
    """mean +- sd string-free tuple"""
    a = np.asarray(a, dtype=np.float64)
    return dict(mean=float(a.mean()), sd=float(a.std(ddof=1)))


# ---------------------------------------------------------------- load A1
a1 = {}
for B in BUDGETS:
    rows = [json.loads(l) for l in open(
        os.path.join(A1DIR, f"experiment_v5_a1_fmnist_B{B}.jsonl")) if l.strip()]
    rows.sort(key=lambda r: r["seed"])
    a1[B] = rows

# S1 deepens(k2) per (B, seed) for the allocation-gap contrast
de_k2 = {}
for B in BUDGETS:
    rows = [json.loads(l) for l in open(
        os.path.join(S1, f"experiment_v5_fmnist_B{B}_k2.jsonl")) if l.strip()]
    de_k2[B] = {r["seed"]: r["deepens"]["nll"] for r in rows}

out = dict(
    config=dict(ablation="a1_a2", budgets=BUDGETS, seeds=10,
                metric="test NLL (raw)", note=(
                    "A1 is an ablation: no pre-registered hypothesis or "
                    "equivalence delta applies; exact sign-flip permutation "
                    "p-values and 95% t-CIs are reported descriptively")),
    a1={}, a2={},
)

seeds = list(range(10))
for B in BUDGETS:
    es = np.array([r["es"]["nll"] for r in a1[B]])
    fin = np.array([r["final"]["nll"] for r in a1[B]])
    es_ts = np.array([r["es"]["ts"]["nll"] for r in a1[B]])
    fin_ts = np.array([r["final"]["ts"]["nll"] for r in a1[B]])
    be = np.array([r["best_epoch"] for r in a1[B]])
    de = np.array([de_k2[B][s] for s in seeds])

    d_t1 = es - fin
    gap_es = de - es          # allocation gap under early-stop rule
    gap_fin = de - fin        # allocation gap under final-weights rule

    out["a1"][f"B{B}"] = dict(
        es_nll=msp(es), final_nll=msp(fin),
        es_ts_nll=msp(es_ts), final_ts_nll=msp(fin_ts),
        best_epoch=dict(mean=float(be.mean()), sd=float(be.std(ddof=1)),
                        median=float(np.median(be)),
                        min=int(be.min()), max=int(be.max()),
                        values=[int(x) for x in be]),
        t1_es_vs_final=dict(diff=msp(d_t1), ci95=ci95(d_t1),
                            p=perm_test_paired(d_t1)),
        t4_gap_de_minus_es=dict(diff=msp(gap_es), ci95=ci95(gap_es),
                                p=perm_test_paired(gap_es)),
        t4_gap_de_minus_final=dict(diff=msp(gap_fin), ci95=ci95(gap_fin),
                                   p=perm_test_paired(gap_fin)),
        per_seed=dict(es=[float(x) for x in es], final=[float(x) for x in fin],
                      best_epoch=[int(x) for x in be],
                      de_k2=[float(x) for x in de]),
    )

# T2/T3 across budgets (paired across the same 10 seeds)
def across(key):
    res = {}
    for (B1, B2) in [(30, 60), (60, 120), (30, 120)]:
        x = np.array([r[key]["nll"] for r in a1[B1]])
        y = np.array([r[key]["nll"] for r in a1[B2]])
        d = y - x
        res[f"{B1}_to_{B2}"] = dict(diff=msp(d), ci95=ci95(d),
                                    p=perm_test_paired(d))
    return res


out["a1"]["t2_es_across_budgets"] = across("es")
out["a1"]["t3_final_across_budgets"] = across("final")

# ---------------------------------------------------------------- load A2
a2 = json.load(open(A2FILE))
for fam, r in a2["families"].items():
    lat = {}
    for bsz in (1, 256):
        rows = r[f"latency_batch{bsz}"]
        lat[f"batch{bsz}"] = dict(
            by_k={str(row["k"]): dict(
                median_ms=row["median_s"] * 1e3,
                mean_ms=row["mean_s"] * 1e3, sd_ms=row["sd_s"] * 1e3,
                per_sample_us=row["per_sample_s"] * 1e6) for row in rows},
            linear=dict(
                slope_ms_per_member=r[f"latency_batch{bsz}_linear"]["slope_s_per_member"] * 1e3,
                r2=r[f"latency_batch{bsz}_linear"]["r2"]))
    out["a2"][fam] = dict(
        params_per_model=r["params_per_model"],
        fp32_mb_per_model=r["fp32_bytes_per_model"] / 1e6,
        storage_mb_by_k={k: v / 1e6 for k, v in r["storage_bytes_by_k"].items()},
        rss_delta_8models_mb=r["rss_delta_8models_mb"], latency=lat)

with open(os.path.join(A1DIR, "analysis_v5_a1a2.json"), "w") as f:
    json.dump(out, f, indent=1)

# ---------------------------------------------------------------- CSVs
with open(os.path.join(A1DIR, "summary_v5_a1.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["budget", "cond", "nll_mean", "nll_sd", "ts_nll_mean",
                "ts_nll_sd", "best_epoch_mean", "best_epoch_sd"])
    for B in BUDGETS:
        r = out["a1"][f"B{B}"]
        w.writerow([B, "single_es", r["es_nll"]["mean"], r["es_nll"]["sd"],
                    r["es_ts_nll"]["mean"], r["es_ts_nll"]["sd"],
                    r["best_epoch"]["mean"], r["best_epoch"]["sd"]])
        w.writerow([B, "single_final", r["final_nll"]["mean"],
                    r["final_nll"]["sd"], r["final_ts_nll"]["mean"],
                    r["final_ts_nll"]["sd"], "", ""])

with open(os.path.join(A1DIR, "summary_v5_a2_latency.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "batch", "k", "median_ms", "mean_ms", "sd_ms",
                "per_sample_us"])
    for fam in ("mlp_fmnist", "cnn_cifar10"):
        for bsz in (1, 256):
            for k, row in out["a2"][fam]["latency"][f"batch{bsz}"]["by_k"].items():
                w.writerow([fam, bsz, k, f"{row['median_ms']:.4f}",
                            f"{row['mean_ms']:.4f}", f"{row['sd_ms']:.4f}",
                            f"{row['per_sample_us']:.3f}"])

with open(os.path.join(A1DIR, "summary_v5_a2_cost.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["family", "params_per_model", "fp32_mb_per_model",
                "storage_mb_k1", "storage_mb_k2", "storage_mb_k3",
                "storage_mb_k5", "storage_mb_k8", "rss_delta_8models_mb"])
    for fam in ("mlp_fmnist", "cnn_cifar10"):
        r = out["a2"][fam]
        st = r["storage_mb_by_k"]
        w.writerow([fam, r["params_per_model"], f"{r['fp32_mb_per_model']:.4f}",
                    f"{st['1']:.4f}", f"{st['2']:.4f}", f"{st['3']:.4f}",
                    f"{st['5']:.4f}", f"{st['8']:.4f}",
                    f"{r['rss_delta_8models_mb']:.2f}"])

# ---------------------------------------------------------------- summary
print("=" * 72)
print("T1  ES vs final (paired, per budget)")
for B in BUDGETS:
    r = out["a1"][f"B{B}"]["t1_es_vs_final"]
    print(f"  B{B:3d}: ES {out['a1'][f'B{B}']['es_nll']['mean']:.4f} "
          f"vs final {out['a1'][f'B{B}']['final_nll']['mean']:.4f} | "
          f"diff {r['diff']['mean']:+.4f} sd {r['diff']['sd']:.4f} "
          f"CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p']:.4f}")
print("T2  ES across budgets (RQ5 core)")
for k, r in out["a1"]["t2_es_across_budgets"].items():
    print(f"  {k}: diff {r['diff']['mean']:+.4f} "
          f"CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p']:.4f}")
print("T3  final across budgets (control)")
for k, r in out["a1"]["t3_final_across_budgets"].items():
    print(f"  {k}: diff {r['diff']['mean']:+.4f} "
          f"CI [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p={r['p']:.4f}")
print("T4  allocation gap DE(k2)-Single under both rules")
for B in BUDGETS:
    r1 = out["a1"][f"B{B}"]["t4_gap_de_minus_es"]
    r2 = out["a1"][f"B{B}"]["t4_gap_de_minus_final"]
    print(f"  B{B:3d}: ES-rule {r1['diff']['mean']:+.4f} (p={r1['p']:.4f}) | "
          f"final-rule {r2['diff']['mean']:+.4f} (p={r2['p']:.4f})")
print("T5  best_epoch")
for B in BUDGETS:
    be = out["a1"][f"B{B}"]["best_epoch"]
    print(f"  B{B:3d}: mean {be['mean']:.1f} sd {be['sd']:.1f} "
          f"range [{be['min']},{be['max']}] values {be['values']}")
print("=" * 72)
print("A2 H6 latency (median ms) and linearity")
for fam in ("mlp_fmnist", "cnn_cifar10"):
    r = out["a2"][fam]
    print(f"  {fam}: {r['params_per_model']:,} params, "
          f"{r['fp32_mb_per_model']:.3f} MB/model, "
          f"RSS(8 models)={r['rss_delta_8models_mb']:.1f} MB")
    for bsz in (1, 256):
        lat = r["latency"][f"batch{bsz}"]
        ks = sorted(int(k) for k in lat["by_k"])
        line = ", ".join(f"k={k}:{lat['by_k'][str(k)]['median_ms']:.3f}"
                         for k in ks)
        print(f"    batch={bsz}: {line}")
        print(f"      slope {lat['linear']['slope_ms_per_member']:.3f} ms/member, "
              f"R2={lat['linear']['r2']:.6f}")
print("wrote analysis_v5_a1a2.json + 3 CSVs to results/a1/")
