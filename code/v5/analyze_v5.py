#!/usr/bin/env python3
"""v5 pre-registered analysis (Plan B, Doc 3 Section 6).

Analysis unit: seed (n=10 per cell), all contrasts seed-paired.
Primary endpoint: raw test NLL. Secondary: ECE-15, Brier, accuracy,
TS-NLL, OOD AUROC (MSP).

Family 1 (confirmation of v4 ordering, per-cell Holm over 5 contrasts):
  DE-SG, SN-SG, SWA-SG, DE-SN, SN-SWA
  exact sign-flip permutation test, two-sided.

Family 2 (THE core inferential family of this study, per-B Holm):
  all pairwise DE(k) vs DE(k') within each B (3+3+6 = 12 contrasts).

Equivalence (TOST): delta_NLL = 0.010, delta_ECE = 0.005; two one-sided
90% t-CIs on the paired mean difference; equivalence declared only when
the full 90% CI lies inside (+/- delta).

Hypothesis checks emitted for the stage gate:
  H1 saturation  : k-curve shape per B; TOST on B120 DE(k5) vs DE(k8)
  H2 amplification: DE-SG gap by B; decomposition into Single vs DE parts
  H3 snapshot gap : DE-SN gap as a function of k within each B
  H5 temperature   : TS-NLL family-2 contrasts vs raw

Outputs: results/v5/analysis_v5.json, summary_v5.csv, contrasts_v5.csv
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import itertools
import json
import math
import os

import numpy as np
from scipy import stats as st

V5 = os.path.join(ROOT, "results", "v5")
CELLS = [(30, 2), (30, 3), (30, 5),
         (60, 2), (60, 3), (60, 5),
         (120, 2), (120, 3), (120, 5), (120, 8),
         (240, 2), (240, 3), (240, 5), (240, 8)]
BUDGETS = [30, 60, 120, 240]
N = 10
DELTA = {"nll": 0.010, "ece15": 0.005}
METRICS = ["nll", "acc", "brier", "ece15", "ts_nll", "auroc_msp_mnist",
           "auroc_msp_kmnist"]
COND = ["single", "deepens", "snapshot", "swa"]


def load_cell(B, k):
    rows = []
    with open(os.path.join(V5, f"experiment_v5_fmnist_B{B}_k{k}.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["seed"])
    assert [r["seed"] for r in rows] == list(range(N)), f"B{B}k{k} bad seeds"
    return rows


def metric_vec(rows, cond, m):
    out = []
    for r in rows:
        c = r[cond]
        if m == "ts_nll":
            v = c["ts"]["nll"]
        elif m.startswith("auroc_msp_"):
            v = c[m.replace("auroc_msp_", "auroc_msp_")]
        else:
            v = c[m]
        out.append(v)
    return np.array(out, dtype=np.float64)


def perm_test_paired(d):
    """Exact two-sided sign-flip permutation test on paired diffs."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    obs = abs(d.mean())
    idx = np.arange(1 << n)
    signs = (((idx[:, None] >> np.arange(n)) & 1) * 2.0) - 1.0
    stat = np.abs((signs * d[None, :]).mean(axis=1))
    return float((stat >= obs - 1e-15).mean())


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = pvals[i] * (m - rank)
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def paired_ci(d, conf=0.90):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    tcrit = float(st.t.ppf(0.5 + conf / 2, n - 1))
    return mean, (mean - tcrit * se, mean + tcrit * se), se


def tost(d, delta):
    """Two one-sided tests: H0: |mean| >= delta vs H1: |mean| < delta.
    Returns (verdict, p_tost, ci90, mean)."""
    d = np.asarray(d, dtype=np.float64)
    n, mean = len(d), float(d.mean())
    se = float(d.std(ddof=1) / math.sqrt(n))
    if not np.isfinite(se) or se == 0:
        se = 1e-300
    df = n - 1
    t_low = (mean + delta) / se    # H0a: mean <= -delta
    t_high = (mean - delta) / se   # H0b: mean >= +delta
    p_low = 1.0 - st.t.cdf(t_low, df)    # P(T > t_low) small if mean >> -delta
    p_high = st.t.cdf(t_high, df)        # P(T < t_high) small if mean << +delta
    p_tost = float(max(p_low, p_high))
    _, ci, _ = paired_ci(d, 0.90)
    verdict = (p_tost < 0.05) and (ci[0] > -delta) and (ci[1] < delta)
    return bool(verdict), p_tost, ci, mean


# ---------------------------------------------------------------- load
data = {(B, k): load_cell(B, k) for B, k in CELLS}

# ---------------------------------------------------------------- summary
summary = []
for (B, k), rows in data.items():
    for cond in COND:
        for m in ["nll", "acc", "brier", "ece15"]:
            v = metric_vec(rows, cond, m)
            summary.append(dict(B=B, k=k, cond=cond, metric=m,
                                mean=float(v.mean()), sd=float(v.std(ddof=1))))
        v = metric_vec(rows, cond, "ts_nll")
        summary.append(dict(B=B, k=k, cond=cond, metric="ts_nll",
                            mean=float(v.mean()), sd=float(v.std(ddof=1))))
        for probe in ["mnist", "kmnist"]:
            v = metric_vec(rows, cond, f"auroc_msp_{probe}")
            summary.append(dict(B=B, k=k, cond=cond,
                                metric=f"auroc_msp_{probe}",
                                mean=float(v.mean()), sd=float(v.std(ddof=1))))

# ------------------------------------------------- family 1 (5 contrasts)
F1_NAMES = [("deepens", "single"), ("snapshot", "single"),
            ("swa", "single"), ("deepens", "snapshot"), ("snapshot", "swa")]
f1_rows = []
for (B, k), rows in data.items():
    cell = []
    for a, b in F1_NAMES:
        d = metric_vec(rows, a, "nll") - metric_vec(rows, b, "nll")
        p = perm_test_paired(d)
        mean, ci, se = paired_ci(d, 0.95)
        cell.append(dict(setting=f"B{B}k{k}", contrast=f"{a.upper()}-{b.upper()}",
                         metric="nll", mean_diff=mean, ci95=ci, p_raw=p,
                         n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum())))
    padj = holm([c["p_raw"] for c in cell])
    for c, pa in zip(cell, padj):
        c["p_holm"] = float(pa)
    f1_rows.extend(cell)

# ------------------------------------------------- family 2 (allocation)
f2_rows = []
for B in BUDGETS:
    ks = [k for (bb, k) in CELLS if bb == B]
    cell = []
    for k1, k2 in itertools.combinations(ks, 2):
        d = (metric_vec(data[(B, k2)], "deepens", "nll")
             - metric_vec(data[(B, k1)], "deepens", "nll"))
        p = perm_test_paired(d)
        mean, ci, se = paired_ci(d, 0.95)
        eq, pT, ci90, _ = tost(d, DELTA["nll"])
        cell.append(dict(setting=f"B{B}", contrast=f"DE(k{k2})-DE(k{k1})",
                         metric="nll", mean_diff=mean, ci95=ci,
                         p_raw=p, tost_equiv=eq, tost_p=pT, ci90=ci90,
                         n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum())))
    padj = holm([c["p_raw"] for c in cell])
    for c, pa in zip(cell, padj):
        c["p_holm"] = float(pa)
    f2_rows.extend(cell)

# TS-NLL family 2 (H5)
f2_ts_rows = []
for B in BUDGETS:
    ks = [k for (bb, k) in CELLS if bb == B]
    cell = []
    for k1, k2 in itertools.combinations(ks, 2):
        d = (metric_vec(data[(B, k2)], "deepens", "ts_nll")
             - metric_vec(data[(B, k1)], "deepens", "ts_nll"))
        p = perm_test_paired(d)
        mean, ci, se = paired_ci(d, 0.95)
        cell.append(dict(setting=f"B{B}", contrast=f"DE(k{k2})-DE(k{k1})",
                         metric="ts_nll", mean_diff=mean, ci95=ci, p_raw=p))
    padj = holm([c["p_raw"] for c in cell])
    for c, pa in zip(cell, padj):
        c["p_holm"] = float(pa)
    f2_ts_rows.extend(cell)

# ------------------------------------------------------------- H1 check
h1 = {"k_curve_de": {}, "tost_B120_k5_k8": None, "tost_B240_k5_k8": None}
for B in BUDGETS:
    ks = [k for (bb, k) in CELLS if bb == B]
    means = {k: float(metric_vec(data[(B, k)], "deepens", "nll").mean())
             for k in ks}
    h1["k_curve_de"][f"B{B}"] = means
d58 = (metric_vec(data[(120, 8)], "deepens", "nll")
       - metric_vec(data[(120, 5)], "deepens", "nll"))
eq, pT, ci90, mean = tost(d58, DELTA["nll"])
h1["tost_B120_k5_k8"] = dict(mean_diff=mean, ci90=ci90, tost_p=pT,
                             equivalent=eq)
# B240 k5 vs k8 TOST (S2 extrapolation cell; H1 saturation criterion)
d58_240 = (metric_vec(data[(240, 8)], "deepens", "nll")
           - metric_vec(data[(240, 5)], "deepens", "nll"))
eq4, pT4, ci904, mean4 = tost(d58_240, DELTA["nll"])
h1["tost_B240_k5_k8"] = dict(mean_diff=mean4, ci90=ci904, tost_p=pT4,
                             equivalent=eq4)
d58e_240 = (metric_vec(data[(240, 8)], "deepens", "ece15")
            - metric_vec(data[(240, 5)], "deepens", "ece15"))
eq4e, pT4e, ci904e, mean4e = tost(d58e_240, DELTA["ece15"])
h1["tost_B240_k5_k8_ece"] = dict(mean_diff=mean4e, ci90=ci904e, tost_p=pT4e,
                                 equivalent=eq4e)
# ECE version for the H1 TOST at B120
d58e = (metric_vec(data[(120, 8)], "deepens", "ece15")
        - metric_vec(data[(120, 5)], "deepens", "ece15"))
eqe, pTe, ci90e, meane = tost(d58e, DELTA["ece15"])
h1["tost_B120_k5_k8_ece"] = dict(mean_diff=meane, ci90=ci90e, tost_p=pTe,
                                 equivalent=eqe)

# ------------------------------------------------------------- H2 check
h2 = {"gap_by_B": {}, "decomposition": {}}
for k in (2, 3, 5):
    gaps = {}
    for B in BUDGETS:
        d = (metric_vec(data[(B, k)], "deepens", "nll")
             - metric_vec(data[(B, k)], "single", "nll"))
        p = perm_test_paired(d)
        mean, ci, _ = paired_ci(d, 0.95)
        gaps[f"B{B}"] = dict(mean=mean, ci95=ci, p_perm=p)
    h2["gap_by_B"][f"k{k}"] = gaps
# decomposition of the k=3 amplification (pre-registered RQ2 mechanism)
sg = {B: metric_vec(data[(B, 3)], "single", "nll").mean() for B in BUDGETS}
de = {B: metric_vec(data[(B, 3)], "deepens", "nll").mean() for B in BUDGETS}
tot = {f"{a}->{b}": (sg[b] - sg[a]) - (de[b] - de[a])
       for a, b in [(30, 60), (60, 120), (120, 240), (30, 240)]}
h2["decomposition"] = dict(single_nll=sg, deepens_nll=de,
                            gap_growth_total=tot,
                            single_share={kk: (sg[int(kk.split('->')[1])] - sg[int(kk.split('->')[0])]) / v
                                         for kk, v in tot.items()})
# fixed-e cross-B pair (e=15): B30k2 vs B120k8 (CONFOUNDED: k 2 vs 8)
d_fixed_e = (metric_vec(data[(120, 8)], "deepens", "nll")
             - metric_vec(data[(30, 2)], "deepens", "nll"))
eqf, pTf, ci90f, meanf = tost(d_fixed_e, DELTA["nll"])
h2["fixed_e15_B30k2_vs_B120k8"] = dict(
    mean_diff=meanf, ci90=ci90f, tost_p=pTf, equivalent=eqf,
    note="k differs (2 vs 8): exploratory only, confounded by ensemble size")
# member training length effect (k fixed=3): DE member NLL by B
h2["de_member_nll_by_B"] = {
    f"B{B}": float(np.mean([r["deepens"]["member_mean_nll"]
                            for r in data[(B, 3)]])) for B in BUDGETS}

# ------------------------------------------------------------- G2 gate
# Pre-registered G2 (Doc 3, Table 7-2): B240 continues the amplification
# trend = adjacent-budget-point 95% CIs of the DE-SG gap (k=3 anchor) do
# NOT overlap AND the direction is consistent (all negative = DE better).
g2 = {"gap_k3_by_B": {}, "verdict": None, "detail": None}
for B in BUDGETS:
    d = (metric_vec(data[(B, 3)], "deepens", "nll")
         - metric_vec(data[(B, 3)], "single", "nll"))
    mean, ci, _ = paired_ci(d, 0.95)
    g2["gap_k3_by_B"][f"B{B}"] = dict(mean=mean, ci95=ci)
adjacent = [(30, 60), (60, 120), (120, 240)]
no_overlap, same_dir = [], []
for a, b in adjacent:
    ca = g2["gap_k3_by_B"][f"B{a}"]["ci95"]
    cb = g2["gap_k3_by_B"][f"B{b}"]["ci95"]
    no_overlap.append(ca[1] < cb[0] or cb[1] < ca[0])
    same_dir.append((g2["gap_k3_by_B"][f"B{a}"]["mean"] < 0)
                     == (g2["gap_k3_by_B"][f"B{b}"]["mean"] < 0))
means = [g2["gap_k3_by_B"][f"B{B}"]["mean"] for B in BUDGETS]
monotone = all(means[i] > means[i + 1] for i in range(len(means) - 1))
g2_pass = all(no_overlap) and all(same_dir)
g2["verdict"] = "PASS" if g2_pass else "FAIL"
g2["detail"] = dict(
    adjacent_no_overlap=dict(zip([f"{a}-{b}" for a, b in adjacent],
                                  no_overlap)),
    direction_consistent=all(same_dir),
    means_monotonically_widening=monotone,
    interpretation=(
        "amplification claim retained across B30-240 (four points)"
        if g2_pass else
        "amplification claim must be downgraded to a local empirical "
        "observation per the pre-registered G2 failure rule"))

# ------------------------------------------------------------- H3 check
h3 = {}
for B in BUDGETS:
    ks = [k for (bb, k) in CELLS if bb == B]
    gaps = {}
    for k in ks:
        d = (metric_vec(data[(B, k)], "deepens", "nll")
             - metric_vec(data[(B, k)], "snapshot", "nll"))
        p = perm_test_paired(d)
        mean, ci, _ = paired_ci(d, 0.95)
        gaps[f"k{k}"] = dict(mean=mean, ci95=ci, p_perm=p,
                             n_pos=int((d > 0).sum()))
    h3[f"B{B}"] = gaps

# ------------------------------------------------------------- H5 check
h5 = {"de_sg_after_ts": {}, "b30_k_effect_after_ts": None,
      "b240_k_effect_after_ts": None}
for k in (2, 3, 5):
    gaps = {}
    for B in BUDGETS:
        d = (metric_vec(data[(B, k)], "deepens", "ts_nll")
             - metric_vec(data[(B, k)], "single", "ts_nll"))
        p = perm_test_paired(d)
        mean, ci, _ = paired_ci(d, 0.95)
        gaps[f"B{B}"] = dict(mean=mean, ci95=ci, p_perm=p)
    h5["de_sg_after_ts"][f"k{k}"] = gaps
# B30 flattening check: best-vs-rest k spread after TS
ks30 = [2, 3, 5]
ts_means = {k: metric_vec(data[(30, k)], "deepens", "ts_nll").mean()
            for k in ks30}
h5["b30_k_effect_after_ts"] = {f"k{k}": float(v) for k, v in ts_means.items()}
# B240 k-effect after TS (S2): does TS shift the optimal k at B240?
ks240 = [2, 3, 5, 8]
ts_means240 = {k: metric_vec(data[(240, k)], "deepens", "ts_nll").mean()
               for k in ks240}
h5["b240_k_effect_after_ts"] = {f"k{k}": float(v)
                                for k, v in ts_means240.items()}

# ------------------------------------------------------ direction counts
dircons = {}
for B in BUDGETS:
    ks = [k for (bb, k) in CELLS if bb == B]
    pairs = list(itertools.combinations(ks, 2))
    raw_dir = {f"k{k1}<k{k2}": int((metric_vec(data[(B, k2)], "deepens", "nll")
             < metric_vec(data[(B, k1)], "deepens", "nll")).sum())
               for k1, k2 in pairs}
    dircons[f"B{B}"] = raw_dir

# ---------------------------------------------------------------- write
out = dict(
    meta=dict(n_seeds=N, cells=[f"B{B}k{k}" for B, k in CELLS],
              delta=DELTA, test="exact sign-flip permutation, two-sided",
              ci="t 95% (raw), t 90% for TOST",
              versions=dict(torch="2.14.1+cpu", numpy="2.1.3")),
    summary=summary,
    family1=f1_rows,
    family2=f2_rows,
    family2_ts=f2_ts_rows,
    H1=h1, H2=h2, H3=h3, H5=h5, G2=g2,
    direction_counts_kcurves=dircons,
)
with open(os.path.join(V5, "analysis_v5.json"), "w") as f:
    json.dump(out, f, indent=1)

import csv
with open(os.path.join(V5, "summary_v5.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["B", "k", "cond", "metric", "mean", "sd"])
    w.writeheader()
    w.writerows(summary)
with open(os.path.join(V5, "contrasts_v5.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["family", "setting", "contrast", "metric",
                                     "mean_diff", "ci_lo", "ci_hi", "p_raw",
                                     "p_holm", "tost_equiv", "tost_p",
                                     "n_pos", "n_neg"])
    w.writeheader()
    for r in f1_rows:
        w.writerow(dict(family="F1", setting=r["setting"],
                        contrast=r["contrast"], metric="nll",
                        mean_diff=round(r["mean_diff"], 6),
                        ci_lo=round(r["ci95"][0], 6),
                        ci_hi=round(r["ci95"][1], 6),
                        p_raw=r["p_raw"], p_holm=round(r["p_holm"], 4),
                        tost_equiv="", tost_p="",
                        n_pos=r["n_pos"], n_neg=r["n_neg"]))
    for r in f2_rows:
        w.writerow(dict(family="F2", setting=r["setting"],
                        contrast=r["contrast"], metric="nll",
                        mean_diff=round(r["mean_diff"], 6),
                        ci_lo=round(r["ci95"][0], 6),
                        ci_hi=round(r["ci95"][1], 6),
                        p_raw=r["p_raw"], p_holm=round(r["p_holm"], 4),
                        tost_equiv=r["tost_equiv"], tost_p=round(r["tost_p"], 5),
                        n_pos=r["n_pos"], n_neg=r["n_neg"]))

# ------------------------------------------------------------ console
print("=" * 70)
print("FAMILY 2 (allocation, per-B Holm) - NLL")
for r in f2_rows:
    print(f"  {r['setting']:6s} {r['contrast']:16s} d={r['mean_diff']:+.4f} "
          f"[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] p_holm={r['p_holm']:.4f} "
          f"equiv={'YES' if r['tost_equiv'] else 'no'} "
          f"({r['n_pos']}+/ {r['n_neg']}-)")
print()
print("H1 k-curves (DeepEns mean NLL):")
for B, ms in h1["k_curve_de"].items():
    print(f"  {B}: " + "  ".join(f"k{k}={v:.4f}" for k, v in ms.items()))
t = h1["tost_B120_k5_k8"]
print(f"H1 TOST B120 k5 vs k8: d={t['mean_diff']:+.4f} ci90=[{t['ci90'][0]:+.4f},"
      f"{t['ci90'][1]:+.4f}] equiv={t['equivalent']}")
print()
print("H1 TOST B240 k5 vs k8: d=%+.4f ci90=[%+.4f,%+.4f] equiv=%s"
      % (h1["tost_B240_k5_k8"]["mean_diff"],
         h1["tost_B240_k5_k8"]["ci90"][0],
         h1["tost_B240_k5_k8"]["ci90"][1],
         h1["tost_B240_k5_k8"]["equivalent"]))
print()
print("H2 amplification (DE-SG gap, k=3):")
for B, g in h2["gap_by_B"]["k3"].items():
    print(f"  {B}: {g['mean']:+.4f} [{g['ci95'][0]:+.4f},{g['ci95'][1]:+.4f}] "
          f"p={g['p_perm']:.4f}")
print(f"  Single NLL by B: {h2['decomposition']['single_nll']}")
print(f"  DE(k3) NLL by B: {h2['decomposition']['deepens_nll']}")
print()
print("G2 gate (adjacent-budget CI separation, k=3 anchor):")
for B, g in g2["gap_k3_by_B"].items():
    print(f"  {B}: {g['mean']:+.4f} [{g['ci95'][0]:+.4f},{g['ci95'][1]:+.4f}]")
print(f"  verdict={g2['verdict']}  detail={g2['detail']}")
print()
print("H3 DE-SN gap by k:")
for B, gs in h3.items():
    print(f"  {B}: " + "  ".join(
        f"{kk}={g['mean']:+.4f}(p={g['p_perm']:.3f})" for kk, g in gs.items()))
print()
print("H5 DE-SG after TS (k=3):")
for B, g in h5["de_sg_after_ts"]["k3"].items():
    print(f"  {B}: {g['mean']:+.4f} p={g['p_perm']:.4f}")
print(f"  B30 TS-NLL by k: {h5['b30_k_effect_after_ts']}")
print()
print("direction counts (k-curve consistency):")
for B, dc in dircons.items():
    print(f"  {B}: {dc}")
