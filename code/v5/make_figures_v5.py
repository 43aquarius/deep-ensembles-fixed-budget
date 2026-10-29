#!/usr/bin/env python3
"""v5 paper figures: the three-axis allocation phase diagram.

Axes: budget B (S1/S2) x domain/regime (S3) x selection rule (A1).
Style: Okabe-Ito palette, Liberation Sans, vector PDF + 600 dpi PNG,
183 mm main figures; all seed-level observations shown.

Outputs -> <repo>/figures/fig{1..5}_*.pdf/.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
for fp in ["/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
           "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
    try:
        fm.fontManager.addfont(fp)
    except Exception:
        pass
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIGDIR = os.path.join(ROOT, "figures")
V5DIR = os.path.join(ROOT, "results", "v5")
S3DIR = os.path.join(ROOT, "results", "s3")
A1DIR = os.path.join(ROOT, "results", "a1")

d = json.load(open(f"{V5DIR}/analysis_v5.json"))
s3 = json.load(open(f"{S3DIR}/analysis_v5_s3.json"))
a1a2 = json.load(open(f"{A1DIR}/analysis_v5_a1a2.json"))

BCOL = {30: "#E69F00", 60: "#009E73", 120: "#0072B2", 240: "#CC79A7"}
COND_COL = {"single": "#999999", "deepens": "#0072B2",
            "snapshot": "#E69F00", "swa": "#009E73"}
LBL = {"single": "Single", "deepens": "DeepEns",
       "snapshot": "Snapshot", "swa": "SWA"}
MM = 1 / 25.4


def save(fig, name, dpi=600):
    fig.savefig(f"{FIGDIR}/{name}.pdf")
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=dpi)
    plt.close(fig)
    print("wrote", name)


def cell_seeds(fam, B, k, cond="deepens", key="nll", ts=False):
    """Seed-level values from raw jsonl (FMNIST v5 grid)."""
    out = []
    with open(f"{V5DIR}/experiment_v5_{fam}_B{B}_k{k}.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r[cond]
            out.append(v["ts"][key] if ts else v[key])
    return np.array(out)


def s3_seeds(domain, B, k, cond="deepens", key="nll", ts=False):
    out = []
    with open(f"{S3DIR}/experiment_v5_{domain}_B{B}_k{k}.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r[cond]
            out.append(v["ts"][key] if ts else v[key])
    return np.array(out)


summ = {(r["B"], r["k"], r["cond"], r["metric"]): (r["mean"], r["sd"])
        for r in d["summary"]}
s3summ = {(r["domain"], r["B"], r["k"], r["cond"], r["metric"]):
          (r["mean"], r["sd"]) for r in s3["summary"]}
BUDGETS = [30, 60, 120, 240]
KS = {30: [2, 3, 5], 60: [2, 3, 5], 120: [2, 3, 5, 8], 240: [2, 3, 5, 8]}

# ══════════════════════════ FIG 1: phase diagram ══════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM),
                         constrained_layout=True)

# (a) k-curves by budget (DE NLL, seed dots)
ax = axes[0]
for B in BUDGETS:
    ks, ms, allx, ally = [], [], [], []
    for k in KS[B]:
        seeds = cell_seeds("fmnist", B, k)
        m = seeds.mean()
        ms.append(m)
        allx += [k] * len(seeds)
        ally += list(seeds)
    ax.plot(KS[B], ms, "-", color=BCOL[B], lw=1.1, marker="o", ms=2.6,
            label=f"B = {B}", zorder=3)
    ax.plot(allx, ally, ".", color=BCOL[B], ms=1.3, alpha=0.45, zorder=2)
ax.set_xlabel("Ensemble size $k$ (members)")
ax.set_ylabel("DeepEns test NLL")
ax.set_xticks([2, 3, 5, 8])
ax.set_title("(a) $k$-curves: optimum shifts with budget", loc="left")
ax.legend(frameon=False, handlelength=1.4, labelspacing=0.25)
ax.annotate("$k^{*}$: 3$\\rightarrow$5$\\rightarrow$8$\\rightarrow$$\\geq$8",
            xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
            fontsize=6.6, color="0.25")

# (b) regime axis: |DE-SG| by domain, log scale
ax = axes[1]
reg = [("MNIST\nB120", "mnist"), ("Fashion-MNIST\nB120", "fmnist"),
       ("CIFAR-10\nB60", "cifar")]
xpos = {0: [0.8, 1.2], 1: [1.8, 2.2], 2: [2.8, 3.2]}
domB = {"mnist": 120, "fmnist": 120, "cifar": 60}
for i, (lab, dom) in enumerate(reg):
    for j, k in enumerate((2, 5)):
        if dom == "fmnist":
            de = cell_seeds("fmnist", 120, k)
            sg = cell_seeds("fmnist", 120, k, cond="single")
        else:
            de = s3_seeds(dom, domB[dom], k)
            sg = s3_seeds(dom, domB[dom], k, cond="single")
        adv = -(de - sg)          # positive = DE better
        x = xpos[i][j] + (0.0 if k == 2 else 0.0)
        ax.plot([x] * len(adv), adv, ".", color=COND_COL["deepens"],
                ms=2.2, alpha=0.5, zorder=2)
        m = adv.mean()
        ax.plot(x, m, "s", color=COND_COL["deepens"], ms=4.5, zorder=3)
        ax.annotate(f"k{k}", (x, m), xytext=(3.5, 0), textcoords="offset points",
                    fontsize=6.0, color="0.25")
ax.axhline(0.020, color="0.4", lw=0.7, ls=":")
ax.annotate("attenuation boundary 0.020", (0.02, 0.020), xycoords="axes fraction",
            xytext=(2, 1.5), textcoords="offset points", fontsize=6.0,
            color="0.35")
ax.set_yscale("log")
ax.set_xticks([1.0, 2.0, 3.0])
ax.set_xticklabels([r[0] for r in reg])
ax.set_xlim(0.5, 3.6)
ax.set_ylabel("DeepEns $-$ Single, NLL (log)")
ax.set_title("(b) Regime axis: advantage magnitude", loc="left")
ax.annotate("near-saturated\n$\\rightarrow$ overconfident",
            xy=(0.99, 0.06), xycoords="axes fraction", ha="right",
            fontsize=6.6, color="0.25")

# (c) selection-rule axis (A1)
ax = axes[2]
Bs = [30, 60, 120]
x = np.arange(3)
a1 = a1a2["a1"]
es_m = [a1[f"B{B}"]["es_nll"]["mean"] for B in Bs]
es_sd = [a1[f"B{B}"]["es_nll"]["sd"] for B in Bs]
fin_m = [a1[f"B{B}"]["final_nll"]["mean"] for B in Bs]
fin_sd = [a1[f"B{B}"]["final_nll"]["sd"] for B in Bs]
de_m = [summ[(B, 2, "deepens", "nll")][0] for B in Bs]
de_sd = [summ[(B, 2, "deepens", "nll")][1] for B in Bs]
ax.errorbar(x - 0.14, fin_m, yerr=fin_sd, fmt="o-", color="#999999",
            lw=1.1, ms=3.2, capsize=1.6, label="Single (final weights)")
ax.errorbar(x, es_m, yerr=es_sd, fmt="s-", color="#D55E00",
            lw=1.1, ms=3.2, capsize=1.6,
            label="Single (early-stopped)")
ax.errorbar(x + 0.14, de_m, yerr=de_sd, fmt="D-", color="#0072B2",
            lw=1.1, ms=3.2, capsize=1.6, label="DeepEns ($k$=2)")
ax.set_xticks(x)
ax.set_xticklabels([f"B = {B}" for B in Bs])
ax.set_ylabel("Test NLL")
ax.set_title("(c) Selection-rule axis (B120 ablation)", loc="left")
ax.legend(frameon=False, handlelength=1.4, labelspacing=0.25, loc="upper left")
ax.annotate("ES-Single beats\nDeepEns($k{=}2$)",
            xy=(2.14, 0.3506), xytext=(1.18, 0.29),
            fontsize=6.4, color="#D55E00",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="#D55E00"))
save(fig, "fig1_phase")

# ══════════════════════════ FIG 2: mechanism ══════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 62 * MM),
                         constrained_layout=True)

# (a) Single NLL vs B (saturation) with DE(k2) for contrast
ax = axes[0]
sg_m = [summ[(B, 2, "single", "nll")][0] for B in BUDGETS]
sg_sd = [summ[(B, 2, "single", "nll")][1] for B in BUDGETS]
dk2 = [summ[(B, 2, "deepens", "nll")][0] for B in BUDGETS]
dk2_sd = [summ[(B, 2, "deepens", "nll")][1] for B in BUDGETS]
ax.errorbar(BUDGETS, sg_m, yerr=sg_sd, fmt="o-", color="#999999",
            lw=1.1, ms=3.4, capsize=1.8, label="Single")
ax.errorbar(BUDGETS, dk2, yerr=dk2_sd, fmt="s-", color="#0072B2",
            lw=1.1, ms=3.4, capsize=1.8, label="DeepEns ($k$=2)")
ax.set_xscale("log")
ax.set_xticks(BUDGETS)
ax.set_xticklabels([str(B) for B in BUDGETS])
ax.set_xlabel("Total budget $B$ (epochs, log)")
ax.set_ylabel("Test NLL")
ax.set_title("(a) Single saturates; the gap is non-monotone", loc="left")
ax.legend(frameon=False, handlelength=1.4, labelspacing=0.25)
ax.annotate("Single worsens\nthen plateaus", xy=(150, 0.662),
            xytext=(34, 0.72), fontsize=6.4, color="0.3",
            arrowprops=dict(arrowstyle="->", lw=0.7, color="0.5"))

# (b) member-length sweet spot: DE NLL vs e=B/k for all cells
ax = axes[1]
for B in BUDGETS:
    es, ys = [], []
    for k in KS[B]:
        es.append(B / k)
        ys.append(summ[(B, k, "deepens", "nll")][0])
    ax.plot(es, ys, "o-", color=BCOL[B], lw=1.0, ms=3.0,
            label=f"B = {B}", zorder=3)
ax.axvspan(10, 30, color="0.92", zorder=1)
ax.annotate("sweet spot\n$e\\in[10,30]$", xy=(20, 0.493),
            fontsize=6.6, ha="center", color="0.3")
ax.set_xscale("log")
ax.set_xticks([5, 10, 20, 40, 60, 120])
ax.set_xticklabels(["5", "10", "20", "40", "60", "120"])
ax.set_xlabel("Member training length $e = B/k$ (epochs, log)")
ax.set_ylabel("DeepEns test NLL")
ax.set_ylim(0.27, 0.53)
ax.set_title("(b) Member-length sweet spot", loc="left")
ax.legend(frameon=False, handlelength=1.4, labelspacing=0.25)

# (c) DE-SG gap vs B for k in {2,3,5} (non-monotone at 240)
ax = axes[2]
h2 = d["H2"]["gap_by_B"]
mk = {"2": "o", "3": "s", "5": "D"}
for k in ("2", "3", "5"):
    xs, ys, lo, hi = [], [], [], []
    for B in BUDGETS:
        b = h2[f"k{k}"][f"B{B}"]
        xs.append(B)
        ys.append(b["mean"])
        lo.append(b["ci95"][0])
        hi.append(b["ci95"][1])
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo),
                              np.array(hi) - np.array(ys)],
                fmt=mk[k] + "-", color=BCOL[{2: 30, 3: 60, 5: 120}[int(k)]],
                lw=1.0, ms=3.2, capsize=1.8, label=f"$k$ = {k}")
ax.axhline(0, color="0.6", lw=0.7)
ax.set_xscale("log")
ax.set_xticks(BUDGETS)
ax.set_xticklabels([str(B) for B in BUDGETS])
ax.set_xlabel("Total budget $B$ (epochs, log)")
ax.set_ylabel("DeepEns $-$ Single (NLL)")
ax.set_title("(c) Amplification peaks at $B{=}120$", loc="left")
ax.legend(frameon=False, handlelength=1.4, labelspacing=0.25)
save(fig, "fig2_mechanism")

# ══════════════════════════ FIG 3: temperature scaling ══════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(183 * MM, 105 * MM),
                         constrained_layout=True)
for i, B in enumerate(BUDGETS):
    for j, (metric, ttl) in enumerate((("nll", "raw NLL"),
                                       ("ts_nll", "after TS"))):
        ax = axes[j, i]
        ks, ms = [], []
        for k in KS[B]:
            ms.append(summ[(B, k, "deepens", metric)][0])
            ks.append(k)
        ax.plot(ks, ms, "o-", color=COND_COL["deepens"], lw=1.1, ms=3.0)
        sgv = summ[(B, 2, "single", metric)][0]
        ax.axhline(sgv, color="#999999", lw=1.0, ls="--")
        ax.annotate("Single", xy=(0.97, sgv), xycoords=("axes fraction", "data"),
                    xytext=(0, -6), textcoords="offset points", ha="right",
                    fontsize=6.2, color="0.35")
        kmin = ks[int(np.argmin(ms))]
        ax.plot([kmin], [min(ms)], "o", ms=7, mfc="none", mec="#D55E00", mew=1.1)
        ax.set_title(f"B = {B} ({ttl})" if j == 0 else ttl, loc="left",
                     fontsize=7.2)
        ax.set_xticks(ks)
        if j == 1:
            ax.set_xlabel("Ensemble size $k$")
        if i == 0:
            ax.set_ylabel("DeepEns test NLL")
save(fig, "fig3_temperature")

# ══════════════════════════ FIG 4: OOD detection ══════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(120 * MM, 55 * MM),
                         constrained_layout=True)
ood = s3["ood_auroc_msp"]

# (a) MNIST-domain probes
ax = axes[0]
groups = [("Single", "single", "#999999"),
          ("DeepEns $k{=}2$", "deepens2", "#0072B2"),
          ("DeepEns $k{=}5$", "deepens5", "#0072B2")]
for gi, (lab, key, col) in enumerate(groups):
    pass  # built below from summary instead
# build from summary rows (metric auroc_msp_{probe})
probes = ["fmnist", "kmnist"]
width = 0.24
for ci, (cond_lab, cond, k, col, hatch) in enumerate([
        ("Single", "single", None, "#999999", None),
        ("DeepEns $k{=}2$", "deepens", 2, "#56B4E9", None),
        ("DeepEns $k{=}5$", "deepens", 5, "#0072B2", None)]):
    for pi, probe in enumerate(probes):
        if k is None:
            m = s3summ[("mnist", 120, 2, "single", f"auroc_msp_{probe}")][0]
            sd = s3summ[("mnist", 120, 2, "single", f"auroc_msp_{probe}")][1]
        else:
            m = s3summ[("mnist", 120, k, "deepens", f"auroc_msp_{probe}")][0]
            sd = s3summ[("mnist", 120, k, "deepens", f"auroc_msp_{probe}")][1]
        x = pi + (ci - 1) * width
        ax.bar(x, m, width * 0.9, color=col, yerr=sd, capsize=1.4,
               error_kw=dict(lw=0.7))
ax.set_xticks(range(len(probes)))
ax.set_xticklabels(["Fashion-MNIST\n(in-dist. for S1)", "KMNIST"])
ax.set_ylabel("OOD AUROC (MSP)")
ax.set_ylim(0.88, 0.97)
ax.set_title("(a) MNIST-trained models", loc="left")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in
           ("#999999", "#56B4E9", "#0072B2")],
          ["Single", "DeepEns $k{=}2$", "DeepEns $k{=}5$"],
          frameon=False, handlelength=1.2, labelspacing=0.25, loc="upper left")

# (b) CIFAR-trained models, CIFAR-100 probe
ax = axes[1]
conds = [("single", None, "#999999", "Single"),
         ("deepens", 2, "#56B4E9", "DeepEns $k{=}2$"),
         ("snapshot", 2, "#E69F00", "Snapshot $k{=}2$"),
         ("deepens", 5, "#0072B2", "DeepEns $k{=}5$"),
         ("snapshot", 5, "#F0C25A", "Snapshot $k{=}5$")]
for ci, (cond, k, col, lab) in enumerate(conds):
    kk = k if k else 2
    m = s3summ[("cifar", 60, kk, cond, "auroc_msp_cifar100")][0]
    sd = s3summ[("cifar", 60, kk, cond, "auroc_msp_cifar100")][1]
    ax.bar(ci, m, 0.62, color=col, yerr=sd, capsize=1.6,
           error_kw=dict(lw=0.7))
    ax.annotate(f"{m:.3f}", (ci, m + sd + 0.004), ha="center", fontsize=6.4)
sgl = s3summ[("cifar", 60, 2, "single", "auroc_msp_cifar100")][0]
ax.axhline(sgl, color="#999999", lw=0.9, ls="--")
ax.set_xticks(range(len(conds)))
ax.set_xticklabels([c[3].replace(" $k{=}", "\n$k{=}$") for c in conds],
                    fontsize=6.4)
ax.set_ylabel("OOD AUROC (MSP)")
ax.set_ylim(0.63, 0.80)
ax.set_title("(b) CIFAR-10-trained models (CIFAR-100 probe)", loc="left")
save(fig, "fig4_ood")

# ══════════════════════════ FIG 5: inference cost ══════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(120 * MM, 55 * MM),
                         constrained_layout=True)
ax = axes[0]
mlp = a1a2["a2"]["mlp_fmnist"]
cnn = a1a2["a2"]["cnn_cifar10"]
for fam, col, lab in ((mlp, "#0072B2", "MLP (Fashion-MNIST)"),
                      (cnn, "#E69F00", "SmallCNN (CIFAR-10)")):
    b256 = fam["latency"]["batch256"]
    ks = sorted(int(k) for k in b256["by_k"])
    med = [b256["by_k"][str(k)]["median_ms"] for k in ks]
    sd = [b256["by_k"][str(k)]["sd_ms"] for k in ks]
    reps = 30 if fam is cnn else 100
    q1 = [m - 1.96 * s / math.sqrt(reps) for m, s in zip(med, sd)]
    q2 = [m + 1.96 * s / math.sqrt(reps) for m, s in zip(med, sd)]
    ax.plot(ks, med, "o-", color=col, lw=1.1, ms=3.2, label=lab)
    ax.fill_between(ks, q1, q2, color=col, alpha=0.18, lw=0)
    sl = b256["linear"]["slope_ms_per_member"]
    r2 = b256["linear"]["r2"]
    base = med[0] - sl * ks[0]
    xx = np.linspace(1, 8, 10)
    ax.plot(xx, sl * xx + base, ls=":", lw=0.9, color=col)
    ax.annotate(f"{sl:.2f} ms/member ($R^2$={r2:.3f})",
                xy=(ks[-1], med[-1]), xytext=(-4, 7),
                textcoords="offset points", ha="right", fontsize=6.2,
                color=col)
ax.set_yscale("log")
ax.set_xticks([1, 2, 3, 5, 8])
ax.set_xlabel("Ensemble size $k$")
ax.set_ylabel("Batch-256 latency, ms (log)")
ax.set_title("(a) Prediction averaging cost (batch 256)", loc="left")
ax.legend(frameon=False, handlelength=1.4, labelspacing=0.25,
          loc="upper left")

ax = axes[1]
ks = sorted(int(k) for k in mlp["storage_mb_by_k"])
ax.plot(ks, [mlp["storage_mb_by_k"][str(k)] for k in ks], "o-",
        color="#0072B2", lw=1.1, ms=3.2, label="MLP: 0.438 MB/model")
ax.plot(ks, [cnn["storage_mb_by_k"][str(k)] for k in ks], "s-",
        color="#E69F00", lw=1.1, ms=3.2, label="SmallCNN: 0.135 MB/model")
ax.set_xticks(ks)
ax.set_xlabel("Ensemble size $k$")
ax.set_ylabel("FP32 storage (MB)")
ax.set_title("(b) Storage is exactly $k\\times$", loc="left")
ax.legend(frameon=False, handlelength=1.4, labelspacing=0.25, loc="upper left")
save(fig, "fig5_cost")

print("ALL FIGURES DONE")
