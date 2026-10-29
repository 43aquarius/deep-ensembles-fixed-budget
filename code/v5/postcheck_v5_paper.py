#!/usr/bin/env python3
"""Post-compile numerical consistency check for the v5 manuscript.

Extracts the PDF text and verifies that every headline number quoted in
the paper matches the canonical analysis artifacts (S1+S2, S3, A1/A2),
plus structural checks (figures, tables, sections, unresolved refs).
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
PAPER_DIR = os.environ.get("V5_PAPER_DIR",
                           os.path.join(ROOT, "manuscript-source"))
import json
import re
import subprocess

SRC = os.path.join(PAPER_DIR, "main.tex")
PDF = os.path.join(PAPER_DIR, "main.pdf")

V5 = os.path.join(ROOT, "results", "v5", "analysis_v5.json")
S3 = os.path.join(ROOT, "results", "s3", "analysis_v5_s3.json")
A1A2 = os.path.join(ROOT, "results", "a1", "analysis_v5_a1a2.json")

d = json.load(open(V5))
s3 = json.load(open(S3))
a1a2 = json.load(open(A1A2))

fail = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fail.append(name)


# ---- extract PDF text via pdftotext if available, else pdftotext via python
try:
    txt = subprocess.run(["pdftotext", PDF, "-"], capture_output=True,
                          text=True, timeout=60).stdout
    _used_pdftotext = True
except FileNotFoundError:
    import pypdf
    r = pypdf.PdfReader(PDF)
    txt = "\n".join((p.extract_text() or "") for p in r.pages)

# normalise typographic minus and thin spaces so numeric matching is
# robust to PDF extraction artefacts
for _bad, _good in (("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-"),
                   ("\u2009", ""), ("\u202f", ""), ("\u00a0", "")):
    txt = txt.replace(_bad, _good)

txt = re.sub(r"(\d),(\d{3})", r"\1\2", txt)  # 109,386 -> 109386

src = open(SRC).read()

# ---- helpers
def F(x, nd=4):
    return f"{x:.{nd}f}"


def summ(B, k, cond, metric="nll"):
    for r in d["summary"]:
        if (r["B"], r["k"], r["cond"], r["metric"]) == (B, k, cond, metric):
            return r["mean"]
    raise KeyError((B, k, cond, metric))


def s3s(dom, B, k, cond, metric="nll"):
    for r in s3["summary"]:
        if (r["domain"], r["B"], r["k"], r["cond"], r["metric"]) == (dom, B, k, cond, metric):
            return r["mean"]
    raise KeyError((dom, B, k, cond, metric))


# ---- 1. structural checks
check("C1 title present", "Split or stretch the budget" in txt)
check("C2 six sections", all(re.search(r"^\d+\s+" + t, txt, re.M) or t in txt
                             for t in ["Introduction", "Related work",
                                       "Study design", "Results",
                                       "Discussion", "Conclusion"]),
      "")
check("C3 five figure files included", all(f in src for f in
      ["fig1_phase", "fig2_mechanism", "fig3_temperature", "fig4_ood",
       "fig5_cost"]))
check("C4 five figure captions", src.count("\\begin{figure}") == 5)
check("C5 six tables", src.count("\\begin{table}") == 6,
      f"found {src.count(chr(92)+'begin{table}')}")
check("C6 no unresolved refs", "??" not in txt)
check("C7 bibliography items", src.count("\\bibitem") == 25,
      f"found {src.count(chr(92)+'bibitem')}")

# ---- 2. main-grid numbers (Table 2 spot checks: all 14 rows)
rows = [(30, 2), (30, 3), (30, 5), (60, 2), (60, 3), (60, 5),
        (120, 2), (120, 3), (120, 5), (120, 8),
        (240, 2), (240, 3), (240, 5), (240, 8)]
ok = True
for B, k in rows:
    for cond in ("single", "deepens"):
        v = F(summ(B, k, cond))
        if v not in txt:
            ok = False
            print(f"   missing in PDF: B{B} k{k} {cond} {v}")
check("N1 all 28 grid NLL values present in PDF", ok)

# ---- 3. headline statistics
h1 = d["H1"]
check("N2 k-curve B120 k8 = 0.2909",
      F(h1["k_curve_de"]["B120"]["8"]) in txt)
check("N3 B240 k8-k5 gap -0.0355",
      F(-d["H1"]["tost_B240_k5_k8"]["mean_diff"]) in txt)
check("N4 B120 TOST equivalent p 8.6e-6",
      "8.6" in txt and "10^{-6}" in src or "8.6 \\times 10^{-6}" in src)

h2 = d["H2"]["gap_by_B"]
for k, B, want in [("k2", 30, "-0.073"), ("k2", 120, "-0.2515"),
                   ("k3", 120, "-0.332"), ("k3", 240, "-0.247")]:
    v = h2[k][f"B{B}"]["mean"]
    check(f"N5 H2 {k} B{B} = {want}",
          (want in txt) or F(v) in txt or F(v, 3) in txt,
          f"computed {v:.4f}")

sg = [summ(B, 2, "single") for B in (30, 60, 120, 240)]
check("N6 single NLL 0.3795/0.5265/0.6672/0.6564",
      all(F(v) in txt for v in sg))

# ---- 4. S3 numbers
check("N7 mnist k2 DE 0.0589", F(s3s("mnist", 120, 2, "deepens")) in txt)
check("N8 cifar k5 DE 0.6302", F(s3s("cifar", 60, 5, "deepens")) in txt)
h4a = s3["H4a"]["k2"]
check("N9 H4a mnist -0.0086", F(h4a["mean_diff"]) in txt)
h4b = s3["H4b"]["k2"]
check("N10 H4b cifar -1.1406", F(h4b["mean_diff"]) in txt)
h4b5 = s3["H4b"]["k5"]
check("N11 H4b k5 -1.3656", F(h4b5["mean_diff"]) in txt)
check("N12 G3 verdict PASS", "G3" in txt and "4/4" in txt)

# ---- 5. A1 numbers
a1 = a1a2["a1"]
for B, es, fin in [(30, 0.3434, 0.3795), (60, 0.3491, 0.5265),
                   (120, 0.3506, 0.6672)]:
    blk = a1[f"B{B}"]
    check(f"N13 A1 B{B} ES {es}", F(blk["es_nll"]["mean"]) in txt)
    check(f"N13 A1 B{B} final {fin}", F(blk["final_nll"]["mean"]) in txt)
t4 = a1["B120"]["t4_gap_de_minus_es"]
check("N14 A1 B120 DE-ES +0.0651", F(t4["diff"]["mean"]) in txt)
t1 = a1["B120"]["t1_es_vs_final"]
check("N15 A1 B120 ES-final -0.3166",
      F(t1["diff"]["mean"]) in txt or F(t1["diff"]["mean"], 3) in txt)
check("N16 40-fold claim", "40" in txt and "fold" in txt)

# ---- 6. A2 numbers
a2 = a1a2["a2"]
mlp = a2["mlp_fmnist"]
cnn = a2["cnn_cifar10"]
check("N17 MLP params 109,386", "109" in txt and "386" in txt)
check("N18 MLP slope 0.48 ms",
      F(mlp["latency"]["batch256"]["linear"]["slope_ms_per_member"], 2) in txt)
check("N19 CNN slope 57.0 ms",
      F(cnn["latency"]["batch256"]["linear"]["slope_ms_per_member"], 1) in txt)
check("N20 storage 0.438/0.135",
      "0.438" in txt and "0.135" in txt)

# ---- 7. verdicts and key phrases
check("P1 only-tested-setting flip claim",
      "only tested setting" in txt)
check("P2 sweet-spot band claim", "10" in txt and "30" in txt and
      "sweet spot" in txt)
check("P3 non-monotone registered deviation",
      "non-monotone" in txt)
check("P4 LLM disclosure present", "large-language-model" in txt)

# ---- 8. study-scale claims (added in v5.0.1; guards against the 22/250 erratum)
import glob as _glob
_v5 = _glob.glob(os.path.join(ROOT, "results", "v5", "experiment_v5_*.jsonl"))
_s3 = _glob.glob(os.path.join(ROOT, "results", "s3", "experiment_v5_*.jsonl"))
_a1 = _glob.glob(os.path.join(ROOT, "results", "a1", "experiment_v5_a1_*.jsonl"))
n_grid = len(_v5) + len(_s3)
n_abl = len(_a1)
n_rec = sum(sum(1 for _ in open(f)) for f in _v5 + _s3 + _a1)
check("N21a filesystem cells 21 (18 grid + 3 ablation)",
      n_grid + n_abl == 21 and n_grid == 18 and n_abl == 3,
      f"(grid {n_grid}, ablation {n_abl})")
check("N21b filesystem replicate records 210", n_rec == 210, f"({n_rec})")
check("N21c per-stage cells S1+S2=14 S3=4", len(_v5) == 14 and len(_s3) == 4,
      f"(v5 {len(_v5)}, s3 {len(_s3)})")
check("N21d paper claims 18 grid + 3 ablation",
      bool(re.search(r"18 grid cells and 3 ablation cells", txt)))
check("N21e paper claims 210 replicate",
      bool(re.search(r"210 replicate", txt)))
check("N21f banned counts absent (22 grid / 250 replicate)",
      not re.search(r"(?<!\d)(22 grid|250 replicate)", txt))
check("N21g per-axis breakdown present (14 + 4 + 3 cells)",
      bool(re.search(r"14 grid cells", txt)) and
      len(re.findall(r"(?<![\d.])4 cells", txt)) >= 2 and
      bool(re.search(r"(?<![\d.])3 cells", txt)))

print()
# pdftotext emits exactly one \f per page (including the last); the pypdf
# fallback text has no \f, so use its reader's page count directly
print("pages:", txt.count(chr(12)) if _used_pdftotext else len(r.pages))
print("POSTCHECK " + ("ALL PASS" if not fail else f"FAIL: {fail}"))
raise SystemExit(0 if not fail else 1)
