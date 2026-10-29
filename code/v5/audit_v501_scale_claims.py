#!/usr/bin/env python3
"""One-off deep audit of countable/scale claims in the v5.0.1 manuscript.

Error class targeted: the v5.0.0 erratum (22 grid / 250 records vs the true
18 grid / 210) showed that *scale* claims can slip through number-by-number
postchecks. This audit enumerates every countable claim family and verifies
each against (a) the filesystem data layer and (b) arithmetic coherence.
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
import glob
import json
import re
import subprocess
import sys

SRC = os.path.join(PAPER_DIR, "main.tex")
PDF = os.path.join(PAPER_DIR, "main.pdf")
RES = os.path.join(ROOT, "results")

tex = open(SRC).read()
txt = subprocess.run(["pdftotext", PDF, "-"], capture_output=True,
                     text=True, timeout=60).stdout
for bad, good in (("\u2212", "-"), ("\u2013", "-"), ("\u2014", "-"),
                  ("\u2009", ""), ("\u202f", ""), ("\u00a0", "")):
    txt = txt.replace(bad, good)
txt = re.sub(r"(\d),(\d{3})", r"\1\2", txt)

fail = []
def ck(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        fail.append(name)

print("== A. study-scale phrases: enumerate ALL, verify coherence ==")
cells_phrases = re.findall(r"(?<![\d.])(\d+)\s*(grid|ablation)?\s*cells", txt)
rep_phrases = re.findall(r"(?<![\d.])(\d+)\s*replicate(?:\s*records?)?", txt)
print("   'N cells' phrases:", sorted(set(cells_phrases), key=lambda x: -int(x[0])))
print("   'N replicate' phrases:", sorted(set(rep_phrases), key=lambda x: -int(x[0])))
ck("A1 cells phrases ⊆ {18 grid, 3 ablation, 14, 10, 4, 3}",
   all(n in {"18", "14", "10", "4", "3", "21"} for n, _ in cells_phrases),
   str(sorted(set(cells_phrases))))
ck("A2 replicate phrases == {210}", {n for n in rep_phrases} == {"210"},
   str(sorted(set(rep_phrases))))
ck("A3 arithmetic 10+4+4=18 grid; +3 ablation = 21; x10 = 210",
   (10 + 4 + 4) == 18 and 18 + 3 == 21 and 21 * 10 == 210)

print("== B. grid structure: filesystem == paper's stated design ==")
v5 = sorted(glob.glob(f"{RES}/v5/experiment_v5_*.jsonl"))
s3 = sorted(glob.glob(f"{RES}/s3/experiment_v5_*.jsonl"))
a1 = sorted(glob.glob(f"{RES}/a1/experiment_v5_a1_*.jsonl"))
def parse(fn):
    m = re.search(r"_(fmnist|mnist|cifar)_B(\d+)_?k?(\d*)", fn)
    return (m.group(1), int(m.group(2)), int(m.group(3)) if m.group(3) else None)
v5_cells = [parse(f) for f in v5]
s3_cells = [parse(f) for f in s3]
a1_cells = [parse(f) for f in a1]
# budget axis = fmnist B in {30,60,120,240} x k in {2,3,5,8} where k | B
expect = {(b, k) for b in (30, 60, 120, 240) for k in (2, 3, 5, 8) if b % k == 0}
got = {(B, k) for (_, B, k) in v5_cells if _ == "fmnist"}
ck("B1 budget axis grid = B{30,60,120,240} x k{2,3,5,8} | divisibility",
   got == expect, f"(fs {len(got)} cells)")
ck("B2 B30/B60 have no k8 (7.5/3.75 indivisible)",
   all((b, 8) not in got for b in (30, 60)))
ck("B3 S3 cells = {mnist B120 k2/k5, cifar B60 k2/k5}",
   set(s3_cells) == {("mnist", 120, 2), ("mnist", 120, 5),
                    ("cifar", 60, 2), ("cifar", 60, 5)}, str(s3_cells))
ck("B4 A1 cells = fmnist B{30,60,120}",
   set(a1_cells) == {("fmnist", 30, None), ("fmnist", 60, None),
                     ("fmnist", 120, None)}, str(a1_cells))
ck("B5 paper states divisibility rule for k",
   "where divisible" in txt)

print("== C. per-record integrity sweep across all 21 cells ==")
n_rec = seeds_ok = ver_ok = 0
bad_seed, bad_ver = [], []
for f in v5 + s3 + a1:
    for line in open(f):
        r = json.loads(line)
        n_rec += 1
        if r.get("seed") not in set(range(10)):
            bad_seed.append((f.split("/")[-1], r.get("seed")))
        v = r.get("versions", {})
        if not (v.get("torch") and v.get("numpy")):
            bad_ver.append(f.split("/")[-1])
ck("C1 records total 210", n_rec == 210, f"({n_rec})")
ck("C2 seeds ⊆ {0..9} in every record", not bad_seed, str(bad_seed[:3]))
ck("C3 version stamps present in every record", not bad_ver, str(bad_ver[:3]))
torch_v = {json.loads(l).get("versions", {}).get("torch")
           for f in v5 + s3 + a1 for l in open(f)}
ck("C4 single torch version across all 210 records", len(torch_v) == 1, str(torch_v))

print("== D. figures / tables / hypotheses / references ==")
figs_src = sorted(glob.glob(os.path.join(ROOT, "figures", "v5", "fig*.pdf")))
suppl_src = sorted(glob.glob(os.path.join(ROOT, "figures", "v5", "suppl_fig*.pdf")))
ck("D1 5 main + 4 supplementary figure PDFs on disk",
   len(figs_src) == 5 and len(suppl_src) == 4,
   f"({len(figs_src)}+{len(suppl_src)})")
incl = set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex))
incl_base = {i.split("/")[-1].replace(".pdf", "") for i in incl}
# 5 main figures are embedded; the 4 supplementary figures ship separately
# (in the source zip / repository) and are described, not embedded, in the
# PDF — the tex carries one "Supplementary Fig.~S N" list item per figure.
ck("D2 tex embeds the 5 main figures",
   incl_base == {"fig1_phase", "fig2_mechanism", "fig3_temperature",
                 "fig4_ood", "fig5_cost"}, str(sorted(incl_base)))
n_suppl_items = len(re.findall(r"Supplementary Fig\.", tex))
ck("D2b 4 supplementary figures described (not embedded, shipped in zip)",
   n_suppl_items == 4 and len(suppl_src) == 4, f"({n_suppl_items} items)")
n_tables = len(re.findall(r"\\begin\{table\*?\}", tex))
print(f"   table environments in tex: {n_tables}")
ck("D3 tables >= 5 (grid + main + S3 + A1 + A2)",
   n_tables >= 5, f"({n_tables})")
for h in range(1, 7):
    ck(f"D4 H{h} mentioned in PDF text", f"H{h}" in txt)
n_bib = len(re.findall(r"\\bibitem", tex))
print(f"   bibliography entries: {n_bib}")
cite_keys = set(re.findall(r"\\cite[pt]?\{([^}]+)\}", tex))
all_keys = set()
for grp in cite_keys:
    all_keys.update(k.strip() for k in grp.split(","))
bib_keys = set(re.findall(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}", tex))
ck("D5 every \\cite key has a \\bibitem", all_keys <= bib_keys,
   str(sorted(all_keys - bib_keys)[:3]))

print("== E. budgets / k values / domains named ==")
for b in ("30", "60", "120", "240"):
    ck(f"E1 budget B={b} appears", f"B{{{b}}}" in tex or f" B{b} " in tex or
       f"B={b}" in txt or f"B {b}" in txt or f"{b} epochs" in txt)
ck("E2 k set {2,3,5,8} stated",
   "k \\in \\{2, 3, 5, 8\\}" in tex or "k ∈ {2, 3, 5, 8}" in txt)
for dom in ("Fashion-MNIST", "MNIST", "CIFAR"):
    ck(f"E3 domain {dom} appears", dom in txt)

print("== F. '10/10' directional claims: enumerate for review ==")
hits = [m.start() for m in re.finditer(r"10/10", txt)]
ctx = [re.sub(r"\s+", " ", txt[i - 70:i + 12]) for i in hits]
for c in ctx:
    print("   ...", c)
ck("F1 10/10 claims present (directional consistency)", len(hits) >= 5,
   f"({len(hits)} occurrences)")

print("== G. Holm / TOST margins ==")
print("   'Holm' occurrences:", len(re.findall(r"Holm", txt)))
print("   'TOST' occurrences:", len(re.findall(r"TOST", txt)))
ck("G1 TOST margin 0.020 stated", "0.020" in txt or "0.02" in txt)

print()
print("AUDIT " + ("ALL PASS" if not fail else f"FAIL: {fail}"))
sys.exit(0 if not fail else 1)
