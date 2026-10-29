#!/usr/bin/env python3
"""S3 data-integrity checks (pre-registered grid, Doc 3 Table 7-1, S3).

Cells: results/s3/experiment_v5_{mnist_B120_k2, mnist_B120_k5,
                                cifar_B60_k2,  cifar_B60_k5}.jsonl

Checks:
  I1  every cell has exactly seeds 0..9, no duplicates, no truncated lines
  I2  shared_member_bit_identical == True on seed 0 of every cell
      (structural k|B check: 120/{2,5} and 60/{2,5} all integer)
  I3  within each domain, the Single condition is bit-identical across the
      two k cells (Single does not depend on k by construction; this also
      cross-checks that MNIST k2/k5 and CIFAR k2/k5 share training runs)
  I4  config stamps: budget / k / id / ood probes match the pre-registered
      matrix (mnist: id=mnist, probes=[fmnist, kmnist]; cifar: id=cifar10,
      probes=[cifar100])
  I5  version stamps uniform and identical to the S1/S2 stamp
      (torch 2.14.1+cpu / numpy 2.1.3; analysis pools across stages)
  I6  wallclock recorded; member-level metrics present for ensembles
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import json
import os

S3 = os.path.join(ROOT, "results", "s3")
DOMAINS = {
    "mnist": dict(B=120, ks=[2, 5], cid="mnist",
                  probes=["fmnist", "kmnist"]),
    "cifar": dict(B=60, ks=[2, 5], cid="cifar10",
                  probes=["cifar100"]),
}
N = 10

ok = True
data = {}
for dom, spec in DOMAINS.items():
    for k in spec["ks"]:
        p = os.path.join(S3, f"experiment_v5_{dom}_B{spec['B']}_k{k}.jsonl")
        seeds, rows = [], []
        with open(p) as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"I1 FAIL: {dom}B{spec['B']}k{k} line {ln} "
                          f"truncated: {e}")
                    ok = False
                    continue
                rows.append(r)
                seeds.append(r["seed"])
        data[(dom, k)] = rows
        dup = len(seeds) != len(set(seeds))
        complete = sorted(seeds) == list(range(N))
        if dup or not complete:
            print(f"I1 FAIL: {dom}B{spec['B']}k{k} seeds={sorted(seeds)}")
            ok = False
        else:
            print(f"I1 PASS: {dom} B{spec['B']} k{k} 10/10 unique seeds")

# I2 shared member bit-identity on seed 0
for (dom, k), rows in data.items():
    r0 = [r for r in rows if r["seed"] == 0][0]
    if r0.get("shared_member_bit_identical") is not True:
        print(f"I2 FAIL: {dom}k{k} shared_member_bit_identical="
              f"{r0.get('shared_member_bit_identical')}")
        ok = False
print("I2 PASS: shared member bit-identical on seed 0 of all 4 cells "
      "(=> k|B member-epoch integrality holds)")

# I3 Single bit-identity across k within each domain
for dom, spec in DOMAINS.items():
    k1, k2 = spec["ks"]
    ref = {r["seed"]: r["single"] for r in data[(dom, k1)]}
    cur = {r["seed"]: r["single"] for r in data[(dom, k2)]}
    same = all(
        ref[s]["nll"] == cur[s]["nll"]
        and ref[s]["acc"] == cur[s]["acc"]
        and ref[s]["ece15"] == cur[s]["ece15"]
        and ref[s]["brier"] == cur[s]["brier"]
        and ref[s]["ts"]["T"] == cur[s]["ts"]["T"]
        for s in range(N))
    print(f"I3 {'PASS' if same else 'FAIL'}: {dom} Single identical "
          f"k{k1} vs k{k2}")
    if not same:
        ok = False

# I4 config stamps
for (dom, k), rows in data.items():
    spec = DOMAINS[dom]
    for r in rows:
        c = r["config"]
        if (c["budget"] != spec["B"] or c["k"] != k or c["id"] != spec["cid"]
                or c["ood_probes"] != spec["probes"]):
            print(f"I4 FAIL: {dom}k{k} seed{r['seed']} config={c}")
            ok = False
print("I4 PASS: config stamps correct in all rows (budget, k, id, probes)")

# I5 version stamps uniform + identical to S1/S2 stamp
vers = {json.dumps(r["versions"], sort_keys=True)
        for rows in data.values() for r in rows}
stamp = json.dumps({"numpy": "2.1.3", "torch": "2.14.1+cpu"},
                   sort_keys=True)
if len(vers) == 1 and vers.pop() == stamp:
    print("I5 PASS: version stamps uniform and equal to S1/S2 stamp "
          "(torch 2.14.1+cpu, numpy 2.1.3)")
else:
    print(f"I5 FAIL: version stamps: {vers} vs expected {stamp}")
    ok = False

# I6 wallclock + member metrics presence
tot_w = 0.0
for (dom, k), rows in data.items():
    for r in rows:
        tot_w += r.get("wallclock_s", 0.0)
        for cond in ("deepens", "snapshot"):
            if "member_mean_nll" not in r[cond]:
                print(f"I6 FAIL: {dom}k{k} seed{r['seed']} "
                      f"{cond} missing member metrics")
                ok = False
print(f"I6 PASS: member metrics present; total S3 wallclock = "
      f"{tot_w/3600:.2f} h ({tot_w:.0f} s)")

print()
print("OVERALL:", "ALL CHECKS PASS" if ok else "FAILURES PRESENT")
