#!/usr/bin/env python3
"""S1 driver (v5, Plan B): run the pre-registered FMNIST grid sequentially.

Pre-registered grid (Doc 3, Table 7-1, S1):
  B in {30, 60, 120} x k in {2, 3, 5}, plus B=120 x k=8  (10 cells, 10 seeds each)

All cells are run fresh under the new environment (torch 2.14.1+cpu,
2 threads), including the k=3/k=5 cells that overlap old v2 records:
per the preregistration (Section 2), old records are exploratory and
must NOT be reused as primary data for v5.

Usage:  python run_s1_driver.py [wall_budget_seconds]   (default 560)
Exit codes: 0 = all 10 cells complete; 42 = wall budget reached (resumable).

Each child invocation gets a --time-budget that leaves headroom for the
longest single seed (~180 s for B120k8) so the outer bash timeout (600 s)
never kills a seed mid-write.
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
import subprocess
import sys
import time

CODE = os.path.join(ROOT, "code")
OUTDIR = os.path.join(ROOT, "results", "v5")
CONFIGS = [
    (30, 2), (30, 3), (30, 5),
    (60, 2), (60, 3), (60, 5),
    (120, 2), (120, 3), (120, 5), (120, 8),
]
N_SEEDS = 10
# longest plausible single-seed wallclock (train + eval + seed-0 bit check)
MAX_SEED_S = 200.0
PY = sys.executable


def done_seeds(path):
    if not os.path.exists(path):
        return set()
    s = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    s.add(json.loads(line)["seed"])
                except json.JSONDecodeError:
                    pass  # truncated tail line (killed mid-write) -> ignore
    return s


def main():
    wall_budget = float(sys.argv[1]) if len(sys.argv) > 1 else 560.0
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()

    for B, k in CONFIGS:
        out = os.path.join(OUTDIR, f"experiment_v5_fmnist_B{B}_k{k}.jsonl")
        while True:
            ds = done_seeds(out)
            if len(ds) >= N_SEEDS:
                print(f"[driver] B{B} k{k}: COMPLETE ({len(ds)}/10 seeds)", flush=True)
                break
            remaining = wall_budget - (time.time() - t0)
            # need enough headroom for a full child invocation: startup + tb + last seed
            tb = remaining - MAX_SEED_S - 30.0
            if tb < 60.0:
                print(f"[driver] wall budget reached "
                      f"(B{B} k{k} at {len(ds)}/10 seeds); resumable exit 42",
                      flush=True)
                sys.exit(42)
            tb = min(tb, 380.0)
            print(f"[driver] B{B} k{k}: {len(ds)}/10 seeds, launching child "
                  f"tb={tb:.0f}s", flush=True)
            r = subprocess.run(
                [PY, "run_experiments_v2.py",
                 "--id", "fmnist", "--budget", str(B), "--k", str(k),
                 "--seeds", "0:10", "--out", out,
                 "--time-budget", str(int(tb))],
                cwd=CODE,
            )
            if r.returncode == 0:
                break
            if r.returncode != 42:
                print(f"[driver] child failed rc={r.returncode}", flush=True)
                sys.exit(r.returncode)
    print("[driver] ALL_S1_CELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
