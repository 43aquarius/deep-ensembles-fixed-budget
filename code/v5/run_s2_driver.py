#!/usr/bin/env python3
"""S2 driver (v5, Plan B): pre-registered FMNIST B240 extrapolation grid.

Pre-registered grid (Doc 3, Table 7-1, S2 — conditional on G1 PASS, which
was reached at the end of S1):
  B = 240 x k in {2, 3, 5, 8}  (4 cells, 10 seeds each)

k | B divisibility: 240/2=120, 240/3=80, 240/5=48, 240/8=30 — all integer
(the B60k8 smoke failure was exactly this violation; verified OK here).

Per-seed epoch accounting (B(2 + (k-1)/k); seed 0 adds B/k for the
bit-identity check -> exactly 3B on seed 0):
  k2: 600/720   k3: 640/720   k5: 672/720   k8: 690/720  (normal/seed-0)

Timing calibrated from S1 (0.40-0.43 s/epoch + ~10 s eval overhead):
  normal seeds 262-300 s; seed 0 ~313 s.  MAX_SEED_S=340, TB cap=330 so a
  child invocation starts a second seed right after seed 0 (~313 s) when
  the pair still fits the 600 s bash timeout (auto-resume makes the rare
  mid-write kill safe: completed seeds are already appended lines, the
  truncated tail line is ignored on resume).

Usage:  python run_s2_driver.py [wall_budget_seconds]   (default 595)
Exit codes: 0 = all 4 cells complete; 42 = wall budget reached (resumable).
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
CONFIGS = [(240, 2), (240, 3), (240, 5), (240, 8)]
N_SEEDS = 10
# longest plausible single-seed wallclock incl. seed-0 bit check (720 ep)
MAX_SEED_S = 340.0
# guard floor for cells whose normal-seed max is not yet observed
GUARD_FLOOR = 290.0
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


def normal_seed_max_s(path):
    """Max wallclock_s among completed NON-seed-0 rows of this cell.

    Seed 0 carries the bit-identity check (3B epochs vs 2B+(k-1)B/k) and
    would inflate the guard, blocking the 2-seeds-per-shard packing that
    the normal seeds allow.  If no normal seed has run yet, return None.
    """
    if not os.path.exists(path):
        return None
    ws = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("seed", 0) != 0 and "wallclock_s" in r:
                ws.append(float(r["wallclock_s"]))
    return max(ws) if ws else None


def main():
    wall_budget = float(sys.argv[1]) if len(sys.argv) > 1 else 595.0
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()

    for B, k in CONFIGS:
        out = os.path.join(OUTDIR, f"experiment_v5_fmnist_B{B}_k{k}.jsonl")
        while True:
            ds = done_seeds(out)
            if len(ds) >= N_SEEDS:
                print(f"[driver] B{B} k{k}: COMPLETE ({len(ds)}/10 seeds)",
                      flush=True)
                break
            elapsed = time.time() - t0
            remaining = wall_budget - elapsed
            # dynamic guard: worst plausible cost of the child's LAST
            # started seed.  Fresh cell (next seed = 0, carries bit check)
            # -> static MAX_SEED_S; else observed normal-seed max + 20 s.
            # A mid-write kill is resumable-safe, so the guard only sizes
            # the packing; it is not a correctness constraint.
            nmax = normal_seed_max_s(out)
            guard = MAX_SEED_S if nmax is None else max(nmax + 20.0,
                                                        GUARD_FLOOR)
            tb = min(remaining - guard - 12.0, 585.0)
            if tb < 60.0:
                print(f"[driver] wall budget reached "
                      f"(B{B} k{k} at {len(ds)}/10 seeds, "
                      f"{elapsed:.0f}s elapsed); resumable exit 42", flush=True)
                sys.exit(42)
            print(f"[driver] B{B} k{k}: {len(ds)}/10 seeds, launching child "
                  f"tb={tb:.0f}s (elapsed {elapsed:.0f}s)", flush=True)
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
    print("[driver] ALL_S2_CELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
