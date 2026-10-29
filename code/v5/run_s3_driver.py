#!/usr/bin/env python3
"""S3 driver (v5, Plan B): pre-registered cross-domain transfer matrix.

Cells (Doc 3, Table 7-1, S3 -- conditional on G2 PASS, reached at end of S2):
  MNIST (MLP, v2 engine):              B120 x k in {2, 5}
  CIFAR-10 (SmallCNN, cifar engine):    B60  x k in {2, 5}
10 seeds each.  Cell order: mnist k2 -> mnist k5 -> cifar k2 -> cifar k5.
Outputs -> results/s3/experiment_v5_{mnist,cifar}_B{B}_k{k}.jsonl

Engines (both survive in the repository, unmodified):
  mnist: code/run_experiments_v2.py --id mnist
         seed-level append; no checkpoints (a kill loses at most the
         in-flight seed; a mid-write kill leaves a truncated tail line,
         sanitized by this driver before every launch)
  cifar: code/run_experiments_cifar.py
         epoch-level checkpoints under results/cifar/ckpts (exact mid-seed
         resume); a ~600 s truncation is the normal end of a shard

Safety protocols (proven across ten environment resets):
  - single-instance lock  scripts/run_s3_driver.pid (+ /proc cmdline check)
  - orphan-engine kill    SIGTERM -> 15 s -> SIGKILL, before any launch
  - jsonl sanitation      truncated tail lines dropped (atomic rewrite)
  - CIFAR ckpt .owner     cell-identity guard against cross-cell mixing;
                          corrupt owner or LATER-cell owner -> exit 3
  - ckpt validation       torch.load probe; corrupt (killed mid-save)
                          checkpoints are deleted and that piece retrains
  - SIGTERM forwarding   a driver termination forwards to the child so no
                          engine outlives its driver

Usage:  python run_s3_driver.py [wall_budget_seconds]   (default 595)
Exit codes: 0 = ALL_S3_CELLS_DONE; 42 = resumable; 3 = .owner violation;
            4 = orphan engines could not be cleared.
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import glob
import json
import os
import signal
import subprocess
import sys
import time

CODE = os.path.join(ROOT, "code")
OUTDIR = os.path.join(ROOT, "results", "s3")
CKPT_DIR = os.path.join(ROOT, "results", "cifar", "ckpts")
OWNER_PATH = os.path.join(CKPT_DIR, ".owner")
LOCK_PATH = os.path.join(__import__("tempfile").gettempdir(), "run_s3_driver.pid")
PY = sys.executable

CELLS = [
    ("mnist", "run_experiments_v2.py", 120, 2),
    ("mnist", "run_experiments_v2.py", 120, 5),
    ("cifar", "run_experiments_cifar.py", 60, 2),
    ("cifar", "run_experiments_cifar.py", 60, 5),
]
N_SEEDS = 10
MAX_SEED_S_MNIST = 240.0   # static packing guard (longest plausible MNIST seed)
CIFAR_HEADROOM = 25.0      # CIFAR child may be truncated mid-seed by design

CURRENT_CHILD = None


def out_path(domain, B, k):
    return os.path.join(OUTDIR, f"experiment_v5_{domain}_B{B}_k{k}.jsonl")


def owner_string(domain, B, k):
    return f"cell={domain}_B{B}_k{k}"


# --------------------------------------------------------------------- lock

def acquire_lock():
    if os.path.exists(LOCK_PATH):
        try:
            pid = int(open(LOCK_PATH).read().strip())
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().decode(errors="replace").replace("\x00", " ")
            if "run_s3_driver" in cmd:
                print(f"[driver] lock held by live driver pid={pid}; "
                      f"exiting 42 (never run two drivers)", flush=True)
                sys.exit(42)
        except (ValueError, ProcessLookupError, FileNotFoundError,
                PermissionError):
            pass  # stale lock from a dead driver
    with open(LOCK_PATH, "w") as f:
        f.write(str(os.getpid()))


# ----------------------------------------------------------- orphan engines

ENGINE_MARKERS = ("run_experiments_v2.py", "run_experiments_cifar.py")


def engine_pids():
    pids = []
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                         text=True).stdout
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid, args = parts
        if any(m in args for m in ENGINE_MARKERS) and "run_s3_driver" not in args:
            pids.append(int(pid))
    return pids


def kill_orphan_engines():
    pids = engine_pids()
    if not pids:
        return
    print(f"[driver] orphan engine(s) {pids} detected; SIGTERM", flush=True)
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(150):
        if not engine_pids():
            break
        time.sleep(0.1)
    for p in engine_pids():
        print(f"[driver] SIGKILL orphan {p}", flush=True)
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(1.0)
    if engine_pids():
        print("[driver] engines survived SIGKILL; exiting 4", flush=True)
        sys.exit(4)
    print("[driver] orphans cleared; epoch-level ckpts make kills safe",
          flush=True)


# ----------------------------------------------------------------- jsonl

def sanitize_jsonl(path):
    """Drop truncated tail lines (killed mid-append) with an atomic rewrite."""
    if not os.path.exists(path):
        return
    kept, dropped = [], 0
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                json.loads(s)
                kept.append(s)
            except json.JSONDecodeError:
                dropped += 1
    if dropped:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for s in kept:
                f.write(s + "\n")
        os.replace(tmp, path)
        print(f"[driver] sanitized {os.path.basename(path)} "
              f"(dropped {dropped} truncated line(s))", flush=True)


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
                    pass  # truncated tail line -> ignore
    return s


# ------------------------------------------------------------ cifar ckpts

def cifar_cell_complete_cleanup():
    """Remove all per-seed checkpoints and the owner marker (cell done)."""
    for p in glob.glob(os.path.join(CKPT_DIR, "s*_*.pt")):
        try:
            os.remove(p)
        except OSError:
            pass
    if os.path.exists(OWNER_PATH):
        try:
            os.remove(OWNER_PATH)
        except OSError:
            pass


def cifar_drop_done_seed_ckpts(done):
    """Crash between append and engine cleanup leaves stray done-seed ckpts."""
    n = 0
    for s in done:
        for p in glob.glob(os.path.join(CKPT_DIR, f"s{s}_*.pt")):
            try:
                os.remove(p)
                n += 1
            except OSError:
                pass
    if n:
        print(f"[driver] dropped {n} ckpt(s) of already-done seeds", flush=True)


def cifar_validate_ckpts():
    """Delete checkpoints that fail to load (killed mid-torch.save)."""
    import torch
    for p in sorted(glob.glob(os.path.join(CKPT_DIR, "s*_*.pt"))):
        try:
            torch.load(p, weights_only=False)
        except Exception as e:  # noqa: BLE001
            print(f"[driver] corrupt ckpt removed: {os.path.basename(p)} "
                  f"({type(e).__name__}); piece will retrain", flush=True)
            try:
                os.remove(p)
            except OSError:
                pass


def cifar_ensure_owner(domain, B, k):
    """Validate/refresh the ckpt-dir owner marker for the current cell.

    .owner holds 'cell=<domain>_B<B>_k<k>'.  Rules:
      - matches current cell          -> keep (exact mid-seed resume)
      - EARLIER cell in CELLS order    -> that cell is finished; delete its
                                          stale ckpts and take ownership
      - LATER cell in CELLS order      -> impossible ordering; exit 3
      - unreadable / unknown format    -> exit 3
      - missing owner + stray ckpts   -> unknown provenance; delete ckpts
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    exp = owner_string(domain, B, k)
    cur = None
    if os.path.exists(OWNER_PATH):
        try:
            cur = open(OWNER_PATH).read().strip()
        except OSError:
            cur = None
        if cur and not cur.startswith("cell="):
            print(f"[driver] .owner corrupt ({cur!r}); exiting 3", flush=True)
            sys.exit(3)
    if cur == exp:
        return
    stray = glob.glob(os.path.join(CKPT_DIR, "s*_*.pt"))
    cell_of = {owner_string(d, b, kk): i for i, (d, _, b, kk) in enumerate(CELLS)}
    if cur is not None and cur != exp:
        if cur not in cell_of or cell_of[cur] > cell_of[exp]:
            print(f"[driver] .owner violation: dir owned by {cur!r} but "
                  f"current cell is {exp!r}; exiting 3", flush=True)
            sys.exit(3)
        print(f"[driver] ckpt dir owned by earlier cell {cur!r}; "
              f"deleting {len(stray)} stale ckpt(s)", flush=True)
        for p in stray:
            try:
                os.remove(p)
            except OSError:
                pass
    elif stray:
        print(f"[driver] no .owner but {len(stray)} ckpt(s) present "
              f"(unknown provenance); deleting", flush=True)
        for p in stray:
            try:
                os.remove(p)
            except OSError:
                pass
    with open(OWNER_PATH, "w") as f:
        f.write(exp + "\n")


# --------------------------------------------------------------- sigterm

def _forward_sigterm(signum, frame):
    global CURRENT_CHILD
    if CURRENT_CHILD is not None and CURRENT_CHILD.poll() is None:
        try:
            CURRENT_CHILD.terminate()
        except ProcessLookupError:
            pass
        deadline = time.time() + 3.0
        while time.time() < deadline and CURRENT_CHILD.poll() is None:
            time.sleep(0.1)
    os._exit(143)


def launch_child(cmd):
    global CURRENT_CHILD
    proc = subprocess.Popen(cmd, cwd=CODE)
    CURRENT_CHILD = proc
    try:
        return proc.wait()
    finally:
        CURRENT_CHILD = None


# ------------------------------------------------------------------ main

def main():
    wall_budget = float(sys.argv[1]) if len(sys.argv) > 1 else 595.0
    signal.signal(signal.SIGTERM, _forward_sigterm)
    acquire_lock()
    kill_orphan_engines()
    os.makedirs(OUTDIR, exist_ok=True)

    # sanitize every existing S3 jsonl once per shard
    for domain, _, B, k in CELLS:
        sanitize_jsonl(out_path(domain, B, k))

    t0 = time.time()
    for domain, engine, B, k in CELLS:
        out = out_path(domain, B, k)
        ds = done_seeds(out)
        if len(ds) >= N_SEEDS:
            # COMPLETED cells skip ckpt ownership handling entirely: a later
            # cell may legitimately already hold the .owner marker (e.g. the
            # previous shard finished this cell and started the next).
            print(f"[driver] {domain} B{B} k{k}: COMPLETE "
                  f"({len(ds)}/10 seeds)", flush=True)
            if domain == "cifar":
                # sweep only if we still own the dir (kill between the last
                # append and the in-loop cleanup); a LATER owner's live
                # in-flight ckpts must never be touched here
                cur = None
                if os.path.exists(OWNER_PATH):
                    try:
                        cur = open(OWNER_PATH).read().strip()
                    except OSError:
                        cur = None
                if cur == owner_string(domain, B, k):
                    cifar_cell_complete_cleanup()
            continue
        if domain == "cifar":
            cifar_ensure_owner(domain, B, k)
            cifar_drop_done_seed_ckpts(ds)
            cifar_validate_ckpts()
        while True:
            ds = done_seeds(out)
            if len(ds) >= N_SEEDS:
                if domain == "cifar":
                    cifar_cell_complete_cleanup()
                print(f"[driver] {domain} B{B} k{k}: COMPLETE "
                      f"({len(ds)}/10 seeds)", flush=True)
                break
            elapsed = time.time() - t0
            remaining = wall_budget - elapsed
            if domain == "mnist":
                tb = min(remaining - MAX_SEED_S_MNIST - 12.0, 585.0)
            else:
                tb = min(remaining - CIFAR_HEADROOM, 585.0)
            if tb < 60.0:
                print(f"[driver] wall budget reached ({domain} B{B} k{k} "
                      f"at {len(ds)}/10 seeds, {elapsed:.0f}s elapsed); "
                      f"resumable exit 42", flush=True)
                sys.exit(42)
            print(f"[driver] {domain} B{B} k{k}: {len(ds)}/10 seeds, "
                  f"launching child tb={tb:.0f}s (elapsed {elapsed:.0f}s)",
                  flush=True)
            if domain == "mnist":
                cmd = [PY, engine, "--id", "mnist", "--budget", str(B),
                       "--k", str(k), "--seeds", "0:10", "--out", out,
                       "--time-budget", str(int(tb))]
            else:
                cmd = [PY, engine, "--budget", str(B), "--k", str(k),
                       "--seeds", "0:10", "--out", out,
                       "--time-budget", str(int(tb))]
            rc = launch_child(cmd)
            if rc == 0:
                continue  # child reports cell complete; loop re-verifies
            if rc != 42:
                print(f"[driver] child failed rc={rc} ({domain} B{B} k{k}); "
                      f"aborting shard for inspection", flush=True)
                sys.exit(rc)
            # rc == 42: child exhausted its budget; loop and re-evaluate
    print("[driver] ALL_S3_CELLS_DONE", flush=True)


if __name__ == "__main__":
    main()
