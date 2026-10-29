#!/usr/bin/env python3
"""Rebuild datasets_cifar.npz from the OFFICIAL CIFAR-10/CIFAR-100 binaries.

Context: the original download_data_cifar.py (HF parquet mirror route)
requires pyarrow, which cannot be installed in this environment (pypi
timeouts).  The HF route's own provenance note states it is content-
identical to the official https://www.cs.toronto.edu/~kriz/ releases, so
building from the official binaries reproduces the same arrays through
the primary source.  Key format, dtypes and integrity checks mirror
download_data_cifar.py exactly:

  cifar10_train_x (50000,32,32,3) uint8   cifar10_train_y (50000,) int64
  cifar10_test_x  (10000,32,32,3) uint8  cifar10_test_y  (10000,) int64
  cifar100_test_x  (10000,32,32,3) uint8  cifar100_test_y (10000,) int64

Checks: per-class counts, label ranges, per-channel means in float64
against the published CIFAR-10 statistics (0.4914, 0.4822, 0.4465).

Resumable: tarballs are downloaded with curl -C - into the same cache
directory layout the HF route used (results/data/hf/), so re-running
this script after an interruption continues the download.

Usage: python3 build_cifar_npz_official.py
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import os
import pickle
import subprocess
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.join(ROOT, "code"))
import paths  # noqa: E402  (resolves DATASETS_CIFAR the same way as v4 code)

BASE = "https://www.cs.toronto.edu/~kriz"
C10 = f"{BASE}/cifar-10-python.tar.gz"
C100 = f"{BASE}/cifar-100-python.tar.gz"

CACHE = os.path.join(ROOT, "results", "data", "hf")
os.makedirs(CACHE, exist_ok=True)
TARS = {
    "cifar-10-python.tar.gz": C10,
    "cifar-100-python.tar.gz": C100,
}


def ensure_downloaded():
    for name, url in TARS.items():
        dst = os.path.join(CACHE, name)
        for attempt in range(6):
            r = subprocess.run(
                ["curl", "-C", "-", "-L", "--retry", "3",
                 "--connect-timeout", "30", "-o", dst, url],
                capture_output=True, text=True)
            if r.returncode == 0:
                print(f"downloaded ok: {name} "
                      f"({os.path.getsize(dst)} bytes)", flush=True)
                break
            print(f"attempt {attempt + 1} failed rc={r.returncode}: "
                  f"{r.stderr.strip()[:200]}", flush=True)
        else:
            raise RuntimeError(f"could not download {name}")
        # minimal size sanity (official sizes ~163MB / ~169MB)
        if os.path.getsize(dst) < 100_000_000:
            raise RuntimeError(f"{name} too small, corrupt download")


def _unpickle(fh):
    return pickle.load(fh, encoding="latin1")


def plane_to_hwc(a):
    """(N, 3072) uint8 rows of R,G,B planes -> (N,32,32,3) HWC uint8."""
    return a.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy()


def main():
    ensure_downloaded()

    tr_x, tr_y, te_x, te_y = [], [], [], []
    with tarfile.open(os.path.join(CACHE, "cifar-10-python.tar.gz")) as tf:
        for member in tf.getmembers():
            n = os.path.basename(member.name)
            if n.startswith("data_batch_") and n.endswith(
                    ("1", "2", "3", "4", "5")):
                d = _unpickle(tf.extractfile(member))
                tr_x.append(d["data"])
                tr_y.extend(d["labels"])
                print(f"parsed {n}: {len(d['labels'])} rows", flush=True)
            elif n == "test_batch":
                d = _unpickle(tf.extractfile(member))
                te_x = d["data"]
                te_y = d["labels"]
                print(f"parsed {n}: {len(d['labels'])} rows", flush=True)
    tr_x = plane_to_hwc(np.concatenate(tr_x, axis=0))
    tr_y = np.asarray(tr_y, dtype=np.int64)
    te_x = plane_to_hwc(np.asarray(te_x))
    te_y = np.asarray(te_y, dtype=np.int64)

    with tarfile.open(os.path.join(CACHE, "cifar-100-python.tar.gz")) as tf:
        m = tf.getmember("cifar-100-python/test")
        d = _unpickle(tf.extractfile(m))
        c100_x = plane_to_hwc(np.asarray(d["data"]))
        c100_y = np.asarray(d["fine_labels"], dtype=np.int64)
    print(f"parsed cifar100 test: {len(c100_y)} rows", flush=True)

    out = dict(
        cifar10_train_x=tr_x, cifar10_train_y=tr_y,
        cifar10_test_x=te_x, cifar10_test_y=te_y,
        cifar100_test_x=c100_x, cifar100_test_y=c100_y,
    )
    np.savez_compressed(paths.DATASETS_CIFAR, **out)
    print("saved", paths.DATASETS_CIFAR, flush=True)
    for k, v in out.items():
        print(f"  {k}: {v.shape} {v.dtype} range=[{v.min()},{v.max()}]")
    print("train class counts:", np.bincount(tr_y))
    print("test  class counts:", np.bincount(te_y))
    print("c100 fine labels in [0,100):", c100_y.min(), c100_y.max())
    # float64 aggregation is essential (float32 drifts over 5e7 elements)
    chan = tr_x.astype(np.float64).mean(axis=(0, 1, 2)) / 255.0
    print("per-channel means (want ~0.4914 0.4822 0.4465):",
          np.round(chan, 4))
    assert tr_x.shape == (50000, 32, 32, 3) and te_x.shape == (10000, 32, 32, 3)
    assert c100_x.shape == (10000, 32, 32, 3)
    assert np.allclose(chan, [0.4914, 0.4822, 0.4465], atol=2e-4), chan
    assert (np.bincount(tr_y) == 5000).all()
    assert (np.bincount(te_y) == 1000).all()
    print("ALL INTEGRITY CHECKS PASS", flush=True)


if __name__ == "__main__":
    main()
