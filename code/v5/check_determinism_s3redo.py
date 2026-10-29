#!/usr/bin/env python3
"""Determinism regression check: rerun vs archived S1 record, bitwise.

Compares every JSON field except wallclock-related keys. Verifies that the
environment rebuild (torch 2.14.1+cpu reinstall) reproduces S1/S2 numbers
bit-exactly, so S3 can proceed under identical conditions.
"""

# --- repo-relative path resolution (v5 release; mirrors code/paths.py) ---
import os
ROOT = os.environ.get(
    "BME_ROOT",
    os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..")),
)
import json
import sys

ARCHIVE = os.path.join(ROOT, "results", "v5", "experiment_v5_fmnist_B30_k2.jsonl")
RERUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3redo_det_check.jsonl")


def load_first(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise RuntimeError(f"no records in {path}")


def diff(a, b, path="$"):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a:
                out.append(f"MISSING in archive: {path}.{key}")
            elif key not in b:
                out.append(f"MISSING in rerun:   {path}.{key}")
            else:
                out.extend(diff(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"LEN {path}: archive {len(a)} vs rerun {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(diff(x, y, f"{path}[{i}]"))
    elif a != b:
        out.append(f"VALUE {path}: archive {a!r} vs rerun {b!r}")
    return out


def _is_time_key(key: str) -> bool:
    kl = key.lower()
    return any(t in kl for t in ("wall", "time_s", "elapsed", "runtime"))


def strip_times(obj):
    if isinstance(obj, dict):
        return {k: strip_times(v) for k, v in obj.items()
                if not (isinstance(v, (int, float)) and _is_time_key(k))}
    if isinstance(obj, list):
        return [strip_times(x) for x in obj]
    return obj


def main():
    arch = load_first(ARCHIVE)
    re = load_first(RERUN)
    a, b = strip_times(arch), strip_times(re)
    d = diff(a, b)
    print(f"archive fields: {len(arch)}; compared after stripping time keys")
    if not d:
        print("BITWISE DETERMINISM: PASS")
        print(f"  seed {arch['seed']} config={arch['config']}")
        sys.exit(0)
    print(f"BITWISE DETERMINISM: FAIL ({len(d)} differences)")
    for line in d[:20]:
        print(" ", line)
    sys.exit(1)


if __name__ == "__main__":
    main()
