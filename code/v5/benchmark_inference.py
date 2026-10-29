#!/usr/bin/env python3
"""A2 inference benchmark (Doc 3 Table 7-1 / H6): measured cost of
probability-averaged prediction, kept strictly separate from the training
budget (Doc 3 Table 5-1).

Measurements (CPU, TORCH_THREADS=2 as in the whole v5 grid):
  - latency of ensembles of k in {1, 2, 3, 5, 8} members (grid values plus
    k=1 = Single), batch sizes 1 and 256, predicted-probability averaging in
    the main-engine recipe (torch float32 forward -> numpy float64 softmax
    -> mean over members)
  - parameter count and fp32 weight storage per model and k x storage
  - resident-memory (VmRSS) increment when holding k models, as a CPU-side
    stand-in for the (GPU) memory column

Latency depends only on (architecture, batch size, k) -- fixed FLOPs, no
data-dependent branching -- so synthetic input tensors are used (noted in
every output table).  Models are randomly initialized; weight VALUES do not
affect latency, only their count and dtype do.

Output: results/a2/benchmark_inference.json (+ printed summary).
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
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
sys.path.insert(0, os.path.join(ROOT, "code"))

from common import MLP
from common_cifar import SmallCNN

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))

OUT = os.path.join(ROOT, "results", "a2", "benchmark_inference.json")
KS = [1, 2, 3, 5, 8]
REPS = {("mlp", 1): 400, ("mlp", 256): 100,
        ("cnn", 1): 200, ("cnn", 256): 30}
WARMUP = 10


def vmrss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return None


def member_probs(model, Xt):
    """Main-engine recipe with the model resident (deployment view):
    torch float32 forward, then numpy float64 softmax."""
    with torch.no_grad():
        z = model(Xt)
    z = z.numpy()
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z.astype(np.float64))
    return e / e.sum(axis=1, keepdims=True)


def bench(models, Xt, reps):
    """Mean +- sd over `reps` repetitions of one full ensemble prediction."""
    for m in models:            # warmup
        member_probs(m, Xt)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ps = [member_probs(m, Xt) for m in models]
        pbar = np.mean(np.stack(ps, axis=0), axis=0)  # probability average
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return dict(mean_s=float(ts.mean()), sd_s=float(ts.std(ddof=1)),
                median_s=float(np.median(ts)),
                reps=reps, n_outputs=int(pbar.shape[0]))


def run_family(name, make_model, in_shape, out_dim):
    torch.manual_seed(0)
    results = {}

    # parameter count / storage per model
    m0 = make_model(12345)
    n_params = sum(p.numel() for p in m0.parameters())
    bytes_per_model = int(sum(p.numel() * p.element_size() for p in m0.parameters()))
    results["params_per_model"] = int(n_params)
    results["fp32_bytes_per_model"] = bytes_per_model

    for bsz in (1, 256):
        torch.manual_seed(0)
        Xt = torch.randn(bsz, *in_shape)

        # k models with distinct init seeds (grid ensemble members)
        models_by_k = {}
        for k in KS:
            models_by_k[k] = [make_model(10000 + 1000 * i + 7) for i in range(k)]

        rows = []
        for k in KS:
            r = bench(models_by_k[k], Xt, REPS[(name, bsz)])
            r.update(k=k, batch=bsz, per_sample_s=r["mean_s"] / bsz)
            rows.append(r)
            del r["n_outputs"]
        results[f"latency_batch{bsz}"] = rows

        # linear fit latency(k) = a + b k  (over the five grid k values)
        ks = np.array([r["k"] for r in rows], dtype=np.float64)
        ys = np.array([r["mean_s"] for r in rows])
        b, a = np.polyfit(ks, ys, 1)
        pred = a + b * ks
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        results[f"latency_batch{bsz}_linear"] = dict(
            slope_s_per_member=float(b), intercept_s=float(a),
            r2=float(1 - ss_res / ss_tot))

    # resident-memory increment of holding k models (batch 256 tensors made)
    rss0 = vmrss_mb()
    held = [make_model(20000 + i) for i in range(8)]
    rss8 = vmrss_mb()
    results["rss_delta_8models_mb"] = (rss8 - rss0) if (rss0 and rss8) else None
    results["rss_note"] = ("VmRSS increment while holding 8 models; "
                          "CPU stand-in for GPU-memory column of H6")
    # storage column (deterministic): k x fp32 weight bytes
    results["storage_bytes_by_k"] = {str(k): k * bytes_per_model for k in KS}
    results["in_shape"] = list(in_shape)
    results["out_dim"] = out_dim
    results["input_note"] = ("synthetic standard-normal tensors; latency "
                             "depends only on (architecture, batch, k)")
    return results


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    out = dict(
        config=dict(threads=int(os.environ.get("TORCH_THREADS", "2")),
                    ks=KS, batches=[1, 256],
                    dtype="fp32 weights; float64 numpy probability averaging",
                    torch=torch.__version__, numpy=np.__version__,
                    machine=os.uname().machine),
        families={},
    )

    t0 = time.time()
    out["families"]["mlp_fmnist"] = run_family(
        "mlp", lambda seed: MLP(seed), (784,), 10)
    out["families"]["cnn_cifar10"] = run_family(
        "cnn", lambda seed: SmallCNN(seed), (3, 32, 32), 10)
    out["wallclock_s"] = time.time() - t0

    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)

    # printed summary
    for fam, r in out["families"].items():
        print(f"=== {fam}: {r['params_per_model']:,} params, "
              f"{r['fp32_bytes_per_model']/1e6:.3f} MB/model ===")
        for bsz in (1, 256):
            print(f"  batch={bsz}:")
            for row in r[f"latency_batch{bsz}"]:
                print(f"    k={row['k']}: {row['mean_s']*1e3:8.3f} +- "
                      f"{row['sd_s']*1e3:6.3f} ms "
                      f"(median {row['median_s']*1e3:8.3f}; "
                      f"{row['per_sample_s']*1e6:8.2f} us/sample)")
            lin = r[f"latency_batch{bsz}_linear"]
            print(f"    linear fit: {lin['slope_s_per_member']*1e3:.3f} "
                  f"ms/member + {lin['intercept_s']*1e3:.3f} ms, "
                  f"R2={lin['r2']:.6f}")
        print(f"  RSS delta holding 8 models: "
              f"{r['rss_delta_8models_mb']:.1f} MB")
    print(f"\nwrote {OUT} ({out['wallclock_s']:.0f}s)")


if __name__ == "__main__":
    main()
