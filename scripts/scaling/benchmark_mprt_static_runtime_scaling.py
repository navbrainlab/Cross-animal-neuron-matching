#!/usr/bin/env python3
"""
End-to-end runtime scaling for the FINAL Static-Atlas MPRT inference path.

Timed production path:
    query_encoding = model.encode_population(query_sample)
    output = model.match_encodings(query_encoding, static_atlas)

The static atlas encoding is precomputed once outside timing, exactly as in the
paper's unified candidate-score exporter.

Reports:
- query population size N
- static atlas size M (fixed)
- encode latency median / p95
- relation-match latency median / p95
- end-to-end latency median / p95 / mean / sample SD
- total peak GPU allocated memory
- incremental peak GPU allocated memory above model+atlas baseline
- log-log empirical scaling slopes

Disk I/O, checkpoint loading, and one-time atlas construction are excluded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch


DEFAULT_REPO = Path("/home/ubuntu/klb/nuclr/nuclr")
DEFAULT_PACKAGE = DEFAULT_REPO / "mprt_net_v1_1"
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO
    / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
    / "rld/fold0/seed42/static_atlas/anchored_pure.pt"
)
DEFAULT_FOLD_ROOT = (
    DEFAULT_REPO / "Data/Dunn_001623/cv5_grouped_v1/fold_0"
)
DEFAULT_OUT = (
    DEFAULT_REPO
    / "runs/mprt_runtime_scaling_v1"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    p.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256, 512],
    )
    p.add_argument("--activity-length", type=int, default=512)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "sd_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median_ms": float(np.median(a)),
        "p95_ms": percentile(a, 95),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def cuda_time_ms(fn, repeats):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]

    for i in range(repeats):
        starts[i].record()
        fn()
        ends[i].record()

    torch.cuda.synchronize()
    return [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]


def make_scaled_sample(base, n: int, seed: int):
    """
    Shape-faithful synthetic population built from one real TRAIN worm.

    For N <= base_N: deterministic evenly distributed subsample.
    For N > base_N: cyclic replication + tiny deterministic jitter to avoid
    exact duplicate coordinates/activity while preserving realistic scale.
    """
    if n < 2:
        raise ValueError("N must be >= 2")

    base_n = int(base.xyz.shape[0])
    if base_n < 2:
        raise RuntimeError("Prototype worm has fewer than two neurons")

    if n <= base_n:
        idx = torch.linspace(0, base_n - 1, steps=n).round().long()
    else:
        idx = torch.arange(n, dtype=torch.long) % base_n

    xyz = base.xyz[idx].clone()
    activity = base.activity[idx].clone()

    g = torch.Generator(device="cpu")
    g.manual_seed(seed + n)

    if n > base_n:
        xyz_scale = xyz.float().std(dim=0, unbiased=False).mean().clamp_min(1e-6)
        act_scale = activity.float().std(unbiased=False).clamp_min(1e-6)

        xyz = xyz + (
            torch.randn(xyz.shape, generator=g, dtype=xyz.dtype)
            * (1e-4 * xyz_scale.to(dtype=xyz.dtype))
        )
        activity = activity + (
            torch.randn(activity.shape, generator=g, dtype=activity.dtype)
            * (1e-4 * act_scale.to(dtype=activity.dtype))
        )

    from mprt_net.data import WormSample

    return WormSample(
        uid=f"runtime_N{n}",
        xyz=xyz,
        activity=activity,
        cell_ids=tuple(f"RUNTIME_{i:05d}" for i in range(n)),
        supervised_mask=torch.ones(n, dtype=torch.bool),
        source_path=f"<runtime-synthetic-N{n}>",
    )


def empirical_slope(rows, key, min_points=3):
    pairs = [
        (float(r["N"]), float(r[key]))
        for r in rows
        if float(r[key]) > 0 and int(r["N"]) >= 64
    ]
    if len(pairs) < min_points:
        return float("nan")

    x = np.log(np.asarray([p[0] for p in pairs], dtype=np.float64))
    y = np.log(np.asarray([p[1] for p in pairs], dtype=np.float64))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def main():
    args = parse_args()

    if not args.package_root.is_dir():
        raise FileNotFoundError(args.package_root)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.fold_root.is_dir():
        raise FileNotFoundError(args.fold_root)

    sys.path.insert(0, str(args.package_root))

    from mprt_net.data import WormCache, split_files
    from mprt_net.evaluate import load_checkpoint

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load FINAL static-atlas checkpoint.
    model, payload = load_checkpoint(args.checkpoint, device)
    model.eval()

    if not getattr(model, "atlas_is_initialized", False):
        raise RuntimeError(
            "Checkpoint does not contain an initialized static atlas; "
            "this is not the final Static MPRT inference path."
        )

    atlas = model.atlas_encoding()
    atlas_size = int(atlas.nodes.shape[0])

    # Use one real TRAIN animal only as a shape/statistics prototype.
    # No labels are used; no accuracy is computed.
    train_files = split_files(args.fold_root, "train")
    if not train_files:
        raise RuntimeError(f"No train NPZs under {args.fold_root}")

    cache = WormCache(activity_length=args.activity_length, max_items=2)
    prototype = cache.get(train_files[0])

    print("=" * 112)
    print("MPRT STATIC-ATLAS — END-TO-END RUNTIME SCALING")
    print("=" * 112)
    print("checkpoint       :", args.checkpoint)
    print("prototype        :", train_files[0])
    print("prototype N      :", int(prototype.xyz.shape[0]))
    print("activity length  :", int(prototype.activity.shape[-1]))
    print("static atlas M   :", atlas_size)
    print("device           :", torch.cuda.get_device_name(device))
    print("warmup           :", args.warmup)
    print("repeats / N      :", args.repeats)
    print("sizes            :", args.sizes)
    print("timed path       : encode_population(query) + match_encodings(query, static_atlas)")
    print("excluded         : disk I/O, checkpoint load, one-time static atlas construction")
    print()

    rows = []

    with torch.inference_mode():
        for n in args.sizes:
            sample_cpu = make_scaled_sample(prototype, int(n), args.seed)
            sample = sample_cpu.to(device)

            # One dry run to catch shape/OOM problems.
            try:
                _ = model.match_encodings(model.encode_population(sample), atlas)
                torch.cuda.synchronize()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                row = {
                    "N": int(n),
                    "atlas_M": atlas_size,
                    "status": "OOM",
                }
                rows.append(row)
                print(f"N={n:4d}  OOM")
                continue

            # Warm up kernels/caches.
            for _ in range(args.warmup):
                _ = model.match_encodings(model.encode_population(sample), atlas)
            torch.cuda.synchronize()

            # Component timing.
            encode_times = cuda_time_ms(
                lambda: model.encode_population(sample),
                args.repeats,
            )

            query_encoding = model.encode_population(sample)
            torch.cuda.synchronize()
            match_times = cuda_time_ms(
                lambda: model.match_encodings(query_encoding, atlas),
                args.repeats,
            )

            # Total inference timing.
            total_times = cuda_time_ms(
                lambda: model.match_encodings(
                    model.encode_population(sample),
                    atlas,
                ),
                args.repeats,
            )

            # Peak memory for one production-path inference.
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            baseline_alloc = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)

            _ = model.match_encodings(model.encode_population(sample), atlas)
            torch.cuda.synchronize()

            peak_alloc = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            increment_alloc = max(0, peak_alloc - baseline_alloc)
            increment_reserved = max(0, peak_reserved - baseline_reserved)

            es = stats(encode_times)
            ms = stats(match_times)
            ts = stats(total_times)

            row = {
                "N": int(n),
                "atlas_M": atlas_size,
                "status": "OK",
                "encode_median_ms": es["median_ms"],
                "encode_p95_ms": es["p95_ms"],
                "match_median_ms": ms["median_ms"],
                "match_p95_ms": ms["p95_ms"],
                "total_mean_ms": ts["mean_ms"],
                "total_sd_ms": ts["sd_ms"],
                "total_median_ms": ts["median_ms"],
                "total_p95_ms": ts["p95_ms"],
                "total_min_ms": ts["min_ms"],
                "total_max_ms": ts["max_ms"],
                "peak_allocated_total_mb": peak_alloc / 2**20,
                "peak_reserved_total_mb": peak_reserved / 2**20,
                "peak_allocated_increment_mb": increment_alloc / 2**20,
                "peak_reserved_increment_mb": increment_reserved / 2**20,
            }
            rows.append(row)

            print(
                f"N={n:4d}  M={atlas_size:4d}  "
                f"encode={es['median_ms']:8.3f} ms  "
                f"match={ms['median_ms']:8.3f} ms  "
                f"total={ts['median_ms']:8.3f} ms  "
                f"p95={ts['p95_ms']:8.3f} ms  "
                f"peak={row['peak_allocated_total_mb']:8.1f} MB  "
                f"+act={row['peak_allocated_increment_mb']:8.1f} MB"
            )

    ok_rows = [r for r in rows if r.get("status") == "OK"]

    time_slope = empirical_slope(ok_rows, "total_median_ms")
    memory_slope = empirical_slope(ok_rows, "peak_allocated_increment_mb")

    print()
    print("=" * 112)
    print("EMPIRICAL SCALING")
    print("=" * 112)
    print(f"latency slope  (N>=64): time ~ N^{time_slope:.3f}")
    print(f"memory slope   (N>=64): incr_mem ~ N^{memory_slope:.3f}")

    # CSV
    csv_path = args.output_dir / "mprt_static_runtime_scaling.csv"
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # JSON provenance
    payload_out = {
        "protocol": "MPRT final static-atlas end-to-end inference runtime scaling v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "fold_root": str(args.fold_root.resolve()),
        "prototype_train_file": str(Path(train_files[0]).resolve()),
        "model_path": (
            "encode_population(query) -> match_encodings(query, precomputed static atlas)"
        ),
        "static_atlas_precomputed_outside_timing": True,
        "excluded_from_latency": [
            "disk IO",
            "NPZ loading",
            "checkpoint loading",
            "one-time static atlas construction",
        ],
        "activity_length": args.activity_length,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "sizes": args.sizes,
        "atlas_size": atlas_size,
        "seed": args.seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "empirical_scaling": {
            "latency_exponent_N_ge_64": time_slope,
            "incremental_peak_memory_exponent_N_ge_64": memory_slope,
        },
        "rows": rows,
    }

    json_path = args.output_dir / "mprt_static_runtime_scaling.json"
    json_path.write_text(
        json.dumps(payload_out, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    print()
    print("CSV :", csv_path)
    print("JSON:", json_path)
    print("DONE")


if __name__ == "__main__":
    main()
