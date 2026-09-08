#!/usr/bin/env python3
"""
GPU runtime scaling comparison on RLD, using the final production inference path
for each learned baseline.

Methods:
  1) fDNC current-grouped final seed42 scorer
  2) GeoTransformer current-grouped seed42 checkpoint
  3) MPRT final Static Atlas, transport disabled at inference (overhead control)
  4) MPRT final Static Atlas, full relation transport

Protocol:
  - one visible CUDA device (launch with CUDA_VISIBLE_DEVICES=1)
  - warmup=20
  - repeats=100
  - CUDA Event timing
  - fixed candidate/reference population M=124
  - query N in {32,64,128,256,512,1024}
  - disk I/O, checkpoint loading, CPU normalization/collation, and one-time atlas
    construction are outside timed regions
  - reports median, p95, mean±sample-SD, and peak allocated GPU memory

Important:
  * GeoTransformer refuses to use the old partition checkpoint. It requires the
    current-grouped selection file produced by the new grouped-CV rerun.
  * "MPRT transport-disabled" is NOT a performance ablation checkpoint. It uses
    the exact same final Static checkpoint and disables relation transport only
    for the runtime measurement, isolating transport overhead.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
MPRT_PACKAGE = ROOT / "mprt_net_v1_1"
MPRT_CHECKPOINT = (
    ROOT
    / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
    / "rld/fold0/seed42/static_atlas/anchored_pure.pt"
)
RLD_FOLD0 = ROOT / "Data/Dunn_001623/cv5_grouped_v1/fold_0"

GEO_ROOT = Path("/home/ubuntu/klb/nuclr/geotransformer_official")
GEO_EXP = GEO_ROOT / "experiments/geotransformer.rld.semantic"
GEO_SELECTION = (
    GEO_ROOT
    / "current_grouped_selection/rld/fold0/seed42/best_checkpoint.txt"
)

DEFAULT_OUT = ROOT / "runs/runtime_scaling_comparison_gpu1_v1"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sizes", nargs="+", type=int, default=[32, 64, 128, 256, 512, 1024])
    p.add_argument("--reference-size", type=int, default=124)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--seed", type=int, default=20260826)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=["fdnc", "geotransformer", "mprt_no_transport", "mprt"],
        default=["fdnc", "geotransformer", "mprt_no_transport", "mprt"],
    )
    return p.parse_args()


def require(path: Path, what: str):
    if not path.exists():
        raise FileNotFoundError(f"{what} missing: {path}")


def stat(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "sd_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def cuda_times(fn: Callable[[], Any], repeats: int):
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return [float(a.elapsed_time(b)) for a, b in zip(starts, ends)]


def benchmark_cuda_fn(
    *,
    method: str,
    n: int,
    m: int,
    fn: Callable[[], Any],
    device: torch.device,
    warmup: int,
    repeats: int,
):
    try:
        # dry run
        _ = fn()
        torch.cuda.synchronize()

        for _ in range(warmup):
            _ = fn()
        torch.cuda.synchronize()

        values = cuda_times(fn, repeats)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        baseline_alloc = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)

        _ = fn()
        torch.cuda.synchronize()

        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

        s = stat(values)
        return {
            "method": method,
            "N": int(n),
            "M": int(m),
            "status": "OK",
            **s,
            "peak_allocated_total_mb": peak_alloc / 2**20,
            "peak_reserved_total_mb": peak_reserved / 2**20,
            "peak_allocated_increment_mb": max(0, peak_alloc - baseline_alloc) / 2**20,
            "peak_reserved_increment_mb": max(0, peak_reserved - baseline_reserved) / 2**20,
        }
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {
            "method": method,
            "N": int(n),
            "M": int(m),
            "status": "OOM",
        }


def resample_rows_np(x: np.ndarray, n: int, seed: int, jitter_fraction: float = 1e-4):
    x = np.asarray(x, dtype=np.float32)
    base_n = len(x)
    if base_n < 2:
        raise ValueError("Need at least 2 prototype rows")
    if n <= base_n:
        idx = np.rint(np.linspace(0, base_n - 1, n)).astype(np.int64)
        out = x[idx].copy()
    else:
        idx = np.arange(n, dtype=np.int64) % base_n
        out = x[idx].copy()
        rng = np.random.default_rng(seed + n)
        scale = float(np.mean(np.std(out, axis=0)))
        scale = max(scale, 1e-6)
        out += rng.normal(0.0, jitter_fraction * scale, size=out.shape).astype(np.float32)
    return out.astype(np.float32, copy=False)


def resample_rows_torch(x: torch.Tensor, n: int, seed: int, jitter_fraction: float = 1e-4):
    cpu = x.detach().cpu().numpy()
    arr = resample_rows_np(cpu, n, seed, jitter_fraction)
    return torch.from_numpy(arr)


# -----------------------------------------------------------------------------
# MPRT
# -----------------------------------------------------------------------------

def setup_mprt(args, device):
    require(MPRT_PACKAGE, "MPRT package")
    require(MPRT_CHECKPOINT, "final Static MPRT checkpoint")
    require(RLD_FOLD0, "RLD current grouped fold0")

    if str(MPRT_PACKAGE) not in sys.path:
        sys.path.insert(0, str(MPRT_PACKAGE))

    from mprt_net.data import WormCache, WormSample, split_files
    from mprt_net.evaluate import load_checkpoint

    model, payload = load_checkpoint(MPRT_CHECKPOINT, device)
    model.eval()

    if not getattr(model, "atlas_is_initialized", False):
        raise RuntimeError("Final MPRT checkpoint has no initialized static atlas")

    atlas = model.atlas_encoding()
    actual_m = int(atlas.nodes.shape[0])
    if actual_m != args.reference_size:
        raise RuntimeError(
            f"MPRT final atlas has M={actual_m}, but benchmark requests M={args.reference_size}. "
            "Use --reference-size equal to the final atlas size for a strict comparison."
        )

    train_files = split_files(RLD_FOLD0, "train")
    if not train_files:
        raise RuntimeError("No RLD fold0 train files")
    proto = WormCache(activity_length=512, max_items=2).get(train_files[0])

    def make_sample(n):
        xyz = resample_rows_torch(proto.xyz, n, args.seed)
        activity = resample_rows_torch(proto.activity, n, args.seed + 100000)
        return WormSample(
            uid=f"runtime_N{n}",
            xyz=xyz,
            activity=activity,
            cell_ids=tuple(f"RUNTIME_{i:05d}" for i in range(n)),
            supervised_mask=torch.ones(n, dtype=torch.bool),
            source_path=f"<runtime-N{n}>",
        ).to(device)

    return {
        "model": model,
        "atlas": atlas,
        "prototype": str(train_files[0]),
        "make_sample": make_sample,
    }


@contextlib.contextmanager
def mprt_transport_enabled(model, enabled: bool):
    cfg = model.config
    old = bool(getattr(cfg, "use_relation_transport"))
    try:
        try:
            cfg.use_relation_transport = bool(enabled)
        except Exception:
            object.__setattr__(cfg, "use_relation_transport", bool(enabled))
        yield
    finally:
        try:
            cfg.use_relation_transport = old
        except Exception:
            object.__setattr__(cfg, "use_relation_transport", old)


def run_mprt(args, device, rows, full: bool):
    ctx = setup_mprt(args, device)
    model, atlas = ctx["model"], ctx["atlas"]
    enabled = bool(full)
    name = "MPRT" if full else "MPRT_transport_disabled"

    with torch.inference_mode(), mprt_transport_enabled(model, enabled):
        for n in args.sizes:
            sample = ctx["make_sample"](n)

            def fn():
                q = model.encode_population(sample)
                return model.match_encodings(q, atlas)

            row = benchmark_cuda_fn(
                method=name,
                n=n,
                m=args.reference_size,
                fn=fn,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            rows.append(row)
            print_row(row)

    del model, atlas
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# fDNC
# -----------------------------------------------------------------------------

def setup_fdnc(args, device):
    # Reuse exact final clean context + official score_pair module from the
    # already-audited robustness evaluator.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from scripts.robustness import eval_rld_robustness_fdnc_final_medoid as fdrob

    ctx = fdrob.load_locked_context(0, device)
    model = ctx["model"]
    model.eval()

    template_xyz = np.asarray(ctx["template"].xyz, dtype=np.float32)
    reference_xyz = resample_rows_np(
        template_xyz, args.reference_size, args.seed + 200000
    )
    reference_xyz = ctx["normalize_xyz"](reference_xyz)
    reference = torch.from_numpy(reference_xyz).to(device=device, dtype=torch.float32)

    return {
        "ctx": ctx,
        "model": model,
        "prototype_xyz": template_xyz,
        "reference": reference,
    }


def run_fdnc(args, device, rows):
    ctx = setup_fdnc(args, device)
    model = ctx["model"]
    fdnc_train = ctx["ctx"]["fdnc_train"]
    normalize_xyz = ctx["ctx"]["normalize_xyz"]
    reference = ctx["reference"]
    prototype = ctx["prototype_xyz"]

    with torch.inference_mode():
        for n in args.sizes:
            q_np = resample_rows_np(prototype, n, args.seed + 300000)
            q_np = normalize_xyz(q_np)
            query = torch.from_numpy(q_np).to(device=device, dtype=torch.float32)

            # Exact production scorer. It computes the directional pair scores;
            # no CPU conversion / Hungarian is included.
            def fn():
                return fdnc_train.score_pair(model, query, reference)

            row = benchmark_cuda_fn(
                method="fDNC",
                n=n,
                m=args.reference_size,
                fn=fn,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            rows.append(row)
            print_row(row)

    del model, reference
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# GeoTransformer
# -----------------------------------------------------------------------------

def parse_geo_checkpoint():
    require(GEO_SELECTION, "GeoTransformer current-grouped selection")
    text = GEO_SELECTION.read_text(encoding="utf-8")

    # Explicitly reject anything that looks like the old rld_fold1_seed42 namespace.
    m = re.search(r"^checkpoint=(.+)$", text, flags=re.M)
    if not m:
        raise RuntimeError(f"Cannot parse checkpoint= from {GEO_SELECTION}")
    ckpt = Path(m.group(1).strip()).expanduser()

    require(ckpt, "GeoTransformer selected checkpoint")
    low = str(ckpt)
    if "/rld_fold1_seed42/" in low or "/rld_fold2_seed42/" in low:
        raise RuntimeError(
            "Refusing old-partition GeoTransformer checkpoint. "
            f"Selected path is {ckpt}"
        )
    if "current_grouped" not in low:
        raise RuntimeError(
            "GeoTransformer checkpoint does not contain 'current_grouped' in its run namespace; "
            "refusing to benchmark ambiguous provenance:\n"
            f"{ckpt}"
        )
    return ckpt


def discover_geo_data_env():
    cfg_text = (GEO_EXP / "config.py").read_text(encoding="utf-8")
    m = re.search(
        r'_C\.data\.dataset_root\s*=\s*os\.environ\.get\(\s*["\']([^"\']+)["\']',
        cfg_text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Cannot discover GeoTransformer RLD dataset-root environment variable")
    return m.group(1)


def recursive_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, list):
        return [recursive_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(recursive_to_device(v, device) for v in x)
    if isinstance(x, dict):
        return {k: recursive_to_device(v, device) for k, v in x.items()}
    return x


def import_geo_modules():
    # GeoTransformer experiment uses local imports named config/dataset/model/loss.
    if str(GEO_ROOT) not in sys.path:
        sys.path.insert(0, str(GEO_ROOT))
    if str(GEO_EXP) not in sys.path:
        sys.path.insert(0, str(GEO_EXP))

    # Ensure we do not accidentally reuse another experiment's plain module.
    for name in ("config", "dataset", "model", "evaluate_semantic"):
        if name in sys.modules:
            del sys.modules[name]

    cfgmod = importlib.import_module("config")
    dsmod = importlib.import_module("dataset")
    modelmod = importlib.import_module("model")
    evalmod = importlib.import_module("evaluate_semantic")
    return cfgmod, dsmod, modelmod, evalmod


def setup_geotransformer(args, device):
    require(GEO_ROOT, "GeoTransformer repo")
    require(GEO_EXP, "GeoTransformer RLD semantic experiment")
    require(RLD_FOLD0, "current grouped RLD fold0")

    ckpt = parse_geo_checkpoint()
    env_name = discover_geo_data_env()
    os.environ[env_name] = str(RLD_FOLD0)
    os.environ["RUN_TAG"] = "runtime_scaling_current_grouped_fold0_seed42"
    os.environ["SEED"] = "42"

    cfgmod, dsmod, modelmod, evalmod = import_geo_modules()
    cfg = cfgmod.make_cfg()
    cfg.data.dataset_root = str(RLD_FOLD0)

    from geotransformer.utils.data import (
        calibrate_neighbors_stack_mode,
        registration_collate_fn_stack_mode,
    )

    # Calibrate using OUTER TRAIN only. This CPU setup is outside timed inference.
    train_dataset = dsmod.build_dataset(cfg, "train")
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )

    model = modelmod.create_model(cfg).to(device)
    evalmod.load_checkpoint(model, str(ckpt))
    model.eval()

    # Shape/statistics prototype from a TRAIN file only.
    proto_path = Path(train_dataset.files[0])
    with np.load(proto_path, allow_pickle=False) as z:
        proto_xyz = np.asarray(z["xyz"], dtype=np.float32)
    proto_xyz = proto_xyz[np.isfinite(proto_xyz).all(axis=1)]
    proto_xyz = dsmod.normalize_cloud(proto_xyz)

    reference = resample_rows_np(
        proto_xyz, args.reference_size, args.seed + 400000
    )

    def make_data(n):
        query = resample_rows_np(proto_xyz, n, args.seed + 500000)

        # Query is "ref", fixed candidate/template is "src", matching the
        # semantic fixed-template evaluator's test-query -> train-template direction.
        ref_ids = np.arange(n, dtype=np.int64)
        src_ids = np.arange(args.reference_size, dtype=np.int64)

        item = {
            "ref_points": query.astype(np.float32),
            "src_points": reference.astype(np.float32),
            "ref_feats": np.ones((n, 1), dtype=np.float32),
            "src_feats": np.ones((args.reference_size, 1), dtype=np.float32),
            "ref_ids": ref_ids,
            "src_ids": src_ids,
            "transform": np.eye(4, dtype=np.float32),
            "ref_name": f"runtime_query_N{n}.npz",
            "src_name": f"runtime_reference_M{args.reference_size}.npz",
            "pair_index": 0,
            "num_shared": int(min(n, args.reference_size)),
        }

        collated = registration_collate_fn_stack_mode(
            [item],
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            neighbor_limits,
            precompute_data=True,
        )
        return recursive_to_device(collated, device)

    return {
        "checkpoint": ckpt,
        "model": model,
        "make_data": make_data,
        "prototype": str(proto_path),
        "neighbor_limits": [int(x) for x in neighbor_limits],
    }


def run_geotransformer(args, device, rows):
    ctx = setup_geotransformer(args, device)
    model = ctx["model"]

    with torch.inference_mode():
        for n in args.sizes:
            # CPU neighborhood construction is deliberately excluded from GPU
            # inference latency, just like disk/preprocessing is excluded for others.
            data_dict = ctx["make_data"](n)

            def fn():
                return model(data_dict)

            row = benchmark_cuda_fn(
                method="GeoTransformer",
                n=n,
                m=args.reference_size,
                fn=fn,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            rows.append(row)
            print_row(row)

            del data_dict
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

def print_row(row):
    if row["status"] != "OK":
        print(
            f"{row['method']:<26s} N={row['N']:4d} M={row['M']:4d} "
            f"{row['status']}",
            flush=True,
        )
        return
    print(
        f"{row['method']:<26s} "
        f"N={row['N']:4d} M={row['M']:4d} "
        f"median={row['median_ms']:9.3f} ms "
        f"p95={row['p95_ms']:9.3f} ms "
        f"mean±sd={row['mean_ms']:9.3f}±{row['sd_ms']:.3f} ms "
        f"peak={row['peak_allocated_total_mb']:9.1f} MB "
        f"+act={row['peak_allocated_increment_mb']:9.1f} MB",
        flush=True,
    )


def empirical_slope(rows, method, field):
    pts = [
        (float(r["N"]), float(r[field]))
        for r in rows
        if r["method"] == method
        and r["status"] == "OK"
        and int(r["N"]) >= 64
        and float(r[field]) > 0
    ]
    if len(pts) < 3:
        return float("nan")
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # With CUDA_VISIBLE_DEVICES=1 this MUST print the physical GPU1 as logical cuda:0.
    print("=" * 126)
    print("UNIFIED GPU RUNTIME SCALING COMPARISON")
    print("=" * 126)
    print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("logical device       :", device)
    print("GPU                  :", torch.cuda.get_device_name(device))
    print("sizes                :", args.sizes)
    print("fixed reference M    :", args.reference_size)
    print("warmup / repeats     :", args.warmup, "/", args.repeats)
    print("methods              :", args.methods)
    print()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        print(
            "WARNING: CUDA_VISIBLE_DEVICES is not exactly '1'. "
            "The user requested physical GPU1.",
            flush=True,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = []
    failures = []

    runners = {
        "fdnc": run_fdnc,
        "geotransformer": run_geotransformer,
        "mprt_no_transport": lambda a, d, r: run_mprt(a, d, r, full=False),
        "mprt": lambda a, d, r: run_mprt(a, d, r, full=True),
    }

    for method in args.methods:
        print()
        print("=" * 126)
        print("METHOD:", method)
        print("=" * 126)
        try:
            runners[method](args, device, rows)
        except Exception as exc:
            failures.append({"method": method, "error": repr(exc)})
            print(f"[FAILED] {method}: {exc}", flush=True)
        torch.cuda.empty_cache()

    # Write raw rows even if one method is unavailable.
    csv_path = args.output_dir / "runtime_scaling_comparison.csv"
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if fields:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    summary = {}
    for method_name in sorted({r["method"] for r in rows}):
        summary[method_name] = {
            "latency_exponent_N_ge_64": empirical_slope(rows, method_name, "median_ms"),
            "incremental_memory_exponent_N_ge_64": empirical_slope(
                rows, method_name, "peak_allocated_increment_mb"
            ),
        }

    json_path = args.output_dir / "runtime_scaling_comparison.json"
    json_path.write_text(
        json.dumps(
            {
                "protocol": "RLD unified GPU runtime scaling comparison v1",
                "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "sizes": args.sizes,
                "reference_size": args.reference_size,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "timing": "CUDA Event; model GPU inference only",
                "excluded": [
                    "disk I/O",
                    "checkpoint loading",
                    "CPU input normalization",
                    "GeoTransformer CPU neighborhood/collate preprocessing",
                    "MPRT one-time static atlas construction",
                ],
                "mprt_transport_disabled_note": (
                    "same final Static checkpoint; relation transport disabled only "
                    "during runtime measurement to isolate transport overhead"
                ),
                "summary": summary,
                "failures": failures,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 126)
    print("EMPIRICAL SCALING, N>=64")
    print("=" * 126)
    for method_name, values in summary.items():
        print(
            f"{method_name:<26s} "
            f"latency ~ N^{values['latency_exponent_N_ge_64']:.3f}    "
            f"incr.memory ~ N^{values['incremental_memory_exponent_N_ge_64']:.3f}"
        )

    print()
    if failures:
        print("FAILURES:")
        for x in failures:
            print(" ", x["method"], "->", x["error"])
    print("CSV :", csv_path)
    print("JSON:", json_path)
    print("DONE")


if __name__ == "__main__":
    main()
