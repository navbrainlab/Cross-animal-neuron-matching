#!/usr/bin/env python3
"""
FULL wall-clock runtime scaling comparison for fixed-reference neuron matching.

Primary metric:
  scores_ready_ms:
      in-memory raw query -> required CPU preprocessing -> host-to-device transfer
      -> complete model inference -> full query x reference score matrix on CPU.

Secondary metric:
  assignment_ready_ms:
      scores_ready pipeline + scipy Hungarian assignment.

This benchmark intentionally includes GeoTransformer's CPU stack-mode
multiscale neighborhood/collation preprocessing, which is excluded by a
GPU-forward-only benchmark.

Excluded for every method:
  - disk I/O / NPZ loading
  - Python/module import
  - checkpoint loading
  - one-time fixed reference / static atlas selection or construction

Fixed-reference deployment semantics:
  - MPRT static atlas is constructed once and resident on GPU.
  - fDNC fixed template is selected once; its normalized tensor is resident on GPU.
  - GeoTransformer fixed reference coordinates are selected once; however its
    official stack-mode collate/precompute graph is pair-dependent and therefore
    is rebuilt for every query and INCLUDED in the online latency.

Methods:
  - fDNC
  - GeoTransformer (CURRENT-GROUPED checkpoint only)
  - MPRT final Static Atlas
  - MPRT same final checkpoint with relation transport disabled (runtime control)

Launch with:
    CUDA_VISIBLE_DEVICES=1 python -u benchmark_full_wallclock_runtime_gpu1.py ...

For a physical GPU1 process, logical device is cuda:0.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


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

DEFAULT_OUT = ROOT / "runs/full_wallclock_runtime_gpu1_v1"


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


def summarize(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "sd_ms": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "median_ms": float(np.median(a)),
        "p95_ms": float(np.percentile(a, 95)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def timed_wallclock(fn: Callable[[], Any], repeats: int):
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        # All pipelines are required to return CPU scores, so GPU work should
        # already be synchronized by D2H. Keep this explicit for safety.
        torch.cuda.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return vals


def run_warmup(fn, warmup: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()


def assignment_from_scores(scores: np.ndarray):
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise RuntimeError(f"Expected 2D score matrix, got {scores.shape}")
    if scores.size == 0:
        return np.empty((0,), np.int64), np.empty((0,), np.int64)
    finite = np.isfinite(scores)
    if not finite.all():
        work = scores.copy()
        work[~finite] = -1e12
    else:
        work = scores
    return linear_sum_assignment(-work)


def resample_rows_np(x: np.ndarray, n: int, seed: int, jitter_fraction: float = 1e-4):
    x = np.asarray(x, dtype=np.float32)
    base_n = len(x)
    if base_n < 2:
        raise ValueError("Need at least two prototype rows")

    if n <= base_n:
        idx = np.rint(np.linspace(0, base_n - 1, n)).astype(np.int64)
        out = x[idx].copy()
    else:
        idx = np.arange(n, dtype=np.int64) % base_n
        out = x[idx].copy()
        rng = np.random.default_rng(seed + n)
        if out.ndim == 1:
            scale = float(np.std(out))
        else:
            scale = float(np.mean(np.std(out, axis=0)))
        scale = max(scale, 1e-6)
        out += rng.normal(
            0.0, jitter_fraction * scale, size=out.shape
        ).astype(np.float32)
    return out.astype(np.float32, copy=False)


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
    x = np.log(np.asarray([p[0] for p in pts]))
    y = np.log(np.asarray([p[1] for p in pts]))
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def benchmark_pipeline(
    *,
    method: str,
    n: int,
    m: int,
    scores_fn: Callable[[], np.ndarray],
    device: torch.device,
    warmup: int,
    repeats: int,
):
    try:
        # Validate output.
        scores = scores_fn()
        if not isinstance(scores, np.ndarray):
            scores = np.asarray(scores)
        if scores.shape != (n, m):
            raise RuntimeError(
                f"{method}: score shape {scores.shape}, expected {(n, m)}"
            )

        # Warm up complete online pipeline.
        run_warmup(scores_fn, warmup)

        # Scores-ready latency.
        score_times = timed_wallclock(scores_fn, repeats)

        # Assignment-ready latency: rerun the complete score pipeline and
        # immediately solve Hungarian on the CPU score matrix.
        def score_plus_assignment():
            s = scores_fn()
            assignment_from_scores(s)
            return s

        run_warmup(score_plus_assignment, min(warmup, 5))
        assign_times = timed_wallclock(score_plus_assignment, repeats)

        # Isolate Hungarian itself on a fixed representative score matrix.
        fixed_scores = scores_fn()
        hung_times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            assignment_from_scores(fixed_scores)
            hung_times.append((time.perf_counter() - t0) * 1000.0)

        # Peak GPU memory of the full scores-ready inference call.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline_alloc = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)

        _ = scores_fn()
        torch.cuda.synchronize()

        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

        ss = summarize(score_times)
        aa = summarize(assign_times)
        hh = summarize(hung_times)

        return {
            "method": method,
            "N": int(n),
            "M": int(m),
            "status": "OK",
            "scores_ready_mean_ms": ss["mean_ms"],
            "scores_ready_sd_ms": ss["sd_ms"],
            "scores_ready_median_ms": ss["median_ms"],
            "scores_ready_p95_ms": ss["p95_ms"],
            "assignment_ready_mean_ms": aa["mean_ms"],
            "assignment_ready_sd_ms": aa["sd_ms"],
            "assignment_ready_median_ms": aa["median_ms"],
            "assignment_ready_p95_ms": aa["p95_ms"],
            "hungarian_only_median_ms": hh["median_ms"],
            "hungarian_only_p95_ms": hh["p95_ms"],
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


def print_row(r):
    if r["status"] != "OK":
        print(
            f"{r['method']:<26s} N={r['N']:4d} M={r['M']:4d} {r['status']}",
            flush=True,
        )
        return

    print(
        f"{r['method']:<26s} "
        f"N={r['N']:4d} M={r['M']:4d} "
        f"scores={r['scores_ready_median_ms']:10.3f} ms "
        f"p95={r['scores_ready_p95_ms']:10.3f} ms "
        f"assign={r['assignment_ready_median_ms']:10.3f} ms "
        f"hung={r['hungarian_only_median_ms']:8.3f} ms "
        f"peak={r['peak_allocated_total_mb']:9.1f} MB",
        flush=True,
    )


# =============================================================================
# MPRT
# =============================================================================

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
        raise RuntimeError("MPRT checkpoint has no initialized static atlas")

    # One-time static atlas construction: intentionally outside online timing.
    atlas = model.atlas_encoding()
    atlas_m = int(atlas.nodes.shape[0])
    if atlas_m != args.reference_size:
        raise RuntimeError(
            f"Final static atlas M={atlas_m}, requested M={args.reference_size}"
        )

    train_files = split_files(RLD_FOLD0, "train")
    proto = WormCache(activity_length=512, max_items=2).get(train_files[0])

    proto_xyz = proto.xyz.detach().cpu().numpy()
    proto_activity = proto.activity.detach().cpu().numpy()

    def make_raw(n):
        # Generated once per N, outside repeated timing. This represents an
        # already-loaded in-memory raw query.
        xyz = resample_rows_np(proto_xyz, n, args.seed + 10000)
        act = resample_rows_np(proto_activity, n, args.seed + 20000)
        return xyz, act

    def make_scores_fn(n):
        raw_xyz, raw_activity = make_raw(n)

        def scores_fn():
            # Query object creation + CPU->GPU transfer ARE included.
            sample_cpu = WormSample(
                uid=f"wallclock_N{n}",
                xyz=torch.from_numpy(raw_xyz),
                activity=torch.from_numpy(raw_activity),
                cell_ids=tuple(f"RUNTIME_{i:05d}" for i in range(n)),
                supervised_mask=torch.ones(n, dtype=torch.bool),
                source_path=f"<wallclock-N{n}>",
            )
            sample = sample_cpu.to(device)
            with torch.inference_mode():
                query = model.encode_population(sample)
                output = model.match_encodings(query, atlas)
                # Final full real-candidate score matrix on CPU.
                scores = output.row_conditional[:, :args.reference_size]
                return scores.detach().float().cpu().numpy()

        return scores_fn

    return model, atlas, make_scores_fn, str(train_files[0])


@contextlib.contextmanager
def transport_enabled(model, enabled):
    cfg = model.config
    old = bool(cfg.use_relation_transport)
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


def run_mprt(args, device, rows, full=True):
    model, atlas, make_scores_fn, proto = setup_mprt(args, device)
    name = "MPRT" if full else "MPRT_transport_disabled"

    with transport_enabled(model, full):
        for n in args.sizes:
            row = benchmark_pipeline(
                method=name,
                n=n,
                m=args.reference_size,
                scores_fn=make_scores_fn(n),
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            rows.append(row)
            print_row(row)

    del model, atlas
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# fDNC
# =============================================================================

def setup_fdnc(args, device):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from scripts.robustness import eval_rld_robustness_fdnc_final_medoid as fdrob

    ctx = fdrob.load_locked_context(0, device)
    model = ctx["model"]
    model.eval()

    template_xyz = np.asarray(ctx["template"].xyz, dtype=np.float32)

    # Fixed-reference preprocessing is one-time in the deployed template protocol.
    ref_raw = resample_rows_np(
        template_xyz, args.reference_size, args.seed + 30000
    )
    ref_norm = ctx["normalize_xyz"](ref_raw)
    reference_gpu = torch.from_numpy(ref_norm).to(
        device=device, dtype=torch.float32
    )

    def make_scores_fn(n):
        raw_query = resample_rows_np(
            template_xyz, n, args.seed + 40000
        )

        def scores_fn():
            # Online query normalization INCLUDED.
            query_norm = ctx["normalize_xyz"](raw_query)
            query_gpu = torch.from_numpy(query_norm).to(
                device=device, dtype=torch.float32
            )
            with torch.inference_mode():
                # Exact official pair scorer.
                _, query_to_reference = ctx["fdnc_train"].score_pair(
                    model, query_gpu, reference_gpu
                )
                scores = query_to_reference[:, :-1]
                if scores.shape[1] != args.reference_size:
                    raise RuntimeError(
                        f"fDNC candidate width {scores.shape[1]} != {args.reference_size}"
                    )
                return scores.detach().float().cpu().numpy()

        return scores_fn

    return model, reference_gpu, make_scores_fn


def run_fdnc(args, device, rows):
    model, reference_gpu, make_scores_fn = setup_fdnc(args, device)

    for n in args.sizes:
        row = benchmark_pipeline(
            method="fDNC",
            n=n,
            m=args.reference_size,
            scores_fn=make_scores_fn(n),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rows.append(row)
        print_row(row)

    del model, reference_gpu
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# GeoTransformer
# =============================================================================

def parse_geo_checkpoint():
    require(GEO_SELECTION, "GeoTransformer current-grouped selection")
    text = GEO_SELECTION.read_text(encoding="utf-8")
    m = re.search(r"^checkpoint=(.+)$", text, flags=re.M)
    if not m:
        raise RuntimeError(f"Cannot parse checkpoint= from {GEO_SELECTION}")

    ckpt = Path(m.group(1).strip()).expanduser()
    require(ckpt, "GeoTransformer selected checkpoint")

    low = str(ckpt)
    if "current_grouped" not in low:
        raise RuntimeError(
            "Refusing GeoTransformer checkpoint without current_grouped provenance: "
            f"{ckpt}"
        )
    return ckpt


def discover_geo_data_env():
    text = (GEO_EXP / "config.py").read_text(encoding="utf-8")
    m = re.search(
        r'_C\.data\.dataset_root\s*=\s*os\.environ\.get\(\s*["\']([^"\']+)["\']',
        text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Cannot discover GeoTransformer dataset-root env variable")
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
    if str(GEO_ROOT) not in sys.path:
        sys.path.insert(0, str(GEO_ROOT))
    if str(GEO_EXP) not in sys.path:
        sys.path.insert(0, str(GEO_EXP))

    for name in ("config", "dataset", "model", "loss", "evaluate_semantic"):
        if name in sys.modules:
            del sys.modules[name]

    cfgmod = importlib.import_module("config")
    dsmod = importlib.import_module("dataset")
    modelmod = importlib.import_module("model")
    lossmod = importlib.import_module("loss")
    evalmod = importlib.import_module("evaluate_semantic")
    return cfgmod, dsmod, modelmod, lossmod, evalmod


def setup_geotransformer(args, device):
    require(GEO_ROOT, "GeoTransformer repo")
    require(GEO_EXP, "GeoTransformer RLD experiment")
    require(RLD_FOLD0, "RLD current grouped fold0")

    ckpt = parse_geo_checkpoint()
    env_name = discover_geo_data_env()
    os.environ[env_name] = str(RLD_FOLD0)
    os.environ["RUN_TAG"] = "full_wallclock_runtime_current_grouped_fold0_seed42"
    os.environ["SEED"] = "42"

    cfgmod, dsmod, modelmod, lossmod, evalmod = import_geo_modules()
    cfg = cfgmod.make_cfg()
    cfg.data.dataset_root = str(RLD_FOLD0)

    from geotransformer.utils.data import (
        calibrate_neighbors_stack_mode,
        registration_collate_fn_stack_mode,
    )

    # Neighbor limits calibration is a training/configuration artifact, not per-query
    # inference, so it is correctly outside the timing.
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

    # In-memory raw coordinates from a TRAIN prototype.
    proto_path = Path(train_dataset.files[0])
    with np.load(proto_path, allow_pickle=False) as z:
        raw_proto = np.asarray(z["xyz"], dtype=np.float32)
    raw_proto = raw_proto[np.isfinite(raw_proto).all(axis=1)]

    # Fixed reference raw coordinates selected once.
    raw_reference = resample_rows_np(
        raw_proto, args.reference_size, args.seed + 50000
    )

    def make_scores_fn(n):
        raw_query = resample_rows_np(
            raw_proto, n, args.seed + 60000
        )

        def scores_fn():
            # IMPORTANT: all official per-query CPU preprocessing is INCLUDED:
            # normalization, feature creation, stack-mode multiscale graph /
            # neighborhood precomputation, then H2D.
            query = dsmod.normalize_cloud(raw_query)
            reference = dsmod.normalize_cloud(raw_reference)

            item = {
                "ref_points": query.astype(np.float32, copy=False),
                "src_points": reference.astype(np.float32, copy=False),
                "ref_feats": np.ones((n, 1), dtype=np.float32),
                "src_feats": np.ones(
                    (args.reference_size, 1), dtype=np.float32
                ),
                "ref_ids": np.arange(n, dtype=np.int64),
                "src_ids": np.arange(args.reference_size, dtype=np.int64),
                "transform": np.eye(4, dtype=np.float32),
                "ref_name": f"wallclock_query_N{n}.npz",
                "src_name": f"wallclock_reference_M{args.reference_size}.npz",
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
            data_gpu = recursive_to_device(collated, device)

            with torch.inference_mode():
                output = model(data_gpu)
                # Official dense score construction INCLUDED.
                dense_ref_by_src = lossmod.build_dense_scores(output)

                # We want query x fixed-reference. Here query was "ref" and
                # fixed reference was "src", so dense already has [N, M].
                scores = dense_ref_by_src
                if scores.shape != (n, args.reference_size):
                    raise RuntimeError(
                        f"Geo dense shape {tuple(scores.shape)} != "
                        f"{(n, args.reference_size)}"
                    )
                return scores.detach().float().cpu().numpy()

        return scores_fn

    return model, make_scores_fn, ckpt, neighbor_limits, proto_path


def run_geotransformer(args, device, rows):
    model, make_scores_fn, ckpt, neighbor_limits, proto = setup_geotransformer(
        args, device
    )

    print("Geo current-grouped checkpoint:", ckpt)
    print("Geo neighbor limits:", [int(x) for x in neighbor_limits])
    print("Geo prototype:", proto)

    for n in args.sizes:
        row = benchmark_pipeline(
            method="GeoTransformer",
            n=n,
            m=args.reference_size,
            scores_fn=make_scores_fn(n),
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        rows.append(row)
        print_row(row)

    del model
    gc.collect()
    torch.cuda.empty_cache()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    print("=" * 132)
    print("FULL WALL-CLOCK FIXED-REFERENCE RUNTIME SCALING")
    print("=" * 132)
    print("CUDA_VISIBLE_DEVICES :", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("logical device       :", device)
    print("GPU                  :", torch.cuda.get_device_name(device))
    print("sizes                :", args.sizes)
    print("fixed reference M    :", args.reference_size)
    print("warmup / repeats     :", args.warmup, "/", args.repeats)
    print("methods              :", args.methods)
    print("primary metric       : in-memory raw query -> CPU full score matrix")
    print("secondary metric     : above + Hungarian assignment")
    print("includes             : CPU query preprocessing, graph/collate, H2D, GPU forward, score materialization, D2H")
    print("excludes             : disk I/O, imports, checkpoint loading, one-time template/atlas construction")
    print()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        print(
            "WARNING: user requested physical GPU1; CUDA_VISIBLE_DEVICES is not exactly '1'.",
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
        print("=" * 132)
        print("METHOD:", method)
        print("=" * 132)

        try:
            runners[method](args, device, rows)
        except Exception as exc:
            failures.append({"method": method, "error": repr(exc)})
            print(f"[FAILED] {method}: {exc}", flush=True)

        gc.collect()
        torch.cuda.empty_cache()

    csv_path = args.output_dir / "full_wallclock_runtime.csv"
    fields = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)

    if fields:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    methods_seen = sorted({r["method"] for r in rows})
    scaling = {}
    for method in methods_seen:
        scaling[method] = {
            "scores_ready_exponent_N_ge_64": empirical_slope(
                rows, method, "scores_ready_median_ms"
            ),
            "assignment_ready_exponent_N_ge_64": empirical_slope(
                rows, method, "assignment_ready_median_ms"
            ),
            "incremental_memory_exponent_N_ge_64": empirical_slope(
                rows, method, "peak_allocated_increment_mb"
            ),
        }

    json_path = args.output_dir / "full_wallclock_runtime.json"
    json_path.write_text(
        json.dumps(
            {
                "protocol": "full wall-clock fixed-reference inference v1",
                "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "logical_device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "sizes": args.sizes,
                "reference_size": args.reference_size,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "primary_metric": (
                    "in-memory raw query -> preprocessing -> H2D -> complete "
                    "model -> full score matrix materialized on CPU"
                ),
                "secondary_metric": "primary metric + scipy Hungarian assignment",
                "included": [
                    "per-query CPU normalization",
                    "per-query feature preparation",
                    "GeoTransformer stack-mode multiscale graph/neighborhood preprocessing",
                    "host-to-device transfer",
                    "GPU model forward",
                    "method-specific final score construction",
                    "device-to-host score materialization",
                ],
                "excluded": [
                    "disk I/O",
                    "module import",
                    "checkpoint loading",
                    "one-time fixed template selection",
                    "one-time static atlas construction",
                    "GeoTransformer one-time neighbor-limit calibration",
                ],
                "fixed_reference_semantics": {
                    "MPRT": "static atlas constructed once and resident on GPU",
                    "fDNC": "fixed template selected once; normalized template tensor resident on GPU",
                    "GeoTransformer": (
                        "fixed reference coordinates selected once, but official "
                        "pair-dependent stack-mode graph/collate preprocessing is rebuilt "
                        "per query and included"
                    ),
                },
                "scaling": scaling,
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
    print("=" * 132)
    print("EMPIRICAL FULL WALL-CLOCK SCALING, N>=64")
    print("=" * 132)
    for method, s in scaling.items():
        print(
            f"{method:<26s} "
            f"scores ~ N^{s['scores_ready_exponent_N_ge_64']:.3f}   "
            f"assignment ~ N^{s['assignment_ready_exponent_N_ge_64']:.3f}   "
            f"memory ~ N^{s['incremental_memory_exponent_N_ge_64']:.3f}"
        )

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(" ", item["method"], "->", item["error"])

    print("\nCSV :", csv_path)
    print("JSON:", json_path)
    print("DONE")


if __name__ == "__main__":
    main()
