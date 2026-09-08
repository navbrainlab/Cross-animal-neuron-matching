from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from ..sinkhorn import contracted_relation_cost


def explicit_relation_cost(
    relation_a: torch.Tensor, relation_b: torch.Tensor, plan: torch.Tensor
) -> torch.Tensor:
    """O(N^4 d) tensor reference; intentionally used only below the memory cap."""

    normalized = plan / plan.sum().clamp_min(1e-8)
    difference = (
        relation_a[:, None, :, None, :] - relation_b[None, :, None, :, :]
    )
    return (
        difference.square().sum(dim=-1) * normalized[None, None, :, :]
    ).sum(dim=(-1, -2))


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _worker(args: argparse.Namespace) -> dict[str, Any]:
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed + args.n)
    relation_a = torch.randn(args.n, args.n, args.dimension, device=device)
    relation_b = torch.randn(args.n, args.n, args.dimension, device=device)
    plan = torch.rand(args.n, args.n, device=device)
    function = contracted_relation_cost if args.method == "contracted" else explicit_relation_cost
    for _ in range(args.warmup):
        output = function(relation_a, relation_b, plan)
        _ = float(output[0, 0])
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None
    timings: list[float] = []
    checksum = 0.0
    for _ in range(args.repeats):
        _synchronize(device)
        started = time.perf_counter()
        output = function(relation_a, relation_b, plan)
        _synchronize(device)
        timings.append(time.perf_counter() - started)
        checksum += float(output[0, 0])
    if device.type == "cuda":
        peak_bytes = max(
            0, torch.cuda.max_memory_allocated(device) - int(baseline_memory or 0)
        )
    else:
        peak_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    return {
        "method": args.method,
        "n": args.n,
        "dimension": args.dimension,
        "device": str(device),
        "repeats": args.repeats,
        "median_seconds": statistics.median(timings),
        "min_seconds": min(timings),
        "max_seconds": max(timings),
        "peak_bytes": peak_bytes,
        "checksum": checksum / args.repeats,
    }


def _worker_command(
    method: str, n: int, args: argparse.Namespace, worker_output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mprt_net.experiments.complexity",
        "--worker",
        "--method",
        method,
        "--n",
        str(n),
        "--dimension",
        str(args.dimension),
        "--warmup",
        str(args.warmup),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--worker-output",
        str(worker_output),
    ]


def _estimate_explicit_bytes(n: int, dimension: int) -> int:
    # Difference + squared temporary + reduced [N^4], conservative float32 estimate.
    return int(4 * n**4 * (2 * dimension + 1))


def _fit_slope(rows: list[dict[str, Any]], metric: str) -> float | None:
    usable = [row for row in rows if row.get(metric, 0) > 0]
    if len(usable) < 2:
        return None
    x = torch.tensor([math.log(float(row["n"])) for row in usable])
    y = torch.tensor([math.log(float(row[metric])) for row in usable])
    centered_x = x - x.mean()
    return float((centered_x * (y - y.mean())).sum() / centered_x.square().sum())


def _agreement(n: int, dimension: int, device: torch.device, seed: int) -> dict[str, float]:
    torch.manual_seed(seed)
    a = torch.randn(n, n, dimension, device=device)
    b = torch.randn(n, n, dimension, device=device)
    p = torch.rand(n, n, device=device)
    expected = explicit_relation_cost(a, b, p)
    actual = contracted_relation_cost(a, b, p)
    error = (actual - expected).abs()
    return {
        "n": n,
        "max_absolute_error": float(error.max()),
        "max_relative_error": float(
            (error / expected.abs().clamp_min(1e-8)).max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Isolated O(N^4) versus contracted O(N^3) FGW benchmark"
    )
    parser.add_argument("--sizes", default="32,64,128,256,512")
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--explicit-max-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--agreement-n", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("contracted", "explicit"), help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.method is None or args.n is None or args.worker_output is None:
            parser.error("Internal worker arguments are incomplete")
        value = _worker(args)
        args.worker_output.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return
    sizes = tuple(int(value) for value in args.sizes.split(","))
    if any(value < 2 for value in sizes):
        parser.error("All sizes must be at least 2")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = args.output.parent / (args.output.stem + "_workers")
    temporary_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for method in ("contracted", "explicit"):
        for n in sizes:
            estimate = _estimate_explicit_bytes(n, args.dimension)
            if method == "explicit" and estimate > args.explicit_max_bytes:
                skipped.append(
                    {
                        "method": method,
                        "n": n,
                        "reason": "estimated temporary memory exceeds cap",
                        "estimated_bytes": estimate,
                    }
                )
                continue
            worker_output = temporary_dir / f"{method}_n{n}.json"
            command = _worker_command(method, n, args, worker_output)
            print(f"benchmark method={method} n={n}", flush=True)
            completed = subprocess.run(command, text=True)
            if completed.returncode != 0:
                skipped.append(
                    {
                        "method": method,
                        "n": n,
                        "reason": f"worker exited {completed.returncode}",
                        "estimated_bytes": estimate,
                    }
                )
                continue
            rows.append(json.loads(worker_output.read_text(encoding="utf-8")))
    device = _device(args.device)
    agreement = _agreement(args.agreement_n, args.dimension, device, args.seed + 999)
    methods = {
        method: [row for row in rows if row["method"] == method]
        for method in ("contracted", "explicit")
    }
    payload = {
        "device": str(device),
        "dimension": args.dimension,
        "sizes": sizes,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "explicit_max_bytes": args.explicit_max_bytes,
        "agreement": agreement,
        "rows": rows,
        "skipped": skipped,
        "empirical_log_log_slopes": {
            method: {
                "time": _fit_slope(method_rows, "median_seconds"),
                "peak_memory": _fit_slope(method_rows, "peak_bytes"),
            }
            for method, method_rows in methods.items()
        },
        "theory": {
            "explicit": "O(d N^4) time and memory",
            "contracted": "O(d N^3) time and O(d N^2) materialized memory",
        },
    }
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
