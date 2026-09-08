#!/usr/bin/env python3
"""Validation-only RLD comparison of baseline vs weak geometry augmentation.

The experiment is deliberately restricted to two conditions:

* baseline: the completed MPRT CV5x3 runs with fixed 5% synthetic dropout;
* weak_geometry: the same training configuration plus weak geometry
  augmentation, retaining the fixed 5% synthetic dropout.

All 5 folds and the locked seeds 1/42/123 are paired.  Static relational atlases are
rebuilt from untouched fold-training animals and evaluated on untouched
fold-validation animals.  Test files are never opened.

The seed-42 namespace is kept compatible with the earlier augmentation pilot,
so completed seed-42 cells (including fold0) are reused safely.  Seeds 1/123
are written to a new CV5x3 namespace.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DATASET = "rld"
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (1, 42, 123)
BASELINE_ROOT_NAME = "mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
LEGACY_SEED42_ROOT_NAME = "mprt_v1_1_rld_augmentation_ablation_seed42_v1"
OUTPUT_ROOT_NAME = "mprt_v1_1_rld_weak_geometry_cv5x3_v1"
CORE_RUNNER_NAME = "run_domain_randomization_seed42.py"
WRAPPER_NAME = "train_with_domain_randomization.py"
PROTOCOL = "mprt_rld_weak_geometry_cv5x3_v1"
LEGACY_PROTOCOL = "mprt_domain_randomization_seed42_v1"
METRICS = ("top1_real", "top5_real", "hungarian_accuracy")


WEAK_GEOMETRY: dict[str, float | int] = {
    # Negative values retain the source trainer's fixed synthetic-drop rate.
    "node_dropout_min": -1.0,
    "node_dropout_max": -1.0,
    "geometry_augmentation_probability": 0.50,
    "geometry_rotation_degrees": 3.0,
    "geometry_anisotropic_scale": 0.03,
    "geometry_shear": 0.01,
    "geometry_jitter_std": 0.005,
    "geometry_warp_std": 0.01,
    "geometry_warp_control_points": 4,
    "geometry_warp_bandwidth": 1.0,
    "augmentation_seed_offset": 2_000_003,
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected a non-empty comma-separated list")
    return result


def parse_int_subset(value: str, allowed: Sequence[int], label: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in parse_csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError(f"Duplicate {label}: {result}")
    unknown = sorted(set(result) - set(allowed))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown {label}: {unknown}")
    return result


def parse_folds(value: str) -> tuple[int, ...]:
    return parse_int_subset(value, FOLDS, "folds")


def parse_seeds(value: str) -> tuple[int, ...]:
    return parse_int_subset(value, SEEDS, "seeds")


def baseline_cell(repo_root: Path, fold: int, seed: int) -> Path:
    return (
        repo_root
        / "runs"
        / BASELINE_ROOT_NAME
        / DATASET
        / f"fold{fold}"
        / f"seed{seed}"
    )


def weak_cell(repo_root: Path, fold: int, seed: int) -> Path:
    # Reuse the exact seed-42 pilot namespace; use the clean CV5x3 namespace
    # for the two new seeds.
    if seed == 42:
        return (
            repo_root
            / "runs"
            / LEGACY_SEED42_ROOT_NAME
            / "weak_geometry"
            / DATASET
            / f"fold{fold}"
            / f"seed{seed}"
        )
    return (
        repo_root
        / "runs"
        / OUTPUT_ROOT_NAME
        / "weak_geometry"
        / DATASET
        / f"fold{fold}"
        / f"seed{seed}"
    )


def output_root(repo_root: Path) -> Path:
    return repo_root / "runs" / OUTPUT_ROOT_NAME


def load_core(repo_root: Path, core_runner: Path | None = None) -> Any:
    path = (
        core_runner.resolve()
        if core_runner is not None
        else (repo_root / CORE_RUNNER_NAME).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    module_name = f"_mprt_domain_randomization_core_{os.getpid()}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load core runner: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise RuntimeError(f"Loaded unexpected core runner: {module.__file__}")
    return module


def configure_core(core: Any, seed: int) -> None:
    """Parameterize the validated seed-42 utilities for one isolated seed."""

    core.SEED = int(seed)
    core.AUGMENTATION = dict(WEAK_GEOMETRY)

    def routed_augmented_cell(repo_root: Path, dataset: str, fold: int) -> Path:
        if dataset != DATASET:
            raise ValueError("This comparison is RLD-only")
        return weak_cell(repo_root, int(fold), int(seed))

    core.augmented_cell = routed_augmented_cell

    # Seeds 1/123 use a protocol name that accurately describes CV5x3.  Seed42
    # keeps the original protocol string so existing pilot specifications can
    # be checked and reused byte-for-byte.
    if seed != 42:
        original_pairwise_spec = core.pairwise_spec
        original_atlas_spec = core.atlas_spec

        def pairwise_spec(*args: Any, **kwargs: Any) -> dict[str, Any]:
            value = original_pairwise_spec(*args, **kwargs)
            value["protocol"] = PROTOCOL
            return value

        def atlas_spec(*args: Any, **kwargs: Any) -> dict[str, Any]:
            value = original_atlas_spec(*args, **kwargs)
            value["protocol"] = PROTOCOL
            return value

        core.pairwise_spec = pairwise_spec
        core.atlas_spec = atlas_spec

    # Fail early if any purported baseline does not actually use the locked
    # fixed 5% synthetic-drop training configuration.
    original_base_training_values = core.base_training_values

    def checked_base_training_values(*args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        parser, values = original_base_training_values(*args, **kwargs)
        observed = float(values.get("synthetic_drop_probability", float("nan")))
        if not np.isfinite(observed) or abs(observed - 0.05) > 1.0e-12:
            raise ValueError(
                "Baseline synthetic_drop_probability is not the locked 0.05: "
                f"observed={observed}"
            )
        return parser, values

    core.base_training_values = checked_base_training_values


def resolve_wrapper(repo_root: Path, wrapper: Path | None) -> Path:
    path = wrapper.resolve() if wrapper is not None else (repo_root / WRAPPER_NAME).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def preflight_baselines(
    repo_root: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    wrapper: Path | None,
    core_runner: Path | None,
) -> None:
    """Audit every locked source cell before launching either GPU worker."""

    package_root = repo_root / "neurid"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    resolve_wrapper(repo_root, wrapper)
    core_path = (
        core_runner.resolve()
        if core_runner is not None
        else (repo_root / CORE_RUNNER_NAME).resolve()
    )
    if not core_path.is_file():
        raise FileNotFoundError(core_path)

    problems: list[str] = []
    for fold in folds:
        fold_root = (
            repo_root
            / "Data/Dunn_001623/cv5_grouped_v1"
            / f"fold_{fold}"
        )
        for split in ("train", "val"):
            split_root = fold_root / split
            if not split_root.is_dir() or not any(split_root.glob("*.npz")):
                problems.append(f"missing RLD NPZ split: {split_root}")
        for seed in seeds:
            cell = baseline_cell(repo_root, fold, seed)
            pairwise = cell / "pairwise/full/best.pt"
            pairwise_args = cell / "pairwise/full/args.json"
            static = cell / "static_atlas/anchored_pure.pt"
            for label, path in (
                ("pairwise checkpoint", pairwise),
                ("pairwise arguments", pairwise_args),
                ("static atlas", static),
            ):
                if not path.is_file():
                    problems.append(
                        f"rld fold{fold} seed{seed} missing {label}: {path}"
                    )
            if pairwise_args.is_file():
                try:
                    values = read_json(pairwise_args)
                    observed = float(values.get("synthetic_drop_probability", float("nan")))
                    if not np.isfinite(observed) or abs(observed - 0.05) > 1.0e-12:
                        problems.append(
                            f"rld fold{fold} seed{seed} synthetic_drop_probability="
                            f"{observed}, expected=0.05"
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                    problems.append(
                        f"rld fold{fold} seed{seed} invalid pairwise args: {error}"
                    )

    if problems:
        formatted = "\n".join(f"  - {problem}" for problem in problems)
        raise RuntimeError(
            "Baseline preflight failed before any new training was launched:\n"
            + formatted
        )
    print(
        f"[PREFLIGHT OK] baseline/static/data cells={len(folds) * len(seeds)} "
        f"folds={','.join(map(str, folds))} seeds={','.join(map(str, seeds))}",
        flush=True,
    )


def fit_one(
    repo_root: Path,
    fold: int,
    seed: int,
    gpu: str,
    wrapper: Path | None,
    core_runner: Path | None,
) -> None:
    package_root = repo_root / "neurid"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    wrapper_path = resolve_wrapper(repo_root, wrapper)
    core = load_core(repo_root, core_runner)
    configure_core(core, seed)
    print(
        f"[FIT] rld fold{fold} seed{seed} weak_geometry gpu={gpu} ",
        f"output={weak_cell(repo_root, fold, seed)}",
        flush=True,
    )
    core.fit_one(
        repo_root,
        package_root,
        wrapper_path,
        DATASET,
        fold,
        gpu,
    )


def run_fit(
    script_path: Path,
    repo_root: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    gpus: Sequence[str],
    wrapper: Path | None,
    core_runner: Path | None,
) -> None:
    tasks = list(itertools.product(folds, seeds))
    buckets = [tasks[index:: len(gpus)] for index in range(len(gpus))]

    def worker(gpu: str, bucket: Sequence[tuple[int, int]]) -> None:
        for fold, seed in bucket:
            command = [
                sys.executable,
                "-u",
                str(script_path),
                "_fit-one",
                "--repo-root",
                str(repo_root),
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--gpu",
                gpu,
            ]
            if wrapper is not None:
                command.extend(("--wrapper", str(wrapper.resolve())))
            if core_runner is not None:
                command.extend(("--core-runner", str(core_runner.resolve())))
            subprocess.run(command, cwd=repo_root, check=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, bucket)
            for gpu, bucket in zip(gpus, buckets)
        ]
        for future in futures:
            future.result()


def evaluation_protocol(seed: int) -> str:
    return LEGACY_PROTOCOL if seed == 42 else PROTOCOL


def evaluate_one(
    core: Any,
    repo_root: Path,
    package_root: Path,
    fold: int,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    import torch
    from mprt_net.data import WormCache, split_files

    configure_core(core, seed)
    cell_paths = core.paths(repo_root, DATASET, fold)
    expected_baseline = baseline_cell(repo_root, fold, seed) / "static_atlas/anchored_pure.pt"
    if cell_paths["baseline_static"].resolve() != expected_baseline.resolve():
        raise RuntimeError(
            f"Unexpected baseline path: {cell_paths['baseline_static']} != {expected_baseline}"
        )
    expected_weak = weak_cell(repo_root, fold, seed) / "static_atlas/anchored_pure.pt"
    if cell_paths["augmented_static"].resolve() != expected_weak.resolve():
        raise RuntimeError(
            f"Unexpected weak-geometry path: {cell_paths['augmented_static']} != {expected_weak}"
        )

    baseline_checkpoint = cell_paths["baseline_static"]
    weak_checkpoint = cell_paths["augmented_static"]
    for checkpoint in (baseline_checkpoint, weak_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    expected = {
        "protocol": evaluation_protocol(seed),
        "dataset": DATASET,
        "fold": fold,
        "seed": seed,
        "split": "val",
        "baseline_checkpoint": str(baseline_checkpoint.resolve()),
        "baseline_sha256": core.sha256(baseline_checkpoint),
        "augmented_checkpoint": str(weak_checkpoint.resolve()),
        "augmented_sha256": core.sha256(weak_checkpoint),
    }
    if cell_paths["evaluation"].is_file() and cell_paths["queries"].is_file():
        result = read_json(cell_paths["evaluation"])
        if result.get("spec") != expected:
            raise RuntimeError(f"Stale validation evaluation: {cell_paths['evaluation']}")
        if result.get("augmentation") != WEAK_GEOMETRY:
            raise RuntimeError(f"Unexpected augmentation: {cell_paths['evaluation']}")
        print(f"[REUSE] rld fold{fold} seed{seed} validation", flush=True)
        return result
    if cell_paths["evaluation"].exists() or cell_paths["queries"].exists():
        raise FileExistsError(f"Partial evaluation: {cell_paths['evaluation'].parent}")

    baseline_model, baseline_state = core.load_static_model(baseline_checkpoint, device)
    weak_model, weak_state = core.load_static_model(weak_checkpoint, device)
    baseline_mapping = {
        str(key): int(value)
        for key, value in baseline_state["atlas_identity_to_slot"].items()
    }
    weak_mapping = {
        str(key): int(value)
        for key, value in weak_state["atlas_identity_to_slot"].items()
    }
    if baseline_mapping != weak_mapping:
        raise RuntimeError("Baseline and weak-geometry atlases use different slots")

    baseline_atlas = baseline_model.atlas_encoding()
    weak_atlas = weak_model.atlas_encoding()
    fold_root = core.dataset_root(repo_root, DATASET, fold)
    files = split_files(fold_root, "val")
    cache = WormCache(activity_length=512, max_items=max(32, len(files)))
    totals = {"baseline": core.Totals(), "augmented": core.Totals()}
    records: list[dict[str, Any]] = []

    with torch.no_grad():
        for number, path in enumerate(files, start=1):
            sample_cpu = cache.get(path)
            sample = sample_cpu.to(device)
            targets = core.target_slots(sample, baseline_mapping)
            outputs = {
                "baseline": baseline_model.match_encodings(
                    baseline_model.encode_population(sample), baseline_atlas
                ),
                "augmented": weak_model.match_encodings(
                    weak_model.encode_population(sample), weak_atlas
                ),
            }
            details: dict[str, dict[int, dict[str, Any]]] = {}
            for method, output in outputs.items():
                value, method_details = core.evaluate_output(output, targets)
                totals[method].update(value)
                details[method] = method_details
            if set(details["baseline"]) != set(details["augmented"]):
                raise RuntimeError("Baseline and weak-geometry query sets differ")
            for query_index in sorted(details["baseline"]):
                baseline_detail = details["baseline"][query_index]
                weak_detail = details["augmented"][query_index]
                records.append(
                    {
                        "dataset": DATASET,
                        "fold": fold,
                        "seed": seed,
                        "split": "val",
                        "uid": sample_cpu.uid,
                        "source_path": sample_cpu.source_path,
                        "node_index": query_index,
                        "target_slot": baseline_detail["target"],
                        "baseline_prediction": baseline_detail["prediction"],
                        "weak_geometry_prediction": weak_detail["prediction"],
                        "baseline_rank": baseline_detail["rank"],
                        "weak_geometry_rank": weak_detail["rank"],
                        "baseline_correct": baseline_detail["correct"],
                        "weak_geometry_correct": weak_detail["correct"],
                    }
                )
            print(
                f"  rld fold{fold} seed{seed} val {number:03d}/{len(files):03d} "
                f"uid={sample_cpu.uid}",
                flush=True,
            )

    metrics = {method: value.compute() for method, value in totals.items()}
    if metrics["baseline"]["queries"] != metrics["augmented"]["queries"]:
        raise RuntimeError("Metric query counts differ")
    result = {
        "spec": expected,
        "dataset": DATASET,
        "fold": fold,
        "seed": seed,
        "split": "val",
        "queries": metrics["baseline"]["queries"],
        "metrics": metrics,
        "delta_top1": float(metrics["augmented"]["top1_real"])
        - float(metrics["baseline"]["top1_real"]),
        "delta_top5": float(metrics["augmented"]["top5_real"])
        - float(metrics["baseline"]["top5_real"]),
        "delta_hungarian": float(metrics["augmented"]["hungarian_accuracy"])
        - float(metrics["baseline"]["hungarian_accuracy"]),
        "augmentation": WEAK_GEOMETRY,
        "test_access": False,
    }
    write_json(cell_paths["evaluation"], result)
    write_csv(cell_paths["queries"], records)
    return result


def sample_stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def win_loss_tie(values: Sequence[float], tolerance: float = 1.0e-12) -> dict[str, int]:
    return {
        "wins": sum(value > tolerance for value in values),
        "losses": sum(value < -tolerance for value in values),
        "ties": sum(abs(value) <= tolerance for value in values),
    }


def exact_sign_flip_pvalue(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return float("nan")
    observed = abs(float(array.mean()))
    total = 1 << len(array)
    masks = np.arange(total, dtype=np.uint32)[:, None]
    bits = (masks >> np.arange(len(array), dtype=np.uint32)[None, :]) & 1
    signs = bits.astype(np.float64) * 2.0 - 1.0
    permuted = np.abs((signs * array[None, :]).mean(axis=1))
    return float(np.mean(permuted >= observed - 1.0e-15))


def hierarchical_bootstrap_ci(
    values: Mapping[tuple[int, int], float],
    folds: Sequence[int],
    seeds: Sequence[int],
    samples: int,
    rng_seed: int,
) -> tuple[float, float]:
    matrix = np.asarray(
        [[float(values[(fold, seed)]) for seed in seeds] for fold in folds],
        dtype=np.float64,
    )
    rng = np.random.default_rng(rng_seed)
    estimates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 10_000):
        count = min(10_000, samples - start)
        sampled_folds = rng.integers(0, len(folds), size=(count, len(folds)))
        sampled_seeds = rng.integers(
            0,
            len(seeds),
            size=(count, len(folds), len(seeds)),
        )
        fold_values = np.empty((count, len(folds)), dtype=np.float64)
        for fold_position in range(len(folds)):
            source_folds = sampled_folds[:, fold_position]
            source_seeds = sampled_seeds[:, fold_position, :]
            fold_values[:, fold_position] = matrix[
                source_folds[:, None], source_seeds
            ].mean(axis=1)
        estimates[start : start + count] = fold_values.mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def summarize(
    results: Sequence[Mapping[str, Any]],
    folds: Sequence[int],
    seeds: Sequence[int],
    bootstrap_samples: int,
) -> dict[str, Any]:
    expected_cells = {(fold, seed) for fold in folds for seed in seeds}
    mapped = {(int(row["fold"]), int(row["seed"])): row for row in results}
    if set(mapped) != expected_cells or len(mapped) != len(results):
        raise ValueError(
            f"Incomplete or duplicate cells: expected={sorted(expected_cells)} "
            f"observed={sorted(mapped)}"
        )

    aggregate: dict[str, Any] = {"baseline": {}, "weak_geometry": {}, "delta": {}}
    for method, output_name in (("baseline", "baseline"), ("augmented", "weak_geometry")):
        for metric in METRICS:
            values = [float(row["metrics"][method][metric]) for row in results]
            aggregate[output_name][metric] = sample_stats(values)

    metric_to_result_key = {
        "top1_real": "delta_top1",
        "top5_real": "delta_top5",
        "hungarian_accuracy": "delta_hungarian",
    }
    for index, metric in enumerate(METRICS):
        result_key = metric_to_result_key[metric]
        keyed_values = {
            (int(row["fold"]), int(row["seed"])): float(row[result_key])
            for row in results
        }
        values = list(keyed_values.values())
        low, high = hierarchical_bootstrap_ci(
            keyed_values,
            folds,
            seeds,
            bootstrap_samples,
            rng_seed=202_608_24 + index,
        )
        aggregate["delta"][metric] = {
            **sample_stats(values),
            **win_loss_tie(values),
            "hierarchical_bootstrap_ci95": [low, high],
            "exact_cell_sign_flip_p_two_sided": exact_sign_flip_pvalue(values),
            "values": values,
        }

    def group_summary(group_key: str, values: Sequence[int]) -> dict[str, Any]:
        grouped: dict[str, Any] = {}
        for group_value in values:
            selected = [row for row in results if int(row[group_key]) == group_value]
            grouped[str(group_value)] = {
                "cells": len(selected),
                "baseline_top1": sample_stats(
                    [float(row["metrics"]["baseline"]["top1_real"]) for row in selected]
                ),
                "weak_geometry_top1": sample_stats(
                    [float(row["metrics"]["augmented"]["top1_real"]) for row in selected]
                ),
                "delta_top1": {
                    **sample_stats([float(row["delta_top1"]) for row in selected]),
                    **win_loss_tie([float(row["delta_top1"]) for row in selected]),
                },
            }
        return grouped

    primary_delta = aggregate["delta"]["top1_real"]
    top5_delta = aggregate["delta"]["top5_real"]
    hungarian_delta = aggregate["delta"]["hungarian_accuracy"]
    selection_checks = {
        "complete_5fold_x_3seed_protocol": (
            set(folds) == set(FOLDS)
            and set(seeds) == set(SEEDS)
            and len(results) == 15
        ),
        "positive_mean_top1_delta": primary_delta["mean"] > 0.0,
        "top1_wins_at_least_10_of_15": primary_delta["wins"] >= 10,
        "top5_noninferior_within_0_25pp": top5_delta["mean"] >= -0.0025,
        "hungarian_noninferior_within_0_25pp": hungarian_delta["mean"] >= -0.0025,
    }
    recommendation = "weak_geometry" if all(selection_checks.values()) else "baseline"
    return {
        "cell_macro": aggregate,
        "by_seed": group_summary("seed", seeds),
        "by_fold": group_summary("fold", folds),
        "selection_rule": {
            "checks": selection_checks,
            "recommended": recommendation,
            "uses_validation_only": True,
        },
    }


def render(results: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    print("\n" + "=" * 126)
    print(f"[RLD BASELINE VS WEAK GEOMETRY] cells={len(results)} split=val")
    print("=" * 126)
    for row in sorted(results, key=lambda item: (int(item["seed"]), int(item["fold"]))):
        baseline = float(row["metrics"]["baseline"]["top1_real"])
        weak = float(row["metrics"]["augmented"]["top1_real"])
        print(
            f"seed{int(row['seed'])} fold{int(row['fold'])}: "
            f"baseline={100 * baseline:6.2f}%  "
            f"weak_geometry={100 * weak:6.2f}%  "
            f"delta={100 * (weak - baseline):+6.2f}pp"
        )

    macro = summary["cell_macro"]
    print("\nOVERALL CELL-MACRO (5 folds x 3 seeds)")
    for method, label in (("baseline", "Baseline"), ("weak_geometry", "Weak geometry")):
        metrics = macro[method]
        print(
            f"{label:15s} "
            f"Top1={100 * metrics['top1_real']['mean']:.2f} ± "
            f"{100 * metrics['top1_real']['sd']:.2f}%  "
            f"Top5={100 * metrics['top5_real']['mean']:.2f} ± "
            f"{100 * metrics['top5_real']['sd']:.2f}%  "
            f"Hung={100 * metrics['hungarian_accuracy']['mean']:.2f} ± "
            f"{100 * metrics['hungarian_accuracy']['sd']:.2f}%"
        )
    for metric, label in (
        ("top1_real", "Weak - baseline Top1"),
        ("top5_real", "Weak - baseline Top5"),
        ("hungarian_accuracy", "Weak - baseline Hung"),
    ):
        value = macro["delta"][metric]
        low, high = value["hierarchical_bootstrap_ci95"]
        print(
            f"{label:23s} {100 * value['mean']:+.2f} ± "
            f"{100 * value['sd']:.2f}pp  "
            f"hier-boot 95% CI [{100 * low:+.2f}, {100 * high:+.2f}]  "
            f"W/L/T={value['wins']}/{value['losses']}/{value['ties']}  "
            f"sign-flip p={value['exact_cell_sign_flip_p_two_sided']:.6f}"
        )

    print("\nBY SEED")
    for seed, value in summary["by_seed"].items():
        delta = value["delta_top1"]
        print(
            f"seed{seed}: baseline={100 * value['baseline_top1']['mean']:.2f}%  "
            f"weak={100 * value['weak_geometry_top1']['mean']:.2f}%  "
            f"delta={100 * delta['mean']:+.2f}pp  "
            f"W/L/T={delta['wins']}/{delta['losses']}/{delta['ties']}"
        )

    rule = summary["selection_rule"]
    print("\nPREDECLARED VALIDATION-ONLY SELECTION CHECKS")
    for name, passed in rule["checks"].items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    print(f"Recommendation: {rule['recommended']}")


def flat_cell_rows(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in sorted(results, key=lambda item: (int(item["seed"]), int(item["fold"]))):
        baseline = result["metrics"]["baseline"]
        weak = result["metrics"]["augmented"]
        rows.append(
            {
                "dataset": DATASET,
                "split": "val",
                "fold": int(result["fold"]),
                "seed": int(result["seed"]),
                "queries": int(result["queries"]),
                "baseline_top1": float(baseline["top1_real"]),
                "weak_geometry_top1": float(weak["top1_real"]),
                "delta_top1": float(result["delta_top1"]),
                "baseline_top5": float(baseline["top5_real"]),
                "weak_geometry_top5": float(weak["top5_real"]),
                "delta_top5": float(result["delta_top5"]),
                "baseline_hungarian": float(baseline["hungarian_accuracy"]),
                "weak_geometry_hungarian": float(weak["hungarian_accuracy"]),
                "delta_hungarian": float(result["delta_hungarian"]),
                "baseline_checkpoint": result["spec"]["baseline_checkpoint"],
                "weak_geometry_checkpoint": result["spec"]["augmented_checkpoint"],
            }
        )
    return rows


def run_evaluate(
    repo_root: Path,
    folds: Sequence[int],
    seeds: Sequence[int],
    gpu: str,
    core_runner: Path | None,
    bootstrap_samples: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    package_root = repo_root / "neurid"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    sys.path.insert(0, str(package_root))
    core = load_core(repo_root, core_runner)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for validation evaluation")
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    results: list[dict[str, Any]] = []
    for seed in seeds:
        for fold in folds:
            results.append(
                evaluate_one(core, repo_root, package_root, fold, seed, device)
            )

    summary = summarize(results, folds, seeds, bootstrap_samples)
    render(results, summary)
    root = output_root(repo_root)
    payload = {
        "protocol": PROTOCOL,
        "dataset": DATASET,
        "split": "val",
        "folds": list(folds),
        "seeds": list(seeds),
        "cells": results,
        "summary": summary,
        "conditions": {
            "baseline": {
                "source_root": BASELINE_ROOT_NAME,
                "synthetic_drop_probability": 0.05,
                "geometry_augmentation_probability": 0.0,
            },
            "weak_geometry": {
                "synthetic_drop_probability": 0.05,
                **WEAK_GEOMETRY,
            },
        },
        "seed42_reuse_root": LEGACY_SEED42_ROOT_NAME,
        "bootstrap_samples": bootstrap_samples,
        "test_access": False,
    }
    summary_path = root / "val_comparison_summary.json"
    cells_path = root / "val_comparison_cells.csv"
    write_json(summary_path, payload)
    write_csv(cells_path, flat_cell_rows(results))
    print(f"\nsaved={summary_path}")
    print(f"saved={cells_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="RLD CV5x3 validation-only baseline vs weak geometry comparison"
    )
    parser.add_argument(
        "phase",
        choices=("fit", "evaluate", "fit-evaluate", "_fit-one"),
        default="fit-evaluate",
        nargs="?",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/ubuntu/klb/nuclr/nuclr"),
    )
    parser.add_argument("--gpus", type=parse_csv, default=("0", "1"))
    parser.add_argument("--folds", type=parse_folds, default=FOLDS)
    parser.add_argument("--seeds", type=parse_seeds, default=SEEDS)
    parser.add_argument("--wrapper", type=Path, default=None)
    parser.add_argument("--core-runner", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    # Internal one-cell worker arguments.
    parser.add_argument("--fold", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--gpu", type=str, default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")

    if args.phase == "_fit-one":
        if args.fold not in FOLDS or args.seed not in SEEDS or args.gpu is None:
            raise ValueError("_fit-one requires valid --fold, --seed, and --gpu")
        fit_one(
            repo_root,
            args.fold,
            args.seed,
            args.gpu,
            args.wrapper,
            args.core_runner,
        )
        return

    gpus = tuple(args.gpus)
    if not gpus:
        raise ValueError("At least one GPU is required")
    preflight_baselines(
        repo_root,
        args.folds,
        args.seeds,
        args.wrapper,
        args.core_runner,
    )
    if args.phase in {"fit", "fit-evaluate"}:
        run_fit(
            Path(__file__).resolve(),
            repo_root,
            args.folds,
            args.seeds,
            gpus,
            args.wrapper,
            args.core_runner,
        )
    if args.phase in {"evaluate", "fit-evaluate"}:
        run_evaluate(
            repo_root,
            args.folds,
            args.seeds,
            gpus[0],
            args.core_runner,
            args.bootstrap_samples,
        )


if __name__ == "__main__":
    main()
