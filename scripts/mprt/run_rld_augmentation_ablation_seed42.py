#!/usr/bin/env python3
"""RLD seed-42 five-fold ablation for node dropout vs geometry augmentation.

This runner deliberately reuses the already completed static baseline and
strong-combined result.  It trains only three missing controlled arms:

* node_only: variable node deletion, no geometry augmentation;
* weak_geometry: baseline 5% node deletion, weak geometry augmentation;
* node_weak_geometry: variable 2--10% deletion plus weak geometry.

Every learned arm rebuilds its identity-anchored static atlas from untouched
fold-train animals and is evaluated on untouched fold-validation animals.  No
test split is opened.  The implementation imports the previously validated
``run_domain_randomization_seed42.py`` utilities so metric definitions and
checkpoint handling remain identical to the strong-combined experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence


SEED = 42
FOLDS = (0, 1, 2, 3, 4)
ABLATION_ROOT_NAME = "mprt_v1_1_rld_augmentation_ablation_seed42_v1"
STRONG_ROOT_NAME = "mprt_v1_1_domain_randomization_seed42_v1"


def augmentation(
    *,
    node_min: float,
    node_max: float,
    probability: float,
    rotation: float,
    scale: float,
    shear: float,
    jitter: float,
    warp: float,
) -> dict[str, float | int]:
    return {
        "node_dropout_min": node_min,
        "node_dropout_max": node_max,
        "geometry_augmentation_probability": probability,
        "geometry_rotation_degrees": rotation,
        "geometry_anisotropic_scale": scale,
        "geometry_shear": shear,
        "geometry_jitter_std": jitter,
        "geometry_warp_std": warp,
        "geometry_warp_control_points": 4,
        "geometry_warp_bandwidth": 1.0,
        "augmentation_seed_offset": 2_000_003,
    }


ARMS: dict[str, dict[str, float | int]] = {
    "node_only": augmentation(
        node_min=0.02,
        node_max=0.15,
        probability=0.0,
        rotation=0.0,
        scale=0.0,
        shear=0.0,
        jitter=0.0,
        warp=0.0,
    ),
    "weak_geometry": augmentation(
        # Negative values tell the wrapper to retain the source run's fixed
        # synthetic-drop probability (currently 0.05).
        node_min=-1.0,
        node_max=-1.0,
        probability=0.50,
        rotation=3.0,
        scale=0.03,
        shear=0.01,
        jitter=0.005,
        warp=0.01,
    ),
    "node_weak_geometry": augmentation(
        node_min=0.02,
        node_max=0.10,
        probability=0.50,
        rotation=3.0,
        scale=0.03,
        shear=0.01,
        jitter=0.005,
        warp=0.01,
    ),
}

STRONG_CONFIG = augmentation(
    node_min=0.02,
    node_max=0.15,
    probability=0.80,
    rotation=10.0,
    scale=0.10,
    shear=0.05,
    jitter=0.015,
    warp=0.04,
)

ARM_LABELS = {
    "baseline": "Baseline (fixed drop 5%)",
    "node_only": "Node-only (drop 2-15%)",
    "weak_geometry": "Weak geometry",
    "node_weak_geometry": "Node + weak geometry",
    "strong_combined": "Strong combined (existing)",
}
METRICS = ("top1_real", "top5_real", "hungarian_accuracy")


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


def ablation_cell(repo_root: Path, arm: str, fold: int) -> Path:
    return (
        repo_root
        / "runs"
        / ABLATION_ROOT_NAME
        / arm
        / "rld"
        / f"fold{fold}"
        / f"seed{SEED}"
    )


def strong_cell(repo_root: Path, fold: int) -> Path:
    return (
        repo_root
        / "runs"
        / STRONG_ROOT_NAME
        / "rld"
        / f"fold{fold}"
        / f"seed{SEED}"
    )


def route_core(core: Any, arm: str, config: Mapping[str, float | int]) -> None:
    """Route core output paths to one isolated ablation namespace."""

    core.AUGMENTATION = dict(config)

    def routed_augmented_cell(repo_root: Path, dataset: str, fold: int) -> Path:
        if dataset != "rld":
            raise ValueError("This ablation runner is RLD-only")
        return ablation_cell(repo_root, arm, fold)

    core.augmented_cell = routed_augmented_cell


def fit_arm(
    core: Any,
    *,
    repo_root: Path,
    package_root: Path,
    wrapper: Path,
    arm: str,
    config: Mapping[str, float | int],
    folds: Sequence[int],
    gpus: Sequence[str],
) -> None:
    route_core(core, arm, config)
    tasks = list(folds)
    buckets = [tasks[index:: len(gpus)] for index in range(len(gpus))]

    def worker(gpu: str, bucket: Sequence[int]) -> None:
        for fold in bucket:
            core.fit_one(repo_root, package_root, wrapper, "rld", fold, gpu)

    print("\n" + "=" * 108, flush=True)
    print(f"[FIT ARM] {arm}: {ARM_LABELS[arm]}", flush=True)
    print("=" * 108, flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, bucket)
            for gpu, bucket in zip(gpus, buckets)
        ]
        for future in futures:
            future.result()


def evaluate_arm(
    core: Any,
    *,
    repo_root: Path,
    package_root: Path,
    arm: str,
    config: Mapping[str, float | int],
    folds: Sequence[int],
    device: Any,
) -> list[dict[str, Any]]:
    route_core(core, arm, config)
    return [
        core.evaluate_one_cell(repo_root, package_root, "rld", fold, device)
        for fold in folds
    ]


def load_strong_results(
    repo_root: Path,
    folds: Sequence[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for fold in folds:
        path = strong_cell(repo_root, fold) / "val_direct_atlas_comparison.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing completed strong-combined result: {path}\n"
                "Finish run_domain_randomization_seed42.py first."
            )
        value = read_json(path)
        if (
            value.get("dataset") != "rld"
            or int(value.get("fold", -1)) != fold
            or int(value.get("seed", -1)) != SEED
            or value.get("split") != "val"
        ):
            raise ValueError(f"Unexpected strong result metadata: {path}")
        recorded = value.get("augmentation")
        if recorded != STRONG_CONFIG:
            raise ValueError(
                f"Strong augmentation specification differs in {path}\n"
                f"expected={STRONG_CONFIG}\nrecorded={recorded}"
            )
        results.append(value)
    return results


def fold_map(results: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    mapped = {int(result["fold"]): result for result in results}
    if len(mapped) != len(results):
        raise ValueError("Duplicate folds in results")
    return mapped


def metric_values(
    arm: str,
    results: Sequence[Mapping[str, Any]],
    metric: str,
) -> list[float]:
    method = "baseline" if arm == "baseline" else "augmented"
    return [float(result["metrics"][method][metric]) for result in results]


def stats(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sd": statistics.stdev(values) if len(values) >= 2 else 0.0,
    }


def aggregate_arm(
    arm: str,
    results: Sequence[Mapping[str, Any]],
    baseline_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"metrics": {}, "delta": {}}
    for metric in METRICS:
        values = metric_values(arm, results, metric)
        baseline_values = metric_values("baseline", baseline_results, metric)
        if len(values) != len(baseline_values):
            raise ValueError("Arm and baseline cell counts differ")
        deltas = [value - base for value, base in zip(values, baseline_values)]
        output["metrics"][metric] = stats(values)
        output["delta"][metric] = {
            **stats(deltas),
            "wins": sum(value > 0.0 for value in deltas),
            "losses": sum(value < 0.0 for value in deltas),
            "ties": sum(value == 0.0 for value in deltas),
            "values": deltas,
        }
    return output


def assert_common_baseline(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    folds: Sequence[int],
) -> None:
    reference = fold_map(all_results["strong_combined"])
    for arm, results in all_results.items():
        current = fold_map(results)
        if set(current) != set(folds):
            raise ValueError(f"{arm} has incomplete folds: {sorted(current)}")
        for fold in folds:
            expected = reference[fold]["metrics"]["baseline"]
            observed = current[fold]["metrics"]["baseline"]
            for metric in METRICS:
                if abs(float(expected[metric]) - float(observed[metric])) > 1.0e-12:
                    raise RuntimeError(
                        f"Baseline changed for {arm} fold{fold} metric={metric}: "
                        f"{observed[metric]} vs {expected[metric]}"
                    )


def render(
    all_results: Mapping[str, Sequence[Mapping[str, Any]]],
    folds: Sequence[int],
) -> dict[str, Any]:
    baseline_results = all_results["strong_combined"]
    ordered_arms = (
        "baseline",
        "node_only",
        "weak_geometry",
        "node_weak_geometry",
        "strong_combined",
    )
    aggregate: dict[str, Any] = {}

    print("\n" + "=" * 124)
    print(f"[RLD AUGMENTATION ABLATION] folds={len(folds)} seed={SEED}")
    print("=" * 124)
    maps = {arm: fold_map(results) for arm, results in all_results.items()}
    for fold in folds:
        baseline = float(maps["strong_combined"][fold]["metrics"]["baseline"]["top1_real"])
        pieces = [f"fold{fold}: baseline={100 * baseline:.2f}%"]
        for arm in ordered_arms[1:]:
            value = float(maps[arm][fold]["metrics"]["augmented"]["top1_real"])
            pieces.append(f"{arm}={100 * value:.2f}% ({100 * (value - baseline):+.2f}pp)")
        print("  ".join(pieces))

    print("\nSUMMARY")
    for arm in ordered_arms:
        results = baseline_results if arm == "baseline" else all_results[arm]
        arm_summary = aggregate_arm(arm, results, baseline_results)
        aggregate[arm] = arm_summary
        metrics = arm_summary["metrics"]
        delta = arm_summary["delta"]["top1_real"]
        print(
            f"{ARM_LABELS[arm]:30s} "
            f"Top1={100 * metrics['top1_real']['mean']:.2f} ± "
            f"{100 * metrics['top1_real']['sd']:.2f}%  "
            f"Top5={100 * metrics['top5_real']['mean']:.2f} ± "
            f"{100 * metrics['top5_real']['sd']:.2f}%  "
            f"Hung={100 * metrics['hungarian_accuracy']['mean']:.2f} ± "
            f"{100 * metrics['hungarian_accuracy']['sd']:.2f}%  "
            f"ΔTop1={100 * delta['mean']:+.2f} ± {100 * delta['sd']:.2f}pp  "
            f"W/L/T={delta['wins']}/{delta['losses']}/{delta['ties']}"
        )

    eligible = [
        arm
        for arm in ordered_arms
        if aggregate[arm]["delta"]["top5_real"]["mean"] >= -0.0025
    ]
    recommended = max(
        eligible,
        key=lambda arm: (
            aggregate[arm]["metrics"]["top1_real"]["mean"],
            aggregate[arm]["metrics"]["hungarian_accuracy"]["mean"],
            aggregate[arm]["metrics"]["top5_real"]["mean"],
        ),
    )
    print(
        "\nValidation-only recommendation "
        f"(Top5 non-inferiority margin 0.25pp): {recommended}"
    )
    aggregate["recommended_arm"] = recommended
    return aggregate


def parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("Expected comma-separated values")
    return result


def parse_folds(value: str) -> tuple[int, ...]:
    try:
        folds = tuple(int(item) for item in parse_csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    unknown = sorted(set(folds) - set(FOLDS))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown folds: {unknown}")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RLD seed42 node-dropout vs geometry augmentation ablation"
    )
    parser.add_argument(
        "phase",
        choices=("fit", "evaluate", "fit-evaluate"),
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
    parser.add_argument(
        "--arms",
        type=parse_csv,
        default=tuple(ARMS),
        help="Subset of new arms to fit/evaluate; strong result is always loaded.",
    )
    parser.add_argument("--wrapper", type=Path, default=None)
    parser.add_argument("--core-runner", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    package_root = repo_root / "mprt_net_v1_1"
    wrapper = (
        args.wrapper.resolve()
        if args.wrapper is not None
        else repo_root / "train_with_domain_randomization.py"
    )
    core_runner = (
        args.core_runner.resolve()
        if args.core_runner is not None
        else repo_root / "run_domain_randomization_seed42.py"
    )
    for path in (package_root, wrapper, core_runner):
        if not path.exists():
            raise FileNotFoundError(path)

    unknown_arms = sorted(set(args.arms) - set(ARMS))
    if unknown_arms:
        raise ValueError(f"Unknown arms: {unknown_arms}; choices={sorted(ARMS)}")
    # Requiring all arms during evaluation prevents an accidental partial table.
    if args.phase in {"evaluate", "fit-evaluate"} and set(args.arms) != set(ARMS):
        raise ValueError("Evaluation requires all three controlled arms")

    # Pin the evaluator before any fallback code can import torch.  Training
    # subprocesses override this value with their individually assigned GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus[0]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(package_root))
    import run_domain_randomization_seed42 as core

    if Path(core.__file__).resolve() != core_runner:
        raise RuntimeError(
            f"Imported unexpected core runner: {core.__file__}; expected={core_runner}"
        )

    if args.phase in {"fit", "fit-evaluate"}:
        for arm in args.arms:
            fit_arm(
                core,
                repo_root=repo_root,
                package_root=package_root,
                wrapper=wrapper,
                arm=arm,
                config=ARMS[arm],
                folds=args.folds,
                gpus=args.gpus,
            )

    if args.phase in {"evaluate", "fit-evaluate"}:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for direct-atlas evaluation")
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda")
        all_results: dict[str, list[dict[str, Any]]] = {
            "strong_combined": load_strong_results(repo_root, args.folds)
        }
        for arm in args.arms:
            all_results[arm] = evaluate_arm(
                core,
                repo_root=repo_root,
                package_root=package_root,
                arm=arm,
                config=ARMS[arm],
                folds=args.folds,
                device=device,
            )
        assert_common_baseline(all_results, args.folds)
        aggregate = render(all_results, args.folds)
        root = repo_root / "runs" / ABLATION_ROOT_NAME
        output = {
            "protocol": "rld_augmentation_ablation_seed42_v1",
            "dataset": "rld",
            "split": "val",
            "seed": SEED,
            "folds": list(args.folds),
            "arms": {
                "baseline": {
                    "synthetic_drop_probability": 0.05,
                    "geometry_augmentation_probability": 0.0,
                },
                **ARMS,
                "strong_combined": STRONG_CONFIG,
            },
            "cells": all_results,
            "aggregate": aggregate,
            "test_access": False,
        }
        write_json(root / "val_ablation_summary.json", output)
        print(f"\nsaved={root / 'val_ablation_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
