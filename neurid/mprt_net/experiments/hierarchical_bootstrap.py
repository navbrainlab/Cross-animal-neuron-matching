from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CellKey = tuple[str, int, int]


def _load_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    entries = value.get("entries") if isinstance(value, dict) else None
    if not isinstance(entries, list):
        raise ValueError("Manifest must be an object containing an entries list")
    selected = [row for row in entries if row.get("split") == split]
    if not selected:
        raise ValueError(f"Manifest contains no split={split!r} entries")
    return selected


def _read_cells(entries: list[dict[str, Any]]) -> dict[CellKey, dict[str, np.ndarray]]:
    grouped: dict[CellKey, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen_paths: set[Path] = set()
    for entry in entries:
        path = Path(entry["query_csv"]).resolve()
        if path in seen_paths:
            raise ValueError(f"Duplicate query CSV in manifest: {path}")
        seen_paths.add(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"pair_id", "correct_a", "correct_b"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError(f"{path} is missing columns {sorted(required)}")
            for row in reader:
                dataset = str(entry.get("dataset", row.get("dataset", "dataset")))
                fold = int(entry.get("fold", row.get("fold", -1)))
                seed = int(entry.get("seed", row.get("seed", -1)))
                key = (dataset, fold, seed)
                grouped[key][str(row["pair_id"])].append(
                    float(row["correct_b"]) - float(row["correct_a"])
                )
    result: dict[CellKey, dict[str, np.ndarray]] = {}
    for key, pairs in grouped.items():
        result[key] = {
            pair: np.asarray(values, dtype=np.float64) for pair, values in pairs.items()
        }
    return result


def _validate_rectangular(cells: dict[CellKey, Any]) -> dict[str, dict[int, tuple[int, ...]]]:
    design: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for dataset, fold, seed in cells:
        design[dataset][fold].add(seed)
    rendered: dict[str, dict[int, tuple[int, ...]]] = {}
    for dataset, folds in design.items():
        seed_sets = {tuple(sorted(seeds)) for seeds in folds.values()}
        if len(seed_sets) != 1:
            raise ValueError(
                f"Dataset {dataset} does not have the same seeds in every fold: {folds}"
            )
        rendered[dataset] = {
            fold: tuple(sorted(seeds)) for fold, seeds in sorted(folds.items())
        }
    return rendered


def _cell_point(pairs: dict[str, np.ndarray]) -> float:
    values = np.concatenate(list(pairs.values()))
    return float(values.mean())


def _cell_bootstrap(pairs: dict[str, np.ndarray], rng: np.random.Generator) -> float:
    names = tuple(pairs)
    sampled = rng.integers(0, len(names), size=len(names))
    total = 0.0
    count = 0
    for index in sampled:
        values = pairs[names[int(index)]]
        query_draw = rng.integers(0, len(values), size=len(values))
        sampled_values = values[query_draw]
        total += float(sampled_values.sum())
        count += len(sampled_values)
    # Pairs are resampled as dependence clusters; query counts retain the
    # benchmark's original micro-accuracy estimand inside each fold/seed cell.
    return total / count


def _point_estimates(
    cells: dict[CellKey, dict[str, np.ndarray]],
    design: dict[str, dict[int, tuple[int, ...]]],
) -> tuple[dict[str, float], float, dict[str, float]]:
    macro: dict[str, float] = {}
    pooled: dict[str, float] = {}
    for dataset, folds in design.items():
        cell_values = [
            _cell_point(cells[(dataset, fold, seed)])
            for fold, seeds in folds.items()
            for seed in seeds
        ]
        macro[dataset] = float(np.mean(cell_values))
        all_values = np.concatenate(
            [
                np.concatenate(list(cells[(dataset, fold, seed)].values()))
                for fold, seeds in folds.items()
                for seed in seeds
            ]
        )
        pooled[dataset] = float(all_values.mean())
    return macro, float(np.mean(list(macro.values()))), pooled


def _one_replicate(
    cells: dict[CellKey, dict[str, np.ndarray]],
    design: dict[str, dict[int, tuple[int, ...]]],
    rng: np.random.Generator,
) -> tuple[dict[str, float], float]:
    estimates: dict[str, float] = {}
    for dataset, folds in design.items():
        fold_ids = tuple(folds)
        sampled_folds = rng.integers(0, len(fold_ids), size=len(fold_ids))
        fold_values: list[float] = []
        for fold_index in sampled_folds:
            fold = fold_ids[int(fold_index)]
            seeds = folds[fold]
            sampled_seeds = rng.integers(0, len(seeds), size=len(seeds))
            seed_values = [
                _cell_bootstrap(cells[(dataset, fold, seeds[int(seed_index)])], rng)
                for seed_index in sampled_seeds
            ]
            fold_values.append(float(np.mean(seed_values)))
        estimates[dataset] = float(np.mean(fold_values))
    return estimates, float(np.mean(list(estimates.values())))


def _interval(values: np.ndarray, point: float) -> dict[str, float]:
    return {
        "estimate": point,
        "bootstrap_mean": float(values.mean()),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "probability_b_superior": float((values > 0.0).mean()),
        "one_sided_p_b_not_better": float((np.count_nonzero(values <= 0.0) + 1) / (len(values) + 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Joint paired hierarchical bootstrap: dataset > fold > seed > pair > query. "
            "The estimand gives equal weight to folds, seeds and datasets."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260865)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1000:
        parser.error("--iterations must be at least 1000")
    entries = _load_manifest(args.manifest, args.split)
    cells = _read_cells(entries)
    design = _validate_rectangular(cells)
    point_by_dataset, joint_point, pooled = _point_estimates(cells, design)
    rng = np.random.default_rng(args.bootstrap_seed)
    draws = {dataset: np.empty(args.iterations) for dataset in design}
    joint_draws = np.empty(args.iterations)
    for iteration in range(args.iterations):
        estimates, joint = _one_replicate(cells, design, rng)
        for dataset, value in estimates.items():
            draws[dataset][iteration] = value
        joint_draws[iteration] = joint
    result = {
        "estimand": "equal-dataset/equal-fold/equal-seed paired Top-1 delta (B-A)",
        "hierarchy": ["dataset", "fold", "seed", "pair", "query"],
        "split": args.split,
        "iterations": args.iterations,
        "bootstrap_seed": args.bootstrap_seed,
        "design": {
            dataset: {
                "folds": len(folds),
                "seeds": list(next(iter(folds.values()))),
                "cells": sum(len(seeds) for seeds in folds.values()),
            }
            for dataset, folds in design.items()
        },
        "datasets": {
            dataset: {
                **_interval(draws[dataset], point_by_dataset[dataset]),
                "pooled_query_delta": pooled[dataset],
            }
            for dataset in design
        },
        "joint_macro": _interval(joint_draws, joint_point),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
