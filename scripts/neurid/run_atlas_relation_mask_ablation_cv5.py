#!/usr/bin/env python3
"""Run the locked CV5 zero-fill versus masked-atlas-relation ablation.

This is an inference-only paired intervention.  Every checkpoint, query,
atlas prototype, transport hyperparameter and evaluation cohort is held fixed;
the masked arm only enables ``atlas_relation_masking``.  Never-co-observed
atlas pairs are identified by the training-only support matrix stored in each
dynamic checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO / "mprt_net_v1_1"
EVALUATOR = REPO / "scripts/mprt/evaluate_mprt_static_atlas_fixed_cohort.py"
CHECKPOINT_ROOT = REPO / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
DATA_ROOTS = {
    "atanas": REPO / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": REPO / "Data/Dunn_001623/cv5_grouped_v1",
}
HANDLINGS = ("zero_fill", "mask_missing")
METRICS = ("covered_only_top1", "covered_only_top5", "covered_only_mrr")


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def current_result(dataset: str, fold: int, seed: int) -> dict[str, Any]:
    path = (
        CHECKPOINT_ROOT
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
        / "atlas_identity_test/metrics.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["modes"]["static"]


def run_cell(
    *,
    dataset: str,
    fold: int,
    seed: int,
    handling: str,
    output_root: Path,
    device: str,
    activity_length: int,
    force: bool,
) -> dict[str, Any]:
    cell = output_root / dataset / f"fold{fold}" / f"seed{seed}" / handling
    result_path = cell / "metrics.json"
    query_path = cell / "queries.csv"
    if not force and result_path.is_file() and query_path.is_file():
        print(f"reuse {result_path.relative_to(REPO)}", flush=True)
        return json.loads(result_path.read_text(encoding="utf-8"))

    cell.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        CHECKPOINT_ROOT
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
        / "dynamic/low_rank_r8/best.pt"
    )
    command = [
        sys.executable,
        str(EVALUATOR),
        "--package-root", str(PACKAGE_ROOT),
        "--dataset-root", str(DATA_ROOTS[dataset] / f"fold_{fold}"),
        "--split", "test",
        "--checkpoint", str(checkpoint),
        "--activity-length", str(activity_length),
        "--device", device,
        "--dataset", dataset,
        "--fold", str(fold),
        "--seed", str(seed),
        "--variant", handling,
        "--atlas-relation-handling", handling,
        "--output", str(result_path),
        "--query-output", str(query_path),
    ]
    print(f"run {dataset} fold={fold} seed={seed} handling={handling}", flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    return json.loads(result_path.read_text(encoding="utf-8"))


def validate_zero_fill(
    dataset: str, fold: int, seed: int, result: dict[str, Any]
) -> None:
    expected = current_result(dataset, fold, seed)
    mappings = {
        "covered_queries": "queries",
        "covered_only_top1": "top1_real",
        "covered_only_top5": "top5_real",
        "covered_only_mrr": "mrr_real",
    }
    for actual_key, expected_key in mappings.items():
        actual = float(result[actual_key])
        reference = float(expected[expected_key])
        if abs(actual - reference) > 1e-6:
            raise RuntimeError(
                f"zero-fill reproduction failed for {dataset}/fold{fold}/seed{seed} "
                f"{actual_key}: {actual} != {reference}"
            )


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        for handling in HANDLINGS:
            selected = [
                row for row in rows
                if row["dataset"] == dataset and row["handling"] == handling
            ]
            summary: dict[str, Any] = {
                "dataset": dataset,
                "atlas_relation_handling": handling,
                "fold_seed_cells": len(selected),
                "folds": len({int(row["fold"]) for row in selected}),
                "seeds": len({int(row["seed"]) for row in selected}),
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_sd"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            output.append(summary)
    return output


def render_readme(
    summaries: list[dict[str, Any]], coverage: list[dict[str, Any]]
) -> str:
    by_key = {
        (row["dataset"], row["atlas_relation_handling"]): row
        for row in summaries
    }
    lines = [
        "# Atlas relation support ablation",
        "",
        "Locked seed-42 CV5 inference-only comparison. Values are the unweighted "
        "mean ± sample SD across five folds on the existing training-atlas-identity "
        "query universe. No model was retrained or selected for this intervention.",
        "",
        "| Atlas relation handling | Atanas Top-1 | Kato/RLD Top-1 |",
        "| --- | ---: | ---: |",
    ]
    for handling, label in (("zero_fill", "Zero fill"), ("mask_missing", "Mask missing relations")):
        values = []
        for dataset in ("atanas", "rld"):
            row = by_key[(dataset, handling)]
            values.append(
                f"{100 * row['covered_only_top1_mean']:.2f} ± "
                f"{100 * row['covered_only_top1_sd']:.2f}%"
            )
        lines.append(f"| {label} | {values[0]} | {values[1]} |")
    lines.extend([
        "",
        "The mask is `M[j,l] = 1[pair_count[j,l] > 0]`; the discrepancy is "
        "divided by supported transport mass. Ordered relation entries, including "
        "the diagonal, are counted because the learned relation field is directed.",
        "",
        "| Dataset | Fold | Unobserved | All | Unobserved fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in coverage:
        name = "Atanas" if row["dataset"] == "atanas" else "Kato/RLD"
        lines.append(
            f"| {name} | {row['fold']} | {row['unobserved_pairs']} | "
            f"{row['all_pairs']} | {100 * row['unobserved_fraction']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="atanas,rld")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "runs/atlas_relation_mask_ablation_cv5_seed42_v1",
    )
    args = parser.parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    folds = parse_ints(args.folds)
    seeds = parse_ints(args.seeds)
    if not datasets or any(dataset not in DATA_ROOTS for dataset in datasets):
        raise ValueError(f"Datasets must be selected from {sorted(DATA_ROOTS)}")
    if not folds or any(fold not in range(5) for fold in folds):
        raise ValueError("Folds must be selected from 0,1,2,3,4")
    if not seeds:
        raise ValueError("At least one seed is required")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    coverage_by_fold: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in datasets:
        for fold in folds:
            for seed in seeds:
                for handling in HANDLINGS:
                    result = run_cell(
                        dataset=dataset,
                        fold=fold,
                        seed=seed,
                        handling=handling,
                        output_root=output_root,
                        device=args.device,
                        activity_length=args.activity_length,
                        force=args.force,
                    )
                    if handling == "zero_fill":
                        validate_zero_fill(dataset, fold, seed, result)
                    row = {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": seed,
                        "handling": handling,
                        **{metric: result[metric] for metric in METRICS},
                        "all_query_top1": result["top1_real"],
                        "queries": result["queries"],
                        "covered_queries": result["covered_queries"],
                    }
                    rows.append(row)
                    total = int(result["atlas_relation_total_pairs"])
                    supported = int(result["atlas_relation_supported_pairs"])
                    coverage_by_fold[(dataset, fold)] = {
                        "dataset": dataset,
                        "fold": fold,
                        "unobserved_pairs": total - supported,
                        "all_pairs": total,
                        "unobserved_fraction": (total - supported) / total,
                    }

    summaries = summarize(rows)
    coverage = [coverage_by_fold[key] for key in sorted(coverage_by_fold)]
    atomic_csv(output_root / "fold_seed_metrics.csv", rows)
    atomic_csv(output_root / "summary.csv", summaries)
    atomic_csv(output_root / "atlas_relation_coverage.csv", coverage)
    audit = {
        "protocol": "locked_static_atlas_relation_mask_ablation_cv5_v1",
        "intervention": "zero_fill versus supported-mass-normalized mask_missing",
        "checkpoint_policy": "existing locked checkpoint; no retraining or reselection",
        "zero_fill_reproduction_guard": "passed",
        "datasets": datasets,
        "folds": folds,
        "seeds": seeds,
        "primary_query_universe": "unique supervised test identities present in outer-train atlas vocabulary",
        "unobserved_pair_denominator": "all ordered atlas relation entries including diagonal",
        "summaries": summaries,
    }
    atomic_json(output_root / "AUDIT.json", audit)
    readme = render_readme(summaries, coverage)
    temporary = (output_root / "README.md.tmp")
    temporary.write_text(readme, encoding="utf-8")
    temporary.replace(output_root / "README.md")
    print(readme)


if __name__ == "__main__":
    main()
