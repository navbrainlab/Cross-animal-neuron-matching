#!/usr/bin/env python3
"""Merge locked CV5x3 component-ablation test results into paper-ready files."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
MAIN = RUNS / "mprt_v1_1_component_ablation_cv5x3_v1" / "test_summary.json"
ACT_ATANAS = RUNS / "mprt_v1_1_activity_only_atanas_cv5x3_v1" / "test_summary.json"
ACT_RLD = RUNS / "mprt_v1_1_activity_only_rld_cv5x3_v1" / "test_summary.json"
OUTPUT = RUNS / "mprt_v1_1_component_ablation_test_cv5x3_final"
FOLDS = set(range(5))
SEEDS = {1, 42, 123}
VARIANTS = ("full", "geometry_only", "node_only", "no_transport", "activity_only")
DISPLAY = {
    "full": "Full NeuRID",
    "geometry_only": "w/o Activity",
    "node_only": "Node-only",
    "no_transport": "w/o Relation Transport",
    "activity_only": "w/o Geometry (Activity-only)",
}
METRICS = ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy")


def read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def key(row: dict) -> tuple[str, int, int, str]:
    return row["dataset"], int(row["fold"]), int(row["seed"]), row["variant"]


def main() -> None:
    sources = [MAIN, ACT_ATANAS, ACT_RLD]
    payloads = [read(path) for path in sources]
    if any(item.get("split") != "test" for item in payloads):
        raise RuntimeError("All inputs must be locked-test summaries")

    main_rows = payloads[0]["rows"]
    activity_rows = payloads[1]["rows"] + payloads[2]["rows"]
    main_full = {key(row): row for row in main_rows if row["variant"] == "full"}
    activity_full = {key(row): row for row in activity_rows if row["variant"] == "full"}
    if main_full.keys() != activity_full.keys():
        raise RuntimeError("Full cell sets differ between locked-test runs")
    for cell in main_full:
        for metric in METRICS:
            if main_full[cell][metric] != activity_full[cell][metric]:
                raise RuntimeError(f"Full mismatch at {cell}: {metric}")

    rows = list(main_rows)
    rows.extend(row for row in activity_rows if row["variant"] == "activity_only")
    rows.sort(key=lambda row: (row["dataset"], VARIANTS.index(row["variant"]), row["fold"], row["seed"]))

    expected = {(dataset, fold, seed, variant) for dataset in ("atanas", "rld") for fold in FOLDS for seed in SEEDS for variant in VARIANTS}
    observed = {key(row) for row in rows}
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError(f"Incomplete or duplicate cells: expected={len(expected)} observed={len(observed)} rows={len(rows)}")

    summary_rows = []
    for dataset in ("atanas", "rld"):
        for variant in VARIANTS:
            selected = [row for row in rows if row["dataset"] == dataset and row["variant"] == variant]
            item = {"dataset": dataset, "variant": variant, "display": DISPLAY[variant], "n_cells": len(selected)}
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_sample_sd"] = statistics.stdev(values)
            summary_rows.append(item)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell_fields = ["dataset", "fold", "seed", "split", "variant", "display", "queries", *METRICS]
    with (OUTPUT / "fold_seed_test_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = ["dataset", "variant", "display", "n_cells", *[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sample_sd")]]
    with (OUTPUT / "component_ablation_test_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# NeuRID component ablation — locked test, 5 folds × 3 seeds",
        "",
        "Values are the unweighted mean ± sample SD across 15 fold-seed cells (folds 0–4; seeds 1, 42, 123). Checkpoints were selected using validation data before the held-out test split was read.",
    ]
    for dataset in ("atanas", "rld"):
        lines.extend(["", f"## {'Atanas' if dataset == 'atanas' else 'RLD (Kato)'}", "", "| Variant | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian Acc. ↑ |", "| --- | ---: | ---: | ---: | ---: |"])
        for row in [item for item in summary_rows if item["dataset"] == dataset]:
            lines.append(
                f"| {row['display']} | {100*row['top1_real_mean']:.2f} ± {100*row['top1_real_sample_sd']:.2f}% "
                f"| {100*row['top5_real_mean']:.2f} ± {100*row['top5_real_sample_sd']:.2f}% "
                f"| {row['mrr_real_mean']:.4f} ± {row['mrr_real_sample_sd']:.4f} "
                f"| {100*row['hungarian_accuracy_mean']:.2f} ± {100*row['hungarian_accuracy_sample_sd']:.2f}% |"
            )
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    provenance = {
        "protocol": "grouped outer CV5x3 locked test",
        "folds": [0, 1, 2, 3, 4],
        "seeds": [1, 42, 123],
        "aggregation": "unweighted mean and sample SD across 15 fold-seed cells",
        "test_policy": "held-out test evaluated only after checkpoint hash lock",
        "evaluation_device": "cpu (CUDA unavailable); model/checkpoint/evaluator unchanged",
        "validation": {
            "expected_cells": 150,
            "observed_cells": len(rows),
            "each_dataset_variant_has_15_cells": all(row["n_cells"] == 15 for row in summary_rows),
            "full_rows_identical_across_source_runs": True,
        },
        "sources": [{"path": str(path.relative_to(REPO)), "sha256": sha256(path)} for path in sources],
        "checkpoint_locks": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256(path)}
            for path in (
                RUNS / "mprt_v1_1_component_ablation_cv5x3_v1" / "LOCKED_CHECKPOINTS.json",
                RUNS / "mprt_v1_1_activity_only_atanas_cv5x3_v1" / "LOCKED_CHECKPOINTS.json",
                RUNS / "mprt_v1_1_activity_only_rld_cv5x3_v1" / "LOCKED_CHECKPOINTS.json",
            )
        ],
    }
    (OUTPUT / "PROTOCOL.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
