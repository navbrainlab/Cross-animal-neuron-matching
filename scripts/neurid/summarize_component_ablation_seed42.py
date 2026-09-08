#!/usr/bin/env python3
"""Create the formal CV5 x seed42 component-ablation summary.

The source table contains the locked held-out-test cells for three seeds.  This
script selects, rather than recomputes, the five canonical seed42 fold cells.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SOURCE = (
    REPO
    / "runs/mprt_v1_1_component_ablation_test_cv5x3_final"
    / "fold_seed_test_cells.csv"
)
OUTPUT = REPO / "runs/mprt_v1_1_component_ablation_test_cv5_seed42_final"
DATASETS = ("atanas", "rld")
FOLDS = set(range(5))
SEED = 42
VARIANTS = ("full", "geometry_only", "node_only", "no_transport", "activity_only")
DISPLAY = {
    "full": "Full NeuRID",
    "geometry_only": "w/o Activity",
    "node_only": "w/o Population Relations (Node-only)",
    "no_transport": "w/o Relation Transport",
    "activity_only": "w/o Geometry (Activity-only)",
}
METRICS = ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    rows = [row for row in source_rows if int(row["seed"]) == SEED]
    expected = {
        (dataset, fold, variant)
        for dataset in DATASETS
        for fold in FOLDS
        for variant in VARIANTS
    }
    observed = {
        (row["dataset"], int(row["fold"]), row["variant"])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise RuntimeError(
            f"Incomplete/duplicate seed42 cells: expected={len(expected)} "
            f"observed={len(observed)} rows={len(rows)}"
        )
    if any(row["split"] != "test" for row in rows):
        raise RuntimeError("Formal ablation summary must use held-out test cells")

    rows.sort(
        key=lambda row: (
            DATASETS.index(row["dataset"]),
            VARIANTS.index(row["variant"]),
            int(row["fold"]),
        )
    )
    for row in rows:
        row["display"] = DISPLAY[row["variant"]]

    summary_rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            selected = [
                row
                for row in rows
                if row["dataset"] == dataset and row["variant"] == variant
            ]
            item = {
                "dataset": dataset,
                "variant": variant,
                "display": DISPLAY[variant],
                "folds": len(selected),
                "seed": SEED,
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in selected]
                item[f"{metric}_mean"] = statistics.mean(values)
                item[f"{metric}_sample_sd"] = statistics.stdev(values)
                item[f"{metric}_fold_values"] = json.dumps(values)
            summary_rows.append(item)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cell_fields = [
        "dataset", "fold", "seed", "split", "variant", "display", "queries", *METRICS
    ]
    with (OUTPUT / "fold_test_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_fields = ["dataset", "variant", "display", "folds", "seed"]
    for metric in METRICS:
        summary_fields.extend(
            (f"{metric}_mean", f"{metric}_sample_sd", f"{metric}_fold_values")
        )
    with (OUTPUT / "component_ablation_seed42_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    lines = [
        "# NeuRID component ablation — CV5 × seed42",
        "",
        "Values are the unweighted mean ± sample SD across the five canonical biological folds. "
        "All rows use seed42 and locked held-out-test evaluation.",
    ]
    for dataset in DATASETS:
        title = "Atanas" if dataset == "atanas" else "RLD (Kato)"
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Variant | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian Acc. ↑ |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in (item for item in summary_rows if item["dataset"] == dataset):
            lines.append(
                f"| {row['display']} "
                f"| {100 * row['top1_real_mean']:.2f} ± {100 * row['top1_real_sample_sd']:.2f}% "
                f"| {100 * row['top5_real_mean']:.2f} ± {100 * row['top5_real_sample_sd']:.2f}% "
                f"| {row['mrr_real_mean']:.4f} ± {row['mrr_real_sample_sd']:.4f} "
                f"| {100 * row['hungarian_accuracy_mean']:.2f} ± "
                f"{100 * row['hungarian_accuracy_sample_sd']:.2f}% |"
            )
    (OUTPUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    protocol = {
        "protocol_id": "component_ablation_cv5_seed42_v1",
        "folds": list(range(5)),
        "seed": SEED,
        "split": "held-out test",
        "aggregation": "unweighted mean and sample SD across five biological folds",
        "canonical_cv_roots": {
            "atanas": "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
            "rld": "Data/Dunn_001623/cv5_grouped_v1",
        },
        "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": sha256(SOURCE),
        "selection": "exactly the seed42 rows from the locked CV5x3 cells; no metric recomputation",
        "legacy_cv5x3_policy": "retained only as a supplementary stability check",
        "validation": {
            "expected_cells": 50,
            "observed_cells": len(rows),
            "each_dataset_variant_has_five_folds": all(
                row["folds"] == 5 for row in summary_rows
            ),
        },
    }
    (OUTPUT / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
