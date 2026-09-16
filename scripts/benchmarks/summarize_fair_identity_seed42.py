#!/usr/bin/env python3
"""Build the five-fold, seed-42 fair-identity benchmark tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


CORE_METHOD_ORDER = ("CPD", "fDNC", "NuCLR", "GeoTransformer")
ATANAS_EXTRA_METHOD_ORDER = ("NGM-v2", "Vanilla FGW")
METHOD_ORDER = (*CORE_METHOD_ORDER, *ATANAS_EXTRA_METHOD_ORDER, "Ours")
METRICS = ("top1", "top5", "mrr", "hungarian")


def read_rows(repo: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline_path = repo / "runs/fair_identity_medoid_template_v1/cv5x3_summary/cells.csv"
    ours_path = repo / "runs/fair_identity_retest_v1/ours/cells.csv"

    with baseline_path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            if int(source["seed"]) != 42:
                continue
            rows.append(
                {
                    "dataset": source["dataset"],
                    "method": source["method"],
                    "fold": int(source["fold"]),
                    "seed": 42,
                    "mode": "",
                    "top1": float(source["top1"]),
                    "top5": float(source["top5"]),
                    "mrr": float(source["mrr"]),
                    "hungarian": float(source["hungarian"]),
                    "queries": int(source["queries"]),
                    "source_path": str(baseline_path.relative_to(repo)),
                }
            )

    with ours_path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            if int(source["seed"]) != 42 or source["mode"] != "static":
                continue
            rows.append(
                {
                    "dataset": source["dataset"],
                    "method": "Ours",
                    "fold": int(source["fold"]),
                    "seed": 42,
                    "mode": "static",
                    "top1": float(source["top1_real"]),
                    "top5": float(source["top5_real"]),
                    "mrr": float(source["mrr_real"]),
                    "hungarian": float(source["hungarian_accuracy"]),
                    "queries": int(source["queries"]),
                    "source_path": str(ours_path.relative_to(repo)),
                }
            )

    for fold in range(5):
        ngm_path = repo / f"runs/unified_benchmark/ngmv2/atanas/fold{fold}/seed42/test_metrics.json"
        with ngm_path.open() as handle:
            source = json.load(handle)
        rows.append(
            {
                "dataset": "atanas", "method": "NGM-v2", "fold": fold, "seed": 42,
                "mode": "", "top1": float(source["top1"]), "top5": float(source["top5"]),
                "mrr": float(source["mrr"]), "hungarian": float(source["hungarian"]),
                "queries": int(source["queries"]), "source_path": str(ngm_path.relative_to(repo)),
            }
        )

    for fold in range(5):
        fgw_path = repo / f"runs/unified_benchmark/fgw_pot/atanas/fold{fold}/test_metrics.json"
        with fgw_path.open() as handle:
            source = json.load(handle)
        rows.append(
            {
                "dataset": "atanas", "method": "Vanilla FGW", "fold": fold, "seed": 42,
                "mode": "deterministic", "top1": float(source["top1"]),
                "top5": float(source["top5"]), "mrr": float(source["mrr"]),
                "hungarian": float(source["hungarian"]), "queries": int(source["queries"]),
                "source_path": str(fgw_path.relative_to(repo)),
            }
        )

    for fold in range(5):
        ngm_path = repo / f"runs/unified_benchmark/ngmv2/rld/fold{fold}/seed42/test_metrics.json"
        with ngm_path.open() as handle:
            source = json.load(handle)
        rows.append(
            {
                "dataset": "rld", "method": "NGM-v2", "fold": fold, "seed": 42,
                "mode": "", "top1": float(source["top1"]), "top5": float(source["top5"]),
                "mrr": float(source["mrr"]), "hungarian": float(source["hungarian"]),
                "queries": int(source["queries"]), "source_path": str(ngm_path.relative_to(repo)),
            }
        )

    for fold in range(5):
        fgw_path = repo / f"runs/unified_benchmark/fgw_pot/rld/fold{fold}/test_metrics.json"
        with fgw_path.open() as handle:
            source = json.load(handle)
        rows.append(
            {
                "dataset": "rld", "method": "Vanilla FGW", "fold": fold, "seed": 42,
                "mode": "deterministic", "top1": float(source["top1"]),
                "top5": float(source["top5"]), "mrr": float(source["mrr"]),
                "hungarian": float(source["hungarian"]), "queries": int(source["queries"]),
                "source_path": str(fgw_path.relative_to(repo)),
            }
        )
    return rows


def validate(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(row)
    expected = {(dataset, method) for dataset in ("atanas", "rld") for method in METHOD_ORDER}
    if set(groups) != expected:
        raise ValueError(f"Unexpected dataset/method groups: {sorted(set(groups) ^ expected)}")
    for key, group in groups.items():
        if len(group) != 5 or len({int(row["fold"]) for row in group}) != 5:
            raise ValueError(f"{key} does not contain five distinct folds")
        if {int(row["seed"]) for row in group} != {42}:
            raise ValueError(f"{key} contains a seed other than 42")


def fmt_pct(mean: float, sd: float) -> str:
    return f"{100 * mean:.2f} ± {100 * sd:.2f}%"


def fmt_unit(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def write_outputs(rows: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["method"]))].append(row)

    detail_fields = (
        "dataset", "method", "fold", "seed", "mode", "top1", "top5", "mrr",
        "hungarian", "queries", "source_path",
    )
    with (output_dir / "benchmark_seed42_fold_cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (str(r["dataset"]), METHOD_ORDER.index(str(r["method"])), int(r["fold"]))))

    summary_fields = ["dataset", "method", "folds", "seed", "aggregation"]
    for metric in METRICS:
        summary_fields.extend((f"{metric}_mean", f"{metric}_sd"))
    summaries: list[dict[str, object]] = []
    for dataset in ("atanas", "rld"):
        for method in METHOD_ORDER:
            group = groups[(dataset, method)]
            summary: dict[str, object] = {
                "dataset": dataset,
                "method": method,
                "folds": 5,
                "seed": 42,
                "aggregation": "unweighted mean and sample SD across 5 biological folds",
            }
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_sd"] = statistics.stdev(values)
            summaries.append(summary)
    with (output_dir / "benchmark_seed42_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    lines = [
        "# Fair-identity benchmark — five folds, seed 42",
        "",
        "Protocol: one result per biological fold; learned methods use seed 42, while deterministic methods are represented once per fold. Values are the unweighted mean ± sample SD across five folds. Ours uses the static Atlas mode used by the main clean benchmark.",
        "",
    ]
    for dataset in ("atanas", "rld"):
        lines.extend(
            [
                f"## {dataset.upper()}",
                "",
                "| Method | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian Acc. ↑ |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for summary in (item for item in summaries if item["dataset"] == dataset):
            bold = "**" if summary["method"] == "Ours" else ""
            method = f"{bold}{summary['method']}{bold}"
            top1 = fmt_pct(float(summary["top1_mean"]), float(summary["top1_sd"]))
            top5 = fmt_pct(float(summary["top5_mean"]), float(summary["top5_sd"]))
            mrr = fmt_unit(float(summary["mrr_mean"]), float(summary["mrr_sd"]))
            hungarian = fmt_pct(float(summary["hungarian_mean"]), float(summary["hungarian_sd"]))
            if bold:
                top1, top5, mrr, hungarian = (f"**{value}**" for value in (top1, top5, mrr, hungarian))
            lines.append(f"| {method} | {top1} | {top5} | {mrr} | {hungarian} |")
        lines.append("")
    lines.extend(
        [
            "## Provenance notes",
            "",
            "- CPD, fDNC, NuCLR, and GeoTransformer are filtered from `runs/fair_identity_medoid_template_v1/cv5x3_summary/cells.csv`.",
            "- NGM-v2 uses its saved seed-42 fold cells under `runs/unified_benchmark/`; Vanilla FGW uses the corresponding deterministic five-fold results.",
            "- Ours is filtered from `runs/fair_identity_retest_v1/ours/cells.csv` with `mode=static`.",
            "- CPD is seed-invariant, so its seed-42 values equal its earlier 5-fold result.",
            "- These CSVs describe the manuscript Atanas/Kato fair-identity benchmark.",
        ]
    )
    (output_dir / "BENCHMARK_SEED42.md").write_text("\n".join(lines) + "\n")

    source_paths = sorted({str(row["source_path"]) for row in rows})
    protocol = {
        "protocol": "fair-identity benchmark; five biological folds; one result per fold",
        "seed": 42,
        "deterministic_methods": ["CPD", "Vanilla FGW"],
        "folds": 5,
        "aggregation": "unweighted mean and sample standard deviation across biological folds",
        "ours_mode": "static",
        "metrics_stored_as": "fractions in CSV; Markdown renders Top-1, Top-5, and Hungarian as percentages",
        "validation": {
            "expected_dataset_method_cells": 14,
            "observed_dataset_method_cells": len(groups),
            "expected_fold_rows": 70,
            "observed_fold_rows": len(rows),
            "all_cells_have_five_distinct_folds": True,
            "all_rows_seed_42": True,
        },
        "sources": [
            {
                "path": path,
                "sha256": hashlib.sha256((output_dir.parents[1] / path).read_bytes()).hexdigest(),
            }
            for path in source_paths
        ],
        "scope_note": "These sources are the manuscript Atanas/Kato fair-identity benchmark.",
    }
    (output_dir / "PROTOCOL.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/fair_identity_seed42_benchmark_v1"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    rows = read_rows(repo)
    validate(rows)
    write_outputs(rows, output_dir)


if __name__ == "__main__":
    main()
