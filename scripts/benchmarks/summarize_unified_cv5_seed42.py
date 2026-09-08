#!/usr/bin/env python3
"""Summarize only five-fold method groups that passed the unified audit."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs/unified_main_benchmark_cv5_seed42_v1"
METRICS = ("top1", "top5", "mrr", "hungarian")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def extract(row: dict[str, str]) -> dict[str, Any]:
    path = REPO / row["metrics_path"]
    data = read_json(path)
    method = row["method"]
    if method in {"CPD", "fDNC"}:
        values = data["metrics"]["template_score"]
        return {
            "queries": int(values["queries"]),
            "top1": float(values["top1"]),
            "top5": float(values["top5"]),
            "mrr": float(values["mrr"]),
            "hungarian": float(values["hungarian_accuracy"]),
        }
    if method == "NuCLR":
        values = data["metrics"]
        return {
            "queries": int(values["queries"]),
            "top1": float(values["ranking_top1"]),
            "top5": float(values["top5"]),
            "mrr": float(values["mrr"]),
            "hungarian": float(values["assignment_top1"]),
        }
    if method == "Ours":
        values = data["modes"]["static"]
        return {
            "queries": int(values["queries"]),
            "top1": float(values["top1_real"]),
            "top5": float(values["top5_real"]),
            "mrr": float(values["mrr_real"]),
            "hungarian": float(values["hungarian_accuracy"]),
        }
    if method in {"Vanilla FGW", "NGM-v2"}:
        return {
            "queries": int(data["queries"]),
            "top1": float(data["top1"]),
            "top5": float(data["top5"]),
            "mrr": float(data["mrr"]),
            "hungarian": float(data["hungarian"]),
        }
    if method == "RGM":
        values = data["test"]
        return {
            "queries": int(values["queries"]),
            "top1": float(values["top1"]),
            "top5": float(values["top5"]),
            "mrr": float(values["mrr"]),
            "hungarian": float(values["hungarian"]),
        }
    if method == "GeoTransformer":
        values = data["test"]
        return {
            "queries": int(values["queries"]),
            "top1": float(values["top1"]),
            "top5": float(values["top5"]),
            "mrr": float(values["mrr"]),
            "hungarian": float(values["hungarian_accuracy"]),
        }
    if method in {"StatAtlas", "CRF_ID"}:
        values = data["metrics"]
        return {
            "queries": int(values["queries"]),
            "top1": float(values["top1"]),
            "top5": float(values["top5"]),
            "mrr": float(values["mrr"]),
            "hungarian": float(values["hungarian"]),
        }
    raise ValueError(f"no verified metric adapter for {method}")


def percent(mean: float, sd: float) -> str:
    return f"{100 * mean:.2f} ± {100 * sd:.2f}%"


def main() -> None:
    audit_path = OUT / "artifact_audit.csv"
    with audit_path.open(newline="", encoding="utf-8") as handle:
        audits = list(csv.DictReader(handle))
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in audits:
        groups[(row["dataset"], row["method"])].append(row)

    cells: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for (dataset, method), rows in sorted(groups.items()):
        rows.sort(key=lambda item: int(item["fold"]))
        if len(rows) != 5 or any(item["status"] != "PASS" for item in rows):
            excluded.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "reason": "requires exactly five PASS cells",
                    "statuses": [item["status"] for item in rows],
                }
            )
            continue
        for row in rows:
            cells.append({**{key: row[key] for key in ("dataset", "method", "fold", "seed", "metrics_path")}, **extract(row)})

    cell_fields = ("dataset", "method", "fold", "seed", "queries", "top1", "top5", "mrr", "hungarian", "metrics_path")
    with (OUT / "verified_fold_cells.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cell_fields)
        writer.writeheader()
        writer.writerows(cells)

    verified_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        verified_groups[(row["dataset"], row["method"])].append(row)
    summaries: list[dict[str, Any]] = []
    for (dataset, method), rows in sorted(verified_groups.items()):
        summary: dict[str, Any] = {"dataset": dataset, "method": method, "folds": 5, "seed": rows[0]["seed"]}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
        summaries.append(summary)
    summary_fields = ("dataset", "method", "folds", "seed", *(f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean", "sd")))
    with (OUT / "verified_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    (OUT / "excluded_groups.json").write_text(json.dumps(excluded, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Unified CV5 × seed42 benchmark — verified cells only",
        "",
        "This is an interim fail-closed table, not yet the complete Main Benchmark. A method appears only after all five folds pass `artifact_audit.csv`.",
        "",
    ]
    for dataset in ("atanas", "rld"):
        lines.extend([f"## {dataset.upper()}", "", "| Method | Top-1 | Top-5 | Hungarian Accuracy |", "| --- | ---: | ---: | ---: |"])
        for row in (item for item in summaries if item["dataset"] == dataset):
            lines.append(
                f"| {row['method']} | {percent(row['top1_mean'], row['top1_sd'])} | "
                f"{percent(row['top5_mean'], row['top5_sd'])} | "
                f"{percent(row['hungarian_mean'], row['hungarian_sd'])} |"
            )
        lines.append("")
    lines.extend(["## Excluded pending rerun/provenance", ""])
    for row in excluded:
        lines.append(f"- {row['dataset']} / {row['method']}: {', '.join(row['statuses'])}")
    (OUT / "VERIFIED_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verified_groups": len(summaries), "excluded_groups": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
