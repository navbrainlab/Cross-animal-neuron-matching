#!/usr/bin/env python3
"""Recompute the canonical main table from a portable audit package.

No repository checkout, raw NPZ, checkpoint, training, or inference is needed.
Top-1/Top-5 are summed from the canonical-complete prediction table.  CPD and
fDNC Hungarian numerators are recovered from their saved fold metrics because
their compact native CSVs did not retain per-query Hungarian assignments.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_hungarian_correct(package: Path, method: str, dataset: str, fold: int) -> int:
    directory = method.lower().replace("-", "_").replace(" ", "_")
    path = package / f"metrics/{directory}/{dataset}/fold{fold}/metrics.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    metric = report["metrics"]["template_score"]
    return round(float(metric["hungarian_accuracy"]) * int(metric["queries"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path, help="Unpacked canonical_main_table_audit_* directory")
    parser.add_argument("--fold-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    package = args.package.resolve()
    rows = read_csv(package / "predictions/canonical_complete_predictions.csv")
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["method"], int(row["fold"]))].append(row)

    folds: list[dict[str, Any]] = []
    for (dataset, method, fold), items in sorted(grouped.items()):
        q = len(items)
        top1_correct = sum(float(row["top1_correct"]) for row in items)
        top5_correct = sum(float(row["top5_correct"]) for row in items)
        if method in {"CPD", "fDNC"}:
            hungarian_correct = aggregate_hungarian_correct(package, method, dataset, fold)
            hungarian_source = "saved native fold numerator"
        else:
            values = [row["hungarian_correct"] for row in items]
            if any(value == "" for value in values):
                raise RuntimeError(f"Missing Hungarian correctness for {method}/{dataset}/fold{fold}")
            hungarian_correct = sum(float(value) for value in values)
            hungarian_source = "canonical-complete per-query rows"
        folds.append({
            "dataset": dataset, "method": method, "fold": fold, "queries": q,
            "top1_correct": top1_correct, "top5_correct": top5_correct,
            "hungarian_correct": hungarian_correct, "top1": top1_correct / q,
            "top5": top5_correct / q, "hungarian": hungarian_correct / q,
            "hungarian_source": hungarian_source,
        })

    summaries: list[dict[str, Any]] = []
    by_method: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in folds:
        by_method[(row["dataset"], row["method"])].append(row)
    for (dataset, method), items in sorted(by_method.items()):
        if len(items) != 5:
            raise RuntimeError(f"Expected five folds for {method}/{dataset}; got {len(items)}")
        output: dict[str, Any] = {"dataset": dataset, "method": method, "folds": 5}
        for metric in ("top1", "top5", "hungarian"):
            values = [float(item[metric]) for item in items]
            output[f"{metric}_mean"] = statistics.mean(values)
            output[f"{metric}_sample_sd"] = statistics.stdev(values)
            output[f"{metric}_display"] = f"{100 * statistics.mean(values):.2f} ± {100 * statistics.stdev(values):.2f}%"
        summaries.append(output)

    saved = read_csv(package / "main_table/unified_fold_metrics.csv")
    expected = {
        (row["dataset"], row["method"], int(row["fold"])): row
        for row in saved if row["status"].startswith("PASS")
    }
    failures: list[str] = []
    for row in folds:
        key = (row["dataset"], row["method"], row["fold"])
        if key not in expected:
            failures.append(f"missing saved fold row: {key}")
            continue
        for metric in ("top1", "top5", "hungarian"):
            if abs(float(row[metric]) - float(expected[key][metric])) > 1e-12:
                failures.append(f"{key} {metric}: recomputed={row[metric]} saved={expected[key][metric]}")
    if failures:
        raise RuntimeError("Recompute mismatch:\n" + "\n".join(failures))

    fold_fields = list(folds[0])
    summary_fields = list(summaries[0])
    if args.fold_output:
        write_csv(args.fold_output, folds, fold_fields)
    if args.summary_output:
        write_csv(args.summary_output, summaries, summary_fields)
    print(json.dumps({
        "status": "PASS", "fold_rows": len(folds), "summary_rows": len(summaries),
        "message": "All recomputed fold Top-1, Top-5, and Hungarian values match the saved main-table fold metrics.",
        "summaries": summaries,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
