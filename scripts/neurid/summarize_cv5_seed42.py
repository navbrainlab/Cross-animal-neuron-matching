#!/usr/bin/env python3
"""Summarize exactly five held-out fold metrics from the formal seed-42 run."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


FOLDS = tuple(range(5))
METRICS = ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy")


def read_cell(path: Path, fold: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing fold-{fold} metrics: {path}")
    cell = json.loads(path.read_text(encoding="utf-8"))
    expected = {"fold": fold, "seed": 42, "split": "test"}
    mismatches = {key: (cell.get(key), value) for key, value in expected.items() if cell.get(key) != value}
    if mismatches:
        raise RuntimeError(f"non-formal fold metadata in {path}: {mismatches}")
    missing = [metric for metric in METRICS if metric not in cell]
    if missing:
        raise RuntimeError(f"missing metrics in {path}: {missing}")
    return cell


def summarize(cells: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "protocol": "five biological folds x single model seed 42",
        "folds": list(FOLDS),
        "seed": 42,
        "aggregation": "unweighted mean and sample SD over five fold-level values",
        "fold_cells": cells,
        "metrics": {},
    }
    for metric in METRICS:
        values = [float(cell[metric]) for cell in cells]
        summary["metrics"][metric] = {
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "fold_values": values,
        }
    return summary


def markdown(summary: dict[str, Any]) -> str:
    labels = {
        "top1_real": "Top-1",
        "top5_real": "Top-5",
        "mrr_real": "MRR",
        "hungarian_accuracy": "Hungarian Accuracy",
    }
    lines = [
        "# Formal CV5 × seed42 summary",
        "",
        "Unweighted mean ± sample SD over exactly five held-out biological folds.",
        "",
        "| Metric | Mean ± SD |",
        "|---|---:|",
    ]
    for metric in METRICS:
        result = summary["metrics"][metric]
        lines.append(
            f"| {labels[metric]} | {100.0 * result['mean']:.2f} ± "
            f"{100.0 * result['sample_sd']:.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    cells = [read_cell(args.run_root / f"fold{fold}" / "test_metrics.json", fold) for fold in FOLDS]
    result = summarize(cells)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown(result), encoding="utf-8")
    print(markdown(result), end="")


if __name__ == "__main__":
    main()
