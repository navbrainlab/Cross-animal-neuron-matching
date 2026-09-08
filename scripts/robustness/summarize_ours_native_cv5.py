#!/usr/bin/env python3
"""Create the publication-facing Ours robustness summary on native CV5 queries.

Aggregation is deliberately paired: average perturbation replicates within each
biological fold first, then report the unweighted mean and sample SD across the
five grouped folds.  Clean cells must reproduce the saved seed-42 benchmark
metrics exactly before any table is emitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/results/ours_static_seed42/ours_all_cells.csv"
DEFAULT_OUTPUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_native_cv5_ours"
BENCHMARK_ROOT = ROOT / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld"
FOLDS = tuple(range(5))
METRICS = ("top1", "top5", "hungarian", "coverage", "effective_top1")
KIND_ORDER = {"coord_noise": 0, "activity_noise": 1, "missing": 2, "outlier": 3}
DISPLAY = {
    "coord_noise": "Coordinate noise",
    "activity_noise": "Activity noise",
    "missing": "Missing neurons",
    "outlier": "Distractors",
}


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def benchmark_clean(fold: int) -> dict[str, float]:
    path = BENCHMARK_ROOT / f"fold{fold}/seed42/atlas_identity_test/metrics.json"
    report = json.loads(path.read_text(encoding="utf-8"))["modes"]["static"]
    return {
        "queries": int(report["queries"]),
        "top1": float(report["top1_real"]),
        "top5": float(report["top5_real"]),
        "hungarian": float(report["hungarian_accuracy"]),
    }


def assert_clean(rows: list[dict]) -> list[dict]:
    audit = []
    for fold in FOLDS:
        expected = benchmark_clean(fold)
        cells = [
            row for row in rows
            if int(row["fold"]) == fold and math.isclose(float(row["severity"]), 0.0)
        ]
        if not cells:
            raise RuntimeError(f"fold{fold}: no severity-zero cells")
        for row in cells:
            differences = {}
            for metric in ("queries", "top1", "top5", "hungarian"):
                got = int(row[metric]) if metric == "queries" else float(row[metric])
                want = expected[metric]
                if got != want:
                    differences[metric] = {"observed": got, "benchmark": want}
            if differences:
                raise RuntimeError(
                    f"fold{fold} {row['kind']} severity=0 does not reproduce benchmark: {differences}"
                )
        audit.append({"fold": fold, **expected, "passed": True})
    return audit


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple[str, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["kind"], float(row["severity"]), int(row["fold"]))].append(row)

    fold_rows = []
    for (kind, severity, fold), cells in sorted(
        grouped.items(), key=lambda item: (KIND_ORDER.get(item[0][0], 99), item[0][1], item[0][2])
    ):
        item = {
            "kind": kind,
            "severity": severity,
            "fold": fold,
            "perturbation_replicates": len(cells),
        }
        for metric in METRICS:
            item[metric] = float(np.mean([float(cell[metric]) for cell in cells]))
        fold_rows.append(item)

    summary = []
    keys = sorted(
        {(row["kind"], row["severity"]) for row in fold_rows},
        key=lambda key: (KIND_ORDER.get(key[0], 99), key[1]),
    )
    for kind, severity in keys:
        cells = [row for row in fold_rows if row["kind"] == kind and row["severity"] == severity]
        observed_folds = sorted(int(row["fold"]) for row in cells)
        if observed_folds != list(FOLDS):
            raise RuntimeError(f"{kind} severity={severity}: folds={observed_folds}")
        item = {"kind": kind, "severity": severity, "folds": len(cells)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in cells], dtype=np.float64)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1))
        summary.append(item)
    return fold_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(mean: float, sd: float) -> str:
    return f"{100 * mean:.2f} ± {100 * sd:.2f}%"


def render_markdown(summary: list[dict], missing_kinds: list[str]) -> str:
    clean_rows = [row for row in summary if math.isclose(float(row["severity"]), 0.0)]
    if not clean_rows:
        raise RuntimeError("No clean row available for the formal table")
    clean = clean_rows[0]
    lines = [
        "# Ours robustness — native grouped CV5 × seed42",
        "",
        "Each perturbation replicate is averaged within fold; values below are the unweighted mean ± sample SD across the same five biological folds as the Main Benchmark. No shared-cohort rescoring is used.",
        "",
        "| Perturbation | Severity | Top-1 | Top-5 | Hungarian | Coverage | Effective Top-1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        "| Clean | 0 | {top1} | {top5} | {hungarian} | {coverage} | {effective} |".format(
            top1=percent(clean["top1_mean"], clean["top1_sd"]),
            top5=percent(clean["top5_mean"], clean["top5_sd"]),
            hungarian=percent(clean["hungarian_mean"], clean["hungarian_sd"]),
            coverage=percent(clean["coverage_mean"], clean["coverage_sd"]),
            effective=percent(clean["effective_top1_mean"], clean["effective_top1_sd"]),
        ),
    ]
    for row in summary:
        if math.isclose(float(row["severity"]), 0.0):
            continue
        lines.append(
            "| {name} | {severity:g} | {top1} | {top5} | {hungarian} | {coverage} | {effective} |".format(
                name=DISPLAY.get(row["kind"], row["kind"]),
                severity=row["severity"],
                top1=percent(row["top1_mean"], row["top1_sd"]),
                top5=percent(row["top5_mean"], row["top5_sd"]),
                hungarian=percent(row["hungarian_mean"], row["hungarian_sd"]),
                coverage=percent(row["coverage_mean"], row["coverage_sd"]),
                effective=percent(row["effective_top1_mean"], row["effective_top1_sd"]),
            )
        )
    lines += [
        "",
        "Clean is represented by severity=0 and reproduces the Main Benchmark: **Top-1 63.93 ± 4.61%, Top-5 79.81 ± 5.01%, Hungarian 61.78 ± 5.18%**.",
        "",
    ]
    if missing_kinds:
        lines.append(
            "Not yet reported: " + ", ".join(DISPLAY.get(kind, kind) for kind in missing_kinds)
            + ". No native-CV5 condition outputs exist for these perturbations, so values are not imputed from the old fixed split or shared cohort."
        )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    rows = read_rows(args.input)
    clean_audit = assert_clean(rows)
    fold_rows, summary = aggregate(rows)
    present = {row["kind"] for row in summary}
    required = ["coord_noise", "activity_noise", "missing", "outlier"]
    missing = [kind for kind in required if kind not in present]

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "fold_level.csv", fold_rows)
    write_csv(args.output / "summary.csv", summary)
    (args.output / "TABLE.md").write_text(render_markdown(summary, missing), encoding="utf-8")
    audit = {
        "protocol": "RLD grouped CV5 x seed42; native held-out evaluation",
        "aggregation": "mean perturbation replicates within fold, then unweighted mean and sample SD across 5 folds",
        "shared_cohort_rescoring": False,
        "severity_zero_benchmark_gate": clean_audit,
        "present_kinds": sorted(present),
        "missing_kinds": missing,
        "complete": not missing,
    }
    (args.output / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(args.output / "TABLE.md")
    print(json.dumps({"clean_gate": "passed", "missing_kinds": missing}, indent=2))


if __name__ == "__main__":
    main()
