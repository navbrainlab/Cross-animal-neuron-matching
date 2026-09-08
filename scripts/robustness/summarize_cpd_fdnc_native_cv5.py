#!/usr/bin/env python3
"""Recompute native-CV5 robustness tables for CPD and current-grouped fDNC."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "runs/rld_robustness_cv5_seed42_v2/results"
BENCHMARK = ROOT / "runs/unified_main_benchmark_cv5_seed42_v1/verified_fold_cells.csv"
OUTPUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_native_cv5_cpd_fdnc"
INPUTS = {
    "CPD": RESULTS / "cpd/cpd_all_cells.csv",
    "fDNC": RESULTS / "fdnc/fdnc_all_cells.csv",
}
SEEDS = {"CPD": "deterministic", "fDNC": "42"}
FOLDS = tuple(range(5))
KIND_ORDER = {"coord_noise": 0, "activity_noise": 1, "missing": 2, "outlier": 3}
KIND_LABEL = {
    "coord_noise": "Coordinate noise",
    "activity_noise": "Activity noise",
    "missing": "Missing neurons",
    "outlier": "Distractors",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def benchmark_cells() -> dict[tuple[str, int], dict]:
    result = {}
    for row in read_csv(BENCHMARK):
        if row["dataset"] == "rld" and row["method"] in INPUTS:
            result[(row["method"], int(row["fold"]))] = row
    expected = {(method, fold) for method in INPUTS for fold in FOLDS}
    if set(result) != expected:
        raise RuntimeError(f"Missing benchmark cells: {sorted(expected - set(result))}")
    return result


def normalize(row: dict, method: str) -> dict:
    coverage_key = "coverage" if "coverage" in row else "coverage_vs_clean"
    return {
        "method": method,
        "fold": int(row["fold"]),
        "kind": row["kind"],
        "severity": float(row["severity"]),
        "perturbation_seed": int(row["perturbation_seed"]),
        "queries": int(row["queries"]),
        "top1": float(row["top1"]),
        "top5": float(row["top5"]),
        "hungarian": float(row["hungarian"]),
        "coverage": float(row[coverage_key]),
        "effective_top1": float(row["effective_top1"]),
    }


def clean_gate(rows: list[dict], benchmark: dict[tuple[str, int], dict]) -> list[dict]:
    audit = []
    for method in INPUTS:
        for fold in FOLDS:
            expected = benchmark[(method, fold)]
            cells = [
                row for row in rows
                if row["method"] == method and row["fold"] == fold
                and math.isclose(row["severity"], 0.0)
            ]
            if not cells:
                raise RuntimeError(f"{method} fold{fold}: no severity-zero cells")
            differences = []
            for row in cells:
                for metric in ("queries", "top1", "top5", "hungarian"):
                    got = row[metric]
                    want = int(expected[metric]) if metric == "queries" else float(expected[metric])
                    if got != want:
                        differences.append({
                            "kind": row["kind"], "metric": metric,
                            "observed": got, "benchmark": want,
                        })
            if differences:
                raise RuntimeError(f"{method} fold{fold} clean mismatch: {differences}")
            audit.append({
                "method": method, "fold": fold, "passed": True,
                "queries": int(expected["queries"]),
                "top1": float(expected["top1"]),
                "top5": float(expected["top5"]),
                "hungarian": float(expected["hungarian"]),
            })
    return audit


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    metrics = ("top1", "top5", "hungarian", "coverage", "effective_top1")
    groups: dict[tuple[str, str, float, int], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["kind"], row["severity"], row["fold"])].append(row)

    fold_rows = []
    for (method, kind, severity, fold), cells in sorted(
        groups.items(),
        key=lambda item: (item[0][0], KIND_ORDER.get(item[0][1], 99), item[0][2], item[0][3]),
    ):
        out = {
            "method": method, "kind": kind, "severity": severity, "fold": fold,
            "perturbation_replicates": len(cells),
        }
        for metric in metrics:
            out[metric] = float(np.mean([cell[metric] for cell in cells]))
        fold_rows.append(out)

    summary = []
    keys = sorted(
        {(row["method"], row["kind"], row["severity"]) for row in fold_rows},
        key=lambda key: (key[0], KIND_ORDER.get(key[1], 99), key[2]),
    )
    for method, kind, severity in keys:
        cells = [
            row for row in fold_rows
            if (row["method"], row["kind"], row["severity"]) == (method, kind, severity)
        ]
        if sorted(row["fold"] for row in cells) != list(FOLDS):
            raise RuntimeError(f"{method} {kind} severity={severity}: incomplete folds")
        out = {"method": method, "kind": kind, "severity": severity, "folds": len(cells)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in cells], dtype=np.float64)
            out[f"{metric}_mean"] = float(values.mean())
            out[f"{metric}_sd"] = float(values.std(ddof=1))
        summary.append(out)
    return fold_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(mean: float, sd: float) -> str:
    return f"{100 * mean:.2f} ± {100 * sd:.2f}%"


def table(method: str, summary: list[dict], missing: list[str]) -> str:
    rows = [row for row in summary if row["method"] == method]
    clean = next(row for row in rows if row["kind"] == "coord_noise" and row["severity"] == 0.0)
    lines = [
        f"# {method} robustness — native grouped CV5 × {SEEDS[method]}",
        "",
        "Within each fold, perturbation replicates are averaged first; values are the unweighted mean ± sample SD across five folds. No shared-cohort rescoring is used.",
        "",
        "| Corruption | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
        "| Clean | 0.00 | {top1} | {hung} | {coverage} | {effective} |".format(
            top1=pct(clean["top1_mean"], clean["top1_sd"]),
            hung=pct(clean["hungarian_mean"], clean["hungarian_sd"]),
            coverage=pct(clean["coverage_mean"], clean["coverage_sd"]),
            effective=pct(clean["effective_top1_mean"], clean["effective_top1_sd"]),
        ),
    ]
    for row in rows:
        if row["severity"] == 0.0:
            continue
        lines.append(
            "| {kind} | {severity:.2f} | {top1} | {hung} | {coverage} | {effective} |".format(
                kind=KIND_LABEL.get(row["kind"], row["kind"]), severity=row["severity"],
                top1=pct(row["top1_mean"], row["top1_sd"]),
                hung=pct(row["hungarian_mean"], row["hungarian_sd"]),
                coverage=pct(row["coverage_mean"], row["coverage_sd"]),
                effective=pct(row["effective_top1_mean"], row["effective_top1_sd"]),
            )
        )
    lines += ["", "Missing conditions: " + ", ".join(KIND_LABEL[kind] for kind in missing) + ".", ""]
    return "\n".join(lines)


def main() -> None:
    normalized = []
    for method, path in INPUTS.items():
        normalized.extend(normalize(row, method) for row in read_csv(path))
    audit = clean_gate(normalized, benchmark_cells())
    fold_rows, summary = aggregate(normalized)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "fold_level.csv", fold_rows)
    write_csv(OUTPUT / "summary.csv", summary)

    missing_by_method = {}
    required = set(KIND_ORDER)
    for method in INPUTS:
        present = {row["kind"] for row in summary if row["method"] == method}
        missing = sorted(required - present, key=lambda kind: KIND_ORDER[kind])
        missing_by_method[method] = missing
        (OUTPUT / f"{method.upper()}_TABLE.md").write_text(
            table(method, summary, missing), encoding="utf-8"
        )
    (OUTPUT / "AUDIT.json").write_text(json.dumps({
        "protocol": "RLD grouped CV5; CPD deterministic; fDNC seed42; native held-out evaluation",
        "aggregation": "mean perturbation replicates within fold, then unweighted mean and sample SD across five folds",
        "shared_cohort_rescoring": False,
        "clean_gate": audit,
        "missing_kinds": missing_by_method,
        "complete": all(not value for value in missing_by_method.values()),
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"clean_gate": "passed", "missing_kinds": missing_by_method}, indent=2))


if __name__ == "__main__":
    main()
