#!/usr/bin/env python3
"""Re-summarize the component ablation under the main clean-benchmark protocol.

This command is intentionally analysis-only: it does not train, select, or
evaluate a checkpoint.  It consumes the already locked per-fold test metrics,
keeps model seed 42, and refuses to publish a summary unless the Full arm
reproduces every Ours/static fold cell in the current clean benchmark.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("atanas", "rld")
FOLDS = tuple(range(5))
SEED = 42
ARMS = ("full", "geometry_only", "node_only", "no_transport")
DISPLAY = {
    "full": "Full MPRT-Net",
    "geometry_only": "w/o Activity",
    "node_only": "w/o Population Relations",
    "no_transport": "w/o Relation Transport",
}
METRIC_MAP = {
    "top1": "top1_real",
    "top5": "top5_real",
    "mrr": "mrr_real",
    "hungarian": "hungarian_accuracy",
}
PROTOCOL_ID = "main_clean_benchmark_component_ablation_cv5_seed42_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def mean_sd(values: Iterable[float]) -> dict[str, float]:
    items = list(values)
    if len(items) != len(FOLDS):
        raise ValueError(f"Expected five biological-fold values, got {len(items)}")
    return {
        "mean": statistics.mean(items),
        "sample_sd": statistics.stdev(items),
    }


def load_benchmark(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        if row["method"] != "Ours" or row.get("mode") != "static":
            continue
        if int(row["seed"]) != SEED:
            raise ValueError(f"Non-seed-42 main benchmark row: {row}")
        key = (row["dataset"], int(row["fold"]))
        if key in selected:
            raise ValueError(f"Duplicate main benchmark cell: {key}")
        selected[key] = row
    expected = {(dataset, fold) for dataset in DATASETS for fold in FOLDS}
    if set(selected) != expected:
        raise ValueError(
            "Main benchmark must contain exactly 2 datasets x 5 Ours/static "
            f"seed-42 folds; missing={sorted(expected - set(selected))}, "
            f"extra={sorted(set(selected) - expected)}"
        )
    return selected


def metric_path(source: Path, dataset: str, fold: int, arm: str) -> Path:
    return source / dataset / f"fold{fold}" / f"seed{SEED}" / "metrics" / "test" / f"{arm}.json"


def gate_full_metric(
    actual: dict[str, Any], expected: dict[str, str], dataset: str, fold: int
) -> None:
    problems: list[str] = []
    if int(actual["queries"]) != int(expected["queries"]):
        problems.append(f"queries {actual['queries']} != {expected['queries']}")
    for benchmark_name, metric_name in METRIC_MAP.items():
        tolerance = 5e-8 if benchmark_name == "mrr" else 1e-12
        left = float(actual[metric_name])
        right = float(expected[benchmark_name])
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
            problems.append(f"{metric_name} {left:.12g} != {right:.12g}")
    if problems:
        raise RuntimeError(
            f"Full-arm clean gate failed for {dataset}/fold{fold}: " + "; ".join(problems)
        )


def collect(
    source: Path, benchmark: dict[tuple[str, int], dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for fold in FOLDS:
            cell: dict[str, dict[str, Any]] = {}
            for arm in ARMS:
                path = metric_path(source, dataset, fold, arm)
                metric = read_json(path)
                expected_metadata = {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": SEED,
                    "split": "test",
                    "variant": arm,
                }
                differences = {
                    name: (metric.get(name), value)
                    for name, value in expected_metadata.items()
                    if metric.get(name) != value
                }
                if differences:
                    raise RuntimeError(f"Metric metadata mismatch in {path}: {differences}")
                cell[arm] = metric
                provenance.append({"path": str(path.resolve()), "sha256": sha256(path)})

            gate_full_metric(cell["full"], benchmark[(dataset, fold)], dataset, fold)
            full_queries = int(cell["full"]["queries"])
            for arm in ARMS:
                metric = cell[arm]
                if int(metric["queries"]) != full_queries:
                    raise RuntimeError(
                        f"Cohort mismatch for {dataset}/fold{fold}/{arm}: "
                        f"{metric['queries']} != Full {full_queries}"
                    )
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": SEED,
                        "split": "test",
                        "arm": arm,
                        "display": DISPLAY[arm],
                        "queries": full_queries,
                        **{name: float(metric[source_name]) for name, source_name in METRIC_MAP.items()},
                    }
                )
    return rows, provenance


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in DATASETS:
        arms: dict[str, Any] = {}
        for arm in ARMS:
            selected = [row for row in rows if row["dataset"] == dataset and row["arm"] == arm]
            if {row["fold"] for row in selected} != set(FOLDS):
                raise RuntimeError(f"Incomplete biological folds for {dataset}/{arm}")
            entry = {metric: mean_sd(row[metric] for row in selected) for metric in METRIC_MAP}
            if arm != "full":
                deltas = []
                for fold in FOLDS:
                    full = next(row for row in rows if row["dataset"] == dataset and row["fold"] == fold and row["arm"] == "full")
                    ablated = next(row for row in selected if row["fold"] == fold)
                    deltas.append(ablated["top1"] - full["top1"])
                entry["top1_delta_vs_full"] = mean_sd(deltas)
            arms[arm] = entry
        datasets[dataset] = {"biological_folds": 5, "arms": arms}
    return {
        "protocol_id": PROTOCOL_ID,
        "split": "test",
        "seed": SEED,
        "aggregation": "unweighted mean and sample SD across five biological folds",
        "datasets": datasets,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Component ablation — main clean-benchmark protocol",
        "",
        "Protocol: the exact grouped five-fold test split and model seed 42 used by the main clean benchmark. "
        "Each value is the unweighted mean ± sample SD across five biological folds. The Full arm passed "
        "a fold-by-fold reproduction gate against the current Ours/static benchmark cells.",
        "",
    ]
    for dataset in DATASETS:
        lines.extend(
            [
                f"## {dataset.upper()}",
                "",
                "| Arm | Top-1 ↑ | Top-5 ↑ | MRR ↑ | Hungarian ↑ | ΔTop-1 vs Full |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm in ARMS:
            item = summary["datasets"][dataset]["arms"][arm]
            def pct(name: str) -> str:
                return f"{100*item[name]['mean']:.2f} ± {100*item[name]['sample_sd']:.2f}%"
            delta = "—"
            if arm != "full":
                value = item["top1_delta_vs_full"]
                delta = f"{100*value['mean']:+.2f} ± {100*value['sample_sd']:.2f} pp"
            lines.append(
                f"| {DISPLAY[arm]} | {pct('top1')} | {pct('top5')} | "
                f"{item['mrr']['mean']:.4f} ± {item['mrr']['sample_sd']:.4f} | "
                f"{pct('hungarian')} | {delta} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=repo / "runs/mprt_v1_1_component_ablation_cv5x3_v1",
    )
    parser.add_argument(
        "--benchmark-cells",
        type=Path,
        default=repo / "runs/fair_identity_seed42_benchmark_v1/benchmark_seed42_fold_cells.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo / "runs/mprt_v1_1_component_ablation_benchmark_seed42_v1",
    )
    args = parser.parse_args()

    benchmark = load_benchmark(args.benchmark_cells)
    rows, sources = collect(args.source, benchmark)
    summary = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    write_rows(args.output / "fold_cells.csv", rows)
    write_json(args.output / "summary.json", summary)
    (args.output / "SUMMARY.md").write_text(render_markdown(summary), encoding="utf-8")
    write_json(
        args.output / "PROTOCOL.json",
        {
            "protocol_id": PROTOCOL_ID,
            "status": "passed",
            "dataset_split": "exact Data/*/cv5_grouped_v1 folds used by the main benchmark",
            "fit_and_atlas": "fold train only; checkpoint selection by fold validation only",
            "evaluation": "complete fold test cohort; no robustness common-cohort filtering",
            "model_seed": SEED,
            "reporting_unit": "one result per biological fold",
            "aggregation": "unweighted mean and sample SD across five folds",
            "clean_reproduction_gate": {
                "reference": str(args.benchmark_cells.resolve()),
                "reference_sha256": sha256(args.benchmark_cells),
                "method": "Ours",
                "mode": "static",
                "queries_top1_top5_hungarian_atol": 1e-12,
                "mrr_atol": 5e-8,
                "folds_passed": 10,
            },
            "source_metrics": sources,
        },
    )
    print(f"PASS: Full reproduces all 10 main-benchmark fold cells; output={args.output}")


if __name__ == "__main__":
    main()
