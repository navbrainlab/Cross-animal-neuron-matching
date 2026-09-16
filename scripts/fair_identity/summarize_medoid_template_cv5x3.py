#!/usr/bin/env python3
"""Audit and summarize fixed outer-training-medoid test results."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/fair_identity_medoid_template_v1/cv5x3_summary"
DATASETS = ("atanas", "rld")
SEEDS = (1, 42, 123)
METRICS = ("top1", "top5", "mrr", "hungarian")
OLD_PAIRWISE = {
    "atanas": {
        "CPD": (0.3924, 0.7429, None), "fDNC": (0.3206, 0.7223, 0.2802),
        "NuCLR": (0.1951, 0.4394, 0.1979), "GeoTransformer": (0.3142, 0.5838, 0.2839),
    },
    "rld": {
        "CPD": (0.0912, 0.2243, None), "fDNC": (0.1895, 0.5235, 0.1570),
        "NuCLR": (0.1321, 0.3968, 0.1184), "GeoTransformer": (0.0951, 0.2805, 0.0738),
    },
}


def load(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def uid(value: str | Path) -> str:
    stem = Path(value).stem
    return stem.split("__")[1] if "__" in stem else stem


def listed_uids(path: Path) -> set[str]:
    return {uid(line.strip()) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")}


def audit_csv(path: Path, reference: str) -> int:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Empty query file: {path}")
    refs = {row["reference_worm"] for row in rows}
    if refs != {reference}:
        raise RuntimeError(f"Non-template reference in {path}: {refs}")
    if any(row["query_worm"] == reference for row in rows):
        raise RuntimeError(f"Template appears as test query: {path}")
    return len(rows)


def collect_cpd(rows: list[dict], audit: list[dict]) -> None:
    for dataset in DATASETS:
        for fold0 in range(5):
            base = ROOT / f"runs/fair_identity_medoid_template_v1/cpd/{dataset}/fold{fold0}"
            result = load(base / "metrics.json")
            if result["protocol"] != "outer_training_geometry_medoid_template_v1":
                raise RuntimeError(f"Wrong CPD protocol: {base}")
            if result["directed_test_test_pairs"] != 0:
                raise RuntimeError(f"CPD test-test pair found: {base}")
            template = result["template_selection"]["template_uid"]
            train_uids = {uid(path) for path in (Path(result["fold_root"]) / "train").glob("*.npz")}
            if template not in train_uids:
                raise RuntimeError(f"CPD template is not in outer train: {base}")
            with (base / "per_query.csv").open(newline="", encoding="utf-8") as handle:
                query_rows = list(csv.DictReader(handle))
            if {x["reference_uid"] for x in query_rows} != {template}:
                raise RuntimeError(f"CPD non-template reference: {base}")
            if any(x["query_uid"] == template for x in query_rows):
                raise RuntimeError(f"CPD template appears as query: {base}")
            metric = result["metrics"]["template_score"]
            for seed in SEEDS:
                rows.append({
                    "dataset": dataset, "method": "CPD", "fold": fold0 + 1,
                    "seed": seed, "seed_invariant": True,
                    "split_family": "Data/cv5_grouped_v1",
                    "top1": metric["top1"], "top5": metric["top5"],
                    "mrr": metric["mrr"], "hungarian": metric["hungarian_accuracy"],
                    "queries": metric["queries"], "template": template,
                })
            audit.append({"dataset": dataset, "method": "CPD", "fold": fold0 + 1,
                          "cells": 3, "reference": template, "test_test_pairs": 0,
                          "status": "pass; deterministic result replicated across seed labels"})


def collect_fdnc(rows: list[dict], audit: list[dict]) -> None:
    root = ROOT / "benchmark_official/runs/fdnc_official_finetuned"
    for dataset in DATASETS:
        for fold in range(1, 6):
            refs = set()
            for seed in SEEDS:
                base = root / dataset / f"fold_{fold}/seed_{seed}/outer_test_medoid_template_v1"
                result = load(base / "result.json")
                if result["evaluation_protocol"] != "outer_training_geometry_medoid_template_v1":
                    raise RuntimeError(f"Wrong fDNC protocol: {base}")
                if result["directed_test_test_pairs"] != 0:
                    raise RuntimeError(f"fDNC test-test pair found: {base}")
                selection = result["template_selection"]
                if selection["selection_split"] != "outer_train_only" or selection["uses_test"]:
                    raise RuntimeError(f"Invalid fDNC template selection: {base}")
                ref = selection["template_worm"]
                if ref not in listed_uids(Path(result["train_list"])):
                    raise RuntimeError(f"fDNC template is not in outer train: {base}")
                refs.add(ref)
                audit_csv(base / "query_level.csv", ref)
                m = result["metrics"]
                rows.append({"dataset": dataset, "method": "fDNC", "fold": fold,
                             "seed": seed, "seed_invariant": False,
                             "split_family": "benchmark_official/protocols",
                             "top1": m["ranking_top1"], "top5": m["top5"],
                             "mrr": m["mrr"], "hungarian": m["assignment_top1"],
                             "queries": m["queries"], "template": ref})
            if len(refs) != 1:
                raise RuntimeError(f"fDNC medoid varies by seed: {dataset} fold {fold}")
            audit.append({"dataset": dataset, "method": "fDNC", "fold": fold,
                          "cells": 3, "reference": next(iter(refs)), "test_test_pairs": 0,
                          "status": "pass"})


def collect_nuclr(rows: list[dict], audit: list[dict]) -> None:
    root = ROOT / "benchmark_official/runs/nuclr_official_scratch50k_cv5x3"
    for dataset in DATASETS:
        for fold in range(1, 6):
            refs = set()
            for seed in SEEDS:
                cell = root / dataset / f"fold_{fold}/seed_{seed}"
                base = cell / "outer_test_medoid_template_v1"
                result = load(base / "result.json")
                protocol = result["protocol"]
                if protocol["test_test_pairing"] or not protocol["test_opened_after_selection"]:
                    raise RuntimeError(f"Invalid NuCLR test protocol: {base}")
                training_protocol = load(cell / "protocol.json")
                selection = training_protocol["template_selection"]
                if selection["selection_split"] != "outer_train_only" or selection["uses_test"]:
                    raise RuntimeError(f"Invalid NuCLR template selection: {cell}")
                ref = protocol["reference_worm"]
                if ref != selection["template_worm"] or ref not in training_protocol["outer_train_worms"]:
                    raise RuntimeError(f"NuCLR reference is not outer-train medoid: {cell}")
                refs.add(ref)
                audit_csv(base / "query_level.csv", ref)
                m = result["metrics"]
                rows.append({"dataset": dataset, "method": "NuCLR", "fold": fold,
                             "seed": seed, "seed_invariant": False,
                             "split_family": "benchmark_official/protocols",
                             "top1": m["ranking_top1"], "top5": m["top5"],
                             "mrr": m["mrr"], "hungarian": m["assignment_top1"],
                             "queries": m["queries"], "template": ref})
            if len(refs) != 1:
                raise RuntimeError(f"NuCLR medoid varies by seed: {dataset} fold {fold}")
            audit.append({"dataset": dataset, "method": "NuCLR", "fold": fold,
                          "cells": 3, "reference": next(iter(refs)), "test_test_pairs": 0,
                          "status": "pass"})


def collect_geo(rows: list[dict], audit: list[dict]) -> None:
    official = ROOT.parent / "geotransformer_official"
    roots = {dataset: official / "cv5x3_results_train_medoid_template_v1" / dataset
             for dataset in DATASETS}
    for dataset, root in roots.items():
        result = load(root / "REPORT.json")
        expected_protocol = "original_semantic_geotransformer_outer_train_geometry_medoid_template_v1"
        if result["protocol"] != expected_protocol:
            raise RuntimeError(f"Unexpected GeoTransformer protocol: {root}")
        if len(result["cells"]) != 15:
            raise RuntimeError(f"GeoTransformer incomplete: {root}")
        selections = {int(item["fold"]): item for item in result["templates"]}
        by_fold = defaultdict(list)
        for cell in result["cells"]:
            fold = int(cell["fold"])
            seed = int(cell["seed"])
            test = cell["test"]
            selection = selections[fold]
            if selection["selection_split"] != "outer_train_only" or selection["uses_test"]:
                raise RuntimeError(f"Invalid GeoTransformer protocol: {dataset} {fold} {seed}")
            ref = cell["template_uid"]
            train_root = official / "data_cv5" / dataset / f"fold_{fold}" / "train"
            train_uids = {path.name for path in train_root.glob("*.npz")}
            if Path(selection["template_path"]).name != ref or ref not in train_uids:
                raise RuntimeError(f"GeoTransformer template is not in outer train: {dataset} {fold} {seed}")
            if any(x["template"] != ref or x["query"] == ref for x in test["per_test_worm"]):
                raise RuntimeError(f"GeoTransformer non-template reference: {dataset} {fold} {seed}")
            rows.append({"dataset": dataset, "method": "GeoTransformer", "fold": fold,
                         "seed": seed, "seed_invariant": False,
                         "split_family": "geotransformer_official/data_cv5",
                         "top1": test["top1"], "top5": test["top5"],
                         "mrr": test["mrr"], "hungarian": test["hungarian_accuracy"],
                         "queries": test["queries"], "template": ref})
            by_fold[fold].append(ref)
        for fold, refs in sorted(by_fold.items()):
            if len(refs) != 3 or len(set(refs)) != 1:
                raise RuntimeError(f"GeoTransformer medoid/cells invalid: {dataset} fold {fold}")
            audit.append({"dataset": dataset, "method": "GeoTransformer", "fold": fold,
                          "cells": 3, "reference": refs[0], "test_test_pairs": 0,
                          "status": "pass"})


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cells: list[dict] = []
    audit: list[dict] = []
    collect_cpd(cells, audit)
    collect_fdnc(cells, audit)
    collect_nuclr(cells, audit)
    collect_geo(cells, audit)
    expected = 2 * 4 * 5 * 3
    if len(cells) != expected:
        raise RuntimeError(f"Expected {expected} rows, got {len(cells)}")

    grouped = defaultdict(list)
    for row in cells:
        grouped[(row["dataset"], row["method"], row["fold"])].append(row)
    folds = []
    for (dataset, method, fold), values in sorted(grouped.items()):
        if {x["seed"] for x in values} != set(SEEDS):
            raise RuntimeError(f"Missing seeds: {dataset} {method} fold {fold}")
        fold_row = {"dataset": dataset, "method": method, "fold": fold}
        for metric in METRICS:
            fold_row[metric] = float(np.mean([x[metric] for x in values]))
        folds.append(fold_row)

    summaries = []
    for dataset in DATASETS:
        for method in ("CPD", "fDNC", "NuCLR", "GeoTransformer"):
            values = [x for x in folds if x["dataset"] == dataset and x["method"] == method]
            if len(values) != 5:
                raise RuntimeError(f"Missing folds: {dataset} {method}")
            row = {"dataset": dataset, "method": method, "folds": 5, "seeds": 3,
                   "aggregation": "3 seeds averaged within fold; mean and sample SD across 5 folds"}
            for metric in METRICS:
                xs = np.asarray([x[metric] for x in values], dtype=float)
                row[f"{metric}_mean"] = float(xs.mean())
                row[f"{metric}_sd"] = float(xs.std(ddof=1))
            summaries.append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "cells.csv", cells)
    write_csv(OUT / "folds.csv", folds)
    write_csv(OUT / "summary.csv", summaries)
    write_csv(OUT / "protocol_audit.csv", audit)
    comparison = []
    for row in summaries:
        old = OLD_PAIRWISE[row["dataset"]][row["method"]]
        for index, metric in enumerate(("top1", "top5", "hungarian")):
            if old[index] is not None:
                comparison.append({"dataset": row["dataset"], "method": row["method"],
                                   "metric": metric, "old_pairwise_mean": old[index],
                                   "medoid_template_mean": row[f"{metric}_mean"],
                                   "delta_percentage_points": 100.0 * (row[f"{metric}_mean"] - old[index])})
    write_csv(OUT / "old_pairwise_vs_medoid.csv", comparison)
    report = {
        "protocol": "outer_training_geometry_medoid_template_v1",
        "aggregation": "3 seeds averaged within fold; mean and sample SD across 5 biological outer folds",
        "cpd_seed_handling": "deterministic fold result replicated across seed labels and marked seed_invariant",
        "split_families": {
            "CPD": "Data/*/cv5_grouped_v1",
            "GeoTransformer": "geotransformer_official/data_cv5",
            "fDNC_and_NuCLR": "benchmark_official/protocols",
        },
        "cross_method_split_warning": "The split families have different fold membership; do not treat cross-method ranking as a strictly paired comparison.",
        "audit": {"status": "passed", "directed_test_test_pairs": 0,
                  "checks": ["single fixed reference per fold", "reference belongs to outer train",
                             "reference identical across seeds", "template never appears as test query"]},
        "summary": summaries,
        "old_pairwise_vs_medoid": comparison,
    }
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
