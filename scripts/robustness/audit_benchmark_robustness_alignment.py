#!/usr/bin/env python3
"""Fail-closed audit of the current Benchmark/Robustness alignment.

The audit is read-only.  It checks biological split membership, severity-zero
native metrics, model-seed checkpoint availability, and required corruption
families.  A cross-method robustness table is publishable only when every gate
passes; common-cohort rescoring is deliberately not accepted as a clean gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


METHODS = ("CPD", "fDNC", "NuCLR", "GeoTransformer", "Ours")
FOLDS = tuple(range(5))
SEEDS = (1, 42, 123)
METRICS = ("queries", "top1", "top5", "mrr", "hungarian")


def uid(value: str | Path) -> str:
    stem = Path(value).stem
    parts = stem.split("__")
    return parts[1] if len(parts) >= 3 else stem


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def path_uids(root: Path, split: str) -> set[str]:
    return {uid(path) for path in (root / split).glob("*.npz")}


def listed_uids(path: Path) -> set[str]:
    return {
        uid(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def benchmark_membership(repo: Path, method: str, fold: int) -> dict[str, set[str]]:
    if method in {"CPD", "Ours"}:
        root = repo / f"Data/Dunn_001623/cv5_grouped_v1/fold_{fold}"
        return {split: path_uids(root, split) for split in ("train", "val", "test")}
    if method in {"fDNC", "NuCLR"}:
        root = repo / f"baselines/official/protocols/rld/fold_{fold + 1}"
        return {split: listed_uids(root / f"{split}.txt") for split in ("train", "val", "test")}
    if method == "GeoTransformer":
        root = repo.parent / f"geotransformer_official/data_cv5/rld/fold_{fold + 1}"
        return {split: path_uids(root, split) for split in ("train", "val", "test")}
    raise ValueError(method)


def shared_membership(repo: Path, fold: int) -> dict[str, set[str]]:
    root = repo / f"Data/Dunn_001623/cv5_grouped_v1/fold_{fold}"
    return {split: path_uids(root, split) for split in ("train", "val", "test")}


def benchmark_seed42(repo: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = read_csv(repo / "runs/fair_identity_seed42_benchmark_v1/benchmark_seed42_fold_cells.csv")
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        if row["dataset"] != "rld" or row["method"] not in METHODS or int(row["seed"]) != 42:
            continue
        fold = int(row["fold"])
        # CPD/fDNC/NuCLR/Geo use display folds 1..5; Ours stores physical 0..4.
        physical_fold = fold if row["method"] == "Ours" else fold - 1
        out[(row["method"], physical_fold)] = {
            "queries": int(row["queries"]),
            "top1": float(row["top1"]),
            "top5": float(row["top5"]),
            "mrr": float(row["mrr"]),
            "hungarian": float(row["hungarian"]),
        }
    expected = {(method, fold) for method in METHODS for fold in FOLDS}
    if set(out) != expected:
        raise RuntimeError(f"Incomplete seed-42 benchmark cells: {sorted(expected - set(out))}")
    return out


def native_clean(repo: Path, method: str) -> dict[int, dict[str, Any]]:
    base = repo / "runs/rld_robustness_cv5_seed42_v2/results"
    out: dict[int, dict[str, Any]] = {}
    if method in {"CPD", "fDNC", "Ours"}:
        filename = {
            "CPD": base / "cpd/cpd_all_cells.csv",
            "fDNC": base / "fdnc/fdnc_all_cells.csv",
            "Ours": base / "ours_static_seed42/ours_all_cells.csv",
        }[method]
        for row in read_csv(filename):
            if row["kind"] != "coord_noise" or float(row["severity"]) != 0.0:
                continue
            out[int(row["fold"])] = {
                "queries": int(row["queries"]),
                "top1": float(row["top1"]),
                "top5": float(row["top5"]),
                "mrr": float(row["mrr"]),
                "hungarian": float(row["hungarian"]),
            }
    elif method == "NuCLR":
        for row in read_csv(base / "nuclr/nuclr_fold_level_summary.csv"):
            if row["kind"] != "coord_noise" or float(row["severity"]) != 0.0:
                continue
            out[int(row["fold"])] = {
                "queries": None,
                "top1": float(row["ranking_top1"]),
                "top5": float(row["top5"]),
                "mrr": float(row["mrr"]),
                "hungarian": float(row["assignment_top1"]),
            }
    elif method == "GeoTransformer":
        for fold in FOLDS:
            data = read_json(base / f"geotransformer/fold{fold}/coord_noise_l0.00_p0/result.json")
            test = data["test"]
            out[fold] = {
                "queries": int(test["queries"]),
                "top1": float(test["top1"]),
                "top5": float(test["top5"]),
                "mrr": float(test["mrr"]),
                "hungarian": float(test["hungarian_accuracy"]),
            }
    return out


def checkpoint_availability(repo: Path) -> dict[str, dict[str, Any]]:
    roots = {
        "fDNC": repo / "runs/fdnc_current_grouped_cv_v2/rld",
        "NuCLR": repo / "baselines/official/runs/nuclr_official_scratch50k_current_cv_seed42/rld",
        "GeoTransformer": repo.parent / "geotransformer_official/current_grouped_selection/rld",
        "Ours": repo / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld",
    }
    report: dict[str, dict[str, Any]] = {
        "CPD": {"deterministic": True, "available_seeds": [42], "complete_cv5": True}
    }
    patterns = {
        "fDNC": "fold{fold}/seed{seed}/selected/best.pt",
        "NuCLR": "fold_{fold}/seed_{seed}/selected/best.pt",
        "GeoTransformer": "fold{fold}/seed{seed}/best_checkpoint.txt",
        "Ours": "fold{fold}/seed{seed}/dynamic/low_rank_r8/best.pt",
    }
    for method, root in roots.items():
        available = [
            seed
            for seed in SEEDS
            if all((root / patterns[method].format(fold=fold, seed=seed)).is_file() for fold in FOLDS)
        ]
        report[method] = {
            "deterministic": False,
            "available_seeds": available,
            "complete_cv5_seed42": 42 in available,
            "complete_cv5x3": set(available) == set(SEEDS),
        }
    return report


def main() -> None:
    repo_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=repo_default)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_default / "runs/rld_robustness_benchmark_alignment_audit_v1",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()

    split_rows = []
    for method in METHODS:
        for fold in FOLDS:
            expected = shared_membership(repo, fold)
            observed = benchmark_membership(repo, method, fold)
            row: dict[str, Any] = {"method": method, "fold": fold}
            exact = True
            for split in ("train", "val", "test"):
                row[f"benchmark_{split}"] = len(observed[split])
                row[f"shared_{split}"] = len(expected[split])
                row[f"{split}_symmetric_difference"] = len(observed[split] ^ expected[split])
                exact &= observed[split] == expected[split]
            row["exact_shared_cv5"] = exact
            split_rows.append(row)

    benchmark = benchmark_seed42(repo)
    clean_rows = []
    for method in METHODS:
        replay = native_clean(repo, method)
        for fold in FOLDS:
            want = benchmark[(method, fold)]
            got = replay.get(fold)
            differences: dict[str, Any] = {}
            if got is None:
                differences["missing"] = True
            else:
                for metric in METRICS:
                    if got.get(metric) is None:
                        differences[metric] = {"benchmark": want[metric], "replay": None}
                        continue
                    tolerance = 5e-8 if metric == "mrr" else 1e-12
                    if metric == "queries":
                        ok = int(got[metric]) == int(want[metric])
                    else:
                        ok = math.isclose(float(got[metric]), float(want[metric]), rel_tol=0.0, abs_tol=tolerance)
                    if not ok:
                        differences[metric] = {"benchmark": want[metric], "replay": got[metric]}
            clean_rows.append({
                "method": method,
                "fold": fold,
                "passed": not differences,
                "differences": differences,
            })

    checkpoints = checkpoint_availability(repo)
    manifest = read_json(repo / "runs/rld_robustness_cv5_seed42_v2/corruptions/MANIFEST.json")
    present_kinds = sorted({str(row["kind"]) for row in manifest["conditions"]})
    required_kinds = ["coord_noise", "missing", "outlier", "activity_noise"]
    dustbin_summary_path = (
        repo
        / "runs/rld_robustness_cv5_seed42_v2/formal_dustbin_ablation_ours/summary.csv"
    )
    dustbin_rows = read_csv(dustbin_summary_path) if dustbin_summary_path.is_file() else []
    dustbin_expected = {
        ("capacity_dustbin", severity)
        for severity in (0.1, 0.2, 0.3, 0.4, 0.5)
    }
    dustbin_observed = {
        (row.get("mode", ""), float(row["severity"]))
        for row in dustbin_rows
        if row.get("severity")
        and row.get("unknown_recall_mean")
        and row.get("dustbin_precision_mean")
        and row.get("dustbin_f1_mean")
    }
    dustbin_metrics_present = dustbin_expected <= dustbin_observed
    corruption_gate = {
        "present": present_kinds,
        "required": required_kinds,
        "missing": sorted(set(required_kinds) - set(present_kinds)),
        "shared_manifest": True,
        "distractor_dustbin_metrics_present": dustbin_metrics_present,
        "distractor_dustbin_summary": str(dustbin_summary_path.resolve()),
    }

    split_pass = all(row["exact_shared_cv5"] for row in split_rows)
    clean_pass = all(row["passed"] for row in clean_rows)
    seed42_pass = all(
        item.get("deterministic") or item.get("complete_cv5_seed42")
        for item in checkpoints.values()
    )
    cv5x3_pass = all(
        item.get("deterministic") or item.get("complete_cv5x3")
        for item in checkpoints.values()
    )
    feature_pass = not corruption_gate["missing"] and corruption_gate["distractor_dustbin_metrics_present"]
    report = {
        "status": "blocked_not_publishable",
        "publication_allowed": False,
        "gates": {
            "same_cv5_membership_all_methods": split_pass,
            "severity_zero_equals_current_benchmark_every_method_fold_seed42": clean_pass,
            "shared_cv5_seed42_checkpoints_available": seed42_pass,
            "shared_cv5x3_checkpoints_available": cv5x3_pass,
            "all_required_corruptions_and_metrics": feature_pass,
        },
        "split_rows": split_rows,
        "clean_rows": clean_rows,
        "checkpoint_availability": checkpoints,
        "corruption_contract": corruption_gate,
        "required_resolution_order": [
            "rerun/revise the main benchmark on one shared cv5_grouped_v1 membership",
            "freeze its exact per-method fold/seed checkpoints, atlases/templates, vocabularies and evaluators",
            "materialize shared activity-noise instances",
            "run severity zero first and block every nonzero condition until all clean cells pass",
            "then run nonzero corruptions without any retuning",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.output / "split_alignment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(split_rows[0]))
        writer.writeheader(); writer.writerows(split_rows)
    clean_flat = [
        {"method": row["method"], "fold": row["fold"], "passed": row["passed"],
         "differences": json.dumps(row["differences"], sort_keys=True)}
        for row in clean_rows
    ]
    with (args.output / "clean_gate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(clean_flat[0]))
        writer.writeheader(); writer.writerows(clean_flat)
    print(json.dumps(report["gates"], indent=2))
    print(f"BLOCKED: current cross-method robustness is not publishable; audit={args.output / 'AUDIT.json'}")


if __name__ == "__main__":
    main()
