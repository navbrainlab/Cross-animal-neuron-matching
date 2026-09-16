#!/usr/bin/env python3
"""Validate archived query predictions and rebuild all zebrafish tables."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from scripts.zebrafish.query_record_io import QUERY_COLUMNS
except ModuleNotFoundError:  # Direct execution from the repository root.
    from query_record_io import QUERY_COLUMNS


METHODS = {
    "cpd": ("CPD", None),
    "fdnc": ("fDNC", 42),
    "nuclr": ("NuCLR", 42),
    "geotransformer": ("GeoTransformer", 42),
    "ngm_v2": ("NGM-v2", 42),
    "vanilla_fgw": ("Vanilla FGW", None),
    "fugw_ga_official": ("FUGW (G+A; official)", None),
    "statatlas": ("StatAtlas", None),
    "crf_id_dagger": ("CRF_ID†", None),
    "ours": ("Ours", 42),
}
EXPECTED_QUERIES = {1: 3536, 2: 3954, 3: 768, 4: 1820, 5: 1464, 6: 938, 7: 2492, 8: 2184}
FISH = {1: "func_20150410", 2: "func_20150417", 3: "func_20161004", 4: "islet_20170202", 5: "islet_20170216", 6: "lineage_20160328", 7: "lineage_20170111", 8: "lineage_20170925"}
T_975_DF7 = 2.3646242510102993


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "queries": n,
        "top1": sum(int(row["top1_correct"]) for row in rows) / n,
        "top5": sum(int(row["top5_correct"]) for row in rows) / n,
        "mrr": sum(float(row["reciprocal_rank"]) for row in rows) / n,
        "hungarian": sum(int(row["hungarian_correct"]) for row in rows) / n,
        "coverage": sum(int(row["covered"]) for row in rows) / n,
    }


def paired_top1_ours_vs_crfid(reported: list[dict]) -> tuple[list[dict], dict]:
    """Compute the paired mean difference and t interval over held-out fish."""
    by_method = {
        method: {int(row["fold"]): row for row in reported if row["method"] == method}
        for method in ("ours", "crf_id_dagger")
    }
    expected_folds = set(range(1, 9))
    for method, rows in by_method.items():
        if set(rows) != expected_folds:
            raise RuntimeError(f"paired Top-1 requires folds 1..8 for {method}")

    paired_rows = []
    for fold in range(1, 9):
        ours = by_method["ours"][fold]
        crfid = by_method["crf_id_dagger"][fold]
        if ours["test_fish"] != crfid["test_fish"]:
            raise RuntimeError(f"paired Top-1 fish mismatch in fold {fold}")
        ours_top1 = float(ours["top1"])
        crfid_top1 = float(crfid["top1"])
        paired_rows.append({
            "fold": fold,
            "test_fish": ours["test_fish"],
            "neurid_top1": ours_top1,
            "crf_id_top1": crfid_top1,
            "delta_top1": ours_top1 - crfid_top1,
        })

    deltas = np.asarray([row["delta_top1"] for row in paired_rows], dtype=np.float64)
    mean_delta = float(deltas.mean())
    sd_delta = float(deltas.std(ddof=1))
    se_delta = sd_delta / np.sqrt(len(deltas))
    margin = T_975_DF7 * se_delta
    summary = {
        "contrast": "NeuRID minus CRF_ID",
        "metric": "Top-1 accuracy",
        "unit": "held-out fish",
        "n": len(deltas),
        "mean_delta": mean_delta,
        "mean_delta_pp": 100 * mean_delta,
        "sd_delta": sd_delta,
        "sd_delta_pp": 100 * sd_delta,
        "se_delta": se_delta,
        "se_delta_pp": 100 * se_delta,
        "ci_level": 0.95,
        "ci_method": "two-sided paired t interval over held-out-fish differences",
        "degrees_of_freedom": 7,
        "t_critical": T_975_DF7,
        "ci_low": mean_delta - margin,
        "ci_high": mean_delta + margin,
        "ci_low_pp": 100 * (mean_delta - margin),
        "ci_high_pp": 100 * (mean_delta + margin),
    }
    return paired_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--test-pairs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reported-fold-results", type=Path, default=None)
    args = parser.parse_args()

    with args.test_pairs.open(encoding="utf-8", newline="") as handle:
        pair_rows = list(csv.DictReader(handle))
    expected_pairs = {(int(r["fold"]), int(r["pair_index"])): r for r in pair_rows}
    fold_rows, all_pair_metrics, errors = [], [], []
    seen = set()

    for method, (display, expected_seed) in METHODS.items():
        for fold in range(1, 9):
            path = args.prediction_root / method / f"fold_{fold}.csv.gz"
            if not path.exists():
                errors.append(f"missing {path}")
                continue
            with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != QUERY_COLUMNS:
                    errors.append(f"{path}: column schema mismatch")
                rows = list(reader)
            if len(rows) != EXPECTED_QUERIES[fold]:
                errors.append(f"{method}/fold_{fold}: {len(rows)} != {EXPECTED_QUERIES[fold]}")

            grouped = defaultdict(list)
            for row in rows:
                key = (method, fold, int(row["pair_index"]), row["direction"], int(row["query_candidate_index"]))
                if key in seen:
                    errors.append(f"duplicate query key: {key}")
                seen.add(key)
                if row["method"] != method or int(row["fold"]) != fold:
                    errors.append(f"metadata mismatch: {key}")
                actual_seed = None if row["seed"] == "" else int(row["seed"])
                if actual_seed != expected_seed:
                    errors.append(f"seed mismatch: {key}: {actual_seed} != {expected_seed}")
                pair_key = (fold, int(row["pair_index"]))
                if pair_key not in expected_pairs:
                    errors.append(f"unknown pair: {pair_key}")
                else:
                    expected = expected_pairs[pair_key]
                    if row["pair_id"] != expected["pair_id"]:
                        errors.append(f"pair-id mismatch: {key}")
                    expected_candidates = expected["r_candidates"] if row["direction"] == "q_to_r" else expected["q_candidates"]
                    if int(row["candidate_count"]) != int(expected_candidates):
                        errors.append(f"candidate-count mismatch: {key}")
                grouped[int(row["pair_index"])].append(row)

            value = metrics(rows)
            fold_rows.append({
                "method": method, "display_name": display, "fold": fold,
                "test_fish": FISH[fold], "seed": "" if expected_seed is None else expected_seed,
                "pairs": len(grouped), **value,
            })
            for pair_index, group in sorted(grouped.items()):
                pair = expected_pairs[(fold, pair_index)]
                all_pair_metrics.append({
                    "method": method, "display_name": display, "fold": fold,
                    "test_fish": FISH[fold], "seed": "" if expected_seed is None else expected_seed,
                    "pair_index": pair_index, "pair_id": pair["pair_id"],
                    **metrics(group),
                })

    summary_rows = []
    for method, (display, expected_seed) in METHODS.items():
        selected = [row for row in fold_rows if row["method"] == method]
        if len(selected) != 8:
            continue
        summary = {"method": method, "display_name": display, "seed": "" if expected_seed is None else expected_seed, "folds": 8}
        for metric in ("top1", "top5", "mrr", "hungarian", "coverage"):
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=1))
        summary_rows.append(summary)

    if errors:
        raise RuntimeError("query-record validation failed:\n" + "\n".join(errors[:50]))
    write_csv(args.output_root / "fold_results.csv", fold_rows)
    write_csv(args.output_root / "pair_results.csv", all_pair_metrics)
    write_csv(args.output_root / "summary.csv", summary_rows)
    lines = [
        "# Zebrafish LOFO8 results rebuilt from per-query predictions", "",
        "Unweighted mean ± sample SD (`ddof=1`) across eight held-out fish. Learned methods use seed 42; deterministic methods have no seed.", "",
        "| Method | Top-1 | Top-5 | MRR | Hungarian |", "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['display_name']} | {100*row['top1_mean']:.2f} ± {100*row['top1_sd']:.2f}% | "
            f"{100*row['top5_mean']:.2f} ± {100*row['top5_sd']:.2f}% | "
            f"{row['mrr_mean']:.4f} ± {row['mrr_sd']:.4f} | "
            f"{100*row['hungarian_mean']:.2f} ± {100*row['hungarian_sd']:.2f}% |"
        )
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    comparison_rows = []
    reported_summary_rows = []
    paired_top1_summary = None
    if args.reported_fold_results is not None:
        with args.reported_fold_results.open(encoding="utf-8", newline="") as handle:
            reported = list(csv.DictReader(handle))
        if len(reported) != len(METHODS) * 8:
            raise RuntimeError(
                f"reported fold table has {len(reported)} rows, expected {len(METHODS) * 8}"
            )
        replay_by_key = {(row["method"], int(row["fold"])): row for row in fold_rows}
        for row in reported:
            key = (row["method"], int(row["fold"]))
            if key not in replay_by_key:
                raise RuntimeError(f"reported fold is absent from replay: {key}")
            replay = replay_by_key[key]
            comparison_rows.append({
                "method": key[0], "fold": key[1],
                "top1_delta": replay["top1"] - float(row["top1"]),
                "top5_delta": replay["top5"] - float(row["top5"]),
                "mrr_delta": replay["mrr"] - float(row["mrr"]),
                "hungarian_delta": replay["hungarian"] - float(row["hungarian"]),
            })
        for method, (display, expected_seed) in METHODS.items():
            selected = [row for row in reported if row["method"] == method]
            summary = {"method": method, "display_name": display, "seed": "" if expected_seed is None else expected_seed, "folds": 8}
            for metric in ("top1", "top5", "mrr", "hungarian"):
                values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_sd"] = float(values.std(ddof=1))
            reported_summary_rows.append(summary)
        write_csv(args.output_root / "reported_summary.csv", reported_summary_rows)
        write_csv(args.output_root / "replay_comparison.csv", comparison_rows)
        paired_rows, paired_top1_summary = paired_top1_ours_vs_crfid(reported)
        write_csv(args.output_root / "paired_top1_ours_vs_crfid.csv", paired_rows)
        (args.output_root / "paired_top1_ours_vs_crfid.json").write_text(
            json.dumps(paired_top1_summary, indent=2) + "\n", encoding="utf-8"
        )
        reported_lines = [
            "# Zebrafish LOFO8 reported results", "",
            "Unweighted mean ± sample SD (`ddof=1`) across the eight archived fold records.", "",
            "| Method | Top-1 | Top-5 | MRR | Hungarian |", "|---|---:|---:|---:|---:|",
        ]
        for row in reported_summary_rows:
            reported_lines.append(
                f"| {row['display_name']} | {100*row['top1_mean']:.2f} ± {100*row['top1_sd']:.2f}% | "
                f"{100*row['top5_mean']:.2f} ± {100*row['top5_sd']:.2f}% | "
                f"{row['mrr_mean']:.4f} ± {row['mrr_sd']:.4f} | "
                f"{100*row['hungarian_mean']:.2f} ± {100*row['hungarian_sd']:.2f}% |"
            )
        reported_lines.extend([
            "", "## Paired Top-1 comparison", "",
            "For each held-out fish, the difference is `NeuRID Top-1 - CRF_ID Top-1`. "
            "The confidence interval is a two-sided paired t interval over the eight fish-level differences.", "",
            f"NeuRID improves Top-1 over CRF_ID by {paired_top1_summary['mean_delta_pp']:.2f} pp "
            f"(paired 95% CI [{paired_top1_summary['ci_low_pp']:.2f}, "
            f"{paired_top1_summary['ci_high_pp']:.2f}] pp).",
            "Because the interval includes zero, these eight fish do not establish a stable positive improvement at the 95% confidence level.",
        ])
        (args.output_root / "reported_summary.md").write_text("\n".join(reported_lines) + "\n", encoding="utf-8")
    validation = {
        "status": "PASS", "methods": len(summary_rows), "fold_results": len(fold_rows),
        "physical_pair_method_records": len(all_pair_metrics),
        "query_records": len(seen), "expected_query_records": len(METHODS) * sum(EXPECTED_QUERIES.values()),
        "aggregation": "unweighted fold mean and sample SD", "sample_sd_ddof": 1,
    }
    if comparison_rows:
        validation["reported_vs_cpu_replay_max_abs_delta"] = {
            metric: max(abs(float(row[metric])) for row in comparison_rows)
            for metric in ("top1_delta", "top5_delta", "mrr_delta", "hungarian_delta")
        }
        validation["replay_tolerance"] = 0.003
        if max(validation["reported_vs_cpu_replay_max_abs_delta"].values()) > 0.003:
            raise RuntimeError("reported/CPU-replay difference exceeds 0.003")
    if paired_top1_summary is not None:
        validation["paired_top1_ours_vs_crfid"] = paired_top1_summary
    (args.output_root / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
