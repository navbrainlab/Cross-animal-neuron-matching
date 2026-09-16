#!/usr/bin/env python3
"""Audit known/unknown target membership in the rejection-comparison outputs."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / ("neurid" if (ROOT / "neurid").is_dir() else "mprt_net_v1_1")
RUNS = ROOT / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld"
CORR = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions/MANIFEST.json"
RESULTS = (
    ROOT
    / "runs/rld_robustness_cv5_seed42_v2/formal_probability_rejection_calibration_ours"
)


def load_state(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CORR)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(PACKAGE.resolve()))
    from mprt_net.data import load_worm, unique_identity_map

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    totals = {
        "checked_condition_recordings": 0,
        "known_query_instances": 0,
        "synthetic_unknown_query_instances": 0,
        "context_only_node_instances_excluded": 0,
        "validation_known_query_instances": 0,
        "validation_synthetic_unknown_query_instances": 0,
    }
    validation_test_uid_disjoint = True
    thresholds_frozen = True
    validation_constraints_met = True
    for fold in range(5):
        checkpoint = args.runs / f"fold{fold}/seed42/dynamic/low_rank_r8/best.pt"
        state = load_state(checkpoint)
        reference = set(map(str, state["atlas_identity_to_slot"]))
        conditions = [
            row
            for row in manifest["conditions"]
            if int(row["fold"]) == fold and str(row["kind"]) == "outlier"
        ]
        threshold_path = args.results / "calibration" / f"fold{fold}" / "thresholds.json"
        thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
        validation_path = (
            args.results
            / "calibration"
            / f"fold{fold}"
            / "pooled_validation_queries.csv.gz"
        )
        with gzip.open(validation_path, "rt", newline="", encoding="utf-8") as handle:
            validation_rows = [dict(row) for row in csv.DictReader(handle)]
        validation_uids = {row["uid"] for row in validation_rows}
        totals["validation_known_query_instances"] += sum(
            row["target_type"] == "known" for row in validation_rows
        )
        totals["validation_synthetic_unknown_query_instances"] += sum(
            row["target_type"] == "synthetic_unknown" for row in validation_rows
        )
        for detail in thresholds["operating_points"].values():
            score = detail["score"]
            threshold = float(detail["threshold"])
            unknown = [
                row for row in validation_rows if row["target_type"] == "synthetic_unknown"
            ]
            known = [row for row in validation_rows if row["target_type"] == "known"]
            recall = sum(float(row[score]) > threshold for row in unknown) / len(unknown)
            false_reject = sum(float(row[score]) > threshold for row in known) / len(known)
            reject_top1 = sum(
                int(row["correct_real"]) and not float(row[score]) > threshold
                for row in known
            ) / len(known)
            validation_constraints_met &= (
                recall + 1e-12 >= float(detail["minimum_unknown_recall"])
                and abs(recall - float(detail["unknown_recall"])) < 1e-12
                and abs(false_reject - float(detail["known_false_reject_rate"])) < 1e-12
                and abs(reject_top1 - float(detail["reject_aware_top1"])) < 1e-12
            )

        test_uids: set[str] = set()
        for condition in conditions:
            name = Path(condition["root"]).name
            query_path = (
                args.results
                / "cells"
                / f"fold{fold}"
                / name
                / "raw_queries.csv.gz"
            )
            observed: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
            with gzip.open(query_path, "rt", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    observed[str(row["uid"])][int(row["node_index"])] = row

            for path in sorted((Path(condition["root"]) / "test").glob("*.npz")):
                sample = load_worm(path)
                test_uids.add(sample.uid)
                unique = unique_identity_map(sample)
                expected: dict[int, str] = {
                    index: "synthetic_unknown"
                    for index, identity in enumerate(sample.cell_ids)
                    if identity.startswith("__OUTLIER_")
                }
                expected.update(
                    {
                        index: "known"
                        for identity, index in unique.items()
                        if identity in reference
                    }
                )
                got = {
                    index: row["target_type"]
                    for index, row in observed[sample.uid].items()
                }
                if got != expected:
                    raise RuntimeError(
                        f"target membership mismatch: fold={fold} condition={name} "
                        f"uid={sample.uid} observed={len(got)} expected={len(expected)}"
                    )
                for index, target_type in expected.items():
                    row = observed[sample.uid][index]
                    identity = sample.cell_ids[index]
                    if row["cell_identity"] != identity:
                        raise RuntimeError(f"saved identity mismatch: {query_path}")
                    if target_type == "known" and (
                        identity not in reference or identity.startswith("__OUTLIER_")
                    ):
                        raise RuntimeError(f"invalid known target: {identity}")
                    if target_type == "synthetic_unknown" and (
                        not identity.startswith("__OUTLIER_") or identity in reference
                    ):
                        raise RuntimeError(f"invalid unknown target: {identity}")
                totals["checked_condition_recordings"] += 1
                totals["known_query_instances"] += sum(
                    target == "known" for target in expected.values()
                )
                totals["synthetic_unknown_query_instances"] += sum(
                    target == "synthetic_unknown" for target in expected.values()
                )
                totals["context_only_node_instances_excluded"] += (
                    sample.num_nodes - len(expected)
                )
            for setting, detail in thresholds["operating_points"].items():
                setting_queries = (
                    args.results
                    / "cells"
                    / f"fold{fold}"
                    / name
                    / setting
                    / "queries.csv.gz"
                )
                with gzip.open(
                    setting_queries, "rt", newline="", encoding="utf-8"
                ) as handle:
                    first = next(csv.DictReader(handle))
                thresholds_frozen &= (
                    first["rejection_score"] != ""
                    and abs(
                        float(first["rejection_threshold"])
                        - float(detail["threshold"])
                    )
                    < 1e-12
                )
        validation_test_uid_disjoint &= validation_uids.isdisjoint(test_uids)

    report = {
        "scope": "all 80 distractor condition cells across five folds",
        **totals,
        "checks": {
            "known_is_unique_supervised_and_present_in_fold_reference": True,
            "synthetic_unknown_has_outlier_prefix_and_is_absent_from_reference": True,
            "query_rows_exactly_equal_known_union_synthetic_unknown": True,
            "native_unlabeled_nodes_are_excluded_from_metric_targets": True,
            "saved_cell_identity_matches_source_node": True,
            "validation_and_test_recording_uids_are_disjoint": validation_test_uid_disjoint,
            "validation_recall_constraints_and_saved_metrics_recompute": validation_constraints_met,
            "one_calibrated_threshold_is_frozen_across_test_conditions": thresholds_frozen,
        },
        "status": (
            "PASS"
            if validation_test_uid_disjoint
            and validation_constraints_met
            and thresholds_frozen
            else "FAIL"
        ),
    }
    output = args.output or args.results / "KNOWN_TARGET_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
