#!/usr/bin/env python3
"""Evaluate hierarchical, validation-calibrated unknown rejection on RLD CV5.

Identity is always the highest-probability real atlas candidate. Unknown
rejection is a separate binary decision. This script compares the legacy
multiclass argmax rule, a binary 0.5 rule, fold-specific dustbin-probability
thresholds, and fold-specific log-margin thresholds.

Validation distractors use the test corruption generator. All requested
nonzero distractor fractions and replicates are pooled to select one threshold
per fold, score, and recall target. Thresholds are frozen for every test
distractor fraction. No model parameter is retrained or selected here.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.stats import t as student_t


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / ("neurid" if (ROOT / "neurid").is_dir() else "mprt_net_v1_1")
RUNS = ROOT / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld"
CORR = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions"
LEGACY_RAW = (
    ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_rejection_comparison_ours"
)
OUT = (
    ROOT
    / "runs/rld_robustness_cv5_seed42_v2/formal_probability_rejection_calibration_ours"
)
RECALL_TARGETS = (0.80, 0.90, 0.95)
RAW_SCHEMA_VERSION = 3
METRIC_NAMES = (
    "known_top1_real",
    "reject_aware_top1",
    "known_false_reject_rate",
    "unknown_recall",
    "unknown_precision",
)


def probability_setting(target: float) -> str:
    return f"probability_val_r{round(100 * target):02d}"


def margin_setting(target: float) -> str:
    return f"log_margin_val_r{round(100 * target):02d}"


def settings(recall_targets: Sequence[float]) -> tuple[str, ...]:
    return (
        "current_argmax",
        "binary_0_5",
        *(probability_setting(target) for target in recall_targets),
        *(margin_setting(target) for target in recall_targets),
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_gzip_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def read_gzip_csv(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def checkpoint(fold: int) -> Path:
    path = RUNS / f"fold{fold}/seed42/dynamic/low_rank_r8/best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def saved_clean(fold: int) -> dict[str, Any]:
    path = RUNS / f"fold{fold}/seed42/atlas_identity_test/metrics.json"
    return read_json(path)["modes"]["static"]


def conditions(manifest: dict[str, Any], fold: int) -> list[dict[str, Any]]:
    rows = [
        dict(row)
        for row in manifest["conditions"]
        if int(row["fold"]) == fold and str(row["kind"]) == "outlier"
    ]
    return sorted(
        rows, key=lambda row: (float(row["severity"]), int(row["perturbation_seed"]))
    )


def identity_targets(sample: Any, identity_to_slot: dict[str, int]) -> torch.Tensor:
    """Return atlas slots only for unique, supervised, reference-present IDs."""
    from mprt_net.data import unique_identity_map

    target = torch.full(
        (sample.num_nodes,), -1, dtype=torch.long, device=sample.xyz.device
    )
    for identity, node_index in unique_identity_map(sample).items():
        slot = identity_to_slot.get(str(identity))
        if slot is not None:
            target[int(node_index)] = int(slot)
    return target


def summarize_queries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    known = [row for row in rows if row["target_type"] == "known"]
    unknown = [row for row in rows if row["target_type"] == "synthetic_unknown"]
    true_positive = sum(int(row["rejected"]) for row in unknown)
    false_positive = sum(int(row["rejected"]) for row in known)
    predicted_positive = true_positive + false_positive
    return {
        "known_queries": len(known),
        "synthetic_unknown_queries": len(unknown),
        "known_top1_real": (
            sum(int(row["correct_real"]) for row in known) / len(known)
            if known
            else None
        ),
        "reject_aware_top1": (
            sum(int(row["correct_reject_aware"]) for row in known) / len(known)
            if known
            else None
        ),
        "known_false_reject_rate": false_positive / len(known) if known else None,
        "unknown_recall": true_positive / len(unknown) if unknown else None,
        "unknown_precision": (
            true_positive / predicted_positive if predicted_positive else None
        ) if unknown else None,
    }


def apply_threshold(
    rows: list[dict[str, Any]], score_name: str, threshold: float
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        rejected = float(row[score_name]) > threshold
        item = dict(row)
        item.update(
            {
                "rejection_score": row[score_name],
                "rejection_threshold": threshold,
                "rejection_comparator": ">",
                "rejected": int(rejected),
                "correct_reject_aware": (
                    int(bool(int(row["correct_real"])) and not rejected)
                    if row["target_type"] == "known"
                    else ""
                ),
            }
        )
        output.append(item)
    return output


def select_threshold_at_minimum_recall(
    rows: list[dict[str, Any]], score_name: str, minimum_unknown_recall: float
) -> dict[str, Any]:
    """Maximize reject-aware Top-1 subject to minimum unknown recall.

    Reject-aware Top-1 is monotone nondecreasing in the threshold. Therefore,
    the optimum is the largest threshold that still rejects at least
    ``ceil(R0 * n_unknown)`` validation unknowns. ``nextafter`` includes all
    ties at the boundary under the strict comparison.
    """
    if not 0.0 < minimum_unknown_recall <= 1.0:
        raise ValueError("minimum_unknown_recall must be in (0, 1]")
    unknown = np.asarray(
        [
            float(row[score_name])
            for row in rows
            if row["target_type"] == "synthetic_unknown"
        ],
        dtype=np.float64,
    )
    if unknown.ndim != 1 or len(unknown) == 0 or not np.isfinite(unknown).all():
        raise ValueError("validation unknown scores must be a non-empty finite vector")
    required = int(math.ceil(minimum_unknown_recall * len(unknown) - 1e-12))
    boundary = float(np.sort(unknown)[::-1][required - 1])
    threshold = float(np.nextafter(boundary, -np.inf))
    if score_name == "dustbin_probability":
        threshold = max(0.0, threshold)
    evaluated = apply_threshold(rows, score_name, threshold)
    metrics = summarize_queries(evaluated)
    observed = float(metrics["unknown_recall"])
    if observed + 1e-12 < minimum_unknown_recall:
        raise AssertionError("selected threshold violates minimum unknown recall")
    return {
        "score": score_name,
        "reject_rule": "score > threshold",
        "threshold": threshold,
        "minimum_unknown_recall": minimum_unknown_recall,
        "required_unknown_rejections": required,
        "boundary_unknown_score": boundary,
        **metrics,
    }


def infer_queries(
    *,
    model: Any,
    identity_to_slot: dict[str, int],
    cache: Any,
    paths: list[Path],
    fold: int,
    severity: float,
    perturbation_seed: int,
) -> list[dict[str, Any]]:
    from mprt_net.experiments.dustbin_robustness import match_with_solver

    rows: list[dict[str, Any]] = []
    atlas = model.atlas_encoding()
    with torch.inference_mode():
        for path in paths:
            sample_cpu = cache.get(path)
            sample = sample_cpu.to(atlas.nodes.device)
            query = model.encode_population(sample)
            target = identity_targets(sample, identity_to_slot)
            synthetic = torch.tensor(
                [identity.startswith("__OUTLIER_") for identity in sample.cell_ids],
                dtype=torch.bool,
                device=target.device,
            )
            if bool(((target >= 0) & synthetic).any()):
                raise RuntimeError(f"synthetic distractor became known: {path}")
            if severity == 0.0 and bool(synthetic.any()):
                raise RuntimeError(f"clean condition contains distractors: {path}")
            if severity > 0.0 and not bool(synthetic.any()):
                raise RuntimeError(f"distractor condition contains none: {path}")

            known_indices = torch.nonzero(target >= 0, as_tuple=False).flatten()
            unknown_indices = torch.nonzero(synthetic, as_tuple=False).flatten()
            probabilities = match_with_solver(
                model, query, atlas, "capacity_dustbin"
            ).row
            real = probabilities[:, :-1]
            max_real, real_prediction = real.max(dim=1)
            dustbin = probabilities[:, -1]
            margin = dustbin - max_real
            tiny = torch.finfo(probabilities.dtype).tiny
            log_margin = dustbin.clamp_min(tiny).log() - max_real.clamp_min(tiny).log()
            for target_type, indices in (
                ("known", known_indices),
                ("synthetic_unknown", unknown_indices),
            ):
                for node_index in indices.detach().cpu().tolist():
                    slot = int(target[node_index]) if target_type == "known" else -1
                    rows.append(
                        {
                            "raw_schema_version": RAW_SCHEMA_VERSION,
                            "fold": fold,
                            "severity": severity,
                            "perturbation_seed": perturbation_seed,
                            "transport_mode": "capacity_dustbin",
                            "uid": sample_cpu.uid,
                            "node_index": node_index,
                            "cell_identity": sample_cpu.cell_ids[node_index],
                            "target_type": target_type,
                            "target_slot": slot,
                            "prediction_slot": int(real_prediction[node_index]),
                            "max_real_probability": float(max_real[node_index]),
                            "dustbin_probability": float(dustbin[node_index]),
                            "dustbin_margin": float(margin[node_index]),
                            "log_dustbin_margin": float(log_margin[node_index]),
                            "correct_real": (
                                int(real_prediction[node_index] == slot)
                                if target_type == "known"
                                else ""
                            ),
                        }
                    )
    return rows


def normalize_cached_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Upgrade legacy per-query rows without changing their probabilities."""
    output = []
    for source in rows:
        row = dict(source)
        dustbin = float(row["dustbin_probability"])
        max_real = float(row["max_real_probability"])
        row["raw_schema_version"] = RAW_SCHEMA_VERSION
        row["dustbin_margin"] = dustbin - max_real
        row["log_dustbin_margin"] = math.log(max(dustbin, 1e-300)) - math.log(
            max(max_real, 1e-300)
        )
        for key in (
            "setting",
            "rejection_score",
            "rejection_threshold",
            "rejection_comparator",
            "rejected",
            "correct_reject_aware",
        ):
            row.pop(key, None)
        output.append(row)
    return output


def materialize_validation_distractors(
    source_paths: list[Path],
    output: Path,
    severity: float,
    perturbation_seed: int,
    force: bool,
) -> list[Path]:
    from scripts.robustness.prepare_rld_robustness_cv5_seed42 import transform

    condition = output / f"outlier_l{severity:.2f}_p{perturbation_seed}"
    condition.mkdir(parents=True, exist_ok=True)
    paths = []
    for source in source_paths:
        destination = condition / source.name
        if force or not destination.is_file():
            np.savez_compressed(
                destination,
                **transform(source, "outlier", severity, perturbation_seed),
            )
        paths.append(destination)
    return paths


def calibrate_fold(
    *,
    model: Any,
    identity_to_slot: dict[str, int],
    cache: Any,
    validation_paths: list[Path],
    fold: int,
    recall_targets: Sequence[float],
    calibration_severities: Sequence[float],
    calibration_seeds: Sequence[int],
    output: Path,
    force: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    fold_dir = output / "calibration" / f"fold{fold}"
    pooled: list[dict[str, Any]] = []
    for severity in calibration_severities:
        if severity <= 0.0:
            raise ValueError("calibration severities must be nonzero")
        for perturbation_seed in calibration_seeds:
            paths = materialize_validation_distractors(
                validation_paths,
                fold_dir / "validation_corruptions",
                severity,
                perturbation_seed,
                force,
            )
            pooled.extend(
                infer_queries(
                    model=model,
                    identity_to_slot=identity_to_slot,
                    cache=cache,
                    paths=paths,
                    fold=fold,
                    severity=severity,
                    perturbation_seed=perturbation_seed,
                )
            )

    operating_points: dict[str, Any] = {}
    for target in recall_targets:
        operating_points[probability_setting(target)] = (
            select_threshold_at_minimum_recall(
                pooled, "dustbin_probability", target
            )
        )
        operating_points[margin_setting(target)] = select_threshold_at_minimum_recall(
            pooled, "log_dustbin_margin", target
        )
    calibration = {
        "fold": fold,
        "split": "validation",
        "selection_objective": (
            "maximize validation reject-aware Top-1 subject to minimum unknown recall"
        ),
        "pooled_distractor_severities": list(calibration_severities),
        "pooled_perturbation_seeds": list(calibration_seeds),
        "known_definition": (
            "unique supervised ground-truth identity present in current atlas reference"
        ),
        "unknown_definition": "materialized validation __OUTLIER_* distractor",
        "excluded": "native unlabeled/uncertain/duplicate/reference-absent nodes",
        "operating_points": operating_points,
    }
    write_json(fold_dir / "thresholds.json", calibration)
    write_gzip_csv(fold_dir / "pooled_validation_queries.csv.gz", pooled)
    return calibration, pooled


def apply_setting(
    rows: list[dict[str, Any]], setting: str, calibration: dict[str, Any]
) -> list[dict[str, Any]]:
    if setting == "current_argmax":
        score_name, threshold = "log_dustbin_margin", 0.0
    elif setting == "binary_0_5":
        score_name, threshold = "dustbin_probability", 0.5
    elif setting in calibration["operating_points"]:
        detail = calibration["operating_points"][setting]
        score_name, threshold = str(detail["score"]), float(detail["threshold"])
    else:
        raise ValueError(f"unknown setting: {setting}")
    output = apply_threshold(rows, score_name, threshold)
    for row in output:
        row["setting"] = setting
    return output


def evaluate_condition(
    *,
    raw_rows: list[dict[str, Any]],
    fold: int,
    severity: float,
    perturbation_seed: int,
    recordings: int,
    calibration: dict[str, Any],
    recall_targets: Sequence[float],
    out_dir: Path,
) -> list[dict[str, Any]]:
    write_gzip_csv(out_dir / "raw_queries.csv.gz", raw_rows)
    results = []
    for setting in settings(recall_targets):
        query_rows = apply_setting(raw_rows, setting, calibration)
        metrics = {
            "raw_schema_version": RAW_SCHEMA_VERSION,
            "fold": fold,
            "severity": severity,
            "perturbation_seed": perturbation_seed,
            "setting": setting,
            "recordings": recordings,
            **summarize_queries(query_rows),
        }
        results.append(metrics)
        setting_dir = out_dir / setting
        write_json(setting_dir / "metrics.json", metrics)
        write_gzip_csv(setting_dir / "queries.csv.gz", query_rows)
    return results


def mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def paired_mean_ci(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Two-sided paired Student-t interval for fold-level differences."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("paired CI requires at least two finite differences")
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    critical = float(student_t.ppf((1.0 + confidence) / 2.0, df=len(array) - 1))
    half_width = critical * sd / math.sqrt(len(array))
    return {
        "mean": mean,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "sd": sd,
        "folds": int(len(array)),
        "confidence": confidence,
    }


def fold_level_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(row["setting"], row["severity"], row["fold"]) for row in cells})
    for setting, severity, fold in keys:
        part = [
            row
            for row in cells
            if (row["setting"], row["severity"], row["fold"])
            == (setting, severity, fold)
        ]
        item: dict[str, Any] = {
            "setting": setting,
            "severity": severity,
            "fold": fold,
            "perturbation_replicates": len(part),
        }
        for metric in METRIC_NAMES:
            values = [
                float(row[metric])
                for row in part
                if row.get(metric) not in (None, "")
                and math.isfinite(float(row[metric]))
            ]
            item[metric] = float(np.mean(values)) if values else ""
        output.append(item)
    return output


def summarize_fold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for setting, severity in sorted({(row["setting"], row["severity"]) for row in rows}):
        part = [
            row
            for row in rows
            if (row["setting"], row["severity"]) == (setting, severity)
        ]
        item: dict[str, Any] = {"setting": setting, "severity": severity, "folds": len(part)}
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in part if row[metric] != ""]
            if values:
                item[f"{metric}_mean"], item[f"{metric}_sd"] = mean_sd(values)
            else:
                item[f"{metric}_mean"], item[f"{metric}_sd"] = "", ""
        output.append(item)
    return output


def overall_operating_points(fold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average severities within fold, then average biological folds."""
    per_fold: list[dict[str, Any]] = []
    for setting, fold in sorted({(row["setting"], row["fold"]) for row in fold_rows}):
        part = [
            row
            for row in fold_rows
            if row["setting"] == setting
            and row["fold"] == fold
            and float(row["severity"]) > 0
        ]
        item: dict[str, Any] = {"setting": setting, "fold": fold}
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in part if row[metric] != ""]
            item[metric] = float(np.mean(values)) if values else ""
        per_fold.append(item)

    output = []
    for setting in sorted({row["setting"] for row in per_fold}):
        part = [row for row in per_fold if row["setting"] == setting]
        item: dict[str, Any] = {"setting": setting, "folds": len(part)}
        for metric in METRIC_NAMES:
            values = [float(row[metric]) for row in part if row[metric] != ""]
            if values:
                item[f"{metric}_mean"], item[f"{metric}_sd"] = mean_sd(values)
            else:
                item[f"{metric}_mean"], item[f"{metric}_sd"] = "", ""
        output.append(item)
    return output


def paired_comparison_rows(
    fold_rows: list[dict[str, Any]], baseline: str = "current_argmax"
) -> list[dict[str, Any]]:
    """Paired fold-level CIs for clean, pooled, and per-severity comparisons.

    Both reported effects use the reviewer-friendly direction in which a
    positive number is an improvement: candidate minus baseline for Top-1,
    and baseline minus candidate for known false rejection.
    """
    indexed = {
        (str(row["setting"]), int(row["fold"]), float(row["severity"])): row
        for row in fold_rows
    }
    candidate_settings = sorted(
        {str(row["setting"]) for row in fold_rows if row["setting"] != baseline}
    )
    folds = sorted({int(row["fold"]) for row in fold_rows})
    nonzero = sorted({float(row["severity"]) for row in fold_rows if float(row["severity"]) > 0})
    scopes: list[tuple[str, float | str, tuple[float, ...]]] = [
        ("clean", 0.0, (0.0,)),
        ("pooled_nonzero", "0.1-0.5", tuple(nonzero)),
        *[("severity", severity, (severity,)) for severity in nonzero],
    ]
    output = []
    for setting in candidate_settings:
        for scope, severity_label, severities in scopes:
            top1_differences = []
            false_reject_reductions = []
            paired_folds = []
            for fold in folds:
                baseline_rows = [
                    indexed.get((baseline, fold, severity)) for severity in severities
                ]
                candidate_rows = [
                    indexed.get((setting, fold, severity)) for severity in severities
                ]
                if any(row is None for row in baseline_rows + candidate_rows):
                    continue
                baseline_top1 = np.mean(
                    [float(row["reject_aware_top1"]) for row in baseline_rows]
                )
                candidate_top1 = np.mean(
                    [float(row["reject_aware_top1"]) for row in candidate_rows]
                )
                baseline_false_reject = np.mean(
                    [float(row["known_false_reject_rate"]) for row in baseline_rows]
                )
                candidate_false_reject = np.mean(
                    [float(row["known_false_reject_rate"]) for row in candidate_rows]
                )
                top1_differences.append(float(candidate_top1 - baseline_top1))
                false_reject_reductions.append(
                    float(baseline_false_reject - candidate_false_reject)
                )
                paired_folds.append(fold)
            top1 = paired_mean_ci(top1_differences)
            false_reject = paired_mean_ci(false_reject_reductions)
            output.append(
                {
                    "baseline": baseline,
                    "setting": setting,
                    "scope": scope,
                    "severity": severity_label,
                    "paired_folds": ";".join(map(str, paired_folds)),
                    "folds": top1["folds"],
                    "ci_method": "two-sided paired Student-t across biological folds",
                    "confidence": top1["confidence"],
                    "delta_reject_aware_top1_pp": 100.0 * top1["mean"],
                    "delta_reject_aware_top1_ci_low_pp": 100.0 * top1["ci_low"],
                    "delta_reject_aware_top1_ci_high_pp": 100.0 * top1["ci_high"],
                    "known_false_reject_reduction_pp": 100.0 * false_reject["mean"],
                    "known_false_reject_reduction_ci_low_pp": 100.0 * false_reject["ci_low"],
                    "known_false_reject_reduction_ci_high_pp": 100.0 * false_reject["ci_high"],
                }
            )
    return output


def _array_metrics(
    rows: list[dict[str, Any]], score_name: str, threshold: float
) -> dict[str, float]:
    known = [row for row in rows if row["target_type"] == "known"]
    unknown = [row for row in rows if row["target_type"] == "synthetic_unknown"]
    known_scores = np.asarray([float(row[score_name]) for row in known])
    unknown_scores = np.asarray([float(row[score_name]) for row in unknown])
    correct = np.asarray([int(row["correct_real"]) for row in known], dtype=bool)
    known_rejected = known_scores > threshold
    unknown_rejected = unknown_scores > threshold
    return {
        "known_false_reject_rate": float(known_rejected.mean()),
        "unknown_recall": float(unknown_rejected.mean()),
        "reject_aware_top1": float((correct & ~known_rejected).mean()),
    }


def test_threshold_curves(
    raw_by_fold: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    all_rows = [row for rows in raw_by_fold.values() for row in rows]
    margin_values = np.asarray(
        [float(row["log_dustbin_margin"]) for row in all_rows], dtype=np.float64
    )
    grids = {
        "dustbin_probability": np.linspace(0.0, 1.0, 201),
        "log_dustbin_margin": np.unique(
            np.quantile(margin_values, np.linspace(0.0, 1.0, 201))
        ),
    }
    # Keep the curve's hierarchy identical to the operating-point table:
    # replicates -> severity within fold -> biological folds. In particular,
    # high-severity conditions contain more unknown rows and must not silently
    # receive more weight merely because their distractor count is larger.
    cells_by_fold: dict[int, list[list[dict[str, Any]]]] = {}
    for fold, fold_rows in raw_by_fold.items():
        grouped: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
        for row in fold_rows:
            grouped[(float(row["severity"]), int(row["perturbation_seed"]))].append(row)
        cells_by_fold[fold] = list(grouped.values())

    output = []
    for score_name, thresholds in grids.items():
        for threshold in thresholds:
            fold_metrics = []
            for fold_cells in cells_by_fold.values():
                cell_metrics = [
                    _array_metrics(rows, score_name, float(threshold))
                    for rows in fold_cells
                ]
                fold_metrics.append(
                    {
                        metric: float(np.mean([row[metric] for row in cell_metrics]))
                        for metric in (
                            "known_false_reject_rate",
                            "unknown_recall",
                            "reject_aware_top1",
                        )
                    }
                )
            item: dict[str, Any] = {"score": score_name, "threshold": float(threshold)}
            for metric in (
                "known_false_reject_rate",
                "unknown_recall",
                "reject_aware_top1",
            ):
                item[f"{metric}_mean"], item[f"{metric}_sd"] = mean_sd(
                    [row[metric] for row in fold_metrics]
                )
            output.append(item)
    return output


def plot_curves(
    curves: list[dict[str, Any]],
    operating_points: list[dict[str, Any]],
    output: Path,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    styles = {
        "dustbin_probability": ("Dustbin probability", "#2468b4", "-"),
        "log_dustbin_margin": ("Log-margin", "#df6b20", "--"),
    }
    for score_name, (label, color, linestyle) in styles.items():
        part = [row for row in curves if row["score"] == score_name]
        part.sort(key=lambda row: float(row["known_false_reject_rate_mean"]))
        axes[0].plot(
            [row["known_false_reject_rate_mean"] for row in part],
            [row["unknown_recall_mean"] for row in part],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
        part.sort(key=lambda row: float(row["unknown_recall_mean"]))
        axes[1].plot(
            [row["unknown_recall_mean"] for row in part],
            [row["reject_aware_top1_mean"] for row in part],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )

    point_labels = {
        "current_argmax": "Current",
        "binary_0_5": "Binary 0.5",
        "probability_val_r80": "Val @ 80%",
        "probability_val_r90": "Val @ 90%",
        "probability_val_r95": "Val @ 95%",
    }
    markers = ("X", "s", "o", "D", "^")
    for marker, (setting, label) in zip(markers, point_labels.items()):
        matches = [row for row in operating_points if row["setting"] == setting]
        if not matches:
            continue
        row = matches[0]
        axes[0].scatter(
            row["known_false_reject_rate_mean"],
            row["unknown_recall_mean"],
            s=48,
            marker=marker,
            edgecolor="black",
            linewidth=0.5,
            label=label,
            zorder=5,
        )
        axes[1].scatter(
            row["unknown_recall_mean"],
            row["reject_aware_top1_mean"],
            s=48,
            marker=marker,
            edgecolor="black",
            linewidth=0.5,
            zorder=5,
        )

    axes[0].set_xlabel("Known false-rejection rate")
    axes[0].set_ylabel("Unknown recall")
    axes[1].set_xlabel("Unknown recall")
    axes[1].set_ylabel("Known reject-aware Top-1")
    for axis in axes:
        axis.set_xlim(left=0.0)
        axis.set_ylim(bottom=0.0)
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output / "rejection_operating_curves.png", dpi=240)
    fig.savefig(output / "rejection_operating_curves.pdf")
    plt.close(fig)
    return True


def rank_gain_conversion(
    summary: list[dict[str, Any]], baseline_summary_path: Path
) -> list[dict[str, Any]]:
    """Compare rejection cost with the capacity-dustbin ranking gain."""
    if not baseline_summary_path.is_file():
        return []
    baseline_rows = read_csv(baseline_summary_path)
    forced = {
        float(row["severity"]): float(row["known_top1_real_mean"])
        for row in baseline_rows
        if row["setting"] == "no_dustbin_forced"
    }
    output = []
    for row in summary:
        severity = float(row["severity"])
        if severity not in forced:
            continue
        real_top1 = float(row["known_top1_real_mean"])
        reject_top1 = float(row["reject_aware_top1_mean"])
        ranking_gain = real_top1 - forced[severity]
        rejection_cost = real_top1 - reject_top1
        output.append(
            {
                "setting": row["setting"],
                "severity": severity,
                "capacity_dustbin_real_top1": real_top1,
                "no_dustbin_forced_top1": forced[severity],
                "ranking_gain": ranking_gain,
                "rejection_cost": rejection_cost,
                "net_gain_over_no_dustbin_forced": ranking_gain - rejection_cost,
                "rejection_cost_below_ranking_gain": rejection_cost < ranking_gain,
            }
        )
    return output


def aggregate(
    cells: list[dict[str, Any]],
    calibrations: list[dict[str, Any]],
    raw_by_fold: dict[int, list[dict[str, Any]]],
    recall_targets: Sequence[float],
    out_root: Path,
    baseline_summary_path: Path,
) -> None:
    write_csv(out_root / "condition_cells.csv", cells)
    fold_rows = fold_level_rows(cells)
    summary = summarize_fold_rows(fold_rows)
    operating = overall_operating_points(fold_rows)
    paired = paired_comparison_rows(fold_rows)
    curves = test_threshold_curves(raw_by_fold)
    conversion = rank_gain_conversion(summary, baseline_summary_path)
    write_csv(out_root / "fold_level.csv", fold_rows)
    write_csv(out_root / "summary.csv", summary)
    write_csv(
        out_root / "clean_known_only.csv",
        [row for row in summary if float(row["severity"]) == 0.0],
    )
    write_csv(
        out_root / "per_severity_unknown_present.csv",
        [row for row in summary if float(row["severity"]) > 0.0],
    )
    write_csv(out_root / "overall_operating_points.csv", operating)
    write_csv(out_root / "unknown_present_pooled.csv", operating)
    write_csv(out_root / "paired_comparisons.csv", paired)
    write_csv(out_root / "test_threshold_curves.csv", curves)
    write_csv(out_root / "rank_gain_conversion.csv", conversion)

    calibration_rows = []
    for row in calibrations:
        for setting, detail in row["operating_points"].items():
            calibration_rows.append(
                {
                    "fold": row["fold"],
                    "setting": setting,
                    "score": detail["score"],
                    "minimum_unknown_recall": detail["minimum_unknown_recall"],
                    "threshold": detail["threshold"],
                    "validation_unknown_recall": detail["unknown_recall"],
                    "validation_known_false_reject_rate": detail["known_false_reject_rate"],
                    "validation_reject_aware_top1": detail["reject_aware_top1"],
                    "validation_known_queries": detail["known_queries"],
                    "validation_unknown_queries": detail["synthetic_unknown_queries"],
                }
            )
    write_csv(out_root / "calibration.csv", calibration_rows)
    figure_written = plot_curves(curves, operating, out_root)

    def pm(row: dict[str, Any], metric: str) -> str:
        mean, sd = row[f"{metric}_mean"], row[f"{metric}_sd"]
        return "N/A" if mean == "" else f"{100*float(mean):.2f} ± {100*float(sd):.2f}%"

    labels = {
        "current_argmax": "Current: p⊥ > pmax",
        "binary_0_5": "Binary: p⊥ > 0.5",
        **{
            probability_setting(target): f"Val p⊥ @ {100*target:.0f}% recall"
            for target in recall_targets
        },
        **{
            margin_setting(target): f"Val log-margin @ {100*target:.0f}% recall"
            for target in recall_targets
        },
    }
    order = {name: index for index, name in enumerate(settings(recall_targets))}
    lines = [
        "# Hierarchical, validation-calibrated unknown rejection",
        "",
        "Kato/RLD grouped CV5 × seed42. One threshold is selected per fold from "
        "pooled validation distractor levels and frozen across all held-out test "
        "levels. No model is retrained.",
        "",
        "Replicates are averaged within fold first; values are the unweighted "
        "mean ± sample SD across five biological folds. Confidence intervals for "
        "differences are two-sided paired Student-t intervals across the same folds.",
        "",
        "## Clean known-only condition (`r_dist = 0`)",
        "",
        "| Rejection rule | Known false reject ↓ | Reject-aware Top-1 ↑ |",
        "|---|---:|---:|",
    ]
    clean_rows = [row for row in summary if float(row["severity"]) == 0.0]
    for row in sorted(clean_rows, key=lambda item: order[item["setting"]]):
        lines.append(
            f"| {labels[row['setting']]} | {pm(row, 'known_false_reject_rate')} | "
            f"{pm(row, 'reject_aware_top1')} |"
        )
    clean_paired = next(
        (
            row
            for row in paired
            if row["setting"] == margin_setting(0.80) and row["scope"] == "clean"
        ),
        None,
    )
    if clean_paired is not None:
        clean_default = next(row for row in clean_rows if row["setting"] == "current_argmax")
        clean_candidate = next(
            row for row in clean_rows if row["setting"] == margin_setting(0.80)
        )
        lines.extend(
            [
                "",
                "On clean data, the 80%-targeted log-margin rule reduced known "
                f"false rejection from {100*float(clean_default['known_false_reject_rate_mean']):.2f}% "
                f"to {100*float(clean_candidate['known_false_reject_rate_mean']):.2f}%, "
                "a paired reduction of "
                f"{clean_paired['known_false_reject_reduction_pp']:.2f} pp "
                f"(95% CI [{clean_paired['known_false_reject_reduction_ci_low_pp']:.2f}, "
                f"{clean_paired['known_false_reject_reduction_ci_high_pp']:.2f}]).",
            ]
        )
    lines.extend(
        [
            "",
            "## Unknown-present conditions (pooled `r_dist = 0.1–0.5`)",
            "",
            "| Rejection rule | Unknown recall ↑ | Known false reject ↓ | Reject-aware Top-1 ↑ | Real-only Top-1 ↑ |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(operating, key=lambda item: order[item["setting"]]):
        lines.append(
            f"| {labels[row['setting']]} | {pm(row, 'unknown_recall')} | "
            f"{pm(row, 'known_false_reject_rate')} | {pm(row, 'reject_aware_top1')} | "
            f"{pm(row, 'known_top1_real')} |"
        )

    primary_paired = next(
        (
            row
            for row in paired
            if row["setting"] == margin_setting(0.90)
            and row["scope"] == "pooled_nonzero"
        ),
        None,
    )
    if primary_paired is not None:
        lines.extend(
            [
                "",
                "### Paired effect of the 90%-targeted log-margin rule",
                "",
                "Relative to the default rule, it reduced known false rejection by "
                f"{primary_paired['known_false_reject_reduction_pp']:.2f} pp "
                f"(95% CI [{primary_paired['known_false_reject_reduction_ci_low_pp']:.2f}, "
                f"{primary_paired['known_false_reject_reduction_ci_high_pp']:.2f}]) "
                "and increased reject-aware Top-1 by "
                f"{primary_paired['delta_reject_aware_top1_pp']:.2f} pp "
                f"(95% CI [{primary_paired['delta_reject_aware_top1_ci_low_pp']:.2f}, "
                f"{primary_paired['delta_reject_aware_top1_ci_high_pp']:.2f}]).",
                "",
                "## Per-severity unknown-present results",
                "",
                "| Rejection rule | Distractor fraction | Unknown recall ↑ | Known false reject ↓ | Reject-aware Top-1 ↑ | Real-only Top-1 ↑ |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
    detailed_rows = [row for row in summary if float(row["severity"]) > 0.0]
    for row in sorted(
        detailed_rows,
        key=lambda item: (float(item["severity"]), order[item["setting"]]),
    ):
        lines.append(
            f"| {labels[row['setting']]} | {float(row['severity']):.2f} | "
            f"{pm(row, 'unknown_recall')} | {pm(row, 'known_false_reject_rate')} | "
            f"{pm(row, 'reject_aware_top1')} | {pm(row, 'known_top1_real')} |"
        )
    lines.extend(
        [
            "",
            "For every non-rejected neuron, identity is decoded as the highest-"
            "probability real candidate. `p⊥` thresholds answer known-vs-unknown "
            "independently of the number of real identity classes. The log-margin "
            "rules are reported only as a robustness check.",
            "",
            "The operating curves are descriptive held-out-test curves. Every marked "
            "validation operating point was selected without access to test labels.",
        ]
    )
    primary = [
        row
        for row in conversion
        if row["setting"] == probability_setting(0.80)
        and float(row["severity"]) == 0.50
    ]
    robust = [
        row
        for row in conversion
        if row["setting"] == margin_setting(0.80)
        and float(row["severity"]) == 0.50
    ]
    if primary and robust:
        p_row, m_row = primary[0], robust[0]
        lines.extend(
            [
                "",
                "At 50% distractors, the capacity-dustbin ranking gain over forced "
                f"no-dustbin matching is {100*float(m_row['ranking_gain']):.2f} points. "
                f"The 80%-recall probability rule costs {100*float(p_row['rejection_cost']):.2f} "
                f"points, whereas the log-margin robustness rule costs "
                f"{100*float(m_row['rejection_cost']):.2f} points and retains a "
                f"{100*float(m_row['net_gain_over_no_dustbin_forced']):+.2f}-point net gain.",
            ]
        )
    (out_root / "TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    audit = {
        "protocol": "RLD grouped CV5 x seed42 hierarchical rejection calibration",
        "raw_schema_version": RAW_SCHEMA_VERSION,
        "positive_class": "materialized __OUTLIER_* synthetic distractors",
        "known_class": "unique supervised identity present in current atlas reference",
        "excluded_targets": "native unlabeled/uncertain/duplicate/reference-absent nodes",
        "calibration_split": "validation only, with independently materialized distractors",
        "calibration_levels_pooled_per_fold": True,
        "one_threshold_per_fold_score_and_recall_target": True,
        "thresholds_frozen_across_test_severities": True,
        "test_labels_used_for_threshold_selection": False,
        "checkpoint_selection_unchanged": True,
        "retrained": False,
        "fold_first_aggregation": True,
        "paired_ci": "two-sided 95% Student-t interval across biological folds",
        "paired_effect_direction": {
            "reject_aware_top1": "candidate minus current default",
            "known_false_reject": "current default minus candidate",
        },
        "recall_targets": list(recall_targets),
        "settings": {
            "current_argmax": "p(dustbin) > max real probability",
            "binary_0_5": "p(dustbin) > 0.5",
            "probability_val": "p(dustbin) > fold-specific validation threshold",
            "log_margin_val": (
                "log p(dustbin) - log(max real probability) > fold-specific validation threshold"
            ),
        },
        "figure_written": figure_written,
        "cells": len(cells),
        "summary_rows": len(summary),
    }
    write_json(out_root / "AUDIT.json", audit)


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split(",") if value.strip())


def parse_ints(text: str) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(",") if value.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CORR / "MANIFEST.json")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--legacy-raw", type=Path, default=LEGACY_RAW)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--severities", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--calibration-severities", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--calibration-seeds", default="0,1,2")
    parser.add_argument("--recall-targets", default="0.8,0.9,0.95")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    folds = parse_ints(args.folds)
    severities = set(parse_floats(args.severities))
    calibration_severities = parse_floats(args.calibration_severities)
    calibration_seeds = parse_ints(args.calibration_seeds)
    recall_targets = parse_floats(args.recall_targets)
    if not folds or not calibration_severities or not calibration_seeds:
        parser.error("folds and calibration grids must be non-empty")
    if any(not 0.0 < target <= 1.0 for target in recall_targets):
        parser.error("--recall-targets values must be in (0, 1]")

    sys.path.insert(0, str(PACKAGE.resolve()))
    from mprt_net.data import WormCache, split_files
    from mprt_net.evaluate import load_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = read_json(args.manifest)
    cells: list[dict[str, Any]] = []
    calibrations: list[dict[str, Any]] = []
    raw_by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for fold in folds:
        ckpt = checkpoint(fold)
        model, state = load_checkpoint(ckpt, device)
        model.eval()
        raw_mapping = state.get("atlas_identity_to_slot")
        if not isinstance(raw_mapping, dict) or not raw_mapping:
            raise RuntimeError(f"missing atlas_identity_to_slot: {ckpt}")
        identity_to_slot = {str(key): int(value) for key, value in raw_mapping.items()}
        fold_conditions = conditions(manifest, fold)
        if not fold_conditions:
            raise RuntimeError(f"fold{fold}: no distractor conditions")
        source_fold = Path(fold_conditions[0]["source_fold"])
        cache = WormCache(activity_length=512, max_items=64)
        calibration, _ = calibrate_fold(
            model=model,
            identity_to_slot=identity_to_slot,
            cache=cache,
            validation_paths=split_files(source_fold, "val"),
            fold=fold,
            recall_targets=recall_targets,
            calibration_severities=calibration_severities,
            calibration_seeds=calibration_seeds,
            output=args.output,
            force=args.force,
        )
        calibrations.append(calibration)
        clean = saved_clean(fold)
        seen_clean = False
        for condition in fold_conditions:
            severity = float(condition["severity"])
            if severity not in severities:
                continue
            perturbation_seed = int(condition["perturbation_seed"])
            condition_name = Path(condition["root"]).name
            cached = (
                args.legacy_raw
                / "cells"
                / f"fold{fold}"
                / condition_name
                / "neurid_original"
                / "queries.csv.gz"
            )
            if cached.is_file():
                raw_rows = normalize_cached_rows(read_gzip_csv(cached))
            else:
                raw_rows = infer_queries(
                    model=model,
                    identity_to_slot=identity_to_slot,
                    cache=cache,
                    paths=split_files(Path(condition["root"]), "test"),
                    fold=fold,
                    severity=severity,
                    perturbation_seed=perturbation_seed,
                )
            condition_out = args.output / "cells" / f"fold{fold}" / condition_name
            results = evaluate_condition(
                raw_rows=raw_rows,
                fold=fold,
                severity=severity,
                perturbation_seed=perturbation_seed,
                recordings=len(condition["files"]),
                calibration=calibration,
                recall_targets=recall_targets,
                out_dir=condition_out,
            )
            if severity > 0.0:
                raw_by_fold[fold].extend(raw_rows)
            for result in results:
                if severity == 0.0 and result["setting"] == "current_argmax":
                    if abs(float(result["known_top1_real"]) - float(clean["top1_real"])) > 1e-12:
                        raise RuntimeError(
                            f"fold{fold} clean Top-1 mismatch: "
                            f"{result['known_top1_real']} != {clean['top1_real']}"
                        )
                    seen_clean = True
                item = dict(result)
                item["execution_device"] = str(device)
                item["checkpoint"] = str(ckpt.resolve())
                item["corruption_root"] = str(Path(condition["root"]).resolve())
                item["raw_probabilities_reused_from"] = (
                    str(cached.resolve()) if cached.is_file() else ""
                )
                cells.append(item)
            if severity > 0.0 and not seen_clean:
                raise RuntimeError(f"fold{fold}: nonzero severity before clean guard")
            print(
                f"[DONE] fold={fold} severity={severity:.2f} p={perturbation_seed}",
                flush=True,
            )
    aggregate(
        cells,
        calibrations,
        raw_by_fold,
        recall_targets,
        args.output,
        args.legacy_raw / "summary.csv",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
