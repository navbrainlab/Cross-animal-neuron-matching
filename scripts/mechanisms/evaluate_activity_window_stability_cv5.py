#!/usr/bin/env python3
"""Evaluate frozen NeurID atlas matching under shorter query activity windows.

The population atlas, model weights, query geometry and evaluation identities
are fixed.  Only the held-out query animal's visible activity interval changes.
Each interval is resampled to the checkpoint's 512-sample input convention.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO / "mprt_net_v1_1"
DATA_ROOTS = {
    "atanas": REPO / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": REPO / "Data/Dunn_001623/cv5_grouped_v1",
}
DEFAULT_CHECKPOINT_ROOT = REPO / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
DEFAULT_REFERENCE_ROOTS = {
    "atanas": REPO / "runs/mprt_v1_1_atlas_medoid_atanas_cv5_test_seeds_42_v1",
    "rld": REPO / "runs/mprt_v1_1_atlas_medoid_rld_cv5_test_seeds_42_v1",
}
DEFAULT_OUTPUT = REPO / "runs/neurid_activity_window_stability_cv5_seed42_v1"
POSITIONS = ("start", "middle", "end")
POSITION_FRACTIONS = {"start": 0.0, "middle": 0.5, "end": 1.0}


@dataclass
class Totals:
    queries: int = 0
    top1: int = 0
    top5: int = 0
    reciprocal_rank_sum: float = 0.0
    hungarian_queries: int = 0
    hungarian_correct: int = 0

    def add(self, other: "Totals") -> None:
        self.queries += other.queries
        self.top1 += other.top1
        self.top5 += other.top5
        self.reciprocal_rank_sum += other.reciprocal_rank_sum
        self.hungarian_queries += other.hungarian_queries
        self.hungarian_correct += other.hungarian_correct

    def metrics(self) -> dict[str, float | int]:
        if self.queries <= 0:
            raise RuntimeError("No evaluable queries")
        if self.hungarian_queries <= 0:
            raise RuntimeError("No evaluable Hungarian queries")
        return {
            "queries": self.queries,
            "top1_real": self.top1 / self.queries,
            "top5_real": self.top5 / self.queries,
            "mrr_real": self.reciprocal_rank_sum / self.queries,
            "hungarian_queries": self.hungarian_queries,
            "hungarian_accuracy": self.hungarian_correct / self.hungarian_queries,
        }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_csv_gz(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def parse_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("durations/folds must be unique")
    return result


def parse_folds(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("expected non-negative comma-separated integers")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("folds must be unique")
    return result


def timestamps_and_rate(path: Path, expected_frames: int) -> tuple[np.ndarray, float]:
    with np.load(path, allow_pickle=False) as data:
        rate = float(np.asarray(data["sampling_rate_hz"]).reshape(-1)[0])
        timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
    if timestamps.shape != (expected_frames,) or not np.all(np.diff(timestamps) > 0):
        timestamps = np.arange(expected_frames, dtype=np.float64) / rate
    timestamps = timestamps - timestamps[0]
    return timestamps, rate


def select_window(
    timestamps: np.ndarray, duration_seconds: int, position: str
) -> tuple[int, int, float, float]:
    """Return a half-open interval with approximately the requested duration."""

    # Median spacing lets the final sample contribute one frame interval.
    dt = float(np.median(np.diff(timestamps)))
    available = float(timestamps[-1] + dt)
    if duration_seconds > available + 1e-6:
        raise ValueError(
            f"requested {duration_seconds}s exceeds available duration {available:.3f}s"
        )
    slack = max(available - duration_seconds, 0.0)
    requested_start = POSITION_FRACTIONS[position] * slack
    lo = int(np.searchsorted(timestamps, requested_start, side="left"))
    hi = int(np.searchsorted(timestamps, requested_start + duration_seconds, side="left"))
    hi = min(max(hi, lo + 2), len(timestamps))
    if hi - lo < 2:
        raise RuntimeError("activity window contains fewer than two frames")
    actual_start = float(timestamps[lo])
    actual_duration = float(timestamps[hi - 1] - timestamps[lo] + dt)
    return lo, hi, actual_start, actual_duration


def targets(sample: Any, identity_to_slot: dict[str, int]) -> tuple[torch.Tensor, dict[int, str]]:
    from mprt_net.data import unique_identity_map

    target = torch.full((sample.num_nodes,), -1, dtype=torch.long, device=sample.xyz.device)
    identities: dict[int, str] = {}
    for identity, node_index in unique_identity_map(sample).items():
        slot = identity_to_slot.get(str(identity))
        if slot is not None:
            target[int(node_index)] = int(slot)
            identities[int(node_index)] = str(identity)
    return target, identities


def evaluate_output(
    output: Any,
    target: torch.Tensor,
    identities: dict[int, str],
    slot_to_identity: list[str],
) -> tuple[Totals, list[dict[str, Any]]]:
    from scipy.optimize import linear_sum_assignment

    probabilities = output.row_conditional.detach()
    real = probabilities[:, :-1]
    valid = (target >= 0) & (target < real.shape[1])
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if indices.numel() == 0:
        return Totals(), []

    selected_real = real.index_select(0, indices)
    selected_target = target.index_select(0, indices)
    target_score = selected_real.gather(1, selected_target[:, None])
    ranks = 1 + (selected_real > target_score).sum(dim=1)
    predictions = selected_real.argmax(dim=1)

    plan = output.plan[:-1, :-1].detach().cpu().numpy()
    row_index, column_index = linear_sum_assignment(-plan)
    assignment = {int(row): int(column) for row, column in zip(row_index, column_index)}

    index_values = indices.detach().cpu().tolist()
    target_values = selected_target.detach().cpu().tolist()
    prediction_values = predictions.detach().cpu().tolist()
    rank_values = ranks.detach().cpu().tolist()
    target_probability = target_score.detach().cpu().flatten().tolist()
    rows: list[dict[str, Any]] = []
    hungarian_correct = 0
    for offset, node_index in enumerate(index_values):
        target_slot = int(target_values[offset])
        prediction_slot = int(prediction_values[offset])
        hungarian_slot = assignment.get(int(node_index), -1)
        hungarian_correct += int(hungarian_slot == target_slot)
        rows.append(
            {
                "node_index": int(node_index),
                "identity": identities[int(node_index)],
                "target_slot": target_slot,
                "prediction_slot": prediction_slot,
                "prediction_identity": slot_to_identity[prediction_slot],
                "rank": int(rank_values[offset]),
                "correct": int(rank_values[offset] == 1),
                "target_probability": float(target_probability[offset]),
                "hungarian_prediction_slot": hungarian_slot,
                "hungarian_correct": int(hungarian_slot == target_slot),
            }
        )
    count = len(rows)
    return (
        Totals(
            queries=count,
            top1=sum(row["correct"] for row in rows),
            top5=sum(row["rank"] <= 5 for row in rows),
            reciprocal_rank_sum=sum(1.0 / row["rank"] for row in rows),
            hungarian_queries=count,
            hungarian_correct=hungarian_correct,
        ),
        rows,
    )


def reference_full_metrics(dataset: str, fold: int) -> dict[str, Any]:
    path = DEFAULT_REFERENCE_ROOTS[dataset] / f"fold{fold}" / "seed42" / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload["full_atlas"]
    return {
        "path": str(path.resolve()),
        "queries": int(result["queries"]),
        "top1_real": float(result["top1_real"]),
        "top5_real": float(result["top5_real"]),
        "mrr_real": float(result["mrr_real"]),
        "hungarian_queries": int(result["hungarian_queries"]),
        "hungarian_accuracy": float(result["hungarian_accuracy"]),
    }


def validate_reference(observed: dict[str, Any], reference: dict[str, Any]) -> None:
    for key in ("queries", "hungarian_queries"):
        if int(observed[key]) != int(reference[key]):
            raise RuntimeError(f"full-record reproduction failed for {key}: {observed[key]} != {reference[key]}")
    for key in ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy"):
        if not math.isclose(float(observed[key]), float(reference[key]), rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(f"full-record reproduction failed for {key}: {observed[key]} != {reference[key]}")


def condition_specs(durations: list[int]) -> list[tuple[str, int | None, str]]:
    specs = [(f"{duration}s", duration, position) for duration in durations for position in POSITIONS]
    specs.append(("full", None, "full"))
    return specs


def evaluate_fold(
    *, dataset: str, fold: int, seed: int, durations: list[int], device: torch.device,
    checkpoint_root: Path, output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    from mprt_net.data import WormSample, _resample_activity, load_worm, split_files
    from mprt_net.evaluate import load_checkpoint

    dataset_root = DATA_ROOTS[dataset] / f"fold_{fold}"
    checkpoint_path = checkpoint_root / dataset / f"fold{fold}" / f"seed{seed}" / "static_atlas/anchored_pure.pt"
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    model.eval()
    if not model.atlas_is_initialized:
        raise RuntimeError(f"static atlas is not initialized: {checkpoint_path}")
    identity_to_slot = {str(key): int(value) for key, value in checkpoint["atlas_identity_to_slot"].items()}
    slot_to_identity = [""] * len(identity_to_slot)
    for identity, slot in identity_to_slot.items():
        slot_to_identity[slot] = identity
    atlas = model.atlas_encoding()
    activity_length = int(checkpoint["train_args"]["activity_length"])
    files = split_files(dataset_root, "test")
    specs = condition_specs(durations)
    condition_totals = {spec: Totals() for spec in specs}
    query_rows: list[dict[str, Any]] = []
    prediction_map: dict[tuple[str, int, str], dict[tuple[str, int, str], int]] = {
        spec: {} for spec in specs
    }
    duration_audit: list[dict[str, Any]] = []

    with torch.inference_mode():
        for animal_number, path in enumerate(files, start=1):
            raw = load_worm(path, activity_length=None)
            timestamps, sampling_rate = timestamps_and_rate(path, raw.activity.shape[1])
            target, identities = targets(raw.to(device), identity_to_slot)
            for duration_key, duration_seconds, position in specs:
                if duration_seconds is None:
                    lo, hi = 0, raw.activity.shape[1]
                    dt = float(np.median(np.diff(timestamps)))
                    actual_start = 0.0
                    actual_duration = float(timestamps[-1] + dt)
                else:
                    lo, hi, actual_start, actual_duration = select_window(
                        timestamps, duration_seconds, position
                    )
                activity = _resample_activity(raw.activity[:, lo:hi], activity_length)
                windowed = WormSample(
                    uid=raw.uid,
                    xyz=raw.xyz,
                    activity=activity,
                    cell_ids=raw.cell_ids,
                    supervised_mask=raw.supervised_mask,
                    source_path=raw.source_path,
                ).to(device)
                query = model.encode_population(windowed)
                output = model.match_encodings(query, atlas)
                totals, rows = evaluate_output(output, target, identities, slot_to_identity)
                condition = (duration_key, duration_seconds, position)
                condition_totals[condition].add(totals)
                duration_audit.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "test_animal_id": raw.uid,
                        "duration_key": duration_key,
                        "position": position,
                        "requested_duration_seconds": "" if duration_seconds is None else duration_seconds,
                        "actual_start_seconds": actual_start,
                        "actual_duration_seconds": actual_duration,
                        "raw_frames": raw.activity.shape[1],
                        "window_frames": hi - lo,
                        "sampling_rate_hz": sampling_rate,
                    }
                )
                for row in rows:
                    key = (raw.uid, int(row["node_index"]), str(row["identity"]))
                    prediction_map[condition][key] = int(row["prediction_slot"])
                    query_rows.append(
                        {
                            "dataset": dataset,
                            "fold": fold,
                            "seed": seed,
                            "test_animal_id": raw.uid,
                            "duration_key": duration_key,
                            "position": position,
                            "requested_duration_seconds": "" if duration_seconds is None else duration_seconds,
                            "window_start_seconds": actual_start,
                            "window_duration_seconds": actual_duration,
                            **row,
                        }
                    )
            print(
                f"window-eval dataset={dataset} fold={fold} animal={animal_number:02d}/{len(files):02d} uid={raw.uid}",
                flush=True,
            )

    position_rows: list[dict[str, Any]] = []
    for condition in specs:
        duration_key, duration_seconds, position = condition
        position_rows.append(
            {
                "dataset": dataset,
                "fold": fold,
                "seed": seed,
                "duration_key": duration_key,
                "duration_seconds": "" if duration_seconds is None else duration_seconds,
                "position": position,
                "recordings": len(files),
                **condition_totals[condition].metrics(),
            }
        )

    full_observed = next(row for row in position_rows if row["duration_key"] == "full")
    reference = reference_full_metrics(dataset, fold)
    validate_reference(full_observed, reference)

    fold_rows: list[dict[str, Any]] = []
    for duration_key, duration_seconds in [(f"{item}s", item) for item in durations] + [("full", None)]:
        selected = [row for row in position_rows if row["duration_key"] == duration_key]
        agreements: list[float] = []
        if duration_seconds is not None:
            maps = [prediction_map[(duration_key, duration_seconds, position)] for position in POSITIONS]
            keys = set(maps[0])
            if any(set(item) != keys for item in maps[1:]):
                raise RuntimeError(f"query universe changed across positions: {dataset}/fold{fold}/{duration_key}")
            for key in keys:
                predictions = [item[key] for item in maps]
                agreements.append(
                    sum(predictions[a] == predictions[b] for a, b in combinations(range(3), 2)) / 3.0
                )
        else:
            agreements = [1.0]
        row: dict[str, Any] = {
            "dataset": dataset,
            "fold": fold,
            "seed": seed,
            "duration_key": duration_key,
            "duration_seconds": "" if duration_seconds is None else duration_seconds,
            "positions": len(selected),
            "recordings": len(files),
            "queries_per_position": int(selected[0]["queries"]),
            "prediction_pair_agreement": float(np.mean(agreements)),
            "position_top1_range": max(float(item["top1_real"]) for item in selected) - min(float(item["top1_real"]) for item in selected),
        }
        for metric in ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy"):
            row[metric] = float(np.mean([float(item[metric]) for item in selected]))
        fold_rows.append(row)

    fold_dir = output_root / dataset / f"fold{fold}"
    atomic_csv(fold_dir / "position_metrics.csv", position_rows, list(position_rows[0]))
    atomic_csv(fold_dir / "duration_metrics.csv", fold_rows, list(fold_rows[0]))
    atomic_csv(fold_dir / "window_audit.csv", duration_audit, list(duration_audit[0]))
    atomic_csv_gz(fold_dir / "query_predictions.csv.gz", query_rows, list(query_rows[0]))
    audit = {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "atlas_identities": len(identity_to_slot),
        "activity_resample_length": activity_length,
        "test_animals": len(files),
        "minimum_available_seconds": min(
            float(item["actual_duration_seconds"])
            for item in duration_audit if item["duration_key"] == "full"
        ),
        "full_record_reproduction": {"status": "exact", "reference": reference, "observed": full_observed},
    }
    atomic_json(fold_dir / "AUDIT.json", audit)
    return position_rows, fold_rows, audit


def summarize(fold_rows: list[dict[str, Any]], durations: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    threshold_result: dict[str, Any] = {}
    for dataset in ("atanas", "rld"):
        ds_rows = [row for row in fold_rows if row["dataset"] == dataset]
        full_by_fold = {int(row["fold"]): row for row in ds_rows if row["duration_key"] == "full"}
        for duration_key, duration_seconds in [(f"{item}s", item) for item in durations] + [("full", None)]:
            selected = [row for row in ds_rows if row["duration_key"] == duration_key]
            if len(selected) != 5:
                raise RuntimeError(f"expected five folds for {dataset}/{duration_key}, found {len(selected)}")
            out: dict[str, Any] = {
                "dataset": dataset,
                "duration_key": duration_key,
                "duration_seconds": "" if duration_seconds is None else duration_seconds,
                "folds": len(selected),
                "positions_per_fold": int(selected[0]["positions"]),
            }
            for metric in (
                "top1_real", "top5_real", "mrr_real", "hungarian_accuracy",
                "prediction_pair_agreement", "position_top1_range",
            ):
                mean, sd = mean_sd(float(row[metric]) for row in selected)
                out[f"{metric}_mean"] = mean
                out[f"{metric}_sd"] = sd
            ratios = [
                float(row["top1_real"]) / float(full_by_fold[int(row["fold"])]["top1_real"])
                for row in selected
            ]
            out["top1_retention_vs_full_mean"], out["top1_retention_vs_full_sd"] = mean_sd(ratios)
            summary_rows.append(out)

        dataset_summary = [row for row in summary_rows if row["dataset"] == dataset]
        thresholds: dict[str, int | None] = {}
        for threshold in (0.90, 0.95, 0.99):
            eligible = [
                int(row["duration_seconds"])
                for row in dataset_summary
                if row["duration_key"] != "full"
                and float(row["top1_retention_vs_full_mean"]) >= threshold
            ]
            thresholds[f"minimum_seconds_at_{int(threshold * 100)}pct_top1_retention"] = min(eligible) if eligible else None
        threshold_result[dataset] = thresholds
    return summary_rows, threshold_result


def readme_text() -> str:
    return """# Frozen-atlas activity-window stability experiment

This experiment changes only the amount and temporal location of activity
visible for each held-out query animal. Model weights, fold-specific
training-only population atlas, query coordinates, candidate identities and
evaluation identities remain fixed.

Windows use physical seconds because Atanas and Kato/RLD have different
sampling rates. Each short interval is independently linearly resampled to
512 samples, matching the frozen checkpoint input convention. Start, middle
and end windows measure location sensitivity without treating them as extra
cross-validation folds. Reported means/SDs use the five outer folds; metrics
within a fold are first averaged over the three positions.

`summary.csv` is the main result table. `fold_duration_metrics.csv` and
`position_metrics.csv` retain fold/position detail. Each fold directory also
contains exact query predictions (gzip CSV), physical window audits and a
full-record reproduction audit. `minimum_duration.json` reports descriptive
90/95/99% Top-1 retention thresholds relative to the full record; these are
not significance-test claims.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="atanas,rld")
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0,1,2,3,4"))
    parser.add_argument("--durations", type=parse_ints, default=parse_ints("15,30,60,120,240,480,900"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if not datasets or set(datasets).difference(DATA_ROOTS):
        parser.error(f"datasets must be selected from {sorted(DATA_ROOTS)}")
    if args.folds != [0, 1, 2, 3, 4]:
        parser.error("the locked summary protocol requires folds 0,1,2,3,4")
    if args.seed != 42:
        parser.error("the locked full-record reference and summary protocol require seed 42")
    sys.path.insert(0, str(PACKAGE_ROOT.resolve()))
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    all_position_rows: list[dict[str, Any]] = []
    all_fold_rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for dataset in datasets:
        for fold in args.folds:
            position_rows, fold_rows, audit = evaluate_fold(
                dataset=dataset,
                fold=fold,
                seed=args.seed,
                durations=args.durations,
                device=device,
                checkpoint_root=args.checkpoint_root,
                output_root=args.output,
            )
            all_position_rows.extend(position_rows)
            all_fold_rows.extend(fold_rows)
            audits.append(audit)
            atomic_csv(args.output / "position_metrics.csv", all_position_rows, list(all_position_rows[0]))
            atomic_csv(args.output / "fold_duration_metrics.csv", all_fold_rows, list(all_fold_rows[0]))

    if datasets != ["atanas", "rld"]:
        raise RuntimeError("locked main summary requires both datasets")
    summary_rows, thresholds = summarize(all_fold_rows, args.durations)
    atomic_csv(args.output / "summary.csv", summary_rows, list(summary_rows[0]))
    atomic_json(
        args.output / "minimum_duration.json",
        {
            "definition": "shortest tested duration whose mean of fold-wise Top-1/full-Top-1 ratios reaches the stated threshold",
            "thresholds": thresholds,
        },
    )
    atomic_json(
        args.output / "AUDIT.json",
        {
            "protocol": "frozen_neurid_static_atlas_query_activity_window_cv5_seed42_v1",
            "device": str(device),
            "datasets": datasets,
            "folds": args.folds,
            "seed": args.seed,
            "durations_seconds": args.durations,
            "positions": list(POSITIONS),
            "activity_resample_length": 512,
            "full_record_reproduction": "exact in every dataset/fold",
            "fold_audits": audits,
        },
    )
    (args.output / "README.md").write_text(readme_text(), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "thresholds": thresholds}, indent=2))


if __name__ == "__main__":
    main()
