#!/usr/bin/env python3
"""Multimodal or position-only temporal-window ensemble for zm9624."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mprt_net.data import WormSample, _resample_activity
from mprt_net.evaluate import load_checkpoint


POSITION_FILE = "shifted_id_2_neuron_pos_acrs_t_result.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match before/after neuron identities using an MPRT window ensemble."
    )
    parser.add_argument("--worm-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--window-length", type=int, default=500)
    parser.add_argument(
        "--gap-windows",
        type=int,
        default=0,
        help="Drop this many boundary-adjacent windows from each half.",
    )
    parser.add_argument(
        "--activity-normalization",
        choices=("none", "per_trace_zscore"),
        default="none",
        help="Normalization fitted independently inside every inference window.",
    )
    parser.add_argument(
        "--long-range-weight",
        type=float,
        default=0.003,
        help=(
            "Small long-range transport-plan tie breaker mixed into the window ensemble "
            "before the final Hungarian solve."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_inputs(worm_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    records = np.load(worm_dir / POSITION_FILE, allow_pickle=True).item()
    shifted_ids = sorted(int(value) for value in records)
    if shifted_ids != list(range(len(shifted_ids))):
        raise ValueError("shifted IDs must be consecutive 0..N-1")
    positions = np.stack(
        [
            np.asarray(records[value]["match_pos_acrs_vol"], dtype=np.float32)
            for value in shifted_ids
        ]
    )
    activity = np.load(worm_dir / "calcium_intensity.npy").astype(
        np.float32, copy=False
    )
    if activity.shape != positions.shape[:2]:
        raise ValueError("Activity and position shapes disagree")
    if not np.isfinite(activity).all():
        raise ValueError("Activity contains non-finite values")
    return positions, activity, shifted_ids


def make_window(
    positions: np.ndarray,
    shifted_ids: Sequence[int],
    start: int,
    end: int,
    uid: str,
    activity_source: np.ndarray | None = None,
    activity_length: int = 512,
    activity_normalization: str = "none",
) -> Tuple[np.ndarray, WormSample]:
    segment = positions[:, start:end].copy()
    frame_centers = np.nanmedian(segment, axis=0)
    if not np.isfinite(frame_centers).all():
        raise ValueError(f"A frame in [{start}, {end}) has no valid positions")
    segment -= frame_centers[None]
    with np.errstate(all="ignore"):
        xyz = np.nanmedian(segment, axis=1).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=1)
    indices = np.flatnonzero(valid)
    if len(indices) < 2:
        raise ValueError(f"Fewer than two valid neurons in [{start}, {end})")
    if activity_source is None:
        window_activity = torch.zeros(
            (len(indices), activity_length), dtype=torch.float32
        )
    else:
        activity = np.asarray(activity_source[valid, start:end], dtype=np.float32)
        if activity_normalization == "per_trace_zscore":
            center = activity.mean(axis=1, keepdims=True)
            scale = activity.std(axis=1, keepdims=True)
            activity = (activity - center) / np.maximum(scale, 1e-6)
        window_activity = _resample_activity(
            torch.from_numpy(np.ascontiguousarray(activity)), activity_length
        )
    sample = WormSample(
        uid=f"{uid}:{start}:{end}",
        xyz=torch.from_numpy(np.ascontiguousarray(xyz[valid])),
        activity=window_activity,
        cell_ids=tuple(str(shifted_ids[index]) for index in indices),
        supervised_mask=torch.zeros(len(indices), dtype=torch.bool),
        source_path=uid,
    )
    return indices, sample


def normalized_log_plan(plan: np.ndarray) -> np.ndarray:
    score = np.log(np.maximum(plan, 1e-12))
    return (score - score.mean()) / max(float(score.std()), 1e-8)


def directional_metrics(score: np.ndarray) -> dict:
    n = score.shape[0]
    identity = np.arange(n)
    row_prediction = score.argmax(axis=1)
    column_prediction = score.argmax(axis=0)
    diagonal = np.diag(score)
    row_rank = 1 + (score > diagonal[:, None]).sum(axis=1)
    column_rank = 1 + (score.T > diagonal[:, None]).sum(axis=1)
    return {
        "queries": 2 * n,
        "argmax_top1_correct": int(
            (row_prediction == identity).sum() + (column_prediction == identity).sum()
        ),
        "argmax_top1_accuracy": float(
            ((row_prediction == identity).sum() + (column_prediction == identity).sum())
            / (2 * n)
        ),
        "top5": float(((row_rank <= 5).sum() + (column_rank <= 5).sum()) / (2 * n)),
        "mrr": float(
            ((1.0 / row_rank).sum() + (1.0 / column_rank).sum()) / (2 * n)
        ),
        "mean_rank": float((row_rank.sum() + column_rank.sum()) / (2 * n)),
        "median_rank": float(np.median(np.concatenate([row_rank, column_rank]))),
    }


def candidate_list(score: np.ndarray, ids: Sequence[int], row: int) -> str:
    order = np.argsort(-score[row])[:5]
    return ";".join(f"{ids[index]}:{score[row, index]:.8f}" for index in order)


def main() -> None:
    args = parse_args()
    if args.window_length < 2:
        raise ValueError("--window-length must be at least 2")
    if args.gap_windows < 0:
        raise ValueError("--gap-windows must be non-negative")
    if not 0.0 <= args.long_range_weight <= 1.0:
        raise ValueError("--long-range-weight must be in [0, 1]")
    positions, activity, shifted_ids = load_inputs(args.worm_dir)
    num_neurons, num_frames, _ = positions.shape

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if not model.config.use_geometry:
        raise ValueError("This evaluator requires a model with geometry enabled")
    model.eval()
    model_activity = activity if model.config.use_activity else None
    activity_length = int(checkpoint.get("train_args", {}).get("activity_length", 512))

    half_length = num_frames // 2
    complete_windows_per_half = half_length // args.window_length
    windows_per_half = complete_windows_per_half - args.gap_windows
    if windows_per_half < 2:
        raise ValueError("Each half must retain at least two complete windows")
    after_start = num_frames - half_length
    before_ranges = [
        (index * args.window_length, (index + 1) * args.window_length)
        for index in range(windows_per_half)
    ]
    after_ranges = [
        (
            after_start + (args.gap_windows + index) * args.window_length,
            after_start + (args.gap_windows + index + 1) * args.window_length,
        )
        for index in range(windows_per_half)
    ]
    before = [
        make_window(
            positions,
            shifted_ids,
            start,
            end,
            f"{args.worm_dir.name}:before",
            model_activity,
            activity_length,
            args.activity_normalization,
        )
        for start, end in before_ranges
    ]
    after = [
        make_window(
            positions,
            shifted_ids,
            start,
            end,
            f"{args.worm_dir.name}:after",
            model_activity,
            activity_length,
            args.activity_normalization,
        )
        for start, end in after_ranges
    ]

    score_sum = np.zeros((num_neurons, num_neurons), dtype=np.float64)
    score_count = np.zeros((num_neurons, num_neurons), dtype=np.int32)
    assignment_votes = np.zeros((num_neurons, num_neurons), dtype=np.int32)
    with torch.no_grad():
        for before_indices, before_sample in before:
            for after_indices, after_sample in after:
                output = model(before_sample.to(device), after_sample.to(device))
                plan = output.plan[:-1, :-1].detach().cpu().numpy()
                pair_score = normalized_log_plan(plan)
                score_sum[np.ix_(before_indices, after_indices)] += pair_score
                score_count[np.ix_(before_indices, after_indices)] += 1
                rows, columns = linear_sum_assignment(-plan)
                assignment_votes[before_indices[rows], after_indices[columns]] += 1

    if (score_count == 0).any():
        missing = int((score_count == 0).sum())
        raise ValueError(f"{missing} neuron pairs were never observed together")
    window_score = score_sum / score_count
    # A very small long-range component resolves unstable global assignments
    # without replacing the motion-preserving short-window evidence.
    before_long_indices, before_long = make_window(
        positions,
        shifted_ids,
        0,
        before_ranges[-1][1],
        f"{args.worm_dir.name}:before_long",
        model_activity,
        activity_length,
        args.activity_normalization,
    )
    after_long_indices, after_long = make_window(
        positions,
        shifted_ids,
        after_ranges[0][0],
        num_frames,
        f"{args.worm_dir.name}:after_long",
        model_activity,
        activity_length,
        args.activity_normalization,
    )
    if len(before_long_indices) != num_neurons or len(after_long_indices) != num_neurons:
        raise ValueError("Long-range tie breaker requires every neuron in both halves")
    with torch.no_grad():
        long_output = model(before_long.to(device), after_long.to(device))
    long_plan = long_output.plan[:-1, :-1].detach().cpu().numpy()

    def standardize(score: np.ndarray) -> np.ndarray:
        return (score - score.mean()) / max(float(score.std()), 1e-8)

    ensemble_score = (
        (1.0 - args.long_range_weight) * standardize(window_score)
        + args.long_range_weight * standardize(long_plan)
    )
    assignment_rows, assignment_columns = linear_sum_assignment(-ensemble_score)
    assignment = dict(zip(assignment_rows.tolist(), assignment_columns.tolist()))
    inverse_assignment = {column: row for row, column in assignment.items()}
    hungarian_correct = int(sum(assignment[row] == row for row in range(num_neurons)))
    metrics = directional_metrics(ensemble_score)
    metrics.update(
        {
            "hungarian_correct": hungarian_correct,
            "hungarian_queries": num_neurons,
            "hungarian_accuracy": float(hungarian_correct / num_neurons),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "ensemble_score.npy", ensemble_score)
    np.save(args.output_dir / "window_score.npy", window_score)
    np.save(args.output_dir / "long_range_plan.npy", long_plan)
    np.save(args.output_dir / "score_count.npy", score_count)
    np.save(args.output_dir / "assignment_votes.npy", assignment_votes)
    matches_path = args.output_dir / "matches.csv"
    with matches_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "before_shifted_id",
            "hungarian_after_shifted_id",
            "hungarian_correct",
            "ensemble_score",
            "assignment_votes",
            "before_to_after_top1_id",
            "after_to_before_top1_id",
            "before_to_after_top5",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        row_top1 = ensemble_score.argmax(axis=1)
        column_top1 = ensemble_score.argmax(axis=0)
        for row, shifted_id in enumerate(shifted_ids):
            column = assignment[row]
            writer.writerow(
                {
                    "before_shifted_id": shifted_id,
                    "hungarian_after_shifted_id": shifted_ids[column],
                    "hungarian_correct": int(row == column),
                    "ensemble_score": f"{ensemble_score[row, column]:.10f}",
                    "assignment_votes": int(assignment_votes[row, column]),
                    "before_to_after_top1_id": shifted_ids[int(row_top1[row])],
                    "after_to_before_top1_id": shifted_ids[int(column_top1[row])],
                    "before_to_after_top5": candidate_list(
                        ensemble_score, shifted_ids, row
                    ),
                }
            )

    summary = {
        "protocol": (
            "multimodal_mprt_temporal_window_log_score_ensemble_v1"
            if model.config.use_activity
            else "position_only_mprt_temporal_window_log_score_ensemble_v1"
        ),
        "worm": args.worm_dir.name,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
        "activity_used": bool(model.config.use_activity),
        "num_neurons": num_neurons,
        "num_frames": num_frames,
        "window_length": args.window_length,
        "activity_length": activity_length,
        "activity_normalization": args.activity_normalization,
        "gap_windows": args.gap_windows,
        "gap_frames": after_ranges[0][0] - before_ranges[-1][1],
        "windows_per_half": windows_per_half,
        "pairwise_model_matches": windows_per_half * windows_per_half,
        "long_range_weight": args.long_range_weight,
        "before_ranges": before_ranges,
        "after_ranges": after_ranges,
        "aggregation": (
            "standardize each real-valued log transport plan and average every "
            "observed neuron-pair score across all before/after window pairs; mix "
            "a small standardized long-range transport-plan tie breaker; then solve "
            "one global Hungarian assignment"
        ),
        "identity_audit": (
            "shifted IDs maintain local tracks within each half and are used as "
            "ground truth only after the ensemble score and assignment are complete"
        ),
        "metrics": metrics,
        "vote_consensus": {
            "mean_max_vote_fraction": float(
                assignment_votes.max(axis=1).mean()
                / (windows_per_half * windows_per_half)
            ),
            "median_max_vote_fraction": float(
                np.median(assignment_votes.max(axis=1))
                / (windows_per_half * windows_per_half)
            ),
        },
        "outputs": {
            "matches": str(matches_path.resolve()),
            "ensemble_score": str((args.output_dir / "ensemble_score.npy").resolve()),
            "window_score": str((args.output_dir / "window_score.npy").resolve()),
            "long_range_plan": str(
                (args.output_dir / "long_range_plan.npy").resolve()
            ),
            "score_count": str((args.output_dir / "score_count.npy").resolve()),
            "assignment_votes": str(
                (args.output_dir / "assignment_votes.npy").resolve()
            ),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

