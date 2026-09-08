#!/usr/bin/env python3
"""Prepare leakage-free position-only or multimodal LOSO zm9624 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


POSITION_FILE = "shifted_id_2_neuron_pos_acrs_t_result.npy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-worm-dir", required=True, type=Path)
    parser.add_argument("--test-worm-dir", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--window-length", type=int, default=500)
    parser.add_argument("--train-windows", type=int, default=9)
    parser.add_argument("--val-windows", type=int, default=3)
    parser.add_argument(
        "--use-activity",
        action="store_true",
        help="Store the matching calcium segment instead of a constant zero tensor.",
    )
    parser.add_argument(
        "--activity-normalization",
        choices=("none", "per_trace_zscore"),
        default="none",
        help="Normalization fitted independently inside every temporal segment.",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Do not read or materialize any held-out-worm test data.",
    )
    return parser.parse_args()


def load_positions(worm_dir: Path) -> Tuple[np.ndarray, List[int]]:
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
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"Expected positions [N,T,3], got {positions.shape}")
    return positions, shifted_ids


def summarize_segment(
    positions: np.ndarray, start: int, end: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    segment = positions[:, start:end].copy()
    observed = np.isfinite(segment).all(axis=2)
    frame_centers = np.nanmedian(segment, axis=0)
    segment -= frame_centers[None]
    with np.errstate(all="ignore"):
        xyz = np.nanmedian(segment, axis=1).astype(np.float32)
    valid = np.isfinite(xyz).all(axis=1)
    counts = observed.sum(axis=1)
    audit = {
        "start": start,
        "end_exclusive": end,
        "frames": end - start,
        "valid_neurons": int(valid.sum()),
        "minimum_observed_frames_among_valid": int(counts[valid].min()),
        "median_observed_frames_among_valid": float(np.median(counts[valid])),
        "mean_observed_fraction": float(observed.mean()),
    }
    return xyz, valid, audit


def write_sample(
    path: Path,
    worm_name: str,
    positions: np.ndarray,
    shifted_ids: List[int],
    start: int,
    end: int,
    activity_source: np.ndarray | None = None,
    activity_normalization: str = "none",
) -> Dict[str, float]:
    xyz, valid, audit = summarize_segment(positions, start, end)
    num_neurons = len(shifted_ids)
    # The verified MPRT data contract requires activity_raw. Position-only
    # preparation uses an explicit constant; multimodal preparation stores the
    # temporally matching calcium window.
    if activity_source is None:
        activity = np.zeros((num_neurons, 2), dtype=np.float32)
    else:
        activity = np.asarray(activity_source[:, start:end], dtype=np.float32)
        if activity_normalization == "per_trace_zscore":
            center = activity.mean(axis=1, keepdims=True)
            scale = activity.std(axis=1, keepdims=True)
            activity = (activity - center) / np.maximum(scale, 1e-6)
    labels = np.asarray([str(value) for value in shifted_ids])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        activity_raw=activity,
        xyz=xyz,
        cell_id=labels,
        labeled_mask=np.ones(num_neurons, dtype=bool),
        certain_mask=np.ones(num_neurons, dtype=bool),
        clean_mask=np.ones(num_neurons, dtype=bool),
        valid_xyz_mask=valid,
        recording_uid=np.asarray(f"{worm_name}:{start}:{end}"),
        segment_start=np.asarray(start),
        segment_end_exclusive=np.asarray(end),
        source_worm=np.asarray(worm_name),
    )
    audit["file"] = str(path.resolve())
    return audit


def main() -> None:
    args = parse_args()
    if args.window_length < 2:
        raise ValueError("--window-length must be at least 2")
    if not args.train_only and args.test_worm_dir is None:
        raise ValueError("--test-worm-dir is required unless --train-only is set")
    train_positions, train_ids = load_positions(args.train_worm_dir)
    test_positions = test_ids = None
    if not args.train_only:
        test_positions, test_ids = load_positions(args.test_worm_dir)
    train_activity = test_activity = None
    if args.use_activity:
        train_activity = np.load(
            args.train_worm_dir / "calcium_intensity.npy"
        ).astype(np.float32, copy=False)
        if not args.train_only:
            test_activity = np.load(
                args.test_worm_dir / "calcium_intensity.npy"
            ).astype(np.float32, copy=False)
        if train_activity.shape != train_positions.shape[:2]:
            raise ValueError("Training activity and position shapes disagree")
        if not args.train_only and test_activity.shape != test_positions.shape[:2]:
            raise ValueError("Test activity and position shapes disagree")
        if not np.isfinite(train_activity).all():
            raise ValueError("Training activity contains non-finite values")
        if not args.train_only and not np.isfinite(test_activity).all():
            raise ValueError("Activity contains non-finite values")
    required = args.window_length * (args.train_windows + args.val_windows)
    if required > train_positions.shape[1]:
        raise ValueError(
            f"Requested {required} train/val frames, only {train_positions.shape[1]} exist"
        )

    records = {"train": [], "val": [], "test": []}
    for index in range(args.train_windows + args.val_windows):
        split = "train" if index < args.train_windows else "val"
        start = index * args.window_length
        end = start + args.window_length
        records[split].append(
            write_sample(
                args.output_root / split / f"segment_{index:02d}.npz",
                args.train_worm_dir.name,
                train_positions,
                train_ids,
                start,
                end,
                train_activity,
                args.activity_normalization,
            )
        )

    if not args.train_only:
        test_length = test_positions.shape[1] // 2
        test_ranges = [
            (0, test_length),
            (test_positions.shape[1] - test_length, test_positions.shape[1]),
        ]
        for name, (start, end) in zip(("before", "after"), test_ranges):
            records["test"].append(
                write_sample(
                    args.output_root / "test" / f"{name}.npz",
                    args.test_worm_dir.name,
                    test_positions,
                    test_ids,
                    start,
                    end,
                    test_activity,
                    args.activity_normalization,
                )
            )

    manifest = {
        "protocol": (
            "multimodal_leave_one_worm_out_temporal_segments_v1"
            if args.use_activity
            else "position_only_leave_one_worm_out_temporal_segments_v1"
        ),
        "train_worm": args.train_worm_dir.name,
        "test_worm": None if args.test_worm_dir is None else args.test_worm_dir.name,
        "test_materialized": not args.train_only,
        "test_identity_labels_available_only_for_post_training_evaluation": True,
        "activity_input": (
            "calcium_intensity.npy segment"
            if args.use_activity
            else "constant zeros; ignored by geometry_only MPRT"
        ),
        "activity_normalization": args.activity_normalization,
        "position_preprocessing": (
            "per-frame population median subtraction followed by per-neuron "
            "temporal median within each segment"
        ),
        "window_length": args.window_length,
        "records": records,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

