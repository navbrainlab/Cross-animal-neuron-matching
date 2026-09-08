#!/usr/bin/env python3
"""Training-free Euclidean baseline using the exact MPRT evaluation targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from mprt_net.data import PairIndex, WormCache, build_pair_targets
from mprt_net.relations import standardize_xyz


def directional(scores: torch.Tensor, targets: torch.Tensor) -> dict[str, float | int]:
    valid = (targets >= 0) & (targets < scores.shape[1])
    if not bool(valid.any()):
        return {"queries": 0, "top1": 0, "top5": 0, "rr": 0.0}
    scores = scores[valid]
    targets = targets[valid]
    target_scores = scores.gather(1, targets[:, None])
    rank = 1 + (scores > target_scores).sum(dim=1)
    return {
        "queries": int(rank.numel()),
        "top1": int((rank <= 1).sum()),
        "top5": int((rank <= 5).sum()),
        "rr": float((1.0 / rank.float()).sum()),
    }


def hungarian(scores: torch.Tensor, row_target: torch.Tensor, col_target: torch.Tensor):
    rows, cols = linear_sum_assignment(-scores.detach().cpu().numpy())
    assignment = {int(row): int(col) for row, col in zip(rows, cols)}
    inverse = {col: row for row, col in assignment.items()}
    num_a, num_b = scores.shape
    correct = total = 0
    for row, target in enumerate(row_target.tolist()):
        if 0 <= target < num_b:
            total += 1
            correct += int(assignment.get(row, -1) == int(target))
    for col, target in enumerate(col_target.tolist()):
        if 0 <= target < num_a:
            total += 1
            correct += int(inverse.get(col, -1) == int(target))
    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--activity-length", type=int, default=128)
    parser.add_argument("--min-shared", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pair-output", type=Path, default=None)
    args = parser.parse_args()

    index = PairIndex(args.dataset_root, args.split, min_shared=args.min_shared)
    cache = WormCache(activity_length=args.activity_length)
    totals = {"queries": 0, "top1": 0, "top5": 0, "rr": 0.0, "hc": 0, "hq": 0}
    pair_rows = []

    for path_a, path_b in index.pairs:
        sample_a, sample_b, targets = build_pair_targets(cache.get(path_a), cache.get(path_b))
        xyz_a = standardize_xyz(sample_a.xyz)
        xyz_b = standardize_xyz(sample_b.xyz)
        scores = -torch.cdist(xyz_a, xyz_b).square()
        left = directional(scores, targets.row_target)
        right = directional(scores.transpose(0, 1), targets.col_target)
        hc, hq = hungarian(scores, targets.row_target, targets.col_target)
        pair = {
            "uid_a": sample_a.uid,
            "uid_b": sample_b.uid,
            "queries": int(left["queries"] + right["queries"]),
            "top1_correct": int(left["top1"] + right["top1"]),
            "top5_correct": int(left["top5"] + right["top5"]),
            "reciprocal_rank_sum": float(left["rr"] + right["rr"]),
            "hungarian_correct": hc,
            "hungarian_queries": hq,
        }
        pair_rows.append(pair)
        totals["queries"] += pair["queries"]
        totals["top1"] += pair["top1_correct"]
        totals["top5"] += pair["top5_correct"]
        totals["rr"] += pair["reciprocal_rank_sum"]
        totals["hc"] += hc
        totals["hq"] += hq

    q = max(int(totals["queries"]), 1)
    hq = max(int(totals["hq"]), 1)
    result = {
        "method": "Euclidean on independently standardized XYZ",
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "pairs": len(pair_rows),
        "queries": int(totals["queries"]),
        "top1_real": float(totals["top1"] / q),
        "top5_real": float(totals["top5"] / q),
        "mrr_real": float(totals["rr"] / q),
        "hungarian_queries": int(totals["hq"]),
        "hungarian_accuracy": float(totals["hc"] / hq),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.pair_output:
        args.pair_output.parent.mkdir(parents=True, exist_ok=True)
        with args.pair_output.open("w", encoding="utf-8") as handle:
            for row in pair_rows:
                handle.write(json.dumps(row) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
