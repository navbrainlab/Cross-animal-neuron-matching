#!/usr/bin/env python3
"""Verify node-order permutation equivariance on a held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from mprt_net.data import PairIndex, WormCache, WormSample, build_pair_targets
from mprt_net.evaluate import load_checkpoint
from mprt_net.metrics import MetricTotals, pair_metrics


def seed_for(*parts: str) -> int:
    digest = hashlib.sha256("||".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def permute(sample: WormSample, seed: int) -> WormSample:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(sample.num_nodes, generator=generator)
    ids = tuple(sample.cell_ids[i] for i in order.tolist())
    return WormSample(
        uid=sample.uid + "::permuted",
        xyz=sample.xyz[order],
        activity=sample.activity[order],
        cell_ids=ids,
        supervised_mask=sample.supervised_mask[order],
        source_path=sample.source_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--activity-length", type=int, default=128)
    parser.add_argument("--min-shared", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-count-difference", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()
    index = PairIndex(args.dataset_root, args.split, min_shared=args.min_shared)
    cache = WormCache(activity_length=args.activity_length)
    original = MetricTotals()
    shuffled = MetricTotals()

    with torch.no_grad():
        for path_a, path_b in index.pairs:
            base_a, base_b = cache.get(path_a), cache.get(path_b)
            sample_a, sample_b, targets = build_pair_targets(base_a, base_b)
            original.update(pair_metrics(model(sample_a.to(device), sample_b.to(device)), targets.to(device)))

            perm_a = permute(base_a, seed_for(str(path_a), "a"))
            perm_b = permute(base_b, seed_for(str(path_b), "b"))
            perm_a, perm_b, perm_targets = build_pair_targets(perm_a, perm_b)
            shuffled.update(
                pair_metrics(
                    model(perm_a.to(device), perm_b.to(device)),
                    perm_targets.to(device),
                )
            )

    a = original.compute()
    b = shuffled.compute()
    count_keys = (
        "queries",
        "real_top1_correct",
        "real_top5_correct",
        "hungarian_queries",
        "hungarian_correct",
    )
    raw_a = {key: int(getattr(original, key)) for key in count_keys}
    raw_b = {key: int(getattr(shuffled, key)) for key in count_keys}
    maximum = max(abs(raw_a[key] - raw_b[key]) for key in count_keys)
    passed = maximum <= args.max_count_difference
    payload = {
        "audit": "independent node-row permutation equivariance",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "pairs": len(index.pairs),
        "original": a,
        "permuted": b,
        "original_counts": raw_a,
        "permuted_counts": raw_b,
        "maximum_count_difference": maximum,
        "allowed_count_difference": args.max_count_difference,
        "pass": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
