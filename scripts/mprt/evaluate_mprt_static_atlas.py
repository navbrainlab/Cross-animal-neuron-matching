#!/usr/bin/env python3
"""Evaluate one final static-atlas MPRT checkpoint against one split.

This is the production-path evaluator:
    query animal -> encode_population(query) -> match_encodings(query, static_atlas)
It intentionally does NOT use the legacy pairwise test-vs-test evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class Totals:
    queries: int = 0
    top1: int = 0
    top5: int = 0
    reciprocal_rank_sum: float = 0.0
    top1_with_dustbin: int = 0
    dustbin_top1: int = 0
    hungarian_queries: int = 0
    hungarian_correct: int = 0

    def metrics(self) -> dict[str, float | int]:
        q = max(self.queries, 1)
        hq = max(self.hungarian_queries, 1)
        return {
            "queries": self.queries,
            "top1_real": self.top1 / q,
            "top5_real": self.top5 / q,
            "mrr_real": self.reciprocal_rank_sum / q,
            "top1_with_dustbin": self.top1_with_dustbin / q,
            "dustbin_top1_rate": self.dustbin_top1 / q,
            "hungarian_queries": self.hungarian_queries,
            "hungarian_accuracy": self.hungarian_correct / hq,
        }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _targets(sample: Any, identity_to_slot: dict[str, int]) -> tuple[torch.Tensor, dict[int, str]]:
    from mprt_net.data import unique_identity_map

    target = torch.full((sample.num_nodes,), -1, dtype=torch.long, device=sample.xyz.device)
    identities: dict[int, str] = {}
    for identity, node_index in unique_identity_map(sample).items():
        slot = identity_to_slot.get(str(identity))
        if slot is None:
            continue
        target[int(node_index)] = int(slot)
        identities[int(node_index)] = str(identity)
    return target, identities


def _evaluate_rows(output: Any, target: torch.Tensor, identities: dict[int, str], uid: str, source_path: str):
    probabilities = output.row_conditional.detach()
    real = probabilities[:, :-1]
    valid = (target >= 0) & (target < real.shape[1])
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    totals = Totals()
    records: list[dict[str, Any]] = []
    if indices.numel() == 0:
        return totals, records

    selected_real = real.index_select(0, indices)
    selected_all = probabilities.index_select(0, indices)
    selected_target = target.index_select(0, indices)
    target_score = selected_real.gather(1, selected_target[:, None])
    ranks = 1 + (selected_real > target_score).sum(dim=1)
    partial_ranks = 1 + (selected_all > target_score).sum(dim=1)
    dustbin_top = selected_all[:, -1] > selected_real.amax(dim=1)
    predictions = selected_real.argmax(dim=1)

    totals.queries = int(indices.numel())
    totals.top1 = int((ranks <= 1).sum())
    totals.top5 = int((ranks <= min(5, real.shape[1])).sum())
    totals.reciprocal_rank_sum = float((1.0 / ranks.float()).sum())
    totals.top1_with_dustbin = int((partial_ranks <= 1).sum())
    totals.dustbin_top1 = int(dustbin_top.sum())

    assignment: dict[int, int] = {}
    try:
        from scipy.optimize import linear_sum_assignment

        row_idx, col_idx = linear_sum_assignment(-output.plan[:-1, :-1].detach().cpu().numpy())
        assignment = {int(r): int(c) for r, c in zip(row_idx, col_idx)}
        target_cpu = target.detach().cpu().tolist()
        valid_cpu = valid.detach().cpu().tolist()
        for row, is_valid in enumerate(valid_cpu):
            if is_valid:
                totals.hungarian_queries += 1
                totals.hungarian_correct += int(assignment.get(row, -1) == int(target_cpu[row]))
    except ImportError:
        pass

    for offset, node_index in enumerate(indices.detach().cpu().tolist()):
        records.append(
            {
                "uid": uid,
                "source_path": source_path,
                "node_index": int(node_index),
                "identity": identities[int(node_index)],
                "target_slot": int(selected_target[offset]),
                "prediction_slot": int(predictions[offset]),
                "rank": int(ranks[offset]),
                "correct": int(ranks[offset] == 1),
                "target_probability": float(target_score[offset]),
                "dustbin_top1": int(dustbin_top[offset]),
                "hungarian_prediction_slot": assignment.get(int(node_index), -1),
                "hungarian_correct": int(assignment.get(int(node_index), -1) == int(selected_target[offset])) if assignment else -1,
            }
        )
    return totals, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(args.package_root.resolve()))
    from mprt_net.data import WormCache, split_files
    from mprt_net.evaluate import load_checkpoint

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    model.eval()
    if not getattr(model, "atlas_is_initialized", False):
        raise RuntimeError(f"Checkpoint has no initialized static atlas: {args.checkpoint}")
    raw_mapping = checkpoint.get("atlas_identity_to_slot")
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise RuntimeError(f"Checkpoint has no atlas_identity_to_slot: {args.checkpoint}")
    identity_to_slot = {str(k): int(v) for k, v in raw_mapping.items()}
    atlas = model.atlas_encoding()
    if int(atlas.nodes.shape[0]) != len(identity_to_slot):
        raise RuntimeError("Atlas tensor size and identity mapping size disagree")

    files = split_files(args.dataset_root, args.split)
    if not files:
        raise FileNotFoundError(f"No {args.split} NPZ files under {args.dataset_root}")
    cache = WormCache(activity_length=args.activity_length, max_items=8)
    total = Totals()
    records: list[dict[str, Any]] = []

    with torch.inference_mode():
        for number, path in enumerate(files, start=1):
            sample_cpu = cache.get(path)
            sample = sample_cpu.to(device)
            query = model.encode_population(sample)
            output = model.match_encodings(query, atlas)
            target, identities = _targets(sample, identity_to_slot)
            cell_total, cell_records = _evaluate_rows(
                output, target, identities, sample_cpu.uid, sample_cpu.source_path
            )
            total.queries += cell_total.queries
            total.top1 += cell_total.top1
            total.top5 += cell_total.top5
            total.reciprocal_rank_sum += cell_total.reciprocal_rank_sum
            total.top1_with_dustbin += cell_total.top1_with_dustbin
            total.dustbin_top1 += cell_total.dustbin_top1
            total.hungarian_queries += cell_total.hungarian_queries
            total.hungarian_correct += cell_total.hungarian_correct
            records.extend(cell_records)
            print(
                f"evaluate {number:03d}/{len(files):03d} uid={sample_cpu.uid} "
                f"queries={cell_total.queries}",
                flush=True,
            )

    result = {
        "dataset": args.dataset,
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "fold": args.fold,
        "seed": args.seed,
        "variant": args.variant,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "atlas_size": len(identity_to_slot),
        "recordings": len(files),
        **total.metrics(),
    }
    _write_json(args.output, result)
    args.query_output.parent.mkdir(parents=True, exist_ok=True)
    with args.query_output.open("w", encoding="utf-8", newline="") as handle:
        if records:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
