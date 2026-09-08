from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import PairIndex, WormCache, WormSample, build_pair_targets
from .evaluate import load_checkpoint


def _direction_records(
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    target: torch.Tensor,
    source: WormSample,
    destination: WormSample,
    direction: str,
) -> list[dict[str, Any]]:
    real_a = probabilities_a[:, :-1]
    real_b = probabilities_b[:, :-1]
    if real_a.shape != real_b.shape:
        raise ValueError("Compared models produced different candidate shapes")
    valid = (target >= 0) & (target < real_a.shape[1])
    source_indices = valid.nonzero(as_tuple=False).flatten()
    if source_indices.numel() == 0:
        return []
    target_indices = target[valid]
    prediction_a = real_a[valid].argmax(dim=1)
    prediction_b = real_b[valid].argmax(dim=1)

    records: list[dict[str, Any]] = []
    for source_index, target_index, pred_a, pred_b in zip(
        source_indices.tolist(),
        target_indices.tolist(),
        prediction_a.tolist(),
        prediction_b.tolist(),
    ):
        correct_a = pred_a == target_index
        correct_b = pred_b == target_index
        records.append(
            {
                "direction": direction,
                "source_uid": source.uid,
                "destination_uid": destination.uid,
                "source_index": source_index,
                "identity": source.cell_ids[source_index],
                "target_index": target_index,
                "prediction_a_index": pred_a,
                "prediction_b_index": pred_b,
                "prediction_a_id": destination.cell_ids[pred_a],
                "prediction_b_id": destination.cell_ids[pred_b],
                "correct_a": correct_a,
                "correct_b": correct_b,
                "a_only_correct": correct_a and not correct_b,
                "b_only_correct": correct_b and not correct_a,
            }
        )
    return records


def _summarize_pair(
    uid_a: str, uid_b: str, records: list[dict[str, Any]]
) -> dict[str, Any]:
    queries = len(records)
    correct_a = sum(record["correct_a"] for record in records)
    correct_b = sum(record["correct_b"] for record in records)
    a_only = sum(record["a_only_correct"] for record in records)
    b_only = sum(record["b_only_correct"] for record in records)
    return {
        "uid_a": uid_a,
        "uid_b": uid_b,
        "queries": queries,
        "correct_a": correct_a,
        "correct_b": correct_b,
        "top1_a": correct_a / max(queries, 1),
        "top1_b": correct_b / max(queries, 1),
        "delta_a_minus_b": (correct_a - correct_b) / max(queries, 1),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "net_rescue_a": a_only - b_only,
    }


def _cluster_bootstrap(
    pair_records: list[dict[str, Any]], iterations: int, seed: int
) -> dict[str, float]:
    if not pair_records:
        raise ValueError("No pair records available for bootstrap")
    rng = np.random.default_rng(seed)
    count = len(pair_records)
    differences = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sampled = rng.integers(0, count, size=count)
        queries = sum(pair_records[index]["queries"] for index in sampled)
        correct_a = sum(pair_records[index]["correct_a"] for index in sampled)
        correct_b = sum(pair_records[index]["correct_b"] for index in sampled)
        differences[iteration] = (correct_a - correct_b) / max(queries, 1)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "iterations": iterations,
        "seed": seed,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "bootstrap_mean": float(differences.mean()),
    }


def _worm_bootstrap(
    pair_records: list[dict[str, Any]], iterations: int, seed: int
) -> dict[str, float]:
    """Resample animals, then evaluate the induced graph of animal pairs."""

    animals = sorted(
        {record["uid_a"] for record in pair_records}
        | {record["uid_b"] for record in pair_records}
    )
    lookup = {
        tuple(sorted((record["uid_a"], record["uid_b"]))): record
        for record in pair_records
    }
    if len(animals) < 2:
        raise ValueError("At least two animals are required for worm bootstrap")
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        # Very rarely all positions contain the same original animal. Resample
        # that draw because it induces no observable cross-animal pair.
        while True:
            sampled = rng.choice(animals, size=len(animals), replace=True)
            induced: list[dict[str, Any]] = []
            for left in range(len(sampled)):
                for right in range(left + 1, len(sampled)):
                    if sampled[left] == sampled[right]:
                        continue
                    key = tuple(sorted((sampled[left], sampled[right])))
                    record = lookup.get(key)
                    if record is not None:
                        induced.append(record)
            if induced:
                break
        queries = sum(record["queries"] for record in induced)
        correct_a = sum(record["correct_a"] for record in induced)
        correct_b = sum(record["correct_b"] for record in induced)
        differences[iteration] = (correct_a - correct_b) / max(queries, 1)
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "animals": len(animals),
        "iterations": iterations,
        "seed": seed,
        "ci95_low": float(lower),
        "ci95_high": float(upper),
        "bootstrap_mean": float(differences.mean()),
    }


@torch.no_grad()
def compare_models(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model_a, checkpoint_a = load_checkpoint(args.checkpoint_a, device)
    model_b, checkpoint_b = load_checkpoint(args.checkpoint_b, device)
    model_a.eval()
    model_b.eval()

    index = PairIndex(args.dataset_root, args.split, min_shared=args.min_shared)
    cache = WormCache(activity_length=args.activity_length, max_items=32)
    all_queries: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []

    for path_a, path_b in index.pairs:
        sample_a, sample_b, targets = build_pair_targets(cache.get(path_a), cache.get(path_b))
        if targets.num_direct_matches == 0:
            continue
        device_a = sample_a.to(device)
        device_b = sample_b.to(device)
        device_targets = targets.to(device)
        output_a = model_a(device_a, device_b)
        output_b = model_b(device_a, device_b)

        queries = _direction_records(
            output_a.row_conditional,
            output_b.row_conditional,
            device_targets.row_target,
            sample_a,
            sample_b,
            f"{sample_a.uid}->{sample_b.uid}",
        )
        queries.extend(
            _direction_records(
                output_a.col_conditional,
                output_b.col_conditional,
                device_targets.col_target,
                sample_b,
                sample_a,
                f"{sample_b.uid}->{sample_a.uid}",
            )
        )
        all_queries.extend(queries)
        pair_summaries.append(_summarize_pair(sample_a.uid, sample_b.uid, queries))

    total_queries = sum(record["queries"] for record in pair_summaries)
    correct_a = sum(record["correct_a"] for record in pair_summaries)
    correct_b = sum(record["correct_b"] for record in pair_summaries)
    a_only = sum(record["a_only_correct"] for record in pair_summaries)
    b_only = sum(record["b_only_correct"] for record in pair_summaries)
    wins = sum(record["delta_a_minus_b"] > 0 for record in pair_summaries)
    losses = sum(record["delta_a_minus_b"] < 0 for record in pair_summaries)
    ties = len(pair_summaries) - wins - losses

    summary = {
        "name_a": args.name_a,
        "name_b": args.name_b,
        "checkpoint_a": str(Path(args.checkpoint_a).resolve()),
        "checkpoint_b": str(Path(args.checkpoint_b).resolve()),
        "checkpoint_epoch_a": checkpoint_a.get("epoch"),
        "checkpoint_epoch_b": checkpoint_b.get("epoch"),
        "split": args.split,
        "pairs": len(pair_summaries),
        "queries": total_queries,
        "top1_real_a": correct_a / max(total_queries, 1),
        "top1_real_b": correct_b / max(total_queries, 1),
        "delta_a_minus_b": (correct_a - correct_b) / max(total_queries, 1),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "net_rescue_a": a_only - b_only,
        "pair_wins_a": wins,
        "pair_ties": ties,
        "pair_losses_a": losses,
        "pair_cluster_bootstrap": _cluster_bootstrap(
            pair_summaries, args.bootstrap_iterations, args.bootstrap_seed
        ),
        "worm_bootstrap": _worm_bootstrap(
            pair_summaries, args.bootstrap_iterations, args.bootstrap_seed + 1
        ),
        "pair_records": pair_summaries,
    }
    return summary, all_queries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired real-only comparison of two NeuRID checkpoints"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260823)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-output", default=None)
    args = parser.parse_args()

    summary, query_records = compare_models(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.query_output:
        query_output = Path(args.query_output)
        query_output.parent.mkdir(parents=True, exist_ok=True)
        if query_records:
            with query_output.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(query_records[0]))
                writer.writeheader()
                writer.writerows(query_records)

    printable = dict(summary)
    printable.pop("pair_records")
    print(json.dumps(printable, indent=2, ensure_ascii=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
