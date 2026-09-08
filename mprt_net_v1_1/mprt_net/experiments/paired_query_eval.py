from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from ..data import PairIndex, WormCache, build_pair_targets
from ..evaluate import load_checkpoint


FIELDS = (
    "dataset",
    "split",
    "fold",
    "seed",
    "pair_id",
    "uid_a",
    "uid_b",
    "direction",
    "query_animal",
    "candidate_animal",
    "identity",
    "query_index",
    "target_index",
    "prediction_a",
    "prediction_b",
    "correct_a",
    "correct_b",
)


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _records_for_direction(
    *,
    probabilities_a: torch.Tensor,
    probabilities_b: torch.Tensor,
    targets: torch.Tensor,
    num_real_candidates: int,
    query_ids: tuple[str, ...],
    uid_a: str,
    uid_b: str,
    direction: str,
    dataset: str,
    split: str,
    fold: int,
    seed: int,
) -> Iterable[dict[str, Any]]:
    valid = (targets >= 0) & (targets < num_real_candidates)
    indices = valid.nonzero(as_tuple=False).flatten()
    if indices.numel() == 0:
        return
    real_a = probabilities_a.index_select(0, indices)[:, :num_real_candidates]
    real_b = probabilities_b.index_select(0, indices)[:, :num_real_candidates]
    prediction_a = real_a.argmax(dim=1).cpu()
    prediction_b = real_b.argmax(dim=1).cpu()
    target = targets.index_select(0, indices).cpu()
    source_uid, candidate_uid = (
        (uid_a, uid_b) if direction == "a_to_b" else (uid_b, uid_a)
    )
    pair_id = "|".join(sorted((uid_a, uid_b)))
    for offset, query_index in enumerate(indices.cpu().tolist()):
        yield {
            "dataset": dataset,
            "split": split,
            "fold": fold,
            "seed": seed,
            "pair_id": pair_id,
            "uid_a": uid_a,
            "uid_b": uid_b,
            "direction": direction,
            "query_animal": source_uid,
            "candidate_animal": candidate_uid,
            "identity": query_ids[query_index],
            "query_index": query_index,
            "target_index": int(target[offset]),
            "prediction_a": int(prediction_a[offset]),
            "prediction_b": int(prediction_b[offset]),
            "correct_a": int(prediction_a[offset] == target[offset]),
            "correct_b": int(prediction_b[offset] == target[offset]),
        }


@torch.no_grad()
def compare_checkpoints(
    *,
    dataset_root: Path,
    split: str,
    checkpoint_a: Path,
    checkpoint_b: Path,
    dataset: str,
    fold: int,
    seed: int,
    device: torch.device,
    activity_length: int,
    min_shared: int,
    max_pairs: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model_a, state_a = load_checkpoint(checkpoint_a, device)
    model_b, state_b = load_checkpoint(checkpoint_b, device)
    model_a.eval()
    model_b.eval()
    index = PairIndex(dataset_root, split, min_shared=min_shared)
    cache = WormCache(activity_length=activity_length, max_items=max(48, len(index.files)))
    pairs = index.pairs[:max_pairs] if max_pairs else index.pairs
    records: list[dict[str, Any]] = []
    for number, (path_a, path_b) in enumerate(pairs, start=1):
        sample_a, sample_b, targets = build_pair_targets(
            cache.get(path_a), cache.get(path_b)
        )
        if targets.num_direct_matches == 0:
            continue
        device_a, device_b = sample_a.to(device), sample_b.to(device)
        output_a = model_a(device_a, device_b)
        output_b = model_b(device_a, device_b)
        target_device = targets.to(device)
        records.extend(
            _records_for_direction(
                probabilities_a=output_a.row_conditional,
                probabilities_b=output_b.row_conditional,
                targets=target_device.row_target,
                num_real_candidates=sample_b.num_nodes,
                query_ids=sample_a.cell_ids,
                uid_a=sample_a.uid,
                uid_b=sample_b.uid,
                direction="a_to_b",
                dataset=dataset,
                split=split,
                fold=fold,
                seed=seed,
            )
        )
        records.extend(
            _records_for_direction(
                probabilities_a=output_a.col_conditional,
                probabilities_b=output_b.col_conditional,
                targets=target_device.col_target,
                num_real_candidates=sample_a.num_nodes,
                query_ids=sample_b.cell_ids,
                uid_a=sample_a.uid,
                uid_b=sample_b.uid,
                direction="b_to_a",
                dataset=dataset,
                split=split,
                fold=fold,
                seed=seed,
            )
        )
        print(
            f"paired_eval pair={number:03d}/{len(pairs):03d} "
            f"uids={sample_a.uid}/{sample_b.uid}",
            flush=True,
        )
    if not records:
        raise RuntimeError("Evaluation produced no real supervised queries")
    correct_a = sum(int(row["correct_a"]) for row in records)
    correct_b = sum(int(row["correct_b"]) for row in records)
    total = len(records)
    summary = {
        "dataset": dataset,
        "dataset_root": str(dataset_root.resolve()),
        "split": split,
        "fold": fold,
        "seed": seed,
        "checkpoint_a": str(checkpoint_a.resolve()),
        "checkpoint_b": str(checkpoint_b.resolve()),
        "checkpoint_epoch_a": state_a.get("epoch"),
        "checkpoint_epoch_b": state_b.get("epoch"),
        "pairs": len({row["pair_id"] for row in records}),
        "queries": total,
        "top1_real_a": correct_a / total,
        "top1_real_b": correct_b / total,
        "delta_b_minus_a": (correct_b - correct_a) / total,
        "a_only_correct": sum(
            int(row["correct_a"] and not row["correct_b"]) for row in records
        ),
        "b_only_correct": sum(
            int(row["correct_b"] and not row["correct_a"]) for row in records
        ),
    }
    return records, summary


def write_outputs(
    records: list[dict[str, Any]], summary: dict[str, Any], csv_path: Path, json_path: Path
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = json_path.with_suffix(json_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(json_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired real-only query export with a stable CSV schema"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, default=-1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    records, summary = compare_checkpoints(
        dataset_root=args.dataset_root,
        split=args.split,
        checkpoint_a=args.checkpoint_a,
        checkpoint_b=args.checkpoint_b,
        dataset=args.dataset,
        fold=args.fold,
        seed=args.seed,
        device=device,
        activity_length=args.activity_length,
        min_shared=args.min_shared,
        max_pairs=args.max_pairs or None,
    )
    write_outputs(records, summary, args.query_output, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved_queries={args.query_output}")


if __name__ == "__main__":
    main()
