from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig
from .data import PairIndex, WormCache, build_pair_targets
from .metrics import MetricTotals, pair_metrics
from .model import MPRTNet


@torch.no_grad()
def evaluate_model(
    model: MPRTNet,
    pair_index: PairIndex,
    cache: WormCache,
    device: torch.device,
    max_pairs: int | None = None,
    collect_pair_records: bool = False,
) -> dict[str, Any]:
    model.eval()
    totals = MetricTotals()
    pair_records: list[dict[str, Any]] = []
    pairs = pair_index.pairs
    if max_pairs is not None and max_pairs > 0:
        pairs = pairs[:max_pairs]
    evaluated_pairs = 0
    for path_a, path_b in pairs:
        sample_a, sample_b, targets = build_pair_targets(cache.get(path_a), cache.get(path_b))
        if targets.num_direct_matches == 0:
            continue
        output = model(sample_a.to(device), sample_b.to(device))
        pair_total = pair_metrics(output, targets.to(device))
        totals.update(pair_total)
        if collect_pair_records:
            pair_record = {
                "uid_a": sample_a.uid,
                "uid_b": sample_b.uid,
                "path_a": sample_a.source_path,
                "path_b": sample_b.source_path,
                "direct_matches": targets.num_direct_matches,
                **pair_total.compute(),
            }
            pair_records.append(pair_record)
        evaluated_pairs += 1
    result = totals.compute()
    result["pairs"] = evaluated_pairs
    if collect_pair_records:
        result["pair_records"] = pair_records
    return result


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[MPRTNet, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model = MPRTNet(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a NeuRID checkpoint")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    parser.add_argument("--pair-output", default=None)
    args = parser.parse_args()

    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    index = PairIndex(args.dataset_root, args.split, min_shared=args.min_shared)
    cache = WormCache(activity_length=args.activity_length)
    result = evaluate_model(
        model,
        index,
        cache,
        device,
        max_pairs=args.max_pairs if args.max_pairs > 0 else None,
        collect_pair_records=bool(args.pair_output),
    )
    pair_records = result.pop("pair_records", None)
    result.update(
        {
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "split": args.split,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
    )
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if args.pair_output and pair_records is not None:
        pair_output = Path(args.pair_output)
        pair_output.parent.mkdir(parents=True, exist_ok=True)
        with pair_output.open("w", encoding="utf-8") as handle:
            for record in pair_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
