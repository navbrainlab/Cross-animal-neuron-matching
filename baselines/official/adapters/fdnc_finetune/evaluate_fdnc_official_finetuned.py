#!/usr/bin/env python3
"""Locked outer-test evaluation for a validation-selected official-fDNC fine-tuned checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
ADAPTER = ROOT / "baselines" / "official" / "adapters" / "fdnc_finetune"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ADAPTER) not in sys.path:
    sys.path.insert(0, str(ADAPTER))

import train_fdnc_official_finetune as ft
from scripts.lib.fair_identity_protocol import select_geometry_medoid

PROTOCOLS = ROOT / "baselines" / "official" / "protocols"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["atanas","rld"], required=True)
    p.add_argument("--fold", type=int, choices=[1,2,3,4,5], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-list", type=Path, default=None)
    p.add_argument("--test-list", type=Path, default=None)
    p.add_argument("--mask-key", default="clean_mask")
    p.add_argument("--coordinate-scale", type=float, default=200.0)
    p.add_argument("--precision", choices=["fp32","bf16"], default="fp32")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--save-dir", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    if args.test_list is None:
        args.test_list = PROTOCOLS / args.dataset / f"fold_{args.fold}" / "test.txt"
    if args.train_list is None:
        args.train_list = PROTOCOLS / args.dataset / f"fold_{args.fold}" / "train.txt"

    device = torch.device(args.device)
    module = ft.load_official_module()

    payload = ft.load_torch(args.checkpoint)
    if payload.get("format") != "fdnc_official_source_finetune_v1":
        raise RuntimeError(f"Unexpected fine-tuned checkpoint format: {payload.get('format')}")

    model = module.NIT_Registration(
        input_dim=3, n_hidden=128, n_layer=6,
        cuda=(device.type=="cuda"), p_rotate=False, feat_trans=False,
    ).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()

    train_records = ft.load_split(args.train_list, args.mask_key, args.coordinate_scale)
    medoid = select_geometry_medoid(
        [record.worm_id for record in train_records],
        [record.xyz for record in train_records],
    )
    template = train_records[medoid.index]
    test_records = ft.load_split(args.test_list, args.mask_key, args.coordinate_scale)

    rows = []
    for query in test_records:
        scores = ft.official_scores(model, template, query, device, args.precision)
        for row in ft.eval_direction(scores, query, template):
            row.update(dataset=args.dataset, fold=args.fold, seed=args.seed, method="fDNC official fine-tuned")
            rows.append(row)

    if not rows:
        raise RuntimeError("No outer-test queries")

    metrics = {
        "queries": len(rows),
        "ranking_top1": float(np.mean([r["top1"] for r in rows])),
        "top3": float(np.mean([r["top3"] for r in rows])),
        "top5": float(np.mean([r["top5"] for r in rows])),
        "top10": float(np.mean([r["top10"] for r in rows])),
        "mrr": float(np.mean([r["rr"] for r in rows])),
        "assignment_top1": float(np.mean([r["hungarian_top1"] for r in rows])),
    }

    args.save_dir.mkdir(parents=True, exist_ok=True)
    with (args.save_dir/"query_level.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    result = {
        "dataset": args.dataset,
        "fold": args.fold,
        "seed": args.seed,
        "method": "fDNC official fine-tuned",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": ft.sha256(args.checkpoint),
        "selected_on": "validation only",
        "train_list": str(args.train_list.resolve()),
        "test_list": str(args.test_list.resolve()),
        "metrics": metrics,
        "evaluation_protocol": "outer_training_geometry_medoid_template_v1",
        "test_to_template_pairs": len(test_records),
        "directed_test_test_pairs": 0,
        "template_selection": {
            "selection_split": "outer_train_only",
            "uses_validation": False,
            "uses_test": False,
            "uses_identity_labels": False,
            "distance": "symmetric mean nearest-neighbour Euclidean distance on preprocessed XYZ",
            "criterion": "minimum mean distance to all other outer-training worms",
            "tie_break": "worm UID, then train-list index",
            "template_worm": template.worm_id,
            "template_path": str(template.path.resolve()),
            "template_train_index": medoid.index,
            "template_mean_distance": medoid.mean_distance,
            "training_candidates": [
                {"worm_id": record.worm_id, "mean_geometry_distance": medoid.mean_distances[index]}
                for index, record in enumerate(train_records)
            ],
        },
        "official_source_modified": False,
        "training_metadata": payload["metadata"],
    }
    (args.save_dir/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
