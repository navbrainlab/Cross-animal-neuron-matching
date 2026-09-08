#!/usr/bin/env python3
"""Evaluate the original semantic GeoTransformer against one train-only medoid.

This intentionally reuses the original model, checkpoints, dense-score builder,
and metric implementation.  The only protocol change is the evaluation pair
set: each held-out test animal is the reference/query and the geometry medoid
selected from that outer fold's training animals is the source/template.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
NUCLR_ROOT = REPOSITORY.parent / "nuclr"
SEEDS = (1, 42, 123)
FOLDS = (1, 2, 3, 4, 5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def parse_best_checkpoint(dataset: str, fold: int, seed: int) -> tuple[Path, int, float]:
    result_dir = REPOSITORY / "cv5x3_results" / dataset / f"fold_{fold}" / f"seed_{seed}"
    selection_path = result_dir / "best_checkpoint.txt"
    text = selection_path.read_text(encoding="utf-8")
    iteration_match = re.search(r"^iter=(\d+)$", text, flags=re.MULTILINE)
    val_match = re.search(r"^val_top1=([0-9.]+)$", text, flags=re.MULTILINE)
    if iteration_match is None or val_match is None:
        raise RuntimeError(f"Malformed checkpoint selection file: {selection_path}")
    iteration = int(iteration_match.group(1))
    experiment = f"geotransformer.{dataset}.semantic"
    checkpoint = (
        REPOSITORY
        / "output"
        / experiment
        / f"{dataset}_fold{fold}_seed{seed}"
        / "snapshots"
        / f"iter-{iteration}.pth.tar"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint, iteration, float(val_match.group(1))


def aggregate_pair_results(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "queries": sum(row["queries"] for row in pair_results),
        "top1_correct": sum(row["top1_correct"] for row in pair_results),
        "top5_correct": sum(row["top5_correct"] for row in pair_results),
        "rr_sum": sum(row["rr_sum"] for row in pair_results),
        "hungarian_correct": sum(row["hungarian_correct"] for row in pair_results),
        "covered": sum(row["covered"] for row in pair_results),
    }
    queries = totals["queries"]
    if queries <= 0:
        raise RuntimeError("The fixed template produced no evaluable identity queries")
    return totals | {
        "pairs": len(pair_results),
        "top1": totals["top1_correct"] / queries,
        "top5": totals["top5_correct"] / queries,
        "mrr": totals["rr_sum"] / queries,
        "hungarian_accuracy": totals["hungarian_correct"] / queries,
        "candidate_coverage": totals["covered"] / queries,
    }


def make_fixed_template_loader(cfg: Any, dataset_module: Any, template_index: int):
    from geotransformer.utils.data import (
        build_dataloader_stack_mode,
        calibrate_neighbors_stack_mode,
        registration_collate_fn_stack_mode,
    )

    train_dataset = dataset_module.build_dataset(cfg, "train")
    test_dataset = dataset_module.build_dataset(cfg, "test")
    template_path = Path(train_dataset.files[template_index])
    template_ids = train_dataset.id_cache[template_index]

    # Preserve the original Dataset.__getitem__ and collate path.  Append the
    # training template as the sole source and replace all test-test pairs by
    # test-reference -> train-template pairs.  Test labels only determine
    # whether a metric query exists; they never select the template or enter
    # model features.
    original_test_count = len(test_dataset.files)
    test_dataset.files.append(str(template_path))
    test_dataset.id_cache.append(template_ids)
    template_dataset_index = original_test_count
    fixed_pairs = []
    excluded_no_shared = []
    template_valid = set(template_ids[template_ids >= 0].tolist())
    for query_index in range(original_test_count):
        query_valid = set(test_dataset.id_cache[query_index][test_dataset.id_cache[query_index] >= 0].tolist())
        shared = len(query_valid.intersection(template_valid))
        if shared:
            fixed_pairs.append((query_index, template_dataset_index, shared))
        else:
            excluded_no_shared.append(test_dataset.files[query_index])
    test_dataset.pairs = fixed_pairs

    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
    )
    loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    return loader, neighbor_limits.tolist(), excluded_no_shared


@torch.no_grad()
def evaluate_checkpoint(model: torch.nn.Module, loader: Any, evaluate_pair: Any, device: torch.device) -> dict[str, Any]:
    model.eval()
    rows = []
    for data_dict in loader:
        ref_name = str(data_dict["ref_name"])
        src_name = str(data_dict["src_name"])
        output = model(move_to_device(data_dict, device))
        metrics = evaluate_pair(output)
        if metrics is None:
            continue
        rows.append({"query": ref_name, "template": src_name} | metrics)
    return aggregate_pair_results(rows) | {"per_test_worm": rows}


def fold_summary(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_names = ("top1", "top5", "mrr", "hungarian_accuracy", "candidate_coverage")
    folds = []
    for fold in FOLDS:
        selected = [cell for cell in cells if cell["fold"] == fold]
        if len(selected) != len(SEEDS):
            raise RuntimeError(f"fold {fold} has {len(selected)} cells, expected {len(SEEDS)}")
        folds.append({"fold": fold} | {
            name: float(np.mean([cell["test"][name] for cell in selected]))
            for name in metric_names
        })
    summary = {}
    for name in metric_names:
        values = np.asarray([row[name] for row in folds], dtype=np.float64)
        summary[name] = {"mean": float(values.mean()), "sample_sd_across_folds": float(values.std(ddof=1))}
    return folds, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("atanas", "rld"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPOSITORY / "cv5x3_results_train_medoid_template_v1",
    )
    args = parser.parse_args()

    experiment = f"geotransformer.{args.dataset}.semantic"
    experiment_dir = REPOSITORY / "experiments" / experiment
    sys.path.insert(0, str(NUCLR_ROOT))
    sys.path.insert(0, str(experiment_dir))
    sys.path.insert(0, str(REPOSITORY))
    dataset_module = importlib.import_module("dataset")
    config_module = importlib.import_module("config")
    model_module = importlib.import_module("model")
    evaluation_module = importlib.import_module("evaluate_semantic")
    from fair_identity_protocol import select_geometry_medoid

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output_root / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    templates = []

    for fold in FOLDS:
        fold_root = REPOSITORY / "data_cv5" / args.dataset / f"fold_{fold}"
        cfg = config_module.make_cfg()
        cfg.data.dataset_root = str(fold_root)
        cfg.test.num_workers = 0
        cfg.coarse_matching.num_correspondences = 96 if args.dataset == "atanas" else 32

        train_dataset = dataset_module.build_dataset(cfg, "train")
        train_clouds = [train_dataset._load_cloud(path)[0] for path in train_dataset.files]
        train_uids = [str(Path(path).relative_to(fold_root / "train")) for path in train_dataset.files]
        medoid = select_geometry_medoid(train_uids, train_clouds)
        template_path = Path(train_dataset.files[medoid.index]).resolve()
        loader, neighbor_limits, excluded = make_fixed_template_loader(
            cfg, dataset_module, medoid.index
        )
        template_record = {
            "fold": fold,
            "selection_split": "outer_train_only",
            "uses_validation": False,
            "uses_test": False,
            "uses_identity_labels": False,
            "criterion": "minimum mean distance to all other outer-training worms",
            "distance": "symmetric mean nearest-neighbour Euclidean distance after the original per-animal normalization",
            "tie_break": "relative training path, then sorted training index",
            "template_uid": medoid.uid,
            "template_path": str(template_path),
            "template_train_index": medoid.index,
            "template_mean_distance": medoid.mean_distance,
            "training_candidates": [
                {"uid": uid, "mean_geometry_distance": medoid.mean_distances[index]}
                for index, uid in enumerate(train_uids)
            ],
            "neighbor_limits": neighbor_limits,
            "test_worms_excluded_for_zero_shared_identity": excluded,
        }
        templates.append(template_record)
        print(f"[template] dataset={args.dataset} fold={fold} uid={medoid.uid} mean_distance={medoid.mean_distance:.8f}", flush=True)

        for seed in SEEDS:
            checkpoint, iteration, val_top1_percent = parse_best_checkpoint(args.dataset, fold, seed)
            checkpoint_digest = sha256(checkpoint)
            cell_path = output_dir / f"fold_{fold}" / f"seed_{seed}.json"
            if cell_path.is_file():
                existing = json.loads(cell_path.read_text(encoding="utf-8"))
                if existing.get("checkpoint_sha256") == checkpoint_digest:
                    print(f"[reuse] fold={fold} seed={seed} {cell_path}", flush=True)
                    cells.append(existing)
                    continue
                raise RuntimeError(f"Refusing to overwrite result for a different checkpoint: {cell_path}")

            model = model_module.create_model(cfg).to(device)
            evaluation_module.load_checkpoint(model, str(checkpoint))
            test = evaluate_checkpoint(model, loader, evaluation_module.evaluate_pair, device)
            cell = {
                "dataset": args.dataset,
                "fold": fold,
                "seed": seed,
                "best_iteration": iteration,
                "validation_top1_percent_used_for_checkpoint_selection": val_top1_percent,
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_digest,
                "template_uid": medoid.uid,
                "test": test,
            }
            cell_path.parent.mkdir(parents=True, exist_ok=True)
            cell_path.write_text(json.dumps(cell, indent=2) + "\n", encoding="utf-8")
            cells.append(cell)
            print(
                f"[result] fold={fold} seed={seed} pairs={test['pairs']} queries={test['queries']} "
                f"Top1={100*test['top1']:.2f}% Top5={100*test['top5']:.2f}% "
                f"MRR={test['mrr']:.4f} Hung={100*test['hungarian_accuracy']:.2f}% "
                f"Coverage={100*test['candidate_coverage']:.2f}%",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fold_means, summary = fold_summary(cells)
    report = {
        "protocol": "original_semantic_geotransformer_outer_train_geometry_medoid_template_v1",
        "dataset": args.dataset,
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "checkpoint_selection": "highest validation Top-1; exact tie uses earlier checkpoint (unchanged from original run)",
        "evaluation_direction": "each outer-test worm -> one outer-training geometry medoid",
        "metric_implementation": "unchanged evaluate_pair from experiments/geotransformer.<dataset>.semantic/evaluate_semantic.py",
        "aggregation": "query-weighted within each cell; 3 seeds averaged within fold; mean and sample SD across 5 folds",
        "templates": templates,
        "cells": cells,
        "fold_means": fold_means,
        "summary": summary,
    }
    report_path = output_dir / "REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path.resolve()), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
