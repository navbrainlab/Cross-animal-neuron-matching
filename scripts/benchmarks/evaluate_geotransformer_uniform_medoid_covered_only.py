#!/usr/bin/env python3
"""Re-test GeoTransformer on the frozen shared medoid-covered cohort.

The validation-selected seed-42 checkpoint is reused unchanged.  Each held-out
test animal is paired with the common outer-train geometry medoid, and metrics
are emitted only for the canonical queries covered by that medoid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.benchmarks.evaluate_uniform_medoid_covered_only import (
    DATA_ROOTS,
    aggregate,
    canonical_queries,
    frozen_medoid,
    uid_for,
    write_query_rows,
)


GEO_ROOT = REPO.parent / "geotransformer_official"
HELPER_PATH = GEO_ROOT / "train_medoid_protocol/evaluate_cv5x3_train_medoid.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_checkpoint(dataset: str, fold: int) -> dict[str, Any]:
    selection = GEO_ROOT / f"current_grouped_selection/{dataset}/fold{fold}/seed42/best_checkpoint.txt"
    text = selection.read_text(encoding="utf-8")
    checkpoint_match = re.search(r"^checkpoint=(.+)$", text, re.MULTILINE)
    iteration_match = re.search(r"^iteration=(\d+)$", text, re.MULTILINE)
    validation_match = re.search(r"^validation_top1_percent=([0-9.eE+-]+)$", text, re.MULTILINE)
    if not (checkpoint_match and iteration_match and validation_match):
        raise RuntimeError(f"Malformed checkpoint selection file: {selection}")
    checkpoint = Path(checkpoint_match.group(1).strip()).resolve()
    if not checkpoint.is_file() or "current_grouped" not in str(checkpoint):
        raise RuntimeError(f"Invalid current-grouped checkpoint: {checkpoint}")
    return {
        "selection_file": selection.resolve(),
        "checkpoint": checkpoint,
        "iteration": int(iteration_match.group(1)),
        "validation_top1_percent": float(validation_match.group(1)),
    }


def import_geo(dataset: str):
    experiment = GEO_ROOT / f"experiments/geotransformer.{dataset}.semantic"
    for path in (REPO, experiment, GEO_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    for name in ("config", "dataset", "model", "loss", "evaluate_semantic"):
        sys.modules.pop(name, None)
    cfgmod = importlib.import_module("config")
    dsmod = importlib.import_module("dataset")
    modelmod = importlib.import_module("model")
    lossmod = importlib.import_module("loss")
    evalmod = importlib.import_module("evaluate_semantic")
    spec = importlib.util.spec_from_file_location("geo_fixed_medoid_helper", HELPER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {HELPER_PATH}")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return cfgmod, dsmod, modelmod, lossmod, evalmod, helper


def install_cpu_compatibility_shims() -> None:
    """Preserve operations while removing two legacy CUDA/contiguity assumptions."""
    torch.Tensor.cuda = lambda self, *args, **kwargs: self  # type: ignore[method-assign]
    index_module = importlib.import_module("geotransformer.modules.ops.index_select")
    original = index_module.index_select

    def compatible_index_select(data: torch.Tensor, index: torch.LongTensor, dim: int) -> torch.Tensor:
        output = data.index_select(dim, index.reshape(-1))
        if index.ndim > 1:
            output_shape = data.shape[:dim] + index.shape + data.shape[dim:][1:]
            output = output.reshape(*output_shape)
        return output

    # Several modules imported the function directly; replace each bound copy.
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "index_select", None) is original:
            setattr(module, "index_select", compatible_index_select)


def dataset_index_for_uid(dataset_obj: Any, wanted_uid: str) -> int:
    hits = [index for index, value in enumerate(dataset_obj.files) if uid_for(Path(value)) == wanted_uid]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one dataset file for {wanted_uid}, got {hits}")
    return hits[0]


def file_lookup(fold_root: Path, split: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted((fold_root / split).rglob("*.npz")):
        uid = uid_for(path)
        if uid in output:
            raise RuntimeError(f"Duplicate UID {uid} under {fold_root / split}")
        output[uid] = path
    return output


def filtered_identities(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"])
        labels = np.asarray(data["cell_id"]).astype(str)
        labeled = np.asarray(data.get("labeled_mask", np.ones(len(labels))), dtype=bool)
    finite = np.isfinite(xyz).all(axis=1)
    labels = labels[finite]
    labeled = labeled[finite]
    return [value.strip() if valid else "" for value, valid in zip(labels, labeled)]


@torch.no_grad()
def evaluate(dataset: str, fold: int, device: torch.device, output_dir: Path) -> dict[str, Any]:
    cfgmod, dsmod, modelmod, lossmod, evalmod, helper = import_geo(dataset)
    if device.type == "cpu":
        # The vendored code constructs temporary tensors using ``.cuda()`` and
        # uses ``view`` on expanded indices.  Both shims are operation-preserving
        # CPU compatibility fixes; model weights and inputs stay explicitly CPU.
        install_cpu_compatibility_shims()
    fold_root = DATA_ROOTS[dataset] / f"fold_{fold}"
    medoid_uid, medoid_path = frozen_medoid(dataset, fold)
    queries = canonical_queries(dataset, fold)
    test_files = file_lookup(fold_root, "test")
    if set(queries) - set(test_files):
        raise RuntimeError(f"Canonical query UIDs absent from GeoTransformer test split: {set(queries) - set(test_files)}")

    cfg = cfgmod.make_cfg()
    cfg.data.dataset_root = str(fold_root)
    cfg.test.num_workers = 0
    # Preserve the setting used by the current-grouped runs.
    cfg.coarse_matching.num_correspondences = 96 if dataset == "atanas" else 32
    train_dataset = dsmod.build_dataset(cfg, "train")
    template_index = dataset_index_for_uid(train_dataset, medoid_uid)
    loader, neighbor_limits, excluded = helper.make_fixed_template_loader(cfg, dsmod, template_index)
    excluded_uids = {uid_for(Path(path)) for path in excluded}
    if excluded_uids.intersection(queries):
        raise RuntimeError(
            "Covered canonical cohort unexpectedly produced excluded test worms: "
            f"{sorted(excluded_uids.intersection(queries))}"
        )

    checkpoint = parse_checkpoint(dataset, fold)
    model = modelmod.create_model(cfg).to(device)
    evalmod.load_checkpoint(model, str(checkpoint["checkpoint"]))
    model.eval()

    reference_labels = filtered_identities(medoid_path)
    rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for data_dict in loader:
        ref_name = str(data_dict["ref_name"])
        query_path = fold_root / "test" / ref_name
        if not query_path.is_file():
            # Files may be nested; basename remains unique in these frozen folds.
            matches = [path for path in test_files.values() if path.name == ref_name]
            if len(matches) != 1:
                raise RuntimeError(f"Cannot resolve GeoTransformer query file {ref_name}")
            query_path = matches[0]
        query_uid = uid_for(query_path)
        wanted = queries.get(query_uid, [])
        if not wanted:
            continue
        seen_uids.add(query_uid)
        query_labels = filtered_identities(query_path)
        output = model(helper.move_to_device(data_dict, device))
        dense = lossmod.build_dense_scores(output)
        if dense.shape != (len(query_labels), len(reference_labels)):
            raise RuntimeError(
                f"Dense score/label mismatch for {query_uid}: {tuple(dense.shape)} vs "
                f"{len(query_labels)}x{len(reference_labels)}"
            )
        src_ids = output["src_ids"]
        score_np = dense.detach().cpu().numpy().astype(np.float64)
        assignment_rows, assignment_cols = linear_sum_assignment(-score_np)
        assignment = {int(row): int(col) for row, col in zip(assignment_rows, assignment_cols)}
        for query_index, identity in wanted:
            if query_index >= len(query_labels) or query_labels[query_index] != identity:
                raise RuntimeError(
                    f"GeoTransformer canonical key unavailable: {dataset} fold{fold} "
                    f"{query_uid} {query_index} {identity}"
                )
            target_id = int(dsmod.stable_cell_id(identity))
            target_cols = torch.where(src_ids == target_id)[0]
            if target_cols.numel() == 0:
                raise RuntimeError(f"Medoid target identity absent after loading: {identity}")
            row_scores = dense[query_index]
            target_score = row_scores[target_cols].max()
            rank = 1 + int((row_scores > target_score).sum().item())
            pred_index = int(row_scores.argmax().item())
            topk = torch.topk(row_scores, k=min(5, row_scores.numel())).indices
            has_candidate = bool((row_scores > -1e8).any().item())
            top1 = int(int(src_ids[pred_index].item()) == target_id)
            top5 = int(has_candidate and bool((src_ids[topk] == target_id).any().item()))
            gt_seen = bool((target_score > -1e8).item())
            rr = 1.0 / rank if gt_seen else 0.0
            assigned = assignment.get(query_index, -1)
            hungarian_correct = int(assigned >= 0 and int(src_ids[assigned].item()) == target_id)
            rows.append({
                "dataset": dataset,
                "fold": fold,
                "seed": 42,
                "method": "GeoTransformer",
                "query_uid": query_uid,
                "query_index": query_index,
                "identity": identity,
                "reference_uid": medoid_uid,
                "rank": rank if gt_seen else "",
                "top1": top1,
                "top5": top5,
                "rr": rr,
                "hungarian_correct": hungarian_correct,
                "predicted_index": pred_index,
                "predicted_identity": reference_labels[pred_index],
            })

    if seen_uids != set(queries):
        raise RuntimeError(f"GeoTransformer did not evaluate all canonical worms: missing={set(queries)-seen_uids}")
    expected_count = sum(len(value) for value in queries.values())
    if len(rows) != expected_count:
        raise RuntimeError(f"GeoTransformer query count mismatch: {len(rows)} != {expected_count}")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_query_rows(output_dir / "query_level.csv", rows)
    metrics = aggregate(rows) | {
        "method": "GeoTransformer",
        "dataset": dataset,
        "fold": fold,
        "model_seed": 42,
        "evaluation_split": "test",
        "query_protocol": "canonical_medoid_covered_only",
        "reference_uid": medoid_uid,
        "reference_path": str(medoid_path),
        "reference_selection": "frozen common outer-train geometry medoid",
        "checkpoint": str(checkpoint["checkpoint"]),
        "checkpoint_sha256": sha256(checkpoint["checkpoint"]),
        "checkpoint_iteration": checkpoint["iteration"],
        "validation_top1_percent_used_for_checkpoint_selection": checkpoint["validation_top1_percent"],
        "checkpoint_selection": "reused frozen validation-selected seed42 checkpoint; no test tuning",
        "selection_file": str(checkpoint["selection_file"]),
        "selection_file_sha256": sha256(checkpoint["selection_file"]),
        "neighbor_limits": neighbor_limits,
        "device": str(device),
        "cpu_legacy_cuda_noop_shim": device.type == "cpu",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("atanas", "rld"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    result = evaluate(args.dataset, args.fold, device, args.output_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
