#!/usr/bin/env python3
"""Replay RLD GeoTransformer robustness with the Table-1 unified medoid.

The validation-selected checkpoint and the common outer-train geometry medoid
are frozen before test corruption.  Only the query/test files are changed.
Prediction-level rows are written so the central robustness summarizer can use
the complete canonical-query denominator and score reference misses as zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmarks.evaluate_geotransformer_uniform_medoid_covered_only import (
    filtered_identities,
    import_geo,
    install_cpu_compatibility_shims,
    parse_checkpoint,
)
from scripts.benchmarks.evaluate_uniform_medoid_covered_only import (
    aggregate,
    canonical_queries,
    frozen_medoid,
    sha256,
    uid_for,
    write_query_rows,
)


DATA_ROOT = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
RUN_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v2"
DEFAULT_CORRUPTION = RUN_ROOT / "corruptions/MANIFEST.json"
DEFAULT_OUTPUT = RUN_ROOT / "results/geotransformer"
MAIN_CLEAN = ROOT / "runs/unified_medoid_covered_only_retest_v1/geotransformer/rld"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--kinds", default="coord_noise,missing,outlier")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--corruption-manifest", type=Path, default=DEFAULT_CORRUPTION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def condition_rows(manifest: dict, folds: set[int], kinds: set[str]):
    rows = []
    for row in manifest["conditions"]:
        fold = int(row["fold"])
        if fold not in folds or row["kind"] not in kinds:
            continue
        name = Path(row["root"]).name
        rows.append((fold, row["kind"], float(row["severity"]),
                     int(row["perturbation_seed"]), name, row))
    order = {"coord_noise": 0, "missing": 1, "outlier": 2}
    return sorted(rows, key=lambda item: (item[0], order[item[1]], item[2], item[3]))


def make_mixed_root(path: Path, clean_fold: Path, test_dir: Path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    for split, target in {
        "train": clean_fold / "train",
        "val": clean_fold / "val",
        "test": test_dir,
    }.items():
        os.symlink(str(target.resolve()), str(path / split), target_is_directory=True)


def dataset_index_for_uid(dataset_obj, wanted_uid: str):
    hits = [i for i, value in enumerate(dataset_obj.files)
            if uid_for(Path(value)) == wanted_uid]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one train entry for {wanted_uid}, got {hits}")
    return hits[0]


def wanted_after_corruption(condition: dict, clean_covered: dict):
    """Map clean source-row keys to current corrupted row indices."""
    output = {}
    for record in condition["files"]:
        worm = str(record["recording_uid"])
        kept = [int(value) for value in record["kept_source_indices"]]
        source_to_current = {source: current for current, source in enumerate(kept)}
        observed = {
            int(item["row"]): str(item["cell_id"])
            for item in record["evaluable_query_rows_after_corruption"]
        }
        selected = []
        for source_index, identity in clean_covered.get(worm, []):
            current_index = source_to_current.get(int(source_index))
            if current_index is None:
                continue
            if observed.get(current_index) != identity:
                raise RuntimeError(
                    "Covered canonical row changed identity: "
                    f"uid={worm} source={source_index} current={current_index} "
                    f"expected={identity!r} observed={observed.get(current_index)!r}"
                )
            selected.append((current_index, identity))
        if selected:
            output[worm] = sorted(selected)
    return output


def resolve_query(test_dir: Path, name: str):
    direct = test_dir / name
    if direct.is_file():
        return direct
    hits = list(test_dir.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"Cannot resolve GeoTransformer query {name}: {hits}")
    return hits[0]


@torch.no_grad()
def evaluate_condition(*, fold: int, condition: dict, condition_name: str,
                       clean_covered: dict, medoid_uid: str, medoid_path: Path,
                       checkpoint: dict, device: torch.device, modules: tuple,
                       model, scratch: Path):
    cfgmod, dsmod, _modelmod, lossmod, _evalmod, helper = modules
    test_dir = Path(condition["root"]) / "test"
    mixed = scratch / f"fold{fold}" / condition_name
    make_mixed_root(mixed, DATA_ROOT / f"fold_{fold}", test_dir)

    cfg = cfgmod.make_cfg()
    cfg.data.dataset_root = str(mixed)
    cfg.test.num_workers = 0
    cfg.coarse_matching.num_correspondences = 32
    train_dataset = dsmod.build_dataset(cfg, "train")
    template_index = dataset_index_for_uid(train_dataset, medoid_uid)
    loader, neighbor_limits, excluded = helper.make_fixed_template_loader(
        cfg, dsmod, template_index
    )

    wanted = wanted_after_corruption(condition, clean_covered)
    reference_labels = filtered_identities(medoid_path)
    rows = []
    inferred_worms = set()
    for data_dict in loader:
        query_path = resolve_query(test_dir, str(data_dict["ref_name"]))
        query_uid = uid_for(query_path)
        selected = wanted.get(query_uid, [])
        if not selected:
            continue
        inferred_worms.add(query_uid)
        query_labels = filtered_identities(query_path)
        output = model(helper.move_to_device(data_dict, device))
        dense = lossmod.build_dense_scores(output)
        if dense.shape != (len(query_labels), len(reference_labels)):
            raise RuntimeError(
                f"Dense score mismatch {query_uid}: {tuple(dense.shape)} vs "
                f"{len(query_labels)}x{len(reference_labels)}"
            )
        src_ids = output["src_ids"]
        score_np = dense.detach().cpu().numpy().astype(np.float64)
        assignment_rows, assignment_cols = linear_sum_assignment(-score_np)
        assignment = {int(r): int(c) for r, c in zip(assignment_rows, assignment_cols)}
        for query_index, identity in selected:
            if query_index >= len(query_labels) or query_labels[query_index] != identity:
                raise RuntimeError(
                    f"Query row unavailable: fold={fold} condition={condition_name} "
                    f"uid={query_uid} index={query_index} identity={identity}"
                )
            target_id = int(dsmod.stable_cell_id(identity))
            target_cols = torch.where(src_ids == target_id)[0]
            if target_cols.numel() == 0:
                raise RuntimeError(f"Frozen medoid identity unavailable: {identity}")
            row_scores = dense[query_index]
            target_score = row_scores[target_cols].max()
            rank = 1 + int((row_scores > target_score).sum().item())
            predicted_index = int(row_scores.argmax().item())
            topk = torch.topk(row_scores, k=min(5, row_scores.numel())).indices
            valid_candidate = bool((row_scores > -1e8).any().item())
            gt_seen = bool((target_score > -1e8).item())
            assigned = assignment.get(query_index, -1)
            rows.append({
                "dataset": "rld",
                "fold": fold,
                "seed": 42,
                "method": "GeoTransformer",
                "query_uid": query_uid,
                "query_index": query_index,
                "identity": identity,
                "reference_uid": medoid_uid,
                "rank": rank if gt_seen else "",
                "top1": int(int(src_ids[predicted_index].item()) == target_id),
                "top5": int(valid_candidate and bool((src_ids[topk] == target_id).any().item())),
                "rr": 1.0 / rank if gt_seen else 0.0,
                "hungarian_correct": int(
                    assigned >= 0 and int(src_ids[assigned].item()) == target_id
                ),
                "predicted_index": predicted_index,
                "predicted_identity": reference_labels[predicted_index],
            })

    metrics = aggregate(rows)
    metrics.update({
        "method": "GeoTransformer",
        "dataset": "rld",
        "fold": fold,
        "model_seed": 42,
        "condition": condition_name,
        "kind": condition["kind"],
        "severity": float(condition["severity"]),
        "perturbation_seed": int(condition["perturbation_seed"]),
        "reference_uid": medoid_uid,
        "reference_path": str(medoid_path.resolve()),
        "reference_selection": "frozen common outer-train geometry medoid",
        "checkpoint": str(checkpoint["checkpoint"]),
        "checkpoint_sha256": sha256(checkpoint["checkpoint"]),
        "checkpoint_iteration": int(checkpoint["iteration"]),
        "validation_top1_percent_used_for_checkpoint_selection": checkpoint["validation_top1_percent"],
        "query_protocol": "query-only corruption; unified-medoid-covered rows emitted; full canonical denominator applied centrally",
        "covered_canonical_rows_surviving": sum(len(value) for value in wanted.values()),
        "inferred_worms": sorted(inferred_worms),
        "excluded_no_shared_identity": [str(value) for value in excluded],
        "neighbor_limits": [int(value) for value in neighbor_limits],
        "device": str(device),
    })
    return rows, metrics


def exact_clean_guard(fold: int, rows: list[dict]):
    baseline_path = MAIN_CLEAN / f"fold{fold}/query_level.csv"
    with baseline_path.open(newline="", encoding="utf-8") as handle:
        baseline = list(csv.DictReader(handle))
    fields = (
        "query_uid", "query_index", "identity", "reference_uid", "rank",
        "top1", "top5", "rr", "hungarian_correct", "predicted_index",
        "predicted_identity",
    )
    current = [{key: str(row[key]) for key in fields} for row in rows]
    expected = [{key: str(row[key]) for key in fields} for row in baseline]
    if current != expected:
        raise RuntimeError(f"GeoTransformer clean query-level replay failed for fold{fold}")


def main():
    args = parse_args()
    folds = {int(value) for value in args.folds.split(",") if value.strip()}
    kinds = {value.strip() for value in args.kinds.split(",") if value.strip()}
    if not folds <= set(range(5)) or not kinds <= {"coord_noise", "missing", "outlier"}:
        raise ValueError((folds, kinds))
    manifest = read_json(args.corruption_manifest.resolve())
    grouped = condition_rows(manifest, folds, kinds)
    device = torch.device(args.device)
    output_root = args.output_root.resolve()
    scratch = output_root / "_mixed_fold_roots"
    output_root.mkdir(parents=True, exist_ok=True)
    completed = []

    for fold in sorted(folds):
        modules = import_geo("rld")
        if device.type == "cpu":
            install_cpu_compatibility_shims()
        cfgmod, _dsmod, modelmod, _lossmod, evalmod, _helper = modules
        checkpoint = parse_checkpoint("rld", fold)
        medoid_uid, medoid_path = frozen_medoid("rld", fold)
        clean_covered = canonical_queries("rld", fold)
        cfg = cfgmod.make_cfg()
        # Match the frozen current-grouped RLD evaluation setting exactly.
        cfg.coarse_matching.num_correspondences = 32
        model = modelmod.create_model(cfg).to(device)
        evalmod.load_checkpoint(model, str(checkpoint["checkpoint"]))
        model.eval()
        print(f"[FOLD {fold}] reference={medoid_uid} checkpoint=iter{checkpoint['iteration']}", flush=True)

        for item_fold, kind, severity, draw, name, condition in grouped:
            if item_fold != fold:
                continue
            directory = output_root / f"fold{fold}/{name}"
            metrics_path = directory / "metrics.json"
            query_path = directory / "query_level.csv"
            if metrics_path.is_file() and query_path.is_file() and not args.overwrite:
                metrics = read_json(metrics_path)
                rows = []
            else:
                rows, metrics = evaluate_condition(
                    fold=fold,
                    condition=condition,
                    condition_name=name,
                    clean_covered=clean_covered,
                    medoid_uid=medoid_uid,
                    medoid_path=medoid_path,
                    checkpoint=checkpoint,
                    device=device,
                    modules=modules,
                    model=model,
                    scratch=scratch,
                )
                directory.mkdir(parents=True, exist_ok=True)
                write_query_rows(query_path, rows)
                write_json(metrics_path, metrics)
            if abs(severity) <= 1e-12:
                if not rows:
                    with query_path.open(newline="", encoding="utf-8") as handle:
                        rows = list(csv.DictReader(handle))
                exact_clean_guard(fold, rows)
            print(
                f"[RESULT] fold={fold} {name} covered_Q={metrics['queries']} "
                f"correct={metrics['top1_correct']} top1_covered={100*metrics['top1']:.2f}%",
                flush=True,
            )
            completed.append({
                "fold": fold, "kind": kind, "severity": severity, "draw": draw,
                "condition": name, "metrics": str(metrics_path),
                "query_level": str(query_path), "reference_uid": medoid_uid,
            })
        del model

    write_json(output_root / "run_manifest.json", {
        "method": "GeoTransformer",
        "protocol": "Table-1 unified medoid; frozen checkpoint/reference; query-only corruption",
        "conditions": completed,
    })
    print(f"DONE: {output_root}")


if __name__ == "__main__":
    main()
