#!/usr/bin/env python3
"""Test NGM-v2 or Vanilla FGW on the frozen shared medoid-covered cohort.

Training/checkpoint selection and validation-selected hyperparameters are kept
frozen.  The only evaluation changes are:

1. use the common outer-train geometry medoid frozen by the NeurID audit;
2. score exactly the canonical queries whose identity occurs in that medoid;
3. emit prediction-level rows so the denominator can be audited directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO = Path(__file__).resolve().parents[2]
THINKMATCH = REPO.parent / "ThinkMatch_official"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA_ROOTS = {
    "atanas": REPO / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": REPO / "Data/Dunn_001623/cv5_grouped_v1",
}
MEDOID_RUNS = {
    dataset: REPO / f"runs/mprt_v1_1_atlas_medoid_{dataset}_cv5_test_seeds_42_v1"
    for dataset in DATA_ROOTS
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def uid_for(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        if "recording_uid" in data.files:
            return str(np.asarray(data["recording_uid"]).reshape(-1)[0])
    parts = path.stem.split("__")
    return parts[1] if len(parts) >= 3 else path.stem


def canonical_queries(dataset: str, fold: int) -> dict[str, list[tuple[int, str]]]:
    path = MEDOID_RUNS[dataset] / f"fold{fold}/seed42/query_level.csv"
    output: dict[str, list[tuple[int, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["fold"]) != fold or int(row["seed"]) != 42:
                raise RuntimeError(f"Fold/seed mismatch in {path}")
            if int(row["medoid_candidate_present"]) != 1:
                continue
            output.setdefault(row["uid"], []).append((int(row["node_index"]), row["identity"]))
    if not output:
        raise RuntimeError(f"No covered canonical queries in {path}")
    for rows in output.values():
        rows.sort()
    return output


def frozen_medoid(dataset: str, fold: int) -> tuple[str, Path]:
    report = read_json(MEDOID_RUNS[dataset] / "summary.json")
    item = report["fold_results"][fold]["medoid"]
    uid = str(item["uid"])
    expected_train = DATA_ROOTS[dataset] / f"fold_{fold}/train"
    matches = [path for path in expected_train.glob("*.npz") if uid_for(path) == uid]
    # Fold files are symlinks into the source dataset, so resolving source_path
    # would move its parent outside fold_N/train.  Validate membership by UID
    # against the actual fold directory and use that fold-local path.
    if len(matches) != 1:
        raise RuntimeError(f"Invalid frozen medoid for {dataset} fold{fold}: {item}")
    path = matches[0]
    return uid, path


def write_query_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset", "fold", "seed", "method", "query_uid", "query_index",
        "identity", "reference_uid", "rank", "top1", "top5", "rr",
        "hungarian_correct", "predicted_index", "predicted_identity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_ngm_prediction_replay(dataset: str, fold: int, replay: Path) -> dict[str, Any]:
    """Require the score-export replay to reproduce the prior saved predictions."""
    baseline = (
        REPO
        / f"runs/unified_medoid_covered_only_retest_v1/ngmv2/{dataset}/fold{fold}/query_level.csv"
    )
    with baseline.open(newline="", encoding="utf-8") as handle:
        old = list(csv.DictReader(handle))
    with replay.open(newline="", encoding="utf-8") as handle:
        new = list(csv.DictReader(handle))
    if len(old) != len(new):
        raise RuntimeError(f"NGM replay row mismatch: old={len(old)}, new={len(new)}")
    exact = {
        "dataset", "fold", "seed", "method", "query_uid", "query_index", "identity",
        "reference_uid", "rank", "top1", "top5", "hungarian_correct",
        "predicted_index", "predicted_identity",
    }
    floating = {"rr"}
    for index, (left, right) in enumerate(zip(old, new)):
        for field in exact:
            if left[field] != right[field]:
                raise RuntimeError(
                    f"NGM prediction replay mismatch row={index} field={field}: "
                    f"saved={left[field]!r}, replay={right[field]!r}"
                )
        for field in floating:
            if not np.isclose(float(left[field]), float(right[field]), rtol=0.0, atol=1e-12):
                raise RuntimeError(
                    f"NGM prediction replay mismatch row={index} field={field}: "
                    f"saved={left[field]!r}, replay={right[field]!r}"
                )
    return {
        "status": "EXACT_SAVED_PREDICTION_REPLAY",
        "saved_prediction_csv": str(baseline.resolve()),
        "saved_prediction_sha256": sha256(baseline),
        "rows_verified": len(old),
        "exact_fields": sorted(exact),
        "numeric_tolerance_fields": {"rr": 1e-12},
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    q = len(rows)
    if q == 0:
        raise RuntimeError("No scored queries")
    return {
        "queries": q,
        "top1_correct": int(sum(int(row["top1"]) for row in rows)),
        "top5_correct": int(sum(int(row["top5"]) for row in rows)),
        "rr_sum": float(sum(float(row["rr"]) for row in rows)),
        "hungarian_correct": int(sum(int(row["hungarian_correct"]) for row in rows)),
        "top1": float(sum(int(row["top1"]) for row in rows) / q),
        "top5": float(sum(int(row["top5"]) for row in rows) / q),
        "mrr": float(sum(float(row["rr"]) for row in rows) / q),
        "hungarian_accuracy": float(sum(int(row["hungarian_correct"]) for row in rows) / q),
    }


def evaluate_fgw(dataset: str, fold: int, output_dir: Path) -> dict[str, Any]:
    module = importlib.import_module(f"scripts.benchmarks.run_official_fgw_{dataset}")
    lock_path = REPO / f"runs/unified_main_benchmark_cv5_seed42_v1/rerun/fgw_pot/{dataset}/fold{fold}/LOCKED_BEFORE_TEST.json"
    lock = read_json(lock_path)
    if lock["status"] != "LOCKED_BEFORE_TEST" or int(lock["fold"]) != fold:
        raise RuntimeError(f"Invalid frozen FGW lock: {lock_path}")
    medoid_uid, medoid_path = frozen_medoid(dataset, fold)
    template = module.load_worm(medoid_path)
    ref_map = module.unique_label_map(template)
    queries = canonical_queries(dataset, fold)
    mean = np.asarray(lock["train_mean"], dtype=np.float64)
    std = np.asarray(lock["train_std"], dtype=np.float64)
    alpha = float(lock["selected_alpha"])
    rows: list[dict[str, Any]] = []

    for worm in module.get_split(DATA_ROOTS[dataset], fold, "test"):
        uid = uid_for(worm.path)
        wanted = queries.get(uid, [])
        if not wanted:
            continue
        qmap = module.unique_label_map(worm)
        for index, identity in wanted:
            if qmap.get(identity) != index or identity not in ref_map:
                raise RuntimeError(f"FGW canonical key unavailable: {dataset} fold{fold} {uid} {index} {identity}")
        plan = module.fgw_plan(
            module.norm_xyz(worm, mean, std),
            module.norm_xyz(template, mean, std),
            alpha,
        )
        assignment_rows, assignment_cols = linear_sum_assignment(-plan)
        assignment = {int(row): int(col) for row, col in zip(assignment_rows, assignment_cols)}
        inverse_ref = {index: identity for identity, index in ref_map.items()}
        for query_index, identity in wanted:
            target_index = int(ref_map[identity])
            scores = plan[query_index]
            # Preserve the official FGW evaluator's stable column-order
            # tie-breaking.  Transport plans contain many exact zeros, so the
            # generic competition-rank rule would strongly inflate Top-5.
            rank = module.rank_of_gt(scores, target_index)
            prediction = int(np.argmax(scores))
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": "deterministic",
                    "method": "Vanilla FGW",
                    "query_uid": uid,
                    "query_index": query_index,
                    "identity": identity,
                    "reference_uid": medoid_uid,
                    "rank": rank,
                    "top1": int(rank == 1),
                    "top5": int(rank <= min(5, len(scores))),
                    "rr": 1.0 / rank,
                    "hungarian_correct": int(assignment.get(query_index, -1) == target_index),
                    "predicted_index": prediction,
                    "predicted_identity": inverse_ref.get(prediction, ""),
                }
            )

    metrics = aggregate(rows)
    metrics.update(
        {
            "method": "Vanilla FGW",
            "dataset": dataset,
            "fold": fold,
            "model_seed": None,
            "deterministic": True,
            "evaluation_split": "test",
            "query_protocol": "canonical_medoid_covered_only",
            "reference_uid": medoid_uid,
            "reference_path": str(medoid_path),
            "reference_selection": "frozen common outer-train geometry medoid",
            "selected_alpha": alpha,
            "hyperparameter_selection": "reused frozen validation-selected alpha; no test tuning",
            "source_lock": str(lock_path.resolve()),
            "source_lock_sha256": sha256(lock_path),
        }
    )
    write_query_rows(output_dir / "query_level.csv", rows)
    return metrics


def import_ngm(dataset: str):
    if str(THINKMATCH) not in sys.path:
        sys.path.insert(0, str(THINKMATCH))
    return importlib.import_module(f"scripts.benchmarks.run_ngmv2_{dataset}_fold")


def evaluate_ngm(dataset: str, fold: int, output_dir: Path, device: torch.device) -> dict[str, Any]:
    module = import_ngm(dataset)
    run_dir = REPO / f"runs/unified_benchmark/ngmv2/{dataset}/fold{fold}/seed42"
    checkpoint = run_dir / "best.pt"
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if int(payload["fold"]) != fold or int(payload["seed"]) != 42:
        raise RuntimeError(f"NGM checkpoint fold/seed mismatch: {checkpoint}")
    model = module.AtanasNGMv2(feature_dim=int(payload["feature_dim"])).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    mean = np.asarray(payload["mean"], dtype=np.float32)
    std = np.asarray(payload["std"], dtype=np.float32)
    medoid_uid, medoid_path = frozen_medoid(dataset, fold)
    template = module.load_worm(medoid_path)
    template_map = module.unique_identity_map(template)
    candidate_labels = list(template_map)
    candidate_indices = np.asarray([template_map[label] for label in candidate_labels], dtype=np.int64)
    candidate_lookup = {index: label for index, label in enumerate(candidate_labels)}
    template_graph = module.make_graph(module.normalized_xyz(template, mean, std), device)
    queries = canonical_queries(dataset, fold)
    rows: list[dict[str, Any]] = []
    tie_rows: list[dict[str, Any]] = []
    score_blocks: list[np.ndarray] = []
    score_query_uids: list[str] = []
    score_query_indices: list[int] = []
    score_gt_identities: list[str] = []
    score_target_columns: list[int] = []

    test_paths = sorted((DATA_ROOTS[dataset] / f"fold_{fold}/test").glob("*.npz"))
    with torch.inference_mode():
        for path in test_paths:
            worm = module.load_worm(path)
            uid = uid_for(path)
            wanted = queries.get(uid, [])
            if not wanted:
                continue
            source_map = module.unique_identity_map(worm)
            for index, identity in wanted:
                if source_map.get(identity) != index or identity not in template_map:
                    raise RuntimeError(f"NGM canonical key unavailable: {dataset} fold{fold} {uid} {index} {identity}")
            source_graph = module.make_graph(module.normalized_xyz(worm, mean, std), device)
            dense = model(source_graph, template_graph)[0].detach().float().cpu().numpy()
            source_indices = np.asarray([index for index, _ in wanted], dtype=np.int64)
            score_matrix = dense[np.ix_(source_indices, candidate_indices)]
            target_columns = np.asarray([candidate_labels.index(identity) for _, identity in wanted], dtype=np.int64)
            score_blocks.append(np.asarray(score_matrix, dtype=np.float32))
            score_query_uids.extend([uid] * len(wanted))
            score_query_indices.extend(int(index) for index, _ in wanted)
            score_gt_identities.extend(identity for _, identity in wanted)
            score_target_columns.extend(int(value) for value in target_columns)
            assignment_rows, assignment_cols = linear_sum_assignment(-score_matrix)
            assignment = {int(row): int(col) for row, col in zip(assignment_rows, assignment_cols)}
            for local_row, ((query_index, identity), target_col) in enumerate(zip(wanted, target_columns)):
                scores = score_matrix[local_row]
                target = scores[target_col]
                greater = int(np.count_nonzero(scores > target))
                tied = int(np.count_nonzero(scores == target))
                rank = 1 + greater
                prediction = int(np.argmax(scores))
                stable_order = np.argsort(-scores, kind="stable")
                stable_top5 = stable_order[: min(5, len(stable_order))]
                top1_fractional = max(0.0, min(1.0, (1 - greater) / tied))
                top5_k = min(5, len(scores))
                top5_fractional = max(0.0, min(1.0, (top5_k - greater) / tied))
                rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": 42,
                        "method": "NGM-v2",
                        "query_uid": uid,
                        "query_index": query_index,
                        "identity": identity,
                        "reference_uid": medoid_uid,
                        "rank": rank,
                        "top1": int(rank == 1),
                        "top5": int(rank <= min(5, len(scores))),
                        "rr": 1.0 / rank,
                        "hungarian_correct": int(assignment.get(local_row, -1) == int(target_col)),
                        "predicted_index": int(candidate_indices[prediction]),
                        "predicted_identity": candidate_lookup[prediction],
                    }
                )
                tie_rows.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "seed": 42,
                        "query_uid": uid,
                        "query_index": query_index,
                        "gt_identity": identity,
                        "target_col": int(target_col),
                        "target_score": float(target),
                        "strictly_greater_candidates": greater,
                        "equal_score_candidates": tied,
                        "rank_min": greater + 1,
                        "rank_max": greater + tied,
                        "top1_competition_credit": int(greater == 0),
                        "top1_fractional_tie_credit": top1_fractional,
                        "top5_competition_credit": int(greater < min(5, len(scores))),
                        "top5_fractional_tie_credit": top5_fractional,
                        "stable_top5_candidate_cols": "|".join(map(str, stable_top5.tolist())),
                        "stable_top5_identities": "|".join(candidate_lookup[int(i)] for i in stable_top5),
                        "stable_top5_scores": "|".join(format(float(scores[int(i)]), ".9g") for i in stable_top5),
                    }
                )

    metrics = aggregate(rows)
    metrics.update(
        {
            "method": "NGM-v2",
            "dataset": dataset,
            "fold": fold,
            "model_seed": 42,
            "evaluation_split": "test",
            "query_protocol": "canonical_medoid_covered_only",
            "reference_uid": medoid_uid,
            "reference_path": str(medoid_path),
            "reference_selection": "frozen common outer-train geometry medoid",
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_epoch": int(payload["epoch"]),
            "checkpoint_selection": "reused frozen validation-selected checkpoint; no test tuning",
            "training_reference_uid": uid_for(find_training_template(dataset, fold, str(payload["template"]))),
            "ranking_tie_policy": {
                "reported_rank": "competition rank = 1 + number of scores strictly greater than the GT score",
                "reported_top5": "credit 1 iff competition rank <= min(5, candidate count); all exact boundary ties receive credit",
                "additional_audit": "tie_diagnostics.csv also provides fractional expected credit under uniform random ordering within exact ties",
            },
            "score_matrix_artifact": "score_matrices.npz",
            "candidate_order_artifact": "candidate_identities.csv",
        }
    )
    write_query_rows(output_dir / "query_level.csv", rows)
    write_csv_rows(output_dir / "tie_diagnostics.csv", tie_rows)
    candidate_rows = [
        {
            "dataset": dataset,
            "fold": fold,
            "reference_uid": medoid_uid,
            "candidate_col": column,
            "reference_node_index": int(candidate_indices[column]),
            "identity": identity,
        }
        for column, identity in enumerate(candidate_labels)
    ]
    write_csv_rows(output_dir / "candidate_identities.csv", candidate_rows)
    all_scores = np.concatenate(score_blocks, axis=0)
    if all_scores.shape != (len(rows), len(candidate_labels)):
        raise RuntimeError(
            f"NGM score export shape mismatch: {all_scores.shape}, "
            f"expected={(len(rows), len(candidate_labels))}"
        )
    np.savez_compressed(
        output_dir / "score_matrices.npz",
        scores=all_scores,
        query_uid=np.asarray(score_query_uids),
        query_index=np.asarray(score_query_indices, dtype=np.int64),
        gt_identity=np.asarray(score_gt_identities),
        target_col=np.asarray(score_target_columns, dtype=np.int64),
        candidate_identity=np.asarray(candidate_labels),
        candidate_reference_index=np.asarray(candidate_indices, dtype=np.int64),
        reference_uid=np.asarray([medoid_uid]),
        fold=np.asarray([fold], dtype=np.int64),
        seed=np.asarray([42], dtype=np.int64),
    )
    metrics["prediction_replay_guard"] = verify_ngm_prediction_replay(
        dataset, fold, output_dir / "query_level.csv"
    )
    metrics["score_matrix_sha256"] = sha256(output_dir / "score_matrices.npz")
    metrics["candidate_order_sha256"] = sha256(output_dir / "candidate_identities.csv")
    metrics["tie_diagnostics_sha256"] = sha256(output_dir / "tie_diagnostics.csv")
    return metrics


def find_training_template(dataset: str, fold: int, name: str) -> Path:
    train = DATA_ROOTS[dataset] / f"fold_{fold}/train"
    hits = list(train.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"Cannot locate original NGM template {name}: {hits}")
    return hits[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("fgw", "ngmv2"), required=True)
    parser.add_argument("--dataset", choices=("atanas", "rld"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.method == "fgw":
        metrics = evaluate_fgw(args.dataset, args.fold, args.output_dir)
    else:
        metrics = evaluate_ngm(args.dataset, args.fold, args.output_dir, device)
    expected = sum(len(rows) for rows in canonical_queries(args.dataset, args.fold).values())
    if int(metrics["queries"]) != expected:
        raise RuntimeError(f"Covered-cohort count mismatch: observed={metrics['queries']}, expected={expected}")
    output = args.output_dir / "test_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
