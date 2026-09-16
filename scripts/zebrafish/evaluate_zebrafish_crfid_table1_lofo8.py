#!/usr/bin/env python3
"""Table-1-core CRF_ID evaluation for zebrafish LOFO8.

The active CRF implementation is the byte-identical Table 1 MATLAB adapter:
fully connected relational graph, uniform node potentials, relative-angle edge
potentials, UGM sum-product LBP, duplicate-resolution loop and conserved node
beliefs.  For each physical q/r pair, a relational atlas is built from all
other unique time points of that fish using stable ``original_row`` IDs.  Both
endpoints and their truth are excluded from atlas construction and inference.

The Table 2 pair score is the overlap of the two Table-1 belief matrices,
``B_q @ B_r.T``; ranking and Hungarian evaluation use the shared benchmark
implementation.  This is a new multi-reference input condition and must not be
presented as the archived single-reference result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat, savemat


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.zebrafish.evaluate_zebrafish_statatlas_table1_lofo8 import (
    PairEpisode,
    endpoint_ids,
    load_pair_episodes,
    pair_manifest_rows,
    sha256_file,
    split_manifest,
    summarize,
    target_arrays,
)
from scripts.zebrafish.query_record_io import QUERY_COLUMNS, records_from_score_matrix


SCHEMA = "zebrafish-crfid-table1-core-lofo8-v1"
METHOD = "crf_id_dagger"
DISPLAY = "CRF_ID†"
OFFICIAL_COMMIT = "1166cdb8b26ca1851011112b620e41b044012518"
FIXED_CONFIG = {
    "graph": "fully_connected",
    "node_potential": "uniform",
    "lambda_PA": 0.0,
    "lambda_LR": 0.0,
    "lambda_DV": 0.0,
    "lambda_geo": 0.0,
    "lambda_angle": 1.0,
    "inference": "UGM_Infer_Conditional+UGM_Infer_LBP",
    "duplicate_resolution": "official iterative clamping",
    "pair_score_fn": "B_q @ B_r.T",
}
MATLAB_DIR = ROOT / "scripts/zebrafish/matlab"
ADAPTER = MATLAB_DIR / "annotation_CRF_atanas.m"
DRIVER = MATLAB_DIR / "run_crfid_table1_jobs.m"


def ordered_hash(values: Iterable[Any]) -> str:
    payload = json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes(crfid_root: Path) -> dict[str, str]:
    paths = {
        "evaluator": Path(__file__).resolve(),
        "table1_matlab_adapter": ADAPTER,
        "matlab_batch_driver": DRIVER,
        "official_get_relative_angles": crfid_root / "Runs/Run_MultiCellCalciumImaging/CRF/get_relative_angles.m",
        "official_duplicate_labels": crfid_root / "Runs/Run_MultiCellCalciumImaging/CRF/duplicate_labels.m",
        "official_compare_hidden": crfid_root / "Runs/Run_MultiCellCalciumImaging/CRF/compare_labels_of_hidden_landmarks.m",
        "official_ugm_conditional": crfid_root / "UGM/infer/UGM_Infer_Conditional.m",
        "official_ugm_lbp": crfid_root / "UGM/infer/UGM_Infer_LBP.m",
        "query_record_io": ROOT / "scripts/zebrafish/query_record_io.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing CRF_ID sources: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def normalize_cloud(xyz: np.ndarray) -> np.ndarray:
    value = np.asarray(xyz, dtype=np.float64)
    value = value - value.mean(axis=0, keepdims=True)
    rms = float(np.sqrt(np.mean(np.sum(value * value, axis=1))))
    if not np.isfinite(rms) or rms < 1e-12:
        raise RuntimeError("degenerate coordinate cloud")
    return value / rms


def matlab_strings(values: list[str]) -> np.ndarray:
    output = np.empty((len(values), 1), dtype=object)
    for index, value in enumerate(values):
        output[index, 0] = value
    return output


def build_relational_atlas(pair: PairEpisode, path: Path) -> dict[str, Any]:
    observations: dict[int, list[np.ndarray]] = {}
    for reference in pair.references:
        xyz = normalize_cloud(reference.xyz)
        for identity, point in zip(reference.tracking_ids, xyz):
            observations.setdefault(int(identity), []).append(point)
    identities = sorted(observations)
    atlas_xyz = np.stack([
        np.mean(np.asarray(observations[identity]), axis=0) for identity in identities
    ])

    # The official Table 1 atlas has more states than query nodes.  A generic
    # latent state is added only when the reference-only vocabulary is smaller
    # than an endpoint's unlabeled node count; no endpoint ID is inspected.
    required_states = max(len(pair.q.xyz), len(pair.r.xyz))
    dummy_count = max(0, required_states - len(identities))
    names = [f"cell{identity}" for identity in identities]
    if dummy_count:
        scale = max(float(np.sqrt(np.mean(np.sum(atlas_xyz * atlas_xyz, axis=1)))), 1.0)
        for dummy in range(dummy_count):
            theta = 2.0 * np.pi * (dummy + 1) / (dummy_count + 1)
            atlas_xyz = np.vstack((atlas_xyz, scale * np.asarray([np.cos(theta), np.sin(theta), 1.0])))
            names.append(f"__DUMMY_{dummy:03d}")

    states = len(names)
    zeros = np.zeros((states, states), dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, {
        "Neuron_head": matlab_strings(names),
        "X_rot": atlas_xyz[:, 0].reshape(-1, 1),
        "Y_rot": atlas_xyz[:, 1].reshape(-1, 1),
        "Z_rot": atlas_xyz[:, 2].reshape(-1, 1),
        "PA_matrix": zeros,
        "LR_matrix": zeros,
        "DV_matrix": zeros,
        "geo_dist": zeros,
    }, do_compression=True)
    return {
        "reference_identities": len(identities),
        "states": states,
        "dummy_states": dummy_count,
        "observation_count_min": min(map(len, observations.values())),
        "observation_count_max": max(map(len, observations.values())),
        "candidate_order_sha256": ordered_hash(names),
    }


def prepare_jobs(episodes: list[PairEpisode], work_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    jobs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    work_dir.mkdir(parents=True, exist_ok=True)
    for pair_index, pair in enumerate(episodes):
        atlas_file = work_dir / f"pair_{pair_index:03d}_atlas.mat"
        atlas_info = build_relational_atlas(pair, atlas_file)
        for side, endpoint in (("q", pair.q), ("r", pair.r)):
            input_file = work_dir / f"pair_{pair_index:03d}_{side}_input.mat"
            output_file = work_dir / f"pair_{pair_index:03d}_{side}_output.mat"
            # This MAT file contains coordinates only. Endpoint tracking IDs
            # remain exclusively in the Python metric process.
            savemat(input_file, {"mu_marker": normalize_cloud(endpoint.xyz)}, do_compression=True)
            jobs.append({
                "job_id": f"{pair.pair_id}__{side}",
                "input_file": str(input_file.resolve()),
                "atlas_file": str(atlas_file.resolve()),
                "output_file": str(output_file.resolve()),
            })
        diagnostics.append({
            "pair_index": pair_index,
            "pair_id": pair.pair_id,
            "specimen_id": pair.q.specimen_id,
            "q_start_frame": pair.q.start_frame,
            "r_start_frame": pair.r.start_frame,
            "reference_count": len(pair.references),
            "reference_start_frames": [record.start_frame for record in pair.references],
            "q_and_r_excluded_from_atlas": True,
            "endpoint_truth_in_matlab_input": False,
            "atlas": atlas_info,
        })
    return jobs, diagnostics


def run_matlab(jobs: list[dict[str, Any]], work_dir: Path, matlab: Path, crfid_root: Path) -> None:
    manifest = work_dir / "matlab_jobs.json"
    write_json(manifest, jobs)
    expression = (
        "run_crfid_table1_jobs('" + str(manifest.resolve()).replace("'", "''") + "','"
        + str(crfid_root.resolve()).replace("'", "''") + "')"
    )
    command = [str(matlab), "-sd", str(MATLAB_DIR), "-batch", expression]
    subprocess.run(command, check=True)
    missing = [job["output_file"] for job in jobs if not Path(job["output_file"]).is_file()]
    if missing:
        raise RuntimeError(f"MATLAB did not emit outputs: {missing}")


def mstr(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode().strip()
    value = np.squeeze(value)
    if value.ndim == 0:
        return mstr(value.item())
    if value.dtype.kind in ("U", "S"):
        return "".join(str(item) for item in value.reshape(-1)).strip()
    if value.size == 1:
        return mstr(value.item())
    return str(value).strip()


def load_beliefs(path: Path, expected_nodes: int) -> tuple[np.ndarray, list[str]]:
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    beliefs = np.asarray(data["conserved_nodeBel"], dtype=np.float64)
    if beliefs.ndim == 1:
        beliefs = beliefs.reshape(1, -1)
    names = [mstr(value) for value in np.asarray(data["Neuron_head"]).reshape(-1)]
    if beliefs.shape != (expected_nodes, len(names)):
        raise RuntimeError(f"malformed CRF output {path}: {beliefs.shape}, states={len(names)}")
    if not np.isfinite(beliefs).all() or not np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError(f"invalid CRF beliefs: {path}")
    return beliefs, names


def evaluate_outputs(episodes: list[PairEpisode], work_dir: Path, fold: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_hashes: list[dict[str, str]] = []
    for pair_index, pair in enumerate(episodes):
        q_path = work_dir / f"pair_{pair_index:03d}_q_output.mat"
        r_path = work_dir / f"pair_{pair_index:03d}_r_output.mat"
        bq, q_names = load_beliefs(q_path, len(pair.q.xyz))
        br, r_names = load_beliefs(r_path, len(pair.r.xyz))
        if q_names != r_names:
            raise RuntimeError(f"{pair.pair_id}: q/r CRF state order differs")
        score = bq @ br.T
        row_target, col_target = target_arrays(pair)
        rows.extend(records_from_score_matrix(
            method=METHOD,
            fold=fold,
            seed=None,
            pair_index=pair_index,
            pair_id=pair.pair_id,
            score=score,
            q_uid=f"{pair.pair_id}__q",
            r_uid=f"{pair.pair_id}__r",
            q_ids=endpoint_ids(pair.pair_id, pair.q.tracking_ids),
            r_ids=endpoint_ids(pair.pair_id, pair.r.tracking_ids),
            row_target=row_target,
            col_target=col_target,
        ))
        output_hashes.extend([
            {"file": q_path.name, "sha256": sha256_file(q_path)},
            {"file": r_path.name, "sha256": sha256_file(r_path)},
        ])
    return rows, {"files": output_hashes, "combined_sha256": ordered_hash(output_hashes)}


def select_fold(args: argparse.Namespace, fold: int) -> None:
    fold_root = args.data_root / f"fold_{fold}" / "matching"
    print(f"[SELECT] fold={fold}: singleton Table 1 configuration; opening VAL only", flush=True)
    episodes = load_pair_episodes(fold_root / "val")
    work = args.run_root / "work" / "val" / f"fold_{fold}"
    jobs, diagnostics = prepare_jobs(episodes, work)
    run_matlab(jobs, work, args.matlab, args.crfid_root)
    rows, raw_outputs = evaluate_outputs(episodes, work, fold)
    validation_metrics = summarize(rows)
    lock = {
        "schema": SCHEMA,
        "status": "LOCKED_BEFORE_TEST",
        "fold": fold,
        "method": METHOD,
        "display_name": DISPLAY,
        "seed": None,
        "deterministic": True,
        "randomness_audit": "adapter calls rng shuffle; active no-landmark CRF/UGM path consumes no random values",
        "official_commit": OFFICIAL_COMMIT,
        "source_hashes": source_hashes(args.crfid_root),
        "selection_split": "val",
        "test_opened_during_selection": False,
        "reference_protocol": "same-fish leave-two-timepoints-out per physical q/r pair",
        "endpoint_truth_used_in_atlas_or_scoring": False,
        "candidate_order": [FIXED_CONFIG],
        "selected_candidate_index": 0,
        "selected_config": FIXED_CONFIG,
        "candidate_results": [{"candidate_index": 0, "config": FIXED_CONFIG, "validation_metrics": validation_metrics}],
        "val_pair_count": len(episodes),
        "val_diagnostics_hash": ordered_hash(diagnostics),
        "val_matlab_outputs": raw_outputs,
        "train_manifest": split_manifest(fold_root / "train"),
        "val_manifest": split_manifest(fold_root / "val"),
    }
    path = args.run_root / "locks" / f"fold_{fold}.json"
    write_json(path, lock)
    print(f"[LOCKED] {path}: {validation_metrics}", flush=True)


def verify_lock(lock: dict[str, Any], args: argparse.Namespace, fold_root: Path, fold: int) -> None:
    guards = (
        lock.get("schema") == SCHEMA,
        int(lock.get("fold", -1)) == fold,
        lock.get("status") == "LOCKED_BEFORE_TEST",
        lock.get("test_opened_during_selection") is False,
        lock.get("selected_config") == FIXED_CONFIG,
        lock.get("source_hashes") == source_hashes(args.crfid_root),
    )
    if not all(guards):
        raise RuntimeError(f"invalid or stale lock for fold {fold}")
    for split in ("train", "val"):
        if lock[f"{split}_manifest"]["sha256"] != split_manifest(fold_root / split)["sha256"]:
            raise RuntimeError(f"{split} data changed after selection")


def test_fold(args: argparse.Namespace, fold: int) -> dict[str, Any]:
    fold_root = args.data_root / f"fold_{fold}" / "matching"
    lock_path = args.run_root / "locks" / f"fold_{fold}.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock(lock, args, fold_root, fold)
    print(f"[TEST] fold={fold}: lock verified; opening TEST", flush=True)
    episodes = load_pair_episodes(fold_root / "test")
    work = args.run_root / "work" / "test" / f"fold_{fold}"
    jobs, diagnostics = prepare_jobs(episodes, work)
    run_matlab(jobs, work, args.matlab, args.crfid_root)
    rows, raw_outputs = evaluate_outputs(episodes, work, fold)
    output_dir = args.run_root / "test" / f"fold_{fold}"
    write_csv(output_dir / "query_predictions.csv", rows, QUERY_COLUMNS)
    write_csv(output_dir / "pair_manifest.csv", pair_manifest_rows(episodes, "test"), pair_manifest_rows(episodes, "test")[0].keys())
    write_json(output_dir / "pair_diagnostics.json", diagnostics)
    result = {
        "schema": SCHEMA,
        "fold": fold,
        "method": METHOD,
        "display_name": DISPLAY,
        "seed": None,
        "deterministic": True,
        "reference_protocol": "same-fish leave-two-timepoints-out per physical q/r pair",
        "selected_config": FIXED_CONFIG,
        "metrics": summarize(rows),
        "physical_pairs": len(episodes),
        "q_and_r_timepoints_used_in_atlas": False,
        "endpoint_truth_used_in_scoring": False,
        "endpoint_truth_used_for_metrics_only": True,
        "test_manifest": split_manifest(fold_root / "test"),
        "lock_sha256": sha256_file(lock_path),
        "source_hashes": source_hashes(args.crfid_root),
        "matlab_outputs": raw_outputs,
    }
    write_json(output_dir / "metrics.json", result)
    return result


def aggregate(args: argparse.Namespace) -> None:
    results = [json.loads((args.run_root / "test" / f"fold_{fold}" / "metrics.json").read_text()) for fold in args.folds]
    output = {
        "schema": SCHEMA,
        "method": METHOD,
        "display_name": DISPLAY,
        "seed": None,
        "deterministic": True,
        "folds": len(results),
        "aggregation": "unweighted mean and sample SD across held-out fish",
        "sample_sd_ddof": 1,
        "reference_protocol": "same-fish leave-two-timepoints-out per physical q/r pair",
        "metrics": {},
    }
    for name in ("top1", "top5", "mrr", "hungarian"):
        values = np.asarray([row["metrics"][name] for row in results], dtype=np.float64)
        output["metrics"][name] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "fold_values": values.tolist()}
    write_json(args.run_root / "aggregate.json", output)
    print(json.dumps(output, indent=2), flush=True)


def audit(args: argparse.Namespace) -> None:
    rows = []
    for fold in args.folds:
        fold_root = args.data_root / f"fold_{fold}" / "matching"
        splits = {}
        for split in ("train", "val", "test"):
            episodes = load_pair_episodes(fold_root / split)
            splits[split] = {
                "physical_pairs": len(episodes),
                "min_references": min(len(pair.references) for pair in episodes),
                "max_references": max(len(pair.references) for pair in episodes),
                "endpoint_exclusion_pass": all(
                    pair.q.start_frame not in {r.start_frame for r in pair.references}
                    and pair.r.start_frame not in {r.start_frame for r in pair.references}
                    for pair in episodes
                ),
            }
        rows.append({"fold": fold, "splits": splits})
    write_json(args.run_root / "data_audit.json", {
        "schema": SCHEMA,
        "status": "PASS",
        "official_commit": OFFICIAL_COMMIT,
        "table1_matlab_adapter_sha256": sha256_file(ADAPTER),
        "reference_protocol": "same-fish leave-two-timepoints-out per physical q/r pair",
        "stable_tracking_id": "original_row",
        "folds": rows,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("audit", "select", "test", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=Path("../Data/Zebrafish_LOFO8_60m"))
    parser.add_argument("--run-root", type=Path, default=Path("../runs/zebrafish_crfid_table1_core_lofo8_v1"))
    parser.add_argument("--crfid-root", type=Path, default=Path("../CRF_Cell_ID"))
    parser.add_argument("--matlab", type=Path, default=Path("/home/ubuntu/MATLAB/R2024b/bin/matlab"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    args = parser.parse_args()
    if any(fold not in range(1, 9) for fold in args.folds):
        raise ValueError("--folds must contain values 1..8")
    args.data_root = args.data_root.resolve()
    args.run_root = args.run_root.resolve()
    args.crfid_root = args.crfid_root.resolve()
    args.matlab = args.matlab.resolve()
    if args.stage == "audit":
        audit(args)
    elif args.stage == "select":
        for fold in args.folds:
            select_fold(args, fold)
    elif args.stage == "test":
        for fold in args.folds:
            test_fold(args, fold)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
