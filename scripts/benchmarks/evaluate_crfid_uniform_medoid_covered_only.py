#!/usr/bin/env python3
"""Rescore official CRF_ID MATLAB predictions on the shared covered cohort.

CRF_ID retains its fixed official 178-state relational atlas.  The frozen
outer-train medoid is used only to define the same canonical covered-only query
denominator used by every method in the main table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.benchmarks.evaluate_uniform_medoid_covered_only import (  # noqa: E402
    DATA_ROOTS,
    MEDOID_RUNS,
    aggregate,
    frozen_medoid,
    uid_for,
    write_query_rows,
)


PARENT = REPO.parent
OFFICIAL_ROOT = PARENT / "CRF_Cell_ID"
ATLAS = OFFICIAL_ROOT / "sample_run/data_neuron_relationship_annotation_updated.mat"
SOURCE_RUNS = {
    dataset: PARENT / f"runs/crfid_official_singleatlas_{dataset}_cv5_v1"
    for dataset in ("atanas", "rld")
}
ADAPTER = OFFICIAL_ROOT / "Atanas_adapter/annotation_CRF_atanas.m"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def mstr(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, np.str_):
        return str(value).strip()
    if isinstance(value, bytes):
        return value.decode().strip()
    if isinstance(value, np.ndarray):
        value = np.squeeze(value)
        if value.ndim == 0:
            return mstr(value.item())
        if value.dtype.kind in ("U", "S"):
            return "".join(str(item) for item in value.reshape(-1)).strip()
        if value.size == 1:
            return mstr(value.item())
    return str(value).strip()


def canonical_id(value: Any) -> str:
    return str(value).strip().rstrip("?")


def test_file_lookup(dataset: str, fold: int) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted((DATA_ROOTS[dataset] / f"fold_{fold}/test").glob("*.npz")):
        uid = uid_for(path)
        if uid in output:
            raise RuntimeError(f"Duplicate test UID {uid}")
        output[uid] = path
    return output


def canonical_queries(dataset: str, fold: int, protocol: str) -> dict[str, list[tuple[int, str]]]:
    path = MEDOID_RUNS[dataset] / f"fold{fold}/seed42/query_level.csv"
    output: dict[str, list[tuple[int, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if protocol == "covered_only" and int(row["medoid_candidate_present"]) != 1:
                continue
            output.setdefault(row["uid"], []).append((int(row["node_index"]), row["identity"]))
    for values in output.values():
        values.sort()
    if not output:
        raise RuntimeError(f"Empty canonical query universe: {dataset} fold{fold} {protocol}")
    return output


def validate_frame_audit(dataset: str, fold: int, path: Path) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    expected_false_key = f"{dataset}_train_atlas_built"
    guards = (
        int(audit["fold"]) == fold,
        int(audit["atlas_count"]) == 1,
        int(audit["atlas_states"]) == 178,
        audit["frame_calibration"] == "outer-fold train only",
        audit["test_labels_used"] is False,
        audit[expected_false_key] is False,
        Path(audit["atlas_file"]).resolve() == ATLAS.resolve(),
    )
    if not all(guards):
        raise RuntimeError(f"Invalid CRF_ID frame audit: {path}")
    return audit


def evaluate(dataset: str, fold: int, query_protocol: str, output_dir: Path) -> dict[str, Any]:
    source_dir = SOURCE_RUNS[dataset] / f"fold{fold}"
    frame_path = source_dir / "frame_audit.json"
    frame_audit = validate_frame_audit(dataset, fold, frame_path)
    medoid_uid, _ = frozen_medoid(dataset, fold)
    queries = canonical_queries(dataset, fold, query_protocol)
    test_files = test_file_lookup(dataset, fold)
    sidecars = sorted((source_dir / "test").glob("worm_*_sidecar.npz"))
    outputs = [Path(str(path).replace("_sidecar.npz", "_output.mat")) for path in sidecars]
    if not sidecars or any(not path.is_file() for path in outputs):
        raise RuntimeError(f"Incomplete CRF_ID MATLAB outputs under {source_dir / 'test'}")

    rows: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    for sidecar_path, output_path in zip(sidecars, outputs):
        with np.load(sidecar_path, allow_pickle=True) as sidecar:
            source = Path(str(sidecar["source"].item()))
            original_indices = np.asarray(sidecar["orig_idx"], dtype=np.int64)
        uid = uid_for(source)
        if uid not in test_files or source.resolve() != test_files[uid].resolve():
            raise RuntimeError(f"CRF_ID sidecar is not from the frozen fold test split: {source}")
        wanted = queries.get(uid, [])
        if not wanted:
            continue
        seen_uids.add(uid)

        with np.load(source, allow_pickle=False) as data:
            xyz = np.asarray(data["xyz"])
            labels = np.asarray(data["cell_id"]).astype(str)
        finite_indices = np.flatnonzero(np.isfinite(xyz).all(axis=1))
        local_for_original = {int(raw): local for local, raw in enumerate(original_indices)}

        matlab = loadmat(output_path, squeeze_me=True, struct_as_record=False)
        beliefs = np.asarray(matlab["conserved_nodeBel"], dtype=np.float64)
        candidates = [mstr(value) for value in np.asarray(matlab["Neuron_head"]).reshape(-1)]
        if beliefs.shape != (len(original_indices), 178) or len(candidates) != 178:
            raise RuntimeError(f"Malformed CRF_ID output: {output_path} {beliefs.shape}")
        if not np.isfinite(beliefs).all() or not np.allclose(beliefs.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"Invalid CRF_ID beliefs: {output_path}")
        if len(candidates) != len(set(candidates)):
            raise RuntimeError(f"Duplicate CRF_ID atlas states: {output_path}")
        state_to_index = {name: index for index, name in enumerate(candidates)}
        hungarian_rows, hungarian_cols = linear_sum_assignment(-beliefs)
        assignment = {int(row): int(col) for row, col in zip(hungarian_rows, hungarian_cols)}

        for query_index, identity in wanted:
            if query_index >= len(finite_indices):
                raise RuntimeError(f"Canonical post-finite index is out of range: {uid}/{query_index}")
            raw_index = int(finite_indices[query_index])
            if canonical_id(labels[raw_index]) != canonical_id(identity) or raw_index not in local_for_original:
                raise RuntimeError(f"CRF_ID canonical key unavailable: {dataset} fold{fold} {uid} {query_index} {identity}")
            local_index = local_for_original[raw_index]
            scores = beliefs[local_index]
            prediction = int(np.argmax(scores))
            predicted_identity = candidates[prediction]
            target_index = state_to_index.get(canonical_id(identity))
            if target_index is None:
                rank: int | str = ""
                top1 = 0
                top5 = 0
                rr = 0.0
            else:
                ranking = np.argsort(-scores, kind="stable")
                rank = int(np.flatnonzero(ranking == target_index)[0]) + 1
                top1 = int(rank == 1)
                top5 = int(rank <= 5)
                rr = 1.0 / rank
            assigned = assignment.get(local_index, -1)
            hungarian_correct = int(
                target_index is not None and assigned >= 0 and candidates[assigned] == canonical_id(identity)
            )
            rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "seed": "deterministic",
                    "method": "CRF_ID",
                    "query_uid": uid,
                    "query_index": query_index,
                    "identity": identity,
                    "reference_uid": "official_crfid_relational_atlas_178",
                    "rank": rank,
                    "top1": top1,
                    "top5": top5,
                    "rr": rr,
                    "hungarian_correct": hungarian_correct,
                    "predicted_index": prediction,
                    "predicted_identity": predicted_identity,
                }
            )

    if seen_uids != set(queries):
        raise RuntimeError(f"CRF_ID missed canonical query worms: {sorted(set(queries) - seen_uids)}")
    expected = sum(len(value) for value in queries.values())
    if len(rows) != expected:
        raise RuntimeError(f"CRF_ID query count mismatch: {len(rows)} != {expected}")
    metrics = aggregate(rows)
    metrics.update(
        {
            "method": "CRF_ID",
            "dataset": dataset,
            "fold": fold,
            "model_seed": None,
            "deterministic_inference": True,
            "randomness_audit": "adapter calls rng shuffle, but the active no-landmark UGM-LBP path consumes no random values",
            "evaluation_split": "test",
            "query_protocol": query_protocol,
            "cohort_medoid_uid": medoid_uid,
            "reference_representation": "fixed official CRF_ID relational atlas",
            "reference_states": 178,
            "reference_atlas": str(ATLAS),
            "reference_atlas_sha256": sha256(ATLAS),
            "crfid_candidate_covered": sum(row["rank"] != "" for row in rows),
            "crfid_candidate_coverage": sum(row["rank"] != "" for row in rows) / len(rows),
            "frame_calibration": "outer-fold train only",
            "frame_shared_train_ids": int(frame_audit["shared_train_ids"]),
            "frame_alignment_rmse": float(frame_audit["alignment_rmse"]),
            "frame_audit": str(frame_path),
            "frame_audit_sha256": sha256(frame_path),
            "official_adapter": str(ADAPTER),
            "official_adapter_sha256": sha256(ADAPTER),
            "matlab_outputs": len(outputs),
            "matlab_outputs_combined_sha256": combined_sha256(outputs),
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_query_rows(output_dir / "query_level.csv", rows)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("atanas", "rld"), required=True)
    parser.add_argument("--fold", type=int, choices=range(5), required=True)
    parser.add_argument(
        "--query-protocol", choices=("canonical_all", "covered_only"), default="canonical_all"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.dataset, args.fold, args.query_protocol, args.output_dir.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
