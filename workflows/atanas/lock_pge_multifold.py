#!/usr/bin/env python3
"""Lock PGE checkpoints before any Atanas outer-test file is materialized."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SEEDS = (1, 42, 123)
MODES = ("anatomy", "real")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--action", choices=["fold", "global"], required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise AssertionError(f"Existing lock differs: {path}")
        print(f"Lock already exists and is identical: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, 0o444)
    print(f"Created immutable lock: {path}")


def code_hashes(root: Path) -> dict[str, str]:
    relative = (
        "hybrid/activity.py",
        "hybrid/extract.py",
        "hybrid/train.py",
        "hybrid/dpge.py",
        "hybrid/train_dpge.py",
        "hybrid/extract_locked_pge_embeddings.py",
        "workflows/atanas/materialize_locked_fold.py",
        "workflows/atanas/materialize_locked_outer_test.py",
        "workflows/atanas/lock_pge_multifold.py",
        "engines/train_fdnc_nuclr_fusion_benchmark.py",
        "engines/train_fdnc_nuclr_relative_geometry_from_v2.py",
        "archive/legacy_experiments/ey/stage1_pretrain_ey_nuclr_official_50k.py",
        "diagnostics/gt_guided_activity_feature_audit_atanas.py",
        "engines/evaluate_atanas_pge_outer_fold.py",
        "engines/summarize_atanas_pge_multifold.py",
        "engines/run_atanas_pge_multifold.sh",
    )
    result = {}
    for name in relative:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = sha256(path)
    return result


def manifest_fold(manifest: dict[str, Any], fold_number: int) -> dict[str, Any]:
    return next(item for item in manifest["folds"] if int(item["fold"]) == fold_number)


def assert_no_outer_artifacts(run_root: Path) -> None:
    forbidden = (
        run_root / "data_outer_test_locked",
        run_root / "pge_embeddings" / "test",
    )
    for path in forbidden:
        if path.exists():
            raise AssertionError(f"Outer-test artifact exists before locking: {path}")
    for path in run_root.glob("pge_cv_v1/*_seed*/outer_test_results.json"):
        raise AssertionError(f"Outer-test result exists before locking: {path}")


def lock_fold(args: argparse.Namespace, manifest: dict[str, Any], manifest_sha: str) -> None:
    if args.fold is None:
        raise ValueError("--fold is required for action=fold")
    root = args.repo_root.resolve()
    fold = manifest_fold(manifest, args.fold)
    run_root = root / "runs" / "atanas_multifold_v1" / f"fold_{args.fold}"
    model_root = run_root / "pge_cv_v1"
    assert_no_outer_artifacts(run_root)
    nuclr_path = run_root / "nuclr_same_only_seed42" / "best.pt"
    if not nuclr_path.is_file():
        raise FileNotFoundError(nuclr_path)
    embedding_manifest_path = run_root / "pge_embeddings" / "manifest_train_val.json"
    if not embedding_manifest_path.is_file():
        raise FileNotFoundError(embedding_manifest_path)
    embedding_manifest = json.loads(
        embedding_manifest_path.read_text(encoding="utf-8")
    )
    if embedding_manifest.get("fdnc_used") or embedding_manifest.get("phase") != "train_val":
        raise AssertionError("Invalid PGE train/val embedding provenance")
    if embedding_manifest.get("split_manifest_sha256") != manifest_sha:
        raise AssertionError("Embedding manifest split provenance mismatch")
    if embedding_manifest.get("nuclr_checkpoint_sha256") != sha256(nuclr_path):
        raise AssertionError("Embedding manifest NuCLR checkpoint mismatch")
    if embedding_manifest["splits"]["train"] != fold["worm_ids"]["train"]:
        raise AssertionError("Embedding manifest train IDs mismatch")
    if embedding_manifest["splits"]["val"] != fold["worm_ids"]["val"]:
        raise AssertionError("Embedding manifest val IDs mismatch")
    for split in ("train", "val"):
        for worm_id, expected_hash in embedding_manifest["file_sha256"][split].items():
            path = run_root / "pge_embeddings" / split / f"{worm_id}.npz"
            if sha256(path) != expected_hash:
                raise AssertionError(f"Embedding changed after extraction: {path}")
    checkpoints: dict[str, Any] = {}
    shared_hashes: dict[int, dict[str, str]] = {seed: {} for seed in SEEDS}
    for mode in MODES:
        for seed in SEEDS:
            run_dir = model_root / f"{mode}_seed{seed}"
            checkpoint_path = run_dir / "best.pt"
            config_path = run_dir / "config.json"
            result_path = run_dir / "final_results.json"
            for path in (checkpoint_path, config_path, result_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            config = json.loads(config_path.read_text(encoding="utf-8"))
            result = json.loads(result_path.read_text(encoding="utf-8"))
            saved_args = config["args"]
            if saved_args["geometry_encoder"] != "pge":
                raise AssertionError(f"{run_dir}: formal encoder is not PGE")
            if saved_args["mode"] != mode or int(saved_args["seed"]) != seed:
                raise AssertionError(f"{run_dir}: mode/seed mismatch")
            if saved_args["split_manifest_sha256"] != manifest_sha:
                raise AssertionError(f"{run_dir}: manifest provenance mismatch")
            audit = config["protocol_audit"]
            if audit["outer_test_data_loaded"] or audit["outer_test_metrics_computed"]:
                raise AssertionError(f"{run_dir}: outer test was accessed")
            if audit["fdnc_checkpoint_loaded"] or audit["fdnc_embedding_array_read"]:
                raise AssertionError(f"{run_dir}: fDNC was used")
            if bool(audit["nuclr_embedding_array_read"]) != (mode == "real"):
                raise AssertionError(f"{run_dir}: activity-read audit mismatch")
            if result.get("outer_test_accessed") or result.get("fdnc_used"):
                raise AssertionError(f"{run_dir}: invalid final audit")
            if sorted(config["train_worms"]) != fold["worm_ids"]["train"]:
                raise AssertionError(f"{run_dir}: train IDs mismatch")
            if sorted(config["val_worms"]) != fold["worm_ids"]["val"]:
                raise AssertionError(f"{run_dir}: val IDs mismatch")
            shared_hashes[seed][mode] = config["initial_shared_model_sha256"]
            metric = result["validation"][f"pge_{mode}"]
            checkpoints[f"{mode}_seed{seed}"] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256(checkpoint_path),
                "config_sha256": sha256(config_path),
                "best_epoch": int(result["best_epoch"]),
                "validation_ranking_top1": float(metric["ranking_top1"]),
            }
    for seed in SEEDS:
        if shared_hashes[seed]["anatomy"] != shared_hashes[seed]["real"]:
            raise AssertionError(f"Seed {seed}: Anatomy/Real initialization mismatch")
    lock = {
        "format": "atanas_pge_fold_protocol_lock_v1",
        "state": "LOCKED_BEFORE_OUTER_TEST",
        "fold": args.fold,
        "split_manifest_sha256": manifest_sha,
        "train_ids": fold["worm_ids"]["train"],
        "val_ids": fold["worm_ids"]["val"],
        "outer_test_ids_unopened": fold["worm_ids"]["test"],
        "nuclr_checkpoint": {
            "path": str(nuclr_path.resolve()),
            "sha256": sha256(nuclr_path),
        },
        "train_val_embedding_manifest": {
            "path": str(embedding_manifest_path.resolve()),
            "sha256": sha256(embedding_manifest_path),
        },
        "pge_checkpoints": checkpoints,
        "code_sha256": code_hashes(root),
        "selection_metric": "inner-validation ranking Top-1",
        "outer_test_accessed": False,
        "fdnc_used": False,
    }
    write_once(model_root / "FOLD_PROTOCOL_LOCK.json", lock)


def lock_global(args: argparse.Namespace, manifest: dict[str, Any], manifest_sha: str) -> None:
    root = args.repo_root.resolve()
    expected_code = code_hashes(root)
    fold_locks: dict[str, Any] = {}
    all_test_ids: list[str] = []
    for fold_number in range(1, 6):
        run_root = root / "runs" / "atanas_multifold_v1" / f"fold_{fold_number}"
        assert_no_outer_artifacts(run_root)
        path = run_root / "pge_cv_v1" / "FOLD_PROTOCOL_LOCK.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        lock = json.loads(path.read_text(encoding="utf-8"))
        if lock.get("state") != "LOCKED_BEFORE_OUTER_TEST":
            raise AssertionError(f"Fold {fold_number} is not locked")
        if lock.get("code_sha256") != expected_code:
            raise AssertionError(f"Fold {fold_number}: code changed between locks")
        if lock.get("split_manifest_sha256") != manifest_sha:
            raise AssertionError(f"Fold {fold_number}: manifest mismatch")
        all_test_ids.extend(lock["outer_test_ids_unopened"])
        fold_locks[str(fold_number)] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
    expected_all = sorted(item["worm_id"] for item in manifest["dataset"]["worms_metadata"])
    if sorted(all_test_ids) != expected_all or len(all_test_ids) != len(set(all_test_ids)):
        raise AssertionError("Outer folds do not partition all worms exactly once")
    lock = {
        "format": "atanas_pge_global_protocol_lock_v1",
        "state": "ALL_FOLDS_LOCKED_BEFORE_ANY_OUTER_TEST",
        "split_manifest_sha256": manifest_sha,
        "code_sha256": expected_code,
        "fold_locks": fold_locks,
        "outer_test_partition_verified": True,
        "model_selection_after_this_lock_forbidden": True,
        "fdnc_used": False,
    }
    output = root / "runs" / "atanas_multifold_v1" / "PGE_GLOBAL_PROTOCOL_LOCK.json"
    write_once(output, lock)


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    manifest_sha = sha256(args.manifest)
    if manifest_sha != args.expected_manifest_sha256.lower():
        raise AssertionError("Locked manifest SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.action == "fold":
        lock_fold(args, manifest, manifest_sha)
    else:
        lock_global(args, manifest, manifest_sha)


if __name__ == "__main__":
    main()
