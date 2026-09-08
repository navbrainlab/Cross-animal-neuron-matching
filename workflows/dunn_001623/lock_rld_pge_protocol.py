#!/usr/bin/env python3
"""Audit and immutably lock the fixed RLD PGE checkpoint set before test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


SEEDS = (1, 42, 123)
MODES = ("anatomy", "real")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
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
        "hybrid/extract_locked_rld_pge_embeddings.py",
        "engines/train_fdnc_nuclr_fusion_benchmark.py",
        "engines/train_fdnc_nuclr_relative_geometry_from_v2.py",
        "archive/legacy_experiments/ey/stage1_pretrain_ey_nuclr_official_50k.py",
        "diagnostics/gt_guided_activity_feature_audit_atanas.py",
        "workflows/dunn_001623/prepare_rld_pge_protocol.py",
        "workflows/dunn_001623/lock_rld_pge_protocol.py",
        "engines/evaluate_rld_pge_locked_test.py",
        "engines/summarize_rld_pge_locked.py",
        "engines/run_rld_pge_locked.sh",
    )
    result: dict[str, str] = {}
    for name in relative:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = sha256(path)
    return result


def assert_no_test_artifacts(run_root: Path) -> None:
    forbidden = (
        run_root / "data_test_locked",
        run_root / "pge_embeddings" / "test",
        run_root / "pge_embeddings" / "manifest_outer_test.json",
    )
    for path in forbidden:
        if path.exists():
            raise AssertionError(f"Test artifact exists before locking: {path}")
    for path in (run_root / "pge_fixed_v1").glob("*_seed*/test_results.json"):
        raise AssertionError(f"Test result exists before locking: {path}")


def as_number(value: Any) -> float:
    return float(value)


def assert_nuclr_protocol(run_root: Path, manifest_sha: str) -> tuple[Path, dict[str, Any]]:
    run_dir = run_root / "nuclr_same_only_seed42"
    checkpoint_path = run_dir / "best.pt"
    config_path = run_dir / "config.json"
    audit_path = run_dir / "protocol_audit.json"
    for path in (checkpoint_path, config_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = dict(checkpoint["args"])
    expected = {
        "model_variant": "raw_nuclr",
        "activity_key": "activity_raw",
        "same_worm_neurons": "all",
        "epochs": 100,
        "batch_size": 8,
        "same_views_per_worm": 8,
        "cross_pairs_per_epoch": 0,
        "optimizer_steps_per_epoch": 88,
        "window_seconds": 30.0,
        "eval_num_windows": 32,
        "lambda_same": 1.0,
        "lambda_cross": 0.0,
        "selection_space": "encoder",
        "selection_metric": "top1",
        "seed": 42,
    }
    for key, expected_value in expected.items():
        observed = saved.get(key)
        if isinstance(expected_value, float):
            if as_number(observed) != expected_value:
                raise AssertionError(f"NuCLR {key}={observed}, expected {expected_value}")
        elif observed != expected_value:
            raise AssertionError(f"NuCLR {key}={observed}, expected {expected_value}")
    if saved.get("init_backbone_checkpoint") is not None or saved.get("init_full_checkpoint") is not None:
        raise AssertionError("RLD NuCLR must be randomly initialized")
    if saved.get("split_manifest_sha256") != manifest_sha:
        raise AssertionError("NuCLR checkpoint manifest provenance mismatch")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["counts"] != {"train": 67, "val": 12, "test": 16}:
        raise AssertionError("NuCLR split count audit mismatch")
    if audit.get("test_worms_referenced_by_training_dataloader") != 0:
        raise AssertionError("RLD test worm entered NuCLR training")
    if audit.get("train_identity_labels_loaded"):
        raise AssertionError("Same-only NuCLR unexpectedly loaded train identities")
    if audit.get("same_only_loss_accessed_cell_id"):
        raise AssertionError("Same-only NuCLR accessed training cell IDs")
    return checkpoint_path, {
        "path": str(checkpoint_path.resolve()),
        "sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path),
        "protocol_audit_sha256": sha256(audit_path),
        "best_epoch": int(checkpoint["epoch"]),
    }


def assert_pge_args(saved: dict[str, Any], mode: str, seed: int) -> None:
    expected: dict[str, Any] = {
        "mode": mode,
        "geometry_encoder": "pge",
        "xyz_scale": 200.0,
        "activity_dim": 256,
        "hidden_dim": 128,
        "activity_hidden": 128,
        "activity_dropout": 0.1,
        "activity_gate_init": 0.1,
        "activity_modality_dropout": 0.1,
        "population_layers": 2,
        "population_heads": 4,
        "population_dropout": 0.1,
        "dpge_heads": 4,
        "dpge_stage1_layers": 2,
        "dpge_stage2_layers": 2,
        "dpge_rbf_bins": 8,
        "dpge_local_k": 8,
        "epochs": 100,
        "max_pairs_per_epoch": 200,
        "minimum_shared": 8,
        "geometry_lr": 2e-4,
        "activity_lr": 5e-4,
        "population_lr": 2e-4,
        "outlier_lr": 2e-4,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "patience": 20,
        "grad_clip": 2.0,
        "coordinate_jitter": 0.01,
        "scale_jitter": 0.05,
        "outlier_weight": 0.25,
        "precision": "bf16",
        "seed": seed,
    }
    for key, expected_value in expected.items():
        observed = saved.get(key)
        if isinstance(expected_value, float):
            if as_number(observed) != expected_value:
                raise AssertionError(f"PGE {mode}/seed{seed} {key}={observed}, expected {expected_value}")
        elif observed != expected_value:
            raise AssertionError(f"PGE {mode}/seed{seed} {key}={observed}, expected {expected_value}")


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    run_root = args.run_root.resolve()
    manifest_sha = sha256(args.manifest)
    if manifest_sha != args.expected_manifest_sha256.lower():
        raise AssertionError("Locked RLD manifest SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fold = manifest["folds"][0]
    train_ids = fold["worm_ids"]["train"]
    val_ids = fold["worm_ids"]["val"]
    test_ids = fold["worm_ids"]["test"]
    if (len(train_ids), len(val_ids), len(test_ids)) != (67, 12, 16):
        raise AssertionError("RLD split counts changed")
    assert_no_test_artifacts(run_root)
    nuclr_path, nuclr_info = assert_nuclr_protocol(run_root, manifest_sha)

    embedding_manifest_path = run_root / "pge_embeddings" / "manifest_train_val.json"
    if not embedding_manifest_path.is_file():
        raise FileNotFoundError(embedding_manifest_path)
    embedding_manifest = json.loads(embedding_manifest_path.read_text(encoding="utf-8"))
    if embedding_manifest.get("fdnc_used") or embedding_manifest.get("phase") != "train_val":
        raise AssertionError("Invalid RLD train/val embedding provenance")
    if embedding_manifest.get("split_manifest_sha256") != manifest_sha:
        raise AssertionError("Embedding manifest split provenance mismatch")
    if embedding_manifest.get("nuclr_checkpoint_sha256") != sha256(nuclr_path):
        raise AssertionError("Embedding manifest NuCLR checkpoint mismatch")
    if embedding_manifest["splits"]["train"] != train_ids:
        raise AssertionError("Embedding manifest train IDs mismatch")
    if embedding_manifest["splits"]["val"] != val_ids:
        raise AssertionError("Embedding manifest val IDs mismatch")
    for split in ("train", "val"):
        for worm_id, expected_hash in embedding_manifest["file_sha256"][split].items():
            path = run_root / "pge_embeddings" / split / f"{worm_id}.npz"
            if sha256(path) != expected_hash:
                raise AssertionError(f"Embedding changed after extraction: {path}")

    model_root = run_root / "pge_fixed_v1"
    checkpoints: dict[str, Any] = {}
    shared_hashes: dict[int, dict[str, str]] = {seed: {} for seed in SEEDS}
    validation_queries: dict[int, dict[str, int]] = {seed: {} for seed in SEEDS}
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
            saved = config["args"]
            assert_pge_args(saved, mode, seed)
            if saved.get("split_manifest_sha256") != manifest_sha:
                raise AssertionError(f"{run_dir}: manifest provenance mismatch")
            audit = config["protocol_audit"]
            if audit["counts"] != {"train": 67, "val": 12, "test": 16}:
                raise AssertionError(f"{run_dir}: split-count audit mismatch")
            if audit["outer_test_data_loaded"] or audit["outer_test_metrics_computed"]:
                raise AssertionError(f"{run_dir}: RLD test was accessed")
            if audit["fdnc_checkpoint_loaded"] or audit["fdnc_embedding_array_read"]:
                raise AssertionError(f"{run_dir}: fDNC was used")
            if bool(audit["nuclr_embedding_array_read"]) != (mode == "real"):
                raise AssertionError(f"{run_dir}: activity-read audit mismatch")
            if result.get("outer_test_accessed") or result.get("fdnc_used"):
                raise AssertionError(f"{run_dir}: invalid final audit")
            if sorted(config["train_worms"]) != train_ids:
                raise AssertionError(f"{run_dir}: train IDs mismatch")
            if sorted(config["val_worms"]) != val_ids:
                raise AssertionError(f"{run_dir}: val IDs mismatch")
            shared_hashes[seed][mode] = config["initial_shared_model_sha256"]
            metric = result["validation"][f"pge_{mode}"]
            validation_queries[seed][mode] = int(metric["queries"])
            checkpoints[f"{mode}_seed{seed}"] = {
                "path": str(checkpoint_path.resolve()),
                "sha256": sha256(checkpoint_path),
                "config_sha256": sha256(config_path),
                "result_sha256": sha256(result_path),
                "best_epoch": int(result["best_epoch"]),
                "validation_ranking_top1": float(metric["ranking_top1"]),
            }
    for seed in SEEDS:
        if shared_hashes[seed]["anatomy"] != shared_hashes[seed]["real"]:
            raise AssertionError(f"Seed {seed}: Anatomy/Real initialization mismatch")
        if validation_queries[seed]["anatomy"] != validation_queries[seed]["real"]:
            raise AssertionError(f"Seed {seed}: paired validation query mismatch")

    code = code_hashes(root)
    fold_lock = {
        "format": "rld_pge_fixed_protocol_fold_lock_v1",
        # This state name is intentionally compatible with the audited generic
        # PGE embedding extractor.
        "state": "LOCKED_BEFORE_OUTER_TEST",
        "fold": 1,
        "split_manifest_sha256": manifest_sha,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "outer_test_ids_unopened": test_ids,
        "nuclr_checkpoint": nuclr_info,
        "train_val_embedding_manifest": {
            "path": str(embedding_manifest_path.resolve()),
            "sha256": sha256(embedding_manifest_path),
        },
        "pge_checkpoints": checkpoints,
        "code_sha256": code,
        "configuration_source": "Atanas locked PGE protocol",
        "selection_metric": "RLD validation ranking Top-1; epoch only",
        "primary_endpoint": "PGE-Real minus PGE-Anatomy ranking Top-1",
        "outer_test_accessed": False,
        "fdnc_used": False,
    }
    fold_lock_path = model_root / "FOLD_PROTOCOL_LOCK.json"
    write_once(fold_lock_path, fold_lock)
    global_lock = {
        "format": "rld_pge_fixed_global_protocol_lock_v1",
        "state": "RLD_ALL_CHECKPOINTS_LOCKED_BEFORE_TEST",
        "split_manifest_sha256": manifest_sha,
        "fold_lock": {
            "path": str(fold_lock_path.resolve()),
            "sha256": sha256(fold_lock_path),
        },
        "code_sha256": code,
        "test_worms": 16,
        "model_selection_after_this_lock_forbidden": True,
        "rld_historical_test_exposure_disclosed": True,
        "fdnc_used": False,
    }
    write_once(run_root / "RLD_PGE_PROTOCOL_LOCK.json", global_lock)


if __name__ == "__main__":
    main()
