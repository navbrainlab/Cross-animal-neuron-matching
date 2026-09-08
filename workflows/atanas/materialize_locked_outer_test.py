#!/usr/bin/env python3
"""Materialize a locked fold's outer-test worms only after checkpoint locking."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--fold-lock", type=Path, required=True)
    parser.add_argument("--global-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    observed = sha256(args.manifest)
    if observed != args.expected_sha256.lower():
        raise AssertionError("Locked manifest SHA256 mismatch")
    fold_lock = json.loads(args.fold_lock.read_text(encoding="utf-8"))
    global_lock = json.loads(args.global_lock.read_text(encoding="utf-8"))
    if fold_lock.get("state") != "LOCKED_BEFORE_OUTER_TEST":
        raise AssertionError("Fold checkpoint set is not locked")
    if global_lock.get("state") != "ALL_FOLDS_LOCKED_BEFORE_ANY_OUTER_TEST":
        raise AssertionError("All folds must be locked before test materialization")
    if int(fold_lock.get("fold", -1)) != args.fold:
        raise AssertionError("Fold-lock number mismatch")
    global_entry = global_lock["fold_locks"].get(str(args.fold))
    if global_entry is None or global_entry["sha256"] != sha256(args.fold_lock):
        raise AssertionError("Fold lock differs from global protocol lock")
    if global_lock.get("split_manifest_sha256") != observed:
        raise AssertionError("Global lock manifest mismatch")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fold = next(item for item in manifest["folds"] if int(item["fold"]) == args.fold)
    metadata = {
        item["worm_id"]: Path(item["source_path"])
        for item in manifest["dataset"]["worms_metadata"]
    }
    if (args.output_root / "train").exists() or (args.output_root / "val").exists():
        raise AssertionError("Outer-test root must not contain train/val")
    output_dir = args.output_root / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = fold["worm_ids"]["test"]
    for worm_id in expected_ids:
        source = metadata[worm_id].resolve()
        destination = output_dir / f"{worm_id}.npz"
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise AssertionError(f"Existing symlink mismatch: {destination}")
        else:
            destination.symlink_to(os.path.relpath(source, start=destination.parent))
    observed_ids = sorted(path.stem for path in output_dir.glob("*.npz"))
    if observed_ids != expected_ids:
        raise AssertionError("Materialized test IDs differ from locked manifest")
    audit = {
        "format": "atanas_locked_fold_outer_test_materialization_v1",
        "fold": args.fold,
        "split_manifest_sha256": observed,
        "fold_lock_sha256": sha256(args.fold_lock),
        "global_lock_sha256": sha256(args.global_lock),
        "outer_test_ids": observed_ids,
        "train_files_materialized": 0,
        "val_files_materialized": 0,
    }
    (args.output_root / "materialization.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
