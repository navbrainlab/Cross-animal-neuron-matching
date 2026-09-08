#!/usr/bin/env python3
"""Materialize train/validation symlinks for a locked Atanas fold.

Outer-test data are deliberately not materialized.
"""

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
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    observed = sha256(args.manifest)
    if observed != args.expected_sha256.lower():
        raise AssertionError("Locked manifest SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    fold = next(
        item for item in manifest["folds"] if int(item["fold"]) == args.fold
    )
    metadata = {
        item["worm_id"]: Path(item["source_path"])
        for item in manifest["dataset"]["worms_metadata"]
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_ids = fold["worm_ids"][split]
        for worm_id in expected_ids:
            source = metadata[worm_id].resolve()
            destination = output_dir / f"{worm_id}.npz"
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != source:
                    raise AssertionError(f"Existing symlink mismatch: {destination}")
            else:
                destination.symlink_to(
                    os.path.relpath(source, start=destination.parent)
                )
        observed_ids = sorted(path.stem for path in output_dir.glob("*.npz"))
        if observed_ids != expected_ids:
            raise AssertionError(
                f"{split}: materialized IDs differ from locked manifest"
            )
    if (args.output_root / "test").exists():
        raise AssertionError("Fold materialization must not contain a test directory")
    audit = {
        "format": "atanas_locked_fold_train_val_materialization",
        "fold": args.fold,
        "split_manifest": str(args.manifest.resolve()),
        "split_manifest_sha256": observed,
        "train_ids": fold["worm_ids"]["train"],
        "val_ids": fold["worm_ids"]["val"],
        "outer_test_ids_assertion_only": fold["worm_ids"]["test"],
        "outer_test_files_materialized": 0,
    }
    (args.output_root / "materialization.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
