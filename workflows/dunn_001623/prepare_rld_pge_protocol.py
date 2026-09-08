#!/usr/bin/env python3
"""Create and materialize the immutable RLD 67/12/16 PGE protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {"train": 67, "val": 12, "test": 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["manifest", "train_val", "test"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-summary", type=Path, required=True)
    parser.add_argument("--expected-summary-sha256", required=True)
    parser.add_argument("--workflow-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--protocol-lock", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def date_from_id(worm_id: str) -> str:
    token = worm_id[:8]
    if len(token) != 8 or not token.isdigit():
        raise ValueError(f"Cannot infer date from {worm_id}")
    return f"{token[:4]}-{token[4:6]}-{token[6:]}"


def write_once(path: Path, text: str, mode: int = 0o444) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise AssertionError(f"Existing immutable file differs: {path}")
        print(f"Immutable file already exists and is identical: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    os.chmod(path, mode)
    print(f"Created immutable file: {path}")


def collect_splits(source_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for split, expected in EXPECTED_COUNTS.items():
        split_root = source_root / split
        paths = sorted(split_root.glob("*.npz"))
        if len(paths) != expected:
            raise AssertionError(
                f"{split}: expected {expected} NPZ files, found {len(paths)}"
            )
        result[split] = paths
    ids = {split: {path.stem for path in paths} for split, paths in result.items()}
    if ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"]:
        raise AssertionError("RLD worm-level splits overlap")
    dates = {split: {date_from_id(item) for item in values} for split, values in ids.items()}
    if dates["train"] & dates["val"] or dates["train"] & dates["test"] or dates["val"] & dates["test"]:
        raise AssertionError("RLD date groups overlap")
    return result


def build_manifest(args: argparse.Namespace) -> None:
    source_root = args.source_root.resolve()
    split_summary = args.split_summary.resolve()
    observed_summary_sha = sha256(split_summary)
    if observed_summary_sha != args.expected_summary_sha256.lower():
        raise AssertionError(
            f"split_summary SHA256 mismatch: {observed_summary_sha}"
        )
    summary = json.loads(split_summary.read_text(encoding="utf-8"))
    if int(summary.get("num_recordings", -1)) != 95:
        raise AssertionError("Expected the canonical full95 RLD summary")
    paths = collect_splits(source_root)
    worm_ids = {
        split: [path.stem for path in split_paths]
        for split, split_paths in paths.items()
    }
    metadata = []
    for split in ("train", "val", "test"):
        for path in paths[split]:
            metadata.append(
                {
                    "worm_id": path.stem,
                    "split": split,
                    "recording_date": date_from_id(path.stem),
                    "source_path": str(path.resolve()),
                }
            )
    manifest: dict[str, Any] = {
        "format": "rld_pge_atanas_fixed_protocol_manifest_v1",
        "dataset": {
            "name": "Dunn/RLD DANDI 001623 full95",
            "source_root": str(source_root),
            "split_summary": str(split_summary),
            "split_summary_sha256": observed_summary_sha,
            "worms_metadata": metadata,
        },
        # Keep the same fold-shaped interface used by the already-audited PGE
        # extractor, but this is one fixed chronological external split.
        "folds": [
            {
                "fold": 1,
                "worm_ids": worm_ids,
                "date_groups": {
                    split: sorted({date_from_id(item) for item in worm_ids[split]})
                    for split in ("train", "val", "test")
                },
            }
        ],
        "protocol": {
            "split": "fixed chronological date-disjoint 67/12/16",
            "configuration_source": "Atanas PGE protocol locked before RLD evaluation",
            "nuclr": "random initialization; Same-only; 32-window encoder; seed42",
            "pge": "no deformation; two geometry-attention stages; match-only",
            "downstream_seeds": [1, 42, 123],
            "primary_endpoint": "PGE-Real minus PGE-Anatomy ranking Top-1",
            "rld_historical_test_exposure_disclosed": True,
        },
    }
    text = json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    manifest_path = args.workflow_root / "LOCKED_SPLIT_MANIFEST.json"
    write_once(manifest_path, text)
    manifest_sha = sha256(manifest_path)
    write_once(
        args.workflow_root / "LOCKED_SPLIT_MANIFEST.sha256",
        manifest_sha + "\n",
    )
    for split in ("train", "val", "test"):
        write_once(
            args.workflow_root / f"{split}_ids.txt",
            "\n".join(worm_ids[split]) + "\n",
        )
    print(json.dumps({"manifest_sha256": manifest_sha, "counts": EXPECTED_COUNTS}, indent=2))


def verify_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    if args.manifest is None or not args.manifest.is_file():
        raise FileNotFoundError("--manifest is required for materialization")
    observed = sha256(args.manifest)
    if not args.expected_manifest_sha256 or observed != args.expected_manifest_sha256.lower():
        raise AssertionError(f"Locked manifest SHA256 mismatch: {observed}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "rld_pge_atanas_fixed_protocol_manifest_v1":
        raise AssertionError("Unexpected RLD manifest format")
    if manifest["dataset"]["split_summary_sha256"] != args.expected_summary_sha256.lower():
        raise AssertionError("Manifest source-summary provenance mismatch")
    return manifest, observed


def verify_test_lock(args: argparse.Namespace, manifest_sha: str) -> dict[str, Any]:
    if args.protocol_lock is None or not args.protocol_lock.is_file():
        raise AssertionError("Test materialization requires --protocol-lock")
    lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"))
    if lock.get("state") != "LOCKED_BEFORE_OUTER_TEST":
        raise AssertionError("RLD checkpoints were not locked before test access")
    if lock.get("split_manifest_sha256") != manifest_sha:
        raise AssertionError("Protocol-lock manifest mismatch")
    return lock


def materialize(args: argparse.Namespace) -> None:
    if args.output_root is None:
        raise ValueError("--output-root is required for materialization")
    manifest, manifest_sha = verify_manifest(args)
    phase_splits = ("train", "val") if args.action == "train_val" else ("test",)
    if args.action == "test":
        lock = verify_test_lock(args, manifest_sha)
        lock_sha = sha256(args.protocol_lock)
    else:
        lock = None
        lock_sha = None
    forbidden = ("test",) if args.action == "train_val" else ("train", "val")
    for split in forbidden:
        if (args.output_root / split).exists():
            raise AssertionError(f"Forbidden split exists in materialization root: {split}")
    metadata = {
        item["worm_id"]: item for item in manifest["dataset"]["worms_metadata"]
    }
    fold = manifest["folds"][0]
    audit_splits: dict[str, list[str]] = {}
    for split in phase_splits:
        output_dir = args.output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)
        expected_ids = fold["worm_ids"][split]
        for worm_id in expected_ids:
            entry = metadata[worm_id]
            if entry["split"] != split:
                raise AssertionError(f"Manifest split mismatch for {worm_id}")
            source = Path(entry["source_path"]).resolve()
            destination = output_dir / f"{worm_id}.npz"
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != source:
                    raise AssertionError(f"Existing symlink mismatch: {destination}")
            else:
                destination.symlink_to(os.path.relpath(source, start=destination.parent))
        observed_ids = sorted(path.stem for path in output_dir.glob("*.npz"))
        if observed_ids != expected_ids:
            raise AssertionError(f"Materialized {split} IDs differ from manifest")
        audit_splits[split] = observed_ids
    audit = {
        "format": f"rld_pge_{args.action}_materialization_v1",
        "split_manifest_sha256": manifest_sha,
        "protocol_lock_sha256": lock_sha,
        "checkpoint_set_locked_before_test": lock is not None,
        "splits": audit_splits,
    }
    path = args.output_root / f"materialization_{args.action}.json"
    text = json.dumps(audit, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise AssertionError(f"Materialization audit differs: {path}")
    path.write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.split_summary = args.split_summary.resolve()
    args.workflow_root = args.workflow_root.resolve()
    if args.output_root is not None:
        args.output_root = args.output_root.resolve()
    if args.action == "manifest":
        build_manifest(args)
    else:
        materialize(args)


if __name__ == "__main__":
    main()
