#!/usr/bin/env python3
"""Prepare leakage-safe zebrafish LOFO8 inputs for MPRT-Net.

``prepare`` converts train/validation only. ``unlock-test`` is accepted only
after a complete pre-test checkpoint manifest exists.  Pair-local identity
strings are converted from NumPy object arrays to fixed-width Unicode so the
MPRT loader can retain ``allow_pickle=False``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


VARIANTS = ("full", "no_transport", "geometry_only", "activity_only")
SEEDS = (42,)
REQUIRED_KEYS = {"activity_raw", "xyz", "cell_id", "pair_id", "side"}


def parse_folds(value: str) -> list[int]:
    if value.strip().lower() in {"all", "1-8"}:
        return list(range(1, 9))
    folds = sorted({int(x) for x in value.replace(",", " ").split()})
    if not folds or any(x < 1 or x > 8 for x in folds):
        raise ValueError("folds must be a subset of 1..8")
    return folds


def scalar_text(z: np.lib.npyio.NpzFile, key: str, default: str = "") -> str:
    if key not in z.files:
        return default
    values = np.asarray(z[key]).reshape(-1)
    if values.size == 0:
        return default
    value = values[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value)


def unicode_array(values: np.ndarray) -> np.ndarray:
    rendered = []
    for value in np.asarray(values).reshape(-1).tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        rendered.append(str(value))
    width = max(1, max(map(len, rendered), default=1))
    return np.asarray(rendered, dtype=f"<U{width}")


def bool_array(
    z: np.lib.npyio.NpzFile,
    key: str,
    n: int,
    default: bool,
) -> np.ndarray:
    if key not in z.files:
        return np.full(n, default, dtype=bool)
    value = np.asarray(z[key], dtype=bool).reshape(-1)
    if value.shape != (n,):
        raise ValueError(f"{key}: expected ({n},), got {value.shape}")
    return value


def convert_record(source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=True) as z:
        missing = REQUIRED_KEYS.difference(z.files)
        if missing:
            raise KeyError(f"{source}: missing {sorted(missing)}")

        activity = np.asarray(z["activity_raw"], dtype=np.float32)
        xyz = np.asarray(z["xyz"], dtype=np.float32)
        labels = unicode_array(z["cell_id"])
        n = len(labels)
        if activity.shape != (n, 128):
            raise ValueError(f"{source}: expected activity ({n},128), got {activity.shape}")
        if xyz.shape != (n, 3):
            raise ValueError(f"{source}: expected xyz ({n},3), got {xyz.shape}")
        if not np.isfinite(activity).all() or not np.isfinite(xyz).all():
            raise ValueError(f"{source}: non-finite model input")

        pair_id = scalar_text(z, "pair_id")
        side = scalar_text(z, "side")
        specimen_id = scalar_text(z, "specimen_id")
        recording_uid = scalar_text(z, "recording_uid", source.stem)
        if not pair_id or side not in {"q", "r"}:
            raise ValueError(f"{source}: invalid pair_id/side")
        prefix = pair_id + "::cell"
        if any(not label.startswith(prefix) for label in labels.tolist()):
            raise ValueError(f"{source}: labels are not pair-local to {pair_id}")

        clean = bool_array(z, "clean_mask", n, True)
        labeled = bool_array(z, "labeled_mask", n, True)
        certain = bool_array(z, "certain_mask", n, True)

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        activity_raw=activity,
        xyz=xyz,
        cell_id=labels,
        clean_mask=clean,
        labeled_mask=labeled,
        certain_mask=certain,
        valid_xyz_mask=np.isfinite(xyz).all(axis=1),
        recording_uid=np.asarray(recording_uid),
        specimen_id=np.asarray(specimen_id),
        pair_id=np.asarray(pair_id),
        side=np.asarray(side),
        source_path=np.asarray(str(source.resolve())),
        source_sampling_rate_hz=np.asarray(4.0, dtype=np.float32),
        stored_activity_points=np.asarray(128, dtype=np.int64),
        window_seconds=np.asarray(30.0, dtype=np.float32),
        gap_minutes=np.asarray(60.0, dtype=np.float32),
    )


def unique_clean_ids(path: Path) -> frozenset[str]:
    with np.load(path, allow_pickle=False) as z:
        ids = np.asarray(z["cell_id"]).astype(str)
        n = len(ids)
        mask = (
            bool_array(z, "clean_mask", n, True)
            & bool_array(z, "labeled_mask", n, True)
            & bool_array(z, "certain_mask", n, True)
        )
    values = [value for value, keep in zip(ids.tolist(), mask.tolist()) if keep]
    counts = Counter(values)
    return frozenset(value for value, count in counts.items() if count == 1)


def audit_split(root: Path, split: str, min_shared: int) -> dict:
    files = sorted((root / split).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No converted files in {root / split}")

    groups: dict[str, dict[str, Path]] = {}
    identities: dict[Path, frozenset[str]] = {}
    specimens: set[str] = set()
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if np.asarray(z["cell_id"]).dtype.kind not in {"U", "S"}:
                raise RuntimeError(f"{path}: unsafe cell_id dtype")
            pair_id = scalar_text(z, "pair_id")
            side = scalar_text(z, "side")
            specimen = scalar_text(z, "specimen_id")
            activity = np.asarray(z["activity_raw"])
            xyz = np.asarray(z["xyz"])
            if activity.ndim != 2 or activity.shape[1] != 128:
                raise RuntimeError(f"{path}: activity shape={activity.shape}")
            if xyz.shape != (activity.shape[0], 3):
                raise RuntimeError(f"{path}: xyz/activity mismatch")
        if not pair_id or side not in {"q", "r"}:
            raise RuntimeError(f"{path}: invalid pair metadata")
        if side in groups.setdefault(pair_id, {}):
            raise RuntimeError(f"{split}: duplicate {pair_id}/{side}")
        groups[pair_id][side] = path
        identities[path] = unique_clean_ids(path)
        specimens.add(specimen)

    shared_counts = []
    for pair_id, sides in groups.items():
        if set(sides) != {"q", "r"}:
            raise RuntimeError(f"{split}: {pair_id} has sides {sorted(sides)}")
        shared = len(identities[sides["q"]].intersection(identities[sides["r"]]))
        if shared < min_shared:
            raise RuntimeError(f"{split}: {pair_id} shared={shared} < {min_shared}")
        shared_counts.append(shared)

    # Pair-local prefixes prove that unrelated pairs cannot share identities.
    for path, values in identities.items():
        with np.load(path, allow_pickle=False) as z:
            pair_id = scalar_text(z, "pair_id")
        if any(not value.startswith(pair_id + "::cell") for value in values):
            raise RuntimeError(f"{path}: cross-pair identity leakage")

    return {
        "split": split,
        "records": len(files),
        "pairs": len(groups),
        "specimens": sorted(specimens),
        "minimum_shared": min(shared_counts),
        "maximum_shared": max(shared_counts),
        "queries_directional_not_precomputed": True,
    }


def convert_split(source: Path, output_fold: Path, split: str, min_shared: int) -> dict:
    target = output_fold / split
    if target.exists():
        report = audit_split(output_fold, split, min_shared)
        print(f"[REUSE] {target}")
        return report

    source_paths = sorted((source / split).glob("*.npz"))
    if not source_paths:
        raise FileNotFoundError(f"No source files in {source / split}")
    building = output_fold / f"{split}.building"
    if building.exists():
        raise FileExistsError(f"Stale build directory exists: {building}")
    building.mkdir(parents=True)
    try:
        for path in source_paths:
            convert_record(path, building / path.name)
        building.rename(target)
        return audit_split(output_fold, split, min_shared)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise


def validate_unlock_manifest(path: Path, folds: list[int]) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Pre-test lock manifest is missing: {path}")
    payload = json.loads(path.read_text())
    if not payload.get("all_training_complete_before_test"):
        raise RuntimeError("Manifest does not certify completed pre-test training")
    entries = payload.get("checkpoints", [])
    found = {
        (int(row["fold"]), int(row["seed"]), str(row["variant"]))
        for row in entries
    }
    required = {
        (fold, seed, variant)
        for fold in folds
        for seed in SEEDS
        for variant in VARIANTS
    }
    missing = sorted(required.difference(found))
    if missing:
        raise RuntimeError(f"Unlock manifest is incomplete: {missing[:5]}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "audit", "unlock-test"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("Data/Zebrafish_LOFO8_joint_from_scratch"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/Zebrafish_MPRT_LOFO8_60m"),
    )
    parser.add_argument("--folds", default="1-8")
    parser.add_argument("--min-shared", type=int, default=20)
    parser.add_argument("--unlock-manifest", type=Path, default=None)
    args = parser.parse_args()

    folds = parse_folds(args.folds)
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    unlock_payload = None
    if args.action == "unlock-test":
        if args.unlock_manifest is None:
            raise ValueError("unlock-test requires --unlock-manifest")
        unlock_payload = validate_unlock_manifest(args.unlock_manifest.resolve(), folds)

    all_reports = {}
    for fold in folds:
        source_fold = source_root / f"fold_{fold}" / "features"
        output_fold = output_root / f"fold_{fold}"
        output_fold.mkdir(parents=True, exist_ok=True)

        if args.action == "prepare":
            reports = {
                split: convert_split(source_fold, output_fold, split, args.min_shared)
                for split in ("train", "val")
            }
        elif args.action == "audit":
            reports = {
                split: audit_split(output_fold, split, args.min_shared)
                for split in ("train", "val")
            }
            if (output_fold / "test").is_dir():
                reports["test"] = audit_split(output_fold, "test", args.min_shared)
        else:
            reports = {
                "train": audit_split(output_fold, "train", args.min_shared),
                "val": audit_split(output_fold, "val", args.min_shared),
                "test": convert_split(source_fold, output_fold, "test", args.min_shared),
            }

        source_protocol_path = source_root / f"fold_{fold}" / "protocol_from_scratch.json"
        source_protocol = (
            json.loads(source_protocol_path.read_text())
            if source_protocol_path.is_file()
            else {}
        )
        protocol = {
            "protocol": "zebrafish_MPRT_LOFO8_60m_seed42",
            "fold": fold,
            "train_fish": source_protocol.get("train_fish", reports["train"]["specimens"]),
            "val_fish": source_protocol.get("val_fish", reports["val"]["specimens"]),
            "test_fish": source_protocol.get("test_fish"),
            "gap_minutes": 60,
            "window_seconds": 30,
            "activity_points": 128,
            "minimum_shared": args.min_shared,
            "identity_scope": "pair-local; no canonical identity shared across fish",
            "test_unlocked": "test" in reports,
            "unlock_manifest": str(args.unlock_manifest.resolve()) if args.unlock_manifest else None,
            "splits": reports,
        }
        (output_fold / "seed42_protocol.json").write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        all_reports[str(fold)] = protocol
        print(
            f"fold={fold} train_pairs={reports['train']['pairs']} "
            f"val_pairs={reports['val']['pairs']} "
            f"test_pairs={reports.get('test', {}).get('pairs', 'LOCKED')}"
        )

    summary = {
        "action": args.action,
        "folds": folds,
        "unlock_manifest_verified": unlock_payload is not None,
        "reports": all_reports,
    }
    if folds == list(range(1, 9)):
        tests = [all_reports[str(fold)].get("test_fish") for fold in folds]
        vals = [all_reports[str(fold)].get("val_fish") for fold in folds]
        if any(value is None for value in tests + vals):
            raise RuntimeError("Source protocols do not identify every val/test fish")
        if len(set(tests)) != 8 or len(set(vals)) != 8:
            raise RuntimeError(f"Invalid LOFO rotation tests={tests} vals={vals}")
        for fold in folds:
            row = all_reports[str(fold)]
            train = set(row["train_fish"])
            val = row["val_fish"]
            test = row["test_fish"]
            if len(train) != 6 or val in train or test in train or val == test:
                raise RuntimeError(
                    f"fold {fold}: invalid split train={sorted(train)} val={val} test={test}"
                )
        summary["test_rotation"] = tests
        summary["validation_rotation"] = vals
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("ZEBRAFISH MPRT LOFO8 DATA AUDIT: PASS")


if __name__ == "__main__":
    main()
