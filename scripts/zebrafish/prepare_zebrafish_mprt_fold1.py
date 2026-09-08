#!/usr/bin/env python3
"""Adapt the locked zebrafish LOFO fold-1 train/val records to NeuRID.

The existing LOFO preparation stores pair-local identity strings as NumPy
object arrays. NeuRID deliberately loads NPZ files with ``allow_pickle=False``.
This adapter rewrites those labels as fixed-width Unicode and audits that only
the biologically legal q/r temporal pairs share identities.

The locked test split is never opened and is never written by this script.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


EXPECTED = {
    "train": {"records": 134, "pairs": 67},
    "val": {"records": 36, "pairs": 18},
}
REQUIRED_KEYS = {"activity_raw", "xyz", "cell_id", "pair_id", "side"}


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
    result = np.asarray(z[key], dtype=bool).reshape(-1)
    if result.shape != (n,):
        raise ValueError(f"{key}: expected ({n},), got {result.shape}")
    return result


def unique_clean_ids(path: Path) -> frozenset[str]:
    with np.load(path, allow_pickle=False) as z:
        labels = np.asarray(z["cell_id"]).astype(str)
        clean = np.asarray(z["clean_mask"], dtype=bool)
        labeled = np.asarray(z["labeled_mask"], dtype=bool)
        certain = np.asarray(z["certain_mask"], dtype=bool)
    values = [x for x, keep in zip(labels.tolist(), clean & labeled & certain) if keep]
    counts = Counter(values)
    return frozenset(value for value, count in counts.items() if count == 1)


def audit_split(split_root: Path, split: str, min_shared: int) -> dict:
    files = sorted(split_root.glob("*.npz"))
    expected = EXPECTED[split]
    if len(files) != expected["records"]:
        raise RuntimeError(
            f"{split}: expected {expected['records']} records, found {len(files)}"
        )

    metadata: dict[Path, tuple[str, str]] = {}
    identities: dict[Path, frozenset[str]] = {}
    groups: dict[str, dict[str, Path]] = {}

    for path in files:
        with np.load(path, allow_pickle=False) as z:
            if np.asarray(z["cell_id"]).dtype.kind not in {"U", "S"}:
                raise RuntimeError(f"{path}: cell_id is not a safe string array")
            pair_id = scalar_text(z, "pair_id")
            side = scalar_text(z, "side")
            activity = np.asarray(z["activity_raw"])
            xyz = np.asarray(z["xyz"])
            if activity.ndim != 2 or activity.shape[1] != 128:
                raise RuntimeError(f"{path}: expected activity [N,128], got {activity.shape}")
            if xyz.shape != (activity.shape[0], 3):
                raise RuntimeError(f"{path}: xyz/activity mismatch {xyz.shape}/{activity.shape}")
            if not np.isfinite(activity).all() or not np.isfinite(xyz).all():
                raise RuntimeError(f"{path}: non-finite input")
        if not pair_id or side not in {"q", "r"}:
            raise RuntimeError(f"{path}: invalid pair metadata pair_id={pair_id!r} side={side!r}")
        if side in groups.setdefault(pair_id, {}):
            raise RuntimeError(f"{split}: duplicate {pair_id}/{side}")
        groups[pair_id][side] = path
        metadata[path] = (pair_id, side)
        identities[path] = unique_clean_ids(path)

    if len(groups) != expected["pairs"]:
        raise RuntimeError(
            f"{split}: expected {expected['pairs']} pair_ids, found {len(groups)}"
        )
    for pair_id, sides in groups.items():
        if set(sides) != {"q", "r"}:
            raise RuntimeError(f"{split}: {pair_id} has sides {sorted(sides)}")

    eligible: list[dict] = []
    illegal_overlaps: list[dict] = []
    for left, right in combinations(files, 2):
        shared = len(identities[left].intersection(identities[right]))
        if shared == 0:
            continue
        pair_left, side_left = metadata[left]
        pair_right, side_right = metadata[right]
        legal = pair_left == pair_right and {side_left, side_right} == {"q", "r"}
        row = {
            "left": left.name,
            "right": right.name,
            "pair_id": pair_left if pair_left == pair_right else "",
            "shared": shared,
        }
        if legal and shared >= min_shared:
            eligible.append(row)
        else:
            illegal_overlaps.append(row)

    if illegal_overlaps:
        raise RuntimeError(
            f"{split}: detected identity leakage or sub-threshold overlap: "
            f"{illegal_overlaps[:3]}"
        )
    if len(eligible) != expected["pairs"]:
        raise RuntimeError(
            f"{split}: expected {expected['pairs']} legal pairs, found {len(eligible)}"
        )

    return {
        "split": split,
        "records": len(files),
        "legal_pairs": len(eligible),
        "minimum_shared": min(row["shared"] for row in eligible),
        "maximum_shared": max(row["shared"] for row in eligible),
        "pairs": eligible,
    }


def convert_record(source: Path, target: Path) -> None:
    with np.load(source, allow_pickle=True) as z:
        missing = REQUIRED_KEYS.difference(z.files)
        if missing:
            raise KeyError(f"{source}: missing {sorted(missing)}")

        activity = np.asarray(z["activity_raw"], dtype=np.float32)
        xyz = np.asarray(z["xyz"], dtype=np.float32)
        labels = unicode_array(z["cell_id"])
        n = len(labels)

        if activity.ndim != 2 or activity.shape != (n, 128):
            raise ValueError(f"{source}: expected activity ({n},128), got {activity.shape}")
        if xyz.shape != (n, 3):
            raise ValueError(f"{source}: expected xyz ({n},3), got {xyz.shape}")
        if not np.isfinite(activity).all() or not np.isfinite(xyz).all():
            raise ValueError(f"{source}: NaN/Inf in model input")

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
        valid_xyz = np.isfinite(xyz).all(axis=1)

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        activity_raw=activity,
        xyz=xyz,
        cell_id=labels,
        clean_mask=clean,
        labeled_mask=labeled,
        certain_mask=certain,
        valid_xyz_mask=valid_xyz,
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


def audit_root(root: Path, min_shared: int) -> dict:
    if (root / "test").exists():
        raise RuntimeError(f"Locked pilot root must not contain test: {root / 'test'}")
    splits = {
        split: audit_split(root / split, split, min_shared)
        for split in ("train", "val")
    }
    return {
        "protocol": "zebrafish_longitudinal_LOFO_fold1_train_val_only",
        "matching_unit": "same fish, q/r windows separated by 60 minutes",
        "train_fish": 6,
        "validation_fish": "func_20150417",
        "locked_test_fish_not_accessed": "func_20150410",
        "activity_points": 128,
        "minimum_shared": min_shared,
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/Data/"
            "Zebrafish_LOFO8_joint_from_scratch/fold_1/features"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/Data/"
            "Zebrafish_MPRT_LOFO8_60m/fold_1"
        ),
    )
    parser.add_argument("--min-shared", type=int, default=20)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if args.audit_only:
        report = audit_root(output_root, args.min_shared)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("ZEBRAFISH MPRT FOLD1 AUDIT: PASS")
        return

    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Missing existing LOFO features: {source_root}\n"
            "Run the previously supplied zebrafish LOFO prepare stage first."
        )

    if output_root.exists():
        protocol_path = output_root / "protocol.json"
        if protocol_path.is_file():
            report = audit_root(output_root, args.min_shared)
            print(f"[REUSE] Valid existing conversion: {output_root}")
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return
        raise FileExistsError(
            f"Refusing to overwrite incomplete output: {output_root}. "
            "Move it aside and rerun."
        )

    building = output_root.with_name(output_root.name + ".building")
    if building.exists():
        raise FileExistsError(f"Stale build directory exists: {building}")
    building.mkdir(parents=True)

    try:
        for split in ("train", "val"):
            source_split = source_root / split
            paths = sorted(source_split.glob("*.npz"))
            expected = EXPECTED[split]["records"]
            if len(paths) != expected:
                raise RuntimeError(
                    f"{source_split}: expected {expected} NPZ files, found {len(paths)}"
                )
            for source in paths:
                convert_record(source, building / split / source.name)

        report = audit_root(building, args.min_shared)
        (building / "protocol.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        output_root.parent.mkdir(parents=True, exist_ok=True)
        building.rename(output_root)
    except Exception:
        shutil.rmtree(building, ignore_errors=True)
        raise

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("ZEBRAFISH MPRT FOLD1 CONVERSION: PASS")
    print("Output:", output_root)


if __name__ == "__main__":
    main()
