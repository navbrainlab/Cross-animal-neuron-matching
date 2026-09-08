#!/usr/bin/env python3
"""Convert official Dunn/DANDI 001623 intermediate pickles to aligned NPZs.

The raw DANDI NWB files contain image volumes but no ROI response table.  The
paper's official Zenodo release (record 17353307) contains pickled
``wbliveDataClass`` objects with aligned dF/F, tracked coordinates, and manual
NeuroPAL identities.  This script converts only explicitly selected experiment
dates and preserves every tracked neuron as a matching candidate.  Supervision
and metrics are restricted to unique, non-empty identities in ``clean_mask``.

Coordinates are the temporal median of the tracked positions.  X/Y pixels are
converted with the 0.16 um (1x1) or 0.32 um (2x2) calibration used by the
authors' NWB writer; Z planes use the recording's ``z_step_size`` metadata.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DATE_RE = re.compile(r"^(\d{8})-")
INVALID_IDS = {"", "MARKER", "NAN", "NONE", "UNKNOWN", "?"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle-root", type=Path, required=True)
    parser.add_argument("--official-lib", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-dates", required=True)
    parser.add_argument("--val-dates", required=True)
    parser.add_argument("--test-dates", required=True)
    parser.add_argument(
        "--records-per-date", type=int, default=3,
        help="Maximum records per selected date; use 0 to include every available record.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_dates(text: str) -> list[str]:
    return sorted({x.strip().replace("-", "") for x in text.split(",") if x.strip()})


def recording_date(path: Path) -> str:
    match = DATE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Cannot infer recording date from {path.name}")
    return match.group(1)


def clean_identity(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().upper()
    return "" if value in INVALID_IDS else value


def finite_trace_rows(activity: np.ndarray) -> np.ndarray:
    """Interpolate isolated non-finite samples without changing ROI order."""
    result = np.asarray(activity, dtype=np.float32).copy()
    grid = np.arange(result.shape[1])
    for row in range(result.shape[0]):
        good = np.isfinite(result[row])
        if good.all():
            continue
        if good.sum() < 2:
            result[row] = 0.0
        else:
            result[row, ~good] = np.interp(grid[~good], grid[good], result[row, good])
    return result


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_pickle(path: Path) -> Any:
    # Importing the original class is required for the official pickle module
    # reference (wbliveDataClass.wbliveDataClass) to resolve.
    with path.open("rb") as handle:
        return pickle.load(handle)


def extract_record(obj: Any, source: Path, split: str, date: str) -> tuple[dict[str, Any], dict[str, Any]]:
    dff = np.asarray(obj.dff, dtype=np.float32)
    labels_raw = np.asarray(obj.ID1, dtype=object).reshape(-1)
    n = labels_raw.size
    if dff.ndim != 2:
        raise ValueError(f"{source}: dff must be 2-D, got {dff.shape}")
    if dff.shape[1] == n:
        activity = dff.T
    elif dff.shape[0] == n:
        activity = dff
    else:
        raise ValueError(f"{source}: dff {dff.shape} does not align with {n} IDs")
    activity = finite_trace_rows(activity)

    coords = []
    for key in ("x", "y", "z"):
        values = np.asarray(getattr(obj, key), dtype=np.float64)
        if values.ndim == 1 and values.size == n:
            coord = values
        elif values.ndim == 2 and values.shape[1] == n:
            coord = np.nanmedian(values, axis=0)
        elif values.ndim == 2 and values.shape[0] == n:
            coord = np.nanmedian(values, axis=1)
        else:
            raise ValueError(f"{source}: {key} {values.shape} does not align with {n} IDs")
        coords.append(coord)
    xyz_voxel = np.column_stack(coords)

    md = getattr(obj, "md", {}) or {}
    gooey = md.get("gooey_args", {}) or {}
    mmc = md.get("mmc_metadata", {}) or {}
    binning = str(mmc.get("binning", "1x1")).lower()
    xy_um = 0.16 if binning == "1x1" else 0.32
    z_um = float(gooey.get("z_step_size", 3.0))
    xyz = xyz_voxel * np.asarray([xy_um, xy_um, z_um], dtype=np.float64)

    labels = np.asarray([clean_identity(x) for x in labels_raw], dtype=str)
    counts = Counter(x for x in labels if x)
    valid_xyz = np.all(np.isfinite(xyz), axis=1)
    clean = np.asarray([bool(x) and counts[x] == 1 for x in labels], dtype=bool) & valid_xyz
    timestamps = np.asarray(obj.timevec, dtype=np.float64).reshape(-1)
    if timestamps.size != activity.shape[1]:
        raise ValueError(
            f"{source}: timestamps={timestamps.size}, activity T={activity.shape[1]}"
        )
    dt = float(np.nanmedian(np.diff(timestamps)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"{source}: invalid timestamp spacing {dt}")

    name = source.stem
    payload = {
        "activity_raw": activity.astype(np.float32, copy=False),
        "xyz": xyz.astype(np.float32),
        "xyz_voxel": xyz_voxel.astype(np.float32),
        "cell_id": labels,
        "cell_id_alt": labels,
        "timestamps": timestamps,
        "roi_index": np.arange(n, dtype=np.int64),
        "aligned_table_id": np.arange(n, dtype=np.int64),
        "labeled_mask": np.asarray([bool(x) for x in labels], dtype=bool),
        "certain_mask": clean,
        "clean_mask": clean,
        "valid_xyz_mask": valid_xyz,
        "sampling_rate_hz": np.asarray(1.0 / dt, dtype=np.float64),
        "recording_uid": np.asarray(name),
        "recording_date": np.asarray(date),
        "subject_id": np.asarray(name),
        "session_id": np.asarray(name),
        "source_pickle": np.asarray(str(source)),
        "source_pickle_md5": np.asarray(md5(source)),
        "dandiset": np.asarray("001623"),
        "source_mode": np.asarray("official_zenodo_wbliveDataClass"),
        "coordinate_calibration_um": np.asarray([xy_um, xy_um, z_um], dtype=np.float32),
        "coordinate_summary": np.asarray("temporal_median"),
        "clean_indices": np.flatnonzero(clean).astype(np.int64),
    }
    row = {
        "recording": name,
        "date": date,
        "split": split,
        "n_neurons": int(n),
        "n_timepoints": int(activity.shape[1]),
        "sampling_rate_hz": 1.0 / dt,
        "num_clean_identities": int(clean.sum()),
        "xyz_finite_rows": int(valid_xyz.sum()),
        "xy_um_per_pixel": xy_um,
        "z_um_per_plane": z_um,
        "source_pickle": str(source),
    }
    return payload, row


def replace_symlink(destination: Path, source: Path, overwrite: bool) -> None:
    if destination.is_symlink() or destination.exists():
        if not overwrite:
            raise FileExistsError(f"Refusing to replace {destination}; pass --overwrite")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    lib = args.official_lib.expanduser().resolve()
    if not (lib / "wbliveDataClass.py").is_file():
        raise FileNotFoundError(f"Missing official class module below {lib}")
    sys.path.insert(0, str(lib))
    __import__("wbliveDataClass")

    split_dates = {
        "train": parse_dates(args.train_dates),
        "val": parse_dates(args.val_dates),
        "test": parse_dates(args.test_dates),
    }
    date_sets = {key: set(value) for key, value in split_dates.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = date_sets[left] & date_sets[right]
        if overlap:
            raise ValueError(f"Date leakage {left}/{right}: {sorted(overlap)}")
    selected_dates = set().union(*date_sets.values())
    date_to_split = {date: split for split, dates in split_dates.items() for date in dates}

    all_pickles = sorted(args.pickle_root.expanduser().resolve().glob("*.pkl"))
    candidates: dict[str, list[Path]] = {date: [] for date in selected_dates}
    for path in all_pickles:
        date = recording_date(path)
        if date in candidates:
            candidates[date].append(path)
    selected: list[Path] = []
    for date in sorted(selected_dates):
        paths = candidates[date]
        if args.records_per_date == 0:
            if not paths:
                raise RuntimeError(f"{date}: no official pickles found")
            selected.extend(paths)
            continue
        if args.records_per_date < 0:
            raise ValueError("--records-per-date must be non-negative")
        if len(paths) < args.records_per_date:
            raise RuntimeError(
                f"{date}: need {args.records_per_date} official pickles, found {len(paths)}"
            )
        selected.extend(paths[: args.records_per_date])

    output = args.output_root.expanduser().resolve()
    all_root = output / "all"
    all_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_root = output / split
        split_root.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for old in split_root.glob("*.npz"):
                old.unlink()

    rows: list[dict[str, Any]] = []
    identity_frequency: Counter[str] = Counter()
    for index, source in enumerate(sorted(selected), start=1):
        date = recording_date(source)
        split = date_to_split[date]
        print(f"[{index}/{len(selected)}] {split}: {source.name}", flush=True)
        obj = load_official_pickle(source)
        payload, row = extract_record(obj, source, split, date)
        destination = all_root / f"{source.stem}.npz"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {destination}")
        np.savez_compressed(destination, **payload)
        replace_symlink(output / split / destination.name, destination, args.overwrite)
        identity_frequency.update(set(payload["cell_id"][payload["clean_mask"]].tolist()))
        row["npz"] = str(destination)
        rows.append(row)
        del obj, payload
        gc.collect()

    manifest = output / "split_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    split_counts = Counter(row["split"] for row in rows)
    if args.records_per_date == 0:
        record_selection = "all locally available official intermediate records on each selected date"
    else:
        record_selection = (
            f"the first {args.records_per_date} locally available official intermediate records "
            "on each selected date"
        )
    summary = {
        "protocol": "dunn_001623_experiment_date_disjoint_v1",
        "selection_basis": (
            "Chronological experiment-date groups only, using " + record_selection + "; "
            "activity, xyz, identities, predictions, and metrics were not used to choose the split."
        ),
        "source_root": str(args.pickle_root.expanduser().resolve()),
        "official_code": str(lib),
        "output_root": str(output),
        "num_recordings": len(rows),
        "split_dates": split_dates,
        "split_counts": {key: int(split_counts[key]) for key in ("train", "val", "test")},
        "date_overlap": {
            "train_val": sorted(date_sets["train"] & date_sets["val"]),
            "train_test": sorted(date_sets["train"] & date_sets["test"]),
            "val_test": sorted(date_sets["val"] & date_sets["test"]),
        },
        "identity_coverage": {
            "num_unique_clean_identities": len(identity_frequency),
            "present_in_all_recordings": sum(v == len(rows) for v in identity_frequency.values()),
        },
        "records": rows,
        "uses_all_local_pickles": len(rows) == len(all_pickles),
        "caveat": (
            "All locally available official intermediate pickles are included."
            if len(rows) == len(all_pickles)
            else "Exploratory subset of locally available official intermediate pickles."
        ),
    }
    write_json(output / "split_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
