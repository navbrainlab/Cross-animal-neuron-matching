#!/usr/bin/env python3
"""Build a unified Atanas/SF activity+position+identity dataset from DANDI 000776.

The script does NOT download the ~25 GiB NWB assets.  It opens each public NWB
through HTTP range reads (remfile + h5py) and reads only these small datasets:

* Calcium activity data and timestamps
* the activity series ROI indices
* activity-aligned neuron ID labels
* activity-aligned neuron coordinates
* calcium imaging grid spacing

The neuron axis is resolved through the RoiResponseSeries ``rois`` vector, so
for output row i, activity_raw[i], xyz[i], and cell_id[i] refer to the same
tracked neuron.

Two outputs are written for every recording:

* full/<split>/<uid>.npz  -- all tracked neurons, including unlabeled neurons
* clean/<split>/<uid>.npz -- only certain, uniquely-labeled neurons

Existing split filenames are used only to preserve the current 70/15/15 worm
assignment; their activity values are never read or copied.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

import h5py
import numpy as np

UID_RE = re.compile(r"20\d{2}-\d{2}-\d{2}-\d{2}")
INVALID_LABELS = {"", "none", "null", "nan", "unknown", "unlabeled", "unlabelled", "?"}

ACTIVITY_ROOT = "/processing/CalciumActivity"
SERIES_ROOT = f"{ACTIVITY_ROOT}/SignalRawFluor/SignalCalciumImResponseSeries"
SEG_ROOT = f"{ACTIVITY_ROOT}/CalciumSeriesSegmentation/Aligned_neuron_coordinates"

PATH_ACTIVITY = f"{SERIES_ROOT}/data"
PATH_ROIS = f"{SERIES_ROOT}/rois"
PATH_TIMESTAMPS = f"{SERIES_ROOT}/timestamps"
PATH_LABELS = f"{SEG_ROOT}/ID_labels"
PATH_TABLE_ID = f"{SEG_ROOT}/id"
PATH_VOXEL = f"{SEG_ROOT}/voxel_mask"
PATH_VOXEL_INDEX = f"{SEG_ROOT}/voxel_mask_index"
PATH_GRID = "/general/optophysiology/CalciumImVol/grid_spacing"
PATH_ALT_LABELS = f"{ACTIVITY_ROOT}/NeuronIDs/labels"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream unified Atanas/SF activity, aligned xyz, and IDs from DANDI 000776"
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("Data/Atanas_activity_npz_70_15_15"),
        help="Existing train/val/test directories; only filenames are used for split assignment.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("Data/Atanas_SF_unified_000776"),
    )
    parser.add_argument("--dandiset-id", default="000776")
    parser.add_argument("--version", default="0.241009.1509")
    parser.add_argument(
        "--local-nwb-root",
        type=Path,
        default=Path(
            "/media/ubuntu/65ccd0d4-7e99-4548-b09a-bee59b6ae7fb/"
            "klb_data/dandi_data/000776"
        ),
        help="Optional local Dandiset root. Complete local NWBs are preferred.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/remfile_sf_unified_cache"),
    )
    parser.add_argument(
        "--prefer-local",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-on-missing-records",
        action="store_true",
        help="Fail instead of skipping split UIDs absent from the published Dandiset.",
    )
    parser.add_argument(
        "--require-label-tables-equal",
        action="store_true",
        help="Fail if Aligned_neuron_coordinates/ID_labels differs from NeuronIDs/labels.",
    )
    return parser.parse_args()


def decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_scalar(value.item())
    return str(value)


def decode_label(value: Any) -> str:
    if isinstance(value, (str, bytes, np.bytes_)):
        text = decode_scalar(value)
    else:
        arr = np.asarray(value)
        if arr.ndim == 0:
            text = decode_scalar(arr.item())
        else:
            parts: list[str] = []
            for item in arr.reshape(-1):
                part = decode_scalar(item)
                if part not in {"", "0", "\x00"}:
                    parts.append(part)
            text = "".join(parts)
    return text.strip().replace(" ", "").replace("\x00", "")


def is_labeled(label: str) -> bool:
    return label.lower() not in INVALID_LABELS


def build_masks(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=str)
    labeled = np.asarray([is_labeled(x) for x in labels], dtype=bool)
    certain = labeled & np.asarray(["?" not in x for x in labels], dtype=bool)

    counts = Counter(labels[certain].tolist())
    clean = certain & np.asarray([counts.get(x, 0) == 1 for x in labels], dtype=bool)
    return labeled, certain, clean


def required_recordings(split_root: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        directory = split_root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing split directory: {directory}")
        for path in sorted(directory.glob("*.npz")):
            match = UID_RE.search(path.stem)
            if not match:
                raise ValueError(f"No recording UID in filename: {path}")
            uid = match.group(0)
            if uid in seen:
                raise ValueError(f"Recording UID appears more than once in split root: {uid}")
            seen.add(uid)
            rows.append((split, uid))
    if not rows:
        raise RuntimeError(f"No NPZ filenames found under {split_root}")
    return rows


def build_dandi_index(dandiset_id: str, version: str) -> dict[str, dict[str, Any]]:
    try:
        from dandi.dandiapi import DandiAPIClient
    except ImportError as exc:
        raise RuntimeError("Install dandi first: python -m pip install -U dandi") from exc

    index: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, list[str]] = {}

    with DandiAPIClient.for_dandi_instance("dandi") as client:
        dandiset = client.get_dandiset(dandiset_id, version)
        for asset in dandiset.get_assets():
            path = asset.path
            if not path.lower().endswith(".nwb"):
                continue
            match = UID_RE.search(path)
            if not match:
                continue
            uid = match.group(0)
            if uid in index:
                duplicates.setdefault(uid, [index[uid]["asset_path"]]).append(path)
                continue
            index[uid] = {
                "asset_path": path,
                "url": asset.get_content_url(follow_redirects=1, strip_query=True),
                "size_bytes": int(getattr(asset, "size", 0) or 0),
            }

    if duplicates:
        raise RuntimeError(f"Multiple NWB assets found for recording UID(s): {duplicates}")
    return index


def complete_local_path(local_root: Optional[Path], asset_path: str) -> Optional[Path]:
    if local_root is None:
        return None
    path = local_root / asset_path
    if not path.is_file():
        return None
    # aria2 leaves this control file while the target file is incomplete.
    if Path(str(path) + ".aria2").exists():
        return None
    return path


@contextmanager
def open_nwb_h5(
    *,
    asset_path: str,
    url: str,
    local_root: Optional[Path],
    cache_dir: Path,
    prefer_local: bool,
) -> Iterator[tuple[h5py.File, str]]:
    local = complete_local_path(local_root, asset_path) if prefer_local else None
    if local is not None:
        try:
            with h5py.File(local, "r") as handle:
                yield handle, "local"
                return
        except OSError as exc:
            print(f"warning: local NWB unusable, using remote stream: {local}: {exc}")

    try:
        import remfile
    except ImportError as exc:
        raise RuntimeError("Install remfile first: python -m pip install -U remfile") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        disk_cache = remfile.DiskCache(str(cache_dir))
        remote = remfile.File(url=url, disk_cache=disk_cache)
    except (AttributeError, TypeError):
        remote = remfile.File(url=url)

    try:
        with h5py.File(remote, "r") as handle:
            yield handle, "stream"
    finally:
        remote.close()


def require_paths(handle: h5py.File, paths: list[str]) -> None:
    missing = [path for path in paths if path not in handle]
    if missing:
        raise KeyError(f"NWB missing required dataset(s): {missing}")


def voxel_centroids(
    voxel_data: np.ndarray,
    voxel_index: np.ndarray,
    expected_rows: int,
) -> np.ndarray:
    raw = np.asarray(voxel_data)
    if raw.dtype.names:
        fields = {name.lower(): name for name in raw.dtype.names}
        for axis in ("x", "y", "z"):
            if axis not in fields:
                raise ValueError(f"voxel_mask missing {axis}; fields={raw.dtype.names}")
        points = np.column_stack(
            [
                np.asarray(raw[fields["x"]], dtype=np.float64),
                np.asarray(raw[fields["y"]], dtype=np.float64),
                np.asarray(raw[fields["z"]], dtype=np.float64),
            ]
        )
        weight_name = fields.get("weight")
        weights = (
            None
            if weight_name is None
            else np.asarray(raw[weight_name], dtype=np.float64)
        )
    else:
        numeric = np.asarray(raw, dtype=np.float64)
        if numeric.ndim != 2:
            raise ValueError(f"Unsupported numeric voxel_mask shape: {numeric.shape}")
        if numeric.shape[1] not in (3, 4) and numeric.shape[0] in (3, 4):
            numeric = numeric.T
        if numeric.shape[1] not in (3, 4):
            raise ValueError(f"Cannot identify xyz columns: {numeric.shape}")
        points = numeric[:, :3]
        weights = numeric[:, 3] if numeric.shape[1] == 4 else None

    ends = np.asarray(voxel_index, dtype=np.int64).reshape(-1)
    if len(ends) != expected_rows:
        raise ValueError(
            f"voxel_mask_index rows={len(ends)}, aligned table rows={expected_rows}"
        )
    if len(ends) and (np.any(np.diff(ends) < 0) or ends[-1] > len(points)):
        raise ValueError("Invalid cumulative voxel_mask_index")

    centroids = np.full((expected_rows, 3), np.nan, dtype=np.float64)
    start = 0
    for row, end_value in enumerate(ends):
        end = int(end_value)
        chunk = points[start:end]
        chunk_weights = None if weights is None else weights[start:end]
        start = end

        valid = np.isfinite(chunk).all(axis=1)
        if chunk_weights is not None:
            valid &= np.isfinite(chunk_weights)
        chunk = chunk[valid]
        if chunk_weights is not None:
            chunk_weights = chunk_weights[valid]
        if not len(chunk):
            continue
        if chunk_weights is not None:
            positive = np.maximum(chunk_weights, 0.0)
            if positive.sum() > 0:
                centroids[row] = np.average(chunk, axis=0, weights=positive)
                continue
        centroids[row] = chunk.mean(axis=0)

    if start != len(points):
        raise ValueError(
            f"voxel_mask_index consumed {start} of {len(points)} voxel entries"
        )
    return centroids


def extract_recording(
    handle: h5py.File,
    *,
    require_label_tables_equal: bool,
) -> dict[str, np.ndarray]:
    require_paths(
        handle,
        [
            PATH_ACTIVITY,
            PATH_ROIS,
            PATH_TIMESTAMPS,
            PATH_LABELS,
            PATH_TABLE_ID,
            PATH_VOXEL,
            PATH_VOXEL_INDEX,
            PATH_GRID,
        ],
    )

    activity = np.asarray(handle[PATH_ACTIVITY], dtype=np.float32)
    timestamps = np.asarray(handle[PATH_TIMESTAMPS], dtype=np.float64).reshape(-1)
    rois = np.asarray(handle[PATH_ROIS], dtype=np.int64).reshape(-1)
    table_ids = np.asarray(handle[PATH_TABLE_ID], dtype=np.int64).reshape(-1)
    raw_labels = np.asarray(handle[PATH_LABELS])
    labels_table = np.asarray([decode_label(x) for x in raw_labels], dtype=str)
    spacing = np.asarray(handle[PATH_GRID], dtype=np.float64).reshape(-1)[:3]

    n_table = len(labels_table)
    if len(table_ids) != n_table:
        raise ValueError(f"table id rows={len(table_ids)}, labels rows={n_table}")
    if activity.ndim != 2:
        raise ValueError(f"activity data must be 2D, got {activity.shape}")
    if len(timestamps) not in activity.shape:
        raise ValueError(
            f"timestamps length {len(timestamps)} does not match activity shape {activity.shape}"
        )

    # Standard NWB layout is time x ROI. Handle the opposite orientation safely.
    if activity.shape[0] == len(timestamps):
        time_by_roi = activity
    elif activity.shape[1] == len(timestamps):
        time_by_roi = activity.T
    else:
        raise ValueError(
            f"Cannot determine activity time axis: data={activity.shape}, timestamps={len(timestamps)}"
        )

    n_activity_rois = time_by_roi.shape[1]
    if len(rois) != n_activity_rois:
        raise ValueError(
            f"rois length={len(rois)}, activity neuron axis={n_activity_rois}"
        )
    if np.any(rois < 0) or np.any(rois >= n_table):
        raise IndexError(
            f"rois contains out-of-range aligned-table index; range=({rois.min()}, {rois.max()}), rows={n_table}"
        )
    if len(np.unique(rois)) != len(rois):
        raise ValueError("RoiResponseSeries rois contains duplicate table-row indices")

    voxel_table = voxel_centroids(
        np.asarray(handle[PATH_VOXEL]),
        np.asarray(handle[PATH_VOXEL_INDEX]),
        expected_rows=n_table,
    )

    if spacing.size != 3 or not np.isfinite(spacing).all():
        raise ValueError(f"Invalid CalciumImVol grid spacing: {spacing}")

    # Resolve all modalities through the RoiResponseSeries table-region vector.
    labels = labels_table[rois]
    xyz_voxel = voxel_table[rois]
    xyz = xyz_voxel * spacing[None, :]
    aligned_ids = table_ids[rois]
    activity_raw = time_by_roi.T

    if PATH_ALT_LABELS in handle:
        alt_table = np.asarray(
            [decode_label(x) for x in np.asarray(handle[PATH_ALT_LABELS])],
            dtype=str,
        )
        if len(alt_table) == n_table:
            alt_labels = alt_table[rois]
            mismatches = np.flatnonzero(alt_labels != labels)
            if require_label_tables_equal and len(mismatches):
                sample = [
                    (int(i), labels[i], alt_labels[i]) for i in mismatches[:20]
                ]
                raise ValueError(
                    f"Aligned ID_labels differs from NeuronIDs/labels at {len(mismatches)} rows; sample={sample}"
                )
        else:
            alt_labels = np.full(len(labels), "", dtype=str)
    else:
        alt_labels = np.full(len(labels), "", dtype=str)

    labeled_mask, certain_mask, clean_mask = build_masks(labels)
    valid_xyz_mask = np.isfinite(xyz).all(axis=1)
    clean_mask &= valid_xyz_mask

    if not (
        activity_raw.shape[0]
        == xyz.shape[0]
        == labels.shape[0]
        == rois.shape[0]
    ):
        raise AssertionError("Output neuron axes are not aligned")

    if len(timestamps) >= 2:
        dt = np.diff(timestamps)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        sampling_rate_hz = float(1.0 / np.median(dt)) if len(dt) else float("nan")
    else:
        sampling_rate_hz = float("nan")

    return {
        "activity_raw": activity_raw.astype(np.float32, copy=False),
        "xyz": xyz.astype(np.float32),
        "xyz_voxel": xyz_voxel.astype(np.float32),
        "cell_id": labels.astype(str),
        "cell_id_alt": alt_labels.astype(str),
        "timestamps": timestamps,
        "roi_index": rois.astype(np.int64),
        "aligned_table_id": aligned_ids.astype(np.int64),
        "grid_spacing": spacing.astype(np.float32),
        "labeled_mask": labeled_mask,
        "certain_mask": certain_mask,
        "clean_mask": clean_mask,
        "valid_xyz_mask": valid_xyz_mask,
        "sampling_rate_hz": np.asarray(sampling_rate_hz, dtype=np.float64),
    }


def save_npz_pair(
    payload: dict[str, np.ndarray],
    *,
    full_path: Path,
    clean_path: Path,
    uid: str,
    split: str,
    dandiset: str,
    asset_path: str,
    source_mode: str,
) -> dict[str, Any]:
    n = int(payload["activity_raw"].shape[0])
    clean_indices = np.flatnonzero(payload["clean_mask"]).astype(np.int64)

    metadata = {
        "recording_uid": np.asarray(uid),
        "split": np.asarray(split),
        "dandiset": np.asarray(dandiset),
        "asset_path": np.asarray(asset_path),
        "source_mode": np.asarray(source_mode),
    }

    full_payload = {**payload, **metadata, "clean_indices": clean_indices}

    clean_payload: dict[str, np.ndarray] = {}
    for key, value0 in payload.items():
        value = np.asarray(value0)
        if value.ndim > 0 and value.shape[0] == n:
            clean_payload[key] = value[clean_indices]
        else:
            clean_payload[key] = value
    clean_payload.update(metadata)
    clean_payload["source_full_indices"] = clean_indices

    # In the clean view every retained row is a certain, unique labeled neuron.
    n_clean = len(clean_indices)
    clean_payload["labeled_mask"] = np.ones(n_clean, dtype=bool)
    clean_payload["certain_mask"] = np.ones(n_clean, dtype=bool)
    clean_payload["clean_mask"] = np.ones(n_clean, dtype=bool)

    full_path.parent.mkdir(parents=True, exist_ok=True)
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(full_path, **full_payload)
    np.savez_compressed(clean_path, **clean_payload)

    labels = np.asarray(payload["cell_id"], dtype=str)
    return {
        "split": split,
        "uid": uid,
        "asset_path": asset_path,
        "source_mode": source_mode,
        "full_npz": str(full_path),
        "clean_npz": str(clean_path),
        "num_neurons": n,
        "num_labeled": int(payload["labeled_mask"].sum()),
        "num_certain": int(payload["certain_mask"].sum()),
        "num_clean": n_clean,
        "num_unlabeled": int((~payload["labeled_mask"]).sum()),
        "num_uncertain": int((payload["labeled_mask"] & ~payload["certain_mask"]).sum()),
        "num_invalid_xyz": int((~payload["valid_xyz_mask"]).sum()),
        "duplicate_certain_labels": sorted(
            [name for name, count in Counter(labels[payload["certain_mask"]]).items() if count > 1]
        ),
        "num_timepoints": int(payload["activity_raw"].shape[1]),
        "sampling_rate_hz": float(np.asarray(payload["sampling_rate_hz"]).item()),
    }


def validate_existing(full_path: Path, clean_path: Path) -> dict[str, Any]:
    with np.load(full_path, allow_pickle=False) as full:
        required = {"activity_raw", "xyz", "cell_id", "clean_mask", "timestamps"}
        missing = required.difference(full.files)
        if missing:
            raise KeyError(f"{full_path} missing {sorted(missing)}")
        n = int(full["activity_raw"].shape[0])
        if full["xyz"].shape != (n, 3) or len(full["cell_id"]) != n:
            raise ValueError(f"{full_path}: inconsistent neuron axes")
        n_clean = int(np.asarray(full["clean_mask"], dtype=bool).sum())
        n_time = int(full["activity_raw"].shape[1])
    with np.load(clean_path, allow_pickle=False) as clean:
        if int(clean["activity_raw"].shape[0]) != n_clean:
            raise ValueError(f"{clean_path}: clean count mismatch")
    return {"num_neurons": n, "num_clean": n_clean, "num_timepoints": n_time}


def write_manifest(output_root: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "uid",
        "asset_path",
        "source_mode",
        "full_npz",
        "clean_npz",
        "num_neurons",
        "num_labeled",
        "num_certain",
        "num_clean",
        "num_unlabeled",
        "num_uncertain",
        "num_invalid_xyz",
        "duplicate_certain_labels",
        "num_timepoints",
        "sampling_rate_hz",
    ]
    with (output_root / "dataset_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    args = parse_args()
    requested = required_recordings(args.split_root)
    print(f"Need {len(requested)} recording UIDs from {args.split_root}")
    print(f"Querying DANDI {args.dandiset_id}@{args.version} metadata...")
    remote_index = build_dandi_index(args.dandiset_id, args.version)

    available = [(split, uid) for split, uid in requested if uid in remote_index]
    missing = [
        {"split": split, "uid": uid, "reason": "not in published Dandiset version"}
        for split, uid in requested
        if uid not in remote_index
    ]
    print(f"Available: {len(available)}/{len(requested)}")
    if missing:
        print("Missing:", ", ".join(item["uid"] for item in missing))
        if args.fail_on_missing_records:
            raise FileNotFoundError(missing)

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for position, (split, uid) in enumerate(available, start=1):
        info = remote_index[uid]
        asset_path = info["asset_path"]
        full_path = args.output_root / "full" / split / f"{uid}.npz"
        clean_path = args.output_root / "clean" / split / f"{uid}.npz"

        if args.reuse_existing and full_path.exists() and clean_path.exists():
            try:
                existing = validate_existing(full_path, clean_path)
                rows.append(
                    {
                        "split": split,
                        "uid": uid,
                        "asset_path": asset_path,
                        "source_mode": "reuse",
                        "full_npz": str(full_path),
                        "clean_npz": str(clean_path),
                        "num_neurons": existing["num_neurons"],
                        "num_clean": existing["num_clean"],
                        "num_timepoints": existing["num_timepoints"],
                    }
                )
                print(
                    f"[{position:02d}/{len(available):02d}] reuse {split}/{uid}: "
                    f"N={existing['num_neurons']}, clean={existing['num_clean']}"
                )
                continue
            except Exception as exc:
                print(f"[{position:02d}/{len(available):02d}] rebuild {split}/{uid}: {exc}")

        try:
            with open_nwb_h5(
                asset_path=asset_path,
                url=info["url"],
                local_root=args.local_nwb_root,
                cache_dir=args.cache_dir,
                prefer_local=args.prefer_local,
            ) as (handle, source_mode):
                payload = extract_recording(
                    handle,
                    require_label_tables_equal=args.require_label_tables_equal,
                )

            result = save_npz_pair(
                payload,
                full_path=full_path,
                clean_path=clean_path,
                uid=uid,
                split=split,
                dandiset=f"{args.dandiset_id}@{args.version}",
                asset_path=asset_path,
                source_mode=source_mode,
            )
            rows.append(result)
            print(
                f"[{position:02d}/{len(available):02d}] {split}/{uid} via {source_mode}: "
                f"N={result['num_neurons']}, labeled={result['num_labeled']}, "
                f"clean={result['num_clean']}, T={result['num_timepoints']}"
            )
        except Exception as exc:
            item = {
                "split": split,
                "uid": uid,
                "asset_path": asset_path,
                "error": f"{type(exc).__name__}: {exc}",
            }
            errors.append(item)
            print(
                f"[{position:02d}/{len(available):02d}] ERROR {split}/{uid}: {item['error']}"
            )

    (args.output_root / "missing_records.json").write_text(
        json.dumps(missing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_root / "extraction_errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if rows:
        write_manifest(args.output_root, rows)

    complete_rows = [row for row in rows if row.get("source_mode") != "reuse"]
    summary = {
        "dandiset": f"{args.dandiset_id}@{args.version}",
        "split_root": str(args.split_root),
        "output_root": str(args.output_root),
        "requested_recordings": len(requested),
        "available_recordings": len(available),
        "created_or_reused_recordings": len(rows),
        "missing_recordings": len(missing),
        "failed_recordings": len(errors),
        "recordings_by_split": dict(Counter(row["split"] for row in rows)),
        "source_modes": dict(Counter(row.get("source_mode", "") for row in rows)),
        "total_neurons_newly_extracted": int(
            sum(int(row.get("num_neurons", 0)) for row in complete_rows)
        ),
        "total_clean_neurons_newly_extracted": int(
            sum(int(row.get("num_clean", 0)) for row in complete_rows)
        ),
        "notes": [
            "full NPZs retain unlabeled neurons for population/spatial context",
            "clean NPZs retain only non-empty, certain, within-worm unique identities with valid xyz",
            "activity, xyz, and identity are aligned through RoiResponseSeries/rois",
            "large raw image datasets are never read",
        ],
    }
    (args.output_root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if errors:
        raise RuntimeError(
            f"Failed on {len(errors)} recording(s); see {args.output_root / 'extraction_errors.json'}"
        )
    if not rows:
        raise RuntimeError("No recordings were extracted")


if __name__ == "__main__":
    main()
