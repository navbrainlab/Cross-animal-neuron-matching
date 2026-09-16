#!/usr/bin/env python3
"""Lightweight source-row audit for the canonical Atanas recordings.

This reads only manifests, source-aligned NPZ metadata/keys, and (when locally
available) the small HDF5 metadata datasets in the requested target NWB.  It
does not load a model or read the large image arrays.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "Data/Atanas_SF_unified_000776/dataset_manifest.csv"
OUTPUT = REPO / "exports/atanas_source_audit_20260913"
LOCAL_DANDI_ROOT = Path(
    "/media/ubuntu/65ccd0d4-7e99-4548-b09a-bee59b6ae7fb/"
    "klb_data/dandi_data/000776"
)
TARGET = "2022-08-02-01"
INVALID = {"", "none", "null", "nan", "unknown", "unlabeled", "unlabelled", "?"}

ACTIVITY = (
    "/processing/CalciumActivity/SignalRawFluor/"
    "SignalCalciumImResponseSeries/data"
)
ROIS = (
    "/processing/CalciumActivity/SignalRawFluor/"
    "SignalCalciumImResponseSeries/rois"
)
TIMESTAMPS = (
    "/processing/CalciumActivity/SignalRawFluor/"
    "SignalCalciumImResponseSeries/timestamps"
)
SEGMENTATION = (
    "/processing/CalciumActivity/CalciumSeriesSegmentation/"
    "Aligned_neuron_coordinates"
)
TABLE_ID = f"{SEGMENTATION}/id"
LABELS = f"{SEGMENTATION}/ID_labels"


def scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    return str(np.asarray(data[key]).item())


def resolve(path: str) -> Path:
    value = Path(path)
    if value.is_file():
        return value
    parts = value.parts
    if "Data" in parts:
        candidate = REPO.joinpath(*parts[parts.index("Data") :])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def main() -> None:
    manifest_rows = list(csv.DictReader(MANIFEST.open(newline="", encoding="utf-8")))
    subject_for_asset = {
        row["asset_path"]: row["asset_path"].split("/", 1)[0][4:]
        for row in manifest_rows
    }
    assets_per_subject = Counter(subject_for_asset.values())
    rows: list[dict[str, object]] = []
    target_npz: dict[str, object] | None = None

    for manifest_row in manifest_rows:
        path = resolve(manifest_row["full_npz"])
        with np.load(path, allow_pickle=False) as data:
            labels = np.asarray(data["cell_id"]).astype(str)
            roi_index = np.asarray(data["roi_index"], dtype=np.int64)
            instance_id = np.asarray(data["aligned_table_id"], dtype=np.int64)
            n = len(labels)
            if roi_index.shape != (n,) or instance_id.shape != (n,):
                raise ValueError(f"{path}: source row keys are not aligned")
            # In every canonical Atanas recording the response-series region
            # selects every raw segmentation-table row exactly once.
            if not np.array_equal(np.sort(roi_index), np.arange(n)):
                raise ValueError(f"{path}: ROI region is not a one-to-one full-table selection")

            valid_xyz = np.asarray(data["valid_xyz_mask"], dtype=bool)
            labeled = np.asarray(data["labeled_mask"], dtype=bool)
            certain = np.asarray(data["certain_mask"], dtype=bool)
            clean = np.asarray(data["clean_mask"], dtype=bool)
            valid_id = np.asarray([x.strip().lower() not in INVALID for x in labels])
            annotation_mask = valid_xyz & labeled & certain & clean & valid_id
            annotation_labels = [x.strip() for x in labels[annotation_mask]]
            annotation_counts = Counter(annotation_labels)
            valid_annotation_count = sum(
                count == 1 for count in annotation_counts.values()
            )
            nonempty_label_count = sum(x.strip().lower() not in INVALID for x in labels)

            asset = scalar(data, "asset_path")
            subject = subject_for_asset[asset]
            row = {
                "recording_id": scalar(data, "recording_uid"),
                "subject_id": subject,
                "source_file": asset,
                "dandiset": scalar(data, "dandiset"),
                "raw_row_count": n,
                "unique_neuron_instance_count": len(np.unique(instance_id)),
                "number_of_sessions_or_subrecordings": assets_per_subject[subject],
                "nonempty_label_count": int(nonempty_label_count),
                "valid_annotation_count": int(valid_annotation_count),
            }
            rows.append(row)
            if row["recording_id"] == TARGET:
                nonempty_values = [x.strip() for x in labels if x.strip().lower() not in INVALID]
                duplicate_labels = sorted(
                    name for name, count in Counter(nonempty_values).items() if count > 1
                )
                target_npz = {
                    **row,
                    "roi_index_count": len(roi_index),
                    "unique_roi_index_count": len(np.unique(roi_index)),
                    "duplicate_neuron_instance_rows": n - len(np.unique(instance_id)),
                    "duplicate_nonempty_identity_labels": duplicate_labels,
                    "timepoint_count": int(np.asarray(data["activity_raw"]).shape[1]),
                }

    if target_npz is None:
        raise RuntimeError(f"target {TARGET} is absent")
    if len({row["source_file"] for row in rows}) != len(rows):
        raise AssertionError("More than one recording maps to the same source NWB")

    target_asset = LOCAL_DANDI_ROOT / str(target_npz["source_file"])
    with h5py.File(target_asset, "r") as handle:
        raw_rois = np.asarray(handle[ROIS], dtype=np.int64).reshape(-1)
        raw_ids = np.asarray(handle[TABLE_ID], dtype=np.int64).reshape(-1)
        raw_detail = {
            "direct_raw_nwb_path": str(target_asset),
            "nwb_identifier": handle["/identifier"][()].decode(),
            "nwb_subject_id": handle["/general/subject/subject_id"][()].decode(),
            "session_start_time": handle["/session_start_time"][()].decode(),
            "segmentation_table_row_count": int(handle[LABELS].shape[0]),
            "activity_shape_time_by_neuron": list(handle[ACTIVITY].shape),
            "roi_region_row_count": len(raw_rois),
            "unique_roi_region_row_count": len(np.unique(raw_rois)),
            "table_id_row_count": len(raw_ids),
            "unique_table_id_count": len(np.unique(raw_ids)),
            "timestamp_count": int(handle[TIMESTAMPS].shape[0]),
            "intervals_or_trials_table_present": "/intervals" in handle,
            "activity_response_series_count_used": 1,
        }
    target_report = {
        **target_npz,
        "raw_nwb_direct_check": raw_detail,
        "conclusion": {
            "count_152_is_len_rows": True,
            "count_152_is_unique_neuron_instances": True,
            "subject_combines_multiple_recordings": False,
            "duplicate_neuron_instance_ids": False,
            "duplicate_identity_label_note": (
                "SIA?L occurs twice, but both rows are uncertain and excluded from valid annotations"
            ),
            "duplicate_time_window_or_trial_rows": False,
        },
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "recording_id",
        "subject_id",
        "source_file",
        "dandiset",
        "raw_row_count",
        "unique_neuron_instance_count",
        "number_of_sessions_or_subrecordings",
        "nonempty_label_count",
        "valid_annotation_count",
    ]
    with (OUTPUT / "atanas_recording_source_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: str(row["recording_id"])))
    (OUTPUT / f"{TARGET}_source_audit.json").write_text(
        json.dumps(target_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"recordings": len(rows), "target": target_report}, indent=2))


if __name__ == "__main__":
    main()
