#!/usr/bin/env python3
"""Build audited Atanas-21 NPZ inputs for the GWOT-MD paper protocol.

Activity is always read from the official ``processed_h5.tar.bz2`` release.
For the 20 recordings available in the project's older DANDI extraction,
labels come from its ``clean_mask`` snapshot.  The one absent recording,
2023-01-19-01, comes from the official Zenodo v4 label JSON.  One concrete
label that is blank in the older extraction (AVJL in 2023-01-23-21) is restored
from that same official JSON.  Every source and override is written to the
provenance file, and the script refuses to emit solver inputs unless all 441
Figure-C.1 common-label counts match.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
PAPER_CODE = ROOT / "baselines/gwot_md"
sys.path.insert(0, str(PAPER_CODE))

from evaluate_paper_top5_strict import PAPER_COMMON_LABELS  # noqa: E402
from solve_gwot_md_atanas21 import (  # noqa: E402
    PAPER_COHORT,
    clean_label,
    find_input_file,
    load_h5_worm,
    load_label_json,
    sha256_file,
)


MISSING_FROM_DANDI = "2023-01-19-01"
V4_RESTORATIONS = {"2023-01-23-21": {103: "AVJL"}}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_digest(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def find_old_npz(root: Path, uid: str) -> Path:
    candidates = sorted(root.rglob(f"{uid}.npz"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{uid}: expected one old full-cohort NPZ under {root}, found {candidates}"
        )
    return candidates[0]


def old_clean_labels(path: Path, expected_n: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        labels = np.asarray(data["cell_id"]).astype(str)
        clean = np.asarray(data["clean_mask"], dtype=bool)
    if labels.shape != (expected_n,) or clean.shape != (expected_n,):
        raise ValueError(
            f"{path}: old label arrays do not match official H5 neuron count {expected_n}"
        )
    return np.asarray(
        [clean_label(label) if keep else "" for label, keep in zip(labels, clean)],
        dtype=str,
    )


def current_concrete_labels(
    uid: str, n: int, label_map: dict[str, dict[int, str]]
) -> np.ndarray:
    if uid not in label_map:
        raise KeyError(f"Official label JSON has no entry for {uid}")
    labels = np.full(n, "", dtype="<U16")
    for index, raw_label in label_map[uid].items():
        label = clean_label(raw_label)
        if label and "?" not in label:
            if not 0 <= index < n:
                raise IndexError(f"{uid}: label index {index} outside 0..{n - 1}")
            labels[index] = label
    return labels


def common_label_matrix(labels: list[np.ndarray]) -> np.ndarray:
    sets = [set(str(x) for x in row if str(x)) for row in labels]
    return np.asarray(
        [[len(left.intersection(right)) for right in sets] for left in sets],
        dtype=np.int16,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Figure-C.1-audited Atanas-21 GWOT-MD inputs"
    )
    parser.add_argument("--h5-dir", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument(
        "--old-npz-root",
        type=Path,
        default=ROOT / "Data/Atanas_SF_unified_000776/full",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "baselines/gwot_md/data_paper_snapshot_v1",
    )
    parser.add_argument("--fallback-sample-rate-hz", type=float, default=1.67)
    parser.add_argument("--verify-h5-sha256", action="store_true")
    parser.add_argument(
        "--allow-figure-mismatch",
        action="store_true",
        help="Diagnostic only: emit inputs even if Figure C.1 does not match",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_dir = args.h5_dir.expanduser().resolve()
    labels_path = args.labels.expanduser().resolve()
    old_root = args.old_npz_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    label_map = load_label_json(labels_path)

    prepared: list[dict[str, Any]] = []
    all_labels: list[np.ndarray] = []
    provenance_rows: list[dict[str, Any]] = []

    for uid, official_name, expected_h5_sha256, paper_count in PAPER_COHORT:
        h5_path = find_input_file(h5_dir, uid, official_name)
        if h5_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError(f"{uid}: expected official H5, got {h5_path}")
        actual_h5_sha256 = sha256_file(h5_path) if args.verify_h5_sha256 else None
        if args.verify_h5_sha256 and actual_h5_sha256 != expected_h5_sha256:
            raise ValueError(
                f"{uid}: H5 SHA-256 {actual_h5_sha256} != {expected_h5_sha256}"
            )

        worm = load_h5_worm(
            h5_path, uid, label_map, args.fallback_sample_rate_hz
        )
        current = current_concrete_labels(uid, worm.n, label_map)
        old_path: Path | None = None
        restorations: list[dict[str, Any]] = []

        if uid == MISSING_FROM_DANDI:
            final_labels = current
            label_source = "Zenodo v4 neuropal_label.json.bz2 (concrete labels only)"
        else:
            old_path = find_old_npz(old_root, uid)
            final_labels = old_clean_labels(old_path, worm.n)
            label_source = "older DANDI extraction cell_id filtered by clean_mask"
            for index, expected_label in V4_RESTORATIONS.get(uid, {}).items():
                observed = str(current[index])
                if observed != expected_label:
                    raise ValueError(
                        f"{uid}[{index}]: official v4 label {observed!r} != {expected_label!r}"
                    )
                if final_labels[index]:
                    raise ValueError(
                        f"{uid}[{index}]: refusing to overwrite old label {final_labels[index]!r}"
                    )
                final_labels[index] = observed
                restorations.append(
                    {
                        "zero_based_index": index,
                        "label": observed,
                        "source": "Zenodo v4 official label JSON",
                    }
                )

        observed_count = len(set(str(x) for x in final_labels if str(x)))
        if observed_count != paper_count:
            raise ValueError(
                f"{uid}: {observed_count} unique concrete labels != paper diagonal {paper_count}"
            )
        timestamps = np.arange(worm.activity.shape[1], dtype=np.float64) / worm.sample_rate_hz
        prepared.append(
            {
                "uid": uid,
                "activity": worm.activity,
                "labels": final_labels,
                "timestamps": timestamps,
                "sample_rate_hz": worm.sample_rate_hz,
            }
        )
        all_labels.append(final_labels)
        provenance_rows.append(
            {
                "uid": uid,
                "neurons": worm.n,
                "timepoints": int(worm.activity.shape[1]),
                "paper_label_count": paper_count,
                "prepared_label_count": observed_count,
                "label_source": label_source,
                "restorations": restorations,
                "h5": {
                    "path": str(h5_path),
                    "expected_sha256": expected_h5_sha256,
                    "verified_sha256": actual_h5_sha256,
                },
                "old_npz": file_digest(old_path) if old_path is not None else None,
            }
        )

    observed_matrix = common_label_matrix(all_labels)
    delta = observed_matrix.astype(int) - PAPER_COMMON_LABELS.astype(int)
    mismatch_indices = np.argwhere(delta != 0)
    audit = {
        "passed": bool(not mismatch_indices.size),
        "observed": observed_matrix.tolist(),
        "expected": PAPER_COMMON_LABELS.tolist(),
        "mismatched_cells": int(len(mismatch_indices)),
        "max_absolute_delta": int(np.max(np.abs(delta))),
        "mismatches": [
            {
                "row": int(i + 1),
                "column": int(j + 1),
                "uid_row": PAPER_COHORT[int(i)][0],
                "uid_column": PAPER_COHORT[int(j)][0],
                "observed": int(observed_matrix[i, j]),
                "expected": int(PAPER_COMMON_LABELS[i, j]),
                "delta": int(delta[i, j]),
            }
            for i, j in mismatch_indices
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "figure_c1_audit.json", audit)
    write_json(
        output_dir / "provenance.json",
        {
            "protocol": "GWOT-MD Atanas-21 paper-input reconstruction",
            "activity_source": "official Atanas processed_h5.tar.bz2",
            "official_label_json": file_digest(labels_path),
            "paper_code": str(PAPER_CODE),
            "figure_c1_audit_passed": audit["passed"],
            "records": provenance_rows,
        },
    )
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["uid", "neurons", "timepoints", "sample_rate_hz", "npz"],
        )
        writer.writeheader()
        for row in prepared:
            writer.writerow(
                {
                    "uid": row["uid"],
                    "neurons": row["activity"].shape[0],
                    "timepoints": row["activity"].shape[1],
                    "sample_rate_hz": row["sample_rate_hz"],
                    "npz": f"{row['uid']}.npz",
                }
            )

    if not audit["passed"] and not args.allow_figure_mismatch:
        raise SystemExit(
            f"Figure C.1 audit failed in {len(mismatch_indices)} cells; "
            f"see {output_dir / 'figure_c1_audit.json'}"
        )

    for row in prepared:
        np.savez_compressed(
            output_dir / f"{row['uid']}.npz",
            activity_raw=np.asarray(row["activity"], dtype=np.float64),
            cell_id=np.asarray(row["labels"], dtype=str),
            timestamps=np.asarray(row["timestamps"], dtype=np.float64),
            sampling_rate_hz=np.asarray(row["sample_rate_hz"], dtype=np.float64),
            recording_uid=np.asarray(row["uid"]),
        )
    print(
        json.dumps(
            {
                "status": "ok" if audit["passed"] else "diagnostic-only",
                "output_dir": str(output_dir),
                "recordings": len(prepared),
                "figure_c1_mismatched_cells": len(mismatch_indices),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
