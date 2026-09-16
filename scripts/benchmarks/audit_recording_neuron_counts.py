#!/usr/bin/env python3
"""Audit observed, annotated, and evaluated neuron counts.

The definitions deliberately mirror the NeurID/MPRT data and evaluation
contract:

* observed: nodes retained by ``mprt_net.data.load_worm``;
* annotated: recording-local unique identities eligible for supervision;
* evaluated: annotated test identities represented in the outer-train union.

The script reads the canonical Atanas and Kato/RLD preprocessing snapshots and
the grouped CV5 manifests.  It does not read predictions or metric files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
INVALID_IDENTITIES = {"", "nan", "none", "null", "unknown", "unk"}
SUBJECT_PATTERN = re.compile(r"(?:^|/)sub-([^/]+)(?:/|$)")


@dataclass(frozen=True)
class RecordingAudit:
    recording_id: str
    subject_id: str
    observed_count: int
    annotated_count: int
    identities: frozenset[str]
    nonempty_label_count: int
    eligible_before_unique_count: int


DATASETS = {
    "Atanas": {
        "manifest": REPO_ROOT / "Data/Atanas_SF_unified_000776/dataset_manifest.csv",
        "cv_root": REPO_ROOT / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
        "id_column": "uid",
        "path_column": "full_npz",
        "organism": "C. elegans",
        "setting": "freely moving whole-brain calcium imaging",
    },
    "Kato": {
        "manifest": REPO_ROOT
        / "Data/Dunn_001623/date_disjoint_full95_v1/split_manifest.csv",
        "cv_root": REPO_ROOT / "Data/Dunn_001623/cv5_grouped_v1",
        "id_column": "recording",
        "path_column": "npz",
        "organism": "C. elegans",
        "setting": "freely moving whole-brain calcium imaging (RLD)",
    },
}


def _mask(data: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    if key not in data.files:
        return np.ones(n, dtype=bool)
    value = np.asarray(data[key], dtype=bool)
    if value.shape != (n,):
        raise ValueError(f"{key}: expected shape {(n,)}, got {value.shape}")
    return value


def _scalar_string(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        return ""
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{key}: expected a scalar, got shape {value.shape}")
    return str(value.item()).strip()


def _subject_id(data: np.lib.npyio.NpzFile, recording_id: str) -> str:
    explicit = _scalar_string(data, "subject_id")
    if explicit:
        return explicit
    asset_path = _scalar_string(data, "asset_path")
    match = SUBJECT_PATTERN.search(asset_path)
    if match:
        return match.group(1)
    # Both canonical datasets use one recording per biological specimen.  This
    # fallback is explicit so that future inputs without subject metadata do
    # not silently collapse multiple recordings into one subject.
    return recording_id


def audit_npz(path: Path, expected_recording_id: str | None = None) -> RecordingAudit:
    with np.load(path, allow_pickle=False) as data:
        missing = {"xyz", "activity_raw", "cell_id"}.difference(data.files)
        if missing:
            raise KeyError(f"{path}: missing keys {sorted(missing)}")

        xyz = np.asarray(data["xyz"])
        labels = np.asarray(data["cell_id"]).astype(str)
        activity = np.asarray(data["activity_raw"])
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"{path}: xyz must have shape [N, 3], got {xyz.shape}")
        n = xyz.shape[0]
        if labels.shape != (n,):
            raise ValueError(f"{path}: cell_id shape {labels.shape} != {(n,)}")
        if activity.ndim != 2 or n not in activity.shape:
            raise ValueError(f"{path}: activity_raw shape {activity.shape} is incompatible with N={n}")

        recording_id = _scalar_string(data, "recording_uid") or path.stem
        if expected_recording_id is not None and recording_id != expected_recording_id:
            raise ValueError(
                f"{path}: recording_uid={recording_id!r}, expected {expected_recording_id!r}"
            )

        # This is the exact node-selection predicate in mprt_net.data.load_worm.
        observed = np.isfinite(xyz).all(axis=1) & _mask(data, "valid_xyz_mask", n)
        valid_id = np.asarray(
            [label.strip().lower() not in INVALID_IDENTITIES for label in labels],
            dtype=bool,
        )
        eligible = observed & valid_id
        for key in ("labeled_mask", "certain_mask", "clean_mask"):
            eligible &= _mask(data, key, n)

        eligible_labels = [label.strip() for label in labels[eligible]]
        counts = Counter(eligible_labels)
        unique_identities = frozenset(label for label, count in counts.items() if count == 1)
        annotated_count = sum(counts[label] for label in unique_identities)
        nonempty = sum(label.strip().lower() not in INVALID_IDENTITIES for label in labels[observed])

        return RecordingAudit(
            recording_id=recording_id,
            subject_id=_subject_id(data, recording_id),
            observed_count=int(observed.sum()),
            annotated_count=int(annotated_count),
            identities=unique_identities,
            nonempty_label_count=int(nonempty),
            eligible_before_unique_count=len(eligible_labels),
        )


def annotation_filter_exclusions(path: Path) -> list[dict[str, object]]:
    """Explain why non-empty labels in one NPZ are not annotations."""
    output: list[dict[str, object]] = []
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"])
        labels = np.asarray(data["cell_id"]).astype(str)
        n = len(labels)
        masks = {
            "finite_xyz": np.isfinite(xyz).all(axis=1),
            "valid_xyz_mask": _mask(data, "valid_xyz_mask", n),
            "labeled_mask": _mask(data, "labeled_mask", n),
            "certain_mask": _mask(data, "certain_mask", n),
            "clean_mask": _mask(data, "clean_mask", n),
        }
        for index, raw_label in enumerate(labels):
            label = raw_label.strip()
            if label.lower() in INVALID_IDENTITIES:
                continue
            reasons = [name for name, values in masks.items() if not values[index]]
            if reasons:
                output.append(
                    {
                        "node_index": index,
                        "raw_identity": label,
                        "reasons": reasons,
                    }
                )
    return output


def resolve_repo_path(raw_path: str, fallback_parent: Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_file():
        return path
    if fallback_parent is not None:
        fallback = fallback_parent / path.name
        if fallback.is_file():
            return fallback
    # Manifests can contain absolute paths from another checkout.  Preserve the
    # suffix beginning at Data/ when resolving them in this checkout.
    parts = path.parts
    if "Data" in parts:
        candidate = REPO_ROOT.joinpath(*parts[parts.index("Data") :])
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(raw_path)


def read_dataset_recordings(dataset: str) -> list[RecordingAudit]:
    config = DATASETS[dataset]
    manifest = Path(config["manifest"])
    rows: list[RecordingAudit] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            recording_id = row[str(config["id_column"])]
            path = resolve_repo_path(row[str(config["path_column"])])
            rows.append(audit_npz(path, recording_id))
    if len({row.recording_id for row in rows}) != len(rows):
        raise ValueError(f"{dataset}: duplicate recording_id in {manifest}")
    return sorted(rows, key=lambda row: row.recording_id)


def _record_path(record: dict[str, object], fold_root: Path) -> Path:
    return resolve_repo_path(
        str(record["link_path"]), fold_root / str(record["split"])
    )


def evaluation_audit(dataset: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    cv_root = Path(DATASETS[dataset]["cv_root"])
    output: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for fold in range(5):
        fold_root = cv_root / f"fold_{fold}"
        manifest = json.loads((fold_root / "manifest.json").read_text(encoding="utf-8"))
        records = manifest["records"]
        train_union: set[str] = set()
        for record in records:
            if record["split"] == "train":
                train_union.update(audit_npz(_record_path(record, fold_root)).identities)

        for record in records:
            if record["split"] != "test":
                continue
            query = audit_npz(_record_path(record, fold_root), str(record["group"]))
            excluded = sorted(query.identities.difference(train_union))
            evaluated = query.identities.intersection(train_union)
            output.append(
                {
                    "fold": fold,
                    "recording_id": query.recording_id,
                    "evaluated_count": len(evaluated),
                }
            )
            for identity in excluded:
                exclusions.append(
                    {
                        "dataset": dataset,
                        "fold": fold,
                        "recording_id": query.recording_id,
                        "identity": identity,
                        "reason": "absent_from_outer_train_identity_union",
                    }
                )
    return sorted(output, key=lambda row: (int(row["fold"]), str(row["recording_id"]))), exclusions


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def range_string(values: Sequence[int]) -> str:
    return f"{min(values)}-{max(values)}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "exports/recording_neuron_count_audit_20260913",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_recordings: dict[str, list[RecordingAudit]] = {}
    all_evaluations: dict[str, list[dict[str, object]]] = {}
    all_exclusions: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    overview_rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        recordings = read_dataset_recordings(dataset)
        all_recordings[dataset] = recordings
        recording_rows = [
            {
                "recording_id": row.recording_id,
                "subject_id": row.subject_id,
                "observed_count": row.observed_count,
                "annotated_count": row.annotated_count,
            }
            for row in recordings
        ]
        write_csv(
            args.output_dir / f"{dataset.lower()}_recording_counts.csv",
            ("recording_id", "subject_id", "observed_count", "annotated_count"),
            recording_rows,
        )

        observed = [row.observed_count for row in recordings]
        annotated = [row.annotated_count for row in recordings]
        subjects = {row.subject_id for row in recordings}
        summary_rows.append(
            {
                "dataset": dataset,
                "number_of_subjects": len(subjects),
                "number_of_recordings": len(recordings),
                "min_observed": min(observed),
                "max_observed": max(observed),
                "mean_observed": f"{mean(observed):.6f}",
                "median_observed": f"{median(observed):.6f}",
                "min_annotated": min(annotated),
                "max_annotated": max(annotated),
                "mean_annotated": f"{mean(annotated):.6f}",
                "median_annotated": f"{median(annotated):.6f}",
            }
        )
        config = DATASETS[dataset]
        overview_rows.append(
            {
                "Dataset": dataset,
                "Organism": config["organism"],
                "Setting": config["setting"],
                "Subjects": len(subjects),
                "Observed": range_string(observed),
                "Annotated": range_string(annotated),
            }
        )

        evaluations, exclusions = evaluation_audit(dataset)
        if len(evaluations) != len(recordings):
            raise AssertionError(
                f"{dataset}: CV test partition has {len(evaluations)} rows for "
                f"{len(recordings)} recordings"
            )
        if {str(row["recording_id"]) for row in evaluations} != {
            row.recording_id for row in recordings
        }:
            raise AssertionError(f"{dataset}: every recording must occur in exactly one test fold")
        annotated_by_id = {row.recording_id: row.annotated_count for row in recordings}
        if any(
            int(row["evaluated_count"]) > annotated_by_id[str(row["recording_id"])]
            for row in evaluations
        ):
            raise AssertionError(f"{dataset}: evaluated_count exceeds annotated_count")
        all_evaluations[dataset] = evaluations
        all_exclusions.extend(exclusions)
        write_csv(
            args.output_dir / f"{dataset.lower()}_evaluation_audit.csv",
            ("fold", "recording_id", "evaluated_count"),
            evaluations,
        )

    write_csv(
        args.output_dir / "dataset_summary.csv",
        (
            "dataset",
            "number_of_subjects",
            "number_of_recordings",
            "min_observed",
            "max_observed",
            "mean_observed",
            "median_observed",
            "min_annotated",
            "max_annotated",
            "mean_annotated",
            "median_annotated",
        ),
        summary_rows,
    )
    write_csv(
        args.output_dir / "table1_dataset_overview.csv",
        ("Dataset", "Organism", "Setting", "Subjects", "Observed", "Annotated"),
        overview_rows,
    )
    write_csv(
        args.output_dir / "evaluation_audit.csv",
        ("dataset", "fold", "recording_id", "evaluated_count"),
        (
            {"dataset": dataset, **row}
            for dataset, rows in all_evaluations.items()
            for row in rows
        ),
    )
    write_csv(
        args.output_dir / "evaluation_exclusions.csv",
        ("dataset", "fold", "recording_id", "identity", "reason"),
        all_exclusions,
    )

    target = next(
        row for row in all_recordings["Atanas"] if row.recording_id == "2022-08-02-01"
    )
    with Path(DATASETS["Atanas"]["manifest"]).open(
        newline="", encoding="utf-8"
    ) as handle:
        target_manifest_row = next(
            row for row in csv.DictReader(handle) if row["uid"] == target.recording_id
        )
    target_path = resolve_repo_path(target_manifest_row["full_npz"])
    target_filter_exclusions = annotation_filter_exclusions(target_path)
    target_eval = next(
        row
        for row in all_evaluations["Atanas"]
        if row["recording_id"] == target.recording_id
    )
    target_exclusions = [
        row["identity"]
        for row in all_exclusions
        if row["dataset"] == "Atanas" and row["recording_id"] == target.recording_id
    ]
    target_report = {
        "dataset": "Atanas",
        "recording_id": target.recording_id,
        "fold": target_eval["fold"],
        "observed_count": target.observed_count,
        "nonempty_label_count": target.nonempty_label_count,
        "nonempty_but_not_annotated_count": len(target_filter_exclusions),
        "nonempty_but_not_annotated_rows": target_filter_exclusions,
        "eligible_before_unique_count": target.eligible_before_unique_count,
        "annotated_count": target.annotated_count,
        "evaluated_count": target_eval["evaluated_count"],
        "annotated_but_not_evaluated_count": len(target_exclusions),
        "annotated_but_not_evaluated_identities": target_exclusions,
    }
    (args.output_dir / "2022-08-02-01_audit.json").write_text(
        json.dumps(target_report, indent=2) + "\n", encoding="utf-8"
    )

    readme = """# Recording-level neuron count audit

This directory is generated by
`python scripts/benchmarks/audit_recording_neuron_counts.py` from the canonical
Atanas and Kato/RLD preprocessing snapshots and grouped CV5 manifests.

Definitions:

- **observed**: all nodes retained by the NeurID loader: finite XYZ and
  `valid_xyz_mask=True`. Unlabeled nodes remain population context.
- **annotated**: observed nodes with a non-empty valid identity and all of
  `labeled_mask`, `certain_mask`, and `clean_mask` true, after requiring the
  identity to occur exactly once within that recording.
- **evaluated**: annotated held-out test nodes whose identity occurs in at
  least one outer-training recording in that fold. Validation identities are
  not used to expand this vocabulary.

The two dataset-specific recording CSVs have exactly the requested four
columns. The two dataset-specific evaluation CSVs have exactly the requested
three columns; `evaluation_audit.csv` combines them with a `dataset` column.
`evaluation_exclusions.csv` records every annotation excluded only because its
identity is absent from the fold's outer-training union.

Subject caveat: Kato/RLD NPZs store `subject_id=recording_uid`; Atanas subject
IDs are read from the NWB/BIDS asset path. Thus the reported distinct-subject
counts follow the canonical metadata/group keys. Biological equivalence of
separately named recordings was not independently re-verified here.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary_rows, "target": target_report}, indent=2))


if __name__ == "__main__":
    main()
