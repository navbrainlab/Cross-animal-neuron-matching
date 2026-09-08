from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


GROUP_KEYS = ("worm_id", "animal_id", "subject_id", "individual_id", "recording_uid")


@dataclass(frozen=True)
class Record:
    path: Path
    source_split: str
    group: str
    supervised_identities: int


def _scalar_string(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Group metadata must be scalar")
    return str(array.reshape(-1)[0]).strip()


def _label_count(z: np.lib.npyio.NpzFile) -> int:
    ids = np.asarray(z["cell_id"]).astype(str)
    mask = np.ones(len(ids), dtype=bool)
    for key in ("labeled_mask", "certain_mask", "clean_mask"):
        if key in z.files:
            mask &= np.asarray(z[key], dtype=bool)
    valid = [
        value.strip()
        for value, keep in zip(ids, mask)
        if keep and value.strip().lower() not in {"", "nan", "none", "null", "unknown", "unk"}
    ]
    counts = Counter(valid)
    return sum(count == 1 for count in counts.values())


def _group_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"file", "group"}.issubset(reader.fieldnames):
            raise ValueError("--group-map CSV needs file and group columns")
        for row in reader:
            result[str(Path(row["file"]).resolve())] = row["group"].strip()
    return result


def _resolve_group(
    path: Path,
    z: np.lib.npyio.NpzFile,
    group_key: str,
    group_regex: re.Pattern[str] | None,
    explicit: dict[str, str],
) -> tuple[str, str]:
    absolute = str(path.resolve())
    if absolute in explicit:
        return explicit[absolute], "group_map"
    if group_key != "auto":
        if group_key not in z.files:
            raise KeyError(f"{path} has no requested group key {group_key!r}")
        raw, source = _scalar_string(z[group_key]), group_key
    else:
        available = [key for key in GROUP_KEYS if key in z.files]
        if available:
            source = available[0]
            raw = _scalar_string(z[source])
        else:
            source = "filename"
            raw = path.stem
    if group_regex is not None:
        match = group_regex.search(raw)
        if not match:
            raise ValueError(f"Group regex did not match {raw!r} from {path}")
        raw = match.group(1) if match.groups() else match.group(0)
        source += "+regex"
    raw = raw.strip()
    if not raw:
        raise ValueError(f"Empty group id for {path}")
    return raw, source


def collect_records(
    dataset_root: Path,
    group_key: str,
    group_regex: str | None,
    group_map_path: Path | None,
) -> tuple[list[Record], set[str]]:
    explicit = _group_map(group_map_path)
    compiled = re.compile(group_regex) if group_regex else None
    records: list[Record] = []
    sources: set[str] = set()
    seen: set[Path] = set()
    for split in ("train", "val", "test"):
        directory = dataset_root / split
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.npz")):
            resolved = path.resolve()
            if resolved in seen:
                raise ValueError(f"The same source file appears more than once: {resolved}")
            seen.add(resolved)
            with np.load(path, allow_pickle=False) as z:
                group, source = _resolve_group(
                    path, z, group_key, compiled, explicit
                )
                labels = _label_count(z)
            sources.add(source)
            records.append(Record(resolved, split, group, labels))
    if not records:
        raise FileNotFoundError(f"No NPZ records under {dataset_root}/{{train,val,test}}")
    return records, sources


def assign_groups(records: list[Record], folds: int, seed: int) -> list[list[str]]:
    by_group: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        by_group[record.group].append(record)
    if len(by_group) < folds:
        raise ValueError(f"Need at least {folds} groups, found {len(by_group)}")
    randomizer = random.Random(seed)
    items = list(by_group.items())
    randomizer.shuffle(items)
    items.sort(
        key=lambda item: (
            -len(item[1]),
            -sum(record.supervised_identities for record in item[1]),
        )
    )
    assigned: list[list[str]] = [[] for _ in range(folds)]
    recording_load = [0] * folds
    label_load = [0] * folds
    for group, group_records in items:
        candidate = min(
            range(folds),
            key=lambda index: (
                recording_load[index],
                label_load[index],
                len(assigned[index]),
                index,
            ),
        )
        assigned[candidate].append(group)
        recording_load[candidate] += len(group_records)
        label_load[candidate] += sum(
            record.supervised_identities for record in group_records
        )
    return assigned


def _safe_name(record: Record) -> str:
    digest = hashlib.sha1(str(record.path).encode("utf-8")).hexdigest()[:8]
    return f"{record.source_split}__{record.path.stem}__{digest}.npz"


def materialize_folds(
    records: list[Record], assigned: list[list[str]], output_root: Path
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Refusing to modify non-empty fold root: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    group_to_bucket = {
        group: bucket for bucket, groups in enumerate(assigned) for group in groups
    }
    fold_manifests: list[dict[str, Any]] = []
    test_coverage: Counter[str] = Counter()
    for fold in range(len(assigned)):
        split_buckets = {
            "test": {fold},
            "val": {(fold + 1) % len(assigned)},
            "train": set(range(len(assigned))) - {fold, (fold + 1) % len(assigned)},
        }
        fold_root = output_root / f"fold_{fold}"
        manifest_rows: list[dict[str, Any]] = []
        split_groups: dict[str, set[str]] = {}
        for split, buckets in split_buckets.items():
            selected_groups = {
                group for group, bucket in group_to_bucket.items() if bucket in buckets
            }
            split_groups[split] = selected_groups
            directory = fold_root / split
            directory.mkdir(parents=True, exist_ok=False)
            for record in records:
                if record.group not in selected_groups:
                    continue
                destination = directory / _safe_name(record)
                destination.symlink_to(record.path)
                manifest_rows.append(
                    {
                        "split": split,
                        "group": record.group,
                        "source_split": record.source_split,
                        "source_path": str(record.path),
                        "link_path": str(destination),
                        "supervised_identities": record.supervised_identities,
                    }
                )
                if split == "test":
                    test_coverage[record.group] += 1
        if not (
            split_groups["train"].isdisjoint(split_groups["val"])
            and split_groups["train"].isdisjoint(split_groups["test"])
            and split_groups["val"].isdisjoint(split_groups["test"])
        ):
            raise AssertionError("Group leakage detected while constructing folds")
        fold_manifest = {
            "fold": fold,
            "root": str(fold_root),
            "groups": {key: sorted(value) for key, value in split_groups.items()},
            "records": manifest_rows,
        }
        (fold_root / "manifest.json").write_text(
            json.dumps(fold_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        fold_manifests.append(fold_manifest)
    expected_groups = set(group_to_bucket)
    if set(test_coverage) != expected_groups or any(
        test_coverage[group] != len([record for record in records if record.group == group])
        for group in expected_groups
    ):
        raise AssertionError("Each source record must appear in test exactly once")
    return {
        "folds": len(assigned),
        "groups": len(expected_groups),
        "records": len(records),
        "bucket_groups": assigned,
        "fold_manifests": fold_manifests,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leakage-audited grouped outer CV folds from locked splits"
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split-seed", type=int, default=20260823)
    parser.add_argument("--group-key", default="auto")
    parser.add_argument(
        "--group-regex",
        default=None,
        help="Optional regex applied to metadata; first capture group is the animal id.",
    )
    parser.add_argument(
        "--group-map",
        type=Path,
        default=None,
        help="Optional CSV with absolute/relative file and group columns.",
    )
    args = parser.parse_args()
    if args.folds < 3:
        parser.error("At least three folds are required for train/val/test rotation")
    records, group_sources = collect_records(
        args.dataset_root,
        args.group_key,
        args.group_regex,
        args.group_map,
    )
    assigned = assign_groups(records, args.folds, args.split_seed)
    summary = materialize_folds(records, assigned, args.output_root)
    summary.update(
        {
            "dataset_root": str(args.dataset_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "split_seed": args.split_seed,
            "group_key": args.group_key,
            "group_regex": args.group_regex,
            "group_sources": sorted(group_sources),
            "warning": (
                "recording_uid/filename was used for at least one record; verify that it "
                "does not split repeated recordings from one biological animal"
                if group_sources.intersection({"recording_uid", "filename", "recording_uid+regex", "filename+regex"})
                else None
            ),
        }
    )
    (args.output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved={args.output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
