#!/usr/bin/env python3
"""Create an experiment-date-disjoint 27/6/5 split for Atanas/SF NPZ files.

The split is selected using recording filenames only. No neural activity, xyz,
identity labels, model predictions, or previous test results are read.

Default protocol (v1):
- group key: first YYYY-MM-DD in each NPZ filename
- exact animal counts: train=27, val=6, test=5
- at least three experiment dates in val and test
- select one feasible group partition deterministically by SHA256(seed, dates)
- materialize with relative symlinks by default

All NPZ files are gathered from SOURCE_ROOT/{train,val,test}; the previous split
assignment is recorded but does not influence the new assignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True,
                   help="Existing root containing train/, val/, test/ NPZ folders.")
    p.add_argument("--output-root", type=Path, required=True,
                   help="New root where train/, val/, test/ will be created.")
    p.add_argument("--train-count", type=int, default=27)
    p.add_argument("--val-count", type=int, default=6)
    p.add_argument("--test-count", type=int, default=5)
    p.add_argument("--min-val-dates", type=int, default=3)
    p.add_argument("--min-test-dates", type=int, default=3)
    p.add_argument("--max-val-dates", type=int, default=6)
    p.add_argument("--max-test-dates", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def extract_date(path: Path) -> str:
    m = DATE_RE.search(path.stem)
    if not m:
        raise ValueError(f"Could not extract YYYY-MM-DD from filename: {path.name}")
    return m.group(1)


def gather(source_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen_names: dict[str, Path] = {}
    seen_resolved: dict[Path, Path] = {}
    for old_split in ("train", "val", "test"):
        folder = source_root / old_split
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.glob("*.npz")):
            if path.name in seen_names:
                raise RuntimeError(
                    f"Duplicate filename {path.name}: {seen_names[path.name]} and {path}"
                )
            resolved = path.resolve()
            if resolved in seen_resolved:
                raise RuntimeError(
                    f"Same source file appears twice: {seen_resolved[resolved]} and {path}"
                )
            seen_names[path.name] = path
            seen_resolved[resolved] = path
            rows.append({
                "filename": path.name,
                "recording": path.stem,
                "date": extract_date(path),
                "old_split": old_split,
                "source_path": path.resolve(),
            })
    return rows


def choose_partition(
    date_counts: dict[str, int],
    train_count: int,
    val_count: int,
    test_count: int,
    min_val_dates: int,
    min_test_dates: int,
    max_val_dates: int,
    max_test_dates: int,
    seed: int,
) -> tuple[set[str], set[str], set[str], str, int, dict[str, float]]:
    dates = sorted(date_counts)
    total = sum(date_counts.values())
    if total != train_count + val_count + test_count:
        raise ValueError(
            f"Total files={total}, requested split total="
            f"{train_count + val_count + test_count}"
        )

    # Metadata-only balancing: use ordinal experiment-date ranks, weighted by the
    # number of animals acquired on each date. This keeps train/val/test centered
    # on a similar part of the acquisition timeline without looking at labels or
    # model results. SHA256(seed, partition) is used only as the final tie-break.
    rank = {d: i / max(1, len(dates) - 1) for i, d in enumerate(dates)}
    overall_mean = sum(rank[d] * date_counts[d] for d in dates) / total

    candidates: list[tuple[tuple[float, float, float, int, str], tuple[str, ...], tuple[str, ...], dict[str, float]]] = []
    for n_test_dates in range(min_test_dates, min(max_test_dates, len(dates)) + 1):
        for test_dates in itertools.combinations(dates, n_test_dates):
            if sum(date_counts[d] for d in test_dates) != test_count:
                continue
            remaining = [d for d in dates if d not in test_dates]
            for n_val_dates in range(min_val_dates, min(max_val_dates, len(remaining)) + 1):
                for val_dates in itertools.combinations(remaining, n_val_dates):
                    if sum(date_counts[d] for d in val_dates) != val_count:
                        continue
                    train_dates = tuple(d for d in remaining if d not in val_dates)
                    if sum(date_counts[d] for d in train_dates) != train_count:
                        continue

                    def weighted_mean(ds: tuple[str, ...]) -> float:
                        denom = sum(date_counts[d] for d in ds)
                        return sum(rank[d] * date_counts[d] for d in ds) / denom

                    means = {
                        "train": weighted_mean(train_dates),
                        "val": weighted_mean(val_dates),
                        "test": weighted_mean(test_dates),
                    }
                    mean_imbalance = sum(abs(x - overall_mean) for x in means.values())
                    val_span = max(rank[d] for d in val_dates) - min(rank[d] for d in val_dates)
                    test_span = max(rank[d] for d in test_dates) - min(rank[d] for d in test_dates)
                    min_holdout_span = min(val_span, test_span)
                    total_holdout_span = val_span + test_span
                    date_count_gap = abs(len(val_dates) - len(test_dates))

                    token = (
                        f"seed={seed}|test={','.join(test_dates)}|"
                        f"val={','.join(val_dates)}"
                    )
                    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
                    objective = (
                        round(mean_imbalance, 12),
                        -round(min_holdout_span, 12),
                        -round(total_holdout_span, 12),
                        date_count_gap,
                        digest,
                    )
                    diagnostics = {
                        "overall_weighted_date_rank": overall_mean,
                        "train_weighted_date_rank": means["train"],
                        "val_weighted_date_rank": means["val"],
                        "test_weighted_date_rank": means["test"],
                        "mean_rank_imbalance": mean_imbalance,
                        "val_date_span": val_span,
                        "test_date_span": test_span,
                    }
                    candidates.append((objective, test_dates, val_dates, diagnostics))

    if not candidates:
        raise RuntimeError(
            "No date-disjoint partition satisfies the requested exact counts and "
            "date-group constraints."
        )

    objective, test_tuple, val_tuple, diagnostics = min(candidates, key=lambda x: x[0])
    test_dates = set(test_tuple)
    val_dates = set(val_tuple)
    train_dates = set(dates) - test_dates - val_dates
    partition_hash = str(objective[-1])
    return train_dates, val_dates, test_dates, partition_hash, len(candidates), diagnostics


def materialize(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        relative = os.path.relpath(src, start=dst.parent)
        dst.symlink_to(relative)
    elif mode == "hardlink":
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()

    if output_root == source_root:
        raise ValueError("--output-root must differ from --source-root")
    if output_root.exists():
        if not args.force:
            raise FileExistsError(
                f"{output_root} already exists. Pass --force to recreate it."
            )
        shutil.rmtree(output_root)

    source_rows = gather(source_root)
    date_counts = Counter(str(row["date"]) for row in source_rows)

    train_dates, val_dates, test_dates, partition_hash, feasible_count, balance_diagnostics = choose_partition(
        dict(date_counts),
        args.train_count,
        args.val_count,
        args.test_count,
        args.min_val_dates,
        args.min_test_dates,
        args.max_val_dates,
        args.max_test_dates,
        args.seed,
    )

    split_by_date = {d: "train" for d in train_dates}
    split_by_date.update({d: "val" for d in val_dates})
    split_by_date.update({d: "test" for d in test_dates})

    output_root.mkdir(parents=True, exist_ok=False)
    for split in ("train", "val", "test"):
        (output_root / split).mkdir()

    manifest: list[dict[str, object]] = []
    for row in sorted(source_rows, key=lambda x: str(x["filename"])):
        new_split = split_by_date[str(row["date"])]
        src = Path(row["source_path"])
        dst = output_root / new_split / str(row["filename"])
        materialize(src, dst, args.mode)
        manifest.append({
            "recording": row["recording"],
            "filename": row["filename"],
            "date": row["date"],
            "old_split": row["old_split"],
            "new_split": new_split,
            "source_path": str(src),
            "output_path": str(dst),
            "materialization": args.mode,
        })

    split_counts = Counter(str(row["new_split"]) for row in manifest)
    dates_by_split: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        dates_by_split[str(row["new_split"])].add(str(row["date"]))

    if split_counts != Counter({
        "train": args.train_count,
        "val": args.val_count,
        "test": args.test_count,
    }):
        raise AssertionError(split_counts)
    if dates_by_split["train"] & dates_by_split["val"]:
        raise AssertionError("train/val date overlap")
    if dates_by_split["train"] & dates_by_split["test"]:
        raise AssertionError("train/test date overlap")
    if dates_by_split["val"] & dates_by_split["test"]:
        raise AssertionError("val/test date overlap")

    summary = {
        "protocol": "experiment_date_disjoint_v1",
        "selection_basis": (
            "Filename date groups and requested split sizes only; no activity, xyz, "
            "identity labels, predictions, or prior metrics were used."
        ),
        "partition_rule": (
            "Enumerate all feasible exact-count date-group partitions. Select the "
            "partition with the smallest worm-weighted experiment-date-rank imbalance; "
            "then prefer wider validation/test date spans. SHA256(seed, partition) is "
            "used only as the final deterministic tie-break."
        ),
        "balance_diagnostics": balance_diagnostics,
        "seed": args.seed,
        "partition_hash": partition_hash,
        "num_feasible_partitions": feasible_count,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "materialization": args.mode,
        "num_recordings": len(manifest),
        "split_counts": dict(sorted(split_counts.items())),
        "date_counts": dict(sorted(date_counts.items())),
        "split_dates": {
            split: sorted(dates_by_split[split])
            for split in ("train", "val", "test")
        },
        "date_overlap": {
            "train_val": sorted(dates_by_split["train"] & dates_by_split["val"]),
            "train_test": sorted(dates_by_split["train"] & dates_by_split["test"]),
            "val_test": sorted(dates_by_split["val"] & dates_by_split["test"]),
        },
    }

    write_csv(output_root / "split_manifest.csv", manifest)
    (output_root / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nFiles by split:")
    for split in ("train", "val", "test"):
        print(f"\n[{split}]")
        for row in manifest:
            if row["new_split"] == split:
                print(f"  {row['filename']}")
    print(f"\nSaved manifest: {output_root / 'split_manifest.csv'}")
    print(f"Saved summary : {output_root / 'split_summary.json'}")


if __name__ == "__main__":
    main()
