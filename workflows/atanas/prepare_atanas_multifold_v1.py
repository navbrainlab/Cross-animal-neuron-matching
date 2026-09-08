#!/usr/bin/env python3
"""Create and lock five date-grouped Atanas outer folds without training.

The only label-derived quantity used for stratification is the number of clean
identities per worm. Neural activity, coordinates, checkpoints, predictions,
and historical validation/test metrics are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import re
import stat
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
INVALID_IDENTITIES = {
    "",
    "-1",
    "none",
    "nan",
    "null",
    "unknown",
    "unk",
    "unlabeled",
    "unlabelled",
    "?",
}
OUTER_FOLDS = 5
INNER_VAL_WORMS = 6
GENERATOR_VERSION = "atanas_multifold_v1_2026-07-31"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(
            "Data/Atanas_SF_unified_000776/date_disjoint_v1/full/"
            "split_manifest.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("workflows/atanas_multifold_v1"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing locked manifest instead of generating it.",
    )
    return parser.parse_args()


def canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_identity(value: Any) -> str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return None if text.lower() in INVALID_IDENTITIES else text


def load_worms(source_manifest: Path) -> list[dict[str, Any]]:
    with source_manifest.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 38:
        raise AssertionError(f"Expected 38 Atanas worms, got {len(source_rows)}")

    worms: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in source_rows:
        worm_id = str(row["recording"])
        if worm_id in seen:
            raise AssertionError(f"Duplicate worm ID: {worm_id}")
        seen.add(worm_id)
        match = DATE_RE.search(worm_id)
        if match is None:
            raise ValueError(f"No date in worm ID: {worm_id}")
        date = match.group(1)
        path = Path(row["source_path"]).resolve()
        with np.load(path, allow_pickle=False) as data:
            if "clean_mask" not in data.files or "cell_id" not in data.files:
                raise KeyError(f"{path}: clean_mask/cell_id missing")
            mask = np.asarray(data["clean_mask"], dtype=bool).reshape(-1)
            raw = np.asarray(data["cell_id"]).reshape(-1)
            if len(mask) != len(raw):
                raise ValueError(f"{path}: clean_mask/cell_id length mismatch")
            identities = [
                normalized_identity(value)
                for value in raw[mask]
            ]
        clean = sorted(identity for identity in identities if identity is not None)
        if len(clean) != len(set(clean)):
            raise AssertionError(f"{worm_id}: duplicate clean identity")
        worms.append(
            {
                "worm_id": worm_id,
                "date_group": date,
                "source_path": str(path),
                "clean_identity_count": len(clean),
                "_clean_identities": frozenset(clean),
            }
        )
    return sorted(worms, key=lambda row: row["worm_id"])


def date_groups(worms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for worm in worms:
        grouped[worm["date_group"]].append(worm)
    return [
        {
            "date": date,
            "worms": tuple(sorted(grouped[date], key=lambda row: row["worm_id"])),
            "worm_count": len(grouped[date]),
            "clean_identity_total": sum(
                int(row["clean_identity_count"]) for row in grouped[date]
            ),
        }
        for date in sorted(grouped)
    ]


def choose_outer_partition(
    groups: list[dict[str, Any]],
    total_clean: int,
    total_worms: int,
    seed: int,
) -> list[set[str]]:
    """MILP: intact dates, 7--8 worms/fold, 3--4 dates/fold, balanced clean means."""
    n_dates = len(groups)
    n_x = n_dates * OUTER_FOLDS
    n_dev = OUTER_FOLDS
    n_vars = n_x + n_dev

    # Absolute deviation of each fold's clean identities per worm from the
    # global mean, represented exactly as:
    # |total_worms * fold_clean - total_clean * fold_worms|.
    objective = np.zeros(n_vars, dtype=np.float64)
    objective[n_x:] = 1.0
    for date_index, group in enumerate(groups):
        for fold in range(OUTER_FOLDS):
            token = f"{seed}|{group['date']}|{fold}".encode("utf-8")
            tie = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
            objective[date_index * OUTER_FOLDS + fold] = (
                (tie / float(2**64 - 1)) * 1e-7
            )

    rows: list[tuple[dict[int, float], float, float]] = []
    # Every date goes to exactly one outer-test fold.
    for date_index in range(n_dates):
        coeff = {
            date_index * OUTER_FOLDS + fold: 1.0
            for fold in range(OUTER_FOLDS)
        }
        rows.append((coeff, 1.0, 1.0))
    # Fold worm counts and date counts.
    for fold in range(OUTER_FOLDS):
        rows.append(
            (
                {
                    date_index * OUTER_FOLDS + fold: float(group["worm_count"])
                    for date_index, group in enumerate(groups)
                },
                7.0,
                8.0,
            )
        )
        rows.append(
            (
                {
                    date_index * OUTER_FOLDS + fold: 1.0
                    for date_index in range(n_dates)
                },
                3.0,
                4.0,
            )
        )
        deviation = {
            date_index * OUTER_FOLDS + fold: float(
                total_worms * int(group["clean_identity_total"])
                - total_clean * int(group["worm_count"])
            )
            for date_index, group in enumerate(groups)
        }
        deviation[n_x + fold] = -1.0
        rows.append((deviation, -np.inf, 0.0))
        rows.append(
            (
                {
                    key: (-value if key != n_x + fold else value)
                    for key, value in deviation.items()
                },
                -np.inf,
                0.0,
            )
        )

    matrix = lil_matrix((len(rows), n_vars), dtype=np.float64)
    lower = np.empty(len(rows), dtype=np.float64)
    upper = np.empty(len(rows), dtype=np.float64)
    for row_index, (coefficients, lb, ub) in enumerate(rows):
        for column, value in coefficients.items():
            matrix[row_index, column] = value
        lower[row_index] = lb
        upper[row_index] = ub

    bounds = Bounds(
        np.zeros(n_vars, dtype=np.float64),
        np.concatenate(
            [
                np.ones(n_x, dtype=np.float64),
                np.full(n_dev, np.inf, dtype=np.float64),
            ]
        ),
    )
    integrality = np.concatenate(
        [np.ones(n_x, dtype=np.int8), np.zeros(n_dev, dtype=np.int8)]
    )
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=bounds,
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"disp": False, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Outer-fold MILP failed: {result.message}")

    raw_folds: list[set[str]] = [set() for _ in range(OUTER_FOLDS)]
    for date_index, group in enumerate(groups):
        assignments = [
            fold
            for fold in range(OUTER_FOLDS)
            if result.x[date_index * OUTER_FOLDS + fold] > 0.5
        ]
        if len(assignments) != 1:
            raise AssertionError((group["date"], assignments))
        raw_folds[assignments[0]].add(str(group["date"]))

    # Remove arbitrary MILP fold-label symmetry.
    return sorted(raw_folds, key=lambda dates: tuple(sorted(dates)))


def pair_statistics(worms: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = sorted(worms, key=lambda row: row["worm_id"])
    shared_counts = [
        len(a["_clean_identities"] & b["_clean_identities"])
        for a, b in itertools.combinations(records, 2)
    ]
    usable = [count for count in shared_counts if count > 0]
    total = len(shared_counts)
    clean_counts = np.asarray(
        [int(row["clean_identity_count"]) for row in records],
        dtype=np.int64,
    )
    union = set().union(*(row["_clean_identities"] for row in records))
    return {
        "worm_count": len(records),
        "date_group_count": len({row["date_group"] for row in records}),
        "clean_identity_per_worm": {
            "mean": float(clean_counts.mean()),
            "min": int(clean_counts.min()),
            "max": int(clean_counts.max()),
            "sum": int(clean_counts.sum()),
        },
        "unique_clean_identity_union": len(union),
        "all_unordered_worm_pairs": total,
        "all_directed_worm_pairs": 2 * total,
        "usable_unordered_worm_pairs": len(usable),
        "usable_directed_worm_pairs": 2 * len(usable),
        "usable_pair_fraction": float(len(usable) / total) if total else 0.0,
        "shared_clean_identities_per_usable_pair": {
            "mean": float(np.mean(usable)) if usable else 0.0,
            "min": int(min(usable)) if usable else 0,
            "max": int(max(usable)) if usable else 0,
        },
        "directed_scorable_queries": 2 * int(sum(usable)),
    }


def choose_inner_validation(
    remaining_dates: set[str],
    groups_by_date: dict[str, dict[str, Any]],
    worms_by_date: dict[str, list[dict[str, Any]]],
    seed: int,
    fold_number: int,
) -> set[str]:
    candidates: list[tuple[tuple[Any, ...], tuple[str, ...]]] = []
    remaining_worms = [
        worm for date in remaining_dates for worm in worms_by_date[date]
    ]
    remaining_mean = float(
        np.mean([worm["clean_identity_count"] for worm in remaining_worms])
    )
    ordered_dates = sorted(remaining_dates)
    rank = {
        date: index / max(1, len(ordered_dates) - 1)
        for index, date in enumerate(ordered_dates)
    }
    remaining_rank_mean = float(
        np.mean(
            [
                rank[worm["date_group"]]
                for worm in remaining_worms
            ]
        )
    )
    for count in range(1, min(5, len(ordered_dates)) + 1):
        for dates in itertools.combinations(ordered_dates, count):
            worms = [worm for date in dates for worm in worms_by_date[date]]
            if len(worms) != INNER_VAL_WORMS:
                continue
            stats = pair_statistics(worms)
            clean_mean = stats["clean_identity_per_worm"]["mean"]
            rank_mean = float(
                np.mean([rank[worm["date_group"]] for worm in worms])
            )
            token = (
                f"{seed}|fold={fold_number}|val={','.join(dates)}"
            ).encode("utf-8")
            digest = hashlib.sha256(token).hexdigest()
            objective = (
                round(abs(clean_mean - remaining_mean), 12),
                -int(stats["usable_unordered_worm_pairs"]),
                -int(stats["directed_scorable_queries"]),
                round(abs(rank_mean - remaining_rank_mean), 12),
                abs(len(dates) - 3),
                digest,
            )
            candidates.append((objective, dates))
    if not candidates:
        raise RuntimeError(
            f"Fold {fold_number}: cannot form an intact-date validation set "
            f"with exactly {INNER_VAL_WORMS} worms"
        )
    return set(min(candidates, key=lambda item: item[0])[1])


def public_worm(worm: dict[str, Any]) -> dict[str, Any]:
    return {
        "worm_id": worm["worm_id"],
        "date_group": worm["date_group"],
        "clean_identity_count": worm["clean_identity_count"],
        "source_path": worm["source_path"],
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_manifest(
    worms: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    groups = date_groups(worms)
    groups_by_date = {group["date"]: group for group in groups}
    worms_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_id = {worm["worm_id"]: worm for worm in worms}
    for worm in worms:
        worms_by_date[worm["date_group"]].append(worm)

    outer_dates = choose_outer_partition(
        groups,
        total_clean=sum(worm["clean_identity_count"] for worm in worms),
        total_worms=len(worms),
        seed=seed,
    )
    all_dates = set(groups_by_date)
    folds: list[dict[str, Any]] = []
    outer_test_occurrences: dict[str, int] = defaultdict(int)
    for fold_index, test_dates in enumerate(outer_dates, start=1):
        remaining_dates = all_dates - test_dates
        val_dates = choose_inner_validation(
            remaining_dates,
            groups_by_date,
            worms_by_date,
            seed,
            fold_index,
        )
        train_dates = remaining_dates - val_dates
        split_dates = {
            "train": sorted(train_dates),
            "val": sorted(val_dates),
            "test": sorted(test_dates),
        }
        split_worms = {
            split: sorted(
                [
                    worm
                    for date in dates
                    for worm in worms_by_date[date]
                ],
                key=lambda row: row["worm_id"],
            )
            for split, dates in split_dates.items()
        }
        ids = {
            split: {worm["worm_id"] for worm in rows}
            for split, rows in split_worms.items()
        }
        intersections = {
            "train_val": sorted(ids["train"] & ids["val"]),
            "train_test": sorted(ids["train"] & ids["test"]),
            "val_test": sorted(ids["val"] & ids["test"]),
        }
        if any(intersections.values()):
            raise AssertionError(f"Fold {fold_index}: worm overlap")
        date_intersections = {
            "train_val": sorted(train_dates & val_dates),
            "train_test": sorted(train_dates & test_dates),
            "val_test": sorted(val_dates & test_dates),
        }
        if any(date_intersections.values()):
            raise AssertionError(f"Fold {fold_index}: date overlap")
        for worm_id in ids["test"]:
            outer_test_occurrences[worm_id] += 1
        folds.append(
            {
                "fold": fold_index,
                "dates": split_dates,
                "worm_ids": {
                    split: sorted(values) for split, values in ids.items()
                },
                "statistics": {
                    split: pair_statistics(rows)
                    for split, rows in split_worms.items()
                },
                "worm_intersections": intersections,
                "date_intersections": date_intersections,
            }
        )

    if set(outer_test_occurrences) != set(by_id):
        raise AssertionError("Not every worm appears in outer test")
    if any(count != 1 for count in outer_test_occurrences.values()):
        raise AssertionError("A worm appears in outer test more than once")
    test_sets = [set(fold["worm_ids"]["test"]) for fold in folds]
    if set().union(*test_sets) != set(by_id):
        raise AssertionError("Outer-test union does not cover all worms")
    for left, right in itertools.combinations(test_sets, 2):
        if left & right:
            raise AssertionError("Outer-test folds overlap")

    outer_stats = [fold["statistics"]["test"] for fold in folds]
    return {
        "format": "atanas_locked_multifold_split_manifest",
        "version": GENERATOR_VERSION,
        "locked": True,
        "seed": seed,
        "selection_basis": (
            "Worm IDs, filename-derived date groups, and per-worm clean identity "
            "counts only. No activity values, coordinates, checkpoints, model "
            "predictions, validation metrics, or historical test metrics used."
        ),
        "outer_partition_rule": (
            "Five mutually exclusive intact-date test folds. MILP constrains each "
            "fold to 7-8 worms and 3-4 date groups and minimizes absolute deviation "
            "of clean identities per worm from the dataset-wide mean."
        ),
        "inner_validation_rule": (
            "Within each outer fold, select an intact-date 6-worm validation set "
            "from the non-test worms. Deterministically minimize clean-count mean "
            "imbalance, then prefer more usable pairs/scorable queries; SHA256 is "
            "the final tie-break."
        ),
        "future_use_policy": {
            "development_reads": ["outer-train", "inner-validation"],
            "outer_test_access": (
                "Exactly once after NuCLR, fDNC, FINDA, checkpoint epoch, windows, "
                "and all hyperparameters are locked for that fold."
            ),
            "outer_test_must_not_drive_model_selection": True,
            "historical_single_split_test_results": "historical only",
        },
        "dataset": {
            "worms": len(worms),
            "date_groups": len(groups),
            "clean_identity_per_worm": pair_statistics(worms)[
                "clean_identity_per_worm"
            ],
            "worms_metadata": [public_worm(worm) for worm in worms],
        },
        "folds": folds,
        "global_assertions": {
            "outer_folds": OUTER_FOLDS,
            "every_worm_outer_test_exactly_once": True,
            "outer_test_sets_pairwise_disjoint": True,
            "outer_test_union_equals_all_worms": True,
            "date_groups_never_split_within_any_fold": True,
            "train_val_test_worm_intersections_empty": True,
            "outer_test_metrics_computed": False,
            "models_trained": False,
        },
        "outer_test_balance": {
            "worm_counts": [
                stats["worm_count"] for stats in outer_stats
            ],
            "clean_identity_means": [
                stats["clean_identity_per_worm"]["mean"]
                for stats in outer_stats
            ],
            "clean_identity_ranges": [
                [
                    stats["clean_identity_per_worm"]["min"],
                    stats["clean_identity_per_worm"]["max"],
                ]
                for stats in outer_stats
            ],
            "usable_unordered_pair_counts": [
                stats["usable_unordered_worm_pairs"]
                for stats in outer_stats
            ],
            "directed_scorable_queries": [
                stats["directed_scorable_queries"]
                for stats in outer_stats
            ],
        },
    }


def file_mode_read_only(path: Path) -> None:
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def generate(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Locked output already exists: {output_root}. "
            "Use --verify; this generator intentionally has no overwrite option."
        )
    worms = load_worms(args.source_manifest.resolve())
    manifest = make_manifest(worms, args.seed)
    manifest_bytes = canonical_json(manifest)
    manifest_sha = sha256_bytes(manifest_bytes)

    output_root.mkdir(parents=True, exist_ok=False)
    manifest_path = output_root / "LOCKED_SPLIT_MANIFEST.json"
    manifest_path.write_bytes(manifest_bytes)
    write_text(
        output_root / "LOCKED_SPLIT_MANIFEST.sha256",
        f"{manifest_sha}  LOCKED_SPLIT_MANIFEST.json\n",
    )
    generated_files = [manifest_path, output_root / "LOCKED_SPLIT_MANIFEST.sha256"]
    for fold in manifest["folds"]:
        fold_dir = output_root / f"fold_{fold['fold']}"
        for split in ("train", "val", "test"):
            path = fold_dir / f"{split}_ids.txt"
            write_text(path, "\n".join(fold["worm_ids"][split]) + "\n")
            generated_files.append(path)
        summary_path = fold_dir / "fold_summary.json"
        summary_path.write_bytes(canonical_json(fold))
        generated_files.append(summary_path)

    checksums_path = output_root / "checksums.sha256"
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(output_root)}"
        for path in sorted(generated_files)
    ]
    write_text(checksums_path, "\n".join(checksum_lines) + "\n")
    generated_files.append(checksums_path)
    for path in generated_files:
        file_mode_read_only(path)
    # Keep directories traversable but not writable as an additional lock.
    for directory, _, _ in os.walk(output_root, topdown=False):
        Path(directory).chmod(
            stat.S_IRUSR
            | stat.S_IXUSR
            | stat.S_IRGRP
            | stat.S_IXGRP
            | stat.S_IROTH
            | stat.S_IXOTH
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"Locked manifest: {manifest_path}")
    print(f"Manifest SHA256: {manifest_sha}")


def verify(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    manifest_path = output_root / "LOCKED_SPLIT_MANIFEST.json"
    checksum_path = output_root / "LOCKED_SPLIT_MANIFEST.sha256"
    expected = checksum_path.read_text(encoding="utf-8").split()[0]
    observed = sha256_file(manifest_path)
    if observed != expected:
        raise AssertionError(
            f"Manifest SHA256 mismatch: expected {expected}, got {observed}"
        )
    for line in (output_root / "checksums.sha256").read_text(
        encoding="utf-8"
    ).splitlines():
        digest, relative = line.split(maxsplit=1)
        relative = relative.strip()
        path = output_root / relative
        if sha256_file(path) != digest:
            raise AssertionError(f"Checksum mismatch: {relative}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not payload["global_assertions"][
        "every_worm_outer_test_exactly_once"
    ]:
        raise AssertionError("Outer-test coverage assertion absent")
    print(f"Verified locked manifest: {manifest_path}")
    print(f"Manifest SHA256: {observed}")


def main() -> None:
    args = parse_args()
    if args.verify:
        verify(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
