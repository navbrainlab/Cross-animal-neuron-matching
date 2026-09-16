"""Export canonical Atanas/Kato per-animal Top-1 counts.

The shared cohort is the fold-specific subset of canonical queries whose GT
identity is present in the frozen single-medoid candidate list.  The exact same
query-key mask is applied to every exported method, including atlas methods.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT = REPO / "results/main_benchmark_audit"
DEFAULT_OUTPUT = REPO / "runs/atanas_kato_per_animal"
DATASET_DISPLAY = {"atanas": "Atanas", "rld": "Kato"}
NEURID_SINGLE = "NeurID (single medoid)"
NEURID_POPULATION = "NeurID (population atlas)"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _key(row: dict[str, str]) -> tuple[str, int, str, int]:
    return row["dataset"], int(row["fold"]), row["query_uid"], int(row["node_index"])


def _binary_top1(row: dict[str, str]) -> int:
    value = float(row["top1_correct"])
    if value not in (0.0, 1.0):
        raise ValueError(f"Non-binary Top-1 credit for {row['method']}: {value}")
    return int(value)


def export(audit: Path, output: Path) -> None:
    predictions = _read_csv(audit / "predictions/canonical_complete_predictions.csv")
    references = _read_csv(audit / "main_table/single_medoid_reference_audit.csv")

    datasets = set(DATASET_DISPLAY)
    predictions = [row for row in predictions if row["dataset"] in datasets]
    methods = sorted({row["method"] for row in predictions})
    if NEURID_SINGLE not in methods or NEURID_POPULATION not in methods:
        raise ValueError("Both NeurID reference protocols must be present")

    by_method: dict[tuple[str, int, str], dict[tuple[str, int, str, int], dict[str, str]]] = {}
    canonical: dict[tuple[str, int], set[tuple[str, int, str, int]]] = {}
    shared: dict[tuple[str, int], set[tuple[str, int, str, int]]] = {}
    for row in predictions:
        dataset, fold, _, _ = _key(row)
        key = _key(row)
        by_method.setdefault((dataset, fold, row["method"]), {})[key] = row
        if row["method"] == NEURID_POPULATION:
            canonical.setdefault((dataset, fold), set()).add(key)
        if row["method"] == NEURID_SINGLE and row["reference_covered"] == "1":
            shared.setdefault((dataset, fold), set()).add(key)

    # Fail closed unless every method has exactly one row for every canonical
    # query.  This prevents a method-specific shared cohort from slipping in.
    for (dataset, fold), canonical_keys in canonical.items():
        if not shared.get((dataset, fold), set()) <= canonical_keys:
            raise ValueError(f"Shared keys are not canonical for {dataset} fold {fold}")
        for method in methods:
            keys = set(by_method.get((dataset, fold, method), {}))
            if keys != canonical_keys:
                raise ValueError(
                    f"Canonical mismatch for {dataset} fold {fold} {method}: "
                    f"expected {len(canonical_keys)}, got {len(keys)}"
                )

    result_rows: list[dict[str, object]] = []
    for dataset, fold in sorted(canonical):
        canonical_keys = canonical[(dataset, fold)]
        shared_keys = shared[(dataset, fold)]
        animals = sorted({key[2] for key in canonical_keys})
        for animal_id in animals:
            animal_keys = {key for key in canonical_keys if key[2] == animal_id}
            animal_shared = animal_keys & shared_keys
            for method in methods:
                rows = by_method[(dataset, fold, method)]
                result_rows.append(
                    {
                        "dataset": DATASET_DISPLAY[dataset],
                        "fold": fold,
                        "animal_id": animal_id,
                        "method": method,
                        "n_eval": len(animal_keys),
                        "n_correct": sum(_binary_top1(rows[key]) for key in animal_keys),
                        "n_shared": len(animal_shared),
                        "n_correct_shared": sum(
                            _binary_top1(rows[key]) for key in animal_shared
                        ),
                    }
                )

    reference_rows: list[dict[str, object]] = []
    for row in references:
        if row["dataset"] not in datasets or row["method"] != NEURID_SINGLE:
            continue
        reference_rows.append(
            {
                "dataset": DATASET_DISPLAY[row["dataset"]],
                "fold": int(row["fold"]),
                "medoid_animal_id": row["reference_uid"],
            }
        )
    reference_rows.sort(key=lambda row: (str(row["dataset"]), int(row["fold"])))
    if len(reference_rows) != 10:
        raise ValueError(f"Expected 10 dataset/fold medoid rows, got {len(reference_rows)}")

    _write_csv(
        output / "atanas_kato_per_animal_method_counts.csv",
        [
            "dataset",
            "fold",
            "animal_id",
            "method",
            "n_eval",
            "n_correct",
            "n_shared",
            "n_correct_shared",
        ],
        result_rows,
    )
    _write_csv(
        output / "atanas_kato_medoid_references.csv",
        ["dataset", "fold", "medoid_animal_id"],
        reference_rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    export(args.audit.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
