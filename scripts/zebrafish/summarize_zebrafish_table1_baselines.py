#!/usr/bin/env python3
"""Build compact, replayable archives for Table-1-core zebrafish baselines."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np


EXPECTED = {1: 3536, 2: 3954, 3: 768, 4: 1820, 5: 1464, 6: 938, 7: 2492, 8: 2184}


def copy_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def archive(
    run_root: Path,
    output_root: Path,
    prediction_root: Path | None = None,
    shared_reference_manifest: Path | None = None,
) -> None:
    aggregate = json.loads((run_root / "aggregate.json").read_text(encoding="utf-8"))
    method = aggregate["method"]
    display = aggregate["display_name"]
    output_root.mkdir(parents=True, exist_ok=True)
    copy_text(run_root / "aggregate.json", output_root / "aggregate.json")
    copy_text(run_root / "data_audit.json", output_root / "DATA_AUDIT.json")

    fold_rows = []
    shared_rows = []
    total = 0
    for fold in range(1, 9):
        lock = run_root / "locks" / f"fold_{fold}.json"
        source_dir = run_root / "test" / f"fold_{fold}"
        destination_dir = output_root / "test" / f"fold_{fold}"
        copy_text(lock, output_root / "locks" / lock.name)
        for name in ("metrics.json", "pair_manifest.csv", "pair_diagnostics.json"):
            copy_text(source_dir / name, destination_dir / name)
        with (source_dir / "pair_manifest.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                shared_rows.append({"fold": fold, **row})
        prediction_source = source_dir / "query_predictions.csv"
        prediction_destination = destination_dir / "query_predictions.csv.gz"
        prediction_destination.parent.mkdir(parents=True, exist_ok=True)
        with prediction_source.open("rb") as source, gzip.open(prediction_destination, "wb") as destination:
            shutil.copyfileobj(source, destination)
        if prediction_root is not None:
            canonical = prediction_root / method / f"fold_{fold}.csv.gz"
            canonical.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(prediction_destination, canonical)
        with prediction_source.open(encoding="utf-8", newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        if count != EXPECTED[fold]:
            raise RuntimeError(f"fold {fold}: {count} != {EXPECTED[fold]}")
        metrics = json.loads((source_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
        fold_rows.append({"method": method, "display_name": display, "fold": fold, **metrics})
        total += count

    with (output_root / "fold_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_rows[0]))
        writer.writeheader()
        writer.writerows(fold_rows)

    replay = {}
    for metric in ("top1", "top5", "mrr", "hungarian"):
        values = np.asarray([row[metric] for row in fold_rows], dtype=np.float64)
        replay[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
        expected = aggregate["metrics"][metric]
        if not np.isclose(replay[metric]["mean"], expected["mean"], rtol=0, atol=1e-15):
            raise RuntimeError(f"{metric}: aggregate replay mismatch")
    validation = {
        "status": "PASS",
        "method": method,
        "folds": 8,
        "query_records": total,
        "expected_query_records": sum(EXPECTED.values()),
        "aggregation": "unweighted mean and sample SD across held-out fish",
        "sample_sd_ddof": 1,
        "replay": replay,
    }
    (output_root / "VALIDATION.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {display}: Table-1-core zebrafish LOFO8", "",
        "This archive contains eight validation locks, test pair manifests, per-query predictions, fold metrics and replay validation.", "",
        "Both q/r endpoints are excluded from the same-fish multi-timepoint reference atlas. Endpoint truth is opened only for metrics.", "",
        "| Metric | Mean ± sample SD across fish |", "|---|---:|",
    ]
    for metric in ("top1", "top5", "mrr", "hungarian"):
        value = replay[metric]
        if metric == "mrr":
            formatted = f"{value['mean']:.4f} ± {value['sd']:.4f}"
        else:
            formatted = f"{100*value['mean']:.2f} ± {100*value['sd']:.2f}%"
        lines.append(f"| {metric} | {formatted} |")
    (output_root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if shared_reference_manifest is not None:
        shared_reference_manifest.parent.mkdir(parents=True, exist_ok=True)
        with shared_reference_manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(shared_rows[0]))
            writer.writeheader()
            writer.writerows(shared_rows)
    print(json.dumps(validation, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, default=None)
    parser.add_argument("--shared-reference-manifest", type=Path, default=None)
    args = parser.parse_args()
    archive(
        args.run_root.resolve(),
        args.output_root.resolve(),
        None if args.prediction_root is None else args.prediction_root.resolve(),
        None if args.shared_reference_manifest is None else args.shared_reference_manifest.resolve(),
    )


if __name__ == "__main__":
    main()
