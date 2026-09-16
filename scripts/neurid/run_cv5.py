#!/usr/bin/env python3
"""Train and evaluate the paper's seed-42 NeuRID protocol on worm CV folds."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "atanas": ROOT / "data" / "atanas",
    "kato_rld": ROOT / "data" / "kato_rld",
}
METRICS = ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="atanas,kato_rld")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "neurid_cv5_seed42")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pairs-per-epoch", type=int, default=128)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run(command: list[str], environment: dict[str, str]) -> None:
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def aggregate(root: Path, datasets: list[str], folds: list[int]) -> None:
    summary: dict[str, object] = {
        "model": "NeuRID paper implementation",
        "seed": 42,
        "aggregation": "unweighted mean and sample SD across biological folds",
        "datasets": {},
    }
    output = summary["datasets"]
    assert isinstance(output, dict)
    for dataset in datasets:
        records = []
        for fold in folds:
            path = root / dataset / f"fold_{fold}" / "test_metrics.json"
            if path.is_file():
                records.append({"fold": fold, **json.loads(path.read_text())})
        metrics = {}
        for key in METRICS:
            values = [float(record[key]) for record in records]
            if values:
                metrics[key] = {
                    "mean": statistics.mean(values),
                    "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                }
        output[dataset] = {"folds": records, "aggregate": metrics}
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown or any(fold not in range(5) for fold in folds):
        raise ValueError(f"invalid datasets/folds: {unknown}, {folds}")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "neurid") + os.pathsep + environment.get("PYTHONPATH", "")
    python = sys.executable
    for dataset in datasets:
        for fold in folds:
            fold_root = DATASETS[dataset] / f"fold_{fold}"
            output = args.output_root / dataset / f"fold_{fold}"
            metrics = output / "test_metrics.json"
            if metrics.is_file() and not args.force:
                print(f"skip existing {metrics}")
                continue
            train = output / "train"
            atlas = output / "anchored_atlas.pt"
            gated_atlas = output / "anchored_atlas_gated.pt"
            train_command = [
                python, "-m", "mprt_net.train", "--dataset-root", str(fold_root),
                "--output-dir", str(train), "--variant", "full", "--seed", "42",
                "--epochs", str(args.epochs), "--pairs-per-epoch", str(args.pairs_per_epoch),
                "--early-stopping-patience", str(args.early_stopping_patience),
                "--device", args.device,
            ]
            if args.force:
                train_command.append("--allow-existing-output")
            run(train_command, environment)
            if atlas.exists() or gated_atlas.exists():
                if not args.force:
                    raise FileExistsError(f"atlas output exists under {output}")
                atlas.unlink(missing_ok=True)
                gated_atlas.unlink(missing_ok=True)
            run([
                python, "-m", "mprt_net.build_anchored_atlas",
                "--dataset-root", str(fold_root), "--checkpoint", str(train / "best.pt"),
                "--output", str(gated_atlas), "--pure-output", str(atlas),
                "--blend-weight", "0.30", "--device", args.device,
            ], environment)
            run([
                python, "-m", "mprt_net.evaluate", "--dataset-root", str(fold_root),
                "--split", "test", "--checkpoint", str(atlas), "--device", args.device,
                "--output", str(metrics),
            ], environment)
            aggregate(args.output_root, datasets, folds)
    aggregate(args.output_root, datasets, folds)


if __name__ == "__main__":
    main()
