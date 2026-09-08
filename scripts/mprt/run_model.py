#!/usr/bin/env python3
"""One-command training and evaluation for NeuRID/MPRT-Net."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "mprt_net_v1_1"


def run(command: list[str]) -> None:
    print("\n> " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def fold_roots(dataset_root: Path) -> list[tuple[int, Path]]:
    candidates = [(fold, dataset_root / f"fold_{fold}") for fold in range(5)]
    present = [path.is_dir() for _, path in candidates]
    if all(present):
        return candidates
    if any(present):
        missing = [str(path) for (_, path), exists in zip(candidates, present) if not exists]
        raise FileNotFoundError("incomplete CV5 directory; missing: " + ", ".join(missing))
    return [(0, dataset_root)]


def validate_split_root(path: Path) -> None:
    for split in ("train", "val", "test"):
        split_root = path / split
        if not split_root.is_dir() or not any(split_root.glob("*.npz")):
            raise FileNotFoundError(f"no NPZ files found in {split_root}")


def train_and_build(data: Path, output: Path, args: argparse.Namespace) -> None:
    pairwise = output / "pairwise_full"
    run([sys.executable, "-m", "mprt_net.self_check", "--dataset-root", str(data),
         "--split", "train", "--device", args.device])
    run([sys.executable, "-m", "mprt_net.train", "--dataset-root", str(data),
         "--output-dir", str(pairwise), "--variant", "full", "--seed", "42",
         "--epochs", str(args.epochs), "--pairs-per-epoch", str(args.pairs_per_epoch),
         "--activity-length", str(args.activity_length), "--device", args.device])
    run([sys.executable, "-m", "mprt_net.build_anchored_atlas", "--dataset-root", str(data),
         "--split", "train", "--checkpoint", str(pairwise / "best.pt"),
         "--output", str(output / "anchored_gated_b030.pt"),
         "--pure-output", str(output / "anchored_pure.pt"),
         "--activity-length", str(args.activity_length), "--blend-weight", "0.30",
         "--gate-temperature", "0.05", "--device", args.device])


def evaluate(fold: int, data: Path, output: Path, args: argparse.Namespace) -> None:
    run([sys.executable, "-m", "scripts.mprt.evaluate_mprt_static_atlas",
         "--package-root", str(PACKAGE_ROOT), "--dataset-root", str(data),
         "--split", "test", "--checkpoint", str(output / "anchored_pure.pt"),
         "--activity-length", str(args.activity_length), "--device", args.device,
         "--dataset", args.dataset_name, "--fold", str(fold), "--seed", "42",
         "--variant", "full", "--output", str(output / "test_metrics.json"),
         "--query-output", str(output / "test_queries.csv")])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dataset-name", default="my_dataset")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pairs-per-epoch", type=int, default=128)
    parser.add_argument("--activity-length", type=int, default=512)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    folds = fold_roots(dataset_root)
    for _, data in folds:
        validate_split_root(data)

    print(f"Detected {'formal CV5' if len(folds) == 5 else 'single dataset'}; seed=42")
    for fold, data in folds:
        train_and_build(data, output_root / f"fold{fold}", args)
    # Test is delayed until every fold has completed training and atlas construction.
    for fold, data in folds:
        evaluate(fold, data, output_root / f"fold{fold}", args)

    if len(folds) == 5:
        run([sys.executable, "-m", "scripts.mprt.summarize_cv5_seed42",
             "--run-root", str(output_root), "--output", str(output_root / "summary.json"),
             "--markdown-output", str(output_root / "SUMMARY.md")])
    print(f"\nFinished. Results: {output_root}")


if __name__ == "__main__":
    main()
