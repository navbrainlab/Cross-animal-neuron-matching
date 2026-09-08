#!/usr/bin/env python3
"""Freeze every validation-selected checkpoint before LOFO test evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


VARIANTS = ("full", "no_transport", "geometry_only", "activity_only")
SEEDS = (42,)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def last_history_epoch(path: Path) -> int:
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        return 0
    return int(json.loads(rows[-1])["epoch"])


def run_dir(run_root: Path, legacy: Path, fold: int, seed: int, variant: str) -> Path:
    if fold == 1 and seed == 42 and variant == "full":
        return legacy
    return run_root / f"fold_{fold}" / f"seed{seed}" / variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--legacy-fold1-full", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    legacy = args.legacy_fold1_full.resolve()
    data_root = args.data_root.resolve()
    entries = []
    problems = []

    for fold in range(1, 9):
        # Fold 1 was intentionally unlocked in the already reported pilot.
        if fold != 1 and (data_root / f"fold_{fold}" / "test").exists():
            problems.append(f"fold {fold}: test split existed before checkpoint lock")
        for seed in SEEDS:
            for variant in VARIANTS:
                directory = run_dir(run_root, legacy, fold, seed, variant)
                checkpoint = directory / "best.pt"
                history = directory / "history.jsonl"
                if not checkpoint.is_file() or not history.is_file():
                    problems.append(f"missing run files: {directory}")
                    continue
                epoch = last_history_epoch(history)
                if epoch < args.epochs:
                    problems.append(f"incomplete history epoch={epoch}: {directory}")
                    continue
                entries.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "variant": variant,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "history": str(history),
                        "last_epoch": epoch,
                        "legacy_reused": directory == legacy,
                    }
                )

    expected = 8 * len(SEEDS) * len(VARIANTS)
    if len(entries) != expected:
        problems.append(f"expected {expected} complete checkpoints, found {len(entries)}")
    if problems:
        print("PRE-TEST LOCK FAILED")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(2)

    payload = {
        "protocol": "zebrafish_MPRT_LOFO8_60m_seed42",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "all_training_complete_before_test": True,
        "folds": list(range(1, 9)),
        "seeds": list(SEEDS),
        "variants": list(VARIANTS),
        "selection_metric": "validation top1_real",
        "known_prior_test_access": (
            "fold1/full/seed42 was evaluated as the locked pilot before the "
            "LOFO8x3 extension; its checkpoint is reused unchanged"
        ),
        "checkpoints": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"PRE-TEST LOCK: PASS ({len(entries)} checkpoints)")
    print("manifest:", args.output.resolve())


if __name__ == "__main__":
    main()
