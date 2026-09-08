from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import torch

from scripts.benchmarks.run_ngmv2_atanas_fold import (
    load_worm,
    AtanasNGMv2,
    evaluate_split,
)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)

    args = p.parse_args()
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    ckpt_path = args.run_dir / "best.pt"

    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    payload = torch.load(
        ckpt_path,
        map_location=device,
    )

    if int(payload["fold"]) != args.fold:
        raise RuntimeError("Checkpoint fold mismatch")

    if int(payload["seed"]) != args.seed:
        raise RuntimeError("Checkpoint seed mismatch")

    mean = np.asarray(
        payload["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        payload["std"],
        dtype=np.float32,
    )

    feature_dim = int(
        payload["feature_dim"]
    )

    template_name = str(
        payload["template"]
    )

    fold_root = (
        args.data_root
        / f"fold_{args.fold}"
    )

    # Template must come from TRAIN only.
    template_path = (
        fold_root
        / "train"
        / template_name
    )

    if not template_path.is_file():
        matches = list(
            (fold_root / "train")
            .rglob(template_name)
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"Cannot uniquely locate train template "
                f"{template_name}: {matches}"
            )

        template_path = matches[0]

    template = load_worm(
        template_path
    )

    model = AtanasNGMv2(
        feature_dim=feature_dim
    ).to(device)

    model.load_state_dict(
        payload["model"]
    )

    model.eval()

    # ------------------------------------------------------
    # LOCK BEFORE TOUCHING TEST DIRECTORY
    # ------------------------------------------------------

    lock = {
        "status": "LOCKED_BEFORE_TEST",
        "method": "NGM-v2 official core adapted",
        "data_root": str(args.data_root.resolve()),
        "fold": args.fold,
        "seed": args.seed,
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_sha256": sha256(ckpt_path),
        "best_epoch": int(payload["epoch"]),
        "selection_metric": "val Hungarian then Top1",
        "val_top1": float(
            payload["val"]["top1"]
        ),
        "val_hungarian": float(
            payload["val"]["hungarian"]
        ),
        "template": template_name,
        "feature_dim": feature_dim,
    }

    lock_path = (
        output_dir
        / "LOCKED_BEFORE_TEST.json"
    )

    lock_path.write_text(
        json.dumps(
            lock,
            indent=2,
        ) + "\n"
    )

    print("=" * 76)
    print("LOCKED BEFORE TEST")
    print(json.dumps(lock, indent=2))
    print("=" * 76)

    # ------------------------------------------------------
    # TEST DIRECTORY FIRST ACCESSED HERE
    # ------------------------------------------------------

    test_paths = sorted(
        (fold_root / "test")
        .glob("*.npz")
    )

    if not test_paths:
        raise RuntimeError(
            "No test files found"
        )

    test = [
        load_worm(path)
        for path in test_paths
    ]

    metrics = evaluate_split(
        model,
        test,
        template,
        mean,
        std,
        device,
    )

    metrics.update(
        {
            "data_root": str(args.data_root.resolve()),
            "fold": args.fold,
            "seed": args.seed,
            "split_counts": {
                split: len(list((fold_root / split).glob("*.npz")))
                for split in ("train", "val", "test")
            },
            "checkpoint_sha256": lock["checkpoint_sha256"],
        }
    )

    output = (
        output_dir
        / "test_metrics.json"
    )

    output.write_text(
        json.dumps(
            metrics,
            indent=2,
        ) + "\n"
    )

    print()
    print("=" * 76)
    print(
        f"NGM-v2 — ATANAS "
        f"FOLD{args.fold} "
        f"SEED{args.seed} "
        f"— LOCKED TEST"
    )
    print("=" * 76)

    print(
        f"Queries     : {metrics['queries']}"
    )
    print(
        f"Top-1       : {100*metrics['top1']:.2f}%"
    )
    print(
        f"Top-5       : {100*metrics['top5']:.2f}%"
    )
    print(
        f"MRR         : {metrics['mrr']:.4f}"
    )
    print(
        f"Hungarian   : {100*metrics['hungarian']:.2f}%"
    )
    print(
        f"Coverage    : {100*metrics['coverage']:.2f}%"
    )

    print("=" * 76)
    print("metrics:", output)


if __name__ == "__main__":
    main()
