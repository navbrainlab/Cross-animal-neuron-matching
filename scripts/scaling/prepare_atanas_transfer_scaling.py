#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import os
import shutil
import numpy as np


ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")

SRC_ROOT = (
    ROOT /
    "Data/Atanas_SF_unified_000776/cv5_grouped_v1"
)

OUT_ROOT = (
    ROOT /
    "Data/Atanas_SF_unified_000776/"
    "transfer_scaling_cv5_v1"
)

KS = [4, 8, 12, 16]
SUBSET_SEED = 20260826


def stable_seed(fold):
    x = f"atanas-transfer-scaling-fold{fold}-{SUBSET_SEED}"
    h = hashlib.sha256(x.encode()).hexdigest()
    return int(h[:8], 16)


def link_all(src, dst):
    dst.mkdir(parents=True, exist_ok=True)

    for p in sorted(src.glob("*.npz")):
        q = dst / p.name

        if q.exists() or q.is_symlink():
            q.unlink()

        os.symlink(p.resolve(), q)


OUT_ROOT.mkdir(parents=True, exist_ok=True)

manifest = {
    "subset_seed": SUBSET_SEED,
    "nested": True,
    "folds": {}
}

for fold in range(5):

    src_fold = SRC_ROOT / f"fold_{fold}"

    train_dir = src_fold / "train"
    val_dir = src_fold / "val"
    test_dir = src_fold / "test"

    train_files = sorted(train_dir.glob("*.npz"))

    if not train_files:
        raise RuntimeError(
            f"No train NPZ files: {train_dir}"
        )

    rng = np.random.default_rng(stable_seed(fold))

    order = np.arange(len(train_files))
    rng.shuffle(order)

    ordered = [train_files[i] for i in order]

    print("=" * 90)
    print(
        f"fold{fold}: "
        f"{len(train_files)} total training worms"
    )

    fold_manifest = {
        "train_total": len(train_files),
        "order": [p.name for p in ordered],
        "subsets": {}
    }

    sizes = KS + [len(train_files)]

    for k in sizes:

        if k > len(train_files):
            continue

        tag = "all" if k == len(train_files) else str(k)

        out = (
            OUT_ROOT /
            f"fold_{fold}" /
            f"k_{tag}"
        )

        if out.exists():
            shutil.rmtree(out)

        (out / "train").mkdir(
            parents=True,
            exist_ok=True
        )

        selected = ordered[:k]

        for p in selected:
            os.symlink(
                p.resolve(),
                out / "train" / p.name
            )

        # Keep EXACT original val/test.
        link_all(val_dir, out / "val")
        link_all(test_dir, out / "test")

        fold_manifest["subsets"][tag] = [
            p.name for p in selected
        ]

        print(
            f"  k={tag:>3}: "
            f"train={len(selected):2d} "
            f"val={len(list(val_dir.glob('*.npz'))):2d} "
            f"test={len(list(test_dir.glob('*.npz'))):2d}"
        )

    manifest["folds"][str(fold)] = fold_manifest


manifest_path = OUT_ROOT / "manifest.json"

manifest_path.write_text(
    json.dumps(
        manifest,
        indent=2
    )
)

print()
print("saved:", manifest_path)
print("DONE")
