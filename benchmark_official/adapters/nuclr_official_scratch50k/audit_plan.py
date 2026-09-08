#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import run_one as r


def main() -> None:
    repo = r.git_info(r.OFFICIAL_REPO)
    print("Official NuCLR repo:")
    print(json.dumps(repo, indent=2))
    if repo["dirty"]:
        raise SystemExit("ERROR: official NuCLR checkout is dirty")

    print("\nOfficial calcium settings used:")
    settings = {
        "view_seconds": r.s1.OFFICIAL_VIEW_SECONDS,
        "max_view_distance_seconds": r.s1.OFFICIAL_MAX_VIEW_DISTANCE_SECONDS,
        "batch_size": r.s1.OFFICIAL_BATCH_SIZE,
        "precision": r.s1.OFFICIAL_PRECISION,
        "max_lr": r.s1.OFFICIAL_MAX_LR,
        "unit_dropout_min_fraction": r.s1.OFFICIAL_UNIT_DROPOUT_MIN_FRACTION,
        "loss_tau": r.s1.OFFICIAL_LOSS_TAU,
        "loss_dcl": r.s1.OFFICIAL_LOSS_DCL,
        "loss_projector": r.s1.OFFICIAL_LOSS_PROJECTOR,
        "loss_full_denom": r.s1.OFFICIAL_LOSS_FULL_DENOM,
        "target_train_steps": r.TARGET_TRAIN_STEPS,
    }
    print(json.dumps(settings, indent=2))

    print("\nFold plans (seed42; split membership is checked from the locked benchmark):")
    print("dataset fold train_worms val_worms sampler_samples steps_per_epoch equivalent_epochs fs_values")
    for ds in ("atanas", "rld"):
        for fold in range(1, 6):
            _, _, _, train_common, val_common = r.locked_train_val_context(ds, fold, 42)
            train_records = r.make_nuclr_records(train_common, "train")
            adapter = r.s1.EYOfficialDataAdapter(train_records, model_num_samples=128, seed=42)
            sampler = adapter.make_first_view_sampler(seed=42)
            sampler.set_epoch(0)
            samples = len(list(iter(sampler)))
            steps = samples // int(r.s1.OFFICIAL_BATCH_SIZE)
            if steps < 1:
                raise RuntimeError(f"{ds} fold{fold}: no full batches")
            eq_epochs = r.TARGET_TRAIN_STEPS / steps
            fs_values = sorted({round(float(x.source_fs), 8) for x in train_records})
            print(
                f"{ds:6s} {fold:>4d} {len(train_common):>11d} {len(val_common):>9d} "
                f"{samples:>15d} {steps:>15d} {eq_epochs:>17.1f} {fs_values}"
            )

    print("\nAUDIT PASSED")
    print("Outer test files were not opened by this audit.")


if __name__ == "__main__":
    main()
