#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
PKG = ROOT / "neurid"

DATA_ROOT = (
    ROOT
    / "Data/Atanas_SF_unified_000776/"
      "transfer_scaling_cv5_v1"
)

RLD_ROOT = (
    ROOT
    / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld"
)

OUT_ROOT = (
    ROOT
    / "runs/mprt_rld_pretrain_atanas_scaling_v1"
)

SEED = 42
KS = [4, 8, 12, 16]


TRAIN_ARGS = [
    "--variant", "full",
    "--seed", str(SEED),
    "--epochs", "80",
    "--pairs-per-epoch", "128",
    "--val-max-pairs", "0",
    "--activity-length", "512",
    "--min-shared", "2",
    "--synthetic-drop-probability", "0.05",
    "--focal-gamma", "2.0",
    "--learning-rate", "2e-4",
    "--weight-decay", "1e-4",
    "--gradient-clip", "1.0",
    "--hidden-dim", "96",
    "--edge-dim", "48",
    "--relation-dim", "8",
    "--activity-channels", "32",
    "--num-heads", "4",
    "--population-layers", "2",
    "--dropout", "0.10",
    "--sinkhorn-iterations", "20",
    "--transport-steps", "2",
    "--structural-weight", "1.0",
    "--cycle-weight", "0.0",
    "--atlas-weight", "0.0",
    "--atlas-blend-weight", "0.0",
    "--device", "cuda",
]


def run(cmd, cwd, gpu, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    print()
    print("=" * 110)
    print("GPU:", gpu)
    print("LOG:", log_path)
    print("CMD:", " ".join(map(str, cmd)))
    print("=" * 110)

    with open(log_path, "w") as f:
        p = subprocess.Popen(
            [str(x) for x in cmd],
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert p.stdout is not None

        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()

        rc = p.wait()

    if rc != 0:
        raise RuntimeError(
            f"Command failed ({rc}): {' '.join(map(str, cmd))}"
        )


def build_atlas(data, checkpoint, atlas_dir, gpu):
    atlas_dir.mkdir(parents=True, exist_ok=True)

    pure = atlas_dir / "anchored_pure.pt"
    gated = atlas_dir / "anchored_gated_b030.pt"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "mprt_net.build_anchored_atlas",
        "--dataset-root", data,
        "--split", "train",
        "--checkpoint", checkpoint,
        "--output", gated,
        "--pure-output", pure,
        "--activity-length", "512",
        "--blend-weight", "0.30",
        "--gate-temperature", "0.05",
        "--device", "cuda",
    ]

    run(
        cmd,
        PKG,
        gpu,
        atlas_dir / "build.log",
    )

    if not pure.is_file():
        raise FileNotFoundError(pure)

    return pure


def run_cell(fold, k, gpu):

    print()
    print("#" * 110)
    print(
        f"SCALING CELL: fold={fold} "
        f"k={k} seed={SEED} GPU={gpu}"
    )
    print("#" * 110)

    data = DATA_ROOT / f"fold_{fold}" / f"k_{k}"

    if not (data / "train").is_dir():
        raise FileNotFoundError(data / "train")
    if not (data / "val").is_dir():
        raise FileNotFoundError(data / "val")

    init = (
        RLD_ROOT
        / f"fold{fold}"
        / f"seed{SEED}"
        / "pairwise/full/best.pt"
    )

    if not init.is_file():
        raise FileNotFoundError(init)

    cell_root = (
        OUT_ROOT
        / f"fold{fold}"
        / f"seed{SEED}"
        / f"k_{k}"
    )

    scratch_train = (
        cell_root / "scratch/pairwise/full"
    )
    transfer_train = (
        cell_root / "rld_pretrained/pairwise/full"
    )

    scratch_atlas_dir = (
        cell_root / "scratch/static_atlas"
    )
    transfer_atlas_dir = (
        cell_root / "rld_pretrained/static_atlas"
    )

    comparison_dir = (
        cell_root / "val_comparison"
    )

    # ============================================================
    # A. SCRATCH
    # ============================================================

    scratch_train.mkdir(
        parents=True,
        exist_ok=True
    )

    scratch_cmd = [
        sys.executable,
        "-u",
        "-m",
        "mprt_net.train",
        "--dataset-root", data,
        "--output-dir", scratch_train,
        *TRAIN_ARGS,
    ]

    run(
        scratch_cmd,
        PKG,
        gpu,
        scratch_train / "train.log",
    )

    scratch_ckpt = scratch_train / "best.pt"

    if not scratch_ckpt.is_file():
        raise FileNotFoundError(scratch_ckpt)

    scratch_atlas = build_atlas(
        data,
        scratch_ckpt,
        scratch_atlas_dir,
        gpu,
    )

    # ============================================================
    # B. RLD PRETRAINED -> ATANAS FINE-TUNE
    # ============================================================

    transfer_train.mkdir(
        parents=True,
        exist_ok=True
    )

    transfer_cmd = [
        sys.executable,
        "-u",
        "-m",
        "mprt_net.train_transfer",
        "--dataset-root", data,
        "--output-dir", transfer_train,
        "--init-checkpoint", init,
        *TRAIN_ARGS,
    ]

    run(
        transfer_cmd,
        PKG,
        gpu,
        transfer_train / "train.log",
    )

    transfer_ckpt = transfer_train / "best.pt"

    if not transfer_ckpt.is_file():
        raise FileNotFoundError(transfer_ckpt)

    transfer_atlas = build_atlas(
        data,
        transfer_ckpt,
        transfer_atlas_dir,
        gpu,
    )

    # ============================================================
    # C. SAME VAL SET, PAIRED COMPARISON
    # ============================================================

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    comparison_json = (
        comparison_dir /
        "scratch_vs_rld_pretrained.json"
    )

    comparison_csv = (
        comparison_dir /
        "scratch_vs_rld_pretrained_queries.csv"
    )

    compare_cmd = [
        sys.executable,
        "-u",
        "-m",
        "mprt_net.compare",
        "--dataset-root", data,
        "--split", "val",
        "--checkpoint-a", scratch_atlas,
        "--checkpoint-b", transfer_atlas,
        "--name-a", "atanas_scratch",
        "--name-b", "rld_pretrained_atanas_ft",
        "--bootstrap-iterations", "10000",
        "--bootstrap-seed",
        str(20260826 + fold * 100 + k),
        "--device", "cuda",
        "--output", comparison_json,
        "--query-output", comparison_csv,
    ]

    run(
        compare_cmd,
        PKG,
        gpu,
        comparison_dir / "compare.log",
    )

    print()
    print(
        f"DONE fold={fold} "
        f"k={k} seed={SEED}"
    )


def worker(gpu, folds):
    for fold in folds:
        for k in KS:
            run_cell(fold, k, gpu)


def main():

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # GPU0: folds 0,2,4
    # GPU1: folds 1,3
    with ThreadPoolExecutor(
        max_workers=2
    ) as ex:

        f0 = ex.submit(
            worker,
            0,
            [0, 2, 4],
        )

        f1 = ex.submit(
            worker,
            1,
            [1, 3],
        )

        f0.result()
        f1.result()

    print()
    print("=" * 110)
    print(
        "ALL SEED42 TARGET-DATA SCALING "
        "CELLS COMPLETE"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
