#!/usr/bin/env python3
"""Strict fold-pure Candidate A/B study using audited fold-local NuCLR pretraining.

The existing Table-1 workflow already trained one T2/ST2 NuCLR for every
dataset/fold/seed using outer-train worms only (30 epochs; no identities; test
dataloader references=0). This runner structurally retains the first T and ST
blocks to form Compact NuCLR T1/ST1, fine-tunes it again on the same outer-train,
selects on inner validation, and only then opens the sealed outer test.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_fold_pure_e2e_candidates_ab as base


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "runs/fold_pure_table1init_candidates_ab_20260817"
PROTOCOL = RUN_ROOT / "FOLD_PURE_PROTOCOL.json"
MANIFESTS = {
    "atanas": ROOT / "table1_clean/workflows/atanas_multifold_v1/LOCKED_SPLIT_MANIFEST.json",
    "rld": ROOT / "table1_clean/workflows/table1_rld_fivefold_evaluable93_v1/LOCKED_SPLIT_MANIFEST.json",
}
TABLE1_RUN = ROOT / "table1_clean/workflows/table1_runs_v1"
NUCLR_BUDGET = {
    "prior_outer_train_epochs": 30,
    "compact_finetune_epochs": 2,
    "optimizer_steps_per_epoch": 8,
    "batch_size": 4,
    "patience": 2,
    "eval_num_windows": 8,
    "extract_num_windows": 16,
}

# Reuse the validated matcher/execution implementation with a fresh run root.
base.RUN_ROOT = RUN_ROOT
base.PROTOCOL = PROTOCOL
base.NUCLR_BUDGET = {
    "epochs": NUCLR_BUDGET["compact_finetune_epochs"],
    "optimizer_steps_per_epoch": NUCLR_BUDGET["optimizer_steps_per_epoch"],
    "batch_size": NUCLR_BUDGET["batch_size"],
    "patience": NUCLR_BUDGET["patience"],
    "eval_num_windows": NUCLR_BUDGET["eval_num_windows"],
    "extract_num_windows": NUCLR_BUDGET["extract_num_windows"],
}


def table1_fold_root(dataset: str, fold: int) -> Path:
    return TABLE1_RUN / dataset / f"fold_{fold}"


def initializer(dataset: str, fold: int, seed: int) -> Path:
    return RUN_ROOT / "initializers" / dataset / f"fold_{fold}" / f"seed_{seed}" / "t1st1_d256/truncated.pt"


def source_checkpoint(dataset: str, fold: int, seed: int) -> Path:
    return table1_fold_root(dataset, fold) / f"nuclr_seed_{seed}/final.pt"


def create_initializer(spec: tuple[str, int, int]) -> dict:
    dataset, fold, seed = spec
    target = initializer(dataset, fold, seed)
    if target.is_file():
        return {"spec": spec, "status": "reused"}
    source = source_checkpoint(dataset, fold, seed)
    output_root = target.parents[1]
    log = output_root / "truncate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(base.PYTHON), str(ROOT / "engines/create_truncated_nuclr_depth_checkpoints.py"),
        "--source", str(source), "--training-script", str(ROOT / "hybrid/activity.py"),
        "--output-root", str(output_root), "--variants", "t1st1_d256:1:1",
    ]
    code, seconds = base.execute(command, log, 1)
    return {"spec": spec, "status": "completed" if code == 0 and target.is_file() else "failed",
            "returncode": code, "seconds": seconds}


def prepare() -> None:
    specs = [(d, f, s) for d in ("atanas", "rld") for f in range(1, 6) for s in base.SEEDS]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(create_initializer, spec) for spec in specs]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            print(json.dumps({"initializer": f"{index}/30", **result}, ensure_ascii=False), flush=True)
            if result["status"] == "failed":
                raise RuntimeError(result)

    protocol = {
        "protocol": "fold-pure end-to-end grouped 5-fold CV x 3 seeds",
        "candidates": {
            "A": "HyQuRP + Compact NuCLR + concat + Population Transformer; no Stage 2; no Sinkhorn",
            "B": "HyQuRP + Compact NuCLR + Stage 2 + Population Transformer; no Sinkhorn",
        },
        "nuclr": (
            "Per dataset/fold/seed: 30-epoch outer-train-only T2/ST2 source; "
            "weight-preserving T1/ST1 truncation; same-fold compact fine-tuning; validation selection"
        ),
        "nuclr_budget": NUCLR_BUDGET,
        "matcher_budget": base.MATCHER_BUDGET,
        "seeds_cover_entire_pipeline": list(base.SEEDS),
        "datasets": {},
        "initializers": [],
    }
    for dataset in ("atanas", "rld"):
        locked = json.loads(MANIFESTS[dataset].read_text())
        raw = base.source_map(dataset)
        folds = []
        for fold_info in locked["folds"]:
            fold = int(fold_info["fold"])
            ids = {key: list(fold_info["worm_ids"][key]) for key in ("train", "val", "test")}
            if set(ids["train"]) & set(ids["val"]) or set(ids["train"]) & set(ids["test"]) or set(ids["val"]) & set(ids["test"]):
                raise RuntimeError(f"{dataset} fold {fold}: overlap")
            fold_root = RUN_ROOT / "folds" / dataset / f"fold_{fold}"
            for split in ("train", "val"):
                for worm_id in ids[split]:
                    base.ensure_link(raw[worm_id], fold_root / "data_train_val_only" / split / f"{worm_id}.npz")
            for worm_id in ids["test"]:
                base.ensure_link(raw[worm_id], fold_root / "sealed_test/test" / f"{worm_id}.npz")
            (fold_root / "test_ids.txt").write_text("\n".join(ids["test"]) + "\n", encoding="utf-8")
            folds.append({
                "fold": fold, "train_ids": ids["train"], "val_ids": ids["val"], "test_ids": ids["test"],
                "counts": {key: len(value) for key, value in ids.items()},
                "date_groups": fold_info.get("dates"),
                "intersections": {"train_val": [], "train_test": [], "val_test": []},
            })
            for seed in base.SEEDS:
                summary_path = table1_fold_root(dataset, fold) / f"nuclr_seed_{seed}/summary.json"
                source_summary = json.loads(summary_path.read_text())
                source = source_checkpoint(dataset, fold, seed)
                if source_summary["checkpoint_sha256"] != base.sha256(source):
                    raise RuntimeError(f"Source checkpoint hash mismatch: {source}")
                if source_summary["outer_test_worms_referenced_by_dataloader"] != 0:
                    raise RuntimeError(f"Source checkpoint referenced test: {source}")
                if source_summary["identity_labels_opened_during_training"]:
                    raise RuntimeError(f"Source checkpoint opened identities: {source}")
                if source_summary["train_worms"] != len(ids["train"]):
                    raise RuntimeError(f"Source train count mismatch: {source}")
                compact = initializer(dataset, fold, seed)
                protocol["initializers"].append({
                    "dataset": dataset, "fold": fold, "seed": seed,
                    "source_checkpoint": str(source.resolve()), "source_sha256": base.sha256(source),
                    "source_train_epochs": source_summary["published_num_epochs"],
                    "source_test_dataloader_references": 0, "source_identity_labels_opened": False,
                    "compact_checkpoint": str(compact.resolve()), "compact_sha256": base.sha256(compact),
                    "truncation": "all surviving T1/ST1 tensors inherited",
                })
        protocol["datasets"][dataset] = {
            "locked_manifest": str(MANIFESTS[dataset].resolve()),
            "locked_manifest_sha256": base.sha256(MANIFESTS[dataset]),
            "num_evaluable_worms": len(set(sum((f["train_ids"] + f["val_ids"] + f["test_ids"] for f in folds), []))),
            "folds": folds,
        }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "FOLD_PURE_PROTOCOL.sha256").write_text(f"{base.sha256(PROTOCOL)}  FOLD_PURE_PROTOCOL.json\n")
    print(json.dumps({"prepared": str(RUN_ROOT), "initializers": 30, "nuclr_jobs": 30, "matcher_jobs": 60}, ensure_ascii=False))


def nuclr_command(dataset: str, fold: int, seed: int, threads: int) -> tuple[list[str], Path]:
    fold_root = RUN_ROOT / "folds" / dataset / f"fold_{fold}"
    save = fold_root / "pipeline" / f"seed_{seed}/nuclr_t1st1"
    counts = json.loads(PROTOCOL.read_text())["datasets"][dataset]["folds"][fold - 1]["counts"]
    init = initializer(dataset, fold, seed)
    provenance = (
        f"{dataset} outer fold {fold}, seed {seed}: Table-1 NuCLR trained 30 epochs on "
        "this outer-train split only; test dataloader references=0; then weight-preserving T1/ST1 truncation"
    )
    command = [
        str(base.PYTHON), "-u", str(ROOT / "hybrid/activity.py"),
        "--train-root", str(fold_root / "data_train_val_only/train"),
        "--val-root", str(fold_root / "data_train_val_only/val"),
        "--test-worm-ids-file", str(fold_root / "test_ids.txt"),
        "--expected-train-worms", str(counts["train"]), "--expected-val-worms", str(counts["val"]),
        "--expected-test-worms", str(counts["test"]), "--split-manifest", str(PROTOCOL),
        "--split-manifest-sha256", base.sha256(PROTOCOL), "--save-dir", str(save),
        "--model-variant", "raw_nuclr", "--init-backbone-checkpoint", str(init),
        "--external-init-checkpoint-sha256", base.sha256(init), "--external-init-provenance", provenance,
        "--audited-init-scope", "outer_train_only", "--activity-key", "activity_raw", "--label-key", "cell_id",
        "--supervision-mask-key", "clean_mask", "--same-worm-neurons", "all", "--source-fs", "4.0",
        "--epochs", str(NUCLR_BUDGET["compact_finetune_epochs"]), "--batch-size", str(NUCLR_BUDGET["batch_size"]),
        "--same-views-per-worm", "8", "--cross-pairs-per-epoch", "0", "--cross-views-per-pair", "2",
        "--optimizer-steps-per-epoch", str(NUCLR_BUDGET["optimizer_steps_per_epoch"]),
        "--cross-window-mode", "independent", "--window-seconds", "30",
        "--eval-num-windows", str(NUCLR_BUDGET["eval_num_windows"]), "--feature-num-windows", "8",
        "--feature-resample-points", "256", "--unit-dropout", "official", "--minimum-positive-matches", "8",
        "--lambda-same", "1", "--lambda-cross", "0", "--temperature", "0.2", "--projector-dim", "128",
        "--full-denom", "--nuclr-patch-size", "4", "--nuclr-dim", "256", "--nuclr-heads", "4",
        "--nuclr-dim-head", "64", "--nuclr-temporal-layers", "1", "--nuclr-spatiotemporal-layers", "1",
        "--nuclr-attention-dropout", "0", "--nuclr-linear-dropout", "0.2", "--nuclr-rot-ratio", "0.5",
        "--backbone-lr", "1e-5", "--branch-lr", "1e-4", "--projector-lr", "1.25e-4",
        "--weight-decay", "0.01", "--warmup-steps", "10", "--min-lr-ratio", "0.1", "--grad-clip", "1",
        "--feature-dropout", "0.1", "--val-every-epochs", "1", "--patience", str(NUCLR_BUDGET["patience"]),
        "--save-every-epochs", "10", "--selection-space", "encoder", "--selection-metric", "top1",
        "--seed", str(seed), "--device", "cpu",
    ]
    return command, save


def run_nuclr(max_workers: int, threads: int) -> None:
    specs = [(d, f, s) for d in ("atanas", "rld") for f in range(1, 6) for s in base.SEEDS]
    def worker(spec: tuple) -> dict:
        command, save = nuclr_command(*spec, threads)
        if (save / "summary.json").is_file() and (save / "best.pt").is_file():
            return {"spec": spec, "status": "reused", "seconds": 0.0}
        code, seconds = base.execute(command, save / "train.log", threads)
        ok = code == 0 and (save / "summary.json").is_file() and (save / "best.pt").is_file()
        return {"spec": spec, "status": "completed" if ok else "failed", "returncode": code, "seconds": seconds}
    base.run_parallel("nuclr", specs, worker, max_workers)


def extract_commands(dataset: str, fold: int, seed: int) -> tuple[list[list[str]], Path]:
    fold_root = RUN_ROOT / "folds" / dataset / f"fold_{fold}"
    pipeline = fold_root / "pipeline" / f"seed_{seed}"
    checkpoint, output = pipeline / "nuclr_t1st1/best.pt", pipeline / "embeddings"
    common = [
        str(base.PYTHON), "-u", str(ROOT / "engines/extract_rld_nuclr_capacity_embeddings_v3.py"),
        "--checkpoint", str(checkpoint), "--training-script", str(ROOT / "hybrid/activity.py"),
        "--output-root", str(output), "--activity-key", "activity_raw", "--label-key", "cell_id",
        "--supervision-mask-key", "clean_mask", "--num-windows", str(NUCLR_BUDGET["extract_num_windows"]),
        "--window-seconds", "30", "--device", "cpu", "--seed", str(seed),
    ]
    return [
        common + ["--data-root", str(fold_root / "data_train_val_only"), "--splits", "train,val"],
        common + ["--data-root", str(fold_root / "sealed_test"), "--splits", "test"],
    ], output


def run_extract(max_workers: int, threads: int) -> None:
    specs = [(d, f, s) for d in ("atanas", "rld") for f in range(1, 6) for s in base.SEEDS]
    def worker(spec: tuple) -> dict:
        d, f, s = spec
        commands, output = extract_commands(d, f, s)
        counts = json.loads(PROTOCOL.read_text())["datasets"][d]["folds"][f - 1]["counts"]
        if all(len(list((output / split).glob("*.npz"))) == counts[split] for split in ("train", "val", "test")):
            return {"spec": spec, "status": "reused", "seconds": 0.0}
        total = 0.0
        for index, command in enumerate(commands):
            code, seconds = base.execute(command, output / f"extract_{index}.log", threads)
            total += seconds
            if code:
                return {"spec": spec, "status": "failed", "returncode": code, "seconds": total}
        return {"spec": spec, "status": "completed", "seconds": total}
    base.run_parallel("extract", specs, worker, max_workers)


def run_matchers(max_workers: int, threads: int) -> None:
    base.run_matchers(max_workers, threads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "nuclr", "extract", "matchers", "all"))
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--threads-per-job", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(); return
    if not PROTOCOL.is_file():
        prepare()
    if args.action in ("nuclr", "all"):
        run_nuclr(args.max_workers, args.threads_per_job)
    if args.action in ("extract", "all"):
        run_extract(args.max_workers, args.threads_per_job)
    if args.action in ("matchers", "all"):
        run_matchers(max(args.max_workers, 16), max(1, args.threads_per_job // 2))


if __name__ == "__main__":
    main()
