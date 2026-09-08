#!/usr/bin/env python3
"""Fold-pure grouped 5-fold x 3-seed Compact-NuCLR Candidate A/B study.

Every dataset/fold/seed fine-tunes a T1/ST1 NuCLR using only the outer-train
split, selects it on outer-validation, then extracts representations. The shared
initializer was pretrained only on the independent EY dataset and is hash-locked.
Candidate A (concat) and Candidate B (Stage 2, no Sinkhorn) share those exact
representations. Matcher checkpoints are selected on validation before test files
are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/ubuntu/anaconda3/envs/nuclr310/bin/python")
RUN_ROOT = ROOT / "runs/fold_pure_e2e_candidates_ab_20260817"
SOURCE_PROTOCOL = ROOT / "runs/ambiguity_aware_cv5x3_ablation/protocol_manifest.json"
PROTOCOL = RUN_ROOT / "FOLD_PURE_PROTOCOL.json"
SEEDS = (1, 42, 123)
DATASET_ROOTS = {
    "atanas": ROOT / "Data/Atanas_SF_unified_000776/date_disjoint_v1/full",
    "rld": ROOT / "Data/Dunn_001623/date_disjoint_full95_v1",
}
EXTERNAL_INIT = RUN_ROOT / "external_init_ey/t1st1_d256/truncated.pt"
EXTERNAL_INIT_PROVENANCE = (
    "EY-only NuCLR Stage-1 seed42; no Atanas or RLD recordings; "
    "weight-preserving T2/ST2 to T1/ST1 truncation before fold-specific fine-tuning"
)

# CPU-feasible, predeclared encoder budget. This is identical across all 30
# fold/seed encoders and is never changed after results are observed.
NUCLR_BUDGET = {
    "epochs": 10,
    "optimizer_steps_per_epoch": 16,
    "batch_size": 4,
    "patience": 4,
    "eval_num_windows": 8,
    "extract_num_windows": 16,
}
MATCHER_BUDGET = {"epochs": 80, "max_pairs_per_epoch": 200, "patience": 15}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_map(dataset: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(DATASET_ROOTS[dataset].rglob("*.npz")):
        resolved = path.resolve()
        if path.stem in paths and paths[path.stem] != resolved:
            raise RuntimeError(f"Conflicting raw files for {dataset}/{path.stem}")
        paths[path.stem] = resolved
    return paths


def ensure_link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Wrong existing link: {link}")
        return
    if link.exists():
        raise RuntimeError(f"Refusing to overwrite {link}")
    link.symlink_to(target.resolve())


def write_list(path: Path, values: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(value.resolve()) for value in values) + "\n", encoding="utf-8")


def prepare() -> None:
    source = json.loads(SOURCE_PROTOCOL.read_text())
    manifest = {
        "protocol": "fold-pure end-to-end grouped 5-fold CV x 3 seeds",
        "candidates": {
            "A": "HyQuRP + Compact NuCLR + concat + Population Transformer; no Stage 2; no Sinkhorn",
            "B": "HyQuRP + Compact NuCLR + Stage 2 + Population Transformer; no Sinkhorn",
        },
        "compact_nuclr": "EY-only initialized T1/ST1 D256, separately fine-tuned for every dataset/fold/seed",
        "external_initializer": {
            "path": str(EXTERNAL_INIT.resolve()),
            "sha256": sha256(EXTERNAL_INIT),
            "provenance": EXTERNAL_INIT_PROVENANCE,
        },
        "seeds_cover_entire_pipeline": list(SEEDS),
        "nuclr_budget": NUCLR_BUDGET,
        "matcher_budget": MATCHER_BUDGET,
        "test_policy": (
            "Raw outer-test is physically absent from NuCLR train/val root. "
            "Test representation extraction occurs only after the validation-selected NuCLR checkpoint. "
            "Matchers open test lists only after validation selects best.pt."
        ),
        "datasets": {},
    }
    for dataset in ("atanas", "rld"):
        raw = source_map(dataset)
        folds_out = []
        for fold_info in source["datasets"][dataset]["folds"]:
            fold = int(fold_info["fold"])
            ids = {
                "train": list(fold_info["train_ids"]),
                "val": list(fold_info["val_ids"]),
                "test": list(fold_info["test_ids"]),
            }
            all_ids = set(ids["train"] + ids["val"] + ids["test"])
            if len(all_ids) != len(ids["train"]) + len(ids["val"]) + len(ids["test"]):
                raise RuntimeError(f"{dataset} fold {fold}: split overlap")
            missing = sorted(all_ids - set(raw))
            if missing:
                raise RuntimeError(f"{dataset} fold {fold}: missing raw IDs {missing}")
            fold_root = RUN_ROOT / "folds" / dataset / f"fold_{fold}"
            train_val_root = fold_root / "data_train_val_only"
            sealed_root = fold_root / "sealed_test"
            for split in ("train", "val"):
                for worm_id in ids[split]:
                    ensure_link(raw[worm_id], train_val_root / split / f"{worm_id}.npz")
            for worm_id in ids["test"]:
                ensure_link(raw[worm_id], sealed_root / "test" / f"{worm_id}.npz")
            (fold_root / "test_ids.txt").write_text("\n".join(ids["test"]) + "\n", encoding="utf-8")
            folds_out.append({
                "fold": fold,
                "train_ids": ids["train"],
                "val_ids": ids["val"],
                "test_ids": ids["test"],
                "counts": {key: len(value) for key, value in ids.items()},
                "intersections": {"train_val": [], "train_test": [], "val_test": []},
                "rld_test_acquisition_dates": sorted({value[:8] for value in ids["test"]}) if dataset == "rld" else None,
            })
        manifest["datasets"][dataset] = {"num_worms": len(raw), "folds": folds_out}
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_ROOT / "FOLD_PURE_PROTOCOL.sha256").write_text(
        f"{sha256(PROTOCOL)}  {PROTOCOL.name}\n", encoding="utf-8"
    )
    print(json.dumps({"prepared": str(RUN_ROOT), "nuclr_jobs": 30, "matcher_jobs": 60}, ensure_ascii=False))


def nuclr_command(dataset: str, fold: int, seed: int, threads: int, benchmark: bool = False) -> tuple[list[str], Path]:
    fold_root = RUN_ROOT / "folds" / dataset / f"fold_{fold}"
    save = fold_root / "pipeline" / f"seed_{seed}" / "nuclr_t1st1"
    counts = json.loads(PROTOCOL.read_text())["datasets"][dataset]["folds"][fold - 1]["counts"]
    epochs = 1 if benchmark else NUCLR_BUDGET["epochs"]
    steps = 4 if benchmark else NUCLR_BUDGET["optimizer_steps_per_epoch"]
    patience = 1 if benchmark else NUCLR_BUDGET["patience"]
    windows = 4 if benchmark else NUCLR_BUDGET["eval_num_windows"]
    command = [
        str(PYTHON), "-u", str(ROOT / "hybrid/activity.py"),
        "--train-root", str(fold_root / "data_train_val_only/train"),
        "--val-root", str(fold_root / "data_train_val_only/val"),
        "--test-worm-ids-file", str(fold_root / "test_ids.txt"),
        "--expected-train-worms", str(counts["train"]),
        "--expected-val-worms", str(counts["val"]),
        "--expected-test-worms", str(counts["test"]),
        "--split-manifest", str(PROTOCOL), "--split-manifest-sha256", sha256(PROTOCOL),
        "--save-dir", str(save), "--model-variant", "raw_nuclr",
        "--init-backbone-checkpoint", str(EXTERNAL_INIT),
        "--external-init-checkpoint-sha256", sha256(EXTERNAL_INIT),
        "--external-init-provenance", EXTERNAL_INIT_PROVENANCE,
        "--activity-key", "activity_raw", "--label-key", "cell_id",
        "--supervision-mask-key", "clean_mask", "--same-worm-neurons", "all",
        "--source-fs", "4.0", "--epochs", str(epochs), "--batch-size", str(NUCLR_BUDGET["batch_size"]),
        "--same-views-per-worm", "8", "--cross-pairs-per-epoch", "0",
        "--cross-views-per-pair", "2", "--optimizer-steps-per-epoch", str(steps),
        "--cross-window-mode", "independent", "--window-seconds", "30",
        "--eval-num-windows", str(windows), "--feature-num-windows", "8",
        "--feature-resample-points", "256", "--unit-dropout", "official",
        "--minimum-positive-matches", "8", "--lambda-same", "1", "--lambda-cross", "0",
        "--temperature", "0.2", "--projector-dim", "128", "--full-denom",
        "--nuclr-patch-size", "4", "--nuclr-dim", "256", "--nuclr-heads", "4",
        "--nuclr-dim-head", "64", "--nuclr-temporal-layers", "1",
        "--nuclr-spatiotemporal-layers", "1", "--nuclr-attention-dropout", "0",
        "--nuclr-linear-dropout", "0.2", "--nuclr-rot-ratio", "0.5",
        "--backbone-lr", "1e-5", "--branch-lr", "1e-4", "--projector-lr", "1.25e-4",
        "--weight-decay", "0.01", "--warmup-steps", "100", "--min-lr-ratio", "0.1",
        "--grad-clip", "1.0", "--feature-dropout", "0.1", "--val-every-epochs", "1",
        "--patience", str(patience), "--save-every-epochs", "10",
        "--selection-space", "encoder", "--selection-metric", "top1",
        "--seed", str(seed), "--device", "cpu",
    ]
    return command, save


def extract_commands(dataset: str, fold: int, seed: int) -> tuple[list[list[str]], Path]:
    fold_root = RUN_ROOT / "folds" / dataset / f"fold_{fold}"
    pipeline = fold_root / "pipeline" / f"seed_{seed}"
    checkpoint = pipeline / "nuclr_t1st1/best.pt"
    output = pipeline / "embeddings"
    common = [
        str(PYTHON), "-u", str(ROOT / "engines/extract_rld_nuclr_capacity_embeddings_v3.py"),
        "--checkpoint", str(checkpoint), "--training-script", str(ROOT / "hybrid/activity.py"),
        "--output-root", str(output), "--activity-key", "activity_raw", "--label-key", "cell_id",
        "--supervision-mask-key", "clean_mask", "--num-windows", str(NUCLR_BUDGET["extract_num_windows"]),
        "--window-seconds", "30", "--device", "cpu", "--seed", str(seed),
    ]
    commands = [
        common + ["--data-root", str(fold_root / "data_train_val_only"), "--splits", "train,val"],
        common + ["--data-root", str(fold_root / "sealed_test"), "--splits", "test"],
    ]
    return commands, output


def matcher_command(dataset: str, fold: int, seed: int, candidate: str, threads: int) -> tuple[list[str], Path]:
    fold_root = RUN_ROOT / "folds" / dataset / f"fold_{fold}"
    pipeline = fold_root / "pipeline" / f"seed_{seed}"
    embeddings = pipeline / "embeddings"
    lists = pipeline / "lists"
    for split in ("train", "val", "test"):
        write_list(lists / f"{split}.txt", sorted((embeddings / split).glob("*.npz")))
    save = pipeline / f"candidate_{candidate.lower()}"
    common = [
        "--activity_dim", "256", "--position_dim", "32", "--d_model", "128",
        "--n_heads", "4", "--n_layers", "3", "--ff_dim", "256", "--dropout", "0.1",
        "--epochs", str(MATCHER_BUDGET["epochs"]),
        "--max_pairs_per_epoch", str(MATCHER_BUDGET["max_pairs_per_epoch"]),
        "--lr", "0.0003", "--weight_decay", "0.0001", "--outlier_weight", "0.25",
        "--grad_clip", "2.0", "--patience", str(MATCHER_BUDGET["patience"]),
        "--warmup_epochs", "3", "--selection_metric", "top1", "--seed", str(seed),
        "--device", "cpu", "--num_threads", str(threads),
    ]
    if candidate == "A":
        command = [
            str(PYTHON), "-u", str(ROOT / "train_hyqurp_nuclr_quantum_crossmodal_v2.py"),
            "--train_list", str(lists / "train.txt"), "--val_list", str(lists / "val.txt"),
            "--test_list", str(lists / "test.txt"), "--save_dir", str(save),
            "--source_data_root", str(DATASET_ROOTS[dataset]), "--position_encoder", "hyqurp",
            "--fusion", "concat", *common,
        ]
    elif candidate == "B":
        dashed = [value.replace("_", "-") if value.startswith("--") else value for value in common]
        command = [
            str(PYTHON), "-u", str(ROOT / "engines/train_ambiguity_aware_geo_activity_transformer.py"),
            "--train-list", str(lists / "train.txt"), "--val-list", str(lists / "val.txt"),
            "--test-list", str(lists / "test.txt"), "--save-dir", str(save),
            "--source-data-root", str(DATASET_ROOTS[dataset]), "--disable-sinkhorn", *dashed,
        ]
    else:
        raise ValueError(candidate)
    return command, save


def execute(command: list[str], log: Path, threads: int) -> tuple[int, float]:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    start = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    return proc.returncode, time.monotonic() - start


def run_parallel(stage: str, specs: list[tuple], worker, max_workers: int) -> None:
    failures = []
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker, spec): spec for spec in specs}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "failed":
                failures.append(result)
            print(json.dumps({"stage": stage, "progress": f"{index}/{len(specs)}", "result": result,
                              "elapsed_minutes": round((time.monotonic() - start) / 60, 2)}, ensure_ascii=False), flush=True)
    status = {"stage": stage, "jobs": len(specs), "failures": failures,
              "elapsed_seconds": time.monotonic() - start}
    (RUN_ROOT / f"status_{stage}.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{stage}: {len(failures)} failures")


def run_nuclr(max_workers: int, threads: int) -> None:
    specs = [(d, f, s) for d in ("atanas", "rld") for f in range(1, 6) for s in SEEDS]
    def worker(spec: tuple) -> dict:
        d, f, s = spec
        command, save = nuclr_command(d, f, s, threads)
        if (save / "summary.json").is_file() and (save / "best.pt").is_file():
            return {"spec": spec, "status": "reused", "seconds": 0.0}
        code, seconds = execute(command, save / "train.log", threads)
        ok = code == 0 and (save / "summary.json").is_file() and (save / "best.pt").is_file()
        return {"spec": spec, "status": "completed" if ok else "failed", "returncode": code, "seconds": seconds}
    run_parallel("nuclr", specs, worker, max_workers)


def run_extract(max_workers: int, threads: int) -> None:
    specs = [(d, f, s) for d in ("atanas", "rld") for f in range(1, 6) for s in SEEDS]
    def worker(spec: tuple) -> dict:
        d, f, s = spec
        commands, output = extract_commands(d, f, s)
        expected = json.loads(PROTOCOL.read_text())["datasets"][d]["folds"][f - 1]["counts"]
        if all(len(list((output / split).glob("*.npz"))) == expected[split] for split in ("train", "val", "test")):
            return {"spec": spec, "status": "reused", "seconds": 0.0}
        total = 0.0
        for index, command in enumerate(commands):
            code, seconds = execute(command, output / f"extract_{index}.log", threads)
            total += seconds
            if code != 0:
                return {"spec": spec, "status": "failed", "returncode": code, "seconds": total}
        ok = all(len(list((output / split).glob("*.npz"))) == expected[split] for split in ("train", "val", "test"))
        return {"spec": spec, "status": "completed" if ok else "failed", "seconds": total}
    run_parallel("extract", specs, worker, max_workers)


def run_matchers(max_workers: int, threads: int) -> None:
    specs = [(d, f, s, c) for d in ("atanas", "rld") for f in range(1, 6) for s in SEEDS for c in ("A", "B")]
    def worker(spec: tuple) -> dict:
        d, f, s, c = spec
        command, save = matcher_command(d, f, s, c, threads)
        if (save / "summary.json").is_file():
            return {"spec": spec, "status": "reused", "seconds": 0.0}
        code, seconds = execute(command, save / "train.log", threads)
        ok = code == 0 and (save / "summary.json").is_file()
        return {"spec": spec, "status": "completed" if ok else "failed", "returncode": code, "seconds": seconds}
    run_parallel("matchers", specs, worker, max_workers)


def benchmark(threads: int) -> None:
    command, _ = nuclr_command("atanas", 1, 1, threads, benchmark=True)
    target = RUN_ROOT / "benchmark/atanas_fold1_seed1_external_init"
    command[command.index("--save-dir") + 1] = str(target)
    code, seconds = execute(command, target / "train.log", threads)
    print(json.dumps({"benchmark": "1 epoch x 4 optimizer steps; 4-window validation", "returncode": code,
                      "seconds": seconds, "log": str(target / "train.log")}, ensure_ascii=False))
    if code != 0:
        raise RuntimeError("Benchmark failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "benchmark", "nuclr", "extract", "matchers", "all"))
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--threads-per-job", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        return
    if not PROTOCOL.is_file():
        prepare()
    if args.action == "benchmark":
        benchmark(args.threads_per_job)
    elif args.action == "nuclr":
        run_nuclr(args.max_workers, args.threads_per_job)
    elif args.action == "extract":
        run_extract(args.max_workers, args.threads_per_job)
    elif args.action == "matchers":
        run_matchers(args.max_workers, args.threads_per_job)
    else:
        run_nuclr(args.max_workers, args.threads_per_job)
        run_extract(args.max_workers, args.threads_per_job)
        run_matchers(max(args.max_workers, 16), max(1, args.threads_per_job // 2))


if __name__ == "__main__":
    main()
