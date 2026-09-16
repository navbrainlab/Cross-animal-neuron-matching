#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _utilities(package_root: Path):
    sys.path.insert(0, str(package_root))
    from mprt_net.experiments import orchestration

    return orchestration


def _source_run(repo_root: Path, dataset: str) -> Path:
    return repo_root / "runs" / "mprt_v1_1" / dataset / "seed42" / "full"


def _run_dir(repo_root: Path, dataset: str, fold: int, seed: int) -> Path:
    return (
        repo_root
        / "runs"
        / "mprt_v1_1_final_cv5x3_s1_42_123"
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
    )


def _paths(repo_root: Path, dataset: str, fold: int, seed: int) -> dict[str, Path]:
    root = _run_dir(repo_root, dataset, fold, seed)
    return {
        "root": root,
        "train": root / "full",
        "baseline": root / "full" / "best.pt",
        "gated": root / "anchored_gated_b030.pt",
        "atlas": root / "anchored_pure.pt",
        "atlas_spec": root / "atlas_spec.json",
    }


def _dataset_fold_root(args: argparse.Namespace, dataset: str, fold: int) -> Path:
    return args.cv_roots[dataset] / f"fold_{fold}"


def _fit_one(task: dict[str, Any], gpu: str, args: argparse.Namespace, util: Any) -> None:
    dataset, fold, seed = task["dataset"], task["fold"], task["seed"]
    fold_root = _dataset_fold_root(args, dataset, fold)
    for split in ("train", "val", "test"):
        if not any((fold_root / split).glob("*.npz")):
            raise FileNotFoundError(f"No {split} NPZ files in {fold_root}")
    source_run = _source_run(args.repo_root, dataset)
    source_checkpoint = source_run / "best.pt"
    parser = util.train_parser(args.package_root)
    source = util.complete_arguments(parser, util.source_arguments(source_run))
    for key in ("cycle_weight", "atlas_weight", "atlas_blend_weight"):
        if float(source.get(key, 0.0)) != 0.0:
            raise ValueError(f"Reference run is not atlas/cycle-free: {key}={source[key]}")
    paths = _paths(args.repo_root, dataset, fold, seed)
    spec = util.train_spec(
        source_checkpoint=source_checkpoint,
        source_values=source,
        dataset_root=fold_root,
        seed=seed,
        variant="full",
    )
    spec.update(protocol="outer_cv5_train60_val20_test20", fold=fold)
    if not util.ensure_clean_or_reusable_run(paths["train"], spec):
        values = dict(source)
        values.update(
            dataset_root=str(fold_root),
            output_dir=str(paths["train"]),
            seed=seed,
            variant="full",
            cycle_weight=0.0,
            atlas_weight=0.0,
            atlas_blend_weight=0.0,
            device="cuda",
            allow_existing_output=False,
        )
        paths["train"].mkdir(parents=True, exist_ok=True)
        util.run_command(
            util.command_from_values(parser, values),
            cwd=args.package_root,
            gpu=gpu,
            log_path=paths["train"] / "train.log",
        )
        util.write_json(paths["train"] / "experiment_spec.json", spec)
    atlas_spec = {
        "method": "pure_identity_anchored_relational_atlas",
        "selection_source": "pure selected on the original Atanas 3-seed validation protocol",
        "dataset_root": str(fold_root.resolve()),
        "atlas_build_split": "train",
        "baseline_checkpoint": str(paths["baseline"].resolve()),
        "baseline_sha256": util.sha256(paths["baseline"]),
        "blend_weight": 1.0,
        "confidence_gating": False,
    }
    if paths["atlas"].is_file() and paths["gated"].is_file() and paths["atlas_spec"].is_file():
        if util.read_json(paths["atlas_spec"]) != atlas_spec:
            raise RuntimeError(f"Existing atlas has a different specification: {paths['root']}")
    else:
        if any(path.exists() for path in (paths["atlas"], paths["gated"], paths["atlas_spec"])):
            raise FileExistsError(f"Partial atlas outputs exist: {paths['root']}")
        util.run_command(
            [
                sys.executable,
                "-u",
                "-m",
                "mprt_net.build_anchored_atlas",
                "--dataset-root",
                str(fold_root),
                "--split",
                "train",
                "--checkpoint",
                str(paths["baseline"]),
                "--output",
                str(paths["gated"]),
                "--pure-output",
                str(paths["atlas"]),
                "--activity-length",
                str(args.activity_length),
                "--blend-weight",
                "0.30",
                "--gate-temperature",
                "0.05",
                "--device",
                "cuda",
            ],
            cwd=args.package_root,
            gpu=gpu,
            log_path=paths["root"] / "atlas_build.log",
        )
        util.write_json(paths["atlas_spec"], atlas_spec)
    _evaluate(task, "val", gpu, args, util)


def _evaluate(
    task: dict[str, Any], split: str, gpu: str, args: argparse.Namespace, util: Any
) -> dict[str, Path]:
    dataset, fold, seed = task["dataset"], task["fold"], task["seed"]
    paths = _paths(args.repo_root, dataset, fold, seed)
    output_dir = paths["root"] / f"{split}_comparison"
    output = output_dir / "pairwise_vs_pure_atlas.json"
    query_output = output_dir / "pairwise_vs_pure_atlas_queries.csv"
    reusable = False
    if output.is_file() and query_output.is_file():
        existing = util.read_json(output)
        expected = {
            "dataset_root": str(_dataset_fold_root(args, dataset, fold).resolve()),
            "split": split,
            "fold": fold,
            "seed": seed,
            "checkpoint_a": str(paths["baseline"].resolve()),
            "checkpoint_b": str(paths["atlas"].resolve()),
        }
        differences = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if differences:
            raise RuntimeError(f"Stale evaluation in {output}: {differences}")
        reusable = True
    elif output.exists() or query_output.exists():
        raise FileExistsError(f"Partial evaluation outputs exist in {output_dir}")
    if not reusable:
        util.run_command(
            [
                sys.executable,
                "-m",
                "mprt_net.experiments.paired_query_eval",
                "--dataset-root",
                str(_dataset_fold_root(args, dataset, fold)),
                "--split",
                split,
                "--checkpoint-a",
                str(paths["baseline"]),
                "--checkpoint-b",
                str(paths["atlas"]),
                "--dataset",
                dataset,
                "--fold",
                str(fold),
                "--seed",
                str(seed),
                "--activity-length",
                str(args.activity_length),
                "--device",
                "cuda",
                "--output",
                str(output),
                "--query-output",
                str(query_output),
            ],
            cwd=args.package_root,
            gpu=gpu,
            log_path=output_dir / "evaluate.log",
        )
    return {"json": output, "csv": query_output}


def _parallel(tasks: list[dict[str, Any]], gpus: tuple[str, ...], fn, args, util) -> None:
    buckets = util.partition(tasks, len(gpus))
    def worker(gpu: str, bucket: list[dict[str, Any]]) -> None:
        for task in bucket:
            fn(task, gpu, args, util)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, bucket) for gpu, bucket in zip(gpus, buckets)]
        for future in futures:
            future.result()


def _lock_payload(tasks: list[dict[str, Any]], args: argparse.Namespace, util: Any) -> dict[str, Any]:
    checkpoints = []
    for task in tasks:
        paths = _paths(args.repo_root, task["dataset"], task["fold"], task["seed"])
        for role in ("baseline", "atlas"):
            path = paths[role]
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints.append(
                {
                    **task,
                    "role": role,
                    "path": str(path.resolve()),
                    "sha256": util.sha256(path),
                }
            )
    return {
        "protocol": "MPRT pure anchored atlas, grouped outer 5-fold x 3 seeds",
        "selection": "checkpoint epoch selected only by each fold's validation split",
        "atlas": "built after training from fold-train labels only; pure rule fixed beforehand",
        "test_policy": "test evaluation forbidden until this lock exists and all hashes match",
        "datasets": {key: str(value.resolve()) for key, value in args.cv_roots.items() if key in args.datasets},
        "folds": list(args.folds),
        "seeds": list(args.seeds),
        "checkpoints": checkpoints,
    }


def _verify_lock(lock: dict[str, Any], tasks: list[dict[str, Any]], args, util) -> None:
    expected = _lock_payload(tasks, args, util)
    if lock != expected:
        raise RuntimeError(
            "Locked protocol/checkpoint hashes no longer match. Do not evaluate test; "
            "start a new named CV run if the model changed."
        )


def _summary(tasks: list[dict[str, Any]], split: str, args, util) -> dict[str, Any]:
    rows = []
    entries = []
    for task in tasks:
        outputs = {
            "json": _paths(args.repo_root, task["dataset"], task["fold"], task["seed"])["root"]
            / f"{split}_comparison"
            / "pairwise_vs_pure_atlas.json",
            "csv": _paths(args.repo_root, task["dataset"], task["fold"], task["seed"])["root"]
            / f"{split}_comparison"
            / "pairwise_vs_pure_atlas_queries.csv",
        }
        result = util.read_json(outputs["json"])
        rows.append(
            {
                **task,
                "split": split,
                "queries": result["queries"],
                "pairwise_top1": result["top1_real_a"],
                "atlas_top1": result["top1_real_b"],
                "atlas_minus_pairwise": result["delta_b_minus_a"],
                "result": str(outputs["json"]),
            }
        )
        entries.append({**task, "split": split, "query_csv": str(outputs["csv"].resolve())})
    aggregate = []
    for dataset in args.datasets:
        selected = [row for row in rows if row["dataset"] == dataset]
        aggregate.append(
            {
                "dataset": dataset,
                "fold_seed_cells": len(selected),
                "pairwise_top1_macro": statistics.mean(row["pairwise_top1"] for row in selected),
                "atlas_top1_macro": statistics.mean(row["atlas_top1"] for row in selected),
                "delta_macro": statistics.mean(row["atlas_minus_pairwise"] for row in selected),
                "delta_cell_sd": statistics.stdev(row["atlas_minus_pairwise"] for row in selected),
            }
        )
    return {"split": split, "rows": rows, "aggregate": aggregate, "entries": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-locked MPRT 5-fold x 3-seed protocol")
    parser.add_argument("phase", choices=("fit-lock", "locked-test", "summarize"))
    parser.add_argument("--repo-root", type=Path, default=Path("/home/ubuntu/klb/nuclr/nuclr"))
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument(
        "--atanas-cv-root",
        type=Path,
        default=Path("/home/ubuntu/klb/nuclr/nuclr/Data/Atanas_SF_unified_000776/cv5_grouped_v1"),
    )
    parser.add_argument(
        "--rld-cv-root",
        type=Path,
        default=Path("/home/ubuntu/klb/nuclr/nuclr/Data/Dunn_001623/cv5_grouped_v1"),
    )
    parser.add_argument("--datasets", default="atanas,rld")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--activity-length", type=int, default=512)
    args = parser.parse_args()
    args.package_root = args.package_root or args.repo_root / "mprt_net_v1_1"
    util = _utilities(args.package_root)
    args.datasets = util.parse_csv(args.datasets)
    args.folds = util.parse_int_csv(args.folds)
    args.seeds = util.parse_int_csv(args.seeds)
    args.gpus = util.parse_csv(args.gpus)
    args.cv_roots = {"atanas": args.atanas_cv_root, "rld": args.rld_cv_root}
    if not set(args.datasets).issubset(args.cv_roots):
        parser.error("--datasets may contain only atanas,rld")
    if len(args.folds) != 5 or len(args.seeds) != 3:
        parser.error("Final protocol requires exactly 5 folds and 3 seeds")
    tasks = [
        {"dataset": dataset, "fold": fold, "seed": seed}
        for dataset in args.datasets
        for fold in args.folds
        for seed in args.seeds
    ]
    protocol_root = args.repo_root / "runs" / "mprt_v1_1_final_cv5x3_s1_42_123"
    lock_path = protocol_root / "LOCKED_CHECKPOINTS.json"
    if args.phase == "fit-lock":
        _parallel(tasks, args.gpus, _fit_one, args, util)
        proposed = _lock_payload(tasks, args, util)
        if lock_path.exists():
            if util.read_json(lock_path) != proposed:
                raise RuntimeError(f"A different lock already exists: {lock_path}")
        else:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            with lock_path.open("x", encoding="utf-8") as handle:
                json.dump(proposed, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        val = _summary(tasks, "val", args, util)
        util.write_json(protocol_root / "val_summary.json", val)
        util.write_json(protocol_root / "val_bootstrap_manifest.json", {"entries": val["entries"]})
        print(f"locked={lock_path}")
        print("Test has not been evaluated. Run the locked-test phase exactly once.")
        return
    if not lock_path.is_file():
        raise FileNotFoundError("Run fit-lock before any test evaluation")
    _verify_lock(util.read_json(lock_path), tasks, args, util)
    if args.phase == "locked-test":
        _parallel(
            tasks,
            args.gpus,
            lambda task, gpu, a, u: _evaluate(task, "test", gpu, a, u),
            args,
            util,
        )
    test = _summary(tasks, "test", args, util)
    util.write_json(protocol_root / "test_summary.json", test)
    manifest = protocol_root / "test_bootstrap_manifest.json"
    util.write_json(manifest, {"entries": test["entries"]})
    print(json.dumps({"aggregate": test["aggregate"]}, indent=2, ensure_ascii=False))
    print(f"bootstrap_manifest={manifest}")


if __name__ == "__main__":
    main()
