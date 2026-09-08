#!/usr/bin/env python3
"""Strict component ablation for the final static-atlas MPRT pipeline.

Arms
----
full            Existing locked full NeuRID checkpoint (reused, not retrained)
geometry_only   w/o Activity: removes activity node AND activity relation inputs
activity_only   w/o Geometry: removes geometry node AND geometry relation inputs
node_only       w/o Population Relations: disables population encoder AND relation transport
no_transport    w/o Relation Transport: keeps relation-conditioned population formation,
                disables cross-population relation transport only

Protocol
--------
* datasets: Atanas + RLD
* grouped outer CV: folds 0..4
* seeds: 1, 42, 123 (identical to the current final MPRT CV protocol)
* fit-lock: train/val only; build each static atlas from fold-train labels only
* locked-test: forbidden until all ablation checkpoints are hashed and locked
* model selection: validation Top-1 only, inherited verbatim from the current full runner

This script deliberately reuses the already-trained Full checkpoints from
``mprt_v1_1_dynamic_residual_atlas_cv5x3_v1``.  ``--datasets`` and
``--ablations`` may select a strict subset, while ``--run-name`` keeps a new
experiment lock separate from any previously completed ablation study.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("atanas", "rld")
FOLDS = (0, 1, 2, 3, 4)
SEEDS = (1, 42, 123)
ALL_ABLATIONS = ("geometry_only", "activity_only", "node_only", "no_transport")
DEFAULT_ABLATIONS = ("geometry_only", "node_only", "no_transport")
DISPLAY = {
    "full": "Full NeuRID",
    "geometry_only": "w/o Activity",
    "activity_only": "w/o Geometry (Activity-only)",
    "node_only": "Node-only",
    "no_transport": "w/o Relation Transport",
}
METRICS = ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy")
PROTOCOL = "mprt_component_ablation_grouped_outer_cv5x3_v1"


def _utilities(package_root: Path):
    sys.path.insert(0, str(package_root))
    from mprt_net.experiments import orchestration

    return orchestration


def _root(args: argparse.Namespace) -> Path:
    return args.repo_root / "runs" / args.run_name


def _final_full_root(repo_root: Path) -> Path:
    # Current final CV namespace. We reuse only the STATIC atlas arm.
    return repo_root / "runs" / "mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"


def _fold_root(args: argparse.Namespace, dataset: str, fold: int) -> Path:
    return args.cv_roots[dataset] / f"fold_{fold}"


def _source_run(args: argparse.Namespace, dataset: str) -> Path:
    # Same canonical source used by the existing final CV runner.
    return args.repo_root / "runs" / "mprt_v1_1" / dataset / "seed42" / "full"


def _cell_root(args: argparse.Namespace, dataset: str, fold: int, seed: int) -> Path:
    return _root(args) / dataset / f"fold{fold}" / f"seed{seed}"


def _variant_root(
    args: argparse.Namespace, dataset: str, fold: int, seed: int, variant: str
) -> Path:
    return _cell_root(args, dataset, fold, seed) / variant


def _pairwise_run(
    args: argparse.Namespace, dataset: str, fold: int, seed: int, variant: str
) -> Path:
    return _variant_root(args, dataset, fold, seed, variant) / "pairwise"


def _pairwise_checkpoint(
    args: argparse.Namespace, dataset: str, fold: int, seed: int, variant: str
) -> Path:
    return _pairwise_run(args, dataset, fold, seed, variant) / "best.pt"


def _static_root(
    args: argparse.Namespace, dataset: str, fold: int, seed: int, variant: str
) -> Path:
    return _variant_root(args, dataset, fold, seed, variant) / "static_atlas"


def _static_checkpoint(
    args: argparse.Namespace, dataset: str, fold: int, seed: int, variant: str
) -> Path:
    return _static_root(args, dataset, fold, seed, variant) / "anchored_pure.pt"


def _full_static_checkpoint(
    args: argparse.Namespace, dataset: str, fold: int, seed: int
) -> Path:
    return (
        _final_full_root(args.repo_root)
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
        / "static_atlas"
        / "anchored_pure.pt"
    )


def _full_pairwise_checkpoint(
    args: argparse.Namespace, dataset: str, fold: int, seed: int
) -> Path:
    return (
        _final_full_root(args.repo_root)
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
        / "pairwise"
        / "full"
        / "best.pt"
    )


def _metric_path(
    args: argparse.Namespace,
    dataset: str,
    fold: int,
    seed: int,
    variant: str,
    split: str,
) -> Path:
    return _cell_root(args, dataset, fold, seed) / "metrics" / split / f"{variant}.json"


def _pair_metric_path(
    args: argparse.Namespace,
    dataset: str,
    fold: int,
    seed: int,
    variant: str,
    split: str,
) -> Path:
    return _cell_root(args, dataset, fold, seed) / "metrics" / split / f"{variant}_queries.csv"


def _comparison_path(
    args: argparse.Namespace,
    dataset: str,
    fold: int,
    seed: int,
    variant: str,
    split: str,
) -> Path:
    return (
        _cell_root(args, dataset, fold, seed)
        / "paired_comparisons"
        / split
        / f"full_vs_{variant}.json"
    )


def _comparison_queries_path(path: Path) -> Path:
    return path.with_name(path.stem + "_queries.csv")


def _require_npz(root: Path, splits: Iterable[str]) -> None:
    for split in splits:
        directory = root / split
        if not directory.is_dir() or not any(directory.glob("*.npz")):
            raise FileNotFoundError(f"No NPZ files in {directory}")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def _validate_variant_semantics(package_root: Path) -> None:
    """Fail fast if the installed package no longer implements our intended ablations."""
    sys.path.insert(0, str(package_root))
    import torch

    from mprt_net.data import WormSample
    from mprt_net.model import MPRTNet
    from mprt_net.train import build_parser, config_for_variant

    parser = build_parser()

    def cfg(variant: str):
        # Give only the two required paths; all architectural values use parser defaults.
        ns = parser.parse_args(
            ["--dataset-root", "/tmp/dummy", "--output-dir", "/tmp/dummy-out", "--variant", variant]
        )
        return config_for_variant(ns)

    full = cfg("full")
    geometry = cfg("geometry_only")
    activity = cfg("activity_only")
    node = cfg("node_only")
    no_transport = cfg("no_transport")

    checks = {
        "full": (
            full.use_geometry
            and full.use_activity
            and full.use_population_encoder
            and full.use_relation_transport
        ),
        "geometry_only": (
            geometry.use_geometry
            and not geometry.use_activity
            and geometry.use_population_encoder
            and geometry.use_relation_transport
        ),
        "activity_only": (
            not activity.use_geometry
            and activity.use_activity
            and activity.use_population_encoder
            and activity.use_relation_transport
        ),
        "node_only": (
            node.use_geometry
            and node.use_activity
            and not node.use_population_encoder
            and not node.use_relation_transport
        ),
        "no_transport": (
            no_transport.use_geometry
            and no_transport.use_activity
            and no_transport.use_population_encoder
            and not no_transport.use_relation_transport
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"Installed MPRT variant semantics changed: {failed}")

    # Numerical non-leakage audit: with activity and node count fixed, changing
    # every coordinate must leave the complete activity-only encoding unchanged.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(20260827)
        model = MPRTNet(activity).eval()
        n, t = 7, 64
        activity_input = torch.randn(n, t)
        common = {
            "uid": "activity-only-geometry-invariance-audit",
            "activity": activity_input,
            "cell_ids": tuple(f"N{i}" for i in range(n)),
            "supervised_mask": torch.ones(n, dtype=torch.bool),
            "source_path": "/tmp/activity-only-geometry-invariance-audit.npz",
        }
        first = WormSample(xyz=torch.randn(n, 3), **common)
        second = WormSample(xyz=1000.0 * torch.randn(n, 3) + 500.0, **common)
        encoded_first = model.encode_population(first)
        encoded_second = model.encode_population(second)
        invariant_fields = ("nodes", "relations", "activity_relations")
        changed = [
            name
            for name in invariant_fields
            if not torch.equal(getattr(encoded_first, name), getattr(encoded_second, name))
        ]
        geometry_is_zero = bool(
            torch.count_nonzero(encoded_first.geometry_relations).item() == 0
            and torch.count_nonzero(encoded_second.geometry_relations).item() == 0
        )
        if changed or not geometry_is_zero:
            raise RuntimeError(
                "activity_only is not geometry-invariant: "
                f"changed_fields={changed}, geometry_relations_zero={geometry_is_zero}"
            )


def _fixed_source(
    args: argparse.Namespace, dataset: str, util: Any
) -> tuple[argparse.ArgumentParser, dict[str, Any], Path]:
    source_run = _source_run(args, dataset)
    source_checkpoint = source_run / "best.pt"
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    parser = util.train_parser(args.package_root)
    source = util.complete_arguments(parser, util.source_arguments(source_run))

    # Component ablation must start from the same pairwise training recipe,
    # without auxiliary atlas/cycle objectives silently leaking in.
    for key in ("cycle_weight", "atlas_weight", "atlas_blend_weight"):
        if key in source and float(source.get(key, 0.0)) != 0.0:
            raise ValueError(f"Canonical source is not auxiliary-free: {key}={source[key]}")
    return parser, source, source_checkpoint


def _training_values(
    args: argparse.Namespace,
    dataset: str,
    fold: int,
    seed: int,
    variant: str,
    util: Any,
) -> tuple[argparse.ArgumentParser, dict[str, Any], Path]:
    parser, source, source_checkpoint = _fixed_source(args, dataset, util)
    values = dict(source)
    values.update(
        dataset_root=str(_fold_root(args, dataset, fold)),
        output_dir=str(_pairwise_run(args, dataset, fold, seed, variant)),
        seed=seed,
        variant=variant,
        device="cuda",
        allow_existing_output=False,
    )
    # These keys exist only in newer MPRT package revisions; changing them to
    # zero is consistent with the existing final grouped-CV runner.
    for key in ("cycle_weight", "atlas_weight", "atlas_blend_weight"):
        if key in values:
            values[key] = 0.0
    return parser, values, source_checkpoint


def _completed_train(run_dir: Path, values: dict[str, Any]) -> tuple[bool, str]:
    core = {
        "best": run_dir / "best.pt",
        "last": run_dir / "last.pt",
        "args": run_dir / "args.json",
        "history": run_dir / "history.jsonl",
        "log": run_dir / "train.log",
    }
    missing = [name for name, path in core.items() if not path.is_file()]
    if missing:
        return False, "missing=" + ",".join(missing)
    try:
        history = [
            json.loads(line)
            for line in core["history"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not history:
            return False, "empty history"
        import torch

        best = torch.load(core["best"], map_location="cpu")
        last = torch.load(core["last"], map_location="cpu")
        best_epoch = int(best["epoch"])
        last_epoch = int(last["epoch"])
        history_epoch = int(history[-1]["epoch"])
        if last_epoch != history_epoch:
            return False, f"last_epoch={last_epoch} history_epoch={history_epoch}"
        epochs = int(values.get("epochs", last_epoch))
        patience = int(values.get("early_stopping_patience", 0) or 0)
        finished_by_limit = last_epoch >= epochs
        finished_by_patience = patience > 0 and last_epoch - best_epoch >= patience
        final_marker = "best_val_top1_real=" in core["log"].read_text(
            encoding="utf-8", errors="replace"
        )
        if not (finished_by_limit or finished_by_patience):
            return False, f"stopped early at epoch={last_epoch}, best_epoch={best_epoch}"
        if not final_marker:
            return False, "training log has no final completion marker"
        return True, f"complete last_epoch={last_epoch} best_epoch={best_epoch}"
    except Exception as exc:  # audit helper; report malformed output instead of reusing it
        return False, f"invalid completion metadata: {exc}"


def _normalize_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _validate_args_json(run_dir: Path, expected: dict[str, Any]) -> None:
    actual = _read_json(run_dir / "args.json")
    expected = _normalize_json(expected)
    differences = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    extra = sorted(set(actual) - set(expected))
    if differences or extra:
        raise RuntimeError(
            f"Training arguments changed in {run_dir}: differences={differences}, extra={extra}"
        )


def _archive_pre_epoch_log(run_dir: Path) -> None:
    log = run_dir / "train.log"
    if not log.is_file():
        return
    for index in range(1, 1000):
        target = run_dir / f"train_pre_epoch_interrupted_{index:03d}.log"
        if not target.exists():
            log.replace(target)
            return
    raise RuntimeError(f"Too many archived logs in {run_dir}")


def _ensure_pairwise(
    task: dict[str, Any], variant: str, gpu: str, args: argparse.Namespace, util: Any
) -> Path:
    dataset, fold, seed = task["dataset"], task["fold"], task["seed"]
    fold_root = _fold_root(args, dataset, fold)
    run_dir = _pairwise_run(args, dataset, fold, seed, variant)
    checkpoint = run_dir / "best.pt"
    parser, values, source_checkpoint = _training_values(
        args, dataset, fold, seed, variant, util
    )
    spec = util.train_spec(
        source_checkpoint=source_checkpoint,
        source_values=util.complete_arguments(parser, util.source_arguments(_source_run(args, dataset))),
        dataset_root=fold_root,
        seed=seed,
        variant=variant,
    )
    spec.update(
        protocol=PROTOCOL,
        fold=fold,
        ablation=variant,
        data_access="train/val only during fit-lock",
        definition={
            "geometry_only": "remove activity node and activity edge inputs",
            "activity_only": "remove geometry node and geometry edge inputs",
            "node_only": "disable native population encoder and relation transport",
            "no_transport": "disable relation transport; retain native population encoder",
        }[variant],
    )
    spec_path = run_dir / "experiment_spec.json"

    if checkpoint.is_file() and spec_path.is_file():
        if _read_json(spec_path) != spec:
            raise RuntimeError(f"Stale experiment spec: {spec_path}")
        _validate_args_json(run_dir, values)
        complete, reason = _completed_train(run_dir, values)
        if not complete:
            raise RuntimeError(f"Checkpoint exists but training is incomplete: {run_dir} ({reason})")
        return checkpoint

    complete, reason = _completed_train(run_dir, values)
    if complete:
        _validate_args_json(run_dir, values)
        _write_json(spec_path, spec)
        print(f"[RECOVER] {run_dir} {reason}", flush=True)
        return checkpoint

    progress = [run_dir / name for name in ("best.pt", "last.pt", "history.jsonl", "experiment_spec.json")]
    args_path = run_dir / "args.json"
    pre_epoch_stub = args_path.is_file() and not any(path.exists() for path in progress)
    if pre_epoch_stub:
        _validate_args_json(run_dir, values)
        _archive_pre_epoch_log(run_dir)
        print(f"[RECOVER PRE-EPOCH] {run_dir}", flush=True)
    elif any(path.exists() for path in [*progress, args_path]):
        raise FileExistsError(f"Partial outputs in {run_dir} ({reason})")

    run_dir.mkdir(parents=True, exist_ok=True)
    util.run_command(
        util.command_from_values(parser, values),
        cwd=args.package_root,
        gpu=gpu,
        log_path=run_dir / "train.log",
    )
    _write_json(spec_path, spec)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    _validate_args_json(run_dir, values)
    complete, reason = _completed_train(run_dir, values)
    if not complete:
        raise RuntimeError(f"Training did not finish cleanly: {run_dir} ({reason})")
    return checkpoint


def _validate_static_metadata(checkpoint: Path, pairwise: Path, fold_root: Path) -> None:
    import torch

    state = torch.load(checkpoint, map_location="cpu")
    metadata = state.get("atlas_build")
    if not isinstance(metadata, dict) or not isinstance(state.get("atlas_identity_to_slot"), dict):
        raise ValueError(f"Not an anchored atlas checkpoint: {checkpoint}")
    expected = {
        "source_checkpoint": str(pairwise.resolve()),
        "dataset_root": str(fold_root.resolve()),
        "split": "train",
        "blend_weight": 1.0,
        "confidence_gating": False,
    }
    differences = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if differences:
        raise RuntimeError(f"Static atlas metadata mismatch in {checkpoint}: {differences}")


def _ensure_static(
    task: dict[str, Any], variant: str, pairwise: Path, gpu: str, args: argparse.Namespace, util: Any
) -> Path:
    dataset, fold, seed = task["dataset"], task["fold"], task["seed"]
    fold_root = _fold_root(args, dataset, fold)
    root = _static_root(args, dataset, fold, seed, variant)
    pure = root / "anchored_pure.pt"
    gated = root / "anchored_gated_b030.pt"
    spec_path = root / "experiment_spec.json"
    spec = {
        "protocol": PROTOCOL,
        "variant": variant,
        "dataset_root": str(fold_root.resolve()),
        "atlas_build_split": "train",
        "pairwise_checkpoint": str(pairwise.resolve()),
        "pairwise_sha256": util.sha256(pairwise),
        "blend_weight": 1.0,
        "confidence_gating": False,
        "activity_length": args.activity_length,
    }

    if pure.is_file() and gated.is_file():
        _validate_static_metadata(pure, pairwise, fold_root)
        if spec_path.is_file() and _read_json(spec_path) != spec:
            raise RuntimeError(f"Stale static atlas spec: {spec_path}")
        if not spec_path.is_file():
            _write_json(spec_path, spec)
        return pure
    if any(path.exists() for path in (pure, gated, spec_path)):
        raise FileExistsError(f"Partial static atlas outputs: {root}")

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
            str(pairwise),
            "--output",
            str(gated),
            "--pure-output",
            str(pure),
            "--activity-length",
            str(args.activity_length),
            "--blend-weight",
            "0.30",
            "--gate-temperature",
            "0.05",
            "--device",
            args.device,
        ],
        cwd=args.package_root,
        gpu=gpu,
        log_path=root / "build.log",
    )
    _validate_static_metadata(pure, pairwise, fold_root)
    _write_json(spec_path, spec)
    return pure


def _ensure_full_reference(task: dict[str, Any], args: argparse.Namespace) -> tuple[Path, Path]:
    dataset, fold, seed = task["dataset"], task["fold"], task["seed"]
    pairwise = _full_pairwise_checkpoint(args, dataset, fold, seed)
    static = _full_static_checkpoint(args, dataset, fold, seed)
    if not pairwise.is_file():
        raise FileNotFoundError(f"Missing final Full pairwise checkpoint: {pairwise}")
    if not static.is_file():
        raise FileNotFoundError(f"Missing final Full static checkpoint: {static}")
    _validate_static_metadata(static, pairwise, _fold_root(args, dataset, fold))
    return pairwise, static


def _fit_cell(task: dict[str, Any], gpu: str, args: argparse.Namespace, util: Any) -> None:
    fold_root = _fold_root(args, task["dataset"], task["fold"])
    _require_npz(fold_root, ("train", "val"))
    _ensure_full_reference(task, args)
    for variant in args.ablations:
        pairwise = _ensure_pairwise(task, variant, gpu, args, util)
        _ensure_static(task, variant, pairwise, gpu, args, util)
    _evaluate_cell(task, "val", gpu, args, util)


def _checkpoint_for_variant(task: dict[str, Any], variant: str, args: argparse.Namespace) -> Path:
    if variant == "full":
        return _full_static_checkpoint(args, task["dataset"], task["fold"], task["seed"])
    return _static_checkpoint(args, task["dataset"], task["fold"], task["seed"], variant)


def _evaluate_metrics(
    task: dict[str, Any], variant: str, split: str, checkpoint: Path, gpu: str,
    args: argparse.Namespace, util: Any
) -> None:
    dataset, fold, seed = task["dataset"], task["fold"], task["seed"]
    output = _metric_path(args, dataset, fold, seed, variant, split)
    pair_output = _pair_metric_path(args, dataset, fold, seed, variant, split)
    expected = {
        "dataset_root": str(_fold_root(args, dataset, fold).resolve()),
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
    }
    if output.is_file() and pair_output.is_file():
        actual = _read_json(output)
        differences = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if differences:
            raise RuntimeError(f"Stale metric output {output}: {differences}")
        return
    if output.exists() or pair_output.exists():
        raise FileExistsError(f"Partial metric outputs: {output.parent}")
    util.run_command(
        [
            sys.executable,
            "-u",
            str(args.static_evaluator),
            "--package-root",
            str(args.package_root),
            "--dataset-root",
            str(_fold_root(args, dataset, fold)),
            "--split",
            split,
            "--checkpoint",
            str(checkpoint),
            "--activity-length",
            str(args.activity_length),
            "--device",
            args.device,
            "--dataset",
            dataset,
            "--fold",
            str(fold),
            "--seed",
            str(seed),
            "--variant",
            variant,
            "--output",
            str(output),
            "--query-output",
            str(pair_output),
        ],
        cwd=args.repo_root,
        gpu=gpu,
        log_path=output.with_suffix(".log"),
    )


def _evaluate_paired(
    task: dict[str, Any], variant: str, split: str, full: Path, ablation: Path,
    gpu: str, args: argparse.Namespace, util: Any
) -> None:
    dataset, fold, seed = task["dataset"], task["fold"], task["seed"]
    output = _comparison_path(args, dataset, fold, seed, variant, split)
    queries = _comparison_queries_path(output)
    expected = {
        "dataset": dataset,
        "dataset_root": str(_fold_root(args, dataset, fold).resolve()),
        "split": split,
        "fold": fold,
        "seed": seed,
        "checkpoint_a": str(full.resolve()),
        "checkpoint_b": str(ablation.resolve()),
    }
    if output.is_file() and queries.is_file():
        actual = _read_json(output)
        differences = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if differences:
            raise RuntimeError(f"Stale paired output {output}: {differences}")
        return
    if output.exists() or queries.exists():
        raise FileExistsError(f"Partial paired outputs: {output.parent}")
    util.run_command(
        [
            sys.executable,
            "-m",
            "mprt_net.experiments.paired_query_eval",
            "--dataset-root",
            str(_fold_root(args, dataset, fold)),
            "--split",
            split,
            "--checkpoint-a",
            str(full),
            "--checkpoint-b",
            str(ablation),
            "--dataset",
            dataset,
            "--fold",
            str(fold),
            "--seed",
            str(seed),
            "--activity-length",
            str(args.activity_length),
            "--device",
            args.device,
            "--output",
            str(output),
            "--query-output",
            str(queries),
        ],
        cwd=args.package_root,
        gpu=gpu,
        log_path=output.with_suffix(".log"),
    )


def _evaluate_cell(task: dict[str, Any], split: str, gpu: str, args, util) -> None:
    checkpoints = {
        variant: _checkpoint_for_variant(task, variant, args)
        for variant in ("full", *args.ablations)
    }
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        _evaluate_metrics(task, name, split, path, gpu, args, util)
    for variant in args.ablations:
        _evaluate_paired(
            task, variant, split, checkpoints["full"], checkpoints[variant], gpu, args, util
        )


def _parallel(tasks: list[dict[str, Any]], args, util, function) -> None:
    buckets = util.partition(tasks, len(args.gpus))

    def worker(gpu: str, bucket: list[dict[str, Any]]) -> None:
        for task in bucket:
            print(f"[CELL GPU{gpu}] {task['dataset']} fold{task['fold']} seed{task['seed']}", flush=True)
            function(task, gpu, args, util)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [
            pool.submit(worker, gpu, bucket)
            for gpu, bucket in zip(args.gpus, buckets)
        ]
        for future in futures:
            future.result()


def _lock_payload(tasks: list[dict[str, Any]], args, util) -> dict[str, Any]:
    checkpoints: list[dict[str, Any]] = []
    for task in tasks:
        for variant in ("full", *args.ablations):
            path = _checkpoint_for_variant(task, variant, args)
            if not path.is_file():
                raise FileNotFoundError(path)
            checkpoints.append(
                {
                    **task,
                    "variant": variant,
                    "checkpoint": str(path.resolve()),
                    "sha256": util.sha256(path),
                }
            )
    return {
        "protocol": PROTOCOL,
        "datasets": {name: str(args.cv_roots[name].resolve()) for name in args.datasets},
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "arms": ["full", *args.ablations],
        "full_policy": "reuse final locked static-atlas checkpoint from existing MPRT grouped CV",
        "ablation_policy": "retrain each ablation from scratch; build its own train-only pure anchored atlas",
        "selection": "validation only; test never used for checkpoint or hyperparameter selection",
        "definitions": {
            variant: {
                "geometry_only": "w/o Activity: no activity node or activity relation inputs",
                "activity_only": "w/o Geometry: no geometry node or geometry relation inputs",
                "node_only": "w/o Population Relations: node matching only; population encoder and relation transport disabled",
                "no_transport": "w/o Relation Transport: native relation-conditioned population formation retained",
            }[variant]
            for variant in args.ablations
        },
        "checkpoints": checkpoints,
    }


def _write_or_verify_lock(tasks: list[dict[str, Any]], args, util) -> Path:
    path = _root(args) / "LOCKED_CHECKPOINTS.json"
    payload = _lock_payload(tasks, args, util)
    if path.is_file():
        if _read_json(path) != payload:
            raise RuntimeError(f"Different lock already exists: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    return path


def _verify_lock(tasks: list[dict[str, Any]], args, util) -> None:
    path = _root(args) / "LOCKED_CHECKPOINTS.json"
    if not path.is_file():
        raise FileNotFoundError("Run fit-lock before any test evaluation")
    if _read_json(path) != _lock_payload(tasks, args, util):
        raise RuntimeError("Checkpoint hashes or protocol changed after lock")


def _mean_sd(values: Iterable[float]) -> dict[str, float]:
    xs = list(values)
    return {
        "mean": statistics.mean(xs),
        "sample_sd": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }


def _summarize(tasks: list[dict[str, Any]], split: str, args) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for variant in ("full", *args.ablations):
            metric = _read_json(
                _metric_path(
                    args, task["dataset"], task["fold"], task["seed"], variant, split
                )
            )
            rows.append(
                {
                    **task,
                    "split": split,
                    "variant": variant,
                    "display": DISPLAY[variant],
                    "queries": int(metric["queries"]),
                    **{name: float(metric[name]) for name in METRICS},
                }
            )

    datasets: dict[str, Any] = {}
    for dataset in args.datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        arms: dict[str, Any] = {}
        for variant in ("full", *args.ablations):
            selected = [row for row in dataset_rows if row["variant"] == variant]
            arm = {metric: _mean_sd(row[metric] for row in selected) for metric in METRICS}
            if variant != "full":
                deltas = []
                for task in tasks:
                    if task["dataset"] != dataset:
                        continue
                    full_row = next(
                        row
                        for row in dataset_rows
                        if row["variant"] == "full"
                        and row["fold"] == task["fold"]
                        and row["seed"] == task["seed"]
                    )
                    ablation_row = next(
                        row
                        for row in dataset_rows
                        if row["variant"] == variant
                        and row["fold"] == task["fold"]
                        and row["seed"] == task["seed"]
                    )
                    deltas.append(ablation_row["top1_real"] - full_row["top1_real"])
                arm["top1_delta_vs_full"] = _mean_sd(deltas)
                arm["cells_beating_full"] = sum(value > 0 for value in deltas)
            arms[variant] = arm
        datasets[dataset] = {"fold_seed_cells": 15, "arms": arms}

    summary = {
        "protocol": PROTOCOL,
        "split": split,
        "arms": ["full", *args.ablations],
        "rows": rows,
        "datasets": datasets,
    }
    output = _root(args) / f"{split}_summary.json"
    _write_json(output, summary)
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 118)
    print(f"MPRT COMPONENT ABLATION — {summary['split'].upper()} — 5 FOLDS x 3 SEEDS")
    print("=" * 118)
    for dataset, content in summary["datasets"].items():
        print(f"\n{dataset.upper()}")
        print(f"{'Variant':28s} {'Top-1':>15s} {'Top-5':>15s} {'MRR':>15s} {'Hungarian':>15s} {'ΔTop1 vs Full':>16s}")
        for variant in summary["arms"]:
            arm = content["arms"][variant]
            def fmt(metric: str, percent: bool = True) -> str:
                item = arm[metric]
                scale = 100.0 if percent else 1.0
                return f"{scale*item['mean']:.2f}±{scale*item['sample_sd']:.2f}" if percent else f"{item['mean']:.4f}±{item['sample_sd']:.4f}"
            delta = "—"
            if variant != "full":
                item = arm["top1_delta_vs_full"]
                delta = f"{100*item['mean']:+.2f}±{100*item['sample_sd']:.2f}pp"
            print(
                f"{DISPLAY[variant]:28s} {fmt('top1_real'):>15s} {fmt('top5_real'):>15s} "
                f"{fmt('mrr_real', False):>15s} {fmt('hungarian_accuracy'):>15s} {delta:>16s}"
            )


def _preflight(args: argparse.Namespace, util: Any) -> None:
    _validate_variant_semantics(args.package_root)
    for dataset in args.datasets:
        for fold in FOLDS:
            root = _fold_root(args, dataset, fold)
            _require_npz(root, ("train", "val"))
        _fixed_source(args, dataset, util)
    tasks = [
        {"dataset": dataset, "fold": fold, "seed": seed}
        for dataset in args.datasets
        for fold in FOLDS
        for seed in SEEDS
    ]
    for task in tasks:
        _ensure_full_reference(task, args)
    print("PRE-FLIGHT PASSED")
    print(
        "Ablation semantics and activity-only geometry invariance are exact; "
        "all grouped-CV/full-reference inputs exist."
    )
    print(f"Selected ablations: {','.join(args.ablations)}")
    print(f"New training runs required: {len(tasks) * len(args.ablations)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("preflight", "fit-lock", "locked-test", "summarize"))
    parser.add_argument("--repo-root", type=Path, default=Path("/home/ubuntu/klb/nuclr/nuclr"))
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument(
        "--static-evaluator",
        type=Path,
        default=None,
        help="Production-path static-atlas evaluator; defaults to scripts/neurid/evaluate_mprt_static_atlas.py",
    )
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
    parser.add_argument(
        "--ablations",
        default=",".join(DEFAULT_ABLATIONS),
        help=f"Comma-separated subset of {ALL_ABLATIONS}",
    )
    parser.add_argument(
        "--run-name",
        default="mprt_v1_1_component_ablation_cv5x3_v1",
        help="Directory name created below repo_root/runs; use a new name for a new lock.",
    )
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device used for locked evaluation; CPU is supported when CUDA is unavailable.",
    )
    parser.add_argument("--activity-length", type=int, default=512)
    args = parser.parse_args()

    args.repo_root = args.repo_root.resolve()
    args.package_root = (args.package_root or args.repo_root / "neurid").resolve()
    args.static_evaluator = (
        args.static_evaluator
        or args.repo_root / "scripts/neurid/evaluate_mprt_static_atlas.py"
    ).resolve()
    args.cv_roots = {
        "atanas": args.atanas_cv_root.resolve(),
        "rld": args.rld_cv_root.resolve(),
    }
    args.datasets = tuple(item.strip().lower() for item in args.datasets.split(",") if item.strip())
    invalid = set(args.datasets) - set(DATASETS)
    if not args.datasets or invalid:
        raise ValueError(f"--datasets must be a subset of {DATASETS}; invalid={sorted(invalid)}")
    args.ablations = tuple(
        item.strip().lower() for item in args.ablations.split(",") if item.strip()
    )
    invalid_ablations = set(args.ablations) - set(ALL_ABLATIONS)
    if not args.ablations or invalid_ablations or len(set(args.ablations)) != len(args.ablations):
        raise ValueError(
            f"--ablations must be a unique subset of {ALL_ABLATIONS}; "
            f"invalid={sorted(invalid_ablations)}"
        )
    if not args.run_name or Path(args.run_name).name != args.run_name:
        raise ValueError("--run-name must be one non-empty directory name")
    if not args.package_root.is_dir():
        raise FileNotFoundError(args.package_root)
    if not args.static_evaluator.is_file():
        raise FileNotFoundError(
            f"Static evaluator not found: {args.static_evaluator}. "
            "Keep scripts/neurid/evaluate_mprt_static_atlas.py available or pass --static-evaluator."
        )

    util = _utilities(args.package_root)
    args.gpus = util.parse_csv(args.gpus)
    _validate_variant_semantics(args.package_root)

    tasks = [
        {"dataset": dataset, "fold": fold, "seed": seed}
        for dataset in args.datasets
        for fold in FOLDS
        for seed in SEEDS
    ]

    if args.phase == "preflight":
        _preflight(args, util)
        return

    if args.phase == "fit-lock":
        _parallel(tasks, args, util, _fit_cell)
        lock = _write_or_verify_lock(tasks, args, util)
        summary = _summarize(tasks, "val", args)
        _print_summary(summary)
        print(f"\nLOCKED: {lock}")
        print("Test has NOT been read. Inspect validation output before running locked-test.")
        return

    _verify_lock(tasks, args, util)
    if args.phase == "locked-test":
        for task in tasks:
            _require_npz(_fold_root(args, task["dataset"], task["fold"]), ("test",))
        _parallel(
            tasks,
            args,
            util,
            lambda task, gpu, a, u: _evaluate_cell(task, "test", gpu, a, u),
        )

    summary = _summarize(tasks, "test", args)
    _print_summary(summary)
    print(f"\nsummary={_root(args) / 'test_summary.json'}")


if __name__ == "__main__":
    main()
