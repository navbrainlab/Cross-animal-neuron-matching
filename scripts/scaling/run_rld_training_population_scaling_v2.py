#!/usr/bin/env python3
"""Publication-oriented RLD training-population scaling experiment.

Protocol
--------
* Keep the official RLD split fixed (67 train / 12 validation / 16 test worms).
* Evaluate training populations 5, 10, 20, 40, and 67.
* For every population below 67, create independent date-balanced nested subsets.
* Train multiple model initializations for every subset.
* Rebuild the anchored atlas from the selected training worms for every cell.
* Select/lock runs using validation data only; test evaluation is a separate phase.
* Aggregate Top-1 and Hungarian accuracy with hierarchical bootstrap 95% CIs.

This script intentionally separates ``fit-lock`` and ``locked-test``.  Do not run
``locked-test`` until training/validation decisions are final.

Expected repository layout and command-line interfaces match the current MPRT
repository under /home/ubuntu/klb/nuclr/nuclr.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_REPO = Path("/home/ubuntu/klb/nuclr/nuclr")
DEFAULT_DATA = DEFAULT_REPO / "Data/Dunn_001623/date_disjoint_full95_v1"
DEFAULT_PACKAGE = DEFAULT_REPO / "mprt_net_v1_1"
DEFAULT_SOURCE_RUN = DEFAULT_REPO / "runs/mprt_v1_1/rld/seed42/full"
DEFAULT_EVALUATOR = DEFAULT_REPO / "scripts/mprt/evaluate_mprt_static_atlas.py"
DEFAULT_RUN_ROOT = (
    DEFAULT_REPO / "runs/mprt_v1_1_rld_training_population_scaling_v2"
)

EXPECTED_SPLIT_COUNTS = {"train": 67, "val": 12, "test": 16}
DEFAULT_SIZES = (5, 10, 20, 40, 67)
DEFAULT_SUBSET_SEEDS = (20260825, 20260826, 20260827)
DEFAULT_MODEL_SEEDS = (1, 42, 123)

ACTIVITY_LENGTH = 512
BLEND_WEIGHT = 0.30
GATE_TEMPERATURE = 0.05


@dataclass(frozen=True)
class Cell:
    train_worms: int
    subset_seed: int | None
    subset_index: int
    model_seed: int

    @property
    def subset_tag(self) -> str:
        return "all" if self.subset_seed is None else f"subset{self.subset_seed}"

    @property
    def key(self) -> str:
        return f"n{self.train_worms}/{self.subset_tag}/seed{self.model_seed}"


def parse_int_csv(value: str) -> tuple[int, ...]:
    values = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError(f"duplicate values are not allowed: {value}")
    return values


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_logged(command: Sequence[str], cwd: Path, log_path: Path, gpu: str) -> None:
    """Run one command with an isolated CUDA_VISIBLE_DEVICES assignment."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    rendered = " ".join(str(x) for x in command)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"cwd: {cwd}\n")
        log.write(f"CUDA_VISIBLE_DEVICES: {gpu}\n")
        log.write(f"command: {rendered}\n\n")
        log.flush()
        result = subprocess.run(
            [str(x) for x in command],
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {result.returncode}; see {log_path}"
        )


def split_files(data_root: Path, split: str) -> list[Path]:
    root = data_root / split
    if not root.is_dir():
        raise FileNotFoundError(f"missing split directory: {root}")
    files = sorted(root.glob("*.npz"))
    if len(files) != EXPECTED_SPLIT_COUNTS[split]:
        raise RuntimeError(
            f"expected {EXPECTED_SPLIT_COUNTS[split]} {split} worms, "
            f"found {len(files)} in {root}"
        )
    duplicate_names = [name for name, n in Counter(x.name for x in files).items() if n > 1]
    if duplicate_names:
        raise RuntimeError(f"duplicate filenames in {root}: {duplicate_names}")
    return files


def date_group(path: Path) -> str:
    """Return the acquisition-date prefix used by the original scaling runner."""
    stem = path.stem
    return stem[:8] if len(stem) >= 8 else stem


def nested_date_balanced_order(files: Sequence[Path], seed: int) -> list[Path]:
    """Round-robin dates after deterministic within/between-date shuffling.

    Prefixes of the returned list are nested and approximately date-balanced.
    """
    rng = np.random.default_rng(seed)
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(files):
        groups[date_group(path)].append(path)

    group_names = sorted(groups)
    rng.shuffle(group_names)
    for name in group_names:
        rng.shuffle(groups[name])

    ordered: list[Path] = []
    round_index = 0
    while len(ordered) < len(files):
        active = [name for name in group_names if round_index < len(groups[name])]
        rng.shuffle(active)
        ordered.extend(groups[name][round_index] for name in active)
        round_index += 1
    if len(ordered) != len(files) or len({x.name for x in ordered}) != len(files):
        raise AssertionError("nested subset construction lost or duplicated worms")
    return ordered


def build_subset_map(
    train_files: Sequence[Path], sizes: Sequence[int], subset_seeds: Sequence[int]
) -> dict[tuple[int, int | None], list[Path]]:
    full_size = len(train_files)
    selections: dict[tuple[int, int | None], list[Path]] = {}
    for seed in subset_seeds:
        order = nested_date_balanced_order(train_files, seed)
        previous: set[str] = set()
        for size in sizes:
            if size == full_size:
                continue
            selected = order[:size]
            current = {x.name for x in selected}
            if not previous.issubset(current):
                raise AssertionError(f"subsets are not nested for seed {seed}, n={size}")
            previous = current
            selections[(size, seed)] = selected
    if full_size in sizes:
        selections[(full_size, None)] = list(sorted(train_files))
    return selections


def build_cells(
    sizes: Sequence[int], subset_seeds: Sequence[int], model_seeds: Sequence[int]
) -> list[Cell]:
    full_size = max(sizes)
    cells: list[Cell] = []
    for size in sizes:
        if size == full_size:
            for model_seed in model_seeds:
                cells.append(Cell(size, None, 0, model_seed))
        else:
            for subset_index, subset_seed in enumerate(subset_seeds):
                for model_seed in model_seeds:
                    cells.append(Cell(size, subset_seed, subset_index, model_seed))
    return cells


def dataset_root_for(run_root: Path, cell: Cell) -> Path:
    return run_root / "data" / f"n{cell.train_worms}" / cell.subset_tag


def cell_root(run_root: Path, cell: Cell) -> Path:
    return run_root / "cells" / f"n{cell.train_worms}" / cell.subset_tag / f"seed{cell.model_seed}"


def metric_path(run_root: Path, split: str, cell: Cell) -> Path:
    return (
        run_root
        / "metrics"
        / split
        / f"n{cell.train_worms}"
        / cell.subset_tag
        / f"seed{cell.model_seed}.json"
    )


def query_path(run_root: Path, split: str, cell: Cell) -> Path:
    return metric_path(run_root, split, cell).with_suffix(".queries.csv")


def safe_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise RuntimeError(
                f"existing symlink points to the wrong file: {destination} -> "
                f"{destination.resolve()} (expected {source.resolve()})"
            )
        return
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {destination}")
    destination.symlink_to(source.resolve())


def materialize_subset(
    run_root: Path,
    cell: Cell,
    selected_train: Sequence[Path],
    val_files: Sequence[Path],
    test_files: Sequence[Path] | None,
) -> Path:
    root = dataset_root_for(run_root, cell)
    for source in selected_train:
        safe_symlink(source, root / "train" / source.name)
    for source in val_files:
        safe_symlink(source, root / "val" / source.name)
    if test_files is not None:
        for source in test_files:
            safe_symlink(source, root / "test" / source.name)

    manifest_path = root / "subset_manifest.json"
    manifest = {
        "protocol": "fixed-rld-67-12-16",
        "train_worms": cell.train_worms,
        "subset_seed": cell.subset_seed,
        "nested_subset": cell.subset_seed is not None,
        "train_files": [x.name for x in selected_train],
        "train_date_counts": dict(
            sorted(Counter(date_group(x) for x in selected_train).items())
        ),
        "validation_files": [x.name for x in val_files],
    }
    if manifest_path.exists():
        existing = json_load(manifest_path)
        if existing != manifest:
            raise RuntimeError(f"subset manifest mismatch: {manifest_path}")
    else:
        json_dump(manifest_path, manifest)
    return root


def selected_for(
    subset_map: dict[tuple[int, int | None], list[Path]], cell: Cell
) -> list[Path]:
    return subset_map[(cell.train_worms, cell.subset_seed)]


def ensure_output_spec(path: Path, expected: dict[str, Any]) -> None:
    if path.exists():
        if json_load(path) != expected:
            raise RuntimeError(
                f"existing output was created by a different experiment spec: {path}"
            )
    else:
        json_dump(path, expected)


def load_orchestration(package_root: Path):
    # Repositories seen in this project use either
    #   <package_root>/mprt_net
    # or an installed package visible from <package_root>.parent.
    for candidate in (str(package_root), str(package_root.parent)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    try:
        from mprt_net.experiments import orchestration as util  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "cannot import mprt_net.experiments.orchestration; "
            f"checked package parent {package_root.parent}"
        ) from exc
    return util


def pairwise_command(
    package_root: Path,
    source_run: Path,
    dataset_root: Path,
    output_dir: Path,
    model_seed: int,
) -> list[str]:
    util = load_orchestration(package_root)
    parser = util.train_parser(package_root)
    source = util.complete_arguments(parser, util.source_arguments(source_run))
    values = dict(source)
    values.update(
        {
            "dataset_root": str(dataset_root),
            "output_dir": str(output_dir),
            "seed": model_seed,
            "variant": "full",
            "cycle_weight": 0.0,
            "atlas_weight": 0.0,
            "atlas_blend_weight": 0.0,
            "device": "cuda",
            "allow_existing_output": False,
        }
    )
    return [str(x) for x in util.command_from_values(parser, values)]


def fit_cell(args: argparse.Namespace, cell: Cell, dataset_root: Path, gpu: str) -> dict[str, Any]:
    root = cell_root(args.run_root, cell)
    pairwise_dir = root / "pairwise" / "full"
    pairwise_checkpoint = pairwise_dir / "best.pt"
    pure_atlas = root / "atlas" / "anchored_pure.pt"
    gated_atlas = root / "atlas" / "anchored_gated.pt"
    val_metrics = metric_path(args.run_root, "val", cell)

    spec = {
        "schema_version": 2,
        "cell": asdict(cell),
        "dataset_root": str(dataset_root.resolve()),
        "source_run": str(args.source_run.resolve()),
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "source_checkpoint_sha256": args.source_checkpoint_sha256,
        "activity_length": args.activity_length,
        "blend_weight": args.blend_weight,
        "gate_temperature": args.gate_temperature,
        "pairwise_cycle_weight": 0.0,
        "pairwise_atlas_weight": 0.0,
        "pairwise_atlas_blend_weight": 0.0,
    }
    ensure_output_spec(root / "experiment_spec.json", spec)

    if not pairwise_checkpoint.exists():
        if pairwise_dir.exists() and any(pairwise_dir.iterdir()):
            raise RuntimeError(
                f"partial pairwise output exists without best.pt; inspect {pairwise_dir}"
            )
        command = pairwise_command(
            args.package_root,
            args.source_run,
            dataset_root,
            pairwise_dir,
            cell.model_seed,
        )
        run_logged(command, args.package_root, root / "logs" / "train_pairwise.log", gpu)
    if not pairwise_checkpoint.is_file():
        raise FileNotFoundError(f"training did not produce {pairwise_checkpoint}")

    if not pure_atlas.exists():
        pure_atlas.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            "-m",
            "mprt_net.build_anchored_atlas",
            "--dataset-root",
            str(dataset_root),
            "--split",
            "train",
            "--checkpoint",
            str(pairwise_checkpoint),
            "--output",
            str(gated_atlas),
            "--pure-output",
            str(pure_atlas),
            "--activity-length",
            str(args.activity_length),
            "--blend-weight",
            str(args.blend_weight),
            "--gate-temperature",
            str(args.gate_temperature),
            "--device",
            "cuda",
        ]
        run_logged(command, args.package_root, root / "logs" / "build_atlas.log", gpu)
    if not pure_atlas.is_file():
        raise FileNotFoundError(f"atlas builder did not produce {pure_atlas}")

    val_queries = query_path(args.run_root, "val", cell)
    if val_metrics.exists() != val_queries.exists():
        raise RuntimeError(
            f"partial validation output for {cell.key}; expected both {val_metrics} "
            f"and {val_queries}"
        )
    if not val_metrics.exists():
        evaluate_cell(args, cell, dataset_root, pure_atlas, "val", gpu)
    return {
        "cell": asdict(cell),
        "pairwise_checkpoint": str(pairwise_checkpoint),
        "pairwise_sha256": sha256_file(pairwise_checkpoint),
        "static_checkpoint": str(pure_atlas),
        "static_sha256": sha256_file(pure_atlas),
        "val_metrics": str(val_metrics),
        "val_metrics_sha256": sha256_file(val_metrics),
        "val_queries": str(val_queries),
        "val_queries_sha256": sha256_file(val_queries),
        "subset_manifest": str(dataset_root / "subset_manifest.json"),
        "subset_manifest_sha256": sha256_file(dataset_root / "subset_manifest.json"),
    }


def evaluate_cell(
    args: argparse.Namespace,
    cell: Cell,
    dataset_root: Path,
    checkpoint: Path,
    split: str,
    gpu: str,
) -> None:
    output = metric_path(args.run_root, split, cell)
    queries = query_path(args.run_root, split, cell)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(args.static_evaluator),
        "--package-root",
        str(args.package_root),
        "--dataset-root",
        str(dataset_root),
        "--split",
        split,
        "--checkpoint",
        str(checkpoint),
        "--activity-length",
        str(args.activity_length),
        "--device",
        "cuda",
        "--dataset",
        "rld",
        "--fold",
        str(cell.subset_index),
        "--seed",
        str(cell.model_seed),
        "--variant",
        "full",
        "--output",
        str(output),
        "--query-output",
        str(queries),
    ]
    log = cell_root(args.run_root, cell) / "logs" / f"evaluate_{split}.log"
    run_logged(command, args.repo_root, log, gpu)
    if not output.is_file() or not queries.is_file():
        raise FileNotFoundError(
            f"evaluation did not produce both {output} and {queries}"
        )


def parallel_cells(
    cells: Sequence[Cell], gpus: Sequence[str], function
) -> list[Any]:
    if not gpus:
        raise ValueError("at least one GPU must be supplied")
    buckets: list[list[Cell]] = [[] for _ in gpus]
    for index, cell in enumerate(cells):
        buckets[index % len(gpus)].append(cell)

    def run_bucket(gpu: str, bucket: Sequence[Cell]) -> list[Any]:
        output: list[Any] = []
        for cell in bucket:
            print(f"[gpu {gpu}] starting {cell.key}", flush=True)
            output.append(function(cell, gpu))
            print(f"[gpu {gpu}] finished {cell.key}", flush=True)
        return output

    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(run_bucket, gpu, bucket)
            for gpu, bucket in zip(gpus, buckets)
            if bucket
        ]
        for future in as_completed(futures):
            results.extend(future.result())
    return results


def normalized_identity(value: Any) -> str | None:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if text.lower() in {"", "-1", "nan", "none", "null", "unknown", "unlabeled"}:
        return None
    return text


def identities_in_npz(path: Path) -> tuple[set[str], int]:
    with np.load(path, allow_pickle=False) as payload:
        label_key = next(
            (key for key in ("cell_id", "cell_id_alt", "cell_ids", "labels") if key in payload),
            None,
        )
        if label_key is None:
            raise KeyError(
                f"could not find an identity key in {path}; available keys={payload.files}"
            )
        labels = np.asarray(payload[label_key]).reshape(-1)
        if "labeled_mask" in payload:
            mask = np.asarray(payload["labeled_mask"]).reshape(-1).astype(bool)
            if mask.shape != labels.shape:
                raise ValueError(f"labeled_mask shape mismatch in {path}")
        else:
            mask = np.ones(labels.shape, dtype=bool)
    normalized = [normalized_identity(x) for x in labels[mask]]
    valid = [x for x in normalized if x is not None]
    return set(valid), len(valid)


def identity_coverage_audit(
    cells: Sequence[Cell],
    subset_map: dict[tuple[int, int | None], list[Path]],
    test_files: Sequence[Path],
) -> dict[str, Any]:
    test_queries: list[tuple[str, str]] = []
    test_identity_union: set[str] = set()
    for path in test_files:
        identities, _ = identities_in_npz(path)
        test_identity_union.update(identities)
        test_queries.extend((path.name, identity) for identity in identities)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int | None]] = set()
    for cell in cells:
        subset_key = (cell.train_worms, cell.subset_seed)
        if subset_key in seen:
            continue
        seen.add(subset_key)
        train_union: set[str] = set()
        for path in subset_map[subset_key]:
            identities, _ = identities_in_npz(path)
            train_union.update(identities)
        covered = sum(identity in train_union for _, identity in test_queries)
        rows.append(
            {
                "train_worms": cell.train_worms,
                "subset_seed": cell.subset_seed,
                "atlas_identity_count": len(train_union),
                "test_identity_count": len(test_identity_union),
                "test_query_identities": len(test_queries),
                "covered_test_query_identities": covered,
                "identity_coverage": covered / len(test_queries) if test_queries else None,
                "missing_test_identities": sorted(test_identity_union - train_union),
            }
        )
    return {
        "definition": (
            "Fraction of per-worm unique labeled test identities present in the union "
            "of identities from the selected training worms. This is descriptive only "
            "and is computed after the validation lock."
        ),
        "rows": sorted(rows, key=lambda x: (x["train_worms"], x["subset_seed"] or -1)),
    }


def protocol_payload(args: argparse.Namespace, cells: Sequence[Cell]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "name": "RLD training-population scaling",
        "fixed_split": {"train": 67, "val": 12, "test": 16},
        "sizes": list(args.sizes),
        "subset_seeds": list(args.subset_seeds),
        "model_seeds": list(args.model_seeds),
        "cells": [asdict(cell) for cell in cells],
        "number_of_training_runs": len(cells),
        "subset_rule": "date-balanced nested prefixes for n<67; one full set for n=67",
        "selection_rule": "validation only; test evaluation forbidden before lock",
        "optimization_rule": (
            "All cells inherit the same training arguments from source_run; only "
            "dataset_root, output_dir, seed, device and disabled auxiliary pairwise "
            "weights are changed. Thus epochs and pairs_per_epoch are held fixed."
        ),
        "data_root": str(args.data_root.resolve()),
        "package_root": str(args.package_root.resolve()),
        "source_run": str(args.source_run.resolve()),
        "static_evaluator": str(args.static_evaluator.resolve()),
        "activity_length": args.activity_length,
        "blend_weight": args.blend_weight,
        "gate_temperature": args.gate_temperature,
    }


def preflight(args: argparse.Namespace) -> tuple[list[Path], list[Path], list[Path], list[Cell], dict]:
    train_files = split_files(args.data_root, "train")
    val_files = split_files(args.data_root, "val")
    test_files = split_files(args.data_root, "test")
    split_names = {
        "train": {x.name for x in train_files},
        "val": {x.name for x in val_files},
        "test": {x.name for x in test_files},
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_names[left] & split_names[right]
        if overlap:
            raise RuntimeError(
                f"worm filenames overlap between {left} and {right}: {sorted(overlap)}"
            )
    if tuple(sorted(args.sizes)) != tuple(args.sizes):
        raise ValueError("--sizes must be strictly increasing")
    if args.sizes[-1] != len(train_files):
        raise ValueError(
            f"the largest training population must be {len(train_files)}, got {args.sizes[-1]}"
        )
    if any(size <= 0 or size > len(train_files) for size in args.sizes):
        raise ValueError(f"invalid sizes: {args.sizes}")
    for required in (
        args.package_root,
        args.source_run,
        args.source_run / "best.pt",
        args.static_evaluator,
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    args.source_checkpoint_sha256 = sha256_file(args.source_run / "best.pt")

    subset_map = build_subset_map(train_files, args.sizes, args.subset_seeds)
    cells = build_cells(args.sizes, args.subset_seeds, args.model_seeds)
    protocol = protocol_payload(args, cells)

    # Import and render one command to catch interface drift before launching GPUs.
    probe = cells[0]
    pairwise_command(
        args.package_root,
        args.source_run,
        dataset_root_for(args.run_root, probe),
        cell_root(args.run_root, probe) / "pairwise" / "full",
        probe.model_seed,
    )
    return train_files, val_files, test_files, cells, {"map": subset_map, "protocol": protocol}


def fit_and_lock(args: argparse.Namespace) -> None:
    _, val_files, _, cells, state = preflight(args)
    subset_map = state["map"]
    protocol = state["protocol"]
    args.run_root.mkdir(parents=True, exist_ok=True)
    protocol_path = args.run_root / "PROTOCOL.json"
    ensure_output_spec(protocol_path, protocol)

    dataset_roots: dict[str, Path] = {}
    for cell in cells:
        root = materialize_subset(
            args.run_root,
            cell,
            selected_for(subset_map, cell),
            val_files,
            test_files=None,
        )
        dataset_roots[cell.key] = root

    def worker(cell: Cell, gpu: str) -> dict[str, Any]:
        return fit_cell(args, cell, dataset_roots[cell.key], gpu)

    records = parallel_cells(cells, args.gpus, worker)
    records.sort(
        key=lambda x: (
            x["cell"]["train_worms"],
            x["cell"]["subset_seed"] or -1,
            x["cell"]["model_seed"],
        )
    )
    lock_payload = {
        "schema_version": 2,
        "status": "validation-selected-and-test-locked",
        "protocol_sha256": sha256_file(protocol_path),
        "records": records,
    }
    # Validate that every cell used the same validation queries before authorizing
    # any test evaluation.
    summarize(args, "val", cells)
    lock_path = args.run_root / "LOCKED_CHECKPOINTS.json"
    if lock_path.exists():
        if json_load(lock_path) != lock_payload:
            raise RuntimeError(
                f"existing lock differs from current artifacts: {lock_path}"
            )
    else:
        json_dump(lock_path, lock_payload)
    print(f"\nValidation lock written to {lock_path}")
    print("Test data have not been evaluated. Inspect validation outputs before locked-test.")


def verify_lock(args: argparse.Namespace, cells: Sequence[Cell]) -> dict[str, Any]:
    lock_path = args.run_root / "LOCKED_CHECKPOINTS.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"missing {lock_path}; run fit-lock before any test evaluation"
        )
    lock = json_load(lock_path)
    records = {
        (
            record["cell"]["train_worms"],
            record["cell"]["subset_seed"],
            record["cell"]["model_seed"],
        ): record
        for record in lock["records"]
    }
    expected_keys = {
        (cell.train_worms, cell.subset_seed, cell.model_seed) for cell in cells
    }
    if set(records) != expected_keys:
        raise RuntimeError("locked cell set does not match the requested experiment")
    protocol_path = args.run_root / "PROTOCOL.json"
    if sha256_file(protocol_path) != lock["protocol_sha256"]:
        raise RuntimeError("PROTOCOL.json changed after validation lock")
    for record in lock["records"]:
        for path_key, hash_key in (
            ("pairwise_checkpoint", "pairwise_sha256"),
            ("static_checkpoint", "static_sha256"),
            ("val_metrics", "val_metrics_sha256"),
            ("val_queries", "val_queries_sha256"),
            ("subset_manifest", "subset_manifest_sha256"),
        ):
            path = Path(record[path_key])
            if not path.is_file() or sha256_file(path) != record[hash_key]:
                raise RuntimeError(f"locked artifact changed or disappeared: {path}")
    return lock


def locked_test(args: argparse.Namespace) -> None:
    _, val_files, test_files, cells, state = preflight(args)
    subset_map = state["map"]
    lock = verify_lock(args, cells)
    record_by_key = {
        (
            record["cell"]["train_worms"],
            record["cell"]["subset_seed"],
            record["cell"]["model_seed"],
        ): record
        for record in lock["records"]
    }

    dataset_roots: dict[str, Path] = {}
    for cell in cells:
        dataset_roots[cell.key] = materialize_subset(
            args.run_root,
            cell,
            selected_for(subset_map, cell),
            val_files,
            test_files=test_files,
        )

    audit = identity_coverage_audit(cells, subset_map, test_files)
    json_dump(args.run_root / "analysis" / "identity_coverage_audit.json", audit)

    def worker(cell: Cell, gpu: str) -> str:
        output = metric_path(args.run_root, "test", cell)
        queries = query_path(args.run_root, "test", cell)
        if output.exists() != queries.exists():
            raise RuntimeError(
                f"partial test output for {cell.key}; expected both {output} and {queries}"
            )
        if output.exists() and queries.exists():
            return str(output)
        record = record_by_key[(cell.train_worms, cell.subset_seed, cell.model_seed)]
        evaluate_cell(
            args,
            cell,
            dataset_roots[cell.key],
            Path(record["static_checkpoint"]),
            "test",
            gpu,
        )
        return str(output)

    parallel_cells(cells, args.gpus, worker)
    summarize(args, "test", cells)


def metric_value(payload: dict[str, Any], candidates: Sequence[str]) -> float:
    for key in candidates:
        if key in payload:
            value = float(payload[key])
            return value / 100.0 if value > 1.0 else value
    raise KeyError(f"none of the metric keys {candidates} appear in {sorted(payload)}")


def bootstrap_hierarchical(
    rows: Sequence[dict[str, Any]],
    metric: str,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """Resample subset realizations, then model seeds within each realization."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["subset_tag"]].append(float(row[metric]))
    labels = sorted(grouped)
    rng = np.random.default_rng(seed)
    boot = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled_labels = rng.choice(labels, size=len(labels), replace=True)
        subset_means = []
        for label in sampled_labels:
            values = np.asarray(grouped[str(label)], dtype=np.float64)
            sampled_values = rng.choice(values, size=len(values), replace=True)
            subset_means.append(float(np.mean(sampled_values)))
        boot[index] = float(np.mean(subset_means))
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(low), float(high)


def read_query_signature(path: Path) -> tuple[str, ...]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row_index, row in enumerate(reader):
            # Exclude predictions/scores so the signature represents the evaluated queries.
            dynamic_tokens = (
                "pred",
                "score",
                "rank",
                "correct",
                "prob",
                "logit",
                "distance",
                "similarity",
                "top1",
                "top5",
                "mrr",
            )
            static_tokens = (
                "file",
                "worm",
                "sample",
                "query",
                "neuron",
                "index",
                "true",
                "ground",
                "label",
                "cell_id",
                "identity",
                "target",
            )
            identifying = {
                key: value
                for key, value in row.items()
                if any(token in key.lower() for token in static_tokens)
                and not any(token in key.lower() for token in dynamic_tokens)
            }
            if not identifying:
                identifying = {"row_index": str(row_index)}
            rows.append(json.dumps(identifying, sort_keys=True))
    return tuple(rows)


def summarize(args: argparse.Namespace, split: str, cells: Sequence[Cell]) -> None:
    rows: list[dict[str, Any]] = []
    query_counts: set[int] = set()
    query_signatures: set[str] = set()
    for cell in cells:
        path = metric_path(args.run_root, split, cell)
        qpath = query_path(args.run_root, split, cell)
        if not path.is_file() or not qpath.is_file():
            raise FileNotFoundError(f"missing {split} evaluation for {cell.key}")
        payload = json_load(path)
        top1 = metric_value(payload, ("top1_real", "top1", "top1_accuracy"))
        hungarian = metric_value(
            payload, ("hungarian_accuracy", "hungarian", "hungarian_acc")
        )
        queries = int(payload.get("queries", payload.get("num_queries", -1)))
        if queries >= 0:
            query_counts.add(queries)
        signature = stable_hash(read_query_signature(qpath))
        query_signatures.add(signature)
        rows.append(
            {
                "train_worms": cell.train_worms,
                "subset_seed": cell.subset_seed,
                "subset_tag": cell.subset_tag,
                "model_seed": cell.model_seed,
                "queries": queries,
                "top1": top1,
                "hungarian": hungarian,
                "metrics_path": str(path),
                "query_path": str(qpath),
            }
        )
    if len(query_counts) > 1:
        raise RuntimeError(f"query counts changed across cells: {sorted(query_counts)}")
    if len(query_signatures) > 1:
        raise RuntimeError(
            "the evaluated query set/order changed across cells; refusing to aggregate"
        )

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["train_worms"]].append(row)
    summaries: list[dict[str, Any]] = []
    for size in sorted(grouped):
        current = grouped[size]
        summary: dict[str, Any] = {
            "train_worms": size,
            "cells": len(current),
            "subset_realizations": len({x["subset_tag"] for x in current}),
            "model_seeds": sorted({x["model_seed"] for x in current}),
        }
        for metric in ("top1", "hungarian"):
            values = np.asarray([x[metric] for x in current], dtype=np.float64)
            low, high = bootstrap_hierarchical(
                current,
                metric,
                args.bootstrap_iterations,
                args.bootstrap_seed + size + (0 if metric == "top1" else 10000),
            )
            summary.update(
                {
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_sd_cells": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0,
                    f"{metric}_ci95_low": low,
                    f"{metric}_ci95_high": high,
                }
            )
        summaries.append(summary)

    analysis_dir = args.run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    json_dump(
        analysis_dir / f"{split}_scaling_summary.json",
        {
            "split": split,
            "ci_definition": (
                "95% percentile hierarchical bootstrap: resample subset realizations, "
                "then model seeds within each sampled subset. For n=67, resample the "
                "three model seeds. The fixed test cohort is not bootstrapped."
            ),
            "bootstrap_iterations": args.bootstrap_iterations,
            "rows": rows,
            "summary": summaries,
        },
    )
    write_summary_csv(analysis_dir / f"{split}_scaling_summary.csv", summaries)
    write_raw_csv(analysis_dir / f"{split}_scaling_raw.csv", rows)
    plot_scaling(analysis_dir, split, summaries)

    print(f"\n{split.upper()} training-population scaling")
    print("worms  cells  Top-1 mean [95% CI]    Hungarian mean [95% CI]")
    for item in summaries:
        print(
            f"{item['train_worms']:>5}  {item['cells']:>5}  "
            f"{100*item['top1_mean']:6.2f}% "
            f"[{100*item['top1_ci95_low']:6.2f}, {100*item['top1_ci95_high']:6.2f}]  "
            f"{100*item['hungarian_mean']:6.2f}% "
            f"[{100*item['hungarian_ci95_low']:6.2f}, "
            f"{100*item['hungarian_ci95_high']:6.2f}]"
        )


def write_summary_csv(path: Path, summaries: Sequence[dict[str, Any]]) -> None:
    fieldnames = [
        "train_worms",
        "cells",
        "subset_realizations",
        "model_seeds",
        "top1_mean",
        "top1_sd_cells",
        "top1_ci95_low",
        "top1_ci95_high",
        "hungarian_mean",
        "hungarian_sd_cells",
        "hungarian_ci95_low",
        "hungarian_ci95_high",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            record = dict(row)
            record["model_seeds"] = ",".join(str(x) for x in row["model_seeds"])
            writer.writerow({key: record[key] for key in fieldnames})


def write_raw_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_scaling(
    analysis_dir: Path, split: str, summaries: Sequence[dict[str, Any]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = np.asarray([x["train_worms"] for x in summaries], dtype=float)
    means = 100 * np.asarray([x["top1_mean"] for x in summaries])
    lows = 100 * np.asarray([x["top1_ci95_low"] for x in summaries])
    highs = 100 * np.asarray([x["top1_ci95_high"] for x in summaries])
    errors = np.vstack((means - lows, highs - means))

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.errorbar(
        sizes,
        means,
        yerr=errors,
        color="#1769aa",
        marker="o",
        markersize=6,
        linewidth=2,
        capsize=3,
        elinewidth=1.2,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes, [str(int(x)) for x in sizes])
    ax.set_xlabel("Number of training worms")
    ax.set_ylabel("Top-1 accuracy (%)")
    ax.grid(True, which="major", axis="both", alpha=0.22, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"RLD training-population scaling ({split})")
    fig.tight_layout()
    stem = analysis_dir / f"{split}_top1_vs_training_worms"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Run the complete fixed-split RLD training-population scaling study.",
    )
    parser.add_argument(
        "phase", choices=("preflight", "fit-lock", "locked-test", "summarize")
    )
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--static-evaluator", type=Path, default=DEFAULT_EVALUATOR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--sizes", type=parse_int_csv, default=DEFAULT_SIZES)
    parser.add_argument(
        "--subset-seeds", type=parse_int_csv, default=DEFAULT_SUBSET_SEEDS
    )
    parser.add_argument(
        "--model-seeds", type=parse_int_csv, default=DEFAULT_MODEL_SEEDS
    )
    parser.add_argument(
        "--gpus",
        type=lambda x: tuple(v.strip() for v in x.split(",") if v.strip()),
        default=("0",),
        help="physical GPU IDs; each GPU runs one experiment cell at a time",
    )
    parser.add_argument("--activity-length", type=int, default=ACTIVITY_LENGTH)
    parser.add_argument("--blend-weight", type=float, default=BLEND_WEIGHT)
    parser.add_argument("--gate-temperature", type=float, default=GATE_TEMPERATURE)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    parser.add_argument(
        "--summary-split",
        choices=("val", "test"),
        default="test",
        help="used only by the summarize phase",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.repo_root = args.repo_root.resolve()
    args.data_root = args.data_root.resolve()
    args.package_root = args.package_root.resolve()
    args.source_run = args.source_run.resolve()
    args.static_evaluator = args.static_evaluator.resolve()
    args.run_root = args.run_root.resolve()

    if args.phase == "preflight":
        train, val, test, cells, state = preflight(args)
        print(json.dumps(state["protocol"], indent=2))
        print(
            f"\nPreflight passed: {len(train)}/{len(val)}/{len(test)} worms; "
            f"{len(cells)} training cells; GPUs={','.join(args.gpus)}"
        )
    elif args.phase == "fit-lock":
        fit_and_lock(args)
    elif args.phase == "locked-test":
        locked_test(args)
    elif args.phase == "summarize":
        _, _, _, cells, _ = preflight(args)
        if args.summary_split == "test":
            verify_lock(args, cells)
        summarize(args, args.summary_split, cells)
    else:
        raise AssertionError(args.phase)


if __name__ == "__main__":
    main()
