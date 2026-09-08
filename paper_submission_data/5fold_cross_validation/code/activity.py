#!/usr/bin/env python3
"""Raw NuCLR: joint same-worm + cross-worm DCL activity training.

This is the canonical activity branch used by Hybrid. It locks validation
checkpoint selection to the encoder space.
The projected space is still reported as a diagnostic and is still used by
the DCL loss, but it can no longer select ``best.pt`` or trigger early stopping.

This unified-data variant uses every finite tracked neuron for same-worm temporal
self-supervision, while cross-worm identity supervision and evaluation are restricted
to the NPZ ``clean_mask``. Activity, coordinates, and labels originate from the same
NWB aligned-neuron table; this script consumes activity only.

The production input is ``raw_nuclr``: raw 30-second activity windows are sent
to the official NuCLR backbone. Historical feature/input ablations are retained
under ``archive/model_variants/activity_feature_ablations.py``.

The default training objective is

    L = lambda_same * L_same_worm + lambda_cross * L_cross_worm_DCL

Set ``--lambda-same 0 --batch-size 1 --cross-views-per-pair 1`` for
the cross-worm-only experiment with exactly two population views per optimizer
step. ``--identity-batch-size B`` then controls the number of aligned,
canonical identities in the B x B contrastive problem.

Use ``--init-full-checkpoint`` to fine-tune an established checkpoint while
preserving both its encoder and DCL projector. The optimizer, scheduler, epoch,
global step, and early-stopping state always start fresh, so checkpoint
selection for the new run remains validation-only.

where both terms use the same decoupled sample-wise contrastive formulation.
Same-worm positives are the same tracked local neuron in two temporal views.
Cross-worm positives are neurons with the same canonical ``cell_id``.

The script intentionally never reads a test split. Checkpoint selection uses only
completely unseen validation worms. Hand-crafted feature scaling is fitted on
train worms only and then applied to validation worms.

Expected companion files in the NuCLR repository root:

* archive/legacy_experiments/ey/stage1_pretrain_ey_nuclr_official_50k.py
* diagnostics/gt_guided_activity_feature_audit_atanas.py

The official NuCLR backbone is reused. A small general DCL module is implemented
here because fused models have a different encoder output interface while the
loss formula remains the paper's DCL objective (tau, symmetric directions,
projector, and optional same-view negatives/full denominator).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# These helpers live in organized subdirectories but are imported by module
# name throughout this training script and by checkpoint-evaluation loaders.
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
for _helper_dir in (
    _SCRIPT_ROOT / "archive" / "legacy_experiments" / "ey",
    _SCRIPT_ROOT / "diagnostics",
):
    if str(_helper_dir) not in sys.path:
        sys.path.insert(0, str(_helper_dir))

import stage1_pretrain_ey_nuclr_official_50k as s1
import gt_guided_activity_feature_audit_atanas as audit
from src.nn.qasa_rotary_attention import (
    QASA_VALUE_MODES,
    install_last_temporal_qasa_value,
    is_qasa_value_parameter,
)


EPS = 1e-8
INVALID_LABELS = {
    "", "-1", "nan", "none", "null", "na", "n/a", "unknown",
    "unlabeled", "unlabelled", "invalid", "?",
}
DYNAMIC_VARIANTS = {
    "dynamic_only",
    "raw_nuclr",
    "dynamic_absolute",
    "dynamic_absolute_population",
}
ABSOLUTE_VARIANTS = {
    "absolute_only",
    "dynamic_absolute",
    "dynamic_absolute_population",
}
POPULATION_VARIANTS = {"dynamic_absolute_population"}


@dataclass
class AtanasRecord:
    worm_id: str
    path: Path
    traces_raw: np.ndarray
    traces_z: np.ndarray
    labels: np.ndarray
    valid_unique_indices: np.ndarray
    same_worm_indices: np.ndarray
    source_fs: float
    identity_labels_loaded: bool = True
    absolute_features: np.ndarray | None = None
    population_features: np.ndarray | None = None

    @property
    def num_neurons(self) -> int:
        return int(self.traces_raw.shape[0])

    @property
    def num_timepoints(self) -> int:
        return int(self.traces_raw.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.num_timepoints / self.source_fs


@dataclass(frozen=True)
class SameSample:
    record: AtanasRecord
    start1: float
    start2: float


@dataclass(frozen=True)
class CrossSample:
    record1: AtanasRecord
    record2: AtanasRecord
    start1: float
    start2: float


@dataclass
class PreparedViews:
    bins1: Tensor | None
    bins2: Tensor | None
    lengths1: Tensor
    lengths2: Tensor
    absolute1: Tensor | None
    absolute2: Tensor | None
    population1: Tensor | None
    population2: Tensor | None
    matches: list[Tensor]
    num_matches: int


@dataclass
class RetrievalAccumulator:
    correct1: int = 0
    correct5: int = 0
    reciprocal_rank_sum: float = 0.0
    queries: int = 0
    positive_sum: float = 0.0
    negative_sum: float = 0.0
    negative_count: int = 0

    @property
    def top1(self) -> float:
        return self.correct1 / max(self.queries, 1)

    @property
    def top5(self) -> float:
        return self.correct5 / max(self.queries, 1)

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank_sum / max(self.queries, 1)

    @property
    def gap(self) -> float:
        if self.queries == 0 or self.negative_count == 0:
            return float("nan")
        return self.positive_sum / self.queries - self.negative_sum / self.negative_count


# -----------------------------------------------------------------------------
# CLI and utilities
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Raw NuCLR activity encoder with same+cross-worm DCL."
    )
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--val-root", type=Path, required=True)
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument(
        "--test-worm-ids-file",
        type=Path,
        default=None,
        help=(
            "Optional ID-only manifest used solely for split assertions. "
            "No test data root is accepted or opened."
        ),
    )
    p.add_argument("--expected-train-worms", type=int, default=0)
    p.add_argument("--expected-val-worms", type=int, default=0)
    p.add_argument("--expected-test-worms", type=int, default=0)
    p.add_argument("--split-manifest", type=Path, default=None)
    p.add_argument("--split-manifest-sha256", default="")
    p.add_argument(
        "--model-variant",
        choices=["raw_nuclr"],
        default="raw_nuclr",
        help="Canonical Hybrid activity input (the only production choice).",
    )
    p.add_argument("--activity-key", type=str, default="activity_raw")
    p.add_argument("--label-key", type=str, default="cell_id")
    p.add_argument(
        "--supervision-mask-key",
        type=str,
        default="clean_mask",
        help=(
            "Boolean NPZ mask defining neurons eligible for cross-worm identity "
            "supervision and evaluation. If absent, unique non-empty labels are used."
        ),
    )
    p.add_argument(
        "--same-worm-neurons",
        choices=["all", "clean"],
        default="all",
        help=(
            "Use every finite tracked neuron or only clean labeled neurons for the "
            "same-worm two-view DCL objective. Cross-worm DCL always uses clean labels."
        ),
    )
    p.add_argument("--source-fs", type=float, default=4.0)

    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument(
        "--init-backbone-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional same-worm NuCLR checkpoint. Only model_state_dict is loaded; "
            "the new shared DCL projector and feature branches start fresh."
        ),
    )
    initialization.add_argument(
        "--init-full-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional checkpoint from this trainer. Load both the complete encoder "
            "model_state_dict and DCL loss_state_dict/projector, but start a fresh "
            "optimizer, schedule, global step, and validation selection."
        ),
    )
    p.add_argument(
        "--external-init-checkpoint-sha256",
        default="",
        help=(
            "Exact SHA256 required to permit a backbone-only initialization in a "
            "locked outer-fold run. The checkpoint must be independently pretrained."
        ),
    )
    p.add_argument(
        "--external-init-provenance",
        default="",
        help="Human-readable independent-dataset provenance for the locked initializer.",
    )
    p.add_argument(
        "--audited-init-scope",
        choices=["external", "outer_train_only"],
        default="external",
        help=(
            "Scope of the hash-locked backbone initializer. outer_train_only is "
            "valid only for a checkpoint trained inside the same outer fold."
        ),
    )
    p.add_argument(
        "--last-temporal-value-mode",
        choices=sorted(QASA_VALUE_MODES),
        default="classical",
        help=(
            "Direct V projection used only in the final temporal attention "
            "executed by the NuCLR activity encoder."
        ),
    )
    p.add_argument("--last-temporal-value-qubits", type=int, default=4)
    p.add_argument("--last-temporal-value-depth", type=int, default=2)
    p.add_argument("--last-temporal-value-chunk-size", type=int, default=256)
    p.add_argument(
        "--qasa-lr",
        type=float,
        default=2e-4,
        help="Learning rate for the newly inserted temporal V bottleneck.",
    )
    p.add_argument(
        "--qasa-freeze-backbone-epochs",
        type=int,
        default=3,
        help=(
            "For a non-classical V extension, initially train only the new "
            "value bottleneck for this many epochs."
        ),
    )
    p.add_argument(
        "--qasa-continuation-control",
        action="store_true",
        help=(
            "Explicitly permit the unchanged classical NuCLR to continue "
            "training from the same source checkpoint as an extra-epochs control."
        ),
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument(
        "--identity-batch-size",
        type=int,
        default=0,
        help=(
            "Maximum number B of shared clean identities encoded per cross-worm "
            "sample. 0 preserves the original behavior. When positive, the two "
            "views contain the same sampled identities in aligned order, giving "
            "an exact B x B cross-worm contrastive problem."
        ),
    )
    p.add_argument("--same-views-per-worm", type=int, default=8)
    p.add_argument(
        "--cross-pairs-per-epoch",
        type=int,
        default=0,
        help="0 uses all unordered train-worm pairs each epoch.",
    )
    p.add_argument("--cross-views-per-pair", type=int, default=2)
    p.add_argument(
        "--optimizer-steps-per-epoch",
        type=int,
        default=0,
        help=(
            "0 derives the epoch length from the active objectives. A positive "
            "value fixes the optimizer-step budget so objective ablations use "
            "the same schedule and number of updates per epoch."
        ),
    )
    p.add_argument(
        "--cross-window-mode",
        choices=["independent", "aligned_quantile"],
        default="independent",
        help=(
            "Atanas behaviors are not synchronized. independent samples unrelated "
            "time windows; aligned_quantile uses the same normalized recording position."
        ),
    )
    p.add_argument("--window-seconds", type=float, default=30.0)
    p.add_argument("--eval-num-windows", type=int, default=8)
    p.add_argument("--feature-num-windows", type=int, default=8)
    p.add_argument("--feature-resample-points", type=int, default=256)

    p.add_argument(
        "--unit-dropout",
        choices=["official", "none"],
        default="official",
    )
    p.add_argument("--minimum-positive-matches", type=int, default=8)

    p.add_argument(
        "--lambda-same",
        type=float,
        default=1.0,
        help=(
            "Weight of the same-worm temporal objective. Set to 0 for the "
            "cross-worm-only two-view experiment."
        ),
    )
    p.add_argument("--lambda-cross", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--projector-dim", type=int, default=128)
    p.add_argument("--full-denom", action=argparse.BooleanOptionalAction, default=True)

    # Explicit architecture knobs. Defaults reproduce the original T2/ST2
    # calcium backbone; fold-pure Compact NuCLR uses T1/ST1 without truncating
    # or loading any checkpoint trained outside the outer fold.
    p.add_argument("--nuclr-patch-size", type=int, default=4)
    p.add_argument("--nuclr-dim", type=int, default=256)
    p.add_argument("--nuclr-heads", type=int, default=4)
    p.add_argument("--nuclr-dim-head", type=int, default=64)
    p.add_argument("--nuclr-temporal-layers", type=int, default=2)
    p.add_argument("--nuclr-spatiotemporal-layers", type=int, default=2)
    p.add_argument("--nuclr-attention-dropout", type=float, default=0.0)
    p.add_argument("--nuclr-linear-dropout", type=float, default=0.2)
    p.add_argument("--nuclr-rot-ratio", type=float, default=0.5)

    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--branch-lr", type=float, default=1e-4)
    p.add_argument("--projector-lr", type=float, default=1.25e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--feature-dropout", type=float, default=0.1)

    p.add_argument("--val-every-epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--save-every-epochs", type=int, default=10)
    p.add_argument(
        "--selection-space",
        choices=["encoder"],
        default="encoder",
        help=(
            "V2 is locked to encoder-space validation selection. Projected "
            "metrics are reported only as diagnostics."
        ),
    )
    p.add_argument(
        "--selection-metric",
        choices=["top1", "top5", "mrr"],
        default="top1",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be >= 1")
    if args.identity_batch_size < 0:
        p.error("--identity-batch-size must be >= 0")
    if args.identity_batch_size > 0:
        if args.identity_batch_size < 2:
            p.error("--identity-batch-size must be 0 or >= 2")
        if args.identity_batch_size < args.minimum_positive_matches:
            p.error(
                "--identity-batch-size must be >= --minimum-positive-matches"
            )
    if args.lambda_same < 0 or args.lambda_cross < 0:
        p.error("--lambda-same and --lambda-cross must be non-negative")
    if args.lambda_same == 0 and args.lambda_cross == 0:
        p.error("At least one of --lambda-same/--lambda-cross must be positive")
    if args.optimizer_steps_per_epoch < 0:
        p.error("--optimizer-steps-per-epoch must be >= 0")
    if not 2 <= args.last_temporal_value_qubits <= 12:
        p.error("--last-temporal-value-qubits must be in [2,12]")
    if args.last_temporal_value_depth < 1:
        p.error("--last-temporal-value-depth must be >= 1")
    if args.last_temporal_value_chunk_size < 1:
        p.error("--last-temporal-value-chunk-size must be >= 1")
    if args.qasa_lr <= 0:
        p.error("--qasa-lr must be positive")
    if args.qasa_freeze_backbone_epochs < 0:
        p.error("--qasa-freeze-backbone-epochs must be non-negative")
    if bool(args.external_init_checkpoint_sha256) != bool(args.external_init_provenance):
        p.error("external initializer SHA256 and provenance must be provided together")
    if args.external_init_checkpoint_sha256 and args.init_backbone_checkpoint is None:
        p.error("external initializer audit is valid only with --init-backbone-checkpoint")
    if args.nuclr_patch_size < 1 or args.nuclr_dim < 1:
        p.error("NuCLR patch size and dimension must be positive")
    if args.nuclr_heads < 1 or args.nuclr_dim_head < 1:
        p.error("NuCLR head count and head dimension must be positive")
    if args.nuclr_temporal_layers < 0 or args.nuclr_spatiotemporal_layers < 0:
        p.error("NuCLR layer counts must be non-negative")
    if (
        args.qasa_continuation_control
        and args.last_temporal_value_mode != "classical"
    ):
        p.error(
            "--qasa-continuation-control is valid only with "
            "--last-temporal-value-mode classical"
        )
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def worm_ids_from_npz_root(root: Path) -> set[str]:
    return {path.stem for path in root.rglob("*.npz")}


def worm_ids_from_manifest(path: Path) -> set[str]:
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token or token.startswith("#"):
            continue
        ids.add(Path(token).stem)
    return ids


def assert_clean_split_protocol(args: argparse.Namespace) -> dict[str, Any]:
    train_ids = worm_ids_from_npz_root(args.train_root)
    val_ids = worm_ids_from_npz_root(args.val_root)
    test_ids = (
        worm_ids_from_manifest(args.test_worm_ids_file)
        if args.test_worm_ids_file is not None
        else set()
    )
    observed = {
        "train": len(train_ids),
        "val": len(val_ids),
        "test": len(test_ids),
    }
    expected = {
        "train": int(args.expected_train_worms),
        "val": int(args.expected_val_worms),
        "test": int(args.expected_test_worms),
    }
    for split, expected_count in expected.items():
        if expected_count > 0 and observed[split] != expected_count:
            raise AssertionError(
                f"{split}: expected {expected_count} worms, found {observed[split]}"
            )
    if args.test_worm_ids_file is not None and expected["test"] <= 0:
        raise AssertionError(
            "--expected-test-worms must be positive with --test-worm-ids-file"
        )

    intersections = {
        "train_val": sorted(train_ids & val_ids),
        "train_test": sorted(train_ids & test_ids),
        "val_test": sorted(val_ids & test_ids),
    }
    overlaps = {key: value for key, value in intersections.items() if value}
    if overlaps:
        raise AssertionError(f"Worm-level split overlap: {overlaps}")

    audit = {
        "train_worms": sorted(train_ids),
        "val_worms": sorted(val_ids),
        "test_worm_ids_assertion_only": sorted(test_ids),
        "counts": observed,
        "intersections": intersections,
        "test_data_root_passed_to_training": False,
        "test_worms_referenced_by_training_dataloader": 0,
    }
    print("=" * 100)
    print("Clean worm-level split assertions")
    print(f"NuCLR train worms: {observed['train']}")
    print(f"NuCLR val worms: {observed['val']}")
    print("NuCLR test worms referenced during training: 0")
    print("train ∩ val  = empty")
    print("train ∩ test = empty")
    print("val ∩ test   = empty")
    print("=" * 100)
    return audit


def clean_identity_statistics(
    records: Sequence[AtanasRecord],
) -> dict[str, float | int]:
    counts = np.asarray(
        [len(record.valid_unique_indices) for record in records], dtype=np.int64
    )
    return {
        "worms": int(len(records)),
        "mean": float(counts.mean()),
        "min": int(counts.min()),
        "max": int(counts.max()),
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def iter_batches(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def infinite_cycle(values: Sequence[Any]) -> Iterable[Any]:
    while True:
        yield from values


def autocast_context(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


def normalize_label(value: Any) -> str:
    text = str(value).strip()
    if text.lower() in INVALID_LABELS:
        return ""
    return text


def unique_valid_indices(labels: np.ndarray) -> np.ndarray:
    normalized = np.asarray([normalize_label(x) for x in labels], dtype=object)
    counts = Counter(x for x in normalized if x)
    return np.asarray(
        [i for i, x in enumerate(normalized) if x and counts[x] == 1],
        dtype=np.int64,
    )


def row_zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mean = values.mean(axis=1, keepdims=True)
    std = values.std(axis=1, keepdims=True)
    return ((values - mean) / np.maximum(std, EPS)).astype(np.float32)


# -----------------------------------------------------------------------------
# Data loading and exact audit feature extraction
# -----------------------------------------------------------------------------


def infer_activity_orientation(activity: np.ndarray, n_labels: int) -> np.ndarray:
    if activity.ndim != 2:
        raise ValueError(f"Expected 2-D activity, got {activity.shape}")
    if activity.shape[0] == n_labels:
        return activity
    if activity.shape[1] == n_labels:
        return activity.T
    raise ValueError(
        f"Neither activity axis matches labels: activity={activity.shape}, labels={n_labels}"
    )


def infer_unlabelled_activity_orientation(activity: np.ndarray) -> np.ndarray:
    """Orient activity without opening identity fields.

    Atanas recordings contain many more time points than tracked neurons.
    Requiring that invariant makes the Same-only loader independent of
    ``cell_id`` and ``clean_mask`` while still failing loudly on ambiguity.
    """
    if activity.ndim != 2:
        raise ValueError(f"Expected 2-D activity, got {activity.shape}")
    if activity.shape[0] == activity.shape[1]:
        raise ValueError(
            f"Cannot infer unlabelled activity orientation from square array {activity.shape}"
        )
    neuron_axis = int(np.argmin(activity.shape))
    oriented = activity if neuron_axis == 0 else activity.T
    if oriented.shape[1] <= oriented.shape[0]:
        raise ValueError(
            "Unlabelled orientation invariant failed: expected timepoints > neurons, "
            f"got {oriented.shape}"
        )
    return oriented


def scalar_from_npz(npz: Any, key: str, fallback: float) -> float:
    candidate_keys = [key]
    if key == "source_fs":
        candidate_keys.append("sampling_rate_hz")
    for candidate in candidate_keys:
        if candidate not in npz.files:
            continue
        value = np.asarray(npz[candidate]).reshape(-1)
        if value.size == 0:
            continue
        try:
            result = float(value[0])
        except (TypeError, ValueError):
            continue
        if math.isfinite(result) and result > 0:
            return result
    return float(fallback)


def load_records(
    root: Path,
    split: str,
    activity_key: str,
    label_key: str,
    fallback_fs: float,
    args: argparse.Namespace,
    *,
    load_identity_labels: bool = True,
) -> list[AtanasRecord]:
    paths = sorted(root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ files found under {root}")
    records: list[AtanasRecord] = []
    for path in paths:
        with np.load(path, allow_pickle=True) as npz:
            if activity_key not in npz.files:
                continue
            raw_activity = np.asarray(npz[activity_key], dtype=np.float32)
            if load_identity_labels:
                if label_key not in npz.files:
                    continue
                labels = np.asarray(npz[label_key]).reshape(-1).astype(str)
                activity = infer_activity_orientation(raw_activity, len(labels))
            else:
                # Deliberately do not index either the identity or supervision
                # mask key. This is the auditable Same-only data path.
                activity = infer_unlabelled_activity_orientation(raw_activity)
                labels = np.full(activity.shape[0], "", dtype=object)
            fs = scalar_from_npz(npz, "source_fs", fallback_fs)
            if load_identity_labels and args.supervision_mask_key in npz.files:
                supervision_mask = np.asarray(
                    npz[args.supervision_mask_key], dtype=bool
                ).reshape(-1)
                if len(supervision_mask) != len(labels):
                    raise ValueError(
                        f"{path.name}: {args.supervision_mask_key} length "
                        f"{len(supervision_mask)} != labels {len(labels)}"
                    )
            else:
                supervision_mask = None
        finite_mask = np.all(np.isfinite(activity), axis=1)
        activity = np.nan_to_num(activity, nan=0.0, posinf=0.0, neginf=0.0)
        unique_idx = unique_valid_indices(labels)
        unique_mask = np.zeros(len(labels), dtype=bool)
        unique_mask[unique_idx] = True
        if supervision_mask is None:
            clean_mask = unique_mask
        else:
            # The saved clean_mask is authoritative, while this extra uniqueness
            # check prevents malformed files from introducing duplicate identities.
            clean_mask = supervision_mask & unique_mask
        clean_mask &= finite_mask
        valid_idx = np.flatnonzero(clean_mask).astype(np.int64)
        if args.same_worm_neurons == "all":
            same_idx = np.flatnonzero(finite_mask).astype(np.int64)
        else:
            same_idx = valid_idx.copy()
        if len(same_idx) < args.minimum_positive_matches:
            print(f"skip {path.name}: same-worm neurons={len(same_idx)}")
            continue
        # Unlabelled train animals are still valid same-worm self-supervised
        # examples when every finite tracked neuron is requested.  They must
        # never enter cross-worm supervision or validation/test retrieval.
        same_only = len(valid_idx) < args.minimum_positive_matches
        if same_only and not (split == "train" and args.same_worm_neurons == "all"):
            print(f"skip {path.name}: clean supervised labels={len(valid_idx)}")
            continue

        # Reuse the exact feature extraction used by the successful GT audit.
        mask = np.zeros(len(labels), dtype=bool)
        mask[valid_idx] = True
        audit_record = audit.WormRecord(
            split=split,
            worm_id=f"{split}:{len(records):04d}:{path.stem}",
            path=path,
            traces_raw=activity,
            labels=labels,
            valid_unique_mask=mask,
            source_fs=fs,
        )
        audit.extract_features(
            audit_record,
            resample_points=args.feature_resample_points,
            window_seconds=args.window_seconds,
            stride_seconds=args.window_seconds,
            num_windows=args.feature_num_windows,
            window_grid_mode="uniform_native",
        )
        records.append(
            AtanasRecord(
                worm_id=audit_record.worm_id,
                path=path,
                traces_raw=activity.astype(np.float32),
                traces_z=row_zscore(activity),
                labels=np.asarray([normalize_label(x) for x in labels]),
                valid_unique_indices=valid_idx,
                same_worm_indices=same_idx,
                source_fs=fs,
                identity_labels_loaded=load_identity_labels,
                absolute_features=np.asarray(
                    audit_record.group_features["absolute"], dtype=np.float32
                ),
                population_features=np.asarray(
                    audit_record.group_features["population"], dtype=np.float32
                ),
            )
        )
        print(
            f"loaded {split} {len(records):02d}: {path.name} "
            f"neurons={activity.shape[0]} clean={len(valid_idx)} "
            f"same_worm={len(same_idx)} same_only={same_only} "
            f"duration={activity.shape[1]/fs:.1f}s"
        )
    if len(records) < 2:
        raise RuntimeError(f"Need at least two usable {split} worms, got {len(records)}")
    return records


def fit_and_apply_feature_scaling(
    train_records: Sequence[AtanasRecord],
    val_records: Sequence[AtanasRecord],
) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for name in ("absolute_features", "population_features"):
        train_values = np.concatenate(
            [
                getattr(r, name)[
                    r.valid_unique_indices
                    if r.identity_labels_loaded
                    else r.same_worm_indices
                ]
                for r in train_records
            ],
            axis=0,
        ).astype(np.float64)
        mean = train_values.mean(axis=0)
        std = train_values.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        for record in itertools.chain(train_records, val_records):
            values = getattr(record, name)
            scaled = ((values - mean) / std).astype(np.float32)
            setattr(record, name, np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0))
        result[name] = {"mean": mean.tolist(), "std": std.tolist()}
    return result


# -----------------------------------------------------------------------------
# Window sampling and official NuCLR tensorization
# -----------------------------------------------------------------------------


def max_window_start(record: AtanasRecord, window_seconds: float) -> float:
    return max(0.0, record.duration_seconds - window_seconds)


def random_start(record: AtanasRecord, window_seconds: float, rng: np.random.Generator) -> float:
    maximum = max_window_start(record, window_seconds)
    return float(rng.uniform(0.0, maximum)) if maximum > 0 else 0.0


def quantile_start(record: AtanasRecord, window_seconds: float, q: float) -> float:
    return float(np.clip(q, 0.0, 1.0) * max_window_start(record, window_seconds))


def resample_rows(values: np.ndarray, target_points: int) -> np.ndarray:
    if values.shape[1] == target_points:
        return values.astype(np.float32, copy=False)
    old_x = np.linspace(0.0, 1.0, values.shape[1], dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, target_points, dtype=np.float64)
    out = np.empty((values.shape[0], target_points), dtype=np.float32)
    for i, row in enumerate(values):
        out[i] = np.interp(new_x, old_x, row).astype(np.float32)
    return out


def extract_window(
    record: AtanasRecord,
    start_seconds: float,
    target_points: int,
    dynamic_mode: str,
    window_seconds: float,
) -> np.ndarray:
    source = record.traces_raw if dynamic_mode == "raw" else record.traces_z
    length = max(2, int(round(window_seconds * record.source_fs)))
    start = int(round(start_seconds * record.source_fs))
    start = min(max(start, 0), max(source.shape[1] - 1, 0))
    end = min(start + length, source.shape[1])
    segment = source[:, start:end]
    if segment.shape[1] < 2:
        segment = source
    return resample_rows(segment, target_points)


def official_dropout_indices(indices: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    if mode == "none" or len(indices) <= 1:
        return indices.copy()
    minimum = int(len(indices) * s1.OFFICIAL_UNIT_DROPOUT_MIN_FRACTION)
    minimum = max(1, minimum)
    number = int(rng.integers(minimum, len(indices) + 1))
    chosen_positions = np.sort(rng.permutation(len(indices))[:number])
    return indices[chosen_positions]


def same_indices_and_match(
    record: AtanasRecord,
    mode: str,
    minimum_matches: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, Tensor]:
    base = record.same_worm_indices
    for _ in range(20):
        idx1 = official_dropout_indices(base, mode, rng)
        idx2 = official_dropout_indices(base, mode, rng)
        pos2 = {int(value): i for i, value in enumerate(idx2)}
        pairs = [(i, pos2[int(value)]) for i, value in enumerate(idx1) if int(value) in pos2]
        if len(pairs) >= minimum_matches:
            return idx1, idx2, torch.tensor(pairs, dtype=torch.long, device=device).T.contiguous()
    idx1 = base.copy()
    idx2 = base.copy()
    match = torch.arange(len(base), dtype=torch.long, device=device).repeat(2, 1)
    return idx1, idx2, match


def cross_indices_and_match(
    record1: AtanasRecord,
    record2: AtanasRecord,
    mode: str,
    minimum_matches: int,
    identity_batch_size: int,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, Tensor]:
    base1 = record1.valid_unique_indices
    base2 = record2.valid_unique_indices
    shared = set(record1.labels[base1]) & set(record2.labels[base2])
    if len(shared) < minimum_matches:
        raise RuntimeError(
            f"{record1.worm_id} and {record2.worm_id}: only {len(shared)} shared IDs"
        )

    if identity_batch_size > 0:
        # In the explicit two-view experiment, B is the neuron/identity batch
        # size rather than the number of worm-pair samples. Select the same B
        # canonical identities from the two worms and keep their order aligned,
        # so the correct B x B cross-worm matches lie on the diagonal.
        ordered_shared = np.asarray(sorted(shared), dtype=object)
        number = min(identity_batch_size, len(ordered_shared))
        selected_positions = np.sort(
            rng.choice(len(ordered_shared), size=number, replace=False)
        )
        selected_labels = ordered_shared[selected_positions]
        original1 = {
            str(record1.labels[index]): int(index)
            for index in base1
        }
        original2 = {
            str(record2.labels[index]): int(index)
            for index in base2
        }
        idx1 = np.asarray(
            [original1[str(label)] for label in selected_labels],
            dtype=np.int64,
        )
        idx2 = np.asarray(
            [original2[str(label)] for label in selected_labels],
            dtype=np.int64,
        )
        match = torch.arange(
            number, dtype=torch.long, device=device
        ).repeat(2, 1)
        return idx1, idx2, match

    for _ in range(20):
        idx1 = official_dropout_indices(base1, mode, rng)
        idx2 = official_dropout_indices(base2, mode, rng)
        map1 = {str(record1.labels[i]): p for p, i in enumerate(idx1) if str(record1.labels[i]) in shared}
        map2 = {str(record2.labels[i]): p for p, i in enumerate(idx2) if str(record2.labels[i]) in shared}
        surviving = sorted(set(map1) & set(map2))
        if len(surviving) >= minimum_matches:
            pairs = [(map1[label], map2[label]) for label in surviving]
            return idx1, idx2, torch.tensor(pairs, dtype=torch.long, device=device).T.contiguous()
    idx1 = base1.copy()
    idx2 = base2.copy()
    map1 = {str(record1.labels[i]): p for p, i in enumerate(idx1) if str(record1.labels[i]) in shared}
    map2 = {str(record2.labels[i]): p for p, i in enumerate(idx2) if str(record2.labels[i]) in shared}
    surviving = sorted(set(map1) & set(map2))
    pairs = [(map1[label], map2[label]) for label in surviving]
    return idx1, idx2, torch.tensor(pairs, dtype=torch.long, device=device).T.contiguous()


def to_bins(values: np.ndarray, backbone: nn.Module, device: torch.device) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(values)).to(device=device, dtype=torch.float32)
    expected = int(backbone.num_latents * backbone.patch_size)
    if tensor.shape[1] != expected:
        raise ValueError(f"Expected {expected} samples, got {tensor.shape[1]}")
    return tensor.reshape(tensor.shape[0], backbone.num_latents, backbone.patch_size).reshape(
        tensor.shape[0] * backbone.num_latents, backbone.patch_size
    )


def gather_features(record: AtanasRecord, indices: np.ndarray, name: str, device: torch.device) -> Tensor:
    values = getattr(record, name)[indices]
    return torch.from_numpy(np.ascontiguousarray(values)).to(device=device, dtype=torch.float32)


# -----------------------------------------------------------------------------
# Model variants
# -----------------------------------------------------------------------------


def make_mlp(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, 256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, output_dim),
        nn.LayerNorm(output_dim),
    )


class AtanasMultiBranchEncoder(nn.Module):
    def __init__(
        self,
        variant: str,
        absolute_dim: int,
        population_dim: int,
        feature_dropout: float,
        device: torch.device,
        last_temporal_value_mode: str = "classical",
        last_temporal_value_qubits: int = 4,
        last_temporal_value_depth: int = 2,
        last_temporal_value_chunk_size: int = 256,
        args: argparse.Namespace | None = None,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.output_dim = 256
        self.dynamic_backbone: nn.Module | None = None
        self.absolute_branch: nn.Module | None = None
        self.population_branch: nn.Module | None = None

        parts = 0
        if variant in DYNAMIC_VARIANTS:
            architecture = {
                "patch_size": int(getattr(args, "nuclr_patch_size", 4)),
                "dim": int(getattr(args, "nuclr_dim", 256)),
                "self_heads": int(getattr(args, "nuclr_heads", 4)),
                "dim_head": int(getattr(args, "nuclr_dim_head", 64)),
                "atn_dropout": float(getattr(args, "nuclr_attention_dropout", 0.0)),
                "lin_dropout": float(getattr(args, "nuclr_linear_dropout", 0.2)),
                "t_layers": int(getattr(args, "nuclr_temporal_layers", 2)),
                "st_layers": int(getattr(args, "nuclr_spatiotemporal_layers", 2)),
                "rot_ratio": float(getattr(args, "nuclr_rot_ratio", 0.5)),
            }
            self.dynamic_backbone = s1.NuclrV2gaCa2(
                ctx_duration=s1.OFFICIAL_VIEW_SECONDS,
                precision=s1.Precision(s1.OFFICIAL_PRECISION),
                **architecture,
            ).to(device)
            expected_samples = int(
                self.dynamic_backbone.num_latents * self.dynamic_backbone.patch_size
            )
            if expected_samples != 128:
                raise RuntimeError(
                    "NuCLR activity protocol requires exactly 128 samples per view; "
                    f"architecture produced {expected_samples}"
                )
            self.output_dim = architecture["dim"]
            install_last_temporal_qasa_value(
                self.dynamic_backbone,
                mode=last_temporal_value_mode,
                qubits=last_temporal_value_qubits,
                depth=last_temporal_value_depth,
                chunk_size=last_temporal_value_chunk_size,
            )
            parts += self.output_dim
        if variant in ABSOLUTE_VARIANTS:
            self.absolute_branch = make_mlp(absolute_dim, 128, feature_dropout)
            parts += 128
        if variant in POPULATION_VARIANTS:
            self.population_branch = make_mlp(population_dim, 128, feature_dropout)
            parts += 128

        if parts == self.output_dim and variant in {"dynamic_only", "raw_nuclr"}:
            self.fusion = nn.Identity()
        elif parts == 128 and variant == "absolute_only":
            self.fusion = nn.Sequential(
                nn.Linear(128, 256), nn.GELU(), nn.LayerNorm(256)
            )
        else:
            self.fusion = nn.Sequential(
                nn.LayerNorm(parts),
                nn.Linear(parts, 256),
                nn.GELU(),
                nn.Dropout(feature_dropout),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
            )

    @property
    def has_dynamic(self) -> bool:
        return self.dynamic_backbone is not None

    def forward(
        self,
        bins: Tensor | None,
        lengths: Tensor,
        absolute: Tensor | None,
        population: Tensor | None,
    ) -> Tensor:
        pieces: list[Tensor] = []
        if self.dynamic_backbone is not None:
            if bins is None:
                raise ValueError("Dynamic variant requires bins")
            pieces.append(self.dynamic_backbone(bins=bins, unit_seqlen=lengths))
        if self.absolute_branch is not None:
            if absolute is None:
                raise ValueError("Absolute branch requires features")
            pieces.append(self.absolute_branch(absolute))
        if self.population_branch is not None:
            if population is None:
                raise ValueError("Population branch requires features")
            pieces.append(self.population_branch(population))
        return self.fusion(torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0])


def temporal_value_kwargs(config: Any) -> dict[str, Any]:
    """Read optional QASA fields from a Namespace or checkpoint dictionary."""
    if isinstance(config, dict):
        get = config.get
    else:
        get = lambda key, default: getattr(config, key, default)
    return {
        "last_temporal_value_mode": get(
            "last_temporal_value_mode", "classical"
        ),
        "last_temporal_value_qubits": int(
            get("last_temporal_value_qubits", 4)
        ),
        "last_temporal_value_depth": int(
            get("last_temporal_value_depth", 2)
        ),
        "last_temporal_value_chunk_size": int(
            get("last_temporal_value_chunk_size", 256)
        ),
    }


class SampleWiseDCL(nn.Module):
    """Paper-style symmetric decoupled contrastive loss for arbitrary encoders."""

    def __init__(self, input_dim: int, projection_dim: int, tau: float, full_denom: bool) -> None:
        super().__init__()
        self.tau = float(tau)
        self.full_denom = bool(full_denom)
        self.projector = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, projection_dim),
        )

    def _directional(self, q: Tensor, c: Tensor, match: Tensor) -> Tensor:
        q = F.normalize(q.float(), dim=-1)
        c = F.normalize(c.float(), dim=-1)
        anchor_idx = match[0]
        positive_idx = match[1]
        q_anchor = q.index_select(0, anchor_idx)
        cross_logits = (q_anchor @ c.T) / self.tau
        row = torch.arange(len(anchor_idx), device=q.device)
        positive = cross_logits[row, positive_idx]

        cross_mask = torch.ones_like(cross_logits, dtype=torch.bool)
        cross_mask[row, positive_idx] = False
        negative_parts = [cross_logits.masked_fill(~cross_mask, -torch.inf)]
        if self.full_denom:
            same_logits = (q_anchor @ q.T) / self.tau
            same_mask = torch.ones_like(same_logits, dtype=torch.bool)
            same_mask[row, anchor_idx] = False
            negative_parts.append(same_logits.masked_fill(~same_mask, -torch.inf))
        negatives = torch.cat(negative_parts, dim=1)
        log_negative_sum = torch.logsumexp(negatives, dim=1)
        return (-positive + log_negative_sum).mean()

    def forward(
        self,
        embedding1: Tensor,
        embedding2: Tensor,
        lengths1: Tensor,
        lengths2: Tensor,
        matches: Sequence[Tensor],
    ) -> Tensor:
        projected1 = self.projector(embedding1)
        projected2 = self.projector(embedding2)
        losses: list[Tensor] = []
        offset1 = 0
        offset2 = 0
        for n1_t, n2_t, match in zip(lengths1, lengths2, matches):
            n1 = int(n1_t.item())
            n2 = int(n2_t.item())
            z1 = projected1[offset1 : offset1 + n1]
            z2 = projected2[offset2 : offset2 + n2]
            losses.append(self._directional(z1, z2, match))
            losses.append(self._directional(z2, z1, match.flip(0)))
            offset1 += n1
            offset2 += n2
        if not losses:
            raise RuntimeError("DCL batch contains no valid samples")
        return torch.stack(losses).mean()


# -----------------------------------------------------------------------------
# Batch preparation
# -----------------------------------------------------------------------------


def dynamic_mode_for_variant(variant: str) -> str:
    return "raw" if variant == "raw_nuclr" else "zscore"


def prepare_same_batch(
    samples: Sequence[SameSample],
    records_variant: str,
    model: AtanasMultiBranchEncoder,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> PreparedViews:
    bins1: list[Tensor] = []
    bins2: list[Tensor] = []
    abs1: list[Tensor] = []
    abs2: list[Tensor] = []
    pop1: list[Tensor] = []
    pop2: list[Tensor] = []
    lengths1: list[int] = []
    lengths2: list[int] = []
    matches: list[Tensor] = []
    target_points = 128

    for sample in samples:
        idx1, idx2, match = same_indices_and_match(
            sample.record,
            args.unit_dropout,
            args.minimum_positive_matches,
            rng,
            device,
        )
        lengths1.append(len(idx1))
        lengths2.append(len(idx2))
        matches.append(match)
        if model.has_dynamic:
            mode = dynamic_mode_for_variant(records_variant)
            view1 = extract_window(sample.record, sample.start1, target_points, mode, args.window_seconds)[idx1]
            view2 = extract_window(sample.record, sample.start2, target_points, mode, args.window_seconds)[idx2]
            bins1.append(to_bins(view1, model.dynamic_backbone, device))
            bins2.append(to_bins(view2, model.dynamic_backbone, device))
        if model.absolute_branch is not None:
            abs1.append(gather_features(sample.record, idx1, "absolute_features", device))
            abs2.append(gather_features(sample.record, idx2, "absolute_features", device))
        if model.population_branch is not None:
            pop1.append(gather_features(sample.record, idx1, "population_features", device))
            pop2.append(gather_features(sample.record, idx2, "population_features", device))

    return PreparedViews(
        bins1=torch.cat(bins1) if bins1 else None,
        bins2=torch.cat(bins2) if bins2 else None,
        lengths1=torch.tensor(lengths1, dtype=torch.long, device=device),
        lengths2=torch.tensor(lengths2, dtype=torch.long, device=device),
        absolute1=torch.cat(abs1) if abs1 else None,
        absolute2=torch.cat(abs2) if abs2 else None,
        population1=torch.cat(pop1) if pop1 else None,
        population2=torch.cat(pop2) if pop2 else None,
        matches=matches,
        num_matches=sum(int(x.shape[1]) for x in matches),
    )


def prepare_cross_batch(
    samples: Sequence[CrossSample],
    variant: str,
    model: AtanasMultiBranchEncoder,
    args: argparse.Namespace,
    rng: np.random.Generator,
    device: torch.device,
) -> PreparedViews:
    bins1: list[Tensor] = []
    bins2: list[Tensor] = []
    abs1: list[Tensor] = []
    abs2: list[Tensor] = []
    pop1: list[Tensor] = []
    pop2: list[Tensor] = []
    lengths1: list[int] = []
    lengths2: list[int] = []
    matches: list[Tensor] = []
    target_points = 128

    for sample in samples:
        idx1, idx2, match = cross_indices_and_match(
            sample.record1,
            sample.record2,
            args.unit_dropout,
            args.minimum_positive_matches,
            args.identity_batch_size,
            rng,
            device,
        )
        lengths1.append(len(idx1))
        lengths2.append(len(idx2))
        matches.append(match)
        if model.has_dynamic:
            mode = dynamic_mode_for_variant(variant)
            view1 = extract_window(sample.record1, sample.start1, target_points, mode, args.window_seconds)[idx1]
            view2 = extract_window(sample.record2, sample.start2, target_points, mode, args.window_seconds)[idx2]
            bins1.append(to_bins(view1, model.dynamic_backbone, device))
            bins2.append(to_bins(view2, model.dynamic_backbone, device))
        if model.absolute_branch is not None:
            abs1.append(gather_features(sample.record1, idx1, "absolute_features", device))
            abs2.append(gather_features(sample.record2, idx2, "absolute_features", device))
        if model.population_branch is not None:
            pop1.append(gather_features(sample.record1, idx1, "population_features", device))
            pop2.append(gather_features(sample.record2, idx2, "population_features", device))

    return PreparedViews(
        bins1=torch.cat(bins1) if bins1 else None,
        bins2=torch.cat(bins2) if bins2 else None,
        lengths1=torch.tensor(lengths1, dtype=torch.long, device=device),
        lengths2=torch.tensor(lengths2, dtype=torch.long, device=device),
        absolute1=torch.cat(abs1) if abs1 else None,
        absolute2=torch.cat(abs2) if abs2 else None,
        population1=torch.cat(pop1) if pop1 else None,
        population2=torch.cat(pop2) if pop2 else None,
        matches=matches,
        num_matches=sum(int(x.shape[1]) for x in matches),
    )


def forward_prepared(model: AtanasMultiBranchEncoder, prepared: PreparedViews) -> tuple[Tensor, Tensor]:
    e1 = model(prepared.bins1, prepared.lengths1, prepared.absolute1, prepared.population1)
    e2 = model(prepared.bins2, prepared.lengths2, prepared.absolute2, prepared.population2)
    return e1, e2


# -----------------------------------------------------------------------------
# Sampling schedules
# -----------------------------------------------------------------------------


def build_same_samples(
    records: Sequence[AtanasRecord],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[SameSample]:
    samples: list[SameSample] = []
    for record in records:
        for _ in range(args.same_views_per_worm):
            samples.append(
                SameSample(
                    record,
                    random_start(record, args.window_seconds, rng),
                    random_start(record, args.window_seconds, rng),
                )
            )
    rng.shuffle(samples)
    return samples


def eligible_cross_pairs(
    records: Sequence[AtanasRecord], minimum_matches: int
) -> list[tuple[AtanasRecord, AtanasRecord]]:
    pairs: list[tuple[AtanasRecord, AtanasRecord]] = []
    for i, a in enumerate(records):
        ids_a = set(a.labels[a.valid_unique_indices])
        for b in records[i + 1 :]:
            if len(ids_a & set(b.labels[b.valid_unique_indices])) >= minimum_matches:
                pairs.append((a, b))
    if not pairs:
        raise RuntimeError("No train worm pairs have enough shared identities")
    return pairs


def build_cross_samples(
    all_pairs: Sequence[tuple[AtanasRecord, AtanasRecord]],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> list[CrossSample]:
    if args.cross_pairs_per_epoch > 0 and args.cross_pairs_per_epoch < len(all_pairs):
        selected = [all_pairs[i] for i in rng.choice(len(all_pairs), args.cross_pairs_per_epoch, replace=False)]
    else:
        selected = list(all_pairs)
    samples: list[CrossSample] = []
    for a, b in selected:
        for _ in range(args.cross_views_per_pair):
            if args.cross_window_mode == "aligned_quantile":
                q = float(rng.random())
                start1 = quantile_start(a, args.window_seconds, q)
                start2 = quantile_start(b, args.window_seconds, q)
            else:
                start1 = random_start(a, args.window_seconds, rng)
                start2 = random_start(b, args.window_seconds, rng)
            samples.append(CrossSample(a, b, start1, start2))
    rng.shuffle(samples)
    return samples


# -----------------------------------------------------------------------------
# Metrics and validation
# -----------------------------------------------------------------------------


def accumulate_direction(q: Tensor, c: Tensor, match: Tensor, acc: RetrievalAccumulator) -> None:
    q = F.normalize(q.float(), dim=-1)
    c = F.normalize(c.float(), dim=-1)
    similarity = q @ c.T
    anchor_idx = match[0]
    positive_idx = match[1]
    selected = similarity.index_select(0, anchor_idx)
    order = torch.argsort(selected, dim=1, descending=True)
    row = torch.arange(len(anchor_idx), device=q.device)
    ranks = (order == positive_idx[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    acc.correct1 += int((ranks <= 1).sum().item())
    acc.correct5 += int((ranks <= 5).sum().item())
    acc.reciprocal_rank_sum += float((1.0 / ranks.float()).sum().item())
    acc.queries += int(len(anchor_idx))
    positives = selected[row, positive_idx]
    acc.positive_sum += float(positives.sum().item())
    negative_mask = torch.ones_like(selected, dtype=torch.bool)
    negative_mask[row, positive_idx] = False
    negatives = selected[negative_mask]
    acc.negative_sum += float(negatives.sum().item())
    acc.negative_count += int(negatives.numel())


@torch.no_grad()
def batch_retrieval(
    e1: Tensor,
    e2: Tensor,
    prepared: PreparedViews,
    criterion: SampleWiseDCL,
) -> tuple[RetrievalAccumulator, RetrievalAccumulator]:
    encoder_acc = RetrievalAccumulator()
    projected_acc = RetrievalAccumulator()
    p1 = criterion.projector(e1.float())
    p2 = criterion.projector(e2.float())
    o1 = 0
    o2 = 0
    for n1_t, n2_t, match in zip(prepared.lengths1, prepared.lengths2, prepared.matches):
        n1 = int(n1_t.item())
        n2 = int(n2_t.item())
        a1 = e1[o1:o1+n1]
        a2 = e2[o2:o2+n2]
        z1 = p1[o1:o1+n1]
        z2 = p2[o2:o2+n2]
        accumulate_direction(a1, a2, match, encoder_acc)
        accumulate_direction(a2, a1, match.flip(0), encoder_acc)
        accumulate_direction(z1, z2, match, projected_acc)
        accumulate_direction(z2, z1, match.flip(0), projected_acc)
        o1 += n1
        o2 += n2
    return encoder_acc, projected_acc


def merge_acc(dst: RetrievalAccumulator, src: RetrievalAccumulator) -> None:
    for field in asdict(dst):
        setattr(dst, field, getattr(dst, field) + getattr(src, field))


@torch.no_grad()
def embed_record(
    record: AtanasRecord,
    model: AtanasMultiBranchEncoder,
    criterion: SampleWiseDCL,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = record.valid_unique_indices
    lengths = torch.tensor([len(indices)], dtype=torch.long, device=device)
    absolute = (
        gather_features(record, indices, "absolute_features", device)
        if model.absolute_branch is not None else None
    )
    population = (
        gather_features(record, indices, "population_features", device)
        if model.population_branch is not None else None
    )
    encoder_views: list[Tensor] = []
    if model.has_dynamic:
        mode = dynamic_mode_for_variant(args.model_variant)
        starts = np.linspace(
            0.0,
            max_window_start(record, args.window_seconds),
            args.eval_num_windows,
        )
        for start in starts:
            view = extract_window(record, float(start), 128, mode, args.window_seconds)[indices]
            bins = to_bins(view, model.dynamic_backbone, device)
            with autocast_context(device):
                encoder_views.append(model(bins, lengths, absolute, population).float())
        encoder = torch.stack(encoder_views).mean(dim=0)
    else:
        with autocast_context(device):
            encoder = model(None, lengths, absolute, population).float()
    projected = criterion.projector(encoder).float()
    return (
        F.normalize(encoder, dim=-1).cpu().numpy(),
        F.normalize(projected, dim=-1).cpu().numpy(),
        record.labels[indices],
    )


def retrieval_pair(
    emb_a: np.ndarray,
    labels_a: np.ndarray,
    emb_b: np.ndarray,
    labels_b: np.ndarray,
) -> dict[str, float]:
    map_b = {str(label): i for i, label in enumerate(labels_b)}
    query_positions = [i for i, label in enumerate(labels_a) if str(label) in map_b]
    if not query_positions:
        return {"queries": 0, "top1": math.nan, "top5": math.nan, "mrr": math.nan}
    sim = emb_a[query_positions] @ emb_b.T
    positives = np.asarray([map_b[str(labels_a[i])] for i in query_positions], dtype=np.int64)
    order = np.argsort(-sim, axis=1)
    ranks = np.argmax(order == positives[:, None], axis=1) + 1
    return {
        "queries": int(len(ranks)),
        "top1": float(np.mean(ranks <= 1)),
        "top5": float(np.mean(ranks <= 5)),
        "mrr": float(np.mean(1.0 / ranks)),
    }


@torch.no_grad()
def evaluate_validation(
    records: Sequence[AtanasRecord],
    model: AtanasMultiBranchEncoder,
    criterion: SampleWiseDCL,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    model.eval()
    criterion.eval()
    cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i, record in enumerate(records, start=1):
        cache[record.worm_id] = embed_record(record, model, criterion, args, device)
        print(f"    embedded {i:02d}/{len(records):02d} {record.path.name}")

    sums = {
        "encoder": {"queries": 0, "top1_num": 0.0, "top5_num": 0.0, "mrr_num": 0.0},
        "projected": {"queries": 0, "top1_num": 0.0, "top5_num": 0.0, "mrr_num": 0.0},
    }
    rows: list[dict[str, Any]] = []
    for a in records:
        for b in records:
            if a is b:
                continue
            ea, pa, la = cache[a.worm_id]
            eb, pb, lb = cache[b.worm_id]
            for space, xa, xb in (("encoder", ea, eb), ("projected", pa, pb)):
                metrics = retrieval_pair(xa, la, xb, lb)
                q = int(metrics["queries"])
                if q == 0:
                    continue
                sums[space]["queries"] += q
                sums[space]["top1_num"] += metrics["top1"] * q
                sums[space]["top5_num"] += metrics["top5"] * q
                sums[space]["mrr_num"] += metrics["mrr"] * q
                rows.append({
                    "space": space,
                    "query_worm": a.worm_id,
                    "candidate_worm": b.worm_id,
                    **metrics,
                })
    result: dict[str, dict[str, float]] = {}
    for space, values in sums.items():
        q = max(int(values["queries"]), 1)
        result[space] = {
            "queries": int(values["queries"]),
            "top1": float(values["top1_num"] / q),
            "top5": float(values["top5_num"] / q),
            "mrr": float(values["mrr_num"] / q),
        }
    model.train()
    criterion.train()
    return result, rows


def metric_value(metrics: dict[str, dict[str, float]], args: argparse.Namespace) -> float:
    # V2 deliberately selects checkpoints and drives early stopping only with
    # encoder-space validation retrieval. Keep args.selection_space in saved
    # metadata for evaluator compatibility, but do not permit projected-space
    # selection to re-enter through a modified command line or checkpoint.
    if args.selection_space != "encoder":
        raise ValueError(
            "V2 requires selection_space='encoder'; projected metrics are diagnostic only"
        )
    return float(metrics["encoder"][args.selection_metric])


# -----------------------------------------------------------------------------
# Optimizer, schedule and checkpoints
# -----------------------------------------------------------------------------


def clean_state_dict(state: dict[str, Tensor]) -> dict[str, Tensor]:
    cleaned: dict[str, Tensor] = {}
    for key, value in state.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def maybe_load_backbone(model: AtanasMultiBranchEncoder, path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if model.dynamic_backbone is None:
        print("warning: --init-backbone-checkpoint ignored for absolute_only")
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise KeyError(f"{path}: missing model_state_dict")
    state = clean_state_dict(checkpoint["model_state_dict"])
    # Accept both a bare official NuCLR checkpoint and a checkpoint previously
    # saved by this Atanas trainer, where the same module is nested below
    # dynamic_backbone.*. This makes cross-worm-only fine-tuning resumable from
    # the established Atanas Raw NuCLR best.pt.
    nested = {
        key[len("dynamic_backbone."):]: value
        for key, value in state.items()
        if key.startswith("dynamic_backbone.")
    }
    if nested:
        state = nested
    elif any(key.startswith("nuclr.") for key in state):
        state = {
            key[len("nuclr."):]: value
            for key, value in state.items()
            if key.startswith("nuclr.")
        }
    model.dynamic_backbone.load_state_dict(state, strict=True)
    print(f"Loaded NuCLR dynamic backbone from {path}")
    return {
        "mode": "backbone_only",
        "path": str(path),
        "train_step": checkpoint.get("global_step", checkpoint.get("train_step")),
        "epoch": checkpoint.get("epoch"),
    }


def maybe_load_full_checkpoint(
    model: AtanasMultiBranchEncoder,
    criterion: SampleWiseDCL,
    path: Path | None,
    expected_variant: str,
) -> dict[str, Any] | None:
    """Load encoder and projector weights without restoring training state."""
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"{path}: checkpoint must be a dictionary")
    missing = [
        key
        for key in ("model_state_dict", "loss_state_dict")
        if key not in checkpoint
    ]
    if missing:
        raise KeyError(f"{path}: missing checkpoint keys {missing}")

    saved_variant = checkpoint.get("model_variant")
    if saved_variant is None and isinstance(checkpoint.get("args"), dict):
        saved_variant = checkpoint["args"].get("model_variant")
    if saved_variant is not None and str(saved_variant) != expected_variant:
        raise ValueError(
            f"{path}: model_variant={saved_variant!r}, expected "
            f"{expected_variant!r}"
        )

    model_state = clean_state_dict(checkpoint["model_state_dict"])
    loss_state = clean_state_dict(checkpoint["loss_state_dict"])
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"{path}: encoder architecture does not match the current run"
        ) from error
    try:
        criterion.load_state_dict(loss_state, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            f"{path}: DCL projector architecture does not match "
            f"--projector-dim={criterion.projector[-1].out_features}"
        ) from error

    print(f"Loaded complete encoder and DCL projector from {path}")
    print("Starting a fresh optimizer, schedule, global step, and validation selection")
    return {
        "mode": "full_encoder_and_projector",
        "path": str(path),
        "train_step": checkpoint.get("global_step", checkpoint.get("train_step")),
        "epoch": checkpoint.get("epoch"),
        "saved_best_metric": checkpoint.get("best_metric"),
        "saved_best_epoch": checkpoint.get("best_epoch"),
    }


def validate_qasa_source_checkpoint(
    path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Verify that QASA extends the split-matched Same-only NuCLR source."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved = dict(checkpoint.get("args", {}))
    for key in ("train_root", "val_root"):
        if key not in saved:
            raise AssertionError(f"QASA source checkpoint is missing {key}")
        observed = Path(saved[key]).resolve()
        expected = Path(getattr(args, key)).resolve()
        if observed != expected:
            raise AssertionError(
                f"QASA source {key} mismatch: {observed} != {expected}"
            )
    if float(saved.get("lambda_same", -1.0)) != 1.0:
        raise AssertionError("QASA source must be the Same-only NuCLR checkpoint")
    if float(saved.get("lambda_cross", -1.0)) != 0.0:
        raise AssertionError("QASA source must not use cross-worm supervision")
    source_mode = str(saved.get("last_temporal_value_mode", "classical"))
    if source_mode != "classical":
        raise AssertionError(
            "QASA experiment must initialise from the original classical NuCLR"
        )
    return {
        "source_last_temporal_value_mode": source_mode,
        "source_train_root": str(Path(saved["train_root"]).resolve()),
        "source_val_root": str(Path(saved["val_root"]).resolve()),
        "source_same_only": True,
    }


def add_group(
    groups: list[dict[str, Any]],
    module: nn.Module | None,
    base_lr: float,
    weight_decay: float,
    seen: set[int],
) -> None:
    if module is None:
        return
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if "norm" in name.lower() or name.endswith("bias") or ".bias" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if decay:
        groups.append({"params": decay, "lr": base_lr, "base_lr": base_lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": base_lr, "base_lr": base_lr, "weight_decay": 0.0})


def add_qasa_value_group(
    groups: list[dict[str, Any]],
    module: nn.Module | None,
    base_lr: float,
    weight_decay: float,
    seen: set[int],
) -> None:
    if module is None:
        return
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in module.named_parameters():
        if (
            not is_qasa_value_parameter("." + name)
            or not parameter.requires_grad
            or id(parameter) in seen
        ):
            continue
        seen.add(id(parameter))
        lowered = name.lower()
        if (
            "norm" in lowered
            or name.endswith("bias")
            or ".bias" in name
            or "circuit_weights" in name
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": base_lr,
                "base_lr": base_lr,
                "weight_decay": weight_decay,
                "name": "qasa_value_decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": base_lr,
                "base_lr": base_lr,
                "weight_decay": 0.0,
                "name": "qasa_value_no_decay",
            }
        )


def build_optimizer(
    model: AtanasMultiBranchEncoder,
    criterion: SampleWiseDCL,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    add_qasa_value_group(
        groups,
        model.dynamic_backbone,
        args.qasa_lr,
        args.weight_decay,
        seen,
    )
    add_group(groups, model.dynamic_backbone, args.backbone_lr, args.weight_decay, seen)
    add_group(groups, model.absolute_branch, args.branch_lr, args.weight_decay, seen)
    add_group(groups, model.population_branch, args.branch_lr, args.weight_decay, seen)
    add_group(groups, model.fusion, args.branch_lr, args.weight_decay, seen)
    add_group(groups, criterion.projector, args.projector_lr, args.weight_decay, seen)
    if not groups:
        raise RuntimeError("No trainable parameters")
    return torch.optim.AdamW(groups)


def set_qasa_epoch_trainability(
    model: AtanasMultiBranchEncoder,
    criterion: SampleWiseDCL,
    args: argparse.Namespace,
    epoch: int,
) -> bool:
    warmup_only = bool(
        args.last_temporal_value_mode != "classical"
        and args.init_full_checkpoint is not None
        and epoch <= args.qasa_freeze_backbone_epochs
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            is_qasa_value_parameter("." + name) if warmup_only else True
        )
    for parameter in criterion.parameters():
        parameter.requires_grad_(not warmup_only)
    return warmup_only


def lr_factor(step: int, total_steps: int, warmup: int, min_ratio: float) -> float:
    if warmup > 0 and step < warmup:
        return max((step + 1) / warmup, 1e-8)
    progress = (step - warmup) / max(total_steps - warmup, 1)
    progress = float(np.clip(progress, 0.0, 1.0))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def set_lr(optimizer: torch.optim.Optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * factor


def save_checkpoint(
    path: Path,
    model: AtanasMultiBranchEncoder,
    criterion: SampleWiseDCL,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    best_metric: float,
    best_epoch: int,
    feature_scalers: dict[str, Any],
    init_info: dict[str, Any] | None,
    validation: dict[str, Any] | None,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "loss_state_dict": criterion.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "format": "atanas_raw_nuclr_unified_v2_encoder_selection",
            "selection_protocol": "validation_encoder_only",
            "model_variant": args.model_variant,
            "args": vars(args),
            "feature_scalers": feature_scalers,
            "initial_backbone": init_info,
            "latest_validation": validation,
        },
        path,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.split_manifest is not None:
        observed_manifest_sha256 = hashlib.sha256(
            args.split_manifest.read_bytes()
        ).hexdigest()
        if (
            args.split_manifest_sha256
            and observed_manifest_sha256 != args.split_manifest_sha256.lower()
        ):
            raise AssertionError(
                "Split manifest SHA256 mismatch: "
                f"expected {args.split_manifest_sha256.lower()}, "
                f"got {observed_manifest_sha256}"
            )
        args.split_manifest = args.split_manifest.resolve()
        args.split_manifest_sha256 = observed_manifest_sha256
    split_audit = assert_clean_split_protocol(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)

    print("Loading Atanas train/validation worms and extracting audit features...")
    train_load_identity_labels = args.lambda_cross > 0
    train_records = load_records(
        args.train_root,
        "train",
        args.activity_key,
        args.label_key,
        args.source_fs,
        args,
        load_identity_labels=train_load_identity_labels,
    )
    val_records = load_records(
        args.val_root, "val", args.activity_key, args.label_key, args.source_fs, args
    )
    feature_scalers = fit_and_apply_feature_scaling(train_records, val_records)
    write_json(args.save_dir / "feature_scalers.json", feature_scalers)

    absolute_dim = int(train_records[0].absolute_features.shape[1])
    population_dim = int(train_records[0].population_features.shape[1])
    model = AtanasMultiBranchEncoder(
        args.model_variant,
        absolute_dim,
        population_dim,
        args.feature_dropout,
        device,
        **temporal_value_kwargs(args),
        args=args,
    ).to(device)
    criterion = SampleWiseDCL(
        input_dim=model.output_dim,
        projection_dim=args.projector_dim,
        tau=args.temperature,
        full_denom=args.full_denom,
    ).to(device)
    if args.init_full_checkpoint is not None:
        init_info = maybe_load_full_checkpoint(
            model,
            criterion,
            args.init_full_checkpoint,
            args.model_variant,
        )
    else:
        init_info = maybe_load_backbone(model, args.init_backbone_checkpoint)
    if init_info is not None and args.test_worm_ids_file is not None:
        external_init = bool(args.external_init_checkpoint_sha256)
        if external_init:
            if args.init_backbone_checkpoint is None or args.init_full_checkpoint is not None:
                raise AssertionError("Locked external initialization must be backbone-only")
            observed = hashlib.sha256(args.init_backbone_checkpoint.read_bytes()).hexdigest()
            expected = args.external_init_checkpoint_sha256.lower()
            if observed != expected:
                raise AssertionError(
                    f"External initializer SHA256 mismatch: expected {expected}, got {observed}"
                )
            init_info.update({
                "audited_initialization_scope": args.audited_init_scope,
                "independent_external_pretraining": args.audited_init_scope == "external",
                "outer_train_only_pretraining": args.audited_init_scope == "outer_train_only",
                "checkpoint_sha256": observed,
                "provenance": args.external_init_provenance,
            })
        else:
            if (
                args.last_temporal_value_mode == "classical"
                and not args.qasa_continuation_control
            ):
                raise AssertionError(
                    "Clean random-initialization protocol forbids loading an "
                    "initialization checkpoint without locked external provenance"
                )
            if args.init_full_checkpoint is None:
                raise AssertionError(
                    "QASA extension requires --init-full-checkpoint so the encoder "
                    "and its DCL projector retain identical provenance"
                )
            init_info.update(validate_qasa_source_checkpoint(args.init_full_checkpoint, args))
            init_info["qasa_continuation_control"] = bool(
                args.qasa_continuation_control
            )
    optimizer = build_optimizer(model, criterion, args)

    cross_pairs = (
        eligible_cross_pairs(train_records, args.minimum_positive_matches)
        if args.lambda_cross > 0
        else []
    )
    cross_worm_ids = {
        record.path.stem
        for pair in cross_pairs
        for record in pair
    }
    train_identity_stats = (
        clean_identity_statistics(train_records)
        if train_load_identity_labels
        else {
            "worms": len(train_records),
            "mean": None,
            "min": None,
            "max": None,
            "status": "not_loaded_for_same_only",
        }
    )
    val_identity_stats = clean_identity_statistics(val_records)
    protocol_audit = {
        **split_audit,
        "usable_loaded_train_worms": len(train_records),
        "usable_loaded_val_worms": len(val_records),
        "cross_supervision_train_worms": len(cross_worm_ids),
        "eligible_cross_worm_train_pairs": len(cross_pairs),
        "validation_directed_worm_pairs": len(val_records) * (len(val_records) - 1),
        "train_clean_identities_per_worm": train_identity_stats,
        "val_clean_identities_per_worm": val_identity_stats,
        "initialization": "random" if init_info is None else init_info,
        "train_identity_labels_loaded": train_load_identity_labels,
        "same_only_loss_accessed_cell_id": bool(
            args.lambda_cross == 0 and train_load_identity_labels
        ),
        "cross_only_same_worm_branch_had_gradient": False
        if args.lambda_same == 0
        else None,
    }
    write_json(args.save_dir / "protocol_audit.json", protocol_audit)
    write_json(
        args.save_dir / "config.json",
        {
            "protocol": "Atanas clean NuCLR 27/6/5; validation-only; test data unopened",
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "split_audit": protocol_audit,
        },
    )
    # Compute the epoch length deterministically for the scheduler.
    nominal_same = (
        len(train_records) * args.same_views_per_worm
        if args.lambda_same > 0
        else 0
    )
    nominal_pairs = (
        min(args.cross_pairs_per_epoch, len(cross_pairs))
        if args.cross_pairs_per_epoch > 0 else len(cross_pairs)
    )
    nominal_cross = (
        nominal_pairs * args.cross_views_per_pair
        if args.lambda_cross > 0
        else 0
    )
    same_steps = (
        math.ceil(nominal_same / args.batch_size)
        if nominal_same > 0
        else 0
    )
    cross_steps = (
        math.ceil(nominal_cross / args.batch_size)
        if nominal_cross > 0
        else 0
    )
    derived_steps_per_epoch = max(1, same_steps, cross_steps)
    steps_per_epoch = (
        int(args.optimizer_steps_per_epoch)
        if args.optimizer_steps_per_epoch > 0
        else derived_steps_per_epoch
    )
    total_steps = max(1, args.epochs * steps_per_epoch)
    protocol_audit["derived_optimizer_steps_per_epoch"] = derived_steps_per_epoch
    protocol_audit["optimizer_steps_per_epoch"] = steps_per_epoch
    write_json(args.save_dir / "protocol_audit.json", protocol_audit)

    print("=" * 100)
    print("Atanas same-worm + cross-worm DCL training V2")
    print(f"variant                  : {args.model_variant}")
    print(
        "last temporal V          : "
        f"{args.last_temporal_value_mode} "
        f"q={args.last_temporal_value_qubits} "
        f"depth={args.last_temporal_value_depth} "
        f"chunk={args.last_temporal_value_chunk_size}"
    )
    print(
        "checkpoint selection      : "
        f"validation encoder/{args.selection_metric} (projected diagnostic only)"
    )
    print(f"train / val worms        : {len(train_records)} / {len(val_records)}")
    print(f"same-worm neuron set     : {args.same_worm_neurons}")
    print(f"supervision mask         : {args.supervision_mask_key}")
    print(f"eligible cross pairs     : {len(cross_pairs)}")
    print(f"cross-supervision worms : {len(cross_worm_ids)}")
    if train_load_identity_labels:
        print(
            "train clean identities   : "
            f"mean={train_identity_stats['mean']:.2f}, "
            f"range={train_identity_stats['min']}-{train_identity_stats['max']}"
        )
    else:
        print("train identity fields     : NOT OPENED (Same-only audit path)")
    print(
        "val clean identities     : "
        f"mean={val_identity_stats['mean']:.2f}, "
        f"range={val_identity_stats['min']}-{val_identity_stats['max']}"
    )
    print(
        "validation directed pairs: "
        f"{protocol_audit['validation_directed_worm_pairs']}"
    )
    print(f"same samples/epoch       : {nominal_same}")
    print(f"cross samples/epoch      : {nominal_cross}")
    print(f"worm-pair sample batch   : {args.batch_size}")
    print(
        "identity batch B         : "
        f"{args.identity_batch_size if args.identity_batch_size > 0 else 'original/all'}"
    )
    print(f"optimizer steps/epoch    : {steps_per_epoch}")
    print(
        "objective                : "
        f"{args.lambda_same:g} * L_same + "
        f"{args.lambda_cross:g} * L_cross_DCL"
    )
    print(f"DCL                      : tau={args.temperature}, full_denom={args.full_denom}")
    print(f"window mode              : {args.cross_window_mode}, {args.window_seconds:g}s")
    print(f"evaluation               : {args.eval_num_windows} uniformly sampled windows")
    print(f"feature dims             : absolute={absolute_dim}, population={population_dim}")
    print(
        "initialization           : "
        f"{init_info['mode'] if init_info is not None else 'random/default'}"
    )
    print(f"device                    : {device}")
    print("=" * 100)

    print("Initial validation:")
    initial_val, initial_rows = evaluate_validation(val_records, model, criterion, args, device)
    write_json(args.save_dir / "initial_validation.json", initial_val)
    with (args.save_dir / "initial_validation_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        if initial_rows:
            w = csv.DictWriter(f, fieldnames=list(initial_rows[0].keys()))
            w.writeheader(); w.writerows(initial_rows)
    best_metric = metric_value(initial_val, args)
    best_epoch = 0
    global_step = 0
    no_improvement = 0
    save_checkpoint(
        args.save_dir / "best.pt", model, criterion, optimizer, args, 0, 0,
        best_metric, best_epoch, feature_scalers, init_info, initial_val,
    )

    log_path = args.save_dir / "training_log.csv"
    for epoch in range(1, args.epochs + 1):
        qasa_warmup_only = set_qasa_epoch_trainability(
            model,
            criterion,
            args,
            epoch,
        )
        if epoch == 1 or (
            args.qasa_freeze_backbone_epochs > 0
            and epoch == args.qasa_freeze_backbone_epochs + 1
        ):
            print(
                f"epoch {epoch}: QASA-only warm-up="
                f"{str(qasa_warmup_only).lower()}"
            )
        model.train(); criterion.train()
        rng = np.random.default_rng(args.seed + epoch * 100003)
        same_samples = (
            build_same_samples(train_records, args, rng)
            if args.lambda_same > 0
            else []
        )
        cross_samples = (
            build_cross_samples(cross_pairs, args, rng)
            if args.lambda_cross > 0
            else []
        )
        same_batches = list(iter_batches(same_samples, args.batch_size))
        cross_batches = list(iter_batches(cross_samples, args.batch_size))
        same_cycle = infinite_cycle(same_batches) if same_batches else None
        cross_cycle = infinite_cycle(cross_batches) if cross_batches else None

        weighted_same = 0.0
        weighted_cross = 0.0
        same_matches = 0
        cross_matches = 0
        grad_sum = 0.0
        same_encoder_acc = RetrievalAccumulator()
        same_projected_acc = RetrievalAccumulator()
        cross_encoder_acc = RetrievalAccumulator()
        cross_projected_acc = RetrievalAccumulator()

        for _ in range(steps_per_epoch):
            factor = lr_factor(global_step, total_steps, args.warmup_steps, args.min_lr_ratio)
            set_lr(optimizer, factor)
            optimizer.zero_grad(set_to_none=True)

            if args.lambda_same > 0:
                assert same_cycle is not None
                same_prepared = prepare_same_batch(
                    next(same_cycle), args.model_variant, model, args, rng, device
                )
                with autocast_context(device):
                    same_e1, same_e2 = forward_prepared(model, same_prepared)
                    same_loss = criterion(
                        same_e1,
                        same_e2,
                        same_prepared.lengths1,
                        same_prepared.lengths2,
                        same_prepared.matches,
                    )
                weighted_same_loss = args.lambda_same * same_loss.float()
                if not torch.isfinite(weighted_same_loss):
                    raise RuntimeError(
                        f"Non-finite same-worm loss at step {global_step}"
                    )
                weighted_same_loss.backward()
                se, sp = batch_retrieval(
                    same_e1.detach(),
                    same_e2.detach(),
                    same_prepared,
                    criterion,
                )
                same_loss_value = float(same_loss.detach())
                same_num_matches = same_prepared.num_matches
                merge_acc(same_encoder_acc, se)
                merge_acc(same_projected_acc, sp)
                weighted_same += same_loss_value * same_num_matches
                same_matches += same_num_matches
                del (
                    same_e1,
                    same_e2,
                    same_loss,
                    weighted_same_loss,
                    same_prepared,
                )

            if args.lambda_cross > 0:
                assert cross_cycle is not None
                cross_prepared = prepare_cross_batch(
                    next(cross_cycle), args.model_variant, model, args, rng, device
                )
                with autocast_context(device):
                    cross_e1, cross_e2 = forward_prepared(model, cross_prepared)
                    cross_loss = criterion(
                        cross_e1,
                        cross_e2,
                        cross_prepared.lengths1,
                        cross_prepared.lengths2,
                        cross_prepared.matches,
                    )
                weighted_cross_loss = args.lambda_cross * cross_loss.float()
                if not torch.isfinite(weighted_cross_loss):
                    raise RuntimeError(
                        f"Non-finite cross-worm loss at step {global_step}"
                    )
                weighted_cross_loss.backward()
                ce, cp = batch_retrieval(
                    cross_e1.detach(),
                    cross_e2.detach(),
                    cross_prepared,
                    criterion,
                )
                merge_acc(cross_encoder_acc, ce)
                merge_acc(cross_projected_acc, cp)
                weighted_cross += (
                    float(cross_loss.detach()) * cross_prepared.num_matches
                )
                cross_matches += cross_prepared.num_matches
                del (
                    cross_e1,
                    cross_e2,
                    cross_loss,
                    weighted_cross_loss,
                    cross_prepared,
                )

            parameters = [
                p for group in optimizer.param_groups for p in group["params"] if p.grad is not None
            ]
            grad = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()

            grad_sum += float(grad)
            global_step += 1

        same_loss_epoch = weighted_same / max(same_matches, 1)
        cross_loss_epoch = weighted_cross / max(cross_matches, 1)
        latest_val: dict[str, dict[str, float]] | None = None
        improved = False
        if epoch % args.val_every_epochs == 0 or epoch == args.epochs:
            print(f"\nValidation after epoch {epoch}:")
            latest_val, val_rows = evaluate_validation(val_records, model, criterion, args, device)
            current = metric_value(latest_val, args)
            if current > best_metric:
                best_metric = current
                best_epoch = epoch
                no_improvement = 0
                improved = True
                write_json(args.save_dir / "best_validation.json", latest_val)
                with (args.save_dir / "best_validation_pairs.csv").open("w", newline="", encoding="utf-8") as f:
                    if val_rows:
                        w = csv.DictWriter(f, fieldnames=list(val_rows[0].keys()))
                        w.writeheader(); w.writerows(val_rows)
            else:
                no_improvement += 1

        row: dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "same_loss": same_loss_epoch,
            "cross_loss": cross_loss_epoch,
            "total_loss": (
                args.lambda_same * same_loss_epoch
                + args.lambda_cross * cross_loss_epoch
            ),
            "same_matches": same_matches,
            "cross_matches": cross_matches,
            "same_encoder_top1": same_encoder_acc.top1,
            "same_projected_top1": same_projected_acc.top1,
            "cross_encoder_top1": cross_encoder_acc.top1,
            "cross_projected_top1": cross_projected_acc.top1,
            "cross_projected_top5": cross_projected_acc.top5,
            "cross_projected_mrr": cross_projected_acc.mrr,
            "cross_projected_gap": cross_projected_acc.gap,
            "mean_grad_norm_before_clip": grad_sum / steps_per_epoch,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
        }
        if latest_val is not None:
            for space in ("encoder", "projected"):
                for metric in ("top1", "top5", "mrr"):
                    row[f"val_{space}_{metric}"] = latest_val[space][metric]
        append_csv(log_path, row)

        save_checkpoint(
            args.save_dir / "last.pt", model, criterion, optimizer, args,
            epoch, global_step, best_metric, best_epoch, feature_scalers,
            init_info, latest_val,
        )
        if improved:
            save_checkpoint(
                args.save_dir / "best.pt", model, criterion, optimizer, args,
                epoch, global_step, best_metric, best_epoch, feature_scalers,
                init_info, latest_val,
            )
        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            save_checkpoint(
                args.save_dir / f"epoch_{epoch:04d}.pt", model, criterion, optimizer,
                args, epoch, global_step, best_metric, best_epoch,
                feature_scalers, init_info, latest_val,
            )

        val_text = ""
        if latest_val is not None:
            val_text = (
                f" | val enc Top1={latest_val['encoder']['top1']:.4f} "
                f"Top5={latest_val['encoder']['top5']:.4f} "
                f"MRR={latest_val['encoder']['mrr']:.4f} "
                f"| diag proj Top1={latest_val['projected']['top1']:.4f}"
            )
        print(
            f"Epoch {epoch:04d}/{args.epochs:04d} | step={global_step:06d} "
            f"| same={same_loss_epoch:.4f} cross={cross_loss_epoch:.4f} "
            f"total={row['total_loss']:.4f} "
            f"| train same proj Top1={same_projected_acc.top1:.4f} "
            f"cross proj Top1={cross_projected_acc.top1:.4f} "
            f"Top5={cross_projected_acc.top5:.4f} "
            f"| gap={cross_projected_acc.gap:.4f} grad={row['mean_grad_norm_before_clip']:.3f}"
            f"{val_text} | best enc/{args.selection_metric}="
            f"{best_metric:.4f}@{best_epoch}"
        )

        if args.patience > 0 and latest_val is not None and no_improvement >= args.patience:
            print(f"Early stopping after {no_improvement} validation checks without improvement.")
            break

    summary = {
        "format": "atanas_raw_nuclr_unified_v2_encoder_selection",
        "selection_protocol": "validation_encoder_only",
        "model_variant": args.model_variant,
        "best_checkpoint": str(args.save_dir / "best.pt"),
        "last_checkpoint": str(args.save_dir / "last.pt"),
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "selection_space": args.selection_space,
        "selection_metric": args.selection_metric,
        "global_step": global_step,
        "objective": (
            f"{args.lambda_same:g} * L_same + "
            f"{args.lambda_cross:g} * L_cross_DCL"
        ),
        "initial_backbone": init_info,
        "protocol_audit": protocol_audit,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "same_branch_backward_calls": global_step if args.lambda_same > 0 else 0,
        "cross_branch_backward_calls": global_step if args.lambda_cross > 0 else 0,
        "same_only_loss_accessed_cell_id": bool(
            args.lambda_cross == 0 and train_load_identity_labels
        ),
        "cross_only_same_worm_branch_had_gradient": False
        if args.lambda_same == 0
        else None,
        "test_embedding_or_metric_computed": False,
    }
    write_json(args.save_dir / "summary.json", summary)
    print("=" * 100)
    print("Training complete")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
