"""Shared protocol, metrics, and CPD code for locked Atanas baselines.

The implementation intentionally keeps the candidate set equal to all neurons
in the reference worm.  Ground-truth clean identities are used only to decide
which query rows are scoreable and which candidate is correct.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


MANIFEST_SHA256 = "8ffc2031920bf9cd096271daedba91d14b93ff2573fb3b436d9cf3b236e0687b"
FDNC_INIT_SHA256 = "ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9"
SEEDS = (1, 42, 123)

# This is the single CPD configuration fixed before this five-fold baseline
# evaluation.  It is the same configuration used in the earlier clean Atanas
# validation comparison.  No mirror branch and no identity-guided selection.
CPD_CONFIG: dict[str, Any] = {
    "normalization": "per_worm_median_divide_200",
    "rigid": {
        "outlier_weight": 0.1,
        "fix_scale": True,
        "max_iterations": 250,
        "sigma2_tolerance": 1e-7,
        "transform": "xy_rotation_xyz_translation",
    },
    "nonrigid": {
        "outlier_weight": 0.1,
        "lambda": 4000.0,
        "beta": 0.25,
        "max_iterations": 150,
        "sigma2_tolerance": 1e-5,
    },
    "identity_used_for_registration": False,
    "gt_selected_mirror_branch": False,
    "score": "negative_euclidean_distance_after_registration",
}

FDNC_CONFIG: dict[str, Any] = {
    "seed": 42,
    "supervision_mask_key": "clean_mask",
    "normalization": "median_scale",
    "normalization_scale": 200.0,
    "unfreeze_last_n": 2,
    "epochs": 60,
    "pairs_per_epoch": 200,
    "minimum_common": 8,
    "backbone_lr": 5e-6,
    "outlier_lr": 5e-5,
    "weight_decay": 0.01,
    "outlier_weight": 0.25,
    "l2sp_weight": 1e-4,
    "gradient_clip": 1.0,
    "scale_jitter": 0.05,
    "coordinate_jitter": 0.01,
    "warmup_epochs": 3,
    "patience": 12,
    "precision": "bf16",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_label(value: Any) -> Optional[str]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {
        "", "-1", "none", "nan", "null", "unknown", "unk", "?",
        "unlabeled", "unlabelled",
    }:
        return None
    return text


def normalize_xyz(xyz: np.ndarray, scale: float = 200.0) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"xyz must be [N,>=3], got {xyz.shape}")
    xyz = xyz[:, :3]
    if not np.isfinite(xyz).all():
        raise ValueError("xyz contains NaN or Inf")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid coordinate scale: {scale}")
    return (xyz - np.median(xyz, axis=0, keepdims=True)) / float(scale)


def unique_label_map(labels: Sequence[Optional[str]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, label in enumerate(labels):
        if label is None:
            continue
        if label in mapping:
            raise ValueError(f"Duplicate clean identity within worm: {label}")
        mapping[label] = index
    return mapping


@dataclass
class WormRecord:
    worm_id: str
    path: Path
    xyz: np.ndarray
    activity: np.ndarray
    labels: list[Optional[str]]


@dataclass
class MetricAccumulator:
    num_queries: int = 0
    top1_hits: int = 0
    top3_hits: int = 0
    top5_hits: int = 0
    top10_hits: int = 0
    reciprocal_rank_sum: float = 0.0
    rank_sum: float = 0.0
    ranks: list[int] = field(default_factory=list)
    assignment_queries: int = 0
    assignment_hits: int = 0

    def add_rank(self, rank: int) -> None:
        rank = int(rank)
        self.num_queries += 1
        self.top1_hits += int(rank <= 1)
        self.top3_hits += int(rank <= 3)
        self.top5_hits += int(rank <= 5)
        self.top10_hits += int(rank <= 10)
        self.reciprocal_rank_sum += 1.0 / max(rank, 1)
        self.rank_sum += rank
        self.ranks.append(rank)

    def add_assignment(self, hits: int, queries: int) -> None:
        self.assignment_hits += int(hits)
        self.assignment_queries += int(queries)

    def to_dict(self) -> dict[str, Any]:
        if not self.num_queries:
            raise AssertionError("No scoreable queries")
        return {
            "queries": int(self.num_queries),
            "num_queries": int(self.num_queries),
            "ranking_top1": self.top1_hits / self.num_queries,
            "top3": self.top3_hits / self.num_queries,
            "top5": self.top5_hits / self.num_queries,
            "top10": self.top10_hits / self.num_queries,
            "mrr": self.reciprocal_rank_sum / self.num_queries,
            "mean_rank": self.rank_sum / self.num_queries,
            "median_rank": float(np.median(np.asarray(self.ranks))),
            "assignment_queries": int(self.assignment_queries),
            "assignment_hits": int(self.assignment_hits),
            "assignment_top1": (
                self.assignment_hits / self.assignment_queries
                if self.assignment_queries else float("nan")
            ),
        }


def evaluate_direction(
    accumulator: MetricAccumulator,
    score_matrix: np.ndarray,
    query: WormRecord,
    reference: WormRecord,
    method: str,
    fold: int,
) -> list[dict[str, Any]]:
    """Evaluate rows=query neurons, columns=all reference neurons."""
    scores = np.asarray(score_matrix, dtype=np.float64)
    expected = (len(query.labels), len(reference.labels))
    if scores.shape != expected:
        raise ValueError(f"score shape {scores.shape}, expected {expected}")
    if not np.isfinite(scores).all():
        raise ValueError(f"{method}: score matrix contains NaN/Inf")

    query_map = unique_label_map(query.labels)
    reference_map = unique_label_map(reference.labels)
    shared = sorted(set(query_map).intersection(reference_map))
    if not shared:
        return []

    row_ind, col_ind = linear_sum_assignment(-scores)
    assignment = {int(row): int(col) for row, col in zip(row_ind, col_ind)}
    assignment_hits = 0
    rows: list[dict[str, Any]] = []
    for identity in shared:
        row = query_map[identity]
        correct_col = reference_map[identity]
        row_scores = scores[row]
        correct_score = float(row_scores[correct_col])
        # Match the existing FINDA evaluator: ties at the correct score receive
        # the optimistic rank 1 + number of strictly larger scores.
        rank = 1 + int(np.count_nonzero(row_scores > correct_score))
        accumulator.add_rank(rank)
        top_col = int(np.argmax(row_scores))
        assigned_col = int(assignment.get(row, -1))
        assignment_correct = (
            assigned_col >= 0 and reference.labels[assigned_col] == identity
        )
        assignment_hits += int(assignment_correct)
        rows.append(
            {
                "fold": int(fold),
                "method": method,
                "query_worm": query.worm_id,
                "reference_worm": reference.worm_id,
                "query_index": int(row),
                "identity": identity,
                "correct_candidate_index": int(correct_col),
                "predicted_candidate_index": top_col,
                "predicted_identity": reference.labels[top_col] or "",
                "rank": rank,
                "correct_score": correct_score,
                "top_score": float(row_scores[top_col]),
                "candidate_count": int(scores.shape[1]),
                "hungarian_candidate_index": assigned_col,
                "hungarian_correct": bool(assignment_correct),
            }
        )
    accumulator.add_assignment(assignment_hits, len(shared))
    return rows


def cosine_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def euclidean_scores(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return -cdist(np.asarray(a), np.asarray(b), metric="euclidean")


def _initial_sigma2(fixed: np.ndarray, moving: np.ndarray) -> float:
    d = fixed.shape[1]
    value = np.square(fixed[None, :, :] - moving[:, None, :]).sum()
    return max(float(value / (d * fixed.shape[0] * moving.shape[0])), 1e-8)


def _expectation(
    fixed: np.ndarray,
    transformed: np.ndarray,
    sigma2: float,
    outlier_weight: float,
) -> np.ndarray:
    """Return CPD posterior P with shape [moving, fixed]."""
    m, d = transformed.shape
    n = fixed.shape[0]
    distance2 = np.square(
        transformed[:, None, :] - fixed[None, :, :]
    ).sum(axis=2)
    numerator = np.exp(-distance2 / max(2.0 * sigma2, 1e-12))
    c = (
        (2.0 * math.pi * sigma2) ** (0.5 * d)
        * outlier_weight
        / max(1.0 - outlier_weight, 1e-12)
        * m
        / n
    )
    denominator = numerator.sum(axis=0, keepdims=True) + c
    return numerator / np.maximum(denominator, 1e-300)


def rigid_cpd_xy(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    outlier_weight: float = 0.1,
    max_iterations: int = 250,
    sigma2_tolerance: float = 1e-7,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rigid CPD restricted to an XY rotation and XYZ translation."""
    fixed = np.asarray(fixed, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    if fixed.shape[1] != 3 or moving.shape[1] != 3:
        raise ValueError("Rigid CPD expects three-dimensional coordinates")
    transformed = moving.copy()
    sigma2 = _initial_sigma2(fixed, moving)
    rotation = np.eye(3, dtype=np.float64)
    translation = np.zeros(3, dtype=np.float64)
    iterations = 0
    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        posterior = _expectation(
            fixed, transformed, sigma2, outlier_weight
        )
        p1 = posterior.sum(axis=1)
        pt1 = posterior.sum(axis=0)
        np_total = float(p1.sum())
        if np_total <= 1e-10:
            raise RuntimeError("Rigid CPD posterior collapsed")
        mu_fixed = (pt1[:, None] * fixed).sum(axis=0) / np_total
        mu_moving = (p1[:, None] * moving).sum(axis=0) / np_total
        x_hat = fixed - mu_fixed
        y_hat = moving - mu_moving

        cross_xy = y_hat[:, :2].T @ posterior @ x_hat[:, :2]
        u, _, vt = np.linalg.svd(cross_xy, full_matrices=False)
        correction = np.eye(2)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
        rotation_xy = u @ correction @ vt
        rotation = np.eye(3, dtype=np.float64)
        rotation[:2, :2] = rotation_xy
        translation = mu_fixed - mu_moving @ rotation
        transformed = moving @ rotation + translation

        residual2 = np.square(
            transformed[:, None, :] - fixed[None, :, :]
        ).sum(axis=2)
        sigma2 = max(
            float((posterior * residual2).sum() / (np_total * 3.0)),
            1e-12,
        )
        if sigma2 <= sigma2_tolerance:
            break
    return transformed, {
        "iterations": iterations,
        "sigma2": float(sigma2),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
    }


def nonrigid_cpd(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    outlier_weight: float = 0.1,
    regularization: float = 4000.0,
    beta: float = 0.25,
    max_iterations: int = 150,
    sigma2_tolerance: float = 1e-5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standard Gaussian-kernel non-rigid CPD."""
    fixed = np.asarray(fixed, dtype=np.float64)
    moving = np.asarray(moving, dtype=np.float64)
    kernel_distance2 = np.square(
        moving[:, None, :] - moving[None, :, :]
    ).sum(axis=2)
    kernel = np.exp(-kernel_distance2 / (2.0 * beta * beta))
    weights = np.zeros_like(moving)
    transformed = moving.copy()
    sigma2 = _initial_sigma2(fixed, moving)
    iterations = 0
    identity = np.eye(moving.shape[0], dtype=np.float64)
    for iteration in range(int(max_iterations)):
        iterations = iteration + 1
        posterior = _expectation(
            fixed, transformed, sigma2, outlier_weight
        )
        p1 = posterior.sum(axis=1)
        np_total = float(p1.sum())
        if np_total <= 1e-10:
            raise RuntimeError("Non-rigid CPD posterior collapsed")
        px = posterior @ fixed
        system = p1[:, None] * kernel + regularization * sigma2 * identity
        target = px - p1[:, None] * moving
        try:
            weights = np.linalg.solve(system, target)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(system, target, rcond=None)[0]
        transformed = moving + kernel @ weights
        residual2 = np.square(
            transformed[:, None, :] - fixed[None, :, :]
        ).sum(axis=2)
        sigma2 = max(
            float((posterior * residual2).sum() / (np_total * fixed.shape[1])),
            1e-12,
        )
        if sigma2 <= sigma2_tolerance:
            break
    return transformed, {
        "iterations": iterations,
        "sigma2": float(sigma2),
        "mean_displacement": float(
            np.linalg.norm(transformed - moving, axis=1).mean()
        ),
    }


def cpd_scores(
    query_xyz: np.ndarray,
    reference_xyz: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Register query (moving) to reference (fixed), then score all pairs."""
    rigid_cfg = CPD_CONFIG["rigid"]
    nonrigid_cfg = CPD_CONFIG["nonrigid"]
    rigid, rigid_info = rigid_cpd_xy(
        reference_xyz,
        query_xyz,
        outlier_weight=float(rigid_cfg["outlier_weight"]),
        max_iterations=int(rigid_cfg["max_iterations"]),
        sigma2_tolerance=float(rigid_cfg["sigma2_tolerance"]),
    )
    transformed, nonrigid_info = nonrigid_cpd(
        reference_xyz,
        rigid,
        outlier_weight=float(nonrigid_cfg["outlier_weight"]),
        regularization=float(nonrigid_cfg["lambda"]),
        beta=float(nonrigid_cfg["beta"]),
        max_iterations=int(nonrigid_cfg["max_iterations"]),
        sigma2_tolerance=float(nonrigid_cfg["sigma2_tolerance"]),
    )
    return euclidean_scores(transformed, reference_xyz), {
        "rigid": rigid_info,
        "nonrigid": nonrigid_info,
    }


def load_embedding_record(path: Path) -> WormRecord:
    with np.load(path, allow_pickle=True) as data:
        required = {"xyz", "nuclr_emb", "labels", "worm_id"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{path}: missing {sorted(missing)}")
        xyz = normalize_xyz(np.asarray(data["xyz"], dtype=np.float64))
        activity = np.asarray(data["nuclr_emb"], dtype=np.float64)
        raw_labels = np.asarray(data["labels"]).reshape(-1)
        worm_raw = np.asarray(data["worm_id"]).reshape(()).item()
        if isinstance(worm_raw, bytes):
            worm_raw = worm_raw.decode("utf-8", errors="replace")
    if xyz.shape[0] != activity.shape[0] or xyz.shape[0] != len(raw_labels):
        raise ValueError(f"{path}: row count mismatch")
    if activity.ndim != 2 or not np.isfinite(activity).all():
        raise ValueError(f"{path}: invalid activity embedding")
    return WormRecord(
        worm_id=str(worm_raw),
        path=path.resolve(),
        xyz=xyz,
        activity=activity,
        labels=[normalize_label(value) for value in raw_labels],
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
