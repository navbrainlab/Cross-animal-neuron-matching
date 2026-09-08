#!/usr/bin/env python3
"""Activity-only GWOT-MD baseline for the grouped Atanas CV5 split.

The experiment is intentionally leakage-safe:

* every neuron in the same coordinate-valid cohort as MPRT-Net is retained;
* xyz is used only to define that cohort and never enters the matching cost;
* labels and quality masks are used only for train-fold h selection and scoring;
* validation is evaluated after h is selected; test is never opened by this file.

GWOT-MD represents a recording by delayed, directed activity-distance matrices.
For a maximum lag h we use equally weighted {D(tau): -h <= tau <= h}; h=0 is
ordinary activity GWOT. Matching minimizes the mean squared structural
distortion over the relation set with a conditional-gradient Gromov-Wasserstein
solver. ``--lag-stride 1`` is the paper-faithful lag grid; a larger stride is an
explicit speed/accuracy trade-off intended only for a preliminary pilot.

Dependencies: numpy, scipy, POT (``pip install POT``).  A tiny scipy.linprog
fallback exists only so ``self-check`` works before POT is installed.

The ``majority-cv5`` command adds the paper's Appendix-F identification
protocol, adapted without leakage to the fixed CV5 split.  For every fold it
draws unique 9-teacher subsets from train, selects h by leave-one-out
identification inside those teachers, and evaluates the held-out validation
worms.  Each teacher contributes its Top-v matched labels (v=5 by default),
and the true label is correct when it is among the k most frequent non-noID
labels (k=5 by default).  The paper-style headline is the median accuracy over
individual-worm evaluations, not the direct pairwise Top-5 above.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment, linprog
from scipy.signal import butter, sosfiltfilt


INVALID_IDS = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "unk",
    "unlabeled",
    "unlabelled",
    "-1",
    "-1.0",
    "noid",
}
DEFAULT_SPLIT_ROOT = Path(
    "/home/ubuntu/klb/nuclr/nuclr/Data/Atanas_SF_unified_000776/"
    "cv5_grouped_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/home/ubuntu/klb/nuclr/nuclr/runs/gwot_md_atanas_cv5_v1"
)
PAPER_URL = "https://openreview.net/forum?id=qAgQqVVwq9"


@dataclass(frozen=True)
class Worm:
    uid: str
    source_path: str
    activity: np.ndarray
    cell_ids: tuple[str, ...]
    supervised_mask: np.ndarray
    sample_rate_hz: float
    timestamp_key: str | None

    @property
    def num_nodes(self) -> int:
        return int(self.activity.shape[0])


@dataclass
class Counts:
    queries: int = 0
    top1_correct: float = 0.0
    top5_correct: float = 0.0
    reciprocal_rank_sum: float = 0.0
    tied_queries: int = 0
    zero_target_queries: int = 0
    hungarian_queries: int = 0
    hungarian_correct: int = 0
    eligible_unique: int = 0

    def add(self, other: "Counts") -> None:
        self.queries += other.queries
        self.top1_correct += other.top1_correct
        self.top5_correct += other.top5_correct
        self.reciprocal_rank_sum += other.reciprocal_rank_sum
        self.tied_queries += other.tied_queries
        self.zero_target_queries += other.zero_target_queries
        self.hungarian_queries += other.hungarian_queries
        self.hungarian_correct += other.hungarian_correct
        self.eligible_unique += other.eligible_unique

    def metrics(self) -> dict[str, float | int]:
        q = max(self.queries, 1)
        hq = max(self.hungarian_queries, 1)
        eligible = max(self.eligible_unique, 1)
        return {
            "queries": self.queries,
            "top1": self.top1_correct / q,
            "top5": self.top5_correct / q,
            "mrr": self.reciprocal_rank_sum / q,
            "tie_affected_queries": self.tied_queries,
            "tie_affected_rate": self.tied_queries / q,
            "zero_target_queries": self.zero_target_queries,
            "zero_target_rate": self.zero_target_queries / q,
            "hungarian_queries": self.hungarian_queries,
            "hungarian_accuracy": self.hungarian_correct / hq,
            "eligible_unique": self.eligible_unique,
            "coverage": self.queries / eligible,
        }


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{k: json_ready(v) for k, v in row.items()} for row in rows])
    os.replace(temporary, path)


def as_bool(z: np.lib.npyio.NpzFile, key: str, n: int, default: bool) -> np.ndarray:
    if key not in z.files:
        return np.full(n, default, dtype=bool)
    value = np.asarray(z[key], dtype=bool)
    if value.shape != (n,):
        raise ValueError(f"{key} must have shape ({n},), got {value.shape}")
    return value


def valid_id(value: str) -> bool:
    return value.strip().lower() not in INVALID_IDS


def infer_sample_rate(
    z: np.lib.npyio.NpzFile, trace_length: int, fallback_hz: float
) -> tuple[float, str | None]:
    for key in ("timestamps", "timestamp", "time", "t"):
        if key not in z.files:
            continue
        stamps = np.asarray(z[key], dtype=np.float64).reshape(-1)
        if stamps.size != trace_length:
            continue
        delta = np.diff(stamps)
        delta = delta[np.isfinite(delta) & (delta > 0)]
        if delta.size:
            return float(1.0 / np.median(delta)), key
    if not math.isfinite(fallback_hz) or fallback_hz <= 0:
        raise ValueError("No valid timestamps and --fallback-sample-rate-hz is invalid")
    return float(fallback_hz), None


def load_worm(path: Path, fallback_sample_rate_hz: float) -> Worm:
    with np.load(path, allow_pickle=False) as z:
        required = {"activity_raw", "xyz", "cell_id"}
        missing = required.difference(z.files)
        if missing:
            raise KeyError(f"{path} is missing keys: {sorted(missing)}")

        xyz = np.asarray(z["xyz"], dtype=np.float64)
        activity = np.asarray(z["activity_raw"], dtype=np.float64)
        cell_ids_raw = np.asarray(z["cell_id"]).astype(str)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be [N,3], got {xyz.shape} in {path}")
        n = xyz.shape[0]
        if activity.ndim != 2:
            raise ValueError(f"activity_raw must be 2-D in {path}")
        if activity.shape[0] != n and activity.shape[1] == n:
            activity = activity.T
        if activity.shape[0] != n or cell_ids_raw.shape != (n,):
            raise ValueError(
                f"Node-count mismatch in {path}: xyz={xyz.shape}, "
                f"activity={activity.shape}, cell_id={cell_ids_raw.shape}"
            )
        sample_rate_hz, timestamp_key = infer_sample_rate(
            z, activity.shape[1], fallback_sample_rate_hz
        )
        finite_xyz = np.isfinite(xyz).all(axis=1)
        finite_xyz &= as_bool(z, "valid_xyz_mask", n, True)
        if finite_xyz.sum() < 2:
            raise ValueError(f"Fewer than two coordinate-valid neurons in {path}")

        labeled = as_bool(z, "labeled_mask", n, True)
        certain = as_bool(z, "certain_mask", n, True)
        clean = as_bool(z, "clean_mask", n, True)
        ids_valid = np.asarray([valid_id(x) for x in cell_ids_raw], dtype=bool)
        supervised = labeled & certain & clean & ids_valid
        uid = str(z["recording_uid"].item()) if "recording_uid" in z.files else path.stem

    activity = np.ascontiguousarray(activity[finite_xyz], dtype=np.float64)
    ids = tuple(x.strip() for x in cell_ids_raw[finite_xyz])
    supervised = supervised[finite_xyz]
    finite = np.isfinite(activity)
    if not finite.all():
        row_median = np.nanmedian(np.where(finite, activity, np.nan), axis=1)
        row_median = np.nan_to_num(row_median, nan=0.0)
        activity = np.where(finite, activity, row_median[:, None])
    return Worm(
        uid=uid,
        source_path=str(path),
        activity=activity,
        cell_ids=ids,
        supervised_mask=np.asarray(supervised, dtype=bool),
        sample_rate_hz=sample_rate_hz,
        timestamp_key=timestamp_key,
    )


def unique_identity_map(worm: Worm) -> dict[str, int]:
    candidates = [
        identity
        for identity, keep in zip(worm.cell_ids, worm.supervised_mask)
        if keep
    ]
    counts = Counter(candidates)
    return {
        identity: index
        for index, (identity, keep) in enumerate(
            zip(worm.cell_ids, worm.supervised_mask)
        )
        if keep and counts[identity] == 1
    }


def highpass_activity(
    worm: Worm, cutoff_hz: float, normalization: str
) -> np.ndarray:
    x = np.asarray(worm.activity, dtype=np.float64)
    if cutoff_hz <= 0:
        filtered = x.copy()
    else:
        nyquist = 0.5 * worm.sample_rate_hz
        if not cutoff_hz < nyquist:
            raise ValueError(
                f"High-pass cutoff {cutoff_hz} Hz is not below Nyquist "
                f"{nyquist:.6g} Hz for {worm.uid}"
            )
        sos = butter(1, cutoff_hz / nyquist, btype="highpass", output="sos")
        default_pad = 3 * (2 * sos.shape[0] + 1)
        if x.shape[1] <= default_pad:
            raise ValueError(f"Trace too short for zero-phase filter in {worm.uid}")
        filtered = sosfiltfilt(sos, x, axis=1)
    if normalization == "zscore":
        filtered -= filtered.mean(axis=1, keepdims=True)
        scale = filtered.std(axis=1, keepdims=True)
        filtered /= np.where(scale > 1e-12, scale, 1.0)
    elif normalization != "none":
        raise ValueError(f"Unknown activity normalization: {normalization}")
    return np.ascontiguousarray(filtered)


def normalize_distance(matrix: np.ndarray, mode: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if mode == "none":
        scale = 1.0
    elif mode == "rms":
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        scale = float(np.sqrt(np.mean(positive * positive))) if positive.size else 1.0
    elif mode == "mean":
        positive = matrix[np.isfinite(matrix) & (matrix > 0)]
        scale = float(np.mean(positive)) if positive.size else 1.0
    else:
        raise ValueError(f"Unknown distance normalization: {mode}")
    return np.ascontiguousarray(matrix / max(scale, 1e-12))


def delayed_distance(x: np.ndarray, lag: int, normalization: str) -> np.ndarray:
    if lag < 0:
        return delayed_distance(x, -lag, normalization).T
    if lag >= x.shape[1] - 2:
        raise ValueError(f"Lag {lag} is too large for trace length {x.shape[1]}")
    left = x[:, : x.shape[1] - lag] if lag else x
    right = x[:, lag:] if lag else x
    length = left.shape[1]
    left_norm = np.einsum("it,it->i", left, left) / length
    right_norm = np.einsum("it,it->i", right, right) / length
    squared = left_norm[:, None] + right_norm[None, :] - 2.0 * (left @ right.T) / length
    distance = np.sqrt(np.maximum(squared, 0.0))
    return normalize_distance(distance, normalization)


def build_relations(
    x: np.ndarray, h: int, normalization: str, lag_stride: int
) -> tuple[np.ndarray, ...]:
    if lag_stride <= 0:
        raise ValueError("lag_stride must be positive")
    if h == 0:
        lags = [0]
    else:
        # Always include both endpoints and zero even for an exploratory stride
        # that does not divide h.  Default stride=1 includes every integer lag.
        lags = sorted(set(range(-h, h + 1, lag_stride)).union({-h, 0, h}))
    positive_cache: dict[int, np.ndarray] = {}
    relations: list[np.ndarray] = []
    for lag in lags:
        absolute = abs(lag)
        if absolute not in positive_cache:
            positive_cache[absolute] = delayed_distance(x, absolute, normalization)
        matrix = positive_cache[absolute]
        relations.append(matrix if lag >= 0 else matrix.T.copy())
    return tuple(relations)


def relation_cost_right(
    coupling: np.ndarray,
    relations_a: Sequence[np.ndarray],
    relations_b: Sequence[np.ndarray],
) -> np.ndarray:
    """Return C where <C,V> = Q(coupling,V)."""
    row_mass = coupling.sum(axis=1)
    col_mass = coupling.sum(axis=0)
    result = np.zeros_like(coupling)
    for a, b in zip(relations_a, relations_b):
        a_term = ((a * a).T @ row_mass)[:, None]
        b_term = ((b * b).T @ col_mass)[None, :]
        result += a_term + b_term - 2.0 * (a.T @ coupling @ b)
    return result / len(relations_a)


def relation_cost_left(
    coupling: np.ndarray,
    relations_a: Sequence[np.ndarray],
    relations_b: Sequence[np.ndarray],
) -> np.ndarray:
    """Return C where <C,U> = Q(U,coupling)."""
    row_mass = coupling.sum(axis=1)
    col_mass = coupling.sum(axis=0)
    result = np.zeros_like(coupling)
    for a, b in zip(relations_a, relations_b):
        a_term = ((a * a) @ row_mass)[:, None]
        b_term = ((b * b) @ col_mass)[None, :]
        result += a_term + b_term - 2.0 * (a @ coupling @ b.T)
    return result / len(relations_a)


def bilinear_objective(
    left: np.ndarray,
    right: np.ndarray,
    relations_a: Sequence[np.ndarray],
    relations_b: Sequence[np.ndarray],
) -> float:
    return float(np.sum(right * relation_cost_right(left, relations_a, relations_b)))


def gw_objective(
    coupling: np.ndarray,
    relations_a: Sequence[np.ndarray],
    relations_b: Sequence[np.ndarray],
) -> float:
    return bilinear_objective(coupling, coupling, relations_a, relations_b)


class RelationOperator:
    """GW-MD contractions, optionally batched on a CUDA device."""

    def __init__(
        self,
        relations_a: Sequence[np.ndarray],
        relations_b: Sequence[np.ndarray],
        requested_device: str,
    ) -> None:
        self.relations_a = relations_a
        self.relations_b = relations_b
        self.backend = "numpy"
        self.device = "cpu"
        self.torch = None
        self.a_t = None
        self.b_t = None
        use_cuda = requested_device == "auto" or requested_device.startswith("cuda")
        if use_cuda:
            try:
                import torch  # type: ignore
            except ModuleNotFoundError:
                if requested_device != "auto":
                    raise RuntimeError(
                        f"--device {requested_device} requires PyTorch with CUDA support"
                    ) from None
                return
            if not torch.cuda.is_available():
                if requested_device != "auto":
                    raise RuntimeError(
                        f"--device {requested_device} requested but CUDA is unavailable"
                    )
                return
            device = "cuda:0" if requested_device == "auto" else requested_device
            self.torch = torch
            self.device = device
            self.backend = "torch"
            self.a_t = torch.as_tensor(
                np.stack(relations_a), dtype=torch.float32, device=device
            )
            self.b_t = torch.as_tensor(
                np.stack(relations_b), dtype=torch.float32, device=device
            )

    def right(self, coupling: np.ndarray) -> np.ndarray:
        if self.backend == "numpy":
            return relation_cost_right(coupling, self.relations_a, self.relations_b)
        torch, a, b = self.torch, self.a_t, self.b_t
        assert torch is not None and a is not None and b is not None
        u = torch.as_tensor(coupling, dtype=torch.float32, device=self.device)
        row_mass, col_mass = u.sum(1), u.sum(0)
        a_term = torch.einsum("rik,i->rk", a * a, row_mass)
        b_term = torch.einsum("rjl,j->rl", b * b, col_mass)
        cross = torch.matmul(torch.matmul(a.transpose(1, 2), u), b)
        result = (a_term[:, :, None] + b_term[:, None, :] - 2.0 * cross).mean(0)
        return result.detach().cpu().numpy().astype(np.float64, copy=False)

    def left(self, coupling: np.ndarray) -> np.ndarray:
        if self.backend == "numpy":
            return relation_cost_left(coupling, self.relations_a, self.relations_b)
        torch, a, b = self.torch, self.a_t, self.b_t
        assert torch is not None and a is not None and b is not None
        u = torch.as_tensor(coupling, dtype=torch.float32, device=self.device)
        row_mass, col_mass = u.sum(1), u.sum(0)
        a_term = torch.einsum("rik,k->ri", a * a, row_mass)
        b_term = torch.einsum("rjl,l->rj", b * b, col_mass)
        cross = torch.matmul(torch.matmul(a, u), b.transpose(1, 2))
        result = (a_term[:, :, None] + b_term[:, None, :] - 2.0 * cross).mean(0)
        return result.detach().cpu().numpy().astype(np.float64, copy=False)

    def bilinear(self, left: np.ndarray, right: np.ndarray) -> float:
        return float(np.sum(right * self.right(left)))

    def objective(self, coupling: np.ndarray) -> float:
        return self.bilinear(coupling, coupling)


def emd_plan(a: np.ndarray, b: np.ndarray, cost: np.ndarray) -> np.ndarray:
    shifted = np.asarray(cost, dtype=np.float64)
    shifted = shifted - np.nanmin(shifted)
    scale = float(np.nanmax(shifted))
    if scale > 0:
        shifted = shifted / scale
    try:
        import ot  # type: ignore

        plan = ot.emd(a, b, shifted, numItermax=200000)
        return np.asarray(plan, dtype=np.float64)
    except ModuleNotFoundError:
        if cost.size > 1600:
            raise RuntimeError(
                "POT is required for real Atanas runs. Install it inside nuclr310 "
                "with: pip install POT"
            ) from None

    n, m = cost.shape
    c = shifted.reshape(-1)
    constraints = np.zeros((n + m, n * m), dtype=np.float64)
    for i in range(n):
        constraints[i, i * m : (i + 1) * m] = 1.0
    for j in range(m):
        constraints[n + j, j::m] = 1.0
    result = linprog(
        c,
        A_eq=constraints[:-1],
        b_eq=np.concatenate([a, b])[:-1],
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"linprog OT fallback failed: {result.message}")
    return result.x.reshape(n, m)


def structural_signature(relations: Sequence[np.ndarray]) -> np.ndarray:
    features: list[np.ndarray] = []
    for relation in relations:
        features.extend(
            [
                relation.mean(axis=1),
                relation.std(axis=1),
                relation.mean(axis=0),
                relation.std(axis=0),
            ]
        )
    signature = np.stack(features, axis=1)
    center = signature.mean(axis=0, keepdims=True)
    scale = signature.std(axis=0, keepdims=True)
    return (signature - center) / np.where(scale > 1e-12, scale, 1.0)


def initial_couplings(
    relations_a: Sequence[np.ndarray], relations_b: Sequence[np.ndarray]
) -> list[tuple[str, np.ndarray]]:
    n, m = relations_a[0].shape[0], relations_b[0].shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    uniform = a[:, None] * b[None, :]
    sig_a, sig_b = structural_signature(relations_a), structural_signature(relations_b)
    signature_cost = (
        np.einsum("if,if->i", sig_a, sig_a)[:, None]
        + np.einsum("jf,jf->j", sig_b, sig_b)[None, :]
        - 2.0 * sig_a @ sig_b.T
    )
    signature = emd_plan(a, b, np.maximum(signature_cost, 0.0))
    return [("uniform", uniform), ("signature", signature)]


def solve_gwot_md(
    relations_a: Sequence[np.ndarray],
    relations_b: Sequence[np.ndarray],
    max_iter: int,
    tolerance: float,
    device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(relations_a) != len(relations_b) or not relations_a:
        raise ValueError("Relation sets must have the same nonzero size")
    n, m = relations_a[0].shape[0], relations_b[0].shape[0]
    mass_a = np.full(n, 1.0 / n)
    mass_b = np.full(m, 1.0 / m)
    operator = RelationOperator(relations_a, relations_b, device)
    candidates: list[tuple[float, np.ndarray, dict[str, Any]]] = []
    for initialization, start in initial_couplings(relations_a, relations_b):
        coupling = np.asarray(start, dtype=np.float64).copy()
        trace: list[float] = [operator.objective(coupling)]
        converged = False
        for iteration in range(1, max_iter + 1):
            gradient = operator.right(coupling)
            gradient += operator.left(coupling)
            vertex = emd_plan(mass_a, mass_b, gradient)
            direction = vertex - coupling
            linear = operator.bilinear(coupling, direction)
            linear += operator.bilinear(direction, coupling)
            quadratic = operator.bilinear(direction, direction)
            if quadratic > 0:
                step = float(np.clip(-linear / (2.0 * quadratic), 0.0, 1.0))
            else:
                step = 1.0 if linear + quadratic < 0 else 0.0
            update = step * direction
            coupling += update
            coupling = np.maximum(coupling, 0.0)
            objective = operator.objective(coupling)
            trace.append(objective)
            if np.sum(np.abs(update)) <= tolerance:
                converged = True
                break
        diagnostics = {
            "initialization": initialization,
            "iterations": iteration,
            "converged": converged,
            "objective": trace[-1],
            "initial_objective": trace[0],
            "contraction_backend": operator.backend,
            "contraction_device": operator.device,
            "row_marginal_max_error": float(np.max(np.abs(coupling.sum(1) - mass_a))),
            "col_marginal_max_error": float(np.max(np.abs(coupling.sum(0) - mass_b))),
        }
        candidates.append((trace[-1], coupling, diagnostics))
    candidates.sort(key=lambda x: x[0])
    best = candidates[0]
    diagnostics = dict(best[2])
    diagnostics["starts"] = [item[2] for item in candidates]
    return best[1], diagnostics


def directional_counts(scores: np.ndarray, mapping: dict[int, int]) -> Counts:
    counts = Counts()
    for source, target in mapping.items():
        row = scores[source]
        target_score = row[target]
        tolerance = max(1e-15, 1e-12 * float(np.max(np.abs(row))))
        greater = int(np.sum(row > target_score + tolerance))
        equal = int(np.sum(np.abs(row - target_score) <= tolerance))
        equal = max(equal, 1)
        # Exact EMD plans are sparse, so most unselected candidates are tied at
        # zero.  Optimistically assigning all ties the best rank makes Top-5
        # approach 100%.  Instead, report the expectation under a uniform random
        # ordering inside each tie block.  With all scores tied this reduces to
        # the correct chance values k/N and mean reciprocal rank at chance.
        top1_credit = float(np.clip(1 - greater, 0, equal)) / equal
        top5_credit = float(np.clip(5 - greater, 0, equal)) / equal
        reciprocal_credit = float(
            np.mean(1.0 / np.arange(greater + 1, greater + equal + 1))
        )
        counts.queries += 1
        counts.top1_correct += top1_credit
        counts.top5_correct += top5_credit
        counts.reciprocal_rank_sum += reciprocal_credit
        counts.tied_queries += int(equal > 1)
        counts.zero_target_queries += int(abs(float(target_score)) <= tolerance)
    return counts


def pair_counts(worm_a: Worm, worm_b: Worm, coupling: np.ndarray) -> Counts:
    ids_a, ids_b = unique_identity_map(worm_a), unique_identity_map(worm_b)
    common = sorted(set(ids_a).intersection(ids_b))
    map_a_to_b = {ids_a[identity]: ids_b[identity] for identity in common}
    map_b_to_a = {ids_b[identity]: ids_a[identity] for identity in common}
    totals = directional_counts(coupling, map_a_to_b)
    totals.add(directional_counts(coupling.T, map_b_to_a))
    row, col = linear_sum_assignment(-coupling)
    assignment = {int(i): int(j) for i, j in zip(row, col)}
    inverse = {j: i for i, j in assignment.items()}
    for source, target in map_a_to_b.items():
        totals.hungarian_queries += 1
        totals.hungarian_correct += int(assignment.get(source, -1) == target)
    for source, target in map_b_to_a.items():
        totals.hungarian_queries += 1
        totals.hungarian_correct += int(inverse.get(source, -1) == target)
    totals.eligible_unique = len(ids_a) + len(ids_b)
    return totals


def build_label_encoding(
    worms: Sequence[Worm],
) -> tuple[list[str], dict[str, np.ndarray], dict[str, set[int]]]:
    """Encode unique, supervised neuron labels; every other node is noID=-1."""
    vocabulary = sorted(
        {
            identity
            for worm in worms
            for identity in unique_identity_map(worm)
        }
    )
    label_to_id = {identity: index for index, identity in enumerate(vocabulary)}
    node_ids: dict[str, np.ndarray] = {}
    presence: dict[str, set[int]] = {}
    for worm in worms:
        encoded = np.full(worm.num_nodes, -1, dtype=np.int32)
        for identity, node_index in unique_identity_map(worm).items():
            encoded[node_index] = label_to_id[identity]
        source_key = str(Path(worm.source_path).resolve())
        node_ids[source_key] = encoded
        presence[source_key] = set(int(x) for x in encoded if x >= 0)
    return vocabulary, node_ids, presence


def top_label_candidates(
    scores: np.ndarray,
    teacher_node_label_ids: np.ndarray,
    v: int,
) -> tuple[np.ndarray, int]:
    """Return paper-ordered Top-v teacher labels and a cutoff-tie audit count."""
    if scores.ndim != 2 or scores.shape[1] != teacher_node_label_ids.shape[0]:
        raise ValueError("Score/teacher-label shape mismatch")
    if v <= 0:
        raise ValueError("v must be positive")
    v_eff = min(v, scores.shape[1])
    # Appendix F ranks matched neurons. Stable sorting makes the otherwise
    # unspecified equal-score order deterministic and matches Counter order
    # downstream. noID nodes remain in the Top-v array as -1 and consume slots.
    order = np.argsort(-scores, axis=1, kind="mergesort")
    candidates = np.full((scores.shape[0], v), -1, dtype=np.int32)
    candidates[:, :v_eff] = teacher_node_label_ids[order[:, :v_eff]]

    cutoff_ties = 0
    if 0 < v_eff < scores.shape[1]:
        ranked = np.take_along_axis(scores, order[:, : v_eff + 1], axis=1)
        row_scale = np.max(np.abs(scores), axis=1)
        tolerance = np.maximum(1e-15, 1e-12 * row_scale)
        cutoff_ties = int(
            np.sum(np.abs(ranked[:, v_eff - 1] - ranked[:, v_eff]) <= tolerance)
        )
    return candidates, cutoff_ties


def query_ranks_from_votes(
    candidate_ids: np.ndarray,
    true_ids: np.ndarray,
    num_labels: int,
    tie_break: str,
) -> np.ndarray:
    """Rank true labels by Appendix-F vote counts; -1 denotes noID."""
    if candidate_ids.ndim != 2 or candidate_ids.shape[0] != true_ids.shape[0]:
        raise ValueError("Candidate/true-label shape mismatch")
    n_query, width = candidate_ids.shape
    counts = np.zeros((n_query, num_labels), dtype=np.int16)
    rows = np.broadcast_to(np.arange(n_query)[:, None], candidate_ids.shape)
    valid = candidate_ids >= 0
    np.add.at(counts, (rows[valid], candidate_ids[valid]), 1)

    ranks = np.full(n_query, np.inf, dtype=np.float64)
    labeled_rows = np.flatnonzero(true_ids >= 0)
    if labeled_rows.size == 0:
        return ranks
    true = true_ids[labeled_rows].astype(np.int64)
    true_counts = counts[labeled_rows, true]
    present = true_counts > 0
    if not np.any(present):
        return ranks

    active_rows = labeled_rows[present]
    active_true = true[present]
    active_counts = counts[active_rows]
    active_true_counts = true_counts[present]
    greater = np.sum(active_counts > active_true_counts[:, None], axis=1)

    if tie_break == "label":
        label_ids = np.arange(num_labels)[None, :]
        before = np.sum(
            (active_counts == active_true_counts[:, None])
            & (label_ids < active_true[:, None]),
            axis=1,
        )
    elif tie_break == "first":
        sentinel = width + 1
        first = np.full((n_query, num_labels), sentinel, dtype=np.int16)
        positions = np.broadcast_to(
            np.arange(width, dtype=np.int16)[None, :], candidate_ids.shape
        )
        np.minimum.at(first, (rows[valid], candidate_ids[valid]), positions[valid])
        active_first = first[active_rows]
        true_first = active_first[np.arange(len(active_rows)), active_true]
        before = np.sum(
            (active_counts == active_true_counts[:, None])
            & (active_first < true_first[:, None]),
            axis=1,
        )
    else:
        raise ValueError(f"Unsupported majority-vote tie break: {tie_break}")

    ranks[active_rows] = 1.0 + greater + before
    return ranks


def evaluate_majority_one_worm(
    query_path: Path,
    teacher_paths: Sequence[Path],
    candidates: Mapping[tuple[str, str], np.ndarray],
    node_label_ids: Mapping[str, np.ndarray],
    label_presence: Mapping[str, set[int]],
    num_labels: int,
    v: int,
    k: int,
    tie_break: str,
) -> dict[str, Any]:
    query_key = str(query_path.resolve())
    teacher_keys = [str(path.resolve()) for path in teacher_paths]
    blocks = [candidates[(query_key, teacher)][:, :v] for teacher in teacher_keys]
    votes = np.concatenate(blocks, axis=1)
    true_ids = node_label_ids[query_key]
    ranks = query_ranks_from_votes(votes, true_ids, num_labels, tie_break)
    labeled = true_ids >= 0
    labeled_true = true_ids[labeled]
    hit = ranks[labeled] <= k

    covered = np.asarray(
        [
            any(int(identity) in label_presence[teacher] for teacher in teacher_keys)
            for identity in labeled_true
        ],
        dtype=bool,
    )
    queries = int(np.sum(labeled))
    correct = int(np.sum(hit))
    covered_queries = int(np.sum(covered))
    covered_correct = int(np.sum(hit & covered))
    return {
        "num_queries": queries,
        "correct": correct,
        "accuracy": float(correct / queries) if queries else math.nan,
        "num_covered_queries": covered_queries,
        "coverage": float(covered_queries / queries) if queries else math.nan,
        "covered_correct": covered_correct,
        "covered_accuracy": (
            float(covered_correct / covered_queries) if covered_queries else math.nan
        ),
    }


def sample_unique_teacher_sets(
    train_files: Sequence[Path],
    teacher_count: int,
    num_sets: int,
    seed: int,
) -> list[tuple[Path, ...]]:
    if not (1 < teacher_count <= len(train_files)):
        raise ValueError("majority teacher count must be in [2, #train worms]")
    total = math.comb(len(train_files), teacher_count)
    if num_sets > total:
        raise ValueError(
            f"Requested {num_sets} teacher sets, but only {total} unique sets exist"
        )
    if num_sets == total and total <= 100000:
        return [tuple(train_files[i] for i in indices) for indices in combinations(range(len(train_files)), teacher_count)]

    rng = np.random.default_rng(seed)
    selected: dict[tuple[int, ...], None] = {}
    while len(selected) < num_sets:
        indices = tuple(
            sorted(
                int(x)
                for x in rng.choice(
                    len(train_files), size=teacher_count, replace=False
                )
            )
        )
        selected.setdefault(indices, None)
    return [tuple(train_files[i] for i in indices) for indices in selected]


def canonical_pair(a: Path, b: Path) -> tuple[Path, Path]:
    if a.resolve() == b.resolve():
        raise ValueError("A GWOT-MD pair must contain two different worms")
    return tuple(sorted((a, b), key=lambda path: str(path.resolve())))  # type: ignore[return-value]


def select_evenly(items: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    positions = np.linspace(0, len(items) - 1, limit)
    indices = sorted(set(int(round(x)) for x in positions))
    return [items[index] for index in indices]


def split_files(split_root: Path, fold: int, split: str) -> list[Path]:
    if split not in {"train", "val"}:
        raise ValueError("This leakage-safe runner only permits train or val")
    directory = split_root / f"fold_{fold}" / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing split directory: {directory}")
    files = sorted(directory.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files in {directory}")
    return files


def majority_cache_signature(
    args: argparse.Namespace,
    fold: int,
    files: Sequence[Path],
    vocabulary: Sequence[str],
) -> str:
    sources = []
    for path in sorted(files, key=lambda item: str(item.resolve())):
        stat = path.stat()
        sources.append(
            {
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = {
        "cache_schema_version": 1,
        "fold": fold,
        "sources": sources,
        "vocabulary": list(vocabulary),
        "highpass_cutoff_hz": args.highpass_cutoff_hz,
        "fallback_sample_rate_hz": args.fallback_sample_rate_hz,
        "activity_normalization": args.activity_normalization,
        "distance_normalization": args.distance_normalization,
        "lag_stride": args.lag_stride,
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
        "majority_v": args.majority_v,
        "solver": "two-start conditional-gradient exact-EMD",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class Experiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.worm_cache: dict[str, Worm] = {}
        self.activity_cache: dict[str, np.ndarray] = {}
        self.relation_cache: dict[tuple[str, int], tuple[np.ndarray, ...]] = {}

    def worm(self, path: Path) -> Worm:
        key = str(path.resolve())
        if key not in self.worm_cache:
            self.worm_cache[key] = load_worm(path, self.args.fallback_sample_rate_hz)
        return self.worm_cache[key]

    def relations(self, path: Path, h: int) -> tuple[np.ndarray, ...]:
        key = (str(path.resolve()), h)
        if key not in self.relation_cache:
            worm = self.worm(path)
            activity_key = key[0]
            if activity_key not in self.activity_cache:
                self.activity_cache[activity_key] = highpass_activity(
                    worm,
                    self.args.highpass_cutoff_hz,
                    self.args.activity_normalization,
                )
            self.relation_cache[key] = build_relations(
                self.activity_cache[activity_key],
                h,
                self.args.distance_normalization,
                self.args.lag_stride,
            )
        return self.relation_cache[key]

    def evaluate_pair(self, a: Path, b: Path, h: int) -> tuple[Counts, dict[str, Any]]:
        start = time.perf_counter()
        worm_a, worm_b = self.worm(a), self.worm(b)
        coupling, diagnostics = solve_gwot_md(
            self.relations(a, h),
            self.relations(b, h),
            max_iter=self.args.max_iter,
            tolerance=self.args.tolerance,
            device=self.args.device,
        )
        counts = pair_counts(worm_a, worm_b, coupling)
        row = {
            "worm_a": worm_a.uid,
            "worm_b": worm_b.uid,
            "path_a": str(a),
            "path_b": str(b),
            "h": h,
            "num_nodes_a": worm_a.num_nodes,
            "num_nodes_b": worm_b.num_nodes,
            **counts.metrics(),
            **{f"solver_{k}": v for k, v in diagnostics.items() if k != "starts"},
            "elapsed_seconds": time.perf_counter() - start,
        }
        return counts, row

    def clear_relations_for_h(self, h: int) -> None:
        stale = [key for key in self.relation_cache if key[1] == h]
        for key in stale:
            del self.relation_cache[key]

    def majority_pair_candidates(
        self,
        a: Path,
        b: Path,
        h: int,
        v: int,
        node_label_ids: Mapping[str, np.ndarray],
        cache_root: Path,
        cache_signature: str,
    ) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
        a, b = canonical_pair(a, b)
        key_a, key_b = str(a.resolve()), str(b.resolve())
        pair_digest = hashlib.sha256(
            (key_a + "\0" + key_b).encode("utf-8")
        ).hexdigest()[:24]
        cache_path = (
            cache_root
            / cache_signature[:16]
            / f"h_{h:03d}"
            / f"{pair_digest}.npz"
        )
        if cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as z:
                stored_signature = str(z["cache_signature"].item())
                stored_a = str(z["path_a"].item())
                stored_b = str(z["path_b"].item())
                stored_h = int(z["h"].item())
                stored_v = int(z["v"].item())
                if (
                    stored_signature != cache_signature
                    or stored_a != key_a
                    or stored_b != key_b
                    or stored_h != h
                    or stored_v != v
                ):
                    raise RuntimeError(f"Majority cache metadata mismatch: {cache_path}")
                top_a_to_b = np.asarray(z["top_a_to_b"], dtype=np.int32)
                top_b_to_a = np.asarray(z["top_b_to_a"], dtype=np.int32)
                audit = {
                    "cache_hit": True,
                    "queries_a_to_b": int(z["queries_a_to_b"].item()),
                    "queries_b_to_a": int(z["queries_b_to_a"].item()),
                    "cutoff_ties_a_to_b": int(z["cutoff_ties_a_to_b"].item()),
                    "cutoff_ties_b_to_a": int(z["cutoff_ties_b_to_a"].item()),
                }
            return {
                (key_a, key_b): top_a_to_b,
                (key_b, key_a): top_b_to_a,
            }, audit

        coupling, diagnostics = solve_gwot_md(
            self.relations(a, h),
            self.relations(b, h),
            max_iter=self.args.max_iter,
            tolerance=self.args.tolerance,
            device=self.args.device,
        )
        top_a_to_b, ties_a_to_b = top_label_candidates(
            coupling, node_label_ids[key_b], v
        )
        top_b_to_a, ties_b_to_a = top_label_candidates(
            coupling.T, node_label_ids[key_a], v
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(cache_path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            cache_signature=np.asarray(cache_signature),
            path_a=np.asarray(key_a),
            path_b=np.asarray(key_b),
            h=np.asarray(h, dtype=np.int32),
            v=np.asarray(v, dtype=np.int32),
            top_a_to_b=top_a_to_b,
            top_b_to_a=top_b_to_a,
            queries_a_to_b=np.asarray(top_a_to_b.shape[0], dtype=np.int32),
            queries_b_to_a=np.asarray(top_b_to_a.shape[0], dtype=np.int32),
            cutoff_ties_a_to_b=np.asarray(ties_a_to_b, dtype=np.int32),
            cutoff_ties_b_to_a=np.asarray(ties_b_to_a, dtype=np.int32),
            solver_objective=np.asarray(diagnostics["objective"], dtype=np.float64),
        )
        os.replace(temporary, cache_path)
        audit = {
            "cache_hit": False,
            "queries_a_to_b": top_a_to_b.shape[0],
            "queries_b_to_a": top_b_to_a.shape[0],
            "cutoff_ties_a_to_b": ties_a_to_b,
            "cutoff_ties_b_to_a": ties_b_to_a,
        }
        return {
            (key_a, key_b): top_a_to_b,
            (key_b, key_a): top_b_to_a,
        }, audit

    def prepare_majority_candidates(
        self,
        pairs: Sequence[tuple[Path, Path]],
        h: int,
        v: int,
        node_label_ids: Mapping[str, np.ndarray],
        cache_root: Path,
        cache_signature: str,
        log_prefix: str,
    ) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, int]]:
        unique_pairs = sorted(
            {canonical_pair(a, b) for a, b in pairs},
            key=lambda pair: (str(pair[0].resolve()), str(pair[1].resolve())),
        )
        candidates: dict[tuple[str, str], np.ndarray] = {}
        audit = {
            "unique_pairs": len(unique_pairs),
            "cache_hits": 0,
            "computed_pairs": 0,
            "directional_queries": 0,
            "cutoff_tied_queries": 0,
        }
        for index, (a, b) in enumerate(unique_pairs, 1):
            directional, row = self.majority_pair_candidates(
                a=a,
                b=b,
                h=h,
                v=v,
                node_label_ids=node_label_ids,
                cache_root=cache_root,
                cache_signature=cache_signature,
            )
            candidates.update(directional)
            audit["cache_hits"] += int(row["cache_hit"])
            audit["computed_pairs"] += int(not row["cache_hit"])
            audit["directional_queries"] += int(row["queries_a_to_b"])
            audit["directional_queries"] += int(row["queries_b_to_a"])
            audit["cutoff_tied_queries"] += int(row["cutoff_ties_a_to_b"])
            audit["cutoff_tied_queries"] += int(row["cutoff_ties_b_to_a"])
            if index == 1 or index % 10 == 0 or index == len(unique_pairs):
                log(
                    f"{log_prefix} h={h:02d} pairs={index}/{len(unique_pairs)} "
                    f"cache_hits={audit['cache_hits']}"
                )
        return candidates, audit

    def tune_h(self, fold: int, train_files: Sequence[Path]) -> dict[str, Any]:
        reference = train_files[0]
        partners = select_evenly(list(train_files[1:]), self.args.tune_max_partners)
        if not partners:
            raise ValueError(f"Fold {fold} needs at least two train worms")
        rows: list[dict[str, Any]] = []
        for h in self.args.h_candidates:
            totals = Counts()
            started = time.perf_counter()
            for index, partner in enumerate(partners, 1):
                counts, _ = self.evaluate_pair(reference, partner, h)
                totals.add(counts)
                log(
                    f"fold={fold} tune h={h:02d} pair={index}/{len(partners)} "
                    f"queries={counts.queries}"
                )
            rows.append(
                {
                    "h": h,
                    "reference": self.worm(reference).uid,
                    "partners": len(partners),
                    **totals.metrics(),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        eligible = [row for row in rows if int(row["queries"]) > 0]
        if not eligible:
            raise RuntimeError(f"Fold {fold}: no labeled train queries available to select h")
        selected = sorted(
            eligible,
            key=lambda row: (-float(row["top1"]), -float(row["mrr"]), int(row["h"])),
        )[0]
        return {
            "fold": fold,
            "selection_split": "train",
            "reference_file": str(reference),
            "partner_files": [str(x) for x in partners],
            "criterion": "query-weighted top1, then mrr, then smaller h",
            "selected_h": int(selected["h"]),
            "candidate_metrics": rows,
        }

    def evaluate_validation(
        self, fold: int, val_files: Sequence[Path], selected_h: int
    ) -> tuple[dict[str, Any], list[dict[str, Any]], Counts]:
        pairs = select_evenly(list(combinations(val_files, 2)), self.args.max_val_pairs)
        totals = Counts()
        rows: list[dict[str, Any]] = []
        started = time.perf_counter()
        for index, (a, b) in enumerate(pairs, 1):
            counts, row = self.evaluate_pair(a, b, selected_h)
            totals.add(counts)
            row = {"fold": fold, "split": "val", **row}
            rows.append(row)
            metrics = totals.metrics()
            log(
                f"fold={fold} val pair={index}/{len(pairs)} h={selected_h} "
                f"pooled_top1={float(metrics['top1']):.4f} queries={totals.queries}"
            )
        summary = {
            "fold": fold,
            "split": "val",
            "selected_h": selected_h,
            "num_worms": len(val_files),
            "num_pairs": len(pairs),
            **totals.metrics(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        return summary, rows, totals


def evaluate_majority_fold(
    experiment: Experiment,
    args: argparse.Namespace,
    fold: int,
    train_files: Sequence[Path],
    val_files: Sequence[Path],
    fold_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Appendix-F majority vote, with teacher selection confined to train."""
    v, k = args.majority_v, args.majority_k
    majority_dir = fold_dir / f"majority_vote_v{v}_k{k}"
    metrics_path = majority_dir / "metrics.json"
    if metrics_path.is_file() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite {metrics_path}; pass --overwrite to rerun. "
            "Validated pair caches will still be reused."
        )
    majority_dir.mkdir(parents=True, exist_ok=True)

    teacher_sets = sample_unique_teacher_sets(
        train_files=train_files,
        teacher_count=args.majority_teacher_count,
        num_sets=args.majority_num_teacher_sets,
        seed=args.majority_seed + fold,
    )
    all_files = list(train_files) + list(val_files)
    all_worms = [experiment.worm(path) for path in all_files]
    vocabulary, node_label_ids, label_presence = build_label_encoding(all_worms)
    if not vocabulary:
        raise RuntimeError(f"Fold {fold}: no valid labels for majority-vote scoring")
    cache_signature = majority_cache_signature(
        args=args,
        fold=fold,
        files=all_files,
        vocabulary=vocabulary,
    )
    cache_root = fold_dir / "majority_pair_cache"

    teacher_set_rows: list[dict[str, Any]] = []
    for split_index, teachers in enumerate(teacher_sets):
        teacher_set_rows.append(
            {
                "fold": fold,
                "teacher_set": split_index,
                "teacher_count": len(teachers),
                "teacher_uids": "|".join(experiment.worm(path).uid for path in teachers),
                "teacher_paths": "|".join(str(path) for path in teachers),
            }
        )
    write_csv(majority_dir / "teacher_sets.csv", teacher_set_rows)

    required_train_pairs = sorted(
        {
            canonical_pair(a, b)
            for teachers in teacher_sets
            for a, b in combinations(teachers, 2)
        },
        key=lambda pair: (str(pair[0].resolve()), str(pair[1].resolve())),
    )
    best: list[dict[str, Any]] = [
        {"h": None, "accuracy": -math.inf, "correct": 0, "queries": 0}
        for _ in teacher_sets
    ]
    inner_rows: list[dict[str, Any]] = []
    cache_audit_rows: list[dict[str, Any]] = []

    for h in args.h_candidates:
        log(
            f"fold={fold} majority inner h={h}: prepare "
            f"{len(required_train_pairs)} unique train pairs"
        )
        candidates, audit = experiment.prepare_majority_candidates(
            pairs=required_train_pairs,
            h=h,
            v=v,
            node_label_ids=node_label_ids,
            cache_root=cache_root,
            cache_signature=cache_signature,
            log_prefix=f"fold={fold} majority-inner",
        )
        cache_audit_rows.append({"fold": fold, "stage": "inner", "h": h, **audit})

        for split_index, teachers in enumerate(teacher_sets):
            total_correct = 0
            total_queries = 0
            individual_accuracies: list[float] = []
            for query in teachers:
                inner_teachers = [teacher for teacher in teachers if teacher != query]
                row = evaluate_majority_one_worm(
                    query_path=query,
                    teacher_paths=inner_teachers,
                    candidates=candidates,
                    node_label_ids=node_label_ids,
                    label_presence=label_presence,
                    num_labels=len(vocabulary),
                    v=v,
                    k=k,
                    tie_break=args.majority_tie_break,
                )
                total_correct += int(row["correct"])
                total_queries += int(row["num_queries"])
                individual_accuracies.append(float(row["accuracy"]))
            accuracy = (
                float(total_correct / total_queries) if total_queries else math.nan
            )
            inner_rows.append(
                {
                    "fold": fold,
                    "teacher_set": split_index,
                    "h": h,
                    "v": v,
                    "k": k,
                    "validation_worms": len(teachers),
                    "queries": total_queries,
                    "correct": total_correct,
                    "micro_accuracy": accuracy,
                    "individual_mean_accuracy": float(np.mean(individual_accuracies)),
                    "individual_median_accuracy": float(np.median(individual_accuracies)),
                }
            )
            # Ascending h plus strict improvement implements the paper's
            # otherwise-unspecified smallest-h tie break.
            if accuracy > float(best[split_index]["accuracy"]):
                best[split_index] = {
                    "h": h,
                    "accuracy": accuracy,
                    "correct": total_correct,
                    "queries": total_queries,
                }
        write_csv(majority_dir / "inner_h_selection.csv", inner_rows)
        write_csv(majority_dir / "cache_audit.csv", cache_audit_rows)
        experiment.clear_relations_for_h(h)

    selected_h_rows: list[dict[str, Any]] = []
    for split_index, (teachers, selected) in enumerate(zip(teacher_sets, best)):
        if selected["h"] is None:
            raise RuntimeError(f"Fold {fold}, teacher set {split_index}: h not selected")
        selected_h_rows.append(
            {
                "fold": fold,
                "teacher_set": split_index,
                "selected_h": int(selected["h"]),
                "inner_micro_accuracy": float(selected["accuracy"]),
                "inner_correct": int(selected["correct"]),
                "inner_queries": int(selected["queries"]),
                "teacher_uids": "|".join(experiment.worm(path).uid for path in teachers),
            }
        )
    write_csv(majority_dir / "selected_h.csv", selected_h_rows)

    sets_by_h: dict[int, list[int]] = {}
    for split_index, selected in enumerate(best):
        sets_by_h.setdefault(int(selected["h"]), []).append(split_index)

    individual_rows: list[dict[str, Any]] = []
    teacher_set_metric_rows: list[dict[str, Any]] = []
    for h in sorted(sets_by_h):
        split_indices = sets_by_h[h]
        required_outer_pairs = sorted(
            {
                canonical_pair(query, teacher)
                for split_index in split_indices
                for teacher in teacher_sets[split_index]
                for query in val_files
            },
            key=lambda pair: (str(pair[0].resolve()), str(pair[1].resolve())),
        )
        log(
            f"fold={fold} majority outer h={h}: teacher_sets={len(split_indices)} "
            f"unique_val_train_pairs={len(required_outer_pairs)}"
        )
        candidates, audit = experiment.prepare_majority_candidates(
            pairs=required_outer_pairs,
            h=h,
            v=v,
            node_label_ids=node_label_ids,
            cache_root=cache_root,
            cache_signature=cache_signature,
            log_prefix=f"fold={fold} majority-outer",
        )
        cache_audit_rows.append({"fold": fold, "stage": "outer", "h": h, **audit})

        for split_index in split_indices:
            teachers = teacher_sets[split_index]
            current_rows: list[dict[str, Any]] = []
            for query in val_files:
                result = evaluate_majority_one_worm(
                    query_path=query,
                    teacher_paths=teachers,
                    candidates=candidates,
                    node_label_ids=node_label_ids,
                    label_presence=label_presence,
                    num_labels=len(vocabulary),
                    v=v,
                    k=k,
                    tie_break=args.majority_tie_break,
                )
                row = {
                    "fold": fold,
                    "teacher_set": split_index,
                    "selected_h": h,
                    "v": v,
                    "k": k,
                    "test_worm": experiment.worm(query).uid,
                    "test_path": str(query),
                    "teacher_count": len(teachers),
                    **result,
                }
                individual_rows.append(row)
                current_rows.append(row)

            total_queries = sum(int(row["num_queries"]) for row in current_rows)
            total_correct = sum(int(row["correct"]) for row in current_rows)
            total_covered = sum(int(row["num_covered_queries"]) for row in current_rows)
            accuracies = np.asarray(
                [float(row["accuracy"]) for row in current_rows], dtype=np.float64
            )
            teacher_set_metric_rows.append(
                {
                    "fold": fold,
                    "teacher_set": split_index,
                    "selected_h": h,
                    "v": v,
                    "k": k,
                    "num_val_worms": len(current_rows),
                    "queries": total_queries,
                    "correct": total_correct,
                    "micro_accuracy": (
                        float(total_correct / total_queries) if total_queries else math.nan
                    ),
                    "individual_mean_accuracy": float(np.mean(accuracies)),
                    "individual_median_accuracy": float(np.median(accuracies)),
                    "coverage": (
                        float(total_covered / total_queries) if total_queries else math.nan
                    ),
                }
            )
        write_csv(majority_dir / "val_individual.csv", individual_rows)
        write_csv(majority_dir / "val_teacher_set.csv", teacher_set_metric_rows)
        write_csv(majority_dir / "cache_audit.csv", cache_audit_rows)
        experiment.clear_relations_for_h(h)

    individual_accuracy = np.asarray(
        [float(row["accuracy"]) for row in individual_rows], dtype=np.float64
    )
    split_micro = np.asarray(
        [float(row["micro_accuracy"]) for row in teacher_set_metric_rows],
        dtype=np.float64,
    )
    coverage = np.asarray(
        [float(row["coverage"]) for row in teacher_set_metric_rows], dtype=np.float64
    )
    selected_h_counter = Counter(int(row["selected_h"]) for row in selected_h_rows)
    directional_queries = sum(int(row["directional_queries"]) for row in cache_audit_rows)
    cutoff_ties = sum(int(row["cutoff_tied_queries"]) for row in cache_audit_rows)
    summary = {
        "fold": fold,
        "split": "val",
        "protocol": "Appendix F majority vote adapted to fixed CV5",
        "v": v,
        "k": k,
        "teacher_count": args.majority_teacher_count,
        "num_unique_teacher_sets": len(teacher_sets),
        "num_train_worms": len(train_files),
        "num_val_worms": len(val_files),
        "unique_validation_queries": int(
            sum(np.sum(node_label_ids[str(path.resolve())] >= 0) for path in val_files)
        ),
        "num_individual_evaluations": len(individual_rows),
        "paper_median_individual_accuracy": float(np.median(individual_accuracy)),
        "individual_mean_accuracy": float(np.mean(individual_accuracy)),
        "individual_sd_accuracy": (
            float(np.std(individual_accuracy, ddof=1))
            if individual_accuracy.size > 1
            else 0.0
        ),
        "individual_q025": float(np.quantile(individual_accuracy, 0.025)),
        "individual_q25": float(np.quantile(individual_accuracy, 0.25)),
        "individual_q75": float(np.quantile(individual_accuracy, 0.75)),
        "individual_q975": float(np.quantile(individual_accuracy, 0.975)),
        "teacher_set_micro_mean": float(np.mean(split_micro)),
        "teacher_set_micro_sd": (
            float(np.std(split_micro, ddof=1)) if split_micro.size > 1 else 0.0
        ),
        "teacher_set_micro_median": float(np.median(split_micro)),
        "coverage_mean": float(np.mean(coverage)),
        "selected_h_distribution": {
            str(h): {
                "count": int(selected_h_counter.get(h, 0)),
                "fraction": float(selected_h_counter.get(h, 0) / len(teacher_sets)),
            }
            for h in args.h_candidates
        },
        "candidate_cutoff_tie_audit": {
            "directional_queries_in_unique_pair_caches": directional_queries,
            "cutoff_tied_queries": cutoff_ties,
            "cutoff_tie_rate": (
                float(cutoff_ties / directional_queries) if directional_queries else 0.0
            ),
        },
        "cache_signature": cache_signature,
        "primary_metric": "paper_median_individual_accuracy",
        "note": (
            "Primary denominator includes every unique supervised query neuron; "
            "coverage-conditioned accuracy is diagnostic only. noID consumes a Top-v "
            "slot and is excluded from final Top-k labels."
        ),
    }
    if k == 5:
        summary["paper_median_individual_top5"] = summary[
            "paper_median_individual_accuracy"
        ]
    write_json(metrics_path, summary)
    return summary, individual_rows, teacher_set_metric_rows


def configuration(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "method": "GWOT-MD activity-only",
        "paper": PAPER_URL,
        "split_root": str(args.split_root),
        "output_root": str(args.output_root),
        "h_candidates_frames": list(args.h_candidates),
        "highpass_cutoff_hz": args.highpass_cutoff_hz,
        "fallback_sample_rate_hz": args.fallback_sample_rate_hz,
        "activity_normalization": args.activity_normalization,
        "distance_normalization": args.distance_normalization,
        "lag_stride": args.lag_stride,
        "relations": "equally weighted D(tau), tau=-h..h at lag_stride; endpoints and zero included",
        "marginals": "uniform",
        "solver": "conditional-gradient with exact EMD linear subproblems",
        "contraction_device_requested": args.device,
        "initializations": ["uniform", "structural_signature"],
        "ranking_tie_policy": "expected credit under uniform ordering within exact tie blocks",
        "max_iter": args.max_iter,
        "tolerance": args.tolerance,
        "tune_reference": "lexicographically first train recording",
        "tune_max_partners": args.tune_max_partners,
        "max_val_pairs": args.max_val_pairs,
        "reuse_h_selection": args.reuse_h_selection,
        "majority_vote": {
            "protocol": "paper Appendix F adapted to held-out CV5 validation",
            "teacher_count": args.majority_teacher_count,
            "num_unique_teacher_sets_per_fold": args.majority_num_teacher_sets,
            "v": args.majority_v,
            "k": args.majority_k,
            "seed": args.majority_seed,
            "vote_tie_break": args.majority_tie_break,
            "h_selection": (
                "pooled leave-one-out v/k accuracy inside each 9-teacher set; "
                "smaller h breaks exact ties"
            ),
            "reporting": "median of per-validation-worm accuracies",
            "noID": "occupies Top-v slots; excluded from final Top-k labels",
        },
        "label_policy": "labeled & certain & clean & valid id & unique within worm",
        "test_access": "forbidden by implementation",
        "deterministic": True,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def parse_folds(value: str) -> list[int]:
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not folds or any(fold not in range(5) for fold in folds):
        raise argparse.ArgumentTypeError("folds must be comma-separated values in 0..4")
    return sorted(set(folds))


def parse_h_candidates(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("h candidates must be nonnegative integers")
    return sorted(set(values))


def preflight(args: argparse.Namespace) -> None:
    log("Preflight starts; only train and val directories will be inspected")
    rows: list[dict[str, Any]] = []
    timestamp_fallbacks = 0
    for fold in args.folds:
        train = split_files(args.split_root, fold, "train")
        val = split_files(args.split_root, fold, "val")
        experiment = Experiment(args)
        for split, files in (("train", train), ("val", val)):
            for path in files:
                worm = experiment.worm(path)
                if worm.timestamp_key is None:
                    timestamp_fallbacks += 1
                if max(args.h_candidates) >= worm.activity.shape[1] - 2:
                    raise ValueError(
                        f"Largest h is invalid for {path}: T={worm.activity.shape[1]}"
                    )
            rows.append(
                {"fold": fold, "split": split, "worms": len(files)}
            )
    try:
        import ot  # type: ignore

        pot_version = getattr(ot, "__version__", "unknown")
    except ModuleNotFoundError:
        pot_version = None
    print(json.dumps({
        "status": "ok" if pot_version else "dependency_missing",
        "POT_version": pot_version,
        "timestamp_fallback_recordings_counted_across_folds": timestamp_fallbacks,
        "splits": rows,
        "next": "pip install POT" if pot_version is None else "run pilot",
    }, indent=2))
    if pot_version is None:
        raise SystemExit(2)


def self_check(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(7)
    n, t = 9, 240
    activity = rng.normal(size=(n, t))
    for neuron in range(n):
        activity[neuron] = np.convolve(
            activity[neuron], np.ones(9) / 9, mode="same"
        )
    permutation = rng.permutation(n)
    other = activity[permutation] + 0.01 * rng.normal(size=(n, t))
    relation_a = build_relations(activity, 5, "rms", 1)
    relation_b = build_relations(other, 5, "rms", 1)
    coupling, diagnostics = solve_gwot_md(
        relation_a, relation_b, 80, 1e-10, device="cpu"
    )
    inverse = np.argsort(permutation)
    recovered = np.argmax(coupling, axis=1)
    accuracy = float(np.mean(recovered == inverse))
    if accuracy < 0.80:
        raise AssertionError(f"Synthetic permutation recovery too low: {accuracy:.3f}")
    if diagnostics["row_marginal_max_error"] > 1e-7:
        raise AssertionError("Row marginal invariant failed")
    if diagnostics["col_marginal_max_error"] > 1e-7:
        raise AssertionError("Column marginal invariant failed")
    vote_candidates = np.asarray(
        [
            [0, 1, -1, 0, 2, 1],
            [0, -1, 1, 0, -1, 1],
        ],
        dtype=np.int32,
    )
    vote_ranks = query_ranks_from_votes(
        vote_candidates,
        np.asarray([1, 2], dtype=np.int32),
        num_labels=3,
        tie_break="first",
    )
    if vote_ranks[0] != 2 or not np.isinf(vote_ranks[1]):
        raise AssertionError(
            f"Majority-vote/noID invariant failed: ranks={vote_ranks.tolist()}"
        )
    print(json.dumps({
        "status": "ok",
        "synthetic_top1": accuracy,
        "majority_vote_check": "ok",
        "solver": diagnostics,
    }, indent=2))


def run(args: argparse.Namespace) -> None:
    folds = [0] if args.command == "pilot" else args.folds
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = configuration(args)
    write_json(args.output_root / "config.json", config)
    fold_summaries: list[dict[str, Any]] = []
    pooled = Counts()
    for fold in folds:
        experiment = Experiment(args)
        fold_dir = args.output_root / f"fold_{fold}"
        if fold_dir.exists() and any(fold_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite nonempty {fold_dir}; pass --overwrite to replace outputs"
            )
        fold_dir.mkdir(parents=True, exist_ok=True)
        train = split_files(args.split_root, fold, "train")
        val = split_files(args.split_root, fold, "val")
        h_selection_path = fold_dir / "h_selection.json"
        if args.reuse_h_selection:
            if not h_selection_path.is_file():
                raise FileNotFoundError(
                    f"--reuse-h-selection requires {h_selection_path}"
                )
            h_selection = json.loads(h_selection_path.read_text(encoding="utf-8"))
            log(
                f"fold={fold}: reuse train-selected h={h_selection['selected_h']} "
                "and recompute validation metrics"
            )
        else:
            log(f"fold={fold}: tune h on train ({len(train)} worms)")
            h_selection = experiment.tune_h(fold, train)
            write_json(h_selection_path, h_selection)
        selected_h = int(h_selection["selected_h"])
        log(f"fold={fold}: selected h={selected_h}; evaluate val ({len(val)} worms)")
        summary, pair_rows, counts = experiment.evaluate_validation(
            fold, val, selected_h
        )
        pooled.add(counts)
        fold_summaries.append(summary)
        write_csv(fold_dir / "val_pairs.csv", pair_rows)
        write_json(fold_dir / "val_metrics.json", summary)

    metric_names = ["top1", "top5", "mrr", "hungarian_accuracy", "coverage"]
    macro: dict[str, Any] = {}
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in fold_summaries])
        macro[f"{name}_mean"] = float(values.mean())
        macro[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    result = {
        "method": "GWOT-MD activity-only",
        "evaluation_split": "val",
        "folds": folds,
        "fold_metrics": fold_summaries,
        "macro_across_folds": macro,
        "pooled_query_weighted": pooled.metrics(),
        "note": "Deterministic baseline: one run per biological fold; no seed duplication.",
    }
    write_json(args.output_root / "val_summary.json", result)
    log("completed")
    print(json.dumps(result, indent=2))


def run_majority(args: argparse.Namespace) -> None:
    args.output_root.mkdir(parents=True, exist_ok=True)
    config = configuration(args)
    config["active_command"] = "majority-cv5"
    config["solver_fidelity_note"] = (
        "The Appendix-F evaluation protocol is reproduced, but pairwise couplings "
        "come from this runner's two-start conditional-gradient solver rather than "
        "the paper's full 21-epsilon x 50-initialization search."
    )
    write_json(args.output_root / "majority_vote_config.json", config)

    fold_summaries: list[dict[str, Any]] = []
    all_individual_rows: list[dict[str, Any]] = []
    all_teacher_set_rows: list[dict[str, Any]] = []
    for fold in args.folds:
        experiment = Experiment(args)
        fold_dir = args.output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train = split_files(args.split_root, fold, "train")
        val = split_files(args.split_root, fold, "val")
        log(
            f"fold={fold}: majority-vote train={len(train)} val={len(val)} "
            f"teacher_sets={args.majority_num_teacher_sets}"
        )
        summary, individual_rows, teacher_set_rows = evaluate_majority_fold(
            experiment=experiment,
            args=args,
            fold=fold,
            train_files=train,
            val_files=val,
            fold_dir=fold_dir,
        )
        fold_summaries.append(summary)
        all_individual_rows.extend(individual_rows)
        all_teacher_set_rows.extend(teacher_set_rows)

    macro: dict[str, Any] = {}
    macro_names = [
        "paper_median_individual_accuracy",
        "individual_mean_accuracy",
        "teacher_set_micro_mean",
        "coverage_mean",
    ]
    for name in macro_names:
        values = np.asarray([float(row[name]) for row in fold_summaries])
        macro[f"{name}_mean"] = float(np.mean(values))
        macro[f"{name}_sd"] = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )

    individual_values = np.asarray(
        [float(row["accuracy"]) for row in all_individual_rows], dtype=np.float64
    )
    teacher_set_micro = np.asarray(
        [float(row["micro_accuracy"]) for row in all_teacher_set_rows],
        dtype=np.float64,
    )
    result = {
        "method": "GWOT-MD activity-only",
        "evaluation_split": "val",
        "protocol": "paper Appendix F majority vote adapted to grouped CV5",
        "folds": args.folds,
        "v": args.majority_v,
        "k": args.majority_k,
        "teacher_count": args.majority_teacher_count,
        "unique_teacher_sets_per_fold": args.majority_num_teacher_sets,
        "fold_metrics": fold_summaries,
        "macro_across_folds": macro,
        "pooled_repeated_teacher_set_evaluations": {
            "num_individual_evaluations": len(all_individual_rows),
            "paper_median_individual_accuracy": float(np.median(individual_values)),
            "individual_mean_accuracy": float(np.mean(individual_values)),
            "individual_sd_accuracy": (
                float(np.std(individual_values, ddof=1))
                if individual_values.size > 1
                else 0.0
            ),
            "num_teacher_set_evaluations": len(all_teacher_set_rows),
            "teacher_set_micro_mean": float(np.mean(teacher_set_micro)),
            "teacher_set_micro_sd": (
                float(np.std(teacher_set_micro, ddof=1))
                if teacher_set_micro.size > 1
                else 0.0
            ),
        },
        "headline_metric": "pooled_repeated_teacher_set_evaluations.paper_median_individual_accuracy",
        "paper_comparison_note": (
            "The paper's 46% is the median per-individual identification accuracy "
            "under v=5, k=5 and 9 teachers. Compare it with the headline median, "
            "not with direct pairwise Top-5. Dataset cohort/splits and this runner's "
            "coupling solver still differ from the paper-exact experiment."
        ),
    }
    if args.majority_k == 5:
        result["pooled_repeated_teacher_set_evaluations"][
            "paper_median_individual_top5"
        ] = result["pooled_repeated_teacher_set_evaluations"][
            "paper_median_individual_accuracy"
        ]
    write_json(
        args.output_root
        / f"majority_vote_v{args.majority_v}_k{args.majority_k}_summary.json",
        result,
    )
    log("majority-cv5 completed")
    print(json.dumps(result, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Leakage-safe activity-only GWOT-MD evaluation on Atanas grouped CV5"
    )
    parser.add_argument(
        "command",
        choices=("self-check", "preflight", "pilot", "cv5", "majority-cv5"),
    )
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("0,1,2,3,4"))
    parser.add_argument(
        "--h-candidates", type=parse_h_candidates, default=parse_h_candidates("0,5,10,15,20,25,30,35,40,45,50")
    )
    parser.add_argument("--highpass-cutoff-hz", type=float, default=0.01)
    parser.add_argument("--fallback-sample-rate-hz", type=float, default=4.0)
    parser.add_argument(
        "--activity-normalization", choices=("none", "zscore"), default="none"
    )
    parser.add_argument(
        "--distance-normalization", choices=("none", "mean", "rms"), default="none"
    )
    parser.add_argument(
        "--lag-stride", type=int, default=1,
        help="1 uses every integer lag (paper-faithful); >1 is a faster pilot approximation",
    )
    parser.add_argument(
        "--device", default="auto",
        help="auto, cpu, or a CUDA device such as cuda:0",
    )
    parser.add_argument(
        "--tune-max-partners", type=int, default=0,
        help="0 uses every other train worm with the fixed train reference",
    )
    parser.add_argument(
        "--max-val-pairs", type=int, default=0,
        help="0 evaluates every unordered validation pair",
    )
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-h-selection", action="store_true",
        help="reuse each existing fold_N/h_selection.json and rerun validation only",
    )
    parser.add_argument(
        "--majority-teacher-count",
        type=int,
        default=9,
        help="paper Appendix F uses 9 labeled teacher worms",
    )
    parser.add_argument(
        "--majority-num-teacher-sets",
        type=int,
        default=1000,
        help="unique random teacher combinations per fold; paper uses 1000",
    )
    parser.add_argument(
        "--majority-v",
        type=int,
        default=5,
        help="Top-v matched neurons from each teacher cast votes",
    )
    parser.add_argument(
        "--majority-k",
        type=int,
        default=5,
        help="true label must be among final Top-k voted labels",
    )
    parser.add_argument("--majority-seed", type=int, default=42)
    parser.add_argument(
        "--majority-tie-break",
        choices=("first", "label"),
        default="first",
        help="order for equal label vote counts; first matches stable Counter order",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.max_iter <= 0 or args.tolerance <= 0 or args.lag_stride <= 0:
        parser.error("--max-iter, --tolerance and --lag-stride must be positive")
    if args.tune_max_partners < 0 or args.max_val_pairs < 0:
        parser.error("pair limits must be nonnegative")
    if (
        args.majority_teacher_count < 2
        or args.majority_num_teacher_sets <= 0
        or args.majority_v <= 0
        or args.majority_k <= 0
    ):
        parser.error("majority teacher count must be >=2 and set/v/k values positive")
    if args.command == "self-check":
        self_check(args)
    elif args.command == "preflight":
        preflight(args)
    elif args.command == "majority-cv5":
        run_majority(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
