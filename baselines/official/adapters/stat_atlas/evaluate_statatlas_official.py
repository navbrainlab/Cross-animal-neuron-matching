#!/usr/bin/env python3
"""
Locked-CV Statistical Atlas benchmark adapter.

Method:
  1) Use the pinned official stat-atlas Atlas.train_atlas implementation.
  2) Position-only: every training neuron is given a constant dummy color channel.
     Hence no RGB/activity information enters the atlas.
  3) Train the atlas using TRAIN worms only.
  4) At test time, align each test point cloud to the atlas using LABEL-FREE
     geometry-only PCA initialization + trimmed similarity ICP.
  5) Convert official atlas position means/covariances to per-neuron identity
     posteriors.
  6) Pair score between two test worms = posterior overlap P_A @ P_B.T.
  7) Reuse benchmark_cv5x3_common.evaluate_pairwise, preserving the exact
     query eligibility, rank, Top-k and Hungarian semantics used by the locked
     benchmark.

Test labels are never passed to alignment or scoring. They are accessed only
inside the common evaluator after the complete score matrix has been produced.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[4]
STAT = ROOT / "baselines/official/third_party/wormid_official/stat-atlas"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(STAT) not in sys.path:
    sys.path.insert(0, str(STAT))

# Pinned official source imports. Do not modify files under third_party/.
from models import Atlas
import utils as statatlas_utils

NeuronModule = importlib.import_module("neurons.Neuron")
ImageModule = importlib.import_module("neurons.Image")
Neuron = NeuronModule.Neuron
Image = ImageModule.Image


# ---------------------------------------------------------------------------
# Position-only official StatAtlas adaptation
# ---------------------------------------------------------------------------
#
# Official Atlas.update_beta has two independent halves:
#   (1) XYZ MCR regression + XYZ Mahalanobis cost
#   (2) color MCR regression + color Mahalanobis cost
#
# For this benchmark we must be strictly position-only.  Therefore we subclass
# the pinned official Atlas and copy ONLY the XYZ half of update_beta verbatim
# in semantics.  The color channel is carried as a constant bookkeeping
# dimension required by train_atlas(), but it is never regressed, scored, or
# used to influence the XYZ transform.
#
# No test labels are used anywhere in this method.
#
import utils as statatlas_utils
import scipy as sp



_OFFICIAL_SCALED_ROTATION = statatlas_utils.scaled_rotation
_SCALED_ROTATION_FALLBACK_COUNT = 0
_MCR_PINV_FALLBACK_COUNT = 0


def _finite_row_mask(x):
    x = np.asarray(x, dtype=np.float64)
    return np.isfinite(x).all(axis=1)


def _fallback_similarity_from_corresponded_rows(X, Y):
    """
    Train-only fallback for degenerate official scaled_rotation.

    X and Y are already identity-row-aligned by official Atlas.sort_mu().
    Use only rows finite in both arrays.  For >=3 common points use the same
    isotropic least-squares similarity fit used elsewhere in this adapter.
    For 1-2 common points, use an isotropic scale from RMS spread (when
    available), identity rotation, and centroid translation.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    mask = _finite_row_mask(X) & _finite_row_mask(Y)
    a, b = X[mask], Y[mask]

    if len(a) >= 3:
        s, R, t = fit_similarity(a, b)
        return float(s), np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64)

    # Very sparse overlap: keep the transform conservative and finite.
    if len(a) >= 1:
        ca, cb = a.mean(0), b.mean(0)
        aa, bb = a - ca, b - cb
        rms_a = float(np.sqrt(np.mean(np.sum(aa * aa, axis=1)))) if len(a) > 1 else 0.0
        rms_b = float(np.sqrt(np.mean(np.sum(bb * bb, axis=1)))) if len(b) > 1 else 0.0
        s = rms_b / rms_a if rms_a > 1e-12 and rms_b > 0 else 1.0
        R = np.eye(3, dtype=np.float64)
        t = cb - s * (ca @ R)
        return float(s), R, t

    # No shared named points.  Use all finite labeled points for centroid/RMS
    # normalization only; do not invent correspondences.
    ax = X[_finite_row_mask(X)]
    by = Y[_finite_row_mask(Y)]
    if len(ax) == 0 or len(by) == 0:
        return 1.0, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    ca, cb = ax.mean(0), by.mean(0)
    aa, bb = ax - ca, by - cb
    rms_a = float(np.sqrt(np.mean(np.sum(aa * aa, axis=1))))
    rms_b = float(np.sqrt(np.mean(np.sum(bb * bb, axis=1))))
    s = rms_b / rms_a if rms_a > 1e-12 and rms_b > 0 else 1.0
    R = np.eye(3, dtype=np.float64)
    t = cb - s * (ca @ R)
    return float(s), R, t


def _scaled_rotation_position_safe(X, Y, sigma=None):
    """
    Keep the pinned official scaled_rotation whenever it returns finite values.
    Fall back only for degenerate low-rank geometry (zero variance / SVD failure).
    """
    global _SCALED_ROTATION_FALLBACK_COUNT
    try:
        out = _OFFICIAL_SCALED_ROTATION(X, Y, sigma)
        S, R, T = out
        finite = (
            np.isfinite(np.asarray(S, dtype=np.float64)).all()
            and np.isfinite(np.asarray(R, dtype=np.float64)).all()
            and np.isfinite(np.asarray(T, dtype=np.float64)).all()
        )
        if finite:
            return out
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        pass

    _SCALED_ROTATION_FALLBACK_COUNT += 1
    return _fallback_similarity_from_corresponded_rows(X, Y)


# Official initialize_atlas resolves utils.scaled_rotation dynamically.
# Patch only this process; third_party source files remain untouched.
statatlas_utils.scaled_rotation = _scaled_rotation_position_safe


def _mcr_position_safe(X, Y, sigma):
    """
    Official MCR_solver first.  If the position design is singular, retry the
    same official solver with Moore-Penrose pinv replacing only np.linalg.inv
    for that call.
    """
    global _MCR_PINV_FALLBACK_COUNT
    try:
        return statatlas_utils.MCR_solver(X, Y, sigma)
    except np.linalg.LinAlgError as exc:
        if "Singular matrix" not in str(exc):
            raise

    _MCR_PINV_FALLBACK_COUNT += 1
    original_inv = np.linalg.inv
    np.linalg.inv = np.linalg.pinv
    try:
        return statatlas_utils.MCR_solver(X, Y, sigma)
    finally:
        np.linalg.inv = original_inv


class PositionOnlyAtlas(Atlas):
    """Pinned official Atlas with update_beta restricted to its XYZ branch."""

    def update_beta(self, X, model):
        # Same allocation shapes as official Atlas.update_beta.
        beta = np.zeros((X.shape[1], X.shape[1], X.shape[2]))
        beta0 = np.zeros((1, X.shape[1], X.shape[2]))
        aligned = np.zeros(X.shape)

        C = X.shape[1] - 3
        cost = [0.0, 0.0]

        # Position-only inverse.  In the official full implementation the
        # position/color cross-covariance is zeroed by estimate_sigma(), so
        # this is exactly the XYZ block needed by the official position cost.
        sigma_pos_inv = np.array([
            np.linalg.inv(model["sigma"][:3, :3, i])
            for i in range(model["sigma"].shape[2])
        ]).transpose([1, 2, 0])

        for j in range(X.shape[2]):
            idx = ~np.isnan(X[:, :, j]).all(1)

            # ---- OFFICIAL POSITION REGRESSION BRANCH ----
            R = _mcr_position_safe(
                np.concatenate(
                    (X[idx, :3, j], np.ones((idx.sum(), 1))),
                    1,
                ),
                model["mu"][idx, :3],
                model["sigma"][:3, :3, idx],
            )

            beta[:3, :3, j] = R[:3, :3]
            beta0[:, :3, j] = R[3, None]

            # Color is deliberately informationally inert.  Carry the
            # constant bookkeeping channel through unchanged.
            if C > 0:
                beta[3:, 3:, j] = np.eye(C)

            aligned[:, :, j] = X[:, :, j] @ beta[:, :, j] + beta0[:, :, j]

            # ---- OFFICIAL POSITION MAHALANOBIS COST BRANCH ----
            cost[0] += sum([
                sp.spatial.distance.mahalanobis(
                    aligned[i, :3, j].squeeze(),
                    model["mu"][i, :3].squeeze(),
                    sigma_pos_inv[:, :, i],
                )
                for i in np.where(idx)[0]
            ])

            # No color regression and no color cost.
            cost[1] = 0.0

        params = {"beta": beta, "beta0": beta0}
        return params, aligned, cost


@dataclass
class AlignmentResult:
    aligned_xyz: np.ndarray
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    objective: float


def np_xyz(record: Any) -> np.ndarray:
    xyz = record.xyz
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"{record.worm_id}: expected xyz [N,>=3], got {xyz.shape}")
    xyz = xyz[:, :3]
    if not np.isfinite(xyz).all():
        raise ValueError(f"{record.worm_id}: xyz contains NaN/Inf")
    return xyz


def np_labels(record: Any) -> np.ndarray:
    labels = record.labels
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    return np.asarray(labels, dtype=np.int64).reshape(-1)


def clean_unique_label_count(record: Any) -> int:
    labels = np_labels(record)
    positions: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        if int(lab) >= 0:
            positions.setdefault(int(lab), []).append(i)
    return int(sum(len(rows) == 1 for rows in positions.values()))


def make_official_train_images(
    records: Sequence[Any],
    *,
    min_train_labels_per_worm: int = 4,
) -> tuple[list[Any], list[Any], list[dict[str, Any]]]:
    """
    Create official Image/Neuron objects from TRAIN worms only.

    StatAtlas spatial alignment needs enough named points to estimate a 3-D
    transform. Worms with fewer than `min_train_labels_per_worm` clean unique
    identities cannot stably contribute to atlas fitting, so they are excluded
    from the method's TRAIN subset only. No validation/test worm is filtered
    here, and no outer-test label is used.
    """
    ims = []
    kept_records = []
    excluded = []

    for record in records:
        xyz = np_xyz(record)
        labels = np_labels(record)

        positions: dict[int, list[int]] = {}
        for i, lab in enumerate(labels.tolist()):
            if int(lab) >= 0:
                positions.setdefault(int(lab), []).append(i)

        unique = {
            lab: rows[0]
            for lab, rows in positions.items()
            if len(rows) == 1
        }

        if len(unique) < int(min_train_labels_per_worm):
            excluded.append({
                "worm_id": str(record.worm_id),
                "clean_unique_labels": int(len(unique)),
                "reason": (
                    f"fewer than {int(min_train_labels_per_worm)} clean unique "
                    "training identities; insufficient for full-rank [x,y,z,1] spatial regression"
                ),
            })
            continue

        neurons = []
        for lab, i in unique.items():
            neuron = Neuron()
            neuron.position = xyz[int(i)].astype(np.float64, copy=True)
            neuron.color = np.zeros((1,), dtype=np.float64)
            neuron.color_readout = np.zeros((1,), dtype=np.float64)
            neuron.annotation = str(int(lab))
            neuron.annotation_confidence = 1.0
            neurons.append(neuron)

        ims.append(
            Image(
                "head",
                neurons=neurons,
                scale=np.ones((3,), dtype=np.float64),
            )
        )
        kept_records.append(record)

    if len(ims) < 3:
        raise RuntimeError(
            f"Only {len(ims)} training worms remain after requiring "
            f">={int(min_train_labels_per_worm)} clean unique labels per worm"
        )

    return ims, kept_records, excluded


def train_official_atlas(
    train_records: Sequence[Any],
    *,
    min_counts: int,
    epsilon_pos: float,
    n_iter: int,
    min_train_labels_per_worm: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ims, kept_train_records, excluded_train_records = make_official_train_images(
        train_records,
        min_train_labels_per_worm=min_train_labels_per_worm,
    )

    # The second epsilon is harmless for the constant dummy color channel.
    model = PositionOnlyAtlas(min_counts=int(min_counts), epsilon=[float(epsilon_pos), 1000.0])

    atlas, aligned_coord, params, cost, counts = model.train_atlas(
        ims,
        bodypart="head",
        neurons=None,
        n_iter=int(n_iter),
        train_indices=list(range(len(ims))),
        match_indices=list(range(len(ims))),
    )

    mu = np.asarray(atlas["mu"], dtype=np.float64)
    sigma = np.asarray(atlas["sigma"], dtype=np.float64)
    names = [str(x) for x in atlas["names"]]

    if mu.ndim != 2 or mu.shape[1] < 3:
        raise RuntimeError(f"Official atlas mu has unexpected shape {mu.shape}")
    if sigma.ndim != 3 or sigma.shape[:2] < (3, 3) or sigma.shape[2] != len(names):
        raise RuntimeError(
            f"Official atlas sigma={sigma.shape}, identities={len(names)}"
        )
    if len(names) < 2:
        raise RuntimeError("Official atlas retained fewer than 2 identities")

    # Only XYZ blocks are used downstream.
    atlas_out = {
        "bodypart": atlas["bodypart"],
        "mu": mu,
        "sigma": sigma,
        "names": names,
    }
    diagnostics = {
        "train_worms_original": [str(r.worm_id) for r in train_records],
        "train_worm_count_original": len(train_records),
        "train_worms_used": [str(r.worm_id) for r in kept_train_records],
        "train_worm_count_used": len(kept_train_records),
        "excluded_train_worms": excluded_train_records,
        "excluded_train_worm_count": len(excluded_train_records),
        "min_train_labels_per_worm": int(min_train_labels_per_worm),
        "atlas_identity_count": len(names),
        "min_counts": int(min_counts),
        "min_required_occurrences": int(min_counts) + 1,  # official code uses counts > min_counts
        "epsilon_pos": float(epsilon_pos),
        "n_iter": int(n_iter),
        "official_cost_history": np.asarray(cost, dtype=object).tolist(),
        "update_beta_policy": (
            "PositionOnlyAtlas copies only the XYZ regression and XYZ Mahalanobis "
            "cost branches of pinned official Atlas.update_beta; color regression "
            "and color cost are disabled"
        ),
        "scaled_rotation_fallback_count": int(_SCALED_ROTATION_FALLBACK_COUNT),
        "mcr_pinv_fallback_count": int(_MCR_PINV_FALLBACK_COUNT),
        "degenerate_geometry_policy": (
            "official scaled_rotation/MCR paths are used when finite/full-rank; "
            "degenerate train-only geometry uses isotropic similarity and/or "
            "Moore-Penrose minimum-norm fallback"
        ),
        "color_information": "constant bookkeeping channel only; identity transform; never scored",
    }
    return atlas_out, diagnostics


def fit_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares row-vector similarity transform b ~= s * a @ R + t."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"fit_similarity shapes {a.shape}, {b.shape}")
    if len(a) < 3:
        return 1.0, np.eye(3), b.mean(0) - a.mean(0)

    ca, cb = a.mean(0), b.mean(0)
    aa, bb = a - ca, b - cb
    h = aa.T @ bb
    u, svals, vt = np.linalg.svd(h, full_matrices=False)
    r = u @ vt  # reflection is allowed; no GT is used.
    denom = float(np.sum(aa * aa))
    scale = float(np.sum(svals) / max(denom, 1e-12))
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    t = cb - scale * (ca @ r)
    return scale, r, t


def compose(
    s1: float, r1: np.ndarray, t1: np.ndarray,
    s2: float, r2: np.ndarray, t2: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    y = s1*x@r1+t1 ; z = s2*y@r2+t2
      = (s2*s1)*x@(r1@r2) + s2*t1@r2+t2
    """
    return (
        float(s2 * s1),
        r1 @ r2,
        float(s2) * (t1 @ r2) + t2,
    )


def apply_transform(
    x: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    return float(scale) * (np.asarray(x, dtype=np.float64) @ rotation) + translation


def pca_initializations(src: np.ndarray, target: np.ndarray):
    cs, ct = src.mean(0), target.mean(0)
    xs, xt = src - cs, target - ct

    _, _, vhs = np.linalg.svd(xs, full_matrices=False)
    _, _, vht = np.linalg.svd(xt, full_matrices=False)
    bs = vhs.T
    bt = vht.T

    rms_s = math.sqrt(float(np.mean(np.sum(xs * xs, axis=1))))
    rms_t = math.sqrt(float(np.mean(np.sum(xt * xt, axis=1))))
    scale = rms_t / max(rms_s, 1e-12)

    for signs in product((-1.0, 1.0), repeat=3):
        d = np.diag(np.asarray(signs, dtype=np.float64))
        r = bs @ d @ bt.T
        t = ct - scale * (cs @ r)
        yield float(scale), r, t


def label_free_align(
    xyz: np.ndarray,
    atlas_mu_xyz: np.ndarray,
    *,
    iterations: int,
    trim_fraction: float,
) -> AlignmentResult:
    """
    Geometry-only PCA + trimmed ICP. No labels are accepted by this function.
    """
    src = np.asarray(xyz, dtype=np.float64)
    tgt = np.asarray(atlas_mu_xyz, dtype=np.float64)
    tgt = tgt[np.isfinite(tgt).all(axis=1)]
    if len(src) < 3 or len(tgt) < 3:
        raise RuntimeError(f"Too few points for label-free alignment: src={len(src)} tgt={len(tgt)}")

    tree = cKDTree(tgt)
    best = None

    for s0, r0, t0 in pca_initializations(src, tgt):
        s, r, t = s0, r0.copy(), t0.copy()

        for _ in range(int(iterations)):
            cur = apply_transform(src, s, r, t)
            dist, idx = tree.query(cur, k=1)

            keep_n = max(3, int(math.ceil(float(trim_fraction) * len(cur))))
            keep = np.argsort(dist, kind="stable")[:keep_n]

            ds, dr, dt = fit_similarity(cur[keep], tgt[idx[keep]])
            s, r, t = compose(s, r, t, ds, dr, dt)

        cur = apply_transform(src, s, r, t)
        dist, _ = tree.query(cur, k=1)
        keep_n = max(3, int(math.ceil(float(trim_fraction) * len(cur))))
        objective = float(np.mean(np.sort(dist)[:keep_n] ** 2))

        candidate = AlignmentResult(cur, s, r, t, objective)
        if best is None or candidate.objective < best.objective:
            best = candidate

    assert best is not None
    return best


def atlas_identity_posterior(
    aligned_xyz: np.ndarray,
    atlas: dict[str, Any],
    *,
    covariance_ridge_fraction: float,
) -> np.ndarray:
    """
    Equal-prior Gaussian identity posterior using only official atlas XYZ mu/sigma.
    """
    x = np.asarray(aligned_xyz, dtype=np.float64)
    mu = np.asarray(atlas["mu"], dtype=np.float64)[:, :3]
    sigma = np.asarray(atlas["sigma"], dtype=np.float64)[:3, :3, :]

    n, k = len(x), len(mu)
    logp = np.empty((n, k), dtype=np.float64)

    # One scale only for numerical stabilization, never tuned on test labels.
    diag_values = []
    for j in range(k):
        c = sigma[:, :, j]
        if np.isfinite(c).all():
            diag_values.extend(np.diag(c).tolist())
    diag_values = np.asarray([v for v in diag_values if np.isfinite(v) and v > 0], dtype=np.float64)
    base_var = float(np.median(diag_values)) if len(diag_values) else 1.0
    ridge = max(1e-10, float(covariance_ridge_fraction) * base_var)

    for j in range(k):
        c = np.asarray(sigma[:, :, j], dtype=np.float64)
        if not np.isfinite(c).all():
            c = np.eye(3, dtype=np.float64) * base_var
        c = 0.5 * (c + c.T) + ridge * np.eye(3)

        eig = np.linalg.eigvalsh(c)
        eig = np.maximum(eig, ridge)
        logdet = float(np.log(eig).sum())
        inv = np.linalg.pinv(c, rcond=1e-10)

        delta = x - mu[j]
        d2 = np.einsum("ni,ij,nj->n", delta, inv, delta)
        logp[:, j] = -0.5 * (d2 + logdet)

    # Stable softmax.
    logp -= np.max(logp, axis=1, keepdims=True)
    p = np.exp(np.clip(logp, -700.0, 0.0))
    p /= np.maximum(p.sum(axis=1, keepdims=True), 1e-300)
    return p


def evaluate_fold(args, dataset: str, fold: int):
    # The benchmark data stack is unrelated to atlas construction.  Keep it
    # lazy so other dataset adapters can reuse the exact official StatAtlas
    # trainer/alignment/posterior implementation without importing NeuRID's
    # model-development modules.
    import scripts.lib.benchmark_cv5x3_common as common

    # Biological folds are identical across seeds. Seed 42 is used only as a
    # route to the locked loader/config; Statistical Atlas itself is deterministic.
    bundle = common.split_bundle(dataset, fold, 42, "cpu")
    train_records = bundle["train"]
    test_records = bundle["test"]

    atlas, train_diag = train_official_atlas(
        train_records,
        min_counts=args.min_counts,
        epsilon_pos=args.epsilon_pos,
        n_iter=args.atlas_iterations,
        min_train_labels_per_worm=args.min_train_labels_per_worm,
    )

    mu_xyz = np.asarray(atlas["mu"], dtype=np.float64)[:, :3]

    posterior = {}
    align_diag = []
    for rec in test_records:
        result = label_free_align(
            np_xyz(rec),
            mu_xyz,
            iterations=args.icp_iterations,
            trim_fraction=args.icp_trim,
        )
        posterior[str(rec.worm_id)] = atlas_identity_posterior(
            result.aligned_xyz,
            atlas,
            covariance_ridge_fraction=args.covariance_ridge_fraction,
        )
        align_diag.append({
            "worm_id": str(rec.worm_id),
            "neurons": int(len(rec.labels)),
            "objective": float(result.objective),
            "scale": float(result.scale),
            "det_rotation": float(np.linalg.det(result.rotation)),
        })

    def score_fn(a, b):
        # Atlas-posterior overlap: probability mass assigned to a shared
        # canonical identity. No labels are consulted here.
        pa = posterior[str(a.worm_id)]
        pb = posterior[str(b.worm_id)]
        score = pa @ pb.T
        if not np.isfinite(score).all():
            raise RuntimeError("StatAtlas pair score contains NaN/Inf")
        return score

    qdf = common.evaluate_pairwise(
        test_records,
        score_fn,
        dataset=dataset,
        fold=fold,
        seed=-1,
        method="Statistical Atlas",
    )

    if qdf.empty:
        raise RuntimeError(f"{dataset} fold {fold}: zero eligible queries")

    fold_metric = {
        "dataset": dataset,
        "fold": int(fold),
        "queries": int(len(qdf)),
        "top1": float(qdf["top1"].mean()),
        "top5": float(qdf["top5"].mean()),
        "mrr": float(qdf["rr"].mean()),
        "hungarian_top1": float(qdf["hungarian_top1"].mean()),
        "atlas_identities": int(len(atlas["names"])),
    }

    run_dir = args.out_root / dataset / f"fold_{fold}"
    run_dir.mkdir(parents=True, exist_ok=True)
    qdf.to_csv(run_dir / "query_level.csv", index=False)

    np.savez_compressed(
        run_dir / "trained_position_atlas.npz",
        mu=np.asarray(atlas["mu"], dtype=np.float64),
        sigma=np.asarray(atlas["sigma"], dtype=np.float64),
        names=np.asarray(atlas["names"], dtype=object),
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(fold_metric, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "diagnostics.json").write_text(
        json.dumps({
            "method": "Statistical Atlas (official train_atlas; position-only; label-free test adapter)",
            "official_repo": str(STAT),
            "test_labels_used_for_alignment": False,
            "test_labels_used_for_score_matrix": False,
            "test_labels_used_only_for_final_metrics": True,
            "training": train_diag,
            "alignment": align_diag,
            "args": {
                "min_counts": args.min_counts,
                "epsilon_pos": args.epsilon_pos,
                "atlas_iterations": args.atlas_iterations,
                "icp_iterations": args.icp_iterations,
                "icp_trim": args.icp_trim,
                "covariance_ridge_fraction": args.covariance_ridge_fraction,
            },
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return qdf, fold_metric


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["atanas", "rld", "all"], default="all")
    p.add_argument("--fold", type=int, default=0, help="0 = all five folds")
    p.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "baselines/official/runs/stat_atlas_official",
    )

    # Official defaults: min_counts=2, epsilon position=1000, n_iter=10.
    p.add_argument("--min-counts", type=int, default=2)
    p.add_argument("--epsilon-pos", type=float, default=1000.0)
    p.add_argument("--atlas-iterations", type=int, default=10)
    p.add_argument(
        "--min-train-labels-per-worm",
        type=int,
        default=4,
        help=(
            "Minimum clean unique identities required for a TRAIN worm to "
            "participate in Statistical Atlas fitting. Sparse training worms "
            "are skipped; validation/test worms are never filtered here."
        ),
    )

    # Fixed label-free adapter parameters. Do not choose these on outer-test labels.
    p.add_argument("--icp-iterations", type=int, default=25)
    p.add_argument("--icp-trim", type=float, default=0.80)
    p.add_argument("--covariance-ridge-fraction", type=float, default=1e-6)

    args = p.parse_args()
    if args.fold not in (0, 1, 2, 3, 4, 5):
        raise ValueError("--fold must be 0 or 1..5")
    if not (0.5 <= args.icp_trim <= 1.0):
        raise ValueError("--icp-trim must be in [0.5,1]")

    args.out_root.mkdir(parents=True, exist_ok=True)
    datasets = ("atanas", "rld") if args.dataset == "all" else (args.dataset,)
    folds = range(1, 6) if args.fold == 0 else (args.fold,)

    all_q = []
    fold_rows = []

    for dataset in datasets:
        for fold in folds:
            print("=" * 100, flush=True)
            print(f"[StatAtlas] dataset={dataset} fold={fold}", flush=True)
            qdf, metric = evaluate_fold(args, dataset, fold)
            all_q.append(qdf)
            fold_rows.append(metric)
            print(json.dumps(metric, indent=2), flush=True)

    q = pd.concat(all_q, ignore_index=True)
    fold_df = pd.DataFrame(fold_rows)

    q.to_csv(args.out_root / "query_level.csv", index=False)
    fold_df.to_csv(args.out_root / "fold_metrics.csv", index=False)

    summary_rows = []
    for dataset, g in fold_df.groupby("dataset", sort=False):
        row = {"dataset": dataset, "method": "Statistical Atlas", "folds": int(len(g))}
        for col in ("top1", "top5", "mrr", "hungarian_top1"):
            vals = g[col].to_numpy(dtype=np.float64)
            row[col + "_mean"] = float(vals.mean())
            row[col + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(row)

        print("\n" + dataset.upper())
        print(
            f"Top1      = {100*row['top1_mean']:.2f}% ± {100*row['top1_sd']:.2f}%\n"
            f"Top5      = {100*row['top5_mean']:.2f}% ± {100*row['top5_sd']:.2f}%\n"
            f"Hungarian = {100*row['hungarian_top1_mean']:.2f}% ± {100*row['hungarian_top1_sd']:.2f}%\n"
            f"MRR       = {row['mrr_mean']:.4f} ± {row['mrr_sd']:.4f}"
        )

    pd.DataFrame(summary_rows).to_csv(args.out_root / "summary.csv", index=False)

    provenance = {
        "method": "Statistical Atlas",
        "official_source": str(STAT),
        "official_training_api": "PositionOnlyAtlas.train_atlas (inherited official loop; XYZ-only update_beta override)",
        "input": "XYZ only; constant dummy color channel",
        "training_data": (
            "locked outer-fold TRAIN worms only; training worms with fewer "
            "than min_train_labels_per_worm clean unique identities are excluded "
            "as method-ineligible for 3-D atlas fitting"
        ),
        "test_alignment": "label-free PCA + trimmed similarity ICP",
        "pair_score": "atlas identity posterior overlap",
        "primary_metric": "direct per-query Top-1 before Hungarian",
        "outer_test_labels": "used only after score matrix generation for metrics",
    }
    (args.out_root / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
