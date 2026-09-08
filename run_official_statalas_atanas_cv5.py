#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")

STAT_REPO = Path(
    "/home/ubuntu/klb/nuclr/nuclr/"
    "benchmark_official/third_party/wormid_official/stat-atlas"
)

DATA_ROOTS = {
    "atanas": ROOT / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": ROOT / "Data/Dunn_001623/cv5_grouped_v1",
}
DATASET = "atanas"
DATA_ROOT = DATA_ROOTS[DATASET]
OUT_ROOT = ROOT / "runs/unified_main_benchmark_cv5_seed42_v1/rerun/statatlas/atanas"

# Official demo values.
ATLAS_EPSILON_POSITION = 0.1
ATLAS_EPSILON_COLOR = 0.1
ATLAS_MIN_COUNTS = 2
ATLAS_ITERATIONS = 100

# Automatic label-free test-time alignment.
ICP_ITERATIONS = 30
ICP_TOL = 1e-7


if not STAT_REPO.is_dir():
    raise FileNotFoundError(STAT_REPO)

sys.path.insert(0, str(STAT_REPO))
sys.path.insert(
    0,
    str(ROOT / "paper_submission_data/5fold_cross_validation/code"),
)

# OFFICIAL GITHUB CODE
from models import Atlas
import utils as stat_utils
from benchmark_official.adapters.stat_atlas.evaluate_statatlas_official import (
    PositionOnlyAtlas as PositionOnlyAtlasSafe,
)


INVALID_LABELS = {
    "",
    "-1",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "unlabeled",
    "unlabelled",
}


# ============================================================================
# Geometry-only StatAtlas adaptation
# ============================================================================
#
# Atanas/SF has no NeuroPAL RGB. The official Atlas.update_beta() always
# fits TWO independent affine models:
#
#   (1) XYZ position
#   (2) RGB color
#
# Feeding constant zero RGB makes the official color MCR regression
# singular. We therefore preserve the OFFICIAL positional branch exactly
# and disable only the unavailable color regression.
#
# IMPORTANT:
#   - position MCR_solver = official stat-atlas/utils.py
#   - position covariance = official StatAtlas covariance
#   - affine alignment = official positional formulation
#   - atlas mean/covariance updates = inherited official code
#   - no test GT is used
#   - no synthetic/random RGB is introduced
#
class GeometryOnlyAtlas(Atlas):

    def update_beta(self, X, model):
        """
        Official StatAtlas positional affine update with the RGB branch disabled.

        X: [N, 3+C, K]
        """

        beta = np.zeros(
            (X.shape[1], X.shape[1], X.shape[2]),
            dtype=np.float64,
        )

        beta0 = np.zeros(
            (1, X.shape[1], X.shape[2]),
            dtype=np.float64,
        )

        aligned = np.zeros_like(
            X,
            dtype=np.float64,
        )

        C = X.shape[1] - 3

        # Color transform is identity only for dimensional compatibility.
        # Since all observed colors are zero, it contributes no information.
        if C > 0:
            for j in range(X.shape[2]):
                beta[3:, 3:, j] = np.eye(C)

        params = {}
        position_cost = 0.0

        for j in range(X.shape[2]):

            # Same validity semantics as official code:
            # a neuron row is present iff not all channels are NaN.
            idx = ~np.isnan(
                X[:, :, j]
            ).all(axis=1)

            if int(idx.sum()) < 4:
                raise RuntimeError(
                    f"Too few observed atlas identities in worm {j}: "
                    f"{int(idx.sum())}"
                )

            # ------------------------------------------------------------
            # EXACT OFFICIAL POSITION BRANCH from models.py:update_beta()
            # ------------------------------------------------------------
            design = np.concatenate(
                (
                    X[idx, :3, j],
                    np.ones(
                        (int(idx.sum()), 1),
                        dtype=np.float64,
                    ),
                ),
                axis=1,
            )

            R = stat_utils.MCR_solver(
                design,
                model["mu"][idx, :3],
                model["sigma"][:3, :3, idx],
            )

            beta[:3, :3, j] = R[:3, :3]
            beta0[:, :3, j] = R[3, None]

            # Block-diagonal transform, exactly matching StatAtlas semantics.
            aligned[:, :, j] = (
                X[:, :, j] @ beta[:, :, j]
                + beta0[:, :, j]
            )

            # Geometry-only Mahalanobis objective.
            observed = np.nonzero(idx)[0]

            for i in observed:
                delta = (
                    aligned[i, :3, j]
                    - model["mu"][i, :3]
                )

                cov = model[
                    "sigma"
                ][:3, :3, i]

                inv_cov = np.linalg.inv(
                    cov
                )

                position_cost += float(
                    np.sqrt(
                        max(
                            delta
                            @ inv_cov
                            @ delta,
                            0.0,
                        )
                    )
                )

        params["beta"] = beta
        params["beta0"] = beta0

        # Keep official two-entry cost API.
        # Color cost is zero because the modality is unavailable.
        cost = [
            float(position_cost),
            0.0,
        ]

        return params, aligned, cost



# ============================================================================
# Minimal adapter to the OFFICIAL StatAtlas Image API.
#
# Atlas.train_atlas() only needs:
#   bodypart
#   scale
#   get_annotations()
#   get_positions()
#   get_colors_readout()
#
# We deliberately give zero RGB because Atanas/SF in this benchmark
# does not provide NeuroPAL RGB.
# ============================================================================

class AtanasImage:

    def __init__(
        self,
        uid: str,
        xyz: np.ndarray,
        labels: list[str],
    ):
        self.uid = str(uid)
        self.bodypart = "head"
        self.scale = np.ones(3, dtype=np.float64)

        self._xyz = np.asarray(
            xyz,
            dtype=np.float64,
        )

        self._labels = list(labels)

        # Compatibility-only neutral RGB.
        # It carries ZERO discriminative information.
        self._colors = np.zeros(
            (len(labels), 3),
            dtype=np.float64,
        )

        if self._xyz.shape != (len(labels), 3):
            raise ValueError(
                f"Bad image shape: "
                f"xyz={self._xyz.shape}, "
                f"labels={len(labels)}"
            )

    def get_annotations(self):
        return list(self._labels)

    def get_positions(self, scale=None):
        if scale is None:
            scale = self.scale

        scale = np.asarray(
            scale,
            dtype=np.float64,
        )

        return self._xyz * scale[None, :]

    def get_colors_readout(self):
        return self._colors.copy()


# ============================================================================
# Dataset IO
# ============================================================================

def find_key(npz, candidates):
    for key in candidates:
        if key in npz.files:
            return key
    return None


def clean_label(value):
    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="ignore",
        )

    value = str(value).strip()

    if value.lower() in INVALID_LABELS:
        return ""

    return value


def load_record(path: Path):
    z = np.load(
        path,
        allow_pickle=True,
    )

    xyz_key = find_key(
        z,
        [
            "xyz",
            "X",
            "coords",
            "coordinates",
            "positions",
        ],
    )

    label_key = find_key(
        z,
        [
            "cell_id",
            "cell_ids",
            "labels",
            "label",
            "identity",
            "identities",
            "ids",
        ],
    )

    if xyz_key is None:
        raise RuntimeError(
            f"No XYZ key in {path}; "
            f"keys={z.files}"
        )

    if label_key is None:
        raise RuntimeError(
            f"No label key in {path}; "
            f"keys={z.files}"
        )

    xyz = np.asarray(
        z[xyz_key],
        dtype=np.float64,
    )

    labels_raw = np.asarray(
        z[label_key],
        dtype=object,
    ).reshape(-1)

    labels = [
        clean_label(x)
        for x in labels_raw
    ]

    if xyz.ndim != 2:
        raise RuntimeError(
            f"Bad XYZ ndim: {xyz.shape}"
        )

    if xyz.shape[1] != 3:
        if xyz.shape[0] == 3:
            xyz = xyz.T
        else:
            raise RuntimeError(
                f"Expected Nx3 XYZ: {path} "
                f"{xyz.shape}"
            )

    if xyz.shape[0] != len(labels):
        raise RuntimeError(
            f"XYZ/label length mismatch: "
            f"{path}: {xyz.shape[0]} "
            f"vs {len(labels)}"
        )

    mask_keys = [
        key
        for key in (
            "labeled_mask",
            "supervised_mask",
            "certain_mask",
            "clean_mask",
        )
        if key in z.files
    ]
    supervised = np.ones(len(labels), dtype=bool)
    for key in mask_keys:
        value = np.asarray(z[key]).astype(bool).reshape(-1)
        if len(value) != len(labels):
            raise RuntimeError(f"Mask length mismatch: {path} key={key}")
        supervised &= value

    finite = np.isfinite(xyz).all(axis=1)

    # Match the current benchmark convention:
    # only unique labeled identities contribute to metrics.
    counts = Counter(
        label
        for label, sup in zip(
            labels,
            supervised,
        )
        if sup and label
    )

    unique_supervised = np.asarray(
        [
            bool(
                finite[i]
                and supervised[i]
                and labels[i]
                and counts[labels[i]] == 1
            )
            for i in range(len(labels))
        ],
        dtype=bool,
    )

    return {
        "path": path,
        "uid": clean_label(np.asarray(z["recording_uid"]).reshape(-1)[0])
        if "recording_uid" in z.files else path.stem,
        "xyz": xyz,
        "labels": labels,
        "finite": finite,
        "supervised": supervised,
        "eval_mask": unique_supervised,
        "xyz_key": xyz_key,
        "label_key": label_key,
        "mask_key": "+".join(mask_keys) if mask_keys else None,
    }


def split_files(fold: int, split: str):
    folder = (
        DATA_ROOT /
        f"fold_{fold}" /
        split
    )

    files = sorted(
        folder.glob("*.npz")
    )

    if not files:
        raise FileNotFoundError(
            f"No NPZ files: {folder}"
        )

    return files


# ============================================================================
# Official StatAtlas training
# ============================================================================

def make_training_image(record):
    """
    Atlas construction is supervised, so only valid unique
    train identities enter the atlas.

    Test labels are NEVER used in fitting.
    """

    idx = np.nonzero(
        record["eval_mask"]
    )[0]

    if len(idx) < 4:
        raise RuntimeError(
            f"Too few annotated neurons: "
            f"{record['path']}"
        )

    xyz = record["xyz"][idx]

    labels = [
        record["labels"][i]
        for i in idx
    ]

    return AtanasImage(
        record["uid"],
        xyz,
        labels,
    )


def train_fold_atlas(fold: int):

    train_paths = split_files(
        fold,
        "train",
    )

    all_records = [
        load_record(p)
        for p in train_paths
    ]

    records = [
        record
        for record in all_records
        if int(record["eval_mask"].sum()) >= 4
    ]
    skipped_records = [
        {
            "uid": record["uid"],
            "path": str(record["path"].resolve()),
            "clean_unique_identities": int(record["eval_mask"].sum()),
            "reason": "fewer than four clean unique training identities",
        }
        for record in all_records
        if int(record["eval_mask"].sum()) < 4
    ]
    if len(records) < 2:
        raise RuntimeError("Fewer than two StatAtlas-eligible outer-train worms")

    images = [
        make_training_image(r)
        for r in records
    ]

    print()
    print("=" * 110)
    print(
        f"STATATLAS OFFICIAL TRAIN — "
        f"{DATASET.upper()} FOLD{fold}"
    )
    print("=" * 110)
    print("training worms :", len(images))
    print("skipped worms  :", len(skipped_records))
    print(
        "train neurons  :",
        [
            int(r["eval_mask"].sum())
            for r in records
        ],
    )
    print(
        "epsilon        :",
        [
            ATLAS_EPSILON_POSITION,
            ATLAS_EPSILON_COLOR,
        ],
    )
    print("min_counts     :", ATLAS_MIN_COUNTS)
    print("iterations     :", ATLAS_ITERATIONS)
    print("RGB            : ZERO / DISABLED")
    print()

    # ========================================================
    # OFFICIAL GITHUB IMPLEMENTATION
    # ========================================================

    # Use the pinned official XYZ update with the adapter's deterministic
    # low-rank/SVD safeguards. This is necessary for sparse Kato folds and
    # does not introduce labels, validation data, or test data.
    atlas_obj = PositionOnlyAtlasSafe(
        min_counts=ATLAS_MIN_COUNTS,
        epsilon=[
            ATLAS_EPSILON_POSITION,
            ATLAS_EPSILON_COLOR,
        ],
    )

    trained_atlas, aligned, params, cost, init_aligned = (
        atlas_obj.train_atlas(
            images,
            "head",
            n_iter=ATLAS_ITERATIONS,
        )
    )

    names = [
        clean_label(x)
        for x in trained_atlas["names"]
    ]

    mu = np.asarray(
        trained_atlas["mu"],
        dtype=np.float64,
    )

    sigma = np.asarray(
        trained_atlas["sigma"],
        dtype=np.float64,
    )

    if mu.shape[0] != len(names):
        raise RuntimeError(
            "Atlas name/mean mismatch"
        )

    if sigma.shape[2] != len(names):
        raise RuntimeError(
            "Atlas name/sigma mismatch"
        )

    print()
    print("atlas identities :", len(names))
    print("atlas mu shape   :", mu.shape)
    print("atlas sigma shape:", sigma.shape)

    return (
        trained_atlas,
        records,
        skipped_records,
    )


# ============================================================================
# Geometry-only Mahalanobis model
# ============================================================================

def atlas_geometry(trained_atlas):

    names = [
        clean_label(x)
        for x in trained_atlas["names"]
    ]

    mu = np.asarray(
        trained_atlas["mu"],
        dtype=np.float64,
    )[:, :3]

    raw_sigma = np.asarray(
        trained_atlas["sigma"],
        dtype=np.float64,
    )[:3, :3, :]

    keep_names = []
    keep_mu = []
    keep_cov = []

    for j, name in enumerate(names):

        if not name:
            continue

        mean = mu[j]
        cov = raw_sigma[:, :, j]

        if not np.isfinite(mean).all():
            continue

        if not np.isfinite(cov).all():
            continue

        # Official model uses epsilon regularization during
        # covariance estimation. The final Github train_atlas()
        # recomputes sigma once without passing reg, so restore
        # the SAME official positional epsilon here for stable
        # out-of-sample Mahalanobis inference.
        cov = (
            0.5 * (cov + cov.T)
            + ATLAS_EPSILON_POSITION
            * np.eye(3)
        )

        keep_names.append(name)
        keep_mu.append(mean)
        keep_cov.append(cov)

    if not keep_names:
        raise RuntimeError(
            "No usable atlas identities"
        )

    mu = np.stack(
        keep_mu,
        axis=0,
    )

    cov = np.stack(
        keep_cov,
        axis=0,
    )

    inv_cov = np.stack(
        [
            np.linalg.pinv(c)
            for c in cov
        ],
        axis=0,
    )

    return (
        keep_names,
        mu,
        cov,
        inv_cov,
    )


def mahalanobis_cost(
    xyz,
    atlas_mu,
    atlas_inv_cov,
):
    """
    Cost[i,j] =
      (x_i - mu_j)^T Sigma_j^-1 (x_i - mu_j)
    """

    diff = (
        xyz[:, None, :]
        - atlas_mu[None, :, :]
    )

    return np.einsum(
        "nmd,mde,nme->nm",
        diff,
        atlas_inv_cov,
        diff,
        optimize=True,
    )


# ============================================================================
# Label-free automatic ICP / affine alignment
#
# Original StatAtlas automatic inference:
# Mahalanobis correspondence + iterative affine update.
#
# The affine update below calls the OFFICIAL utils.MCR_solver.
# No test identity is used.
# ============================================================================

def initial_center_scale(
    xyz,
    atlas_mu,
):

    q_center = np.median(
        xyz,
        axis=0,
    )

    a_center = np.median(
        atlas_mu,
        axis=0,
    )

    q0 = xyz - q_center
    a0 = atlas_mu - a_center

    q_scale = np.sqrt(
        np.mean(
            np.sum(q0 * q0, axis=1)
        )
    )

    a_scale = np.sqrt(
        np.mean(
            np.sum(a0 * a0, axis=1)
        )
    )

    if (
        not np.isfinite(q_scale)
        or q_scale < 1e-8
    ):
        scale = 1.0
    else:
        scale = a_scale / q_scale

    aligned = (
        q0 * scale
        + a_center
    )

    return aligned


def assignment(cost):

    rows, cols = linear_sum_assignment(
        cost
    )

    return (
        np.asarray(rows, dtype=int),
        np.asarray(cols, dtype=int),
    )


def official_affine_update(
    original_xyz,
    matched_query_rows,
    matched_atlas_cols,
    atlas_mu,
    atlas_cov,
):
    """
    Use OFFICIAL stat-atlas/utils.py MCR_solver.

    This is the same solver called by Atlas.update_beta().
    """

    q = original_xyz[
        matched_query_rows
    ]

    y = atlas_mu[
        matched_atlas_cols
    ]

    covariance = atlas_cov[
        matched_atlas_cols
    ].transpose(1, 2, 0)

    design = np.concatenate(
        [
            q,
            np.ones(
                (len(q), 1),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    try:
        R = stat_utils.MCR_solver(
            design,
            y,
            covariance,
        )
    except np.linalg.LinAlgError as exc:
        if "Singular matrix" not in str(exc):
            raise
        original_inv = np.linalg.inv
        np.linalg.inv = np.linalg.pinv
        try:
            R = stat_utils.MCR_solver(
                design,
                y,
                covariance,
            )
        finally:
            np.linalg.inv = original_inv

    R = np.asarray(
        R,
        dtype=np.float64,
    )

    if R.shape != (4, 3):
        raise RuntimeError(
            f"MCR_solver returned "
            f"{R.shape}, expected (4,3)"
        )

    aligned = (
        original_xyz @ R[:3, :]
        + R[3, :][None, :]
    )

    return aligned, R


def automatic_align(
    original_xyz,
    atlas_mu,
    atlas_cov,
    atlas_inv_cov,
):

    aligned = initial_center_scale(
        original_xyz,
        atlas_mu,
    )

    previous_pairs = None

    best = None

    for iteration in range(
        ICP_ITERATIONS
    ):

        cost = mahalanobis_cost(
            aligned,
            atlas_mu,
            atlas_inv_cov,
        )

        rows, cols = assignment(cost)

        if len(rows) < 4:
            raise RuntimeError(
                "Too few ICP correspondences"
            )

        objective = float(
            cost[rows, cols].mean()
        )

        if (
            best is None
            or objective < best["objective"]
        ):
            best = {
                "objective": objective,
                "aligned": aligned.copy(),
                "cost": cost.copy(),
                "rows": rows.copy(),
                "cols": cols.copy(),
                "iteration": iteration,
            }

        pairs = tuple(
            zip(
                rows.tolist(),
                cols.tolist(),
            )
        )

        if pairs == previous_pairs:
            break

        previous_pairs = pairs

        new_aligned, R = (
            official_affine_update(
                original_xyz,
                rows,
                cols,
                atlas_mu,
                atlas_cov,
            )
        )

        delta = float(
            np.sqrt(
                np.mean(
                    (
                        new_aligned
                        - aligned
                    ) ** 2
                )
            )
        )

        aligned = new_aligned

        if delta < ICP_TOL:
            break

    # Evaluate final state too.
    final_cost = mahalanobis_cost(
        aligned,
        atlas_mu,
        atlas_inv_cov,
    )

    rows, cols = assignment(
        final_cost
    )

    final_obj = float(
        final_cost[
            rows,
            cols,
        ].mean()
    )

    if (
        best is None
        or final_obj < best["objective"]
    ):
        best = {
            "objective": final_obj,
            "aligned": aligned.copy(),
            "cost": final_cost.copy(),
            "rows": rows.copy(),
            "cols": cols.copy(),
            "iteration": iteration + 1,
        }

    return best


# ============================================================================
# Ranked Hungarian assignments
#
# WormID StatAtlas protocol:
# rank 1 = Hungarian
# rank 2 = forbid rank-1 pairs, rerun
# ...
# rank 5.
# ============================================================================

def repeated_hungarian(
    cost,
    ranks=5,
):

    work = np.asarray(
        cost,
        dtype=np.float64,
    ).copy()

    assignments = []

    for rank in range(ranks):

        rows, cols = linear_sum_assignment(
            work
        )

        mapping = {
            int(r): int(c)
            for r, c in zip(rows, cols)
        }

        assignments.append(mapping)

        for r, c in zip(rows, cols):
            work[r, c] = np.inf

    return assignments


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_record(
    record,
    atlas_names,
    atlas_mu,
    atlas_cov,
    atlas_inv_cov,
):

    finite_idx = np.nonzero(
        record["finite"]
    )[0]

    xyz = record["xyz"][
        finite_idx
    ]

    original_to_query = {
        int(original): int(i)
        for i, original
        in enumerate(finite_idx)
    }

    result = automatic_align(
        xyz,
        atlas_mu,
        atlas_cov,
        atlas_inv_cov,
    )

    cost = result["cost"]

    candidate_index = {
        name: j
        for j, name
        in enumerate(atlas_names)
    }

    # Standard benchmark retrieval rankings.
    local_order = np.argsort(
        cost,
        axis=1,
    )

    # Official globally coherent ranks.
    global_ranked = (
        repeated_hungarian(
            cost,
            ranks=5,
        )
    )

    top_assignment = (
        global_ranked[0]
    )

    rows = []

    eval_original = np.nonzero(
        record["eval_mask"]
        & record["finite"]
    )[0]

    for original_row in eval_original:

        q = original_to_query[
            int(original_row)
        ]

        gt = record["labels"][
            original_row
        ]

        covered = (
            gt in candidate_index
        )

        local_top1 = 0
        local_top5 = 0
        reciprocal_rank = 0.0
        hungarian_correct = 0
        official_top5 = 0
        local_rank = None

        if covered:

            target = candidate_index[gt]

            ordered = local_order[q]

            position = np.nonzero(
                ordered == target
            )[0]

            if len(position) != 1:
                raise RuntimeError(
                    "Target rank failure"
                )

            local_rank = (
                int(position[0]) + 1
            )

            local_top1 = int(
                local_rank == 1
            )

            local_top5 = int(
                local_rank <= 5
            )

            reciprocal_rank = (
                1.0 / local_rank
            )

            hungarian_correct = int(
                top_assignment.get(
                    q,
                    -1,
                )
                == target
            )

            official_top5 = int(
                any(
                    mapping.get(q, -1)
                    == target
                    for mapping
                    in global_ranked
                )
            )

        pred_local = (
            atlas_names[
                int(local_order[q, 0])
            ]
            if len(atlas_names)
            else ""
        )

        pred_hung = (
            atlas_names[
                top_assignment[q]
            ]
            if q in top_assignment
            else ""
        )

        rows.append(
            {
                "worm": record["uid"],
                "source_path": str(
                    record["path"]
                ),
                "query_row": int(
                    original_row
                ),
                "gt_label": gt,
                "covered": int(covered),
                "local_prediction":
                    pred_local,
                "hungarian_prediction":
                    pred_hung,
                "local_rank":
                    local_rank,
                "top1": local_top1,
                "top5": local_top5,
                "mrr":
                    reciprocal_rank,
                "hungarian":
                    hungarian_correct,
                "official_top5":
                    official_top5,
                "icp_objective":
                    result["objective"],
                "icp_best_iteration":
                    result["iteration"],
            }
        )

    return rows


def summarize(rows):

    q = len(rows)

    if q == 0:
        raise RuntimeError(
            "No evaluation queries"
        )

    covered = sum(
        r["covered"]
        for r in rows
    )

    return {
        "queries": q,
        "covered_queries": covered,
        "coverage": covered / q,
        "top1": sum(
            r["top1"]
            for r in rows
        ) / q,
        "top5": sum(
            r["top5"]
            for r in rows
        ) / q,
        "mrr": sum(
            r["mrr"]
            for r in rows
        ) / q,
        "hungarian": sum(
            r["hungarian"]
            for r in rows
        ) / q,
        "official_repeated_hungarian_top5":
            sum(
                r["official_top5"]
                for r in rows
            ) / q,
        "covered_top1": (
            sum(
                r["top1"]
                for r in rows
                if r["covered"]
            )
            / covered
            if covered
            else 0.0
        ),
        "covered_hungarian": (
            sum(
                r["hungarian"]
                for r in rows
                if r["covered"]
            )
            / covered
            if covered
            else 0.0
        ),
    }


def write_csv(path, rows):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(rows)


def sha256(path):

    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(
                1024 * 1024
            )
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def git_commit():

    try:
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(STAT_REPO),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
    except Exception:
        return None


# ============================================================================
# Fold
# ============================================================================

def run_fold(
    fold,
    split,
):

    out = (
        OUT_ROOT /
        f"fold{fold}"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    trained_atlas, train_records, skipped_train_records = (
        train_fold_atlas(fold)
    )

    (
        atlas_names,
        atlas_mu,
        atlas_cov,
        atlas_inv_cov,
    ) = atlas_geometry(
        trained_atlas
    )

    atlas_file = (
        out /
        "trained_atlas.pkl"
    )

    with atlas_file.open(
        "wb"
    ) as f:
        pickle.dump(
            trained_atlas,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    eval_paths = split_files(
        fold,
        split,
    )

    all_rows = []

    print()
    print(
        f"STATATLAS LABEL-FREE "
        f"EVALUATION fold{fold} "
        f"split={split}"
    )
    print(
        "atlas candidates:",
        len(atlas_names),
    )

    for number, path in enumerate(
        eval_paths,
        start=1,
    ):

        record = load_record(path)

        rows = evaluate_record(
            record,
            atlas_names,
            atlas_mu,
            atlas_cov,
            atlas_inv_cov,
        )

        all_rows.extend(rows)

        if rows:
            m = summarize(rows)
            print(
                f"[{number:02d}/"
                f"{len(eval_paths):02d}] "
                f"{record['uid']} "
                f"Q={m['queries']:4d} "
                f"Cov={100*m['coverage']:6.2f}% "
                f"T1={100*m['top1']:6.2f}% "
                f"T5={100*m['top5']:6.2f}% "
                f"Hung={100*m['hungarian']:6.2f}%"
            )
        else:
            print(
                f"[{number:02d}/{len(eval_paths):02d}] "
                f"{record['uid']} skipped=no_evaluable_atlas_identity"
            )

    metrics = summarize(
        all_rows
    )

    payload = {
        "method":
            "StatAtlas official Github "
            "geometry-only adaptation",
        "official_repo":
            str(STAT_REPO),
        "official_commit":
            git_commit(),
        "dataset": DATASET,
        "fold": fold,
        "split": split,
        "fold_root": str((DATA_ROOT / f"fold_{fold}").resolve()),
        "split_counts": {
            name: len(split_files(fold, name))
            for name in ("train", "val", "test")
        },
        "model_seed": None,
        "deterministic": True,
        "training_protocol": {
            "train_only_atlas": True,
            "test_labels_used_for_alignment":
                False,
            "rgb_used": False,
            "dummy_rgb":
                [0.0, 0.0, 0.0],
            "atlas_iterations":
                ATLAS_ITERATIONS,
            "atlas_epsilon_position":
                ATLAS_EPSILON_POSITION,
            "atlas_epsilon_color":
                ATLAS_EPSILON_COLOR,
            "atlas_min_counts":
                ATLAS_MIN_COUNTS,
            "automatic_inference":
                "Mahalanobis ICP with "
                "official utils.MCR_solver",
            "icp_iterations":
                ICP_ITERATIONS,
        },
        "train_worms":
            len(train_records),
        "skipped_train_worms":
            skipped_train_records,
        "eval_worms":
            len(eval_paths),
        "atlas_candidates":
            len(atlas_names),
        "atlas_names":
            atlas_names,
        "atlas_sha256":
            sha256(atlas_file),
        "metrics":
            metrics,
    }

    (
        out /
        f"{split}_metrics.json"
    ).write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_csv(
        out /
        f"{split}_queries.csv",
        all_rows,
    )

    print()
    print(
        f"FOLD{fold} {split.upper()}: "
        f"Q={metrics['queries']} "
        f"Top1={100*metrics['top1']:.2f}% "
        f"Top5={100*metrics['top5']:.2f}% "
        f"MRR={metrics['mrr']:.4f} "
        f"Hung={100*metrics['hungarian']:.2f}% "
        f"Cov={100*metrics['coverage']:.2f}% "
        f"OfficialTop5="
        f"{100*metrics['official_repeated_hungarian_top5']:.2f}%"
    )

    return payload


# ============================================================================
# Aggregate
# ============================================================================

def aggregate(results):

    metrics = [
        r["metrics"]
        for r in results
    ]

    keys = [
        "top1",
        "top5",
        "mrr",
        "hungarian",
        "coverage",
        "covered_top1",
        "official_repeated_hungarian_top5",
    ]

    summary = {}

    for key in keys:

        x = np.asarray(
            [
                m[key]
                for m in metrics
            ],
            dtype=float,
        )

        summary[key] = {
            "mean":
                float(x.mean()),
            "sample_sd":
                float(
                    x.std(ddof=1)
                )
                if len(x) > 1
                else 0.0,
            "fold_values":
                x.tolist(),
        }

    return summary


def main():

    global DATASET, DATA_ROOT, OUT_ROOT

    p = argparse.ArgumentParser()

    p.add_argument(
        "--dataset",
        choices=["atanas", "rld"],
        default="atanas",
    )

    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
    )

    p.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
    )

    p.add_argument(
        "--split",
        choices=[
            "val",
            "test",
        ],
        default="test",
    )

    args = p.parse_args()

    DATASET = args.dataset
    DATA_ROOT = DATA_ROOTS[DATASET]
    OUT_ROOT = args.out_root or (
        ROOT /
        "runs/unified_main_benchmark_cv5_seed42_v1/"
        f"rerun/statatlas/{DATASET}"
    )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 110)
    print(
        f"OFFICIAL STATALAS -> {DATASET.upper()} "
        "GROUPED CV5"
    )
    print("=" * 110)
    print(
        "repo   :", STAT_REPO
    )
    print(
        "commit :", git_commit()
    )
    print(
        "data   :", DATA_ROOT
    )
    print(
        "split  :", args.split
    )
    print(
        "folds  :", args.folds
    )
    print(
        "TEST GT USED DURING MATCHING: NO"
    )
    print(
        "RGB USED: NO"
    )
    print("=" * 110)

    results = []

    for fold in args.folds:

        results.append(
            run_fold(
                fold,
                args.split,
            )
        )

    summary = aggregate(
        results
    )

    aggregate_payload = {
        "method":
            "StatAtlas official Github "
            "geometry-only adaptation",
        "official_commit":
            git_commit(),
        "dataset":
            DATASET,
        "split":
            args.split,
        "folds":
            args.folds,
        "fold_results":
            results,
        "summary":
            summary,
    }

    aggregate_file = (
        OUT_ROOT /
        f"{args.split}_aggregate.json"
    )

    aggregate_file.write_text(
        json.dumps(
            aggregate_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print(
        "FINAL — MEAN ± SAMPLE SD "
        "ACROSS BIOLOGICAL FOLDS"
    )
    print("=" * 110)

    for key in [
        "top1",
        "top5",
        "mrr",
        "hungarian",
        "coverage",
        "covered_top1",
        "official_repeated_hungarian_top5",
    ]:

        m = summary[key]["mean"]
        sd = summary[key]["sample_sd"]

        if key == "mrr":
            print(
                f"{key:36s}: "
                f"{m:.4f} ± {sd:.4f}"
            )
        else:
            print(
                f"{key:36s}: "
                f"{100*m:.2f} ± "
                f"{100*sd:.2f}%"
            )

    print("=" * 110)
    print(
        "saved:",
        aggregate_file,
    )


if __name__ == "__main__":
    main()
