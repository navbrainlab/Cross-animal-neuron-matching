#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Cross-animal population relation mechanism analysis

A:
    Same / Hard / Random cross-animal relation-profile similarity

B:
    Identity x Identity cross-animal relation-profile similarity heatmap

D:
    Cross-animal relation stability vs REAL MPRT per-identity Top-1 accuracy

The analysis is label-aligned:
For a query neuron i in worm a, its population relation profile is compared
with candidate neuron j in worm b using only shared anchor identities.

Supported relation modes
------------------------
geo:
    Population relation = normalized pairwise Euclidean distance.

act:
    Population relation = neuron-neuron activity correlation.

multi:
    Geometry and activity relation profiles are independently standardized
    and concatenated before cross-animal correlation.

precomputed:
    Read an NxN relation matrix directly from each NPZ using --relation-key.
    This is preferable if you have exported the exact MPRT relation matrix.

Supported prediction formats
----------------------------
CSV / JSONL / JSON / NPZ.

The loader automatically searches common field names for:
    GT identity:
        gt_id, true_id, target_id, cell_id, label, gt_label, true_label

    Predicted identity:
        pred_id, predicted_id, prediction, pred_label, predicted_label

    Optional correctness:
        correct, is_correct, top1_correct

If your predictions use different field names, use:
    --gt-field ...
    --pred-field ...

Important
---------
For the final paper figure, use test predictions only for Panel D.
Panels A/B describe a property of the biological population and can use all
animals, but you can restrict them with --splits if desired.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    pearsonr,
    spearmanr,
)


# ============================================================
# General helpers
# ============================================================

def mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_label(x):
    if x is None:
        return None

    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")

    if isinstance(x, np.generic):
        x = x.item()

    x = str(x).strip()

    bad = {
        "",
        "nan",
        "none",
        "null",
        "unknown",
        "unlabeled",
        "?",
        "-1",
    }

    if x.lower() in bad:
        return None

    return x


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan

    return float(pearsonr(x, y)[0])


def standardize_vector(x):
    x = np.asarray(x, dtype=np.float64)

    mu = np.mean(x)
    sd = np.std(x)

    if sd < 1e-10:
        return np.zeros_like(x)

    return (x - mu) / sd


def normalize_xyz(xyz):
    """
    Translation + global-scale normalization.

    No rotation alignment is applied here.
    Pairwise Euclidean distances are rotation invariant already.
    """
    xyz = np.asarray(xyz, dtype=np.float64)

    xyz = xyz - np.mean(xyz, axis=0, keepdims=True)

    rms = np.sqrt(
        np.mean(
            np.sum(xyz ** 2, axis=1)
        )
    )

    if rms < 1e-10:
        rms = 1.0

    return xyz / rms


def percentile_ci(values, alpha=0.05):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]

    if len(v) == 0:
        return np.nan, np.nan

    return (
        float(np.quantile(v, alpha / 2)),
        float(np.quantile(v, 1 - alpha / 2)),
    )


# ============================================================
# Dataset
# ============================================================

@dataclass
class Worm:
    name: str
    path: str

    labels: List[str]
    label_to_idx: Dict[str, int]

    xyz: np.ndarray
    xyz_raw: np.ndarray

    activity: Optional[np.ndarray]

    geo_relation: np.ndarray
    act_relation: Optional[np.ndarray]

    precomputed_relation: Optional[np.ndarray]


def discover_npz(data_root, splits):
    root = Path(data_root)

    found = []

    # Prefer explicit split directories when present
    for split in splits:
        p = root / split

        if p.exists():
            found.extend(
                sorted(str(x) for x in p.rglob("*.npz"))
            )

    # fallback
    if not found:
        found = sorted(
            str(x) for x in root.rglob("*.npz")
        )

    # remove likely generated outputs
    bad_tokens = (
        "prediction",
        "metric",
        "checkpoint",
        "embedding",
        "cache",
    )

    out = []

    for p in found:
        low = p.lower()
        if any(x in low for x in bad_tokens):
            continue
        out.append(p)

    return out


def load_one_worm(
    path,
    relation_mode,
    relation_key=None,
):
    d = np.load(path, allow_pickle=True)

    if "xyz" not in d:
        return None

    if "cell_id" in d:
        labels_raw = d["cell_id"]
    elif "cell_id_alt" in d:
        labels_raw = d["cell_id_alt"]
    else:
        return None

    xyz = np.asarray(d["xyz"])

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        return None

    n = xyz.shape[0]

    mask = np.ones(n, dtype=bool)

    if "labeled_mask" in d:
        lm = np.asarray(d["labeled_mask"]).astype(bool)

        if len(lm) == n:
            mask &= lm

    keep = []
    labels = []
    seen = set()

    for i in range(n):
        if not mask[i]:
            continue

        lb = clean_label(labels_raw[i])

        if lb is None:
            continue

        # Duplicate IDs inside one animal make relation alignment ambiguous.
        # Keep first occurrence.
        if lb in seen:
            continue

        seen.add(lb)
        keep.append(i)
        labels.append(lb)

    if len(keep) < 5:
        return None

    keep = np.asarray(keep, dtype=int)

    xyz_raw = xyz[keep].astype(np.float64)
    xyz_norm = normalize_xyz(xyz_raw)

    label_to_idx = {
        x: i
        for i, x in enumerate(labels)
    }

    # --------------------------------------------------------
    # Geometry relation
    # --------------------------------------------------------

    delta = (
        xyz_norm[:, None, :]
        -
        xyz_norm[None, :, :]
    )

    geo = np.sqrt(
        np.sum(delta ** 2, axis=-1)
    )

    # We use NEGATIVE distance:
    # larger relation value = geometrically more similar / closer.
    geo = -geo

    # --------------------------------------------------------
    # Activity relation
    # --------------------------------------------------------

    activity = None
    act = None

    if "activity_raw" in d:
        activity = np.asarray(
            d["activity_raw"]
        )[keep].astype(np.float64)

        if activity.ndim == 2 and activity.shape[1] >= 3:

            # normalize each neuron's time series
            mu = activity.mean(axis=1, keepdims=True)
            sd = activity.std(axis=1, keepdims=True)

            sd[sd < 1e-10] = 1.0

            activity = (activity - mu) / sd

            act = np.corrcoef(activity)

            act = np.nan_to_num(
                act,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

    if relation_mode in ("act", "multi") and act is None:
        return None

    # --------------------------------------------------------
    # Precomputed relation
    # --------------------------------------------------------

    precomputed = None

    if relation_mode == "precomputed":
        if relation_key is None:
            raise RuntimeError(
                "--relation-mode precomputed requires --relation-key"
            )

        if relation_key not in d:
            raise RuntimeError(
                f"{path}\nmissing relation key: {relation_key}\n"
                f"available keys = {list(d.keys())}"
            )

        full_r = np.asarray(
            d[relation_key]
        )

        if full_r.shape[0] != n or full_r.shape[1] != n:
            raise RuntimeError(
                f"Expected NxN relation matrix under {relation_key}, "
                f"got {full_r.shape} for N={n}"
            )

        precomputed = full_r[
            np.ix_(keep, keep)
        ].astype(np.float64)

    return Worm(
        name=Path(path).stem,
        path=path,

        labels=labels,
        label_to_idx=label_to_idx,

        xyz=xyz_norm,
        xyz_raw=xyz_raw,

        activity=activity,

        geo_relation=geo,
        act_relation=act,

        precomputed_relation=precomputed,
    )


def load_worms(
    data_root,
    splits,
    relation_mode,
    relation_key,
):
    paths = discover_npz(
        data_root,
        splits,
    )

    print(
        f"[DATA] discovered {len(paths)} candidate NPZ files"
    )

    worms = []

    for p in paths:
        try:
            w = load_one_worm(
                p,
                relation_mode=relation_mode,
                relation_key=relation_key,
            )

            if w is not None:
                worms.append(w)

        except Exception as e:
            print(
                f"[WARN] failed loading {p}: {e}"
            )

    print(
        f"[DATA] usable worms = {len(worms)}"
    )

    if len(worms) < 2:
        raise RuntimeError(
            "Need at least two usable animals."
        )

    counts = [len(w.labels) for w in worms]

    print(
        f"[DATA] labeled neurons per worm: "
        f"min={min(counts)} "
        f"median={np.median(counts):.1f} "
        f"max={max(counts)}"
    )

    return worms


# ============================================================
# Relation profiles
# ============================================================

def common_identities(wa, wb):
    return sorted(
        set(wa.labels)
        &
        set(wb.labels)
    )


def get_relation(w, mode):
    if mode == "geo":
        return w.geo_relation

    if mode == "act":
        return w.act_relation

    if mode == "precomputed":
        return w.precomputed_relation

    raise ValueError(mode)


def scalar_relation_similarity(
    wa,
    ida,
    wb,
    idb,
    mode,
    min_common_anchors,
):
    """
    Relation-profile similarity between query identity ida in animal A
    and candidate identity idb in animal B.

    Profiles are aligned through shared named anchor neurons.
    """

    anchors = sorted(
        set(wa.labels)
        &
        set(wb.labels)
    )

    # Do not use either compared identity as its own anchor
    anchors = [
        x
        for x in anchors
        if x != ida and x != idb
    ]

    if len(anchors) < min_common_anchors:
        return np.nan

    if (
        ida not in wa.label_to_idx
        or
        idb not in wb.label_to_idx
    ):
        return np.nan

    Ra = get_relation(wa, mode)
    Rb = get_relation(wb, mode)

    if Ra is None or Rb is None:
        return np.nan

    ia = wa.label_to_idx[ida]
    ib = wb.label_to_idx[idb]

    aa = np.array(
        [
            wa.label_to_idx[x]
            for x in anchors
        ],
        dtype=int,
    )

    bb = np.array(
        [
            wb.label_to_idx[x]
            for x in anchors
        ],
        dtype=int,
    )

    va = Ra[ia, aa]
    vb = Rb[ib, bb]

    return safe_corr(va, vb)


def multimodal_relation_similarity(
    wa,
    ida,
    wb,
    idb,
    min_common_anchors,
):
    """
    Geometry + activity.

    The two relation modalities are standardized independently,
    concatenated, and compared through Pearson correlation.

    This avoids arbitrary dominance from different numeric scales.
    """

    anchors = sorted(
        set(wa.labels)
        &
        set(wb.labels)
    )

    anchors = [
        x
        for x in anchors
        if x != ida and x != idb
    ]

    if len(anchors) < min_common_anchors:
        return np.nan

    if (
        ida not in wa.label_to_idx
        or
        idb not in wb.label_to_idx
    ):
        return np.nan

    if (
        wa.act_relation is None
        or
        wb.act_relation is None
    ):
        return np.nan

    ia = wa.label_to_idx[ida]
    ib = wb.label_to_idx[idb]

    aa = np.array(
        [wa.label_to_idx[x] for x in anchors]
    )

    bb = np.array(
        [wb.label_to_idx[x] for x in anchors]
    )

    ga = wa.geo_relation[ia, aa]
    gb = wb.geo_relation[ib, bb]

    aa_rel = wa.act_relation[ia, aa]
    ab_rel = wb.act_relation[ib, bb]

    ga = standardize_vector(ga)
    gb = standardize_vector(gb)

    aa_rel = standardize_vector(aa_rel)
    ab_rel = standardize_vector(ab_rel)

    va = np.concatenate(
        [ga, aa_rel]
    )

    vb = np.concatenate(
        [gb, ab_rel]
    )

    return safe_corr(
        va,
        vb,
    )


def relation_similarity(
    wa,
    ida,
    wb,
    idb,
    mode,
    min_common_anchors,
):
    if mode == "multi":
        return multimodal_relation_similarity(
            wa,
            ida,
            wb,
            idb,
            min_common_anchors,
        )

    return scalar_relation_similarity(
        wa,
        ida,
        wb,
        idb,
        mode,
        min_common_anchors,
    )


# ============================================================
# Hard negatives
# ============================================================

def hard_negative_ids(
    target_worm,
    true_id,
    candidate_ids,
    hard_k,
):
    """
    Hard negatives are spatially closest wrong identities
    in the TARGET animal.

    This is intentionally defined independently of the tested
    relation similarity.
    """

    if true_id not in target_worm.label_to_idx:
        return []

    gt_idx = target_worm.label_to_idx[
        true_id
    ]

    p = target_worm.xyz[
        gt_idx
    ]

    pool = [
        x
        for x in candidate_ids
        if (
            x != true_id
            and
            x in target_worm.label_to_idx
        )
    ]

    scored = []

    for cid in pool:
        q = target_worm.xyz[
            target_worm.label_to_idx[cid]
        ]

        dist = float(
            np.linalg.norm(p - q)
        )

        scored.append(
            (dist, cid)
        )

    scored.sort()

    return [
        cid
        for _, cid in scored[:hard_k]
    ]


# ============================================================
# Panel A
# ============================================================

def compute_panel_a(
    worms,
    mode,
    min_common_anchors,
    hard_k,
    random_k,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    rows = []

    # Use ordered animal pairs:
    # hard negative depends on target animal.
    for ai in range(len(worms)):
        for bi in range(len(worms)):

            if ai == bi:
                continue

            wa = worms[ai]
            wb = worms[bi]

            shared = common_identities(
                wa,
                wb,
            )

            if len(shared) < (
                min_common_anchors + 2
            ):
                continue

            for gt in shared:

                # SAME
                s = relation_similarity(
                    wa,
                    gt,
                    wb,
                    gt,
                    mode,
                    min_common_anchors,
                )

                if np.isfinite(s):
                    rows.append({
                        "source_worm": wa.name,
                        "target_worm": wb.name,
                        "query_id": gt,
                        "candidate_id": gt,
                        "group": "Same",
                        "similarity": s,
                    })

                # HARD
                hard = hard_negative_ids(
                    wb,
                    gt,
                    shared,
                    hard_k,
                )

                for cid in hard:
                    s = relation_similarity(
                        wa,
                        gt,
                        wb,
                        cid,
                        mode,
                        min_common_anchors,
                    )

                    if np.isfinite(s):
                        rows.append({
                            "source_worm": wa.name,
                            "target_worm": wb.name,
                            "query_id": gt,
                            "candidate_id": cid,
                            "group": "Hard",
                            "similarity": s,
                        })

                # RANDOM
                hard_set = set(hard)

                random_pool = [
                    x
                    for x in shared
                    if (
                        x != gt
                        and
                        x not in hard_set
                    )
                ]

                if random_pool:
                    k = min(
                        random_k,
                        len(random_pool),
                    )

                    chosen = rng.choice(
                        random_pool,
                        size=k,
                        replace=False,
                    )

                    for cid in chosen:
                        cid = str(cid)

                        s = relation_similarity(
                            wa,
                            gt,
                            wb,
                            cid,
                            mode,
                            min_common_anchors,
                        )

                        if np.isfinite(s):
                            rows.append({
                                "source_worm": wa.name,
                                "target_worm": wb.name,
                                "query_id": gt,
                                "candidate_id": cid,
                                "group": "Random",
                                "similarity": s,
                            })

    return pd.DataFrame(
        rows
    )


def panel_a_statistics(df):
    rows = []

    for g in (
        "Same",
        "Hard",
        "Random",
    ):
        vals = df.loc[
            df["group"] == g,
            "similarity"
        ].dropna().values

        lo, hi = percentile_ci(
            vals
        )

        rows.append({
            "group": g,
            "n": len(vals),
            "mean": np.mean(vals),
            "median": np.median(vals),
            "std": np.std(vals, ddof=1),
            "ci95_low": lo,
            "ci95_high": hi,
        })

    # Important test:
    # Same versus Hard
    same = df.loc[
        df.group == "Same",
        "similarity"
    ].values

    hard = df.loc[
        df.group == "Hard",
        "similarity"
    ].values

    rand = df.loc[
        df.group == "Random",
        "similarity"
    ].values

    print("\n[PANEL A]")

    for r in rows:
        print(
            f"{r['group']:8s} "
            f"N={r['n']:7d} "
            f"mean={r['mean']:.4f} "
            f"median={r['median']:.4f} "
            f"CI=[{r['ci95_low']:.4f},"
            f"{r['ci95_high']:.4f}]"
        )

    if len(same) and len(hard):
        u, p = mannwhitneyu(
            same,
            hard,
            alternative="greater",
        )

        print(
            f"Same > Hard: "
            f"Mann-Whitney U={u:.1f}, "
            f"p={p:.4e}"
        )

    if len(hard) and len(rand):
        u, p = mannwhitneyu(
            hard,
            rand,
            alternative="greater",
        )

        print(
            f"Hard > Random: "
            f"Mann-Whitney U={u:.1f}, "
            f"p={p:.4e}"
        )

    return pd.DataFrame(
        rows
    )


def plot_panel_a(
    df,
    out,
    title,
):
    order = [
        "Same",
        "Hard",
        "Random",
    ]

    vals = [
        df.loc[
            df.group == g,
            "similarity"
        ].dropna().values
        for g in order
    ]

    fig, ax = plt.subplots(
        figsize=(5.8, 5.0)
    )

    violin = ax.violinplot(
        vals,
        positions=[1, 2, 3],
        widths=0.75,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )

    for body in violin["bodies"]:
        body.set_alpha(0.35)

    ax.boxplot(
        vals,
        positions=[1, 2, 3],
        widths=0.20,
        showfliers=False,
    )

    ax.set_xticks(
        [1, 2, 3]
    )

    ax.set_xticklabels(
        order
    )

    ax.set_ylabel(
        "Cross-animal relation-profile similarity"
    )

    ax.set_title(
        title
    )

    ax.axhline(
        0,
        linewidth=0.8,
        alpha=0.4,
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Panel B
# ============================================================

def identity_frequency(worms):
    c = Counter()

    for w in worms:
        c.update(
            w.labels
        )

    return c


def choose_heatmap_identities(
    worms,
    top_n,
    min_appearances,
):
    freq = identity_frequency(
        worms
    )

    candidates = [
        (k, v)
        for k, v in freq.items()
        if v >= min_appearances
    ]

    # deterministic:
    # frequency descending, then identity lexicographic
    candidates.sort(
        key=lambda x: (
            -x[1],
            x[0],
        )
    )

    return [
        x[0]
        for x in candidates[:top_n]
    ]


def compute_panel_b(
    worms,
    identities,
    mode,
    min_common_anchors,
):
    n = len(
        identities
    )

    matrix = np.full(
        (n, n),
        np.nan,
        dtype=np.float64,
    )

    counts = np.zeros(
        (n, n),
        dtype=np.int64,
    )

    long_rows = []

    for i, ida in enumerate(
        identities
    ):
        for j, idb in enumerate(
            identities
        ):

            similarities = []

            for ai in range(
                len(worms)
            ):
                for bi in range(
                    len(worms)
                ):

                    if ai == bi:
                        continue

                    wa = worms[ai]
                    wb = worms[bi]

                    if (
                        ida not in wa.label_to_idx
                        or
                        idb not in wb.label_to_idx
                    ):
                        continue

                    s = relation_similarity(
                        wa,
                        ida,
                        wb,
                        idb,
                        mode,
                        min_common_anchors,
                    )

                    if np.isfinite(s):
                        similarities.append(
                            s
                        )

            if similarities:
                matrix[i, j] = np.mean(
                    similarities
                )

                counts[i, j] = len(
                    similarities
                )

                long_rows.append({
                    "query_identity": ida,
                    "candidate_identity": idb,
                    "mean_similarity": matrix[i, j],
                    "n_cross_animal_pairs": counts[i, j],
                })

    return (
        matrix,
        counts,
        pd.DataFrame(long_rows),
    )


def panel_b_statistics(
    matrix,
    identities,
):
    diag = []

    hardest_offdiag = []

    margins = []

    for i in range(
        len(identities)
    ):
        d = matrix[i, i]

        off = np.delete(
            matrix[i],
            i,
        )

        off = off[
            np.isfinite(off)
        ]

        if not np.isfinite(d):
            continue

        if len(off) == 0:
            continue

        h = np.max(
            off
        )

        diag.append(
            d
        )

        hardest_offdiag.append(
            h
        )

        margins.append(
            d - h
        )

    print("\n[PANEL B]")

    print(
        f"Mean diagonal similarity         = "
        f"{np.mean(diag):.4f}"
    )

    print(
        f"Mean strongest off-diagonal     = "
        f"{np.mean(hardest_offdiag):.4f}"
    )

    print(
        f"Mean identity diagonal margin   = "
        f"{np.mean(margins):.4f}"
    )

    print(
        f"Fraction positive margin        = "
        f"{np.mean(np.asarray(margins) > 0):.4f}"
    )

    return pd.DataFrame({
        "identity": [
            identities[i]
            for i in range(len(identities))
            if (
                np.isfinite(matrix[i, i])
                and
                np.any(
                    np.isfinite(
                        np.delete(
                            matrix[i],
                            i,
                        )
                    )
                )
            )
        ],
        "diagonal_similarity": diag,
        "strongest_wrong_similarity": hardest_offdiag,
        "diagonal_margin": margins,
    })


def plot_panel_b(
    matrix,
    identities,
    out,
    title,
):
    fig, ax = plt.subplots(
        figsize=(9.2, 8.0)
    )

    im = ax.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(
            len(identities)
        )
    )

    ax.set_yticks(
        np.arange(
            len(identities)
        )
    )

    ax.set_xticklabels(
        identities,
        rotation=90,
        fontsize=7,
    )

    ax.set_yticklabels(
        identities,
        fontsize=7,
    )

    ax.set_xlabel(
        "Candidate neuron identity"
    )

    ax.set_ylabel(
        "Query neuron identity"
    )

    ax.set_title(
        title
    )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Mean cross-animal relation similarity"
    )

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Per-identity relation stability for D
# ============================================================

def compute_identity_stability(
    worms,
    mode,
    min_appearances,
    min_common_anchors,
):
    freq = identity_frequency(
        worms
    )

    identities = sorted([
        x
        for x, n in freq.items()
        if n >= min_appearances
    ])

    rows = []

    for identity in identities:

        similarities = []

        for ai, bi in combinations(
            range(len(worms)),
            2,
        ):
            wa = worms[ai]
            wb = worms[bi]

            if (
                identity not in wa.label_to_idx
                or
                identity not in wb.label_to_idx
            ):
                continue

            s = relation_similarity(
                wa,
                identity,
                wb,
                identity,
                mode,
                min_common_anchors,
            )

            if np.isfinite(s):
                similarities.append(
                    s
                )

        if similarities:
            lo, hi = percentile_ci(
                similarities
            )

            rows.append({
                "identity": identity,
                "n_animal_pairs": len(similarities),
                "relation_stability": float(
                    np.mean(similarities)
                ),
                "median_relation_stability": float(
                    np.median(similarities)
                ),
                "stability_ci95_low": lo,
                "stability_ci95_high": hi,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Prediction loader for REAL MPRT accuracy
# ============================================================

DEFAULT_GT_FIELDS = [
    "gt_id",
    "true_id",
    "target_id",
    "gt_label",
    "true_label",
    "target_label",
    "cell_id",
    "label",
    "y_true",
]

DEFAULT_PRED_FIELDS = [
    "pred_id",
    "predicted_id",
    "prediction",
    "pred_label",
    "predicted_label",
    "top1_id",
    "top1_label",
    "y_pred",
]

DEFAULT_CORRECT_FIELDS = [
    "correct",
    "is_correct",
    "top1_correct",
]


def first_existing(
    columns,
    candidates,
):
    lower_to_actual = {
        str(x).lower(): x
        for x in columns
    }

    for c in candidates:
        if c.lower() in lower_to_actual:
            return lower_to_actual[
                c.lower()
            ]

    return None


def dataframe_from_json(
    path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return pd.DataFrame(
            obj
        )

    if not isinstance(obj, dict):
        raise RuntimeError(
            "Unsupported JSON format"
        )

    # common nested prediction lists
    for k in (
        "predictions",
        "queries",
        "results",
        "records",
        "samples",
        "items",
    ):
        if (
            k in obj
            and
            isinstance(obj[k], list)
        ):
            return pd.DataFrame(
                obj[k]
            )

    # dict of equal-length arrays
    try:
        return pd.DataFrame(
            obj
        )
    except Exception:
        raise RuntimeError(
            "JSON does not contain per-query records."
        )


def dataframe_from_npz(
    path,
):
    d = np.load(
        path,
        allow_pickle=True,
    )

    arrays = {}

    lengths = []

    for k in d.files:
        arr = np.asarray(
            d[k]
        )

        if arr.ndim == 1:
            arrays[k] = arr
            lengths.append(
                len(arr)
            )

    if not arrays:
        raise RuntimeError(
            "No 1D per-query arrays found in NPZ."
        )

    # use modal length
    target_len = Counter(
        lengths
    ).most_common(1)[0][0]

    arrays = {
        k: v
        for k, v in arrays.items()
        if len(v) == target_len
    }

    return pd.DataFrame(
        arrays
    )


def read_prediction_file(
    path,
):
    suffix = Path(path).suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(
            path
        )

    if suffix == ".tsv":
        return pd.read_csv(
            path,
            sep="\t",
        )

    if suffix == ".jsonl":
        return pd.read_json(
            path,
            lines=True,
        )

    if suffix == ".json":
        return dataframe_from_json(
            path
        )

    if suffix == ".npz":
        return dataframe_from_npz(
            path
        )

    raise RuntimeError(
        f"Unsupported format: {suffix}"
    )


def discover_prediction_files(
    pred_root,
):
    root = Path(
        pred_root
    )

    if root.is_file():
        return [
            str(root)
        ]

    patterns = [
        "*.csv",
        "*.tsv",
        "*.jsonl",
        "*.json",
        "*.npz",
    ]

    files = []

    for pat in patterns:
        files.extend(
            str(x)
            for x in root.rglob(pat)
        )

    # Strongly prefer filenames suggesting query predictions.
    positive_words = (
        "prediction",
        "predictions",
        "query",
        "queries",
        "test_pred",
        "test_result",
    )

    preferred = [
        x
        for x in files
        if any(
            w in Path(x).name.lower()
            for w in positive_words
        )
    ]

    if preferred:
        files = preferred

    # Exclude obvious aggregate metrics/checkpoints
    files = [
        x
        for x in files
        if not any(
            bad in Path(x).name.lower()
            for bad in (
                "metrics",
                "summary",
                "checkpoint",
                "config",
                "history",
            )
        )
    ]

    return sorted(
        files
    )


def normalize_prediction_dataframe(
    df,
    source_file,
    gt_field=None,
    pred_field=None,
):
    """
    Normalize either:

    1) Per-query prediction format:
         gt_id, pred_id

    or

    2) Unified candidate-score long format:
         query_uid, gt_label, candidate_label, score

       In long format, Top-1 prediction for each query is:
           argmax_candidate score

    Returns one row per query:
         gt_id, pred_id, correct, query_uid, source_file
    """

    if len(df) == 0:
        return None

    cols = list(df.columns)

    lower_map = {
        str(c).lower(): c
        for c in cols
    }

    # ========================================================
    # CASE 1:
    # Unified candidate-score format
    # ========================================================

    required_long = {
        "query_uid",
        "gt_label",
        "candidate_label",
        "score",
    }

    if required_long.issubset(
        set(lower_map.keys())
    ):
        q_col = lower_map["query_uid"]
        gt_col = lower_map["gt_label"]
        cand_col = lower_map["candidate_label"]
        score_col = lower_map["score"]

        tmp = df[
            [
                q_col,
                gt_col,
                cand_col,
                score_col,
            ]
        ].copy()

        tmp.columns = [
            "query_uid",
            "gt_id",
            "candidate_id",
            "score",
        ]

        tmp["gt_id"] = tmp["gt_id"].map(
            clean_label
        )

        tmp["candidate_id"] = tmp[
            "candidate_id"
        ].map(
            clean_label
        )

        tmp["score"] = pd.to_numeric(
            tmp["score"],
            errors="coerce",
        )

        tmp = tmp[
            tmp["query_uid"].notna()
            &
            tmp["gt_id"].notna()
            &
            tmp["candidate_id"].notna()
            &
            tmp["score"].notna()
        ].copy()

        if len(tmp) == 0:
            return None

        # ----------------------------------------------------
        # Audit:
        # each query should have exactly one GT label
        # ----------------------------------------------------

        gt_counts = (
            tmp.groupby("query_uid")["gt_id"]
            .nunique()
        )

        bad_queries = gt_counts[
            gt_counts != 1
        ]

        if len(bad_queries) > 0:
            raise RuntimeError(
                f"{source_file}: "
                f"{len(bad_queries)} queries have "
                f"multiple gt_label values."
            )

        # ----------------------------------------------------
        # Top-1 = candidate with maximum raw matching score
        # ----------------------------------------------------

        best_idx = (
            tmp.groupby("query_uid")["score"]
            .idxmax()
        )

        best = tmp.loc[
            best_idx,
            [
                "query_uid",
                "gt_id",
                "candidate_id",
                "score",
            ]
        ].copy()

        best = best.rename(
            columns={
                "candidate_id": "pred_id",
                "score": "top1_score",
            }
        )

        best["correct"] = (
            best["gt_id"]
            ==
            best["pred_id"]
        ).astype(int)

        best["source_file"] = str(
            source_file
        )

        return best.reset_index(
            drop=True
        )

    # ========================================================
    # CASE 2:
    # Already one-row-per-query prediction format
    # ========================================================

    gt = (
        gt_field
        if gt_field in cols
        else None
    )

    pred = (
        pred_field
        if pred_field in cols
        else None
    )

    if gt is None:
        gt = first_existing(
            cols,
            DEFAULT_GT_FIELDS,
        )

    if pred is None:
        pred = first_existing(
            cols,
            DEFAULT_PRED_FIELDS,
        )

    correct_field = first_existing(
        cols,
        DEFAULT_CORRECT_FIELDS,
    )

    if gt is None:
        return None

    if pred is None and correct_field is None:
        return None

    out = pd.DataFrame()

    out["gt_id"] = [
        clean_label(x)
        for x in df[gt].values
    ]

    if pred is not None:
        out["pred_id"] = [
            clean_label(x)
            for x in df[pred].values
        ]

        out["correct"] = (
            out["gt_id"]
            ==
            out["pred_id"]
        ).astype(int)

    else:
        out["correct"] = pd.to_numeric(
            df[correct_field],
            errors="coerce",
        ).fillna(0).astype(int)

        out["pred_id"] = None

    # Preserve query id if available
    q_field = first_existing(
        cols,
        [
            "query_uid",
            "query_id",
            "query_index",
            "uid",
        ],
    )

    if q_field is not None:
        out["query_uid"] = df[
            q_field
        ].astype(str).values
    else:
        out["query_uid"] = [
            f"row_{i}"
            for i in range(len(out))
        ]

    out["source_file"] = str(
        source_file
    )

    out = out[
        out["gt_id"].notna()
    ].copy()

    return out.reset_index(
        drop=True
    )


def load_mprt_predictions(
    pred_root,
    gt_field=None,
    pred_field=None,
):
    files = discover_prediction_files(
        pred_root
    )

    print(
        f"[PRED] candidate files = {len(files)}"
    )

    frames = []

    used = []

    for p in files:
        try:
            df = read_prediction_file(
                p
            )

            norm = normalize_prediction_dataframe(
                df,
                p,
                gt_field=gt_field,
                pred_field=pred_field,
            )

            if norm is None:
                continue

            if len(norm) == 0:
                continue

            frames.append(
                norm
            )

            used.append(
                p
            )

            print(
                f"[PRED] accepted "
                f"{p} "
                f"N={len(norm)}"
            )

        except Exception as e:
            # Lots of unrelated JSON files may exist in run directories.
            continue

    if not frames:
        raise RuntimeError(
            "\nCould not find per-query MPRT predictions.\n\n"
            "The prediction files must contain GT identity and either "
            "predicted identity or a correct flag.\n\n"
            "Examples:\n"
            "  gt_id,pred_id\n"
            "  true_id,predicted_id\n"
            "  cell_id,top1_id\n"
            "  gt_id,correct\n\n"
            "If your field names differ, pass:\n"
            "  --gt-field YOUR_GT_FIELD "
            "--pred-field YOUR_PRED_FIELD\n"
        )

    pred = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"[PRED] total accepted queries = "
        f"{len(pred)}"
    )

    print(
        f"[PRED] overall Top-1 = "
        f"{100 * pred.correct.mean():.2f}%"
    )

    return pred


def compute_per_identity_accuracy(
    pred_df,
    min_queries,
):
    rows = []

    for identity, sub in pred_df.groupby(
        "gt_id"
    ):
        n = len(
            sub
        )

        if n < min_queries:
            continue

        acc = float(
            sub.correct.mean()
        )

        rows.append({
            "identity": identity,
            "n_test_queries": n,
            "mprt_top1": acc,
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Panel D
# ============================================================

def compute_panel_d(
    stability_df,
    accuracy_df,
):
    merged = stability_df.merge(
        accuracy_df,
        on="identity",
        how="inner",
    )

    if len(merged) < 3:
        raise RuntimeError(
            f"Only {len(merged)} identities overlap between "
            "relation stability and MPRT predictions."
        )

    rho, p = spearmanr(
        merged["relation_stability"].values,
        merged["mprt_top1"].values,
    )

    pearson_r, pearson_p = pearsonr(
        merged["relation_stability"].values,
        merged["mprt_top1"].values,
    )

    print("\n[PANEL D]")

    print(
        f"Identities             = "
        f"{len(merged)}"
    )

    print(
        f"Spearman rho           = "
        f"{rho:.4f}"
    )

    print(
        f"Spearman p             = "
        f"{p:.4e}"
    )

    print(
        f"Pearson r              = "
        f"{pearson_r:.4f}"
    )

    print(
        f"Pearson p              = "
        f"{pearson_p:.4e}"
    )

    return (
        merged,
        rho,
        p,
        pearson_r,
        pearson_p,
    )


def plot_panel_d(
    df,
    rho,
    p,
    out,
    title,
    annotate=False,
):
    x = df[
        "relation_stability"
    ].values

    y = (
        100
        *
        df["mprt_top1"].values
    )

    fig, ax = plt.subplots(
        figsize=(5.8, 5.0)
    )

    ax.scatter(
        x,
        y,
        s=35,
        alpha=0.75,
    )

    # Simple least-squares visual trend.
    # Statistical claim remains Spearman.
    if len(x) >= 2:
        z = np.polyfit(
            x,
            y,
            1,
        )

        xx = np.linspace(
            np.min(x),
            np.max(x),
            100,
        )

        yy = (
            z[0] * xx
            +
            z[1]
        )

        ax.plot(
            xx,
            yy,
            linewidth=1.5,
        )

    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                row["identity"],
                (
                    row["relation_stability"],
                    100 * row["mprt_top1"],
                ),
                fontsize=6,
                alpha=0.75,
            )

    ax.set_xlabel(
        "Cross-animal population-relation stability"
    )

    ax.set_ylabel(
        "MPRT Top-1 accuracy per identity (%)"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.2
    )

    text = (
        f"Spearman $\\rho$ = {rho:.3f}\n"
        f"$p$ = {p:.2e}\n"
        f"$n$ = {len(df)} identities"
    )

    ax.text(
        0.04,
        0.96,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            alpha=0.85,
        ),
    )

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Optional combined 3-panel figure
# ============================================================

def plot_combined(
    df_a,
    matrix,
    identities,
    df_d,
    rho,
    p,
    out,
    dataset_name,
    mode,
):
    fig = plt.figure(
        figsize=(16.0, 4.8)
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[
            0.8,
            1.15,
            0.9,
        ],
    )

    # ----------------------
    # A
    # ----------------------
    ax = fig.add_subplot(
        gs[0, 0]
    )

    order = [
        "Same",
        "Hard",
        "Random",
    ]

    vals = [
        df_a.loc[
            df_a.group == g,
            "similarity"
        ].dropna().values
        for g in order
    ]

    vp = ax.violinplot(
        vals,
        positions=[1, 2, 3],
        widths=0.7,
        showextrema=False,
    )

    for b in vp["bodies"]:
        b.set_alpha(
            0.35
        )

    ax.boxplot(
        vals,
        positions=[1, 2, 3],
        widths=0.18,
        showfliers=False,
    )

    ax.set_xticks(
        [1, 2, 3]
    )

    ax.set_xticklabels(
        order
    )

    ax.set_ylabel(
        "Relation-profile similarity"
    )

    ax.set_title(
        "(A) Identity specificity"
    )

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    # ----------------------
    # B
    # ----------------------
    ax = fig.add_subplot(
        gs[0, 1]
    )

    im = ax.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )

    ax.set_xticks(
        np.arange(
            len(identities)
        )
    )

    ax.set_yticks(
        np.arange(
            len(identities)
        )
    )

    ax.set_xticklabels(
        identities,
        rotation=90,
        fontsize=6,
    )

    ax.set_yticklabels(
        identities,
        fontsize=6,
    )

    ax.set_xlabel(
        "Candidate identity"
    )

    ax.set_ylabel(
        "Query identity"
    )

    ax.set_title(
        "(B) Cross-identity relation similarity"
    )

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.03,
    )

    # ----------------------
    # D
    # ----------------------
    ax = fig.add_subplot(
        gs[0, 2]
    )

    x = df_d[
        "relation_stability"
    ].values

    y = (
        100
        *
        df_d["mprt_top1"].values
    )

    ax.scatter(
        x,
        y,
        s=30,
        alpha=0.75,
    )

    if len(x) >= 2:
        coef = np.polyfit(
            x,
            y,
            1,
        )

        xx = np.linspace(
            np.min(x),
            np.max(x),
            100,
        )

        ax.plot(
            xx,
            coef[0] * xx + coef[1],
            linewidth=1.4,
        )

    ax.set_xlabel(
        "Relation stability"
    )

    ax.set_ylabel(
        "MPRT per-identity Top-1 (%)"
    )

    ax.set_title(
        "(D) Stability predicts matching"
    )

    ax.grid(
        alpha=0.2
    )

    ax.text(
        0.05,
        0.95,
        (
            f"Spearman $\\rho$={rho:.3f}\n"
            f"$p$={p:.2e}"
        ),
        transform=ax.transAxes,
        va="top",
    )

    fig.suptitle(
        f"{dataset_name}: stable and identity-specific "
        f"population relations ({mode})",
        fontsize=14,
    )

    fig.tight_layout()

    fig.savefig(
        out,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-root",
        required=True,
    )

    p.add_argument(
        "--pred-root",
        required=True,
        help=(
            "MPRT test prediction file or directory "
            "containing per-query prediction files"
        ),
    )

    p.add_argument(
        "--save-dir",
        required=True,
    )

    p.add_argument(
        "--dataset-name",
        default="Dataset",
    )

    p.add_argument(
        "--splits",
        nargs="+",
        default=[
            "train",
            "val",
            "test",
        ],
    )

    p.add_argument(
        "--relation-mode",
        choices=[
            "geo",
            "act",
            "multi",
            "precomputed",
        ],
        default="geo",
    )

    p.add_argument(
        "--relation-key",
        default=None,
        help=(
            "NPZ key containing exact NxN model relation matrix "
            "when --relation-mode precomputed"
        ),
    )

    p.add_argument(
        "--min-common-anchors",
        type=int,
        default=8,
    )

    p.add_argument(
        "--hard-k",
        type=int,
        default=3,
    )

    p.add_argument(
        "--random-k",
        type=int,
        default=1,
    )

    p.add_argument(
        "--heatmap-top-n",
        type=int,
        default=25,
    )

    p.add_argument(
        "--min-appearances",
        type=int,
        default=5,
    )

    p.add_argument(
        "--min-test-queries",
        type=int,
        default=5,
        help=(
            "Minimum number of MPRT test queries for an identity "
            "to enter Panel D"
        ),
    )

    p.add_argument(
        "--gt-field",
        default=None,
    )

    p.add_argument(
        "--pred-field",
        default=None,
    )

    p.add_argument(
        "--annotate-d",
        action="store_true",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


def main():
    args = parse_args()

    mkdir(
        args.save_dir
    )

    print("=" * 100)
    print(
        "CROSS-ANIMAL POPULATION RELATION MECHANISM ANALYSIS"
    )
    print("=" * 100)

    print(
        f"Dataset       : {args.dataset_name}"
    )

    print(
        f"Relation mode : {args.relation_mode}"
    )

    print(
        f"Data root     : {args.data_root}"
    )

    print(
        f"Pred root     : {args.pred_root}"
    )

    print(
        f"Save dir      : {args.save_dir}"
    )

    # ========================================================
    # Load biological population
    # ========================================================

    worms = load_worms(
        args.data_root,
        args.splits,
        args.relation_mode,
        args.relation_key,
    )

    # ========================================================
    # PANEL A
    # ========================================================

    df_a = compute_panel_a(
        worms,
        mode=args.relation_mode,
        min_common_anchors=args.min_common_anchors,
        hard_k=args.hard_k,
        random_k=args.random_k,
        seed=args.seed,
    )

    if len(df_a) == 0:
        raise RuntimeError(
            "Panel A generated zero valid comparisons."
        )

    df_a.to_csv(
        Path(args.save_dir)
        /
        "panelA_same_hard_random_raw.csv",
        index=False,
    )

    summary_a = panel_a_statistics(
        df_a
    )

    summary_a.to_csv(
        Path(args.save_dir)
        /
        "panelA_same_hard_random_summary.csv",
        index=False,
    )

    plot_panel_a(
        df_a,
        str(
            Path(args.save_dir)
            /
            "panelA_same_hard_random.png"
        ),
        title=(
            f"{args.dataset_name}: "
            f"cross-animal relation specificity"
        ),
    )

    # ========================================================
    # PANEL B
    # ========================================================

    heatmap_ids = choose_heatmap_identities(
        worms,
        top_n=args.heatmap_top_n,
        min_appearances=args.min_appearances,
    )

    print(
        f"\n[PANEL B] identities selected = "
        f"{len(heatmap_ids)}"
    )

    matrix, counts, long_b = compute_panel_b(
        worms,
        heatmap_ids,
        mode=args.relation_mode,
        min_common_anchors=args.min_common_anchors,
    )

    np.save(
        Path(args.save_dir)
        /
        "panelB_similarity_matrix.npy",
        matrix,
    )

    np.save(
        Path(args.save_dir)
        /
        "panelB_count_matrix.npy",
        counts,
    )

    pd.DataFrame({
        "identity": heatmap_ids
    }).to_csv(
        Path(args.save_dir)
        /
        "panelB_identities.csv",
        index=False,
    )

    long_b.to_csv(
        Path(args.save_dir)
        /
        "panelB_similarity_long.csv",
        index=False,
    )

    stats_b = panel_b_statistics(
        matrix,
        heatmap_ids,
    )

    stats_b.to_csv(
        Path(args.save_dir)
        /
        "panelB_diagonal_statistics.csv",
        index=False,
    )

    plot_panel_b(
        matrix,
        heatmap_ids,
        str(
            Path(args.save_dir)
            /
            "panelB_identity_heatmap.png"
        ),
        title=(
            f"{args.dataset_name}: "
            f"cross-animal identity-specific relations"
        ),
    )

    # ========================================================
    # PANEL D: biological stability
    # ========================================================

    stability = compute_identity_stability(
        worms,
        mode=args.relation_mode,
        min_appearances=args.min_appearances,
        min_common_anchors=args.min_common_anchors,
    )

    stability.to_csv(
        Path(args.save_dir)
        /
        "panelD_identity_relation_stability.csv",
        index=False,
    )

    # ========================================================
    # PANEL D: REAL MPRT predictions
    # ========================================================

    predictions = load_mprt_predictions(
        args.pred_root,
        gt_field=args.gt_field,
        pred_field=args.pred_field,
    )

    predictions.to_csv(
        Path(args.save_dir)
        /
        "panelD_loaded_mprt_predictions.csv",
        index=False,
    )

    accuracy = compute_per_identity_accuracy(
        predictions,
        min_queries=args.min_test_queries,
    )

    accuracy.to_csv(
        Path(args.save_dir)
        /
        "panelD_mprt_per_identity_accuracy.csv",
        index=False,
    )

    (
        df_d,
        rho,
        p,
        pearson_r,
        pearson_p,
    ) = compute_panel_d(
        stability,
        accuracy,
    )

    df_d["spearman_rho_global"] = rho
    df_d["spearman_p_global"] = p
    df_d["pearson_r_global"] = pearson_r
    df_d["pearson_p_global"] = pearson_p

    df_d.to_csv(
        Path(args.save_dir)
        /
        "panelD_stability_vs_mprt_accuracy.csv",
        index=False,
    )

    plot_panel_d(
        df_d,
        rho,
        p,
        str(
            Path(args.save_dir)
            /
            "panelD_stability_vs_mprt_accuracy.png"
        ),
        title=(
            f"{args.dataset_name}: relation stability "
            f"vs MPRT accuracy"
        ),
        annotate=args.annotate_d,
    )

    # ========================================================
    # Combined A+B+D
    # ========================================================

    plot_combined(
        df_a,
        matrix,
        heatmap_ids,
        df_d,
        rho,
        p,
        str(
            Path(args.save_dir)
            /
            "figure_population_relation_mechanism_ABD.png"
        ),
        dataset_name=args.dataset_name,
        mode=args.relation_mode,
    )

    print("\n" + "=" * 100)
    print("FINISHED")
    print("=" * 100)

    print(
        f"Panel A : "
        f"{Path(args.save_dir) / 'panelA_same_hard_random.png'}"
    )

    print(
        f"Panel B : "
        f"{Path(args.save_dir) / 'panelB_identity_heatmap.png'}"
    )

    print(
        f"Panel D : "
        f"{Path(args.save_dir) / 'panelD_stability_vs_mprt_accuracy.png'}"
    )

    print(
        f"Combined: "
        f"{Path(args.save_dir) / 'figure_population_relation_mechanism_ABD.png'}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()