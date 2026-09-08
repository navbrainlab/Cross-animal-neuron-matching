#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overall population-relation similarity heatmap across animals.

Outputs:
    1 x 3 heatmap:
        Geometry | Activity | Multimodal

Each cell (a,b):
    overall similarity between animal a and animal b
    computed on the shared labeled identities only.

Important:
    Use a dataset root with UNIQUE animals, e.g.
    - Atanas: Data/Atanas_SF_unified_000776/date_disjoint_v1/full
    - RLD   : Data/Dunn_001623/date_disjoint_full95_v1

Do NOT use cv5_grouped_v1 directly for this plot, because the same
biological animal appears in multiple folds and would be duplicated.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Style
# ============================================================

def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 400,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ============================================================
# Helpers
# ============================================================

def clean_label(x):
    if x is None:
        return None

    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")

    if isinstance(x, np.generic):
        x = x.item()

    x = str(x).strip()

    bad = {
        "", "nan", "none", "null", "unknown", "unlabeled", "?", "-1"
    }

    if x.lower() in bad:
        return None

    return x


def normalize_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz - np.mean(xyz, axis=0, keepdims=True)

    rms = np.sqrt(np.mean(np.sum(xyz ** 2, axis=1)))
    if rms < 1e-12:
        rms = 1.0

    return xyz / rms


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan

    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def zscore_vec(v):
    v = np.asarray(v, dtype=np.float64)
    mu = np.mean(v)
    sd = np.std(v)
    if sd < 1e-12:
        return np.zeros_like(v)
    return (v - mu) / sd


# ============================================================
# Worm object
# ============================================================

class Worm:
    def __init__(
        self,
        name,
        path,
        labels,
        label_to_idx,
        geo_relation,
        act_relation,
    ):
        self.name = name
        self.path = path
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.geo_relation = geo_relation
        self.act_relation = act_relation


# ============================================================
# Data loading
# ============================================================

def discover_npz(data_root, splits=None):
    root = Path(data_root)

    files = []

    if splits:
        for split in splits:
            p = root / split
            if p.exists():
                files.extend(sorted(str(x) for x in p.rglob("*.npz")))

    if not files:
        files = sorted(str(x) for x in root.rglob("*.npz"))

    # remove obvious non-sample files if any
    bad_tokens = (
        "prediction",
        "metric",
        "checkpoint",
        "embedding",
        "cache",
    )

    keep = []
    for p in files:
        low = p.lower()
        if any(tok in low for tok in bad_tokens):
            continue
        keep.append(p)

    # deduplicate absolute paths
    keep = sorted(set(keep))
    return keep


def load_one_worm(path, require_activity=False):
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

    keep_idx = []
    labels = []
    seen = set()

    for i in range(n):
        if not mask[i]:
            continue

        lb = clean_label(labels_raw[i])
        if lb is None:
            continue

        if lb in seen:
            continue

        seen.add(lb)
        keep_idx.append(i)
        labels.append(lb)

    if len(keep_idx) < 5:
        return None

    keep_idx = np.asarray(keep_idx, dtype=int)

    xyz_kept = xyz[keep_idx].astype(np.float64)
    xyz_norm = normalize_xyz(xyz_kept)

    # Geometry relation = negative pairwise distance
    delta = xyz_norm[:, None, :] - xyz_norm[None, :, :]
    geo = -np.sqrt(np.sum(delta ** 2, axis=-1))

    act = None
    if "activity_raw" in d:
        activity = np.asarray(d["activity_raw"])[keep_idx].astype(np.float64)
        if activity.ndim == 2 and activity.shape[1] >= 3:
            mu = activity.mean(axis=1, keepdims=True)
            sd = activity.std(axis=1, keepdims=True)
            sd[sd < 1e-12] = 1.0
            activity = (activity - mu) / sd
            act = np.corrcoef(activity)
            act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

    if require_activity and act is None:
        return None

    label_to_idx = {lb: i for i, lb in enumerate(labels)}

    return Worm(
        name=Path(path).stem,
        path=path,
        labels=labels,
        label_to_idx=label_to_idx,
        geo_relation=geo,
        act_relation=act,
    )


def load_worms(data_root, splits=None):
    paths = discover_npz(data_root, splits=splits)
    print(f"[DATA] discovered {len(paths)} candidate NPZ files")

    worms = []
    for p in paths:
        try:
            w = load_one_worm(p, require_activity=False)
            if w is not None:
                worms.append(w)
        except Exception as e:
            print(f"[WARN] failed loading {p}: {e}")

    print(f"[DATA] usable worms = {len(worms)}")

    if len(worms) < 2:
        raise RuntimeError("Need at least two usable animals.")

    counts = [len(w.labels) for w in worms]
    print(
        f"[DATA] labeled neurons per worm: "
        f"min={min(counts)} median={np.median(counts):.1f} max={max(counts)}"
    )

    return worms


# ============================================================
# Pairwise overall similarity
# ============================================================

def shared_identities(wa, wb):
    return sorted(set(wa.labels) & set(wb.labels))


def upper_triangle_vector(R):
    iu = np.triu_indices(R.shape[0], k=1)
    return R[iu]


def overall_similarity_geo(wa, wb, min_shared_ids):
    shared = shared_identities(wa, wb)
    if len(shared) < min_shared_ids:
        return np.nan, len(shared)

    ia = np.array([wa.label_to_idx[x] for x in shared], dtype=int)
    ib = np.array([wb.label_to_idx[x] for x in shared], dtype=int)

    Ra = wa.geo_relation[np.ix_(ia, ia)]
    Rb = wb.geo_relation[np.ix_(ib, ib)]

    va = upper_triangle_vector(Ra)
    vb = upper_triangle_vector(Rb)

    return safe_corr(va, vb), len(shared)


def overall_similarity_act(wa, wb, min_shared_ids):
    shared = shared_identities(wa, wb)
    if len(shared) < min_shared_ids:
        return np.nan, len(shared)

    if wa.act_relation is None or wb.act_relation is None:
        return np.nan, len(shared)

    ia = np.array([wa.label_to_idx[x] for x in shared], dtype=int)
    ib = np.array([wb.label_to_idx[x] for x in shared], dtype=int)

    Ra = wa.act_relation[np.ix_(ia, ia)]
    Rb = wb.act_relation[np.ix_(ib, ib)]

    va = upper_triangle_vector(Ra)
    vb = upper_triangle_vector(Rb)

    return safe_corr(va, vb), len(shared)


def overall_similarity_multi(wa, wb, min_shared_ids):
    shared = shared_identities(wa, wb)
    if len(shared) < min_shared_ids:
        return np.nan, len(shared)

    if wa.act_relation is None or wb.act_relation is None:
        return np.nan, len(shared)

    ia = np.array([wa.label_to_idx[x] for x in shared], dtype=int)
    ib = np.array([wb.label_to_idx[x] for x in shared], dtype=int)

    Rga = wa.geo_relation[np.ix_(ia, ia)]
    Rgb = wb.geo_relation[np.ix_(ib, ib)]
    Rag = upper_triangle_vector(Rga)
    Rbg = upper_triangle_vector(Rgb)

    Raa = wa.act_relation[np.ix_(ia, ia)]
    Rab = wb.act_relation[np.ix_(ib, ib)]
    Ract_a = upper_triangle_vector(Raa)
    Ract_b = upper_triangle_vector(Rab)

    va = np.concatenate([zscore_vec(Rag), zscore_vec(Ract_a)])
    vb = np.concatenate([zscore_vec(Rbg), zscore_vec(Ract_b)])

    return safe_corr(va, vb), len(shared)


def compute_all_matrices(worms, min_shared_ids):
    n = len(worms)

    geo = np.full((n, n), np.nan, dtype=np.float64)
    act = np.full((n, n), np.nan, dtype=np.float64)
    multi = np.full((n, n), np.nan, dtype=np.float64)
    shared_counts = np.zeros((n, n), dtype=np.int64)

    rows = []

    for i in range(n):
        for j in range(n):
            if i == j:
                geo[i, j] = np.nan
                act[i, j] = np.nan
                multi[i, j] = np.nan
                shared_counts[i, j] = len(worms[i].labels)
                continue

            g, k = overall_similarity_geo(worms[i], worms[j], min_shared_ids)
            a, _ = overall_similarity_act(worms[i], worms[j], min_shared_ids)
            m, _ = overall_similarity_multi(worms[i], worms[j], min_shared_ids)

            geo[i, j] = g
            act[i, j] = a
            multi[i, j] = m
            shared_counts[i, j] = k

            rows.append({
                "animal_a": worms[i].name,
                "animal_b": worms[j].name,
                "n_shared_ids": k,
                "geo_similarity": g,
                "act_similarity": a,
                "multi_similarity": m,
            })

    long_df = pd.DataFrame(rows)
    return geo, act, multi, shared_counts, long_df


# ============================================================
# Optional sorting
# ============================================================

def sort_by_average_similarity(worms, matrix):
    avg = np.nanmean(matrix, axis=1)
    order = np.argsort(avg)[::-1]
    worms_sorted = [worms[i] for i in order]
    return worms_sorted, order


# ============================================================
# Plot
# ============================================================

def plot_three_heatmaps(
    worms,
    geo,
    act,
    multi,
    out_png,
    out_pdf,
    dataset_name,
):
    names = [w.name for w in worms]

    fig, axes = plt.subplots(
        1, 3,
        figsize=(14.5, 4.8),
        constrained_layout=False,
    )

    matrices = [geo, act, multi]
    titles = ["Geometry", "Activity", "Multimodal"]
    letters = ["A", "B", "C"]

    # shared color scale
    all_vals = np.concatenate([
        geo[np.isfinite(geo)],
        act[np.isfinite(act)],
        multi[np.isfinite(multi)],
    ])

    if len(all_vals) == 0:
        raise RuntimeError("All similarity matrices are empty.")

    vmin = max(-1.0, float(np.nanmin(all_vals)))
    vmax = min(1.0, float(np.nanmax(all_vals)))

    ims = []

    for ax, M, title, letter in zip(axes, matrices, titles, letters):
        im = ax.imshow(
            M,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        ims.append(im)

        ax.set_title(title, pad=10, fontweight="medium")
        ax.set_xticks(np.arange(len(names)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticklabels(names)

        ax.set_xlabel("Animal")
        if ax is axes[0]:
            ax.set_ylabel("Animal")

        ax.text(
            -0.12, 1.03, letter,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="bottom",
            ha="left",
            clip_on=False,
        )

    cbar = fig.colorbar(
        ims[-1],
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Overall population-relation similarity")

    fig.suptitle(
        f"{dataset_name}: overall population-relation similarity across animals",
        fontsize=12.5,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.96,
        bottom=0.22,
        top=0.86,
        wspace=0.20,
    )

    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# Summary stats
# ============================================================

def print_summary(name, M):
    vals = M[np.isfinite(M)]
    print(
        f"[{name}] "
        f"animal-pairs={len(vals)} "
        f"mean={np.mean(vals):.4f} "
        f"median={np.median(vals):.4f} "
        f"std={np.std(vals, ddof=1):.4f} "
        f"min={np.min(vals):.4f} "
        f"max={np.max(vals):.4f}"
    )


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data-root", required=True)
    p.add_argument("--dataset-name", default="Dataset")
    p.add_argument("--save-dir", required=True)

    p.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Optional split directories, e.g. train val test"
    )

    p.add_argument(
        "--min-shared-ids",
        type=int,
        default=8,
        help="Minimum number of shared identities required to compare two animals"
    )

    p.add_argument(
        "--sort-by",
        choices=["none", "geo", "act", "multi"],
        default="none",
        help="Optional row/column reordering by average similarity"
    )

    return p.parse_args()


def main():
    args = parse_args()
    setup_style()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("OVERALL POPULATION-RELATION SIMILARITY HEATMAP")
    print("=" * 100)
    print(f"Dataset       : {args.dataset-name if hasattr(args, 'dataset-name') else args.dataset_name}")
    print(f"Data root     : {args.data_root}")
    print(f"Save dir      : {args.save_dir}")
    print(f"Min shared IDs: {args.min_shared_ids}")
    print(f"Sort by       : {args.sort_by}")

    worms = load_worms(args.data_root, splits=args.splits)

    geo, act, multi, shared_counts, long_df = compute_all_matrices(
        worms=worms,
        min_shared_ids=args.min_shared_ids,
    )

    # optional sorting
    order = np.arange(len(worms))

    if args.sort_by != "none":
        ref = {"geo": geo, "act": act, "multi": multi}[args.sort_by]
        worms, order = sort_by_average_similarity(worms, ref)

        geo = geo[np.ix_(order, order)]
        act = act[np.ix_(order, order)]
        multi = multi[np.ix_(order, order)]
        shared_counts = shared_counts[np.ix_(order, order)]

    names = [w.name for w in worms]

    # save raw matrices
    np.save(save_dir / "geo_overall_similarity.npy", geo)
    np.save(save_dir / "act_overall_similarity.npy", act)
    np.save(save_dir / "multi_overall_similarity.npy", multi)
    np.save(save_dir / "shared_identity_counts.npy", shared_counts)

    pd.DataFrame({"animal": names}).to_csv(
        save_dir / "animal_order.csv", index=False
    )

    long_df.to_csv(
        save_dir / "overall_population_relation_similarity_long.csv",
        index=False
    )

    pd.DataFrame(geo, index=names, columns=names).to_csv(
        save_dir / "geo_overall_similarity.csv"
    )
    pd.DataFrame(act, index=names, columns=names).to_csv(
        save_dir / "act_overall_similarity.csv"
    )
    pd.DataFrame(multi, index=names, columns=names).to_csv(
        save_dir / "multi_overall_similarity.csv"
    )
    pd.DataFrame(shared_counts, index=names, columns=names).to_csv(
        save_dir / "shared_identity_counts.csv"
    )

    print_summary("Geometry", geo)
    print_summary("Activity", act)
    print_summary("Multimodal", multi)

    out_png = save_dir / "overall_population_relation_similarity_heatmaps.png"
    out_pdf = save_dir / "overall_population_relation_similarity_heatmaps.pdf"

    plot_three_heatmaps(
        worms=worms,
        geo=geo,
        act=act,
        multi=multi,
        out_png=out_png,
        out_pdf=out_pdf,
        dataset_name=args.dataset_name,
    )

    print("\n" + "=" * 100)
    print("SAVED")
    print("=" * 100)
    print(f"PNG : {out_png}")
    print(f"PDF : {out_pdf}")
    print(f"CSV : {save_dir / 'overall_population_relation_similarity_long.csv'}")


if __name__ == "__main__":
    main()