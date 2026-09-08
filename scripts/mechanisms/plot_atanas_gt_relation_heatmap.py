#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Atanas GT-ordered population relation heatmaps.

This script draws GT identity x GT identity heatmaps:
    Geometry | Activity | Multimodal

Each cell (i, j) is the mean relation value across all animals
that contain both GT identities i and j.

Compared with animal x animal similarity heatmaps, this figure
has semantically meaningful axes (GT neuron identities).

Outputs:
    - gt_relation_heatmaps_1x3.png
    - gt_relation_heatmaps_1x3.pdf
    - geo_mean_relation.csv
    - act_mean_relation.csv
    - multi_mean_relation.csv
    - pair_support_counts.csv
    - selected_identities.csv
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
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
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

BAD_LABELS = {
    "", "nan", "none", "null", "unknown", "unlabeled", "?", "-1"
}


def clean_label(x):
    if x is None:
        return None

    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="ignore")

    if isinstance(x, np.generic):
        x = x.item()

    x = str(x).strip()

    if x.lower() in BAD_LABELS:
        return None

    return x


def normalize_xyz(xyz):
    xyz = np.asarray(xyz, dtype=np.float64)
    xyz = xyz - np.mean(xyz, axis=0, keepdims=True)

    rms = np.sqrt(np.mean(np.sum(xyz ** 2, axis=1)))
    if rms < 1e-12:
        rms = 1.0

    return xyz / rms


def zscore_matrix_upper(M):
    """
    Z-score only the upper-triangle entries of a symmetric matrix,
    then restore to a symmetric matrix. Diagonal set to 0.
    """
    M = np.asarray(M, dtype=np.float64)
    n = M.shape[0]
    iu = np.triu_indices(n, k=1)

    v = M[iu]
    mu = np.mean(v)
    sd = np.std(v)

    if sd < 1e-12:
        vz = np.zeros_like(v)
    else:
        vz = (v - mu) / sd

    Z = np.zeros_like(M)
    Z[iu] = vz
    Z[(iu[1], iu[0])] = vz
    np.fill_diagonal(Z, 0.0)
    return Z


# ============================================================
# Worm object
# ============================================================

class Worm:
    def __init__(self, name, labels, label_to_idx, geo, act, multi):
        self.name = name
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.geo_relation = geo
        self.act_relation = act
        self.multi_relation = multi


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

    files = sorted(set(files))
    return files


def load_one_worm(path):
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

    # Geometry relation: negative Euclidean distance
    delta = xyz_norm[:, None, :] - xyz_norm[None, :, :]
    geo = -np.sqrt(np.sum(delta ** 2, axis=-1))

    # Activity relation: correlation
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

    if act is None:
        # If activity missing, skip this worm for this figure
        return None

    # Multimodal relation:
    # z-score geo and act separately within a worm, then average
    geo_z = zscore_matrix_upper(geo)
    act_z = zscore_matrix_upper(act)
    multi = 0.5 * (geo_z + act_z)

    label_to_idx = {lb: i for i, lb in enumerate(labels)}

    return Worm(
        name=Path(path).stem,
        labels=labels,
        label_to_idx=label_to_idx,
        geo=geo,
        act=act,
        multi=multi,
    )


def load_worms(data_root, splits=None):
    paths = discover_npz(data_root, splits=splits)
    print(f"[DATA] discovered {len(paths)} candidate NPZ files")

    worms = []
    for p in paths:
        try:
            w = load_one_worm(p)
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
# Identity selection
# ============================================================

def select_identities(worms, min_appearances=8, top_n=None):
    counts = {}

    for w in worms:
        for lb in w.labels:
            counts[lb] = counts.get(lb, 0) + 1

    df = pd.DataFrame({
        "identity": list(counts.keys()),
        "n_animals": list(counts.values()),
    })

    df = df.sort_values(
        ["n_animals", "identity"],
        ascending=[False, True]
    ).reset_index(drop=True)

    df = df[df["n_animals"] >= min_appearances].copy()

    if top_n is not None:
        df = df.head(top_n).copy()

    identities = df["identity"].tolist()

    print(
        f"[GT] selected identities = {len(identities)} "
        f"(min appearances = {min_appearances})"
    )

    return identities, df


# ============================================================
# Aggregate relation matrices across animals
# ============================================================

def aggregate_mean_relation(worms, identities):
    n = len(identities)
    idx_of = {lb: i for i, lb in enumerate(identities)}

    geo_sum = np.zeros((n, n), dtype=np.float64)
    act_sum = np.zeros((n, n), dtype=np.float64)
    multi_sum = np.zeros((n, n), dtype=np.float64)
    count = np.zeros((n, n), dtype=np.int64)

    for w in worms:
        present = [lb for lb in identities if lb in w.label_to_idx]

        if len(present) < 2:
            continue

        local_idx = np.array([w.label_to_idx[lb] for lb in present], dtype=int)
        global_idx = np.array([idx_of[lb] for lb in present], dtype=int)

        G = w.geo_relation[np.ix_(local_idx, local_idx)]
        A = w.act_relation[np.ix_(local_idx, local_idx)]
        M = w.multi_relation[np.ix_(local_idx, local_idx)]

        for ii, gi in enumerate(global_idx):
            for jj, gj in enumerate(global_idx):
                geo_sum[gi, gj] += G[ii, jj]
                act_sum[gi, gj] += A[ii, jj]
                multi_sum[gi, gj] += M[ii, jj]
                count[gi, gj] += 1

    geo_mean = np.divide(
        geo_sum, count,
        out=np.full_like(geo_sum, np.nan),
        where=(count > 0)
    )
    act_mean = np.divide(
        act_sum, count,
        out=np.full_like(act_sum, np.nan),
        where=(count > 0)
    )
    multi_mean = np.divide(
        multi_sum, count,
        out=np.full_like(multi_sum, np.nan),
        where=(count > 0)
    )

    return geo_mean, act_mean, multi_mean, count


# ============================================================
# Plot
# ============================================================

def plot_heatmaps(
    identities,
    geo_mean,
    act_mean,
    multi_mean,
    pair_count,
    out_png,
    out_pdf,
):
    fig, axes = plt.subplots(
        1, 3,
        figsize=(15.0, 4.8),
        constrained_layout=False,
    )

    matrices = [geo_mean, act_mean, multi_mean]
    titles = ["Geometry", "Activity", "Multimodal"]
    letters = ["A", "B", "C"]

    # Use independent color scales because modalities have different raw ranges
    cmaps = ["viridis", "viridis", "viridis"]

    for ax, M, title, letter, cmap in zip(axes, matrices, titles, letters, cmaps):
        # mask entries with insufficient support
        masked = M.copy()

        im = ax.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
        )

        ax.set_title(title, pad=10, fontweight="medium")
        ax.set_xticks(np.arange(len(identities)))
        ax.set_yticks(np.arange(len(identities)))
        ax.set_xticklabels(identities, rotation=90)
        ax.set_yticklabels(identities)

        ax.set_xlabel("GT identity")
        if ax is axes[0]:
            ax.set_ylabel("GT identity")

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
            im,
            ax=ax,
            fraction=0.046,
            pad=0.02,
        )
        if title == "Geometry":
            cbar.set_label("Mean relation value")
        elif title == "Activity":
            cbar.set_label("Mean correlation")
        else:
            cbar.set_label("Mean multimodal relation")

    fig.suptitle(
        "Atanas: GT-ordered population relation structure",
        fontsize=12.5,
        y=0.98,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.24,
        top=0.84,
        wspace=0.23,
    )

    fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--data-root",
        default="Data/Atanas_SF_unified_000776/date_disjoint_v1/full",
    )
    p.add_argument(
        "--save-dir",
        default="runs/mprt_v1_1/mechanism_population_relation/atanas_gt_relation_heatmap",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    p.add_argument(
        "--min-appearances",
        type=int,
        default=8,
        help="Minimum number of animals in which a GT identity must appear",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Maximum number of GT identities to display",
    )

    return p.parse_args()


def main():
    args = parse_args()
    setup_style()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("ATANAS GT-ORDERED POPULATION RELATION HEATMAP")
    print("=" * 100)
    print(f"Data root       : {args.data_root}")
    print(f"Save dir        : {args.save_dir}")
    print(f"Min appearances : {args.min_appearances}")
    print(f"Top N identities: {args.top_n}")

    worms = load_worms(args.data_root, splits=args.splits)

    identities, id_df = select_identities(
        worms=worms,
        min_appearances=args.min_appearances,
        top_n=args.top_n,
    )

    geo_mean, act_mean, multi_mean, pair_count = aggregate_mean_relation(
        worms=worms,
        identities=identities,
    )

    # Save outputs
    id_df.to_csv(save_dir / "selected_identities.csv", index=False)

    pd.DataFrame(geo_mean, index=identities, columns=identities).to_csv(
        save_dir / "geo_mean_relation.csv"
    )
    pd.DataFrame(act_mean, index=identities, columns=identities).to_csv(
        save_dir / "act_mean_relation.csv"
    )
    pd.DataFrame(multi_mean, index=identities, columns=identities).to_csv(
        save_dir / "multi_mean_relation.csv"
    )
    pd.DataFrame(pair_count, index=identities, columns=identities).to_csv(
        save_dir / "pair_support_counts.csv"
    )

    out_png = save_dir / "gt_relation_heatmaps_1x3.png"
    out_pdf = save_dir / "gt_relation_heatmaps_1x3.pdf"

    plot_heatmaps(
        identities=identities,
        geo_mean=geo_mean,
        act_mean=act_mean,
        multi_mean=multi_mean,
        pair_count=pair_count,
        out_png=out_png,
        out_pdf=out_pdf,
    )

    print("\n" + "=" * 100)
    print("SAVED")
    print("=" * 100)
    print(f"PNG : {out_png}")
    print(f"PDF : {out_pdf}")
    print(f"CSV : {save_dir / 'selected_identities.csv'}")


if __name__ == "__main__":
    main()