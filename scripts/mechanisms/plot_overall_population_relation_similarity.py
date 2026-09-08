#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overall cross-animal population-relation similarity
===================================================

Each observation = one animal pair.

For each pair of animals (a, b):
    1) find shared GT identities C_ab
    2) build relation matrices on those shared identities
    3) vectorize upper triangle
    4) compute Pearson correlation between the two vectors

Outputs:
    - overall_population_relation_similarity.png
    - overall_population_relation_similarity.pdf
    - overall_population_relation_similarity_long.csv
    - overall_population_relation_similarity_summary.csv

Optional:
    --with-shuffle
        also compute identity-shuffled control for each animal pair
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

try:
    from scipy.stats import wilcoxon
    HAVE_WILCOXON = True
except Exception:
    HAVE_WILCOXON = False


# ============================================================
# Style
# ============================================================

def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 600,
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


def upper_triangle_vector(M):
    iu = np.triu_indices(M.shape[0], k=1)
    return M[iu]


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


def zscore_upper_to_symmetric(M):
    """
    Z-score only the upper triangle of a symmetric matrix,
    then restore symmetry. Diagonal is set to 0.
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
# Data object
# ============================================================

class Worm:
    def __init__(self, name, labels, label_to_idx, geo_relation, act_relation, multi_relation):
        self.name = name
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.geo_relation = geo_relation
        self.act_relation = act_relation
        self.multi_relation = multi_relation


# ============================================================
# Data loading
# ============================================================

def discover_npz(data_root, splits=None):
    root = Path(data_root)
    files = []

    if splits is not None and len(splits) > 0:
        for split in splits:
            p = root / split
            if p.exists():
                files.extend(sorted(str(x) for x in p.rglob("*.npz")))

    if not files:
        files = sorted(str(x) for x in root.rglob("*.npz"))

    return sorted(set(files))


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

    # -------- geometry relation --------
    xyz_kept = xyz[keep_idx].astype(np.float64)
    xyz_norm = normalize_xyz(xyz_kept)

    delta = xyz_norm[:, None, :] - xyz_norm[None, :, :]
    geo = -np.sqrt(np.sum(delta ** 2, axis=-1))  # negative distance

    # -------- activity relation --------
    if "activity_raw" not in d:
        return None

    activity = np.asarray(d["activity_raw"])[keep_idx].astype(np.float64)
    if activity.ndim != 2 or activity.shape[1] < 3:
        return None

    mu = activity.mean(axis=1, keepdims=True)
    sd = activity.std(axis=1, keepdims=True)
    sd[sd < 1e-12] = 1.0
    activity = (activity - mu) / sd

    act = np.corrcoef(activity)
    act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)

    # -------- multimodal relation --------
    # Define multimodal relation as within-worm average of
    # z-scored geometry relation and z-scored activity relation.
    geo_z = zscore_upper_to_symmetric(geo)
    act_z = zscore_upper_to_symmetric(act)
    multi = 0.5 * (geo_z + act_z)

    label_to_idx = {lb: i for i, lb in enumerate(labels)}

    return Worm(
        name=Path(path).stem,
        labels=labels,
        label_to_idx=label_to_idx,
        geo_relation=geo,
        act_relation=act,
        multi_relation=multi,
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
# Pairwise overall similarity
# ============================================================

def shared_identities(wa, wb):
    return sorted(set(wa.labels) & set(wb.labels))


def relation_similarity_for_shared(Ra, Rb, ia, ib):
    Sa = Ra[np.ix_(ia, ia)]
    Sb = Rb[np.ix_(ib, ib)]

    va = upper_triangle_vector(Sa)
    vb = upper_triangle_vector(Sb)

    return safe_corr(va, vb)


def shuffled_similarity_for_shared(Ra, Rb, ia, ib, rng, repeats):
    Sa = Ra[np.ix_(ia, ia)]
    Sb = Rb[np.ix_(ib, ib)]

    va = upper_triangle_vector(Sa)

    vals = []
    k = len(ia)

    for _ in range(repeats):
        perm = rng.permutation(k)
        Sb_perm = Sb[np.ix_(perm, perm)]
        vb = upper_triangle_vector(Sb_perm)
        vals.append(safe_corr(va, vb))

    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return np.nan

    return float(np.mean(vals))


def compute_pairwise_overall_similarity(
    worms,
    min_shared_ids=8,
    with_shuffle=False,
    shuffle_repeats=50,
    seed=42,
):
    rng = np.random.default_rng(seed)

    rows = []

    n = len(worms)

    for i in range(n):
        for j in range(i + 1, n):
            wa = worms[i]
            wb = worms[j]

            shared = shared_identities(wa, wb)
            k = len(shared)

            if k < min_shared_ids:
                continue

            ia = np.array([wa.label_to_idx[x] for x in shared], dtype=int)
            ib = np.array([wb.label_to_idx[x] for x in shared], dtype=int)

            aligned = {
                "Geometry": relation_similarity_for_shared(
                    wa.geo_relation, wb.geo_relation, ia, ib
                ),
                "Activity": relation_similarity_for_shared(
                    wa.act_relation, wb.act_relation, ia, ib
                ),
                "Multimodal": relation_similarity_for_shared(
                    wa.multi_relation, wb.multi_relation, ia, ib
                ),
            }

            shuffled = {}
            if with_shuffle:
                shuffled = {
                    "Geometry": shuffled_similarity_for_shared(
                        wa.geo_relation, wb.geo_relation, ia, ib, rng, shuffle_repeats
                    ),
                    "Activity": shuffled_similarity_for_shared(
                        wa.act_relation, wb.act_relation, ia, ib, rng, shuffle_repeats
                    ),
                    "Multimodal": shuffled_similarity_for_shared(
                        wa.multi_relation, wb.multi_relation, ia, ib, rng, shuffle_repeats
                    ),
                }

            for modality in ["Geometry", "Activity", "Multimodal"]:
                rows.append({
                    "animal_a": wa.name,
                    "animal_b": wb.name,
                    "n_shared_ids": k,
                    "modality": modality,
                    "condition": "Aligned",
                    "similarity": aligned[modality],
                })

                if with_shuffle:
                    rows.append({
                        "animal_a": wa.name,
                        "animal_b": wb.name,
                        "n_shared_ids": k,
                        "modality": modality,
                        "condition": "Shuffled",
                        "similarity": shuffled[modality],
                    })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "No valid animal pairs found. "
            "Try lowering --min-shared-ids, but do not set it too low."
        )

    return df


# ============================================================
# Summary
# ============================================================

def summarize(df):
    out = (
        df.groupby(["modality", "condition"], as_index=False)
        .agg(
            valid_pairs=("similarity", lambda x: int(np.isfinite(x).sum())),
            mean_similarity=("similarity", "mean"),
            median_similarity=("similarity", "median"),
            std_similarity=("similarity", "std"),
            min_similarity=("similarity", "min"),
            max_similarity=("similarity", "max"),
        )
    )
    return out


def paired_wilcoxon_table(df):
    if not HAVE_WILCOXON:
        return pd.DataFrame()

    if "Shuffled" not in set(df["condition"].unique()):
        return pd.DataFrame()

    rows = []

    for modality in ["Geometry", "Activity", "Multimodal"]:
        sub = df[df["modality"] == modality].copy()

        pivot = sub.pivot_table(
            index=["animal_a", "animal_b"],
            columns="condition",
            values="similarity",
            aggfunc="mean",
        ).reset_index()

        if not {"Aligned", "Shuffled"}.issubset(pivot.columns):
            continue

        paired = pivot.dropna(subset=["Aligned", "Shuffled"]).copy()

        if len(paired) < 3:
            continue

        try:
            stat, p = wilcoxon(
                paired["Aligned"].values,
                paired["Shuffled"].values,
                alternative="greater",
            )
        except Exception:
            stat, p = np.nan, np.nan

        rows.append({
            "modality": modality,
            "n_pairs": len(paired),
            "aligned_mean": paired["Aligned"].mean(),
            "shuffled_mean": paired["Shuffled"].mean(),
            "mean_diff": (paired["Aligned"] - paired["Shuffled"]).mean(),
            "wilcoxon_stat": stat,
            "wilcoxon_p_greater": p,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plot
# ============================================================

def plot_distribution_figure(
    df,
    out_png,
    out_pdf,
    dataset_name,
):
    """
    Compact ICLR-style visualization of cross-animal population-relation
    similarity. One observation is one unordered animal pair.

    Each violin uses all valid animal pairs.

    Visual encoding:
        violin       = full distribution
        vertical bar = interquartile range (25%-75%)
        open circle  = median
        number       = median value

    Individual points are intentionally omitted because 703 overplotted points
    obscure the distribution. The horizontal dashed line marks Pearson r = 0.
    """

    # ========================================================
    # Configuration
    # ========================================================

    modalities = [
        "Geometry",
        "Activity",
        "Multimodal",
    ]

    positions = np.arange(
        1,
        len(modalities) + 1,
    )

    # Only use correctly aligned animal pairs
    plot_df = df[
        df["condition"] == "Aligned"
    ].copy()

    # ========================================================
    # Collect values
    # ========================================================

    values_all = []

    for modality in modalities:

        vals = (
            plot_df.loc[
                plot_df["modality"] == modality,
                "similarity",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        if len(vals) == 0:
            raise RuntimeError(
                f"No valid values for modality: {modality}"
            )

        values_all.append(vals)

    # Count the actual observational units rather than repeating n above every
    # modality. All modalities should normally use these same animal pairs.
    pair_table = plot_df[["animal_a", "animal_b"]].drop_duplicates()
    n_pairs = len(pair_table)
    n_animals = len(
        set(pair_table["animal_a"].astype(str))
        | set(pair_table["animal_b"].astype(str))
    )

    modality_counts = [len(v) for v in values_all]
    if len(set(modality_counts)) != 1 or modality_counts[0] != n_pairs:
        print(
            "[WARN] modality-specific valid-pair counts differ: "
            + ", ".join(
                f"{m}={n}" for m, n in zip(modalities, modality_counts)
            )
        )

    # ========================================================
    # Figure
    # ========================================================

    # 3.35 in fits one ICLR column. Use the PDF output in the manuscript.
    fig, ax = plt.subplots(figsize=(3.35, 2.55))

    # ========================================================
    # Violin plots
    # ========================================================

    vp = ax.violinplot(
        dataset=values_all,
        positions=positions,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.18,
    )

    # Color-blind-friendly, restrained modality colors.
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_linewidth(0.75)
        body.set_alpha(0.24)

    # ========================================================
    # IQR + median
    # ========================================================

    for x_pos, vals, color in zip(
        positions,
        values_all,
        colors,
    ):

        q25, med, q75 = np.percentile(
            vals,
            [25, 50, 75],
        )

        # IQR: a slim rounded line is cleaner than the previous dark square.
        ax.plot(
            [x_pos, x_pos],
            [q25, q75],
            color=color,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=4,
        )

        # Median: open circle remains legible in print and at single-column size.
        ax.scatter(
            [x_pos],
            [med],
            s=22,
            facecolor="white",
            edgecolor=color,
            linewidth=1.2,
            zorder=5,
        )

        text_y = q75 + 0.035
        text_y = min(text_y, 0.985)

        ax.text(
            x_pos,
            text_y,
            f"{med:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            fontweight="semibold",
            color="0.18",
            zorder=6,
        )

    # ========================================================
    # Axes
    # ========================================================

    ax.set_xlim(
        0.5,
        3.5,
    )

    # Pearson r is theoretically in [-1, 1]. Avoid the large empty negative
    # half when all observed values are positive, but never hide negative data.
    observed_min = min(float(np.min(v)) for v in values_all)
    y_lower = max(-1.0, min(-0.05, observed_min - 0.05))
    ax.set_ylim(y_lower, 1.02)

    ax.set_xticks(
        positions,
    )

    ax.set_xticklabels(
        modalities,
    )

    # ceil keeps every tick inside the selected limits; using floor here would
    # silently expand a -0.05 lower limit to -0.20.
    tick_start = np.ceil(y_lower / 0.2) * 0.2
    yticks = np.arange(tick_start, 1.001, 0.2)
    if not np.any(np.isclose(yticks, 0.0)):
        yticks = np.sort(np.append(yticks, 0.0))
    ax.set_yticks(yticks)

    ax.set_ylabel(
        "Population-relation similarity, $r$",
        labelpad=5,
    )

    # ========================================================
    # Grid
    # ========================================================

    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        color="#E6E6E6",
        linewidth=0.5,
        alpha=0.8,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    # A short panel title is sufficient; the full statement belongs in the
    # manuscript caption.
    ax.set_title(
        dataset_name,
        loc="left",
        fontsize=9,
        fontweight="semibold",
        pad=6,
    )

    ax.text(
        1.0,
        1.015,
        f"{n_animals} animals · {n_pairs} pairs",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="0.40",
    )

    # Pearson r = 0 reference line. It remains visible even when the lower
    # axis is cropped to the observed positive range.
    ax.axhline(
        0.0,
        color="0.55",
        linewidth=0.7,
        linestyle=(0, (3, 2)),
        zorder=1,
    )

    # ========================================================
    # Clean publication style
    # ========================================================

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3.0,
        width=0.7,
    )

    # ========================================================
    # Layout
    # ========================================================

    fig.tight_layout(pad=0.45)

    # ========================================================
    # Save
    # ========================================================

    fig.savefig(
        out_png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        out_pdf,
        bbox_inches="tight",
        facecolor="white",
    )

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
        "--dataset-name",
        default="Atanas",
    )
    p.add_argument(
        "--save-dir",
        default="runs/mprt_v1_1/mechanism_population_relation/atanas_overall_similarity",
    )
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
    )
    p.add_argument(
        "--min-shared-ids",
        type=int,
        default=8,
        help="Minimum number of shared GT identities required for an animal pair",
    )
    p.add_argument(
        "--with-shuffle",
        action="store_true",
        help="Also compute identity-shuffled control",
    )
    p.add_argument(
        "--shuffle-repeats",
        type=int,
        default=50,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return p.parse_args()


def main():
    args = parse_args()
    setup_style()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("OVERALL CROSS-ANIMAL POPULATION-RELATION SIMILARITY")
    print("=" * 100)
    print(f"Dataset         : {args.dataset_name}")
    print(f"Data root       : {args.data_root}")
    print(f"Save dir        : {args.save_dir}")
    print(f"Min shared IDs  : {args.min_shared_ids}")
    print(f"With shuffle    : {args.with_shuffle}")
    if args.with_shuffle:
        print(f"Shuffle repeats : {args.shuffle_repeats}")

    worms = load_worms(args.data_root, splits=args.splits)

    df = compute_pairwise_overall_similarity(
        worms=worms,
        min_shared_ids=args.min_shared_ids,
        with_shuffle=args.with_shuffle,
        shuffle_repeats=args.shuffle_repeats,
        seed=args.seed,
    )

    # save long csv
    long_csv = save_dir / "overall_population_relation_similarity_long.csv"
    df.to_csv(long_csv, index=False)

    summary = summarize(df)
    summary_csv = save_dir / "overall_population_relation_similarity_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print("\n[SUMMARY]")
    print(summary.to_string(index=False))

    if args.with_shuffle:
        stats = paired_wilcoxon_table(df)
        if len(stats) > 0:
            stats_csv = save_dir / "overall_population_relation_similarity_aligned_vs_shuffled.csv"
            stats.to_csv(stats_csv, index=False)

            print("\n[ALIGNED vs SHUFFLED]")
            print(stats.to_string(index=False))

    out_png = save_dir / "overall_population_relation_similarity.png"
    out_pdf = save_dir / "overall_population_relation_similarity.pdf"

    plot_distribution_figure(
        df=df,
        out_png=out_png,
        out_pdf=out_pdf,
        dataset_name=args.dataset_name,
    )

    print("\n" + "=" * 100)
    print("SAVED")
    print("=" * 100)
    print(f"PNG : {out_png}")
    print(f"PDF : {out_pdf}")
    print(f"CSV : {long_csv}")
    print(f"CSV : {summary_csv}")


if __name__ == "__main__":
    main()
