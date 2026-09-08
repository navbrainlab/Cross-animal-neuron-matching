#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Relation specificity margin vs MPRT Top-1

1 x 3 publication-style scatter figure:
    Geometry | Activity | Multimodal

Each point = one neuron identity.

x:
    identity-specific relational margin
        margin_k = Same_k - Hard_k

y:
    real MPRT per-identity Top-1 accuracy

point size:
    relation support = number of unique animals in which the
    identity contributes valid relation comparisons

Statistics:
    Spearman rho + two-sided p-value

Inputs are EXISTING outputs from:
    analyze_cross_animal_population_relation_mechanism.py

No retraining / re-evaluation required.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",

        "font.size": 9,

        "axes.titlesize": 11,
        "axes.labelsize": 10,

        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,

        "axes.linewidth": 0.8,

        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "legend.fontsize": 8,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "savefig.dpi": 400,

        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# ============================================================
# Load relation-analysis CSV
# ============================================================

def load_relation_raw(path):
    """
    Load raw Same / Hard / Random cross-animal relation comparisons.

    Expected columns:
        source_worm
        target_worm
        query_id
        group
        similarity
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Relation CSV not found:\n{path}"
        )

    df = pd.read_csv(path)

    required = {
        "source_worm",
        "target_worm",
        "query_id",
        "group",
        "similarity",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"\nInvalid relation CSV:\n{path}\n"
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()

    # Normalize identity strings
    df["query_id"] = (
        df["query_id"]
        .astype(str)
        .str.strip()
    )

    # Normalize group names
    df["group"] = (
        df["group"]
        .astype(str)
        .str.strip()
    )

    # Force numeric similarity
    df["similarity"] = pd.to_numeric(
        df["similarity"],
        errors="coerce",
    )

    # Remove invalid rows
    df = df[
        df["similarity"].notna()
        &
        df["query_id"].notna()
        &
        df["source_worm"].notna()
        &
        df["target_worm"].notna()
    ].copy()

    valid_groups = {
        "Same",
        "Hard",
        "Random",
    }

    unknown_groups = (
        set(df["group"].unique())
        -
        valid_groups
    )

    if unknown_groups:
        print(
            "[WARN] unexpected relation groups:",
            sorted(unknown_groups),
        )

    return df.reset_index(drop=True)
# ============================================================
# Per-identity margin
# ============================================================

def compute_identity_margin(df):
    """
    IMPORTANT:

    Hard contains K hard-negative candidate rows per query.

    Therefore we must first average Hard candidates WITHIN:
        source worm
        target worm
        query identity

    Otherwise Hard receives K times more weight than Same.

    Then:
        margin(pair, identity)
          = Same(pair, identity) - mean Hard(pair, identity)

    Finally:
        margin(identity)
          = mean over cross-animal pairs
    """

    relation = df[
        df["group"].isin(
            ["Same", "Hard"]
        )
    ].copy()

    # --------------------------------------------------------
    # First average within biological animal-pair/query
    # --------------------------------------------------------

    pair_identity = (
        relation
        .groupby(
            [
                "source_worm",
                "target_worm",
                "query_id",
                "group",
            ],
            as_index=False,
        )
        ["similarity"]
        .mean()
    )

    # --------------------------------------------------------
    # Same / Hard become columns
    # --------------------------------------------------------

    piv = (
        pair_identity
        .pivot_table(
            index=[
                "source_worm",
                "target_worm",
                "query_id",
            ],
            columns="group",
            values="similarity",
            aggfunc="mean",
        )
        .reset_index()
    )

    piv.columns.name = None

    if (
        "Same" not in piv.columns
        or
        "Hard" not in piv.columns
    ):
        raise RuntimeError(
            "Could not construct Same/Hard paired values."
        )

    piv = piv.dropna(
        subset=[
            "Same",
            "Hard",
        ]
    ).copy()

    piv["pair_margin"] = (
        piv["Same"]
        -
        piv["Hard"]
    )

    # --------------------------------------------------------
    # One row per identity
    # --------------------------------------------------------

    rows = []

    for identity, sub in piv.groupby(
        "query_id"
    ):

        # Unique worms participating in valid comparisons
        animals = set(
            sub["source_worm"].astype(str)
        ) | set(
            sub["target_worm"].astype(str)
        )

        rows.append({
            "identity": str(identity),

            "same_similarity": float(
                sub["Same"].mean()
            ),

            "hard_similarity": float(
                sub["Hard"].mean()
            ),

            "stability_margin": float(
                sub["pair_margin"].mean()
            ),

            "median_margin": float(
                sub["pair_margin"].median()
            ),

            "n_animal_pairs": int(
                len(sub)
            ),

            "support_animals": int(
                len(animals)
            ),
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Load MPRT accuracy
# ============================================================

def load_accuracy(path):
    df = pd.read_csv(path)

    required = {
        "identity",
        "mprt_top1",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"\n{path}\n"
            f"Missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df = df.copy()

    df["identity"] = (
        df["identity"]
        .astype(str)
        .str.strip()
    )

    df["mprt_top1"] = pd.to_numeric(
        df["mprt_top1"],
        errors="coerce",
    )

    # Accept either 0-1 or already-percent values
    finite = df["mprt_top1"].dropna()

    if len(finite) > 0 and finite.max() > 1.5:
        df["mprt_top1"] = (
            df["mprt_top1"]
            /
            100.0
        )

    return df


# ============================================================
# Optional support size scaling
# ============================================================

def map_point_sizes(
    support,
    min_size=20,
    max_size=70,
):
    support = np.asarray(
        support,
        dtype=float,
    )

    if len(support) == 0:
        return np.array([])

    smin = np.min(
        support
    )

    smax = np.max(
        support
    )

    if smax <= smin:
        return np.full(
            len(support),
            35.0,
        )

    z = (
        support - smin
    ) / (
        smax - smin
    )

    # sqrt scaling is visually less aggressive
    z = np.sqrt(
        z
    )

    return (
        min_size
        +
        z
        *
        (
            max_size
            -
            min_size
        )
    )


# ============================================================
# Optional bootstrap CI for Spearman
# ============================================================

def bootstrap_spearman(
    x,
    y,
    n_boot=10000,
    seed=42,
):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return np.nan, np.nan

    rng = np.random.default_rng(
        seed
    )

    vals = []

    n = len(x)

    for _ in range(
        n_boot
    ):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        xb = x[idx]
        yb = y[idx]

        if (
            np.std(xb) < 1e-12
            or
            np.std(yb) < 1e-12
        ):
            continue

        r, _ = spearmanr(
            xb,
            yb,
        )

        if np.isfinite(r):
            vals.append(
                r
            )

    if not vals:
        return np.nan, np.nan

    lo, hi = np.quantile(
        vals,
        [
            0.025,
            0.975,
        ],
    )

    return (
        float(lo),
        float(hi),
    )


# ============================================================
# Optional binned trend
# ============================================================

def draw_binned_median(
    ax,
    x,
    y,
    bins=5,
):
    """
    Draw a subtle non-parametric trend:
    median y within quantiles of x.

    This is NOT a regression fit.
    """

    df = pd.DataFrame({
        "x": x,
        "y": y,
    }).dropna()

    if len(df) < bins * 3:
        return

    try:
        df["bin"] = pd.qcut(
            df["x"],
            q=bins,
            duplicates="drop",
        )
    except Exception:
        return

    trend = (
        df.groupby(
            "bin",
            observed=True,
        )
        .agg(
            x=("x", "median"),
            y=("y", "median"),
        )
        .reset_index(
            drop=True
        )
    )

    if len(trend) < 2:
        return

    ax.plot(
        trend["x"],
        trend["y"],
        linewidth=1.5,
        alpha=0.70,
        zorder=4,
    )

    ax.scatter(
        trend["x"],
        trend["y"],
        s=18,
        marker="D",
        zorder=5,
    )


def make_figure(
    mode_data,
    output_png,
    output_pdf,
    dataset_name,
    draw_trend=False,   # 保留参数兼容，但不再使用
    bootstrap=10000,
    seed=42,
):
    """
    Publication-style 1x3 scatter figure.

    Improvements:
    1. Independent x-axis ranges for Geo / Activity / Multimodal.
    2. No support-size encoding or confusing legend.
    3. Shared y-axis and shared x-axis description.
    4. Spearman rho/p only; no regression line.
    """

    modes = [
        "Geometry",
        "Activity",
        "Multimodal",
    ]

    letters = [
        "A",
        "B",
        "C",
    ]

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.2, 3.55),
        sharey=True,
    )

    stats_rows = []

    for panel_i, (ax, mode) in enumerate(
        zip(axes, modes)
    ):
        df = mode_data[mode].copy()

        x = df["stability_margin"].to_numpy(
            dtype=float
        )

        y = (
            100.0
            *
            df["mprt_top1"].to_numpy(
                dtype=float
            )
        )

        valid = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

        x = x[valid]
        y = y[valid]

        # ====================================================
        # Spearman
        # ====================================================
        rho, p = spearmanr(
            x,
            y,
        )

        ci_lo, ci_hi = bootstrap_spearman(
            x,
            y,
            n_boot=bootstrap,
            seed=seed + panel_i,
        )

        stats_rows.append({
            "mode": mode,
            "n_identities": len(x),
            "spearman_rho": float(rho),
            "spearman_p": float(p),
            "rho_ci95_low": float(ci_lo),
            "rho_ci95_high": float(ci_hi),
            "mean_margin": float(np.mean(x)),
            "median_margin": float(np.median(x)),
            "fraction_positive_margin": float(
                np.mean(x > 0)
            ),
        })

        # ====================================================
        # Scatter
        # ====================================================

        # Every identity receives the same visual weight.
        ax.scatter(
            x,
            y,
            s=26,
            alpha=0.58,
            linewidths=0,
            zorder=3,
        )

        # Same = Hard reference
        ax.axvline(
            0.0,
            linestyle=(0, (4, 3)),
            linewidth=1.0,
            alpha=0.50,
            zorder=1,
        )

        # ====================================================
        # Independent x range for EACH modality
        # ====================================================

        xmin = float(
            np.min(x)
        )

        xmax = float(
            np.max(x)
        )

        span = xmax - xmin

        if span < 1e-8:
            span = max(
                abs(xmin),
                0.1,
            )

        pad = 0.05 * span

        # Give a little extra room so x=0 does not touch edge
        xlo = min(
            xmin - pad,
            -0.03 * span,
        )

        xhi = max(
            xmax + pad,
            0.03 * span,
        )

        ax.set_xlim(
            xlo,
            xhi,
        )

        # ====================================================
        # Shared y
        # ====================================================

        ax.set_ylim(
            -2,
            102,
        )

        ax.set_yticks(
            np.arange(
                0,
                101,
                20,
            )
        )

        # horizontal grid only
        ax.grid(
            axis="y",
            linewidth=0.55,
            alpha=0.16,
        )

        ax.grid(
            axis="x",
            visible=False,
        )

        # ====================================================
        # Statistics annotation
        # ====================================================

        if p < 0.001:
            p_text = r"$p<0.001$"
        else:
            p_text = rf"$p={p:.3f}$"

        stat_text = (
            rf"$\rho={rho:.2f}$"
            "\n"
            f"{p_text}"
            "\n"
            rf"$n={len(x)}$"
        )

        ax.text(
            0.045,
            0.955,
            stat_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.3,
            linespacing=1.25,
        )

        # ====================================================
        # Panel title
        # ====================================================

        ax.set_title(
            mode,
            fontsize=11,
            fontweight="medium",
            pad=10,
        )

        # Panel letters outside plotting area
        ax.text(
            -0.12,
            1.065,
            letters[panel_i],
            transform=ax.transAxes,
            fontsize=13.5,
            fontweight="bold",
            va="top",
            ha="left",
            clip_on=False,
        )

        # ====================================================
        # Axis style
        # ====================================================

        ax.tick_params(
            direction="out",
            width=0.8,
            length=3.5,
        )

        ax.spines["left"].set_linewidth(
            0.8
        )

        ax.spines["bottom"].set_linewidth(
            0.8
        )

        ax.spines["top"].set_visible(
            False
        )

        ax.spines["right"].set_visible(
            False
        )

    # ========================================================
    # Shared labels
    # ========================================================

    axes[0].set_ylabel(
        "Per-identity Top-1 accuracy (%)",
        fontsize=10,
        labelpad=7,
    )

    # One shared x-axis label instead of repeating 3 times
    fig.supxlabel(
        "Identity-specific relation margin (Same − Hard)",
        fontsize=10,
        y=0.015,
    )

    # ========================================================
    # Main title
    # ========================================================

    fig.suptitle(
        "Relational specificity is associated with matching performance",
        fontsize=12.5,
        fontweight="medium",
        y=1.015,
    )

    # Dataset label can stay small rather than dominating title
    fig.text(
        0.5,
        0.955,
        dataset_name,
        ha="center",
        va="top",
        fontsize=9,
        alpha=0.65,
    )

    # ========================================================
    # Layout
    # ========================================================

    fig.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.18,
        top=0.82,
        wspace=0.25,
    )

    fig.savefig(
        output_png,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    return pd.DataFrame(
        stats_rows
    )


# ============================================================
# Main
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base-dir",
        default=(
            "runs/mprt_v1_1/"
            "mechanism_population_relation"
        ),
    )

    p.add_argument(
        "--dataset-prefix",
        default="atanas",
    )

    p.add_argument(
        "--dataset-name",
        default="Atanas",
    )

    p.add_argument(
        "--accuracy-csv",
        default=None,
    )

    p.add_argument(
        "--out-dir",
        default=None,
    )

    p.add_argument(
        "--draw-binned-trend",
        action="store_true",
        help=(
            "Draw a subtle 5-bin median trend. "
            "This is preferable to an OLS regression line "
            "when reporting Spearman correlation."
        ),
    )

    p.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
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

    base = Path(
        args.base_dir
    )

    dirs = {
        "Geometry":
            base
            /
            f"{args.dataset_prefix}_geo",

        "Activity":
            base
            /
            f"{args.dataset_prefix}_act",

        "Multimodal":
            base
            /
            f"{args.dataset_prefix}_multi",
    }

    # --------------------------------------------------------
    # Accuracy already produced by successful Panel D
    # --------------------------------------------------------

    if args.accuracy_csv is not None:
        accuracy_path = Path(
            args.accuracy_csv
        )
    else:
        accuracy_path = (
            dirs["Geometry"]
            /
            "panelD_mprt_per_identity_accuracy.csv"
        )

    if not accuracy_path.exists():
        raise FileNotFoundError(
            "\nMPRT per-identity accuracy CSV not found:\n"
            f"{accuracy_path}\n"
        )

    accuracy = load_accuracy(
        accuracy_path
    )

    print("=" * 100)
    print(
        "RELATION SPECIFICITY MARGIN VS MPRT TOP-1"
    )
    print("=" * 100)

    print(
        f"Dataset       : {args.dataset_name}"
    )

    print(
        f"Accuracy file : {accuracy_path}"
    )

    print(
        f"Accuracy IDs  : {len(accuracy)}"
    )

    mode_data = {}

    margin_tables = {}

    # --------------------------------------------------------
    # Compute margin for each relation mode
    # --------------------------------------------------------

    for mode, directory in dirs.items():

        raw_path = (
            directory
            /
            "panelA_same_hard_random_raw.csv"
        )

        if not raw_path.exists():
            raise FileNotFoundError(
                f"\nMissing {mode} relation file:\n"
                f"{raw_path}\n"
            )

        relation = load_relation_raw(
            raw_path
        )

        margin = compute_identity_margin(
            relation
        )

        margin_tables[
            mode
        ] = margin

        merged = margin.merge(
            accuracy,
            on="identity",
            how="inner",
        )

        merged = merged[
            np.isfinite(
                merged["stability_margin"]
            )
            &
            np.isfinite(
                merged["mprt_top1"]
            )
        ].copy()

        mode_data[
            mode
        ] = merged

        print(
            f"\n[{mode}]"
        )

        print(
            f"relation identities = "
            f"{len(margin)}"
        )

        print(
            f"matched to MPRT     = "
            f"{len(merged)}"
        )

        print(
            f"mean margin         = "
            f"{merged['stability_margin'].mean():.4f}"
        )

        print(
            f"median margin       = "
            f"{merged['stability_margin'].median():.4f}"
        )

        print(
            f"positive margin     = "
            f"{100 * np.mean(merged['stability_margin'] > 0):.2f}%"
        )

        rho, p = spearmanr(
            merged["stability_margin"],
            merged["mprt_top1"],
        )

        print(
            f"Spearman rho        = "
            f"{rho:.4f}"
        )

        print(
            f"Spearman p          = "
            f"{p:.4e}"
        )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    if args.out_dir is not None:
        out_dir = Path(
            args.out_dir
        )
    else:
        out_dir = (
            base
            /
            f"{args.dataset_prefix}_margin_vs_top1"
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save all per-mode tables
    for mode, table in margin_tables.items():
        slug = mode.lower()

        table.to_csv(
            out_dir
            /
            f"{slug}_identity_margin.csv",
            index=False,
        )

        mode_data[
            mode
        ].to_csv(
            out_dir
            /
            f"{slug}_margin_vs_accuracy.csv",
            index=False,
        )

    png = (
        out_dir
        /
        "relation_margin_vs_mprt_top1_1x3.png"
    )

    pdf = (
        out_dir
        /
        "relation_margin_vs_mprt_top1_1x3.pdf"
    )

    stats = make_figure(
        mode_data=mode_data,
        output_png=png,
        output_pdf=pdf,
        dataset_name=args.dataset_name,
        draw_trend=args.draw_binned_trend,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )

    stats.to_csv(
        out_dir
        /
        "relation_margin_vs_top1_statistics.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("FINAL STATISTICS")
    print("=" * 100)

    print(
        stats.to_string(
            index=False
        )
    )

    print("\n" + "=" * 100)
    print("SAVED")
    print("=" * 100)

    print(
        f"PNG : {png}"
    )

    print(
        f"PDF : {pdf}"
    )

    print(
        f"CSV : "
        f"{out_dir / 'relation_margin_vs_top1_statistics.csv'}"
    )


if __name__ == "__main__":
    main()