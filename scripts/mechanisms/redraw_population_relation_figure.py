#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Publication-style redraw for cross-animal population relation analysis.

Main figure
-----------
(A) Geometry / Activity / Multimodal:
    Same / Hard / Random relation similarity
    Point estimate + 95% bootstrap interval over animal-pair means.

(B) Multimodal identity-specific relation margin:
        margin_i = mean(Same_i) - mean(Hard_i)
    All identities are ranked by margin.

(C) Multimodal identity-specific margin vs real MPRT per-identity Top-1.
    Spearman correlation is reported.
    No regression line is drawn to avoid visually overstating a weak relation.

Optional supplementary figure
-----------------------------
Raw multimodal identity × identity heatmap.

This script uses EXISTING mechanism-analysis outputs.
It does not rerun the model or relation extraction.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# Style
# ============================================================

def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,

        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "figure.dpi": 150,
        "savefig.dpi": 400,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def clean_axis(ax):
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out")
    ax.grid(
        axis="y",
        linewidth=0.5,
        alpha=0.18,
    )


# ============================================================
# IO
# ============================================================

def load_panel_a(path):
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
            f"{path} missing columns: {missing}"
        )

    return df


def load_accuracy(path):
    df = pd.read_csv(path)

    required = {
        "identity",
        "mprt_top1",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"{path} missing columns: {missing}"
        )

    return df


# ============================================================
# Panel A statistics
# ============================================================

def animal_pair_means(df):
    """
    First aggregate neuron-level comparisons within each ordered animal pair.

    This prevents a pair containing more neurons from automatically receiving
    more visual weight than a pair containing fewer neurons.

    Output:
        source_worm, target_worm, group, similarity
    """

    pair_df = (
        df.groupby(
            [
                "source_worm",
                "target_worm",
                "group",
            ],
            as_index=False,
        )["similarity"]
        .mean()
    )

    return pair_df


def bootstrap_mean_ci(
    values,
    n_boot=10000,
    seed=42,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    mean = float(
        np.mean(values)
    )

    if len(values) == 1:
        return mean, mean, mean

    rng = np.random.default_rng(
        seed
    )

    # chunked bootstrap to avoid excessive memory
    boot_means = np.empty(
        n_boot,
        dtype=float,
    )

    chunk = 1000

    for start in range(
        0,
        n_boot,
        chunk,
    ):
        end = min(
            start + chunk,
            n_boot,
        )

        size = end - start

        idx = rng.integers(
            0,
            len(values),
            size=(
                size,
                len(values),
            ),
        )

        boot_means[start:end] = (
            values[idx].mean(axis=1)
        )

    lo, hi = np.quantile(
        boot_means,
        [0.025, 0.975],
    )

    return (
        mean,
        float(lo),
        float(hi),
    )


def summarize_modes(
    dfs,
    n_boot,
    seed,
):
    rows = []

    group_order = [
        "Same",
        "Hard",
        "Random",
    ]

    for mode, df in dfs.items():

        pair_df = animal_pair_means(
            df
        )

        for gi, group in enumerate(
            group_order
        ):
            vals = pair_df.loc[
                pair_df["group"] == group,
                "similarity",
            ].values

            mean, lo, hi = bootstrap_mean_ci(
                vals,
                n_boot=n_boot,
                seed=seed + gi,
            )

            rows.append({
                "mode": mode,
                "group": group,
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_animal_pairs": len(vals),
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# Identity-specific margin
# ============================================================

def compute_identity_specificity(df):
    """
    Calculate identity-specific Same-Hard margin.

    Importantly:
    1. Hard candidates are averaged within each animal pair/query first.
    2. Same and Hard are then averaged across animal pairs for each identity.

    margin_i =
        E_pair[Same similarity for identity i]
        -
        E_pair[mean Hard similarity for identity i]
    """

    # One value per:
    # source animal × target animal × neuron identity × group.
    per_pair_identity = (
        df[df["group"].isin(["Same", "Hard"])]
        .groupby(
            [
                "source_worm",
                "target_worm",
                "query_id",
                "group",
            ],
            as_index=False,
        )["similarity"]
        .mean()
    )

    pivot = (
        per_pair_identity
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

    pivot.columns.name = None

    pivot = pivot.dropna(
        subset=[
            "Same",
            "Hard",
        ]
    ).copy()

    pivot["pair_margin"] = (
        pivot["Same"]
        -
        pivot["Hard"]
    )

    rows = []

    for identity, sub in pivot.groupby(
        "query_id"
    ):
        margin_vals = sub[
            "pair_margin"
        ].values

        same_vals = sub[
            "Same"
        ].values

        hard_vals = sub[
            "Hard"
        ].values

        rows.append({
            "identity": identity,

            "n_animal_pairs": len(sub),

            "same_similarity": float(
                np.mean(same_vals)
            ),

            "hard_similarity": float(
                np.mean(hard_vals)
            ),

            "specificity_margin": float(
                np.mean(margin_vals)
            ),

            "median_specificity_margin": float(
                np.median(margin_vals)
            ),
        })

    out = pd.DataFrame(
        rows
    )

    return out


# ============================================================
# Main figure
# ============================================================

def make_main_figure(
    summary,
    specificity,
    accuracy,
    output_png,
    output_pdf,
    dataset_name,
):
    # --------------------------------------------------------
    # Merge specificity with MPRT performance
    # --------------------------------------------------------

    merged = specificity.merge(
        accuracy,
        on="identity",
        how="inner",
    )

    merged = merged[
        np.isfinite(
            merged["specificity_margin"]
        )
        &
        np.isfinite(
            merged["mprt_top1"]
        )
    ].copy()

    if len(merged) >= 3:
        rho, p = spearmanr(
            merged["specificity_margin"],
            merged["mprt_top1"],
        )
    else:
        rho = np.nan
        p = np.nan

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(13.2, 3.8)
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[
            1.05,
            1.05,
            1.00,
        ],
        wspace=0.34,
    )

    # ========================================================
    # A
    # ========================================================

    ax = fig.add_subplot(
        gs[0, 0]
    )

    modes = [
        "Geometry",
        "Activity",
        "Multimodal",
    ]

    groups = [
        "Same",
        "Hard",
        "Random",
    ]

    # Small horizontal offsets create a clean grouped point plot
    offsets = {
        "Same": -0.18,
        "Hard": 0.00,
        "Random": 0.18,
    }

    markers = {
        "Same": "o",
        "Hard": "s",
        "Random": "^",
    }

    for group in groups:

        xs = []
        ys = []
        yerr_low = []
        yerr_high = []

        for x, mode in enumerate(
            modes
        ):
            row = summary[
                (summary["mode"] == mode)
                &
                (summary["group"] == group)
            ]

            if len(row) != 1:
                continue

            row = row.iloc[0]

            xpos = (
                x
                +
                offsets[group]
            )

            xs.append(
                xpos
            )

            ys.append(
                row["mean"]
            )

            yerr_low.append(
                row["mean"]
                -
                row["ci_low"]
            )

            yerr_high.append(
                row["ci_high"]
                -
                row["mean"]
            )

        ax.errorbar(
            xs,
            ys,
            yerr=[
                yerr_low,
                yerr_high,
            ],
            fmt=markers[group],
            markersize=5.2,
            linewidth=1.15,
            capsize=2.7,
            label=group,
            zorder=3,
        )

    ax.set_xticks(
        range(
            len(modes)
        )
    )

    ax.set_xticklabels(
        modes
    )

    ax.set_ylabel(
        "Cross-animal relation similarity"
    )

    ax.set_ylim(
        -0.02,
        1.00,
    )

    ax.legend(
        frameon=False,
        loc="lower left",
        ncol=1,
    )

    clean_axis(
        ax
    )

    ax.text(
        -0.16,
        1.05,
        "A",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )

    ax.set_title(
        "Relational stability and discrimination",
        pad=8,
    )

    # ========================================================
    # B
    # ========================================================

    ax = fig.add_subplot(
        gs[0, 1]
    )

    ranked = (
        specificity
        .sort_values(
            "specificity_margin",
            ascending=True,
        )
        .reset_index(
            drop=True
        )
    )

    x = np.arange(
        1,
        len(ranked) + 1,
    )

    y = ranked[
        "specificity_margin"
    ].values

    ax.plot(
        x,
        y,
        linewidth=1.0,
        alpha=0.72,
        zorder=2,
    )

    ax.scatter(
        x,
        y,
        s=14,
        alpha=0.85,
        zorder=3,
    )

    ax.axhline(
        0,
        linewidth=0.9,
        linestyle="--",
        alpha=0.65,
        zorder=1,
    )

    positive_fraction = float(
        np.mean(
            y > 0
        )
    )

    median_margin = float(
        np.median(y)
    )

    ax.text(
        0.04,
        0.95,
        (
            f"{100 * positive_fraction:.0f}% positive margin\n"
            f"median = {median_margin:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.8,
    )

    ax.set_xlabel(
        "Neuron identities ranked by specificity"
    )

    ax.set_ylabel(
        "Same − hard similarity"
    )

    clean_axis(
        ax
    )

    ax.text(
        -0.16,
        1.05,
        "B",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )

    ax.set_title(
        "Identity-specific multimodal relations",
        pad=8,
    )

    # ========================================================
    # C
    # ========================================================

    ax = fig.add_subplot(
        gs[0, 2]
    )

    ax.scatter(
        merged["specificity_margin"],
        100 * merged["mprt_top1"],
        s=24,
        alpha=0.72,
    )

    ax.set_xlabel(
        "Multimodal identity-specific margin"
    )

    ax.set_ylabel(
        "MPRT Top-1 per identity (%)"
    )

    clean_axis(
        ax
    )

    if np.isfinite(
        rho
    ):
        if p < 1e-3:
            p_text = "$p<10^{-3}$"
        else:
            p_text = f"$p={p:.3f}$"

        stat_text = (
            f"Spearman $\\rho={rho:.2f}$\n"
            f"{p_text}\n"
            f"$n={len(merged)}$ identities"
        )

        ax.text(
            0.04,
            0.96,
            stat_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8.8,
        )

    ax.text(
        -0.16,
        1.05,
        "C",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
    )

    ax.set_title(
        "Relational specificity and matching",
        pad=8,
    )

    # --------------------------------------------------------
    # Global formatting
    # --------------------------------------------------------

    fig.suptitle(
        f"{dataset_name}: cross-animal population relations",
        fontsize=12.5,
        y=1.035,
    )

    fig.savefig(
        output_png,
        bbox_inches="tight",
        dpi=400,
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    return (
        ranked,
        merged,
        rho,
        p,
    )


# ============================================================
# Supplementary heatmap
# ============================================================

def make_supplement_heatmap(
    matrix_path,
    identity_path,
    output_png,
    output_pdf,
    dataset_name,
):
    matrix = np.load(
        matrix_path
    )

    id_df = pd.read_csv(
        identity_path
    )

    identities = id_df[
        "identity"
    ].astype(str).tolist()

    if matrix.shape != (
        len(identities),
        len(identities),
    ):
        raise RuntimeError(
            "Heatmap matrix and identity list have inconsistent shapes."
        )

    # Rank identities by their diagonal margin.
    margins = []

    for i in range(
        len(identities)
    ):
        diag = matrix[
            i,
            i,
        ]

        off = np.delete(
            matrix[i],
            i,
        )

        off = off[
            np.isfinite(off)
        ]

        if (
            np.isfinite(diag)
            and
            len(off) > 0
        ):
            margin = (
                diag
                -
                np.max(off)
            )
        else:
            margin = -np.inf

        margins.append(
            margin
        )

    order = np.argsort(
        margins
    )[::-1]

    matrix = matrix[
        np.ix_(
            order,
            order,
        )
    ]

    identities = [
        identities[i]
        for i in order
    ]

    fig, ax = plt.subplots(
        figsize=(6.2, 5.4)
    )

    im = ax.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        aspect="equal",
        interpolation="nearest",
    )

    n = len(
        identities
    )

    ax.set_xticks(
        np.arange(n)
    )

    ax.set_yticks(
        np.arange(n)
    )

    ax.set_xticklabels(
        identities,
        rotation=90,
        fontsize=6.7,
    )

    ax.set_yticklabels(
        identities,
        fontsize=6.7,
    )

    ax.set_xlabel(
        "Candidate identity"
    )

    ax.set_ylabel(
        "Query identity"
    )

    ax.set_title(
        f"{dataset_name}: multimodal cross-identity relation similarity"
    )

    # Explicitly mark the GT diagonal without altering heatmap values.
    for i in range(n):
        rect = Rectangle(
            (
                i - 0.5,
                i - 0.5,
            ),
            1,
            1,
            fill=False,
            linewidth=0.65,
            edgecolor="white",
        )

        ax.add_patch(
            rect
        )

    cbar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.035,
    )

    cbar.set_label(
        "Cross-animal relation similarity"
    )

    fig.tight_layout()

    fig.savefig(
        output_png,
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        output_pdf,
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
        "--base-dir",
        default=(
            "runs/mprt_v1_1/"
            "mechanism_population_relation"
        ),
    )

    p.add_argument(
        "--dataset-prefix",
        default="atanas",
        help="Directory prefix: atanas or rld",
    )

    p.add_argument(
        "--dataset-name",
        default="Atanas",
    )

    p.add_argument(
        "--accuracy-csv",
        default=None,
        help=(
            "MPRT per-identity accuracy CSV. "
            "Defaults to <base>/<prefix>_geo/"
            "panelD_mprt_per_identity_accuracy.csv"
        ),
    )

    p.add_argument(
        "--out-dir",
        default=None,
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

    geo_dir = (
        base
        /
        f"{args.dataset_prefix}_geo"
    )

    act_dir = (
        base
        /
        f"{args.dataset_prefix}_act"
    )

    multi_dir = (
        base
        /
        f"{args.dataset_prefix}_multi"
    )

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else
        base
        /
        f"{args.dataset_prefix}_publication"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "Geometry":
            geo_dir
            /
            "panelA_same_hard_random_raw.csv",

        "Activity":
            act_dir
            /
            "panelA_same_hard_random_raw.csv",

        "Multimodal":
            multi_dir
            /
            "panelA_same_hard_random_raw.csv",
    }

    print("=" * 90)
    print("PUBLICATION FIGURE REDRAW")
    print("=" * 90)

    dfs = {}

    for mode, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {mode} Panel A file:\n{path}"
            )

        dfs[mode] = load_panel_a(
            path
        )

        print(
            f"[LOAD] {mode:10s} "
            f"N={len(dfs[mode]):,} "
            f"{path}"
        )

    # --------------------------------------------------------
    # A summary
    # --------------------------------------------------------

    summary = summarize_modes(
        dfs,
        n_boot=args.bootstrap,
        seed=args.seed,
    )

    summary.to_csv(
        out_dir
        /
        "panelA_publication_summary.csv",
        index=False,
    )

    print("\n[PANEL A SUMMARY]")
    print(
        summary[
            [
                "mode",
                "group",
                "mean",
                "ci_low",
                "ci_high",
                "n_animal_pairs",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # B/C use multimodal identity specificity
    # --------------------------------------------------------

    specificity = compute_identity_specificity(
        dfs["Multimodal"]
    )

    specificity.to_csv(
        out_dir
        /
        "multimodal_identity_specificity.csv",
        index=False,
    )

    pos = np.mean(
        specificity[
            "specificity_margin"
        ] > 0
    )

    print("\n[MULTIMODAL SPECIFICITY]")
    print(
        f"identities = {len(specificity)}"
    )
    print(
        f"mean margin = "
        f"{specificity['specificity_margin'].mean():.4f}"
    )
    print(
        f"median margin = "
        f"{specificity['specificity_margin'].median():.4f}"
    )
    print(
        f"positive identities = "
        f"{100 * pos:.2f}%"
    )

    # --------------------------------------------------------
    # MPRT accuracy
    # --------------------------------------------------------

    if args.accuracy_csv:
        accuracy_path = Path(
            args.accuracy_csv
        )
    else:
        accuracy_path = (
            geo_dir
            /
            "panelD_mprt_per_identity_accuracy.csv"
        )

    if not accuracy_path.exists():
        raise FileNotFoundError(
            "\nCould not find real MPRT per-identity accuracy:\n"
            f"{accuracy_path}\n\n"
            "Run the successful geo D analysis first, or pass:\n"
            "--accuracy-csv PATH"
        )

    accuracy = load_accuracy(
        accuracy_path
    )

    print(
        f"\n[ACCURACY] loaded "
        f"{len(accuracy)} identities from "
        f"{accuracy_path}"
    )

    # --------------------------------------------------------
    # Main figure
    # --------------------------------------------------------

    main_png = (
        out_dir
        /
        "population_relation_main_figure.png"
    )

    main_pdf = (
        out_dir
        /
        "population_relation_main_figure.pdf"
    )

    (
        ranked,
        merged,
        rho,
        p,
    ) = make_main_figure(
        summary=summary,
        specificity=specificity,
        accuracy=accuracy,
        output_png=main_png,
        output_pdf=main_pdf,
        dataset_name=args.dataset_name,
    )

    ranked.to_csv(
        out_dir
        /
        "panelB_ranked_identity_specificity.csv",
        index=False,
    )

    merged.to_csv(
        out_dir
        /
        "panelC_specificity_vs_mprt_accuracy.csv",
        index=False,
    )

    print("\n[PANEL C]")
    print(
        f"overlapping identities = "
        f"{len(merged)}"
    )

    print(
        f"Spearman rho = "
        f"{rho:.4f}"
    )

    print(
        f"p = "
        f"{p:.4e}"
    )

    # --------------------------------------------------------
    # Supplementary multimodal heatmap
    # --------------------------------------------------------

    matrix_path = (
        multi_dir
        /
        "panelB_similarity_matrix.npy"
    )

    ids_path = (
        multi_dir
        /
        "panelB_identities.csv"
    )

    if (
        matrix_path.exists()
        and
        ids_path.exists()
    ):
        supp_png = (
            out_dir
            /
            "supp_multimodal_identity_heatmap.png"
        )

        supp_pdf = (
            out_dir
            /
            "supp_multimodal_identity_heatmap.pdf"
        )

        make_supplement_heatmap(
            matrix_path=matrix_path,
            identity_path=ids_path,
            output_png=supp_png,
            output_pdf=supp_pdf,
            dataset_name=args.dataset_name,
        )

        print(
            f"\n[SUPPLEMENT] {supp_png}"
        )

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)

    print(
        f"Main PNG : {main_png}"
    )

    print(
        f"Main PDF : {main_pdf}"
    )

    print(
        f"Data     : {out_dir}"
    )


if __name__ == "__main__":
    main()