#!/usr/bin/env python3
"""Draw a Figure-F-style full relation-profile matrix with gap histograms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mechanisms.plot_relation_profile_similarity_heatmaps import (
    compute_matrices,
    hard_cells,
    load_formal_query_identities,
    select_and_order_identities,
    select_pair,
)


DEFAULT_FOLD_ROOT = (
    REPO_ROOT / "Data/Atanas_SF_unified_000776/cv5_grouped_v1/fold_0"
)
DEFAULT_QUERY_EXPORT = (
    REPO_ROOT
    / "NeuRID_reproducibility/results/visualization_export_atanas_seed42"
    / "neurid_query_neurons_atanas_cv5_seed42.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs/mprt_v1_1/mechanism_population_relation"
    / "atanas_relation_profile_figure_f"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--query-export", type=Path, default=DEFAULT_QUERY_EXPORT)
    parser.add_argument("--hard-k", type=int, default=3)
    parser.add_argument("--min-common-anchors", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_gap_table(matrices: dict, identities: list, hard: list) -> pd.DataFrame:
    hard_index = {(row, column) for row, column, _, _ in hard}
    rows = []
    for mode, display in (("geo", "Geometry"), ("act", "Activity")):
        matrix = matrices[mode]
        for i, query_id in enumerate(identities):
            same = float(matrix[i, i])
            for j, candidate_id in enumerate(identities):
                if i == j:
                    continue
                value = float(matrix[i, j])
                rows.append(
                    {
                        "mode": display,
                        "query_identity": query_id,
                        "candidate_identity": candidate_id,
                        "same_similarity": same,
                        "candidate_similarity": value,
                        "same_minus_candidate": same - value,
                        "is_spatial_hard": (i, j) in hard_index,
                    }
                )
    return pd.DataFrame(rows)


def draw_figure(
    matrices: dict,
    identities: list,
    hard: list,
    gaps: pd.DataFrame,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cmap = plt.get_cmap("RdBu_r")
    try:
        cmap = cmap.copy()
    except AttributeError:
        pass
    cmap.set_bad("#BDBDBD")
    norm = Normalize(vmin=-1.0, vmax=1.0)

    fig = plt.figure(figsize=(11.6, 13.2))
    grid = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 0.032, 0.34],
        height_ratios=[1, 1],
        left=0.105,
        right=0.945,
        bottom=0.075,
        top=0.965,
        wspace=0.25,
        hspace=0.28,
    )
    colorbar_axis = fig.add_subplot(grid[:, 1])
    hard_lookup = {(row, column) for row, column, _, _ in hard}
    last_image = None

    for panel, (mode, display) in enumerate((("geo", "Geometry"), ("act", "Activity"))):
        heat_axis = fig.add_subplot(grid[panel, 0])
        hist_axis = fig.add_subplot(grid[panel, 2])
        matrix = matrices[mode]
        last_image = heat_axis.imshow(
            np.ma.masked_invalid(matrix),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        heat_axis.set_title(display, pad=7)
        heat_axis.set_xticks(np.arange(len(identities)))
        heat_axis.set_yticks(np.arange(len(identities)))
        heat_axis.set_xticklabels(
            identities, rotation=90, ha="center", va="top", fontsize=3.35
        )
        heat_axis.set_yticklabels(identities, fontsize=3.35)
        heat_axis.tick_params(length=0, pad=0.8)
        heat_axis.set_xlabel("Training-reference candidate identity", labelpad=5)
        heat_axis.set_ylabel("Held-out query identity")
        heat_axis.text(
            -0.10,
            1.025,
            chr(ord("A") + panel),
            transform=heat_axis.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

        # Thin overlays remain legible in the vector export without obscuring
        # the dense matrix in the raster preview.
        for index in range(len(identities)):
            heat_axis.add_patch(
                Rectangle(
                    (index - 0.47, index - 0.47),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor="#111111",
                    linewidth=0.34,
                    zorder=4,
                )
            )
        if hard_lookup:
            hard_rows, hard_columns = zip(*sorted(hard_lookup))
            heat_axis.scatter(
                hard_columns,
                hard_rows,
                s=2.2,
                facecolor="#F2A900",
                edgecolor="none",
                marker="s",
                zorder=5,
            )

        mode_gaps = gaps.loc[gaps["mode"] == display]
        all_values = mode_gaps["same_minus_candidate"].astype(float).values
        hard_values = mode_gaps.loc[
            mode_gaps["is_spatial_hard"], "same_minus_candidate"
        ].astype(float).values
        bins = np.linspace(-2.0, 2.0, 81)
        hist_axis.hist(
            all_values,
            bins=bins,
            density=True,
            color="#6B8EAD",
            alpha=0.62,
            linewidth=0,
            label="All wrong candidates",
        )
        hist_axis.hist(
            hard_values,
            bins=bins,
            density=True,
            histtype="step",
            color="#F2A900",
            linewidth=1.35,
            label="Spatial Hard",
        )
        hist_axis.axvline(0, color="#555555", linestyle=(0, (4, 3)), linewidth=0.9)
        ymax = hist_axis.get_ylim()[1]
        hist_axis.vlines(
            hard_values,
            0,
            0.025 * ymax,
            color="#F2A900",
            alpha=0.38,
            linewidth=0.35,
        )
        hist_axis.set_xlim(-2, 2)
        hist_axis.set_xlabel("Similarity gap\nSame − candidate")
        hist_axis.set_ylabel("Density", rotation=270, labelpad=12)
        hist_axis.yaxis.set_label_position("right")
        hist_axis.yaxis.tick_right()
        hist_axis.set_title(
            "Identity-separation gaps\nHard median = %.3f"
            % float(np.median(hard_values)),
            pad=7,
        )
        hist_axis.spines["top"].set_visible(False)
        hist_axis.spines["left"].set_visible(False)
        if panel == 0:
            hist_axis.legend(frameon=False, fontsize=7, loc="upper right")

    colorbar = fig.colorbar(last_image, cax=colorbar_axis)
    colorbar.set_label("Relation-profile similarity (Pearson r)")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    figure_legend = [
        Patch(facecolor="none", edgecolor="#111111", linewidth=0.7, label="Correct identity"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#F2A900", markersize=4, label="Spatial Hard candidate"),
        Patch(facecolor="#BDBDBD", edgecolor="none", label="Missing / not computable"),
    ]
    fig.legend(
        handles=figure_legend,
        loc="lower center",
        bbox_to_anchor=(0.47, 0.012),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(output_dir / ("relation_profile_similarity_figure_f.%s" % suffix), **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal = load_formal_query_identities(args.query_export, args.fold)
    selected, pair_candidates = select_pair(args.fold_root, formal)
    shared_count, query_name, reference_name, query, reference, shared = selected
    identities, _ = select_and_order_identities(reference, shared, len(shared))
    hard, all_hard = hard_cells(reference, shared, identities, args.hard_k)
    matrices, long_table = compute_matrices(
        query, reference, identities, args.min_common_anchors
    )
    gaps = make_gap_table(matrices, identities, hard)
    draw_figure(matrices, identities, hard, gaps, args.output_dir)

    long_table.to_csv(args.output_dir / "full_similarity_matrix_long.csv", index=False)
    gaps.to_csv(args.output_dir / "similarity_gap_distribution.csv", index=False)
    pd.DataFrame(
        {
            "display_order": np.arange(len(identities)),
            "identity": identities,
            "hard_candidates": [";".join(all_hard[x]) for x in identities],
        }
    ).to_csv(args.output_dir / "identity_order_and_hard_candidates.csv", index=False)
    np.save(args.output_dir / "geometry_similarity_matrix.npy", matrices["geo"])
    np.save(args.output_dir / "activity_similarity_matrix.npy", matrices["act"])

    mode_summary = {}
    for display in ("Geometry", "Activity"):
        part = gaps.loc[gaps["mode"] == display]
        hard_part = part.loc[part["is_spatial_hard"]]
        mode_summary[display] = {
            "all_wrong_candidate_gap_median": float(
                part["same_minus_candidate"].median()
            ),
            "spatial_hard_gap_mean": float(
                hard_part["same_minus_candidate"].mean()
            ),
            "spatial_hard_gap_median": float(
                hard_part["same_minus_candidate"].median()
            ),
            "fraction_spatial_hard_gaps_positive": float(
                (hard_part["same_minus_candidate"] > 0).mean()
            ),
        }
    audit = {
        "dataset": "Atanas/SF 000776",
        "fold": args.fold,
        "query_record": query_name,
        "query_split": "held-out test",
        "reference_record": reference_name,
        "reference_split": "train",
        "pair_selection_rule": (
            "Within the fixed fold, maximize formal shared-identity coverage; "
            "break ties by query and reference record name."
        ),
        "pair_candidates_considered": len(pair_candidates),
        "shared_formal_identities": shared_count,
        "displayed_identities": len(identities),
        "identity_order_rule": (
            "Ascending normalized training-reference x, then y, z, then identity name."
        ),
        "similarity_definition": (
            "Pearson correlation between relation profiles aligned on common named "
            "anchors, excluding query and candidate identities."
        ),
        "gap_definition": "S_same - S_candidate for every wrong candidate",
        "hard_definition": (
            "Three spatially nearest wrong identities in the training reference "
            "from the full shared-identity pool."
        ),
        "shared_heatmap_color_scale": [-1.0, 1.0],
        "shared_gap_histogram_range": [-2.0, 2.0],
        "missing_color": "#BDBDBD",
        "missing_cells": {
            mode: int(np.isnan(matrix).sum()) for mode, matrix in matrices.items()
        },
        "summary": mode_summary,
        "query_export": str(args.query_export),
        "query_export_sha256": sha256(args.query_export),
    }
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
