#!/usr/bin/env python3
"""Draw paired Geometry/Activity relation-profile similarity heatmaps.

The similarity calculation and spatial hard-negative definition are imported
from the Same-Hard mechanism analysis so the heatmap is a direct visualization
of the same quantity rather than a new score.
"""

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
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mechanisms.analyze_cross_animal_population_relation_mechanism import (
    common_identities,
    hard_negative_ids,
    load_one_worm,
    relation_similarity,
)


DEFAULT_FOLD_ROOT = (
    REPO_ROOT / "Data/Atanas_SF_unified_000776/cv5_grouped_v1/fold_0"
)
DEFAULT_QUERY_EXPORT = (
    REPO_ROOT
    / "NeuRID_reproducibility/results/visualization_export_atanas_seed42"
    / "neurid_query_neurons_atanas_cv5_seed42.csv"
)
DEFAULT_GEO_RAW = (
    REPO_ROOT
    / "runs/mprt_v1_1/mechanism_population_relation/atanas_geo"
    / "panelA_same_hard_random_raw.csv"
)
DEFAULT_ACT_RAW = (
    REPO_ROOT
    / "runs/mprt_v1_1/mechanism_population_relation/atanas_act"
    / "panelA_same_hard_random_raw.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs/mprt_v1_1/mechanism_population_relation"
    / "atanas_relation_profile_heatmap"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--query-export", type=Path, default=DEFAULT_QUERY_EXPORT)
    parser.add_argument("--geo-raw", type=Path, default=DEFAULT_GEO_RAW)
    parser.add_argument("--act-raw", type=Path, default=DEFAULT_ACT_RAW)
    parser.add_argument("--n-identities", type=int, default=18)
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


def clean_record_name(path: Path) -> str:
    return path.resolve().stem


def load_formal_query_identities(path: Path, fold: int) -> dict:
    table = pd.read_csv(path)
    required = {"fold", "animal_id", "gt_identity", "is_evaluated"}
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("Missing query-export columns: %s" % sorted(missing))
    flag = table["is_evaluated"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if flag.isna().any():
        raise RuntimeError("Unrecognized is_evaluated value")
    table = table.loc[flag & (table["fold"] == fold)].copy()
    return {
        str(animal): set(part["gt_identity"].astype(str))
        for animal, part in table.groupby("animal_id")
    }


def load_unique_worms(paths) -> list:
    worms = []
    seen = set()
    for link in sorted(paths):
        resolved = link.resolve()
        name = clean_record_name(resolved)
        if name in seen:
            continue
        worm = load_one_worm(str(resolved), relation_mode="act")
        if worm is None:
            raise RuntimeError("Could not load record: %s" % resolved)
        worms.append(worm)
        seen.add(name)
    return worms


def select_pair(fold_root: Path, formal_ids: dict):
    query_worms = load_unique_worms((fold_root / "test").glob("*.npz"))
    reference_worms = load_unique_worms((fold_root / "train").glob("*.npz"))
    candidates = []
    for query in query_worms:
        if query.name not in formal_ids:
            continue
        for reference in reference_worms:
            shared = sorted(
                set(common_identities(query, reference)) & formal_ids[query.name]
            )
            candidates.append((len(shared), query.name, reference.name, query, reference, shared))
    if not candidates:
        raise RuntimeError("No held-out query/training-reference pair could be formed")
    # Maximize only formal shared-identity coverage; deterministic name tie-breaks.
    candidates.sort(key=lambda row: (-row[0], row[1], row[2]))
    return candidates[0], candidates


def select_and_order_identities(reference, shared: list, n_identities: int) -> tuple:
    if len(shared) < n_identities:
        raise RuntimeError("Requested more identities than the selected pair shares")
    coordinates = {
        identity: reference.xyz[reference.label_to_idx[identity]] for identity in shared
    }
    centroid = np.mean(np.stack([coordinates[x] for x in shared]), axis=0)
    nearest = sorted(
        shared,
        key=lambda identity: (
            float(np.linalg.norm(coordinates[identity] - centroid)),
            identity,
        ),
    )[:n_identities]
    ordered = sorted(
        nearest,
        key=lambda identity: tuple(float(x) for x in coordinates[identity]) + (identity,),
    )
    return ordered, centroid


def compute_matrices(query, reference, identities: list, min_common_anchors: int):
    matrices = {}
    rows = []
    pair_anchors = set(common_identities(query, reference))
    for mode in ("geo", "act"):
        matrix = np.full((len(identities), len(identities)), np.nan, dtype=float)
        for i, query_id in enumerate(identities):
            for j, candidate_id in enumerate(identities):
                value = relation_similarity(
                    query,
                    query_id,
                    reference,
                    candidate_id,
                    mode,
                    min_common_anchors,
                )
                matrix[i, j] = value
                anchors = pair_anchors - {query_id, candidate_id}
                rows.append(
                    {
                        "mode": "Geometry" if mode == "geo" else "Activity",
                        "query_record": query.name,
                        "reference_record": reference.name,
                        "query_identity": query_id,
                        "candidate_identity": candidate_id,
                        "similarity": value,
                        "n_common_anchors": len(anchors),
                    }
                )
        matrices[mode] = matrix
    return matrices, pd.DataFrame(rows)


def hard_cells(reference, shared: list, identities: list, hard_k: int):
    identity_to_column = {identity: index for index, identity in enumerate(identities)}
    cells = []
    full = {}
    for row, query_id in enumerate(identities):
        hard = hard_negative_ids(reference, query_id, shared, hard_k)
        full[query_id] = hard
        for candidate_id in hard:
            if candidate_id in identity_to_column:
                cells.append((row, identity_to_column[candidate_id], query_id, candidate_id))
    return cells, full


def validate_against_panel_a(
    long_table: pd.DataFrame,
    raw_path: Path,
    query_name: str,
    reference_name: str,
    mode_name: str,
) -> dict:
    raw = pd.read_csv(raw_path)
    raw = raw.loc[
        (raw["source_worm"].astype(str) == query_name)
        & (raw["target_worm"].astype(str) == reference_name)
        & raw["group"].isin(["Same", "Hard"])
    ].copy()
    calculated = long_table.loc[long_table["mode"] == mode_name]
    checked = 0
    maximum_error = 0.0
    for _, row in raw.iterrows():
        match = calculated.loc[
            (calculated["query_identity"] == str(row["query_id"]))
            & (calculated["candidate_identity"] == str(row["candidate_id"]))
        ]
        if len(match) == 0:
            continue
        error = abs(float(match.iloc[0]["similarity"]) - float(row["similarity"]))
        maximum_error = max(maximum_error, error)
        checked += 1
    if checked == 0 or maximum_error > 1e-12:
        raise RuntimeError(
            "%s heatmap failed Panel-A equivalence: checked=%d, max_error=%g"
            % (mode_name, checked, maximum_error)
        )
    return {"checked_cells": checked, "maximum_absolute_error": maximum_error}


def draw_heatmaps(matrices: dict, identities: list, hard: list, output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
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
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.75), sharex=True, sharey=True)
    last_image = None
    for panel, (axis, mode, title) in enumerate(
        zip(axes, ("geo", "act"), ("Geometry", "Activity"))
    ):
        last_image = axis.imshow(
            np.ma.masked_invalid(matrices[mode]),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title, pad=7)
        axis.set_xticks(np.arange(len(identities)))
        axis.set_yticks(np.arange(len(identities)))
        axis.set_xticklabels(identities, rotation=90, ha="center", va="top", fontsize=6.7)
        axis.set_yticklabels(identities, fontsize=6.7)
        axis.tick_params(length=0, pad=1.7)
        for index in range(len(identities)):
            axis.add_patch(
                Rectangle(
                    (index - 0.47, index - 0.47),
                    0.94,
                    0.94,
                    fill=False,
                    edgecolor="#111111",
                    linewidth=1.15,
                    zorder=4,
                )
            )
        for row, column, _, _ in hard:
            axis.add_patch(
                Rectangle(
                    (column - 0.40, row - 0.40),
                    0.80,
                    0.80,
                    fill=False,
                    edgecolor="#F2A900",
                    linewidth=1.15,
                    zorder=5,
                )
            )
        axis.text(
            -0.11,
            1.03,
            chr(ord("A") + panel),
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    axes[0].set_ylabel("Held-out query identity")
    for axis in axes:
        axis.set_xlabel("Training-reference candidate identity", labelpad=5)
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.25, top=0.92, wspace=0.10)
    colorbar_axis = fig.add_axes([0.885, 0.25, 0.018, 0.67])
    colorbar = fig.colorbar(last_image, cax=colorbar_axis)
    colorbar.set_label("Relation-profile similarity (Pearson r)")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    legend = [
        Patch(facecolor="none", edgecolor="#111111", linewidth=1.15, label="Correct identity"),
        Patch(facecolor="none", edgecolor="#F2A900", linewidth=1.15, label="Spatial Hard candidate"),
        Patch(facecolor="#BDBDBD", edgecolor="none", label="Missing / not computable"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.47, -0.01),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(output_dir / ("relation_profile_similarity_heatmaps.%s" % suffix), **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    formal = load_formal_query_identities(args.query_export, args.fold)
    selected, pair_candidates = select_pair(args.fold_root, formal)
    shared_count, query_name, reference_name, query, reference, shared = selected
    identities, centroid = select_and_order_identities(
        reference, shared, args.n_identities
    )
    hard, all_hard = hard_cells(reference, shared, identities, args.hard_k)
    matrices, long_table = compute_matrices(
        query, reference, identities, args.min_common_anchors
    )
    validation = {
        "Geometry": validate_against_panel_a(
            long_table, args.geo_raw, query_name, reference_name, "Geometry"
        ),
        "Activity": validate_against_panel_a(
            long_table, args.act_raw, query_name, reference_name, "Activity"
        ),
    }
    draw_heatmaps(matrices, identities, hard, args.output_dir)

    identity_rows = []
    for order, identity in enumerate(identities):
        xyz = reference.xyz[reference.label_to_idx[identity]]
        identity_rows.append(
            {
                "display_order": order,
                "identity": identity,
                "reference_x_normalized": float(xyz[0]),
                "reference_y_normalized": float(xyz[1]),
                "reference_z_normalized": float(xyz[2]),
                "distance_to_reference_shared_centroid": float(
                    np.linalg.norm(xyz - centroid)
                ),
                "hard_candidates_full_common_pool": ";".join(all_hard[identity]),
                "displayed_hard_candidates": ";".join(
                    candidate
                    for candidate in all_hard[identity]
                    if candidate in set(identities)
                ),
            }
        )
    pd.DataFrame(
        identity_rows,
        columns=[
            "display_order",
            "identity",
            "reference_x_normalized",
            "reference_y_normalized",
            "reference_z_normalized",
            "distance_to_reference_shared_centroid",
            "hard_candidates_full_common_pool",
            "displayed_hard_candidates",
        ],
    ).to_csv(
        args.output_dir / "displayed_identities.csv", index=False
    )
    long_table.to_csv(
        args.output_dir / "similarity_values.csv",
        index=False,
        columns=[
            "mode",
            "query_record",
            "reference_record",
            "query_identity",
            "candidate_identity",
            "similarity",
            "n_common_anchors",
        ],
    )

    audit = {
        "dataset": "Atanas/SF 000776",
        "fold": args.fold,
        "query_record": query_name,
        "query_split": "held-out test",
        "reference_record": reference_name,
        "reference_split": "train",
        "pair_selection_rule": (
            "Within fold %d, maximize the number of formal held-out query identities "
            "shared with one training reference record; break ties by query and "
            "reference record name." % args.fold
        ),
        "pair_candidates_considered": len(pair_candidates),
        "shared_formal_identities": shared_count,
        "subset_selection_rule": (
            "Choose the %d shared identities nearest the centroid of all shared "
            "identities in the normalized 3D coordinates of the training reference."
            % args.n_identities
        ),
        "identity_order_rule": (
            "Ascending normalized reference x, then y, z, then identity name."
        ),
        "displayed_identities": identities,
        "n_displayed_identities": len(identities),
        "similarity_definition": (
            "Pearson correlation between relation profiles aligned on common named "
            "anchors, excluding the query and candidate identities themselves."
        ),
        "geometry_relation": "negative Euclidean distance after per-recording normalization",
        "activity_relation": "neuron-neuron activity correlation",
        "min_common_anchors": args.min_common_anchors,
        "hard_definition": (
            "Three spatially nearest wrong identities in the training reference, "
            "selected from the full shared-identity pool independently of relation similarity."
        ),
        "hard_k": args.hard_k,
        "displayed_hard_cells": len(hard),
        "rows_with_at_least_one_displayed_hard": len(set(row for row, _, _, _ in hard)),
        "shared_color_scale": [-1.0, 1.0],
        "missing_color": "#BDBDBD",
        "missing_cells": {
            mode: int(np.isnan(matrix).sum()) for mode, matrix in matrices.items()
        },
        "panel_a_equivalence": validation,
        "fold_root": str(args.fold_root),
        "query_export": str(args.query_export),
        "query_export_sha256": sha256(args.query_export),
        "geo_raw": str(args.geo_raw),
        "geo_raw_sha256": sha256(args.geo_raw),
        "act_raw": str(args.act_raw),
        "act_raw_sha256": sha256(args.act_raw),
        "query_record_source_npz": query.path,
        "reference_record_source_npz": reference.path,
        "query_fold_materialization": str(
            next(
                path
                for path in sorted((args.fold_root / "test").glob("*.npz"))
                if clean_record_name(path) == query_name
            )
        ),
        "reference_fold_materialization": str(
            next(
                path
                for path in sorted((args.fold_root / "train").glob("*.npz"))
                if clean_record_name(path) == reference_name
            )
        ),
    }
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
