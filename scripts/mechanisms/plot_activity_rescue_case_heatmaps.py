#!/usr/bin/env python3
"""Find and plot real Full-rescues linked to activity relation specificity."""

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
from matplotlib.lines import Line2D
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
from scripts.mechanisms.plot_relation_profile_similarity_heatmaps import (
    validate_against_panel_a,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "data/atanas"
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "results/visualization_export_atanas_seed42"
    / "neurid_query_neurons_atanas_cv5_seed42.csv"
)
DEFAULT_GEO_RAW = (
    REPO_ROOT
    / "runs/atanas_geo_relations"
    / "panelA_same_hard_random_raw.csv"
)
DEFAULT_ACT_RAW = (
    REPO_ROOT
    / "runs/atanas_activity_relations"
    / "panelA_same_hard_random_raw.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs/atanas_activity_rescue_cases"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--geo-raw", type=Path, default=DEFAULT_GEO_RAW)
    parser.add_argument("--act-raw", type=Path, default=DEFAULT_ACT_RAW)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--hard-k", type=int, default=3)
    parser.add_argument("--min-common-anchors", type=int, default=8)
    parser.add_argument("--n-display", type=int, default=6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_name(path: Path) -> str:
    return path.resolve().stem


def load_worms(paths) -> dict:
    output = {}
    for link in sorted(paths):
        name = record_name(link)
        if name in output:
            continue
        worm = load_one_worm(str(link.resolve()), relation_mode="act")
        if worm is None:
            raise RuntimeError("Could not load %s" % link)
        output[name] = worm
    return output


def load_predictions(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    required = {
        "fold",
        "animal_id",
        "neuron_row",
        "gt_identity",
        "pred_geometry",
        "pred_full",
        "is_evaluated",
    }
    missing = required - set(table.columns)
    if missing:
        raise RuntimeError("Missing prediction columns: %s" % sorted(missing))
    flag = table["is_evaluated"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if flag.isna().any():
        raise RuntimeError("Unrecognized is_evaluated value")
    table = table.loc[flag].copy()
    if table.duplicated(["fold", "animal_id", "neuron_row"]).any():
        raise RuntimeError("Formal query keys are not unique")
    table["gt_identity"] = table["gt_identity"].astype(str)
    table["pred_geometry"] = table["pred_geometry"].astype(str)
    table["pred_full"] = table["pred_full"].astype(str)
    table["wo_activity_correct"] = table["pred_geometry"] == table["gt_identity"]
    table["full_correct"] = table["pred_full"] == table["gt_identity"]
    return table


def select_references(data_root: Path, predictions: pd.DataFrame, folds: list):
    selections = {}
    audits = []
    worm_cache = {}
    for fold in folds:
        fold_root = data_root / ("fold_%d" % fold)
        queries = load_worms((fold_root / "test").glob("*.npz"))
        references = load_worms((fold_root / "train").glob("*.npz"))
        worm_cache[fold] = {"query": queries, "reference": references}
        fold_predictions = predictions.loc[predictions["fold"] == fold]
        for query_name, part in fold_predictions.groupby("animal_id"):
            query_name = str(query_name)
            if query_name not in queries:
                raise RuntimeError("Missing fold query record %s" % query_name)
            formal_ids = set(part["gt_identity"])
            candidates = []
            for reference_name, reference in references.items():
                shared = sorted(
                    formal_ids
                    & set(common_identities(queries[query_name], reference))
                )
                candidates.append((len(shared), reference_name, reference, shared))
            candidates.sort(key=lambda row: (-row[0], row[1]))
            coverage, reference_name, reference, shared = candidates[0]
            selections[(fold, query_name)] = {
                "query": queries[query_name],
                "reference": reference,
                "reference_name": reference_name,
                "shared": shared,
            }
            audits.append(
                {
                    "fold": fold,
                    "query_record": query_name,
                    "reference_record": reference_name,
                    "formal_queries": len(formal_ids),
                    "shared_formal_identities": coverage,
                    "training_references_considered": len(candidates),
                }
            )
    return selections, pd.DataFrame(audits), worm_cache


def similarity_set(query, reference, true_id: str, hard: list, min_anchors: int, mode: str):
    same = relation_similarity(
        query, true_id, reference, true_id, mode, min_anchors
    )
    hard_scores = [
        relation_similarity(query, true_id, reference, candidate, mode, min_anchors)
        for candidate in hard
    ]
    values = np.asarray(hard_scores, dtype=float)
    if not np.isfinite(same) or not np.isfinite(values).all():
        return None
    return {
        "same": float(same),
        "hard": [float(value) for value in values],
        "mean_margin": float(same - values.mean()),
        "strongest_margin": float(same - values.max()),
        "max_absolute_gap": float(np.max(np.abs(same - values))),
    }


def evaluate_rescues(
    predictions: pd.DataFrame,
    selections: dict,
    hard_k: int,
    min_common_anchors: int,
) -> pd.DataFrame:
    rescued = predictions.loc[(~predictions["wo_activity_correct"]) & predictions["full_correct"]]
    rows = []
    for _, prediction in rescued.iterrows():
        fold = int(prediction["fold"])
        query_name = str(prediction["animal_id"])
        true_id = str(prediction["gt_identity"])
        selected = selections[(fold, query_name)]
        shared = selected["shared"]
        if true_id not in shared:
            continue
        hard = hard_negative_ids(
            selected["reference"], true_id, shared, hard_k
        )
        if len(hard) != hard_k:
            continue
        geometry = similarity_set(
            selected["query"],
            selected["reference"],
            true_id,
            hard,
            min_common_anchors,
            "geo",
        )
        activity = similarity_set(
            selected["query"],
            selected["reference"],
            true_id,
            hard,
            min_common_anchors,
            "act",
        )
        if geometry is None or activity is None:
            continue
        row = {
            "fold": fold,
            "query_record": query_name,
            "neuron_row": int(prediction["neuron_row"]),
            "identity": true_id,
            "reference_record": selected["reference_name"],
            "wo_activity_prediction": str(prediction["pred_geometry"]),
            "full_prediction": str(prediction["pred_full"]),
            "hard_candidates": ";".join(hard),
            "wo_activity_prediction_is_hard": str(prediction["pred_geometry"]) in hard,
            "geometry_same": geometry["same"],
            "geometry_hard_scores": ";".join("%.12g" % x for x in geometry["hard"]),
            "geometry_mean_margin": geometry["mean_margin"],
            "geometry_strongest_margin": geometry["strongest_margin"],
            "geometry_max_absolute_hard_gap": geometry["max_absolute_gap"],
            "activity_same": activity["same"],
            "activity_hard_scores": ";".join("%.12g" % x for x in activity["hard"]),
            "activity_mean_margin": activity["mean_margin"],
            "activity_strongest_margin": activity["strongest_margin"],
            "activity_max_absolute_hard_gap": activity["max_absolute_gap"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def choose_cases(cases: pd.DataFrame, n_display: int):
    if len(cases) == 0:
        raise RuntimeError("No eligible Full-rescue queries")
    geometry_threshold = float(cases["geometry_max_absolute_hard_gap"].quantile(0.25))
    cases = cases.copy()
    cases["geometry_ambiguous_bottom_quartile"] = (
        cases["geometry_max_absolute_hard_gap"] <= geometry_threshold
    )
    cases["activity_beats_strongest_hard"] = cases["activity_strongest_margin"] > 0
    cases["strict_case"] = (
        cases["wo_activity_prediction_is_hard"]
        & cases["geometry_ambiguous_bottom_quartile"]
        & cases["activity_beats_strongest_hard"]
    )

    pair_keys = ["fold", "query_record", "reference_record"]
    grouped = cases.groupby(pair_keys)
    pair_counts = grouped[
        ["strict_case", "wo_activity_prediction_is_hard"]
    ].sum().reset_index()
    pair_counts = pair_counts.rename(
        columns={
            "strict_case": "strict_cases",
            "wo_activity_prediction_is_hard": "hard_prediction_rescues",
        }
    )
    pair_sizes = grouped.size().reset_index(name="eligible_rescues")
    pair_counts = pair_counts.merge(pair_sizes, on=pair_keys, how="left")
    pair_counts = pair_counts.sort_values(
        ["strict_cases", "hard_prediction_rescues", "eligible_rescues", "fold", "query_record", "reference_record"],
        ascending=[False, False, False, True, True, True],
    )
    chosen_pair = pair_counts.iloc[0]
    pair_cases = cases.loc[
        (cases["fold"] == int(chosen_pair["fold"]))
        & (cases["query_record"] == str(chosen_pair["query_record"]))
        & (cases["reference_record"] == str(chosen_pair["reference_record"]))
    ].copy()
    pair_cases["selection_tier"] = np.where(
        pair_cases["strict_case"],
        0,
        np.where(
            pair_cases["wo_activity_prediction_is_hard"]
            & pair_cases["activity_beats_strongest_hard"],
            1,
            np.where(pair_cases["activity_beats_strongest_hard"], 2, 3),
        ),
    )
    pair_cases = pair_cases.sort_values(
        [
            "selection_tier",
            "activity_strongest_margin",
            "geometry_max_absolute_hard_gap",
            "identity",
        ],
        ascending=[True, False, True, True],
    )
    selected = pair_cases.head(min(n_display, len(pair_cases))).copy()
    return cases, selected, pair_counts, geometry_threshold


def build_case_matrices(selected: pd.DataFrame, selections: dict, min_anchors: int):
    first = selected.iloc[0]
    fold = int(first["fold"])
    query_name = str(first["query_record"])
    pair = selections[(fold, query_name)]
    candidates = set(selected["identity"])
    for value in selected["hard_candidates"]:
        candidates.update(str(value).split(";"))
    candidates = sorted(candidates)
    rows = list(selected["identity"])
    matrices = {}
    long_rows = []
    for mode, display in (("geo", "Geometry"), ("act", "Activity")):
        matrix = np.full((len(rows), len(candidates)), np.nan, dtype=float)
        for i, query_id in enumerate(rows):
            for j, candidate_id in enumerate(candidates):
                value = relation_similarity(
                    pair["query"],
                    query_id,
                    pair["reference"],
                    candidate_id,
                    mode,
                    min_anchors,
                )
                matrix[i, j] = value
                long_rows.append(
                    {
                        "mode": display,
                        "query_identity": query_id,
                        "candidate_identity": candidate_id,
                        "similarity": value,
                    }
                )
        matrices[mode] = matrix
    return rows, candidates, matrices, pd.DataFrame(long_rows)


def draw_case_heatmaps(
    selected: pd.DataFrame,
    rows: list,
    candidates: list,
    matrices: dict,
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.5,
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
    width = max(11.0, 0.40 * len(candidates) * 2 + 3.2)
    height = max(4.6, 0.62 * len(rows) + 2.4)
    fig, axes = plt.subplots(1, 2, figsize=(width, height), sharex=True, sharey=True)
    candidate_index = {identity: index for index, identity in enumerate(candidates)}
    last_image = None
    for panel, (axis, mode, title) in enumerate(
        zip(axes, ("geo", "act"), ("Geometry", "Activity"))
    ):
        last_image = axis.imshow(
            np.ma.masked_invalid(matrices[mode]),
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="auto",
        )
        axis.set_title(title, pad=7)
        axis.set_xticks(np.arange(len(candidates)))
        axis.set_xticklabels(
            candidates, rotation=90, ha="center", va="top", fontsize=9.5
        )
        axis.set_yticks(np.arange(len(rows)))
        axis.set_yticklabels(rows, fontsize=10.5)
        axis.tick_params(length=0, pad=3)
        for row_index, (_, case) in enumerate(selected.iterrows()):
            correct_column = candidate_index[str(case["identity"])]
            axis.add_patch(
                Rectangle(
                    (correct_column - 0.47, row_index - 0.43),
                    0.94,
                    0.86,
                    fill=False,
                    edgecolor="#111111",
                    linewidth=1.35,
                    zorder=4,
                )
            )
            for hard_id in str(case["hard_candidates"]).split(";"):
                hard_column = candidate_index[hard_id]
                axis.add_patch(
                    Rectangle(
                        (hard_column - 0.39, row_index - 0.35),
                        0.78,
                        0.70,
                        fill=False,
                        edgecolor="#F2A900",
                        linewidth=1.15,
                        zorder=5,
                    )
                )
            wrong_column = candidate_index.get(str(case["wo_activity_prediction"]))
            if wrong_column is not None:
                axis.plot(
                    wrong_column,
                    row_index,
                    marker="x",
                    markersize=6.5,
                    markeredgewidth=1.5,
                    color="#7A3E9D",
                    linestyle="none",
                    zorder=6,
                )
        axis.text(
            -0.07,
            1.04,
            chr(ord("A") + panel),
            transform=axis.transAxes,
            fontsize=15,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        axis.set_xlabel("Training-reference candidate identity", labelpad=5)
    axes[0].set_ylabel("Rescued held-out query identity")
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.30, top=0.88, wspace=0.08)
    colorbar_axis = fig.add_axes([0.91, 0.30, 0.013, 0.58])
    colorbar = fig.colorbar(last_image, cax=colorbar_axis)
    colorbar.set_label("Relation-profile similarity (Pearson r)")
    colorbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    colorbar.ax.tick_params(labelsize=10)
    legend = [
        Patch(facecolor="none", edgecolor="#111111", linewidth=1.35, label="Correct / Full prediction"),
        Patch(facecolor="none", edgecolor="#F2A900", linewidth=1.15, label="Spatial Hard candidate"),
        Line2D([0], [0], marker="x", color="#7A3E9D", linestyle="none", markersize=6.5, label="w/o Activity prediction"),
        Patch(facecolor="#BDBDBD", edgecolor="none", label="Missing / not computable"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.49, 0.01),
        ncol=4,
        frameon=False,
        fontsize=10.5,
    )
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(output_dir / ("activity_rescue_case_heatmaps.%s" % suffix), **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(args.predictions)
    selections, reference_audit, _ = select_references(
        args.data_root, predictions, args.folds
    )
    cases = evaluate_rescues(
        predictions,
        selections,
        args.hard_k,
        args.min_common_anchors,
    )
    cases, selected, pair_counts, geometry_threshold = choose_cases(
        cases, args.n_display
    )
    rows, candidates, matrices, long_table = build_case_matrices(
        selected, selections, args.min_common_anchors
    )
    selected_pair = selected.iloc[0]
    panel_a_validation = {
        "Geometry": validate_against_panel_a(
            long_table,
            args.geo_raw,
            str(selected_pair["query_record"]),
            str(selected_pair["reference_record"]),
            "Geometry",
        ),
        "Activity": validate_against_panel_a(
            long_table,
            args.act_raw,
            str(selected_pair["query_record"]),
            str(selected_pair["reference_record"]),
            "Activity",
        ),
    }
    draw_case_heatmaps(selected, rows, candidates, matrices, args.output_dir)

    total_queries = int(len(predictions))
    total_rescued = int(((~predictions["wo_activity_correct"]) & predictions["full_correct"]).sum())
    eligible = int(len(cases))
    hard_rescues = int(cases["wo_activity_prediction_is_hard"].sum())
    activity_strong = int(cases["activity_beats_strongest_hard"].sum())
    strict = int(cases["strict_case"].sum())
    summary = {
        "dataset": "Atanas/SF 000776",
        "seed": 42,
        "formal_held_out_queries": total_queries,
        "full_correct_wo_activity_wrong_queries": total_rescued,
        "eligible_rescues_with_fixed_training_reference_and_three_hard_candidates": eligible,
        "rescues_where_wo_activity_prediction_is_spatial_hard": hard_rescues,
        "fraction_hard_prediction_among_eligible_rescues": hard_rescues / eligible,
        "rescues_where_activity_beats_strongest_hard": activity_strong,
        "fraction_activity_beats_strongest_hard_among_eligible_rescues": activity_strong / eligible,
        "geometry_ambiguity_definition": (
            "Bottom quartile of max_j |S_same - S_hard,j| among eligible rescues"
        ),
        "geometry_ambiguity_threshold": geometry_threshold,
        "strict_case_definition": (
            "w/o Activity prediction is spatial Hard AND geometry ambiguity is in "
            "the bottom quartile AND activity strongest-hard margin is positive"
        ),
        "strict_cases": strict,
        "fraction_strict_among_eligible_rescues": strict / eligible,
        "fraction_strict_among_all_full_rescues": strict / total_rescued,
        "reference_selection_rule": (
            "For each held-out query record, select the same-fold training record "
            "with maximum formal shared-identity coverage; break ties by record name."
        ),
        "case_pair_selection_rule": (
            "Choose the fixed query-reference pair with the most strict cases; then "
            "rank cases by strict tier, descending activity strongest-hard margin, "
            "ascending geometry max absolute hard gap, and identity name."
        ),
        "selected_fold": int(selected_pair["fold"]),
        "selected_query_record": str(selected_pair["query_record"]),
        "selected_reference_record": str(selected_pair["reference_record"]),
        "selected_query_identities": rows,
        "displayed_candidate_identities": candidates,
        "n_selected_cases": len(selected),
        "hard_k": args.hard_k,
        "min_common_anchors": args.min_common_anchors,
        "same_hard_mean_margin_definition": "S_same - mean_j S_hard,j",
        "strongest_hard_margin_definition": "S_same - max_j S_hard,j",
        "shared_color_scale": [-1.0, 1.0],
        "missing_cells": {
            mode: int(np.isnan(matrix).sum()) for mode, matrix in matrices.items()
        },
        "panel_a_equivalence": panel_a_validation,
        "prediction_source": str(args.predictions),
        "prediction_source_sha256": sha256(args.predictions),
        "geo_same_hard_source": str(args.geo_raw),
        "geo_same_hard_source_sha256": sha256(args.geo_raw),
        "activity_same_hard_source": str(args.act_raw),
        "activity_same_hard_source_sha256": sha256(args.act_raw),
    }

    cases.to_csv(args.output_dir / "eligible_rescue_cases.csv", index=False)
    selected.to_csv(args.output_dir / "selected_cases.csv", index=False)
    pair_counts.to_csv(args.output_dir / "pair_case_counts.csv", index=False)
    reference_audit.to_csv(args.output_dir / "fixed_reference_selection.csv", index=False)
    long_table.to_csv(args.output_dir / "displayed_similarity_values.csv", index=False)
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
