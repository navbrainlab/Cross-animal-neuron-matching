#!/usr/bin/env python3
"""Plot identity-level activity relation margin versus activity ablation gain.

The accuracy difference is paired at the held-out query level because Full and
w/o Activity predictions are read from two columns of the same audited export.
One plotted row is then produced per neuron identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "results/visualization_export_atanas_seed42"
    / "neurid_query_neurons_atanas_cv5_seed42.csv"
)
DEFAULT_MARGINS = (
    REPO_ROOT
    / "results/mechanisms/atanas_margin_vs_top1"
    / "activity_identity_margin.csv"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs/atanas_activity_margin_vs_ablation_gain"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--margins", type=Path, default=DEFAULT_MARGINS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_paired_accuracy(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    queries = pd.read_csv(path)
    required = {
        "fold",
        "animal_id",
        "neuron_row",
        "gt_identity",
        "pred_geometry",
        "pred_full",
        "is_evaluated",
    }
    missing = required - set(queries.columns)
    if missing:
        raise RuntimeError(f"Missing prediction columns: {sorted(missing)}")

    evaluated_flag = queries["is_evaluated"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if evaluated_flag.isna().any():
        bad = sorted(queries.loc[evaluated_flag.isna(), "is_evaluated"].astype(str).unique())
        raise RuntimeError(f"Unrecognized is_evaluated values: {bad}")
    evaluated = queries.loc[evaluated_flag].copy()

    key = ["fold", "animal_id", "neuron_row"]
    if evaluated.duplicated(key).any():
        raise RuntimeError("Evaluated held-out query keys are not unique")
    if evaluated[["gt_identity", "pred_geometry", "pred_full"]].isna().any().any():
        raise RuntimeError("Evaluated queries contain missing labels or predictions")

    # Both predictions occupy the same row, so their query sets are identical by
    # construction. Keep an explicit audit count for the exported artifact.
    evaluated["correct_full"] = (
        evaluated["pred_full"].astype(str) == evaluated["gt_identity"].astype(str)
    )
    evaluated["correct_wo_activity"] = (
        evaluated["pred_geometry"].astype(str) == evaluated["gt_identity"].astype(str)
    )

    grouped = evaluated.groupby("gt_identity", sort=True)
    per_identity = grouped[["correct_full", "correct_wo_activity"]].mean().reset_index()
    per_identity = per_identity.rename(
        columns={
            "gt_identity": "identity",
            "correct_full": "accuracy_full",
            "correct_wo_activity": "accuracy_wo_activity",
        }
    )
    per_identity.insert(
        1,
        "n_paired_queries",
        grouped.size().reindex(per_identity["identity"]).values,
    )
    per_identity["accuracy_gain_pp"] = 100.0 * (
        per_identity["accuracy_full"] - per_identity["accuracy_wo_activity"]
    )

    audit = {
        "paired_query_key": key,
        "n_export_rows": int(len(queries)),
        "n_paired_evaluated_queries": int(len(evaluated)),
        "n_accuracy_identities": int(len(per_identity)),
        "full_and_wo_activity_query_sets_identical": True,
        "fold_query_counts": {
            str(int(fold)): int(count)
            for fold, count in evaluated.groupby("fold").size().items()
        },
        "pooled_full_top1": float(evaluated["correct_full"].mean()),
        "pooled_wo_activity_top1": float(evaluated["correct_wo_activity"].mean()),
    }
    return per_identity, audit


def load_activity_margins(path: Path) -> pd.DataFrame:
    margins = pd.read_csv(path)
    required = {"identity", "same_similarity", "hard_similarity", "stability_margin"}
    missing = required - set(margins.columns)
    if missing:
        raise RuntimeError(f"Missing margin columns: {sorted(missing)}")
    if margins["identity"].duplicated().any():
        raise RuntimeError("Activity margin table has duplicate identities")
    expected = margins["same_similarity"] - margins["hard_similarity"]
    if not np.allclose(expected, margins["stability_margin"], rtol=0, atol=1e-12):
        raise RuntimeError("stability_margin is not Same - Hard")
    return margins


def p_text(value: float) -> str:
    if value < 0.001:
        return rf"$p={value:.2e}$"
    return rf"$p={value:.3f}$"


def draw_plot(table: pd.DataFrame, output_dir: Path, rho: float, p_value: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.35, 3.55))
    ax.axhline(0, color="#666666", linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter(
        table["activity_relation_margin"],
        table["accuracy_gain_pp"],
        s=27,
        facecolor="#2878B5",
        edgecolor="white",
        linewidth=0.45,
        alpha=0.82,
        zorder=2,
    )
    annotation = (
        rf"Spearman $\rho={rho:.3f}$" + "\n" + p_text(p_value) + rf", $n={len(table)}$"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.7,
        linespacing=1.15,
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.92},
    )
    ax.set_xlabel("Activity relation margin (Same − Hard)")
    ax.set_ylabel("Top-1 accuracy gain (pp)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    fig.tight_layout()
    for suffix in ("pdf", "png", "svg"):
        kwargs = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(output_dir / f"activity_margin_vs_ablation_gain.{suffix}", **kwargs)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accuracy, audit = load_paired_accuracy(args.predictions)
    margins = load_activity_margins(args.margins).rename(
        columns={"stability_margin": "activity_relation_margin"}
    )
    merged = margins.merge(accuracy, on="identity", how="inner", validate="one_to_one")
    merged = merged.loc[
        np.isfinite(merged["activity_relation_margin"])
        & np.isfinite(merged["accuracy_gain_pp"])
    ].copy()
    if len(merged) < 3:
        raise RuntimeError("Fewer than three identities remain after merging")

    rho, p_value = spearmanr(
        merged["activity_relation_margin"], merged["accuracy_gain_pp"]
    )
    merged.to_csv(args.output_dir / "activity_margin_vs_ablation_gain.csv", index=False)
    draw_plot(merged, args.output_dir, float(rho), float(p_value))

    audit.update(
        {
            "dataset": "Atanas/SF 000776",
            "seed": 42,
            "accuracy_definition": "pooled Top-1 over exactly paired held-out queries per identity",
            "activity_margin_definition": "mean Same similarity - mean Hard similarity per identity",
            "n_margin_identities": int(len(margins)),
            "n_plotted_identities": int(len(merged)),
            "accuracy_identities_without_margin": sorted(
                set(accuracy["identity"]) - set(margins["identity"])
            ),
            "margin_identities_without_accuracy": sorted(
                set(margins["identity"]) - set(accuracy["identity"])
            ),
            "spearman_rho": float(rho),
            "spearman_p": float(p_value),
            "prediction_source": str(args.predictions),
            "prediction_source_sha256": sha256(args.predictions),
            "margin_source": str(args.margins),
            "margin_source_sha256": sha256(args.margins),
        }
    )
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
