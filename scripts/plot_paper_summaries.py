#!/usr/bin/env python3
"""Regenerate paper summary figures from the frozen public CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
COLORS = {"Atanas": "#3973ac", "Kato": "#8e2f4f"}
METHOD_COLORS = {
    "NeurID (single medoid)": "#3973ac",
    "NeurID (population atlas)": "#8e2f4f",
}


def activity_window(output: Path) -> None:
    table = pd.read_csv(ROOT / "results/activity_window_cv5_seed42/summary.csv")
    table = table[table["duration_key"].isin(["15s", "60s", "120s", "240s", "480s", "900s", "full"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for dataset, display in (("atanas", "Atanas"), ("rld", "Kato")):
        rows = table[(table["dataset"] == dataset) & (table["duration_key"] != "full")].copy()
        x = rows["duration_seconds"].to_numpy(float)
        color = COLORS[display]
        axes[0].errorbar(x, 100 * rows["top1_real_mean"], 100 * rows["top1_real_sd"], marker="o", color=color, label=display)
        full = float(table[(table["dataset"] == dataset) & (table["duration_key"] == "full")]["top1_real_mean"].iloc[0])
        axes[0].axhline(100 * full, color=color, linestyle="--", alpha=0.65)
        axes[1].errorbar(x, 100 * rows["prediction_pair_agreement_mean"], 100 * rows["prediction_pair_agreement_sd"], marker="o", color=color, label=display)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Activity duration (s)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[1].set_ylabel("Prediction agreement (%)")
    axes[0].set_title("A")
    axes[1].set_title("B")
    fig.savefig(output / "activity_window_summary.pdf")
    fig.savefig(output / "activity_window_summary.png", dpi=300)
    plt.close(fig)


def per_animal(output: Path) -> None:
    table = pd.read_csv(ROOT / "results/per_animal/atanas_kato_per_animal_method_counts.csv")
    methods = list(METHOD_COLORS)
    table = table[table["method"].isin(methods)].copy()
    table["full"] = table["n_correct"] / table["n_eval"]
    table["shared"] = table["n_correct_shared"] / table["n_shared"].replace(0, np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), constrained_layout=True, sharey=True)
    rng = np.random.default_rng(42)
    panel = 0
    for row, cohort in enumerate(("full", "shared")):
        for column, dataset in enumerate(("Atanas", "Kato")):
            axis = axes[row, column]
            arrays = [table[(table["dataset"] == dataset) & (table["method"] == method)][cohort].dropna().to_numpy() * 100 for method in methods]
            boxes = axis.boxplot(arrays, positions=[1, 2], widths=0.52, patch_artist=True, showfliers=False)
            for patch, method in zip(boxes["boxes"], methods):
                patch.set(facecolor="white", edgecolor=METHOD_COLORS[method], linewidth=1.5)
            for index, (values, method) in enumerate(zip(arrays, methods), start=1):
                jitter = rng.uniform(-0.12, 0.12, size=len(values))
                axis.scatter(index + jitter, values, s=14, facecolors="none", edgecolors=METHOD_COLORS[method], linewidths=0.7)
            axis.set_xticks([1, 2], ["Single\nmedoid", "Population\natlas"])
            axis.set_title(f"{chr(ord('A') + panel)}  {dataset} — {cohort.title()}", loc="left")
            panel += 1
            axis.set_ylim(-3, 103)
            axis.spines[["top", "right"]].set_visible(False)
            if column == 0:
                axis.set_ylabel("Per-animal Top-1 (%)")
    fig.savefig(output / "Atanas_Kato_boxplots.pdf")
    fig.savefig(output / "Atanas_Kato_boxplots.png", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    activity_window(args.output)
    per_animal(args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
