#!/usr/bin/env python3
"""Plot Ours unknown detection and no-dustbin ablation on formal RLD CV5."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = (
    ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_dustbin_ablation_ours/summary.csv"
)
DEFAULT_OUTPUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/figures"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(mode, severity) for mode in ("capacity_dustbin", "ordinary")
                for severity in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)}
    observed = {(row["mode"], float(row["severity"])) for row in rows}
    if observed != expected or len(rows) != 12:
        raise RuntimeError(f"incomplete dustbin summary: missing={expected-observed}")
    return rows


def series(rows, mode: str, metric: str, include_clean: bool = True):
    part = [row for row in rows if row["mode"] == mode]
    if not include_clean:
        part = [row for row in part if float(row["severity"]) > 0]
    part.sort(key=lambda row: float(row["severity"]))
    if any(row[f"{metric}_mean"] == "" for row in part):
        raise RuntimeError(f"undefined plotted metric: {mode}/{metric}")
    x = np.asarray([float(row["severity"]) for row in part])
    mean = 100 * np.asarray([float(row[f"{metric}_mean"]) for row in part])
    sd = 100 * np.asarray([float(row[f"{metric}_sd"]) for row in part])
    return x, mean, np.maximum(0, mean - sd), np.minimum(100, mean + sd)


def draw(ax, rows, *, mode, metric, label, color, marker, linestyle="-"):
    x, mean, low, high = series(
        rows, mode, metric, include_clean=(metric == "known_top1_real")
    )
    ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
    ax.plot(
        x, mean, color=color, marker=marker, markerfacecolor="white",
        markeredgewidth=0.9, linestyle=linestyle, linewidth=1.8,
        markersize=4.5, label=label,
    )


def style_axis(ax, title: str, ylabel: str, include_clean: bool):
    ax.set_title(title, pad=4)
    ax.set_xlabel(r"Distractor fraction, $r_{\mathrm{dist}}$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 102)
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ticks = [0, .1, .2, .3, .4, .5] if include_clean else [.1, .2, .3, .4, .5]
    labels = [f"{int(100*x)}%" for x in ticks]
    ax.set_xlim((-0.02 if include_clean else 0.08), 0.52)
    ax.set_xticks(ticks, labels)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(width=0.6, length=2.8, color="#666666")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stem", default="Figure_X_unknown_dustbin_ablation")
    args = parser.parse_args()
    rows = read_rows(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "axes.titlesize": 8.7, "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))

    draw(axes[0], rows, mode="capacity_dustbin", metric="unknown_recall",
         label="Recall", color="#E64B35", marker="o")
    draw(axes[0], rows, mode="capacity_dustbin", metric="dustbin_precision",
         label="Precision", color="#377EB8", marker="^")
    draw(axes[0], rows, mode="capacity_dustbin", metric="dustbin_f1",
         label="F1", color="#2E8B57", marker="s")
    style_axis(axes[0], "(a) Unknown rejection", "Detection score (%)", False)

    draw(axes[1], rows, mode="capacity_dustbin", metric="unknown_auroc",
         label="AUROC", color="#8E5BB7", marker="D")
    draw(axes[1], rows, mode="capacity_dustbin", metric="unknown_auprc",
         label="AUPRC", color="#F39C12", marker="v")
    style_axis(axes[1], "(b) Dustbin ranking", "Ranking score (%)", False)

    draw(axes[2], rows, mode="capacity_dustbin", metric="known_top1_real",
         label="With dustbin", color="#E64B35", marker="o")
    draw(axes[2], rows, mode="ordinary", metric="known_top1_real",
         label="No dustbin", color="#777777", marker="x", linestyle="--")
    style_axis(axes[2], "(c) No-dustbin ablation", "Known Top-1 (%)", True)

    for ax in axes:
        ax.legend(frameon=False, loc="lower left", handlelength=2.3)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.22, top=0.90, wspace=0.34)

    metadata = {
        "Title": "Unknown detection and no-dustbin ablation",
        "Subject": "RLD grouped CV5 seed42 held-out distractor robustness",
        "Keywords": "unknown dustbin ablation RLD CV5 seed42 distractors",
    }
    for suffix, kwargs in (
        ("pdf", {"metadata": metadata}),
        ("svg", {"metadata": {"Title": metadata["Title"],
                               "Description": metadata["Subject"]}}),
        ("png", {"dpi": 600, "metadata": {"Title": metadata["Title"]}}),
    ):
        path = args.output_dir / f"{args.stem}.{suffix}"
        fig.savefig(path, facecolor="white", **kwargs)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
