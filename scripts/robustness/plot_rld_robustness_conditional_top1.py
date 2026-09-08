#!/usr/bin/env python3
"""Plot native-CV5 RLD robustness from the formal seed42 summaries."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/rld_robustness_cv5_seed42_v2"

METHODS = {
    "ours": {
        "label": "Ours", "color": "#E64B35", "linestyle": "-",
        "marker": "o", "linewidth": 2.4, "markersize": 4.6,
        "alpha": 0.12, "zorder": 6,
    },
    "fdnc": {
        "label": "fDNC", "color": "#377EB8", "linestyle": "--",
        "marker": "^", "linewidth": 1.55, "markersize": 4.4,
        "alpha": 0.07, "zorder": 4,
    },
    "cpd": {
        "label": "CPD", "color": "#777777", "linestyle": "-.",
        "marker": "D", "linewidth": 1.45, "markersize": 3.8,
        "alpha": 0.07, "zorder": 3,
    },
    "nuclr": {
        "label": "NuCLR", "color": "#8E5BB7", "linestyle": ":",
        "marker": "s", "linewidth": 1.55, "markersize": 3.9,
        "alpha": 0.07, "zorder": 2,
    },
    "geo": {
        "label": "GeoTransformer", "color": "#2E8B57",
        "linestyle": (0, (6.0, 2.4)), "marker": "x",
        "linewidth": 1.45, "markersize": 4.3,
        "alpha": 0.07, "zorder": 1,
    },
}
PLOT_ORDER = ("geo", "nuclr", "cpd", "fdnc", "ours")
LEGEND_ORDER = ("ours", "fdnc", "cpd", "geo", "nuclr")
PANELS = (
    ("coord_noise", "(a) Coordinate noise", r"Coordinate noise, $\sigma$"),
    ("missing", "(b) Missing neurons", r"Missing fraction, $r_{\mathrm{miss}}$"),
    ("outlier", "(c) Distractor neurons", r"Distractor fraction, $r_{\mathrm{dist}}$"),
)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_rows(run_root: Path):
    """Normalize the five native-CV5 summaries to one plotting schema."""
    rows = []

    for row in _read_csv(run_root / "formal_native_cv5_ours/summary.csv"):
        rows.append({
            "method": "ours", "kind": row["kind"],
            "severity": row["severity"], "top1_mean": row["top1_mean"],
            "top1_sd": row["top1_sd"],
        })

    for row in _read_csv(
        run_root / "formal_native_cv5_cpd_fdnc/summary.csv"
    ):
        method = {"CPD": "cpd", "fDNC": "fdnc"}.get(row["method"])
        if method is None:
            continue
        rows.append({
            "method": method, "kind": row["kind"],
            "severity": row["severity"], "top1_mean": row["top1_mean"],
            "top1_sd": row["top1_sd"],
        })

    numeric_sources = {
        "geo": (
            run_root / "results/geotransformer/geotransformer_macro_summary.csv",
            "top1_mean", "top1_sd",
        ),
        "nuclr": (
            run_root / "results/nuclr/nuclr_macro_summary.csv",
            "ranking_top1_mean", "ranking_top1_sd",
        ),
    }
    for method, (path, mean_col, sd_col) in numeric_sources.items():
        for row in _read_csv(path):
            rows.append({
                "method": method, "kind": row["kind"],
                "severity": row["severity"], "top1_mean": row[mean_col],
                "top1_sd": row[sd_col],
            })

    expected = {
        (method, kind, severity)
        for method in METHODS
        for kind, _, _ in PANELS
        for severity in (
            (0.0, 0.02, 0.05, 0.10, 0.20)
            if kind == "coord_noise"
            else (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)
        )
    }
    observed = {
        (row["method"], row["kind"], float(row["severity"])) for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"formal native-CV5 plot input is incomplete: "
            f"rows={len(rows)}, expected={len(expected)}, "
            f"missing={missing}, extra={extra}"
        )

    clean_expected = {
        "ours": 0.6393, "fdnc": 0.4511, "cpd": 0.2651,
        "geo": 0.1652, "nuclr": 0.1014,
    }
    for method, target in clean_expected.items():
        clean = [
            float(row["top1_mean"]) for row in rows
            if row["method"] == method
            and row["kind"] == "coord_noise"
            and float(row["severity"]) == 0.0
        ]
        if len(clean) != 1 or abs(clean[0] - target) > 5e-5:
            raise RuntimeError(
                f"{method} severity=0 does not reproduce the formal Main "
                f"Benchmark: observed={clean}, expected~={target}"
            )
    return rows


def series(rows, method: str, kind: str):
    part = [x for x in rows if x["method"] == method and x["kind"] == kind]
    part.sort(key=lambda x: float(x["severity"]))
    return (
        np.asarray([float(x["severity"]) for x in part]),
        100 * np.asarray([float(x["top1_mean"]) for x in part]),
        100 * np.asarray([
            max(0.0, float(x["top1_mean"]) - float(x["top1_sd"]))
            for x in part
        ]),
        100 * np.asarray([
            min(1.0, float(x["top1_mean"]) + float(x["top1_sd"]))
            for x in part
        ]),
    )


def annotate_ours(ax, x, y, panel_index: int):
    if panel_index == 0:
        ax.annotate(
            f"{y[0]:.1f}%", (x[0], y[0]), xytext=(5, 5),
            textcoords="offset points", color=METHODS["ours"]["color"],
            fontsize=7.2, fontweight="semibold", ha="left", va="bottom",
            annotation_clip=False, zorder=8,
        )
    max_offsets = ((-2, 14), (-2, 11), (-1, 5))
    ax.annotate(
        f"{y[-1]:.1f}%", (x[-1], y[-1]), xytext=max_offsets[panel_index],
        textcoords="offset points", color=METHODS["ours"]["color"],
        fontsize=7.2, fontweight="semibold", ha="right", va="bottom",
        annotation_clip=False, zorder=8,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--stem", default="Figure_X_robustness_conditional_top1")
    args = ap.parse_args()
    rows = read_rows(args.run_root)
    out_dir = args.output_dir or args.run_root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 6.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.7,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4), sharey=True)

    for panel_index, (ax, (kind, title, xlabel)) in enumerate(zip(axes, PANELS)):
        for method in PLOT_ORDER:
            style = METHODS[method]
            x, y, lo, hi = series(rows, method, kind)
            ax.fill_between(
                x, lo, hi, color=style["color"], alpha=style["alpha"],
                linewidth=0, zorder=style["zorder"] - 0.5,
            )
            ax.plot(
                x, y, color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], linewidth=style["linewidth"],
                markersize=style["markersize"], markerfacecolor="white",
                markeredgewidth=0.9, zorder=style["zorder"],
                solid_capstyle="round", dash_capstyle="round",
            )
            if method == "ours":
                annotate_ours(ax, x, y, panel_index)

        ax.set_title(title, pad=4.0)
        ax.set_xlabel(xlabel, labelpad=4.0)
        ax.set_ylim(0, 70)
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#666666")
        ax.spines["bottom"].set_color("#666666")
        ax.tick_params(axis="both", width=0.6, length=2.8, color="#666666")

        if kind == "coord_noise":
            ticks = np.asarray([0.00, 0.02, 0.05, 0.10, 0.20])
            ax.set_xlim(-0.008, 0.208)
            ax.set_xticks(ticks, ["0", "0.02", "0.05", "0.10", "0.20"])
        else:
            ticks = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            ax.set_xlim(-0.02, 0.52)
            ax.set_xticks(ticks, ["0%", "10%", "20%", "30%", "40%", "50%"])
    axes[0].set_ylabel("Top-1 accuracy (%)", labelpad=4.0)
    handles = [
        Line2D(
            [0], [0], label=METHODS[m]["label"], color=METHODS[m]["color"],
            linestyle=METHODS[m]["linestyle"], marker=METHODS[m]["marker"],
            linewidth=METHODS[m]["linewidth"], markersize=METHODS[m]["markersize"],
            markerfacecolor="white", markeredgewidth=0.9,
        ) for m in LEGEND_ORDER
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, 0.99), handlelength=2.55, columnspacing=1.4,
        handletextpad=0.55, borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.082, right=0.995, bottom=0.245, top=0.80, wspace=0.18)

    metadata = {
        "Title": "Robustness to synthetic perturbations",
        "Subject": "Native-CV5 Top-1 mean and fold SD under held-out test corruption",
        "Keywords": "RLD robustness CV5 seed42 coordinate noise missing neurons distractors",
    }
    for suffix, kwargs in (
        ("pdf", {"metadata": metadata}),
        ("svg", {"metadata": {"Title": metadata["Title"], "Description": metadata["Subject"]}}),
        ("png", {"dpi": 600, "metadata": {"Title": metadata["Title"]}}),
    ):
        path = out_dir / f"{args.stem}.{suffix}"
        fig.savefig(path, bbox_inches=None, facecolor="white", **kwargs)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
