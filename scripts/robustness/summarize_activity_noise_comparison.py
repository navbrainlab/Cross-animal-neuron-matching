#!/usr/bin/env python3
"""Build the formal RLD activity-noise comparison table and figure."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OURS = ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_native_cv5_ours/summary.csv"
NUCLR = ROOT / "runs/rld_robustness_cv5_seed42_v2/results/nuclr/nuclr_macro_summary.csv"
FGW = ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_activity_noise_v1/vanilla_fgw/summary.csv"
GWOT_CELLS = ROOT / "runs/unified_main_benchmark_cv5_seed42_v1/verified_fold_cells.csv"
OUTPUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_activity_noise_v1"
LEVELS = (0.0, 0.10, 0.20, 0.50, 1.00, 2.00)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def exact_levels(rows: list[dict], method: str) -> None:
    got = sorted(float(row["severity"]) for row in rows)
    if got != list(LEVELS):
        raise RuntimeError(f"{method}: expected levels={LEVELS}, observed={got}")


def normalized_rows() -> list[dict]:
    ours_src = [row for row in read_csv(OURS) if row["kind"] == "activity_noise"]
    nuclr_src = [row for row in read_csv(NUCLR) if row["kind"] == "activity_noise"]
    fgw_src = read_csv(FGW)
    exact_levels(ours_src, "Ours")
    exact_levels(nuclr_src, "NuCLR")
    exact_levels(fgw_src, "Vanilla FGW")

    rows: list[dict] = []
    specs = (
        ("Ours", ours_src, {
            "top1": "top1", "hungarian": "hungarian", "coverage": "coverage",
            "effective_top1": "effective_top1",
        }),
        ("NuCLR", nuclr_src, {
            "top1": "ranking_top1", "hungarian": "assignment_top1",
            "coverage": "coverage_vs_clean_eligible", "effective_top1": "effective_top1",
        }),
        ("Vanilla FGW", fgw_src, {
            "top1": "top1", "hungarian": "hungarian", "coverage": "coverage",
            "effective_top1": "effective_top1",
        }),
    )
    for method, source, names in specs:
        for src in sorted(source, key=lambda row: float(row["severity"])):
            row = {
                "method": method,
                "severity": float(src["severity"]),
                "folds": int(src.get("folds", src.get("num_folds", 5))),
                "status": "evaluated" if method != "Vanilla FGW" else "certified_activity_invariant",
            }
            for dst, prefix in names.items():
                row[f"{dst}_mean"] = float(src[f"{prefix}_mean"])
                row[f"{dst}_sd"] = float(src[f"{prefix}_sd"])
            rows.append(row)
    return rows


def assert_clean(rows: list[dict]) -> None:
    # The displayed Main values are used as a coarse second guard; source evaluators
    # independently enforce exact per-fold replay before producing these summaries.
    displayed = {"Ours": (63.93, 4.61), "NuCLR": (10.14, 7.62), "Vanilla FGW": (7.99, 2.44)}
    for method, (mean_pct, sd_pct) in displayed.items():
        clean = next(row for row in rows if row["method"] == method and row["severity"] == 0.0)
        got = (round(100 * clean["top1_mean"], 2), round(100 * clean["top1_sd"], 2))
        if got != (mean_pct, sd_pct):
            raise RuntimeError(f"{method}: severity=0 {got} != Main {(mean_pct, sd_pct)}")


def pct(row: dict, key: str) -> str:
    return f"{100 * row[key + '_mean']:.2f} ± {100 * row[key + '_sd']:.2f}%"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_table(rows: list[dict], gwot_missing: bool) -> str:
    lines = [
        "# Activity-noise robustness — RLD grouped CV5 × seed42",
        "",
        "Noise is added only to held-out-test `activity_raw` as independent Gaussian noise with standard deviation `severity × each neuron's temporal SD`. Three corruption draws are averaged within each fold, followed by the unweighted mean ± sample SD across the same five biological folds. Severity 0 must reproduce each method's formal Main Benchmark cells.",
        "",
        "| Method | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = {"NuCLR": 0, "GWOT-MD": 1, "Vanilla FGW": 2, "Ours": 3}
    by_method = sorted(rows, key=lambda row: (order[row["method"]], row["severity"]))
    inserted_gwot = False
    for row in by_method:
        if not inserted_gwot and row["method"] == "Vanilla FGW" and gwot_missing:
            lines.append("| GWOT-MD | — | N/A | N/A | N/A | N/A |")
            inserted_gwot = True
        lines.append(
            f"| {row['method']} | {row['severity']:.2f} | {pct(row, 'top1')} | "
            f"{pct(row, 'hungarian')} | {pct(row, 'coverage')} | "
            f"{pct(row, 'effective_top1')} |"
        )
    lines.extend([
        "",
        "Vanilla FGW is flat by construction in the locked formal implementation: it consumes geometry only. Its values are backed by a 1,520-file input-invariance audit rather than imputed from an unrelated run.",
        "",
    ])
    if gwot_missing:
        lines.extend([
            "GWOT-MD is intentionally N/A: no canonical RLD held-out-test Main Benchmark cells exist, so severity=0 cannot be locked. Historical GWOT-MD files are validation-only and are not substituted.",
            "",
        ])
    return "\n".join(lines)


def plot(rows: list[dict]) -> None:
    styles = {
        "NuCLR": ("#D55E00", "s"),
        "Vanilla FGW": ("#7F7F7F", "^"),
        "Ours": ("#0072B2", "o"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    for method, (color, marker) in styles.items():
        cells = sorted((row for row in rows if row["method"] == method), key=lambda row: row["severity"])
        # Treat the predefined stress levels as ordered experimental categories;
        # this keeps the dense 0/0.1/0.2 labels legible while retaining exact labels.
        x = np.arange(len(cells), dtype=float)
        for ax, key, title in zip(axes, ("top1", "hungarian"), ("Top-1", "Hungarian accuracy")):
            y = 100 * np.asarray([row[f"{key}_mean"] for row in cells])
            e = 100 * np.asarray([row[f"{key}_sd"] for row in cells])
            ax.errorbar(x, y, yerr=e, label=method, color=color, marker=marker,
                        linewidth=2, markersize=5, capsize=2.5)
            ax.set_title(title)
            ax.set_xlabel("Activity-noise severity")
            ax.set_ylabel("Accuracy (%)")
            ax.grid(alpha=0.22, linewidth=0.7)
            ax.set_xticks(np.arange(len(LEVELS), dtype=float))
            ax.set_xticklabels([f"{level:g}" for level in LEVELS])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(OUTPUT / f"Figure_activity_noise_comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = normalized_rows()
    assert_clean(rows)
    with GWOT_CELLS.open(newline="", encoding="utf-8-sig") as handle:
        gwot = [row for row in csv.DictReader(handle) if row["dataset"] == "rld" and row["method"] == "GWOT-MD"]
    gwot_missing = len(gwot) != 5
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "activity_noise_comparison.csv", rows)
    (OUTPUT / "TABLE.md").write_text(render_table(rows, gwot_missing), encoding="utf-8")
    plot(rows)
    audit = {
        "protocol": "RLD cv5_grouped_v1 x seed42; native held-out-test activity corruption",
        "aggregation": "average 3 corruption draws within fold, then mean and sample SD over 5 folds",
        "severity_zero_main_gate": "passed for Ours, NuCLR, and Vanilla FGW",
        "methods": {
            "Ours": "complete_evaluated",
            "NuCLR": "complete_evaluated",
            "Vanilla FGW": "complete_certified_activity_invariant",
            "GWOT-MD": "blocked_missing_formal_main_severity_zero" if gwot_missing else "main_available_robustness_pending",
        },
        "gwot_md_values_imputed": False,
    }
    (OUTPUT / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT / "TABLE.md")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
