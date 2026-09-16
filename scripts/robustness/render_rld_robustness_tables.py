#!/usr/bin/env python3
"""Render the audited five-method RLD robustness CSV as paper-ready Markdown."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/rld_robustness_cv5_seed42_v2"
METHODS = (("cpd", "CPD"), ("fdnc", "fDNC"), ("nuclr", "NuCLR"),
           ("geo", "GeoTransformer"), ("ours", "Ours"))
KINDS = (("coord_noise", "Coordinate noise"),
         ("missing", "Missing neurons"),
         ("outlier", "Inserted distractors"))
METRICS = ("top1", "hungarian", "coverage", "effective_top1")


def fmt(row: dict, metric: str, bold: bool = False) -> str:
    text = (
        f"{100 * float(row[metric + '_mean']):.2f}% "
        f"[{100 * float(row[metric + '_ci_low']):.2f}, "
        f"{100 * float(row[metric + '_ci_high']):.2f}]"
    )
    return f"**{text}**" if bold else text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    source = args.run_root / "hierarchical_bootstrap_summary.csv"
    output = args.output or args.run_root / "FINAL_ROBUSTNESS_TABLES.md"
    with source.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 85:
        raise RuntimeError(f"expected 85 audited rows, found {len(rows)}")

    lines = [
        "# Final RLD robustness tables",
        "",
        "All cells use the canonical main-table query cohort and are reported as mean "
        "[95% paired hierarchical-bootstrap CI], with 10,000 replicates.",
        "",
    ]
    for method, label in METHODS:
        part = [x for x in rows if x["method"] == method]
        if len(part) != 17:
            raise RuntimeError(f"{method}: expected 17 rows, found {len(part)}")
        lookup = {(x["kind"], float(x["severity"])): x for x in part}
        lines.extend([
            f"## {label}", "",
            "| Corruption | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        clean = lookup[("coord_noise", 0.0)]
        lines.append(
            "| **Clean** | 0.00 | " + " | ".join(fmt(clean, x, True) for x in METRICS) + " |"
        )
        for kind, kind_label in KINDS:
            severities = sorted(s for k, s in lookup if k == kind and s > 0)
            for i, severity in enumerate(severities):
                row = lookup[(kind, severity)]
                bold = severity == severities[-1]
                name = kind_label if i == 0 else ""
                lines.append(
                    f"| {name} | {severity:.2f} | "
                    + " | ".join(fmt(row, x, bold) for x in METRICS) + " |"
                )
        lines.append("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output.resolve())


if __name__ == "__main__":
    main()
