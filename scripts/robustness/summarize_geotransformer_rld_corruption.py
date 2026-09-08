#!/usr/bin/env python3
"""
Summarize CURRENT-GROUPED GeoTransformer RLD controlled-corruption results.

Protocol:
  1) for each biological fold, average the 3 perturbation seeds;
  2) across 5 biological folds, report mean ± sample SD;
  3) severity=0 defines the clean query denominator for each fold;
  4) missing-neuron conditions additionally report survival coverage and
     effective Top-1 = correct Top-1 / clean-query denominator.

The script accepts several common result.json/metrics.json schemas and refuses
to silently mix old GeoTransformer partitions if checkpoint provenance is
present and does not contain "current_grouped".
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
DEFAULT_ROOT = (
    ROOT
    / "runs/rld_robustness_cv5_seed42_v1/results"
    / "geotransformer_current_grouped_seed42_v1"
)

COND_RE = re.compile(
    r"^(coord_noise|missing|outlier|distractor)_l([0-9]+(?:\.[0-9]+)?)_p([0-9]+)$"
)
FOLD_RE = re.compile(r"^fold_?([0-4])$")


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("atanas", "rld"), default="rld")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def first_numeric(obj, keys):
    if not isinstance(obj, dict):
        return None
    for k in keys:
        if k in obj and obj[k] is not None:
            try:
                return float(obj[k])
            except Exception:
                pass
    return None


def first_int(obj, keys):
    x = first_numeric(obj, keys)
    return None if x is None else int(round(x))


def candidate_metric_blocks(obj):
    blocks = []
    if isinstance(obj, dict):
        blocks.append(obj)
        for k in ("metrics", "test", "result", "template_score", "derived"):
            v = obj.get(k)
            if isinstance(v, dict):
                blocks.append(v)
                for kk in ("metrics", "test", "template_score", "derived"):
                    vv = v.get(kk)
                    if isinstance(vv, dict):
                        blocks.append(vv)
    return blocks


def extract_metric(obj, keys):
    for b in candidate_metric_blocks(obj):
        x = first_numeric(b, keys)
        if x is not None:
            return x
    return math.nan


def extract_int(obj, keys):
    for b in candidate_metric_blocks(obj):
        x = first_int(b, keys)
        if x is not None:
            return x
    return None


def checkpoint_strings(obj):
    out = []
    def visit(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                visit(v, str(k))
        elif isinstance(x, list):
            for v in x:
                visit(v, key)
        elif isinstance(x, str):
            if "checkpoint" in key.lower():
                out.append(x)
    visit(obj)
    return out


def parse_path(path: Path):
    fold = None
    cond = None
    for part in path.parts:
        m = FOLD_RE.match(part)
        if m:
            fold = int(m.group(1))
        m = COND_RE.match(part)
        if m:
            kind, sev, ps = m.groups()
            cond = (kind, float(sev), int(ps), part)
    return fold, cond


def load_records(root: Path):
    files = sorted(set(root.rglob("result.json")) | set(root.rglob("metrics.json")))
    records = []
    used_cells = set()

    for p in files:
        fold, cond = parse_path(p)
        if fold is None or cond is None:
            continue
        kind, severity, pseed, condition = cond
        cell = (fold, kind, severity, pseed)
        if cell in used_cells:
            # Prefer result.json over metrics.json if both exist.
            if p.name != "result.json":
                continue

        obj = json.loads(p.read_text(encoding="utf-8"))

        ckpts = checkpoint_strings(obj)
        for ck in ckpts:
            low = ck.lower()
            if "geotransformer" in low and "current_grouped" not in low:
                raise RuntimeError(
                    f"OLD/AMBIGUOUS GeoTransformer checkpoint in {p}:\n{ck}"
                )

        top1 = extract_metric(obj, ("top1", "ranking_top1", "top1_real"))
        top5 = extract_metric(obj, ("top5", "top5_real"))
        mrr = extract_metric(obj, ("mrr", "mrr_real"))
        hung = extract_metric(
            obj, ("hungarian_accuracy", "assignment_top1", "hungarian")
        )
        cov = extract_metric(
            obj, ("candidate_coverage", "coverage", "gt_candidate_coverage")
        )
        q = extract_int(obj, ("queries", "num_queries"))

        # Some schemas store integer correct counts directly.
        top1_correct = extract_metric(obj, ("top1_correct", "correct_top1"))
        if math.isnan(top1) and q and top1_correct is not None and not math.isnan(top1_correct):
            top1 = top1_correct / q

        if q is None:
            raise RuntimeError(f"Could not extract query count from {p}")
        if math.isnan(top1):
            raise RuntimeError(f"Could not extract Top-1 from {p}")

        records.append({
            "fold": fold,
            "kind": "outlier" if kind == "distractor" else kind,
            "severity": severity,
            "perturbation_seed": pseed,
            "condition": condition,
            "queries": q,
            "top1": top1,
            "top5": top5,
            "mrr": mrr,
            "hungarian": hung,
            "candidate_coverage": cov,
            "source": str(p),
        })
        used_cells.add(cell)

    if not records:
        raise RuntimeError(
            f"No GeoTransformer corruption result.json/metrics.json cells found under:\n{root}"
        )
    return pd.DataFrame(records)


def mean_sd(x):
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if len(a) == 0:
        return math.nan, math.nan
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else math.nan


def main():
    a = args()
    root = a.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    out = a.out or root
    out.mkdir(parents=True, exist_ok=True)

    df = load_records(root)

    # Require all five biological folds.
    folds = sorted(df["fold"].unique().tolist())
    if folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Expected folds 0..4, found {folds}")

    # Get severity-0 clean denominator per biological fold.
    clean = df[np.isclose(df["severity"], 0.0)].copy()
    if clean.empty:
        raise RuntimeError("No severity=0 cells found; cannot establish clean denominator")

    clean_q = {}
    for fold, part in clean.groupby("fold"):
        qs = sorted(set(int(x) for x in part["queries"]))
        if len(qs) != 1:
            raise RuntimeError(
                f"fold{fold}: severity=0 query denominator inconsistent: {qs}"
            )
        clean_q[int(fold)] = qs[0]

        # severity=0 should reproduce exactly across p-seeds/kinds.
        for metric in ("top1", "top5", "mrr", "hungarian", "candidate_coverage"):
            vals = part[metric].dropna().to_numpy(float)
            if len(vals) > 1 and np.max(np.abs(vals - vals[0])) > 1e-10:
                raise RuntimeError(
                    f"fold{fold}: severity=0 {metric} is not exact across conditions: {vals}"
                )

    df["clean_queries"] = df["fold"].map(clean_q).astype(int)
    df["survival_coverage"] = df["queries"] / df["clean_queries"]
    df["effective_top1"] = df["top1"] * df["survival_coverage"]

    raw_csv = out / "geotransformer_corruption_raw_cells.csv"
    df.to_csv(raw_csv, index=False)

    # First average perturbation seeds inside biological fold.
    metrics = [
        "queries", "top1", "top5", "mrr", "hungarian",
        "candidate_coverage", "survival_coverage", "effective_top1",
    ]
    fold_cells = (
        df.groupby(["kind", "severity", "fold"], as_index=False)[metrics]
        .mean()
    )
    fold_csv = out / "geotransformer_corruption_fold_cells.csv"
    fold_cells.to_csv(fold_csv, index=False)

    macro_rows = []
    for (kind, severity), part in fold_cells.groupby(["kind", "severity"], sort=True):
        row = {
            "kind": kind,
            "severity": severity,
            "folds": int(part["fold"].nunique()),
        }
        for m in metrics:
            mu, sd = mean_sd(part[m].to_numpy(float))
            row[f"{m}_mean"] = mu
            row[f"{m}_sd"] = sd
        macro_rows.append(row)

    macro = pd.DataFrame(macro_rows).sort_values(["kind", "severity"])
    macro_csv = out / "geotransformer_macro_summary.csv"
    macro.to_csv(macro_csv, index=False)

    # Paper-ready native-CV5 table.  Keep method-native denominators and use
    # survival coverage for missing-neuron conditions; candidate coverage is
    # retained in the CSV as a separate method-specific diagnostic.
    labels = {
        "coord_noise": "Coordinate noise",
        "missing": "Missing neurons",
        "outlier": "Distractors",
    }
    table_rows = []
    clean_row = macro[
        (macro["kind"] == "coord_noise") & np.isclose(macro["severity"], 0.0)
    ].iloc[0]
    table_rows.append(("Clean", clean_row))
    for kind in ("coord_noise", "missing", "outlier"):
        part = macro[(macro["kind"] == kind) & (macro["severity"] > 0)].sort_values(
            "severity"
        )
        for row in part.itertuples(index=False):
            table_rows.append((labels[kind], row))

    def pct(row, metric):
        return f"{100 * float(getattr(row, metric + '_mean')):.2f} ± " \
               f"{100 * float(getattr(row, metric + '_sd')):.2f}%"

    lines = [
        "# GeoTransformer robustness — native grouped CV5 × seed42",
        "",
        "Each perturbation seed is averaged within biological fold first; values are the "
        "unweighted mean ± sample SD across the same five folds as the Main Benchmark. "
        "No shared-cohort rescoring is used.",
        "",
        "| Corruption | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    export_rows = []
    for label, row in table_rows:
        severity = float(row.severity)
        values = {
            "corruption": label,
            "severity": severity,
            "top1": pct(row, "top1"),
            "hungarian": pct(row, "hungarian"),
            "coverage": pct(row, "survival_coverage"),
            "effective_top1": pct(row, "effective_top1"),
        }
        export_rows.append(values)
        lines.append(
            f"| {label} | {severity:.2f} | {values['top1']} | "
            f"{values['hungarian']} | {values['coverage']} | "
            f"{values['effective_top1']} |"
        )
    lines.extend([
        "",
        "Coverage is the surviving evaluable-query fraction relative to the clean fold. "
        "Activity noise is not applicable because GeoTransformer consumes geometry only.",
        "",
    ])
    table_md = out / "GEOTRANSFORMER_TABLE.md"
    table_md.write_text("\n".join(lines), encoding="utf-8")
    formal_csv = out / "geotransformer_formal_table.csv"
    pd.DataFrame(export_rows).to_csv(formal_csv, index=False)
    if a.dataset == "rld":
        formal_dir = ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_native_cv5_geotransformer"
        formal_dir.mkdir(parents=True, exist_ok=True)
        (formal_dir / "TABLE.md").write_text("\n".join(lines), encoding="utf-8")
        pd.DataFrame(export_rows).to_csv(formal_dir / "summary.csv", index=False)

    print("=" * 128)
    print(
        f"GEOTRANSFORMER CURRENT-GROUPED {a.dataset.upper()} "
        "— CONTROLLED CORRUPTION MACRO"
    )
    print("=" * 128)
    for kind in ("coord_noise", "missing", "outlier"):
        sub = macro[macro["kind"] == kind]
        if sub.empty:
            continue
        print(f"\n[{kind}]")
        for r in sub.itertuples(index=False):
            base = (
                f"sev={r.severity:0.2f} "
                f"Top1={100*r.top1_mean:6.2f}±{100*r.top1_sd:5.2f}% "
                f"Top5={100*r.top5_mean:6.2f}±{100*r.top5_sd:5.2f}% "
                f"MRR={r.mrr_mean:.4f}±{r.mrr_sd:.4f} "
                f"Hung={100*r.hungarian_mean:6.2f}±{100*r.hungarian_sd:5.2f}% "
            )
            if np.isfinite(r.candidate_coverage_mean):
                base += (
                    f"CandCov={100*r.candidate_coverage_mean:6.2f}"
                    f"±{100*r.candidate_coverage_sd:5.2f}% "
                )
            if kind == "missing":
                base += (
                    f"SurvCov={100*r.survival_coverage_mean:6.2f}"
                    f"±{100*r.survival_coverage_sd:5.2f}% "
                    f"EffTop1={100*r.effective_top1_mean:6.2f}"
                    f"±{100*r.effective_top1_sd:5.2f}%"
                )
            print(base)

    # Endpoint deltas relative to severity 0 within each corruption type.
    print("\n" + "=" * 128)
    print("ENDPOINT DROP FROM CLEAN")
    print("=" * 128)
    for kind in ("coord_noise", "missing", "outlier"):
        sub = macro[macro["kind"] == kind].sort_values("severity")
        if sub.empty:
            continue
        zero = sub.iloc[0]
        end = sub.iloc[-1]
        print(
            f"{kind:<12s}: severity {end.severity:.2f} | "
            f"ΔTop1={100*(end.top1_mean-zero.top1_mean):+6.2f} pp | "
            f"ΔHung={100*(end.hungarian_mean-zero.hungarian_mean):+6.2f} pp | "
            f"ΔMRR={end.mrr_mean-zero.mrr_mean:+.4f}",
            end="",
        )
        if kind == "missing":
            print(
                f" | EffTop1={100*end.effective_top1_mean:.2f}% "
                f"| Survival={100*end.survival_coverage_mean:.2f}%"
            )
        else:
            print()

    print("\nraw  :", raw_csv)
    print("fold :", fold_csv)
    print("macro:", macro_csv)
    print("table:", table_md)
    print("DONE")


if __name__ == "__main__":
    main()
