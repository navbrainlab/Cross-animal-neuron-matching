#!/usr/bin/env python3
"""One paired estimand and hierarchical bootstrap for every RLD method.

The bootstrap resamples biological folds, then worms within each sampled fold,
then shared corruption-draw IDs.  Identical resampling indices are used for all
methods, which permits paired method contrasts once every result root is present.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs/rld_robustness_cv5_seed42_v2"
COND_RE = re.compile(r"^(coord_noise|missing|outlier)_l([0-9]+(?:\.[0-9]+)?)_p([0-9]+)$")
METRICS = ("top1", "top5", "mrr", "hungarian", "coverage", "effective_top1", "retention")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def condition(name: str):
    m = COND_RE.match(name)
    if not m:
        raise ValueError(name)
    return m.group(1), float(m.group(2)), int(m.group(3))


def uid_from_geo(name: str) -> str:
    stem = Path(name).stem
    parts = stem.split("__")
    if len(parts) < 2:
        raise ValueError(name)
    return parts[1]


def aggregate_query_rows(rows: list[dict], adapter: str):
    out = defaultdict(lambda: np.zeros(5, dtype=float))
    for row in rows:
        if adapter in {"cpd", "fdnc"}:
            uid = row["query_uid"]
            values = (1, float(row["top1"]), float(row["top5"]),
                      float(row["rr"]), 0.0)
        elif adapter == "ours":
            uid = row["uid"]
            rank = int(row["rank"])
            values = (1, float(row["correct"]), float(rank <= 5),
                      1.0 / rank, float(row["hungarian_correct"]))
        elif adapter == "nuclr":
            uid = row["query_worm"]
            values = (1, float(row["top1"]), float(row["top5"]),
                      float(row["rr"]), float(row["hungarian_top1"]))
        else:
            raise ValueError(adapter)
        out[uid] += np.asarray(values)
    return dict(out)


def load_cell(method: str, root: Path, fold: int, name: str):
    base = root / f"fold{fold}" / name
    if method == "cpd":
        report_root = base / "original_runner"
        out = aggregate_query_rows(read_csv(report_root / "per_query.csv"), method)
        report = json.loads((report_root / "metrics.json").read_text(encoding="utf-8"))
        for row in report["per_animal"]:
            out[row["query_uid"]][4] = float(row["hungarian_accuracy"]) * int(row["queries"])
        return out
    if method == "fdnc":
        out = aggregate_query_rows(read_csv(base / "per_query.csv"), method)
        report = json.loads((base / "metrics.json").read_text(encoding="utf-8"))
        for row in report["per_animal"]:
            out[row["query_uid"]][4] = float(row["hungarian_accuracy"]) * int(row["queries"])
        return out
    if method == "ours":
        return aggregate_query_rows(read_csv(base / "queries.csv"), method)
    if method == "nuclr":
        return aggregate_query_rows(read_csv(base / "query_level.csv"), method)
    if method == "geo":
        report = json.loads((base / "result.json").read_text(encoding="utf-8"))
        out = {}
        for row in report["test"]["per_test_worm"]:
            out[uid_from_geo(row["query"])] = np.asarray([
                row["queries"], row["top1_correct"], row["top5_correct"],
                row["rr_sum"], row["hungarian_correct"],
            ], dtype=float)
        return out
    raise ValueError(method)


def parse_manifest(path: Path):
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    worms = defaultdict(set)
    common_queries = {}
    for row in report["conditions"]:
        fold = int(row["fold"])
        name = Path(row["root"]).name
        kind, severity, draw = condition(name)
        rows.append((fold, name, kind, severity, draw))
        common_queries[(fold, kind, severity, draw)] = {}
        for rec in row["files"]:
            uid = str(rec["recording_uid"])
            worms[fold].add(uid)
            common_queries[(fold, kind, severity, draw)][uid] = len(
                rec["evaluable_query_rows_after_corruption"]
            )
    return sorted(rows), {k: sorted(v) for k, v in worms.items()}, common_queries


def ratios(counts: np.ndarray, current_q: np.ndarray, clean_q: np.ndarray):
    reported_q, top1, top5, rr, hung = counts.sum(axis=0)
    q, cq = current_q.sum(), clean_q.sum()
    if reported_q > q + 1e-9:
        raise RuntimeError(f"method emitted {reported_q} queries outside common cohort of {q}")
    if q <= 0 or cq <= 0:
        return np.full(6, np.nan)
    return np.asarray([top1/q, top5/q, rr/q, hung/q, q/cq, top1/cq])


def point_for_method(cells, common_queries, worms, folds, kind, severity):
    fold_values = []
    for fold in folds:
        uids = worms[fold]
        clean = cells[(fold, "coord_noise", 0.0, 0)]
        clean_q = np.asarray([
            common_queries[(fold, "coord_noise", 0.0, 0)][uid] for uid in uids
        ])
        draws = sorted(k[3] for k in cells if k[:3] == (fold, kind, severity))
        draw_values = []
        for draw in draws:
            cell = cells[(fold, kind, severity, draw)]
            counts = np.stack([cell.get(uid, np.zeros(5)) for uid in uids])
            current_q = np.asarray([
                common_queries[(fold, kind, severity, draw)][uid] for uid in uids
            ])
            draw_values.append(ratios(counts, current_q, clean_q))
        fold_values.append(np.nanmean(draw_values, axis=0))
    value = np.nanmean(fold_values, axis=0)
    clean_eff = point_for_clean(cells, common_queries, worms, folds)
    return np.r_[value, value[5] / clean_eff]


def point_for_clean(cells, common_queries, worms, folds):
    values = []
    for fold in folds:
        uids = worms[fold]
        clean = cells[(fold, "coord_noise", 0.0, 0)]
        counts = np.stack([clean.get(uid, np.zeros(5)) for uid in uids])
        clean_q = np.asarray([
            common_queries[(fold, "coord_noise", 0.0, 0)][uid] for uid in uids
        ])
        values.append(ratios(counts, clean_q, clean_q)[5])
    return float(np.nanmean(values))


def bootstrap_once(cells, common_queries, worms, folds, kind, severity, rng):
    sampled_folds = rng.choice(folds, size=len(folds), replace=True)
    fold_values, clean_values = [], []
    for fold in sampled_folds:
        uids = worms[int(fold)]
        picked = rng.integers(0, len(uids), size=len(uids))
        clean = cells[(int(fold), "coord_noise", 0.0, 0)]
        clean_counts = np.stack([clean.get(uids[i], np.zeros(5)) for i in picked])
        clean_q = np.asarray([
            common_queries[(int(fold), "coord_noise", 0.0, 0)][uids[i]] for i in picked
        ])
        clean_values.append(ratios(clean_counts, clean_q, clean_q)[5])
        draws = sorted(k[3] for k in cells if k[:3] == (int(fold), kind, severity))
        sampled_draws = rng.choice(draws, size=len(draws), replace=True)
        draw_values = []
        for draw in sampled_draws:
            cell = cells[(int(fold), kind, severity, int(draw))]
            counts = np.stack([cell.get(uids[i], np.zeros(5)) for i in picked])
            current_q = np.asarray([
                common_queries[(int(fold), kind, severity, int(draw))][uids[i]]
                for i in picked
            ])
            draw_values.append(ratios(counts, current_q, clean_q))
        fold_values.append(np.nanmean(draw_values, axis=0))
    value = np.nanmean(fold_values, axis=0)
    clean_eff = float(np.nanmean(clean_values))
    return np.r_[value, value[5] / clean_eff]


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    ap.add_argument("--methods", default="cpd,fdnc,ours,nuclr,geo")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    method_roots = {
        "cpd": args.run_root / "results/cpd",
        "fdnc": args.run_root / "results/fdnc",
        "ours": args.run_root / "results/ours_static_seed42",
        "nuclr": args.run_root / "results/nuclr",
        "geo": args.run_root / "results/geotransformer",
    }
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    manifest_rows, worms, common_queries = parse_manifest(
        args.run_root / "corruptions/MANIFEST.json"
    )
    folds = sorted(worms)
    expected = len(manifest_rows)
    summaries = []
    sample_cache = {}
    estimate_cache = {}
    for method in methods:
        cells = {}
        missing = []
        for fold, name, kind, severity, draw in manifest_rows:
            try:
                cells[(fold, kind, severity, draw)] = load_cell(
                    method, method_roots[method], fold, name
                )
            except FileNotFoundError as exc:
                missing.append(str(exc.filename or exc))
        if missing:
            message = f"{method}: {len(missing)}/{expected} condition outputs missing"
            if not args.allow_incomplete:
                raise RuntimeError(message)
            print("[SKIP]", message)
            continue
        grid = sorted({(kind, severity) for _, _, kind, severity, _ in manifest_rows})
        rng = np.random.default_rng(args.seed)
        for kind, severity in grid:
            estimate = point_for_method(cells, common_queries, worms, folds, kind, severity)
            samples = np.stack([
                bootstrap_once(cells, common_queries, worms, folds, kind, severity, rng)
                for _ in range(args.bootstrap)
            ])
            sample_cache[(method, kind, severity)] = samples
            estimate_cache[(method, kind, severity)] = estimate
            row = {"method": method, "kind": kind, "severity": severity,
                   "folds": len(folds), "bootstrap_replicates": args.bootstrap}
            for j, metric in enumerate(METRICS):
                valid = samples[:, j][np.isfinite(samples[:, j])]
                row[f"{metric}_mean"] = float(estimate[j])
                row[f"{metric}_ci_low"] = float(np.quantile(valid, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(valid, 0.975))
            summaries.append(row)
    if not summaries:
        raise RuntimeError("no complete method outputs")
    out = args.run_root / "hierarchical_bootstrap_summary.csv"
    write_csv(out, summaries)
    pairwise = []
    completed_methods = sorted({row["method"] for row in summaries})
    grid = sorted({(row["kind"], row["severity"]) for row in summaries})
    for method_a, method_b in combinations(completed_methods, 2):
        for kind, severity in grid:
            key_a = (method_a, kind, severity)
            key_b = (method_b, kind, severity)
            if key_a not in sample_cache or key_b not in sample_cache:
                continue
            diffs = sample_cache[key_a] - sample_cache[key_b]
            estimate = estimate_cache[key_a] - estimate_cache[key_b]
            row = {"method_a": method_a, "method_b": method_b, "kind": kind,
                   "severity": severity, "contrast": "method_a_minus_method_b"}
            for j, metric in enumerate(METRICS):
                valid = diffs[:, j][np.isfinite(diffs[:, j])]
                row[f"{metric}_difference"] = float(estimate[j])
                row[f"{metric}_ci_low"] = float(np.quantile(valid, 0.025))
                row[f"{metric}_ci_high"] = float(np.quantile(valid, 0.975))
            pairwise.append(row)
    if pairwise:
        write_csv(args.run_root / "hierarchical_bootstrap_pairwise.csv", pairwise)
    meta = {
        "estimand": "unweighted draw mean within fold; unweighted biological-fold mean",
        "query_cohort": "shared manifest query rows; method-unrepresentable labels remain in the denominator and score zero",
        "bootstrap": "resample folds, worms within fold, and shared corruption draw IDs; paired indices across methods",
        "replicates": args.bootstrap,
        "seed": args.seed,
        "methods": completed_methods,
        "metrics": list(METRICS),
    }
    (args.run_root / "hierarchical_bootstrap_summary.json").write_text(
        json.dumps({"metadata": meta, "rows": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out)


if __name__ == "__main__":
    main()
