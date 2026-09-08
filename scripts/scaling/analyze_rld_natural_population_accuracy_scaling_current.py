#!/usr/bin/env python3
"""
Natural population-size accuracy scaling on the FINAL locked RLD benchmark.

This is deliberately NOT a neuron-deletion experiment. No neuron is removed,
duplicated, padded, or corrupted. Every held-out test worm is evaluated exactly
as in the already-exported clean unified benchmark; we only stratify the locked
predictions by the worm's ORIGINAL natural population size N.

Default comparison:
    Ours (Static MPRT) vs fDNC, seed42, 5 biological outer folds.

Why seed42 by default?
    It matches the locked seed used by the runtime/robustness analyses. The CLI
    can additionally analyze seeds 1 42 123 if the exact final score files exist.

Primary metric:
    pooled out-of-fold Top-1 within population-size bins.

Also reports:
    Top-5, MRR, Hungarian, macro worm Top-1,
    query-worm bootstrap 95% CI,
    paired MPRT - fDNC Top-1 difference,
    Spearman correlation between natural N and worm-level accuracy.

Strict guards:
  * use current RLD grouped fold test files only;
  * each test worm must occur in exactly one biological outer fold;
  * score query signatures must be EXACTLY identical between compared methods;
  * no old GeoTransformer scores are used here;
  * candidate-score files are never modified.

Outputs:
  runs/rld_natural_accuracy_scaling_v1/
    worm_metadata.csv
    worm_metrics.csv
    scaling_summary.csv
    paired_delta_summary.csv
    correlations.csv
    protocol.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
CV_ROOT = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
DEFAULT_OUT = ROOT / "runs/rld_natural_accuracy_scaling_current_v2"

METHODS = {
    "MPRT": ROOT / "runs/unified_benchmark/ours_static/rld",
    "fDNC": ROOT / "runs/accuracy_scaling_current_v1/fdnc_current_grouped/rld",
}

INVALID = {"", "nan", "none", "null", "-1", "unknown", "unk", "?", "unlabeled", "unlabelled"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--bins", type=int, default=4, help="Number of natural-N quantile bins.")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20260826)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def norm_label(x: Any) -> str:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    s = str(x).strip()
    return "" if s.lower() in INVALID else s


def recording_uid(path: Path) -> str:
    """
    Canonicalize current grouped filenames such as:
        test__20240902-15-36-41__6c56c3c3.npz
    -> 20240902-15-36-41
    """
    stem = path.stem
    stem = re.sub(r"^(?:train|val|test)__", "", stem)
    stem = re.sub(r"__[0-9a-fA-F]{8,64}$", "", stem)
    return stem


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_labeled_count(labels: np.ndarray, mask: np.ndarray) -> int:
    vals = []
    for x, keep in zip(labels, mask):
        if not bool(keep):
            continue
        s = norm_label(x)
        if s:
            vals.append(s)
    counts = pd.Series(vals).value_counts() if vals else pd.Series(dtype=int)
    return int((counts == 1).sum())


def load_worm_metadata(folds: list[int]) -> pd.DataFrame:
    rows = []
    seen = {}

    for fold in folds:
        test_root = CV_ROOT / f"fold_{fold}" / "test"
        files = sorted(test_root.rglob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No test NPZ files under {test_root}")

        for path in files:
            uid = recording_uid(path)
            if uid in seen:
                raise RuntimeError(
                    f"Biological test worm appears in >1 outer fold: {uid}\n"
                    f"  first={seen[uid]}\n  second={path}"
                )
            seen[uid] = path

            with np.load(path, allow_pickle=False) as z:
                if "xyz" not in z:
                    raise KeyError(f"{path}: missing xyz")
                xyz = np.asarray(z["xyz"])
                n_raw = int(xyz.shape[0])
                finite = np.isfinite(xyz).all(axis=1)
                n_finite = int(finite.sum())

                labels = np.asarray(z["cell_id"]) if "cell_id" in z else np.asarray([""] * n_raw)
                if len(labels) != n_raw:
                    raise RuntimeError(f"{path}: xyz/cell_id length mismatch")

                if "labeled_mask" in z:
                    mask = np.asarray(z["labeled_mask"], dtype=bool)
                elif "clean_mask" in z:
                    mask = np.asarray(z["clean_mask"], dtype=bool)
                else:
                    mask = np.ones(n_raw, dtype=bool)

                if len(mask) != n_raw:
                    raise RuntimeError(f"{path}: mask length mismatch")

                mask = mask & finite
                n_unique_labeled = unique_labeled_count(labels, mask)

            rows.append({
                "fold": fold,
                "group_uid": uid,
                "population_n": n_finite,
                "raw_n": n_raw,
                "unique_labeled_n": n_unique_labeled,
                "path": str(path.resolve()),
            })

    frame = pd.DataFrame(rows).sort_values(["fold", "group_uid"]).reset_index(drop=True)
    if frame["group_uid"].duplicated().any():
        raise RuntimeError("Duplicate group_uid after fold audit")
    return frame


def load_long_scores(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {
        "query_uid", "group_uid", "gt_label",
        "candidate_label", "score", "assignment_score", "reference_uid",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")

    for col in ("query_uid", "group_uid", "gt_label", "candidate_label", "reference_uid"):
        frame[col] = frame[col].astype(str)

    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    frame["assignment_score"] = pd.to_numeric(frame["assignment_score"], errors="raise")

    if not np.isfinite(frame["score"]).all():
        raise RuntimeError(f"{path}: non-finite ranking score")
    if not np.isfinite(frame["assignment_score"]).all():
        raise RuntimeError(f"{path}: non-finite assignment score")
    return frame


def aggregate_references(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Match the final unified benchmark's reference_reducer='mean'.
    Metadata are invariant across repeated train references.
    """
    keys = ["query_uid", "group_uid", "gt_label", "candidate_label"]
    out = (
        frame.groupby(keys, as_index=False, sort=False)[["score", "assignment_score"]]
        .mean()
    )
    return out


def query_signature(frame: pd.DataFrame) -> tuple[tuple[str, str, str], ...]:
    x = frame[["query_uid", "group_uid", "gt_label"]].drop_duplicates()
    x = x.sort_values(["group_uid", "query_uid", "gt_label"])
    return tuple(map(tuple, x.astype(str).itertuples(index=False, name=None)))


@dataclass
class WormMetric:
    method: str
    seed: int
    fold: int
    group_uid: str
    queries: int
    top1_correct: int
    top5_correct: int
    rr_sum: float
    hungarian_correct: int
    hungarian_queries: int

    def row(self):
        q = max(self.queries, 1)
        hq = max(self.hungarian_queries, 1)
        return {
            "method": self.method,
            "seed": self.seed,
            "fold": self.fold,
            "group_uid": self.group_uid,
            "queries": self.queries,
            "top1_correct": self.top1_correct,
            "top5_correct": self.top5_correct,
            "rr_sum": self.rr_sum,
            "hungarian_correct": self.hungarian_correct,
            "hungarian_queries": self.hungarian_queries,
            "top1": self.top1_correct / q if self.queries else math.nan,
            "top5": self.top5_correct / q if self.queries else math.nan,
            "mrr": self.rr_sum / q if self.queries else math.nan,
            "hungarian": self.hungarian_correct / hq if self.hungarian_queries else math.nan,
        }


def compute_worm_metrics(frame: pd.DataFrame, method: str, seed: int, fold: int) -> list[WormMetric]:
    results = []

    for group_uid, group in frame.groupby("group_uid", sort=False):
        qmeta = group[["query_uid", "gt_label"]].drop_duplicates()
        query_ids = qmeta["query_uid"].astype(str).tolist()
        gt_map = dict(qmeta.astype(str).itertuples(index=False, name=None))

        top1 = top5 = 0
        rr_sum = 0.0

        for qid, qrows in group.groupby("query_uid", sort=False):
            gt = str(qrows["gt_label"].iloc[0])
            labels = qrows["candidate_label"].astype(str).to_numpy()
            scores = qrows["score"].to_numpy(dtype=float)
            idx = np.flatnonzero(labels == gt)

            # Final unified benchmark uses missing_gt_policy='incorrect':
            # absent GT remains in denominator with no credit.
            if len(idx) == 1:
                target = float(scores[int(idx[0])])
                rank = 1 + int(np.sum(scores > target))
                top1 += int(rank <= 1)
                top5 += int(rank <= min(5, len(scores)))
                rr_sum += 1.0 / rank
            elif len(idx) > 1:
                raise RuntimeError(f"{method} fold{fold} seed{seed}: duplicate GT candidate {gt} for {qid}")

        candidate_labels = sorted(group["candidate_label"].astype(str).unique())
        row_of = {qid: i for i, qid in enumerate(query_ids)}
        col_of = {lab: j for j, lab in enumerate(candidate_labels)}
        matrix = np.full((len(query_ids), len(candidate_labels)), -1e12, dtype=np.float64)

        for row in group.itertuples(index=False):
            qi = row_of[str(row.query_uid)]
            cj = col_of[str(row.candidate_label)]
            matrix[qi, cj] = max(matrix[qi, cj], float(row.assignment_score))

        ar, ac = linear_sum_assignment(-matrix)
        assignment = {int(r): int(c) for r, c in zip(ar, ac)}
        hh = 0
        for qid, qi in row_of.items():
            gt = gt_map[qid]
            pred_col = assignment.get(qi, -1)
            if pred_col >= 0 and candidate_labels[pred_col] == gt:
                hh += 1

        results.append(
            WormMetric(
                method=method,
                seed=seed,
                fold=fold,
                group_uid=str(group_uid),
                queries=len(query_ids),
                top1_correct=top1,
                top5_correct=top5,
                rr_sum=rr_sum,
                hungarian_correct=hh,
                hungarian_queries=len(query_ids),
            )
        )

    return results


def quantile_bins(meta: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    if n_bins < 2:
        raise ValueError("--bins must be >=2")

    x = meta.copy()
    # rank(method='first') prevents qcut failure when multiple worms share N.
    ranks = x["population_n"].rank(method="first")
    x["size_bin_index"] = pd.qcut(ranks, q=n_bins, labels=False).astype(int)

    labels = {}
    for b, part in x.groupby("size_bin_index", sort=True):
        lo = int(part["population_n"].min())
        hi = int(part["population_n"].max())
        med = float(part["population_n"].median())
        labels[int(b)] = f"Q{int(b)+1}: N={lo}-{hi} (median {med:g})"

    x["size_bin"] = x["size_bin_index"].map(labels)
    return x


def pooled_metrics(part: pd.DataFrame) -> dict[str, float]:
    q = int(part["queries"].sum())
    hq = int(part["hungarian_queries"].sum())
    return {
        "worms": int(part["group_uid"].nunique()),
        "queries": q,
        "mean_population_n": float(part.drop_duplicates("group_uid")["population_n"].mean()),
        "median_population_n": float(part.drop_duplicates("group_uid")["population_n"].median()),
        "top1": float(part["top1_correct"].sum() / q) if q else math.nan,
        "top5": float(part["top5_correct"].sum() / q) if q else math.nan,
        "mrr": float(part["rr_sum"].sum() / q) if q else math.nan,
        "hungarian": float(part["hungarian_correct"].sum() / hq) if hq else math.nan,
        "macro_worm_top1": float(part.groupby("group_uid")["top1"].mean().mean()),
    }


def cluster_bootstrap_top1(
    part: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    """
    Query-worm cluster bootstrap. If multiple model seeds are present, first pool
    counts within each worm across seeds, so resampling unit remains biological worm.
    """
    by_worm = (
        part.groupby("group_uid", as_index=False)[["top1_correct", "queries"]]
        .sum()
    )
    if by_worm.empty:
        return math.nan, math.nan

    hits = by_worm["top1_correct"].to_numpy(dtype=np.float64)
    qs = by_worm["queries"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(by_worm)
    vals = np.empty(iterations, dtype=np.float64)

    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        denom = qs[idx].sum()
        vals[i] = hits[idx].sum() / denom if denom > 0 else np.nan

    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def paired_bootstrap_delta(
    part_a: pd.DataFrame,
    part_b: pd.DataFrame,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """
    Paired worm bootstrap for pooled Top-1 delta A-B.
    """
    a = part_a.groupby("group_uid")[["top1_correct", "queries"]].sum()
    b = part_b.groupby("group_uid")[["top1_correct", "queries"]].sum()
    common = sorted(set(a.index) & set(b.index))
    if not common:
        return math.nan, math.nan, math.nan

    # Strict paired query universe implies denominators should agree by worm.
    aq = a.loc[common, "queries"].to_numpy(float)
    bq = b.loc[common, "queries"].to_numpy(float)
    if not np.array_equal(aq, bq):
        raise RuntimeError("Paired methods have different per-worm query denominators")

    ah = a.loc[common, "top1_correct"].to_numpy(float)
    bh = b.loc[common, "top1_correct"].to_numpy(float)

    point = float((ah.sum() - bh.sum()) / aq.sum())
    rng = np.random.default_rng(seed)
    n = len(common)
    vals = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        denom = aq[idx].sum()
        vals[i] = (ah[idx].sum() - bh[idx].sum()) / denom
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 118)
    print("RLD NATURAL POPULATION-SIZE ACCURACY SCALING — LOCKED CLEAN TEST")
    print("=" * 118)
    print("folds       :", args.folds)
    print("seeds       :", args.seeds)
    print("size bins   :", args.bins, "quantile bins over ORIGINAL natural test-worm N")
    print("bootstrap   :", args.bootstrap, "query-worm cluster resamples")
    print("corruption  : NONE")
    print("model rerun : NONE; reads locked candidate-score exports")
    print()

    meta = load_worm_metadata(args.folds)
    meta = quantile_bins(meta, args.bins)
    meta.to_csv(args.output_dir / "worm_metadata.csv", index=False)

    print("N distribution:")
    print(meta["population_n"].describe().to_string())
    print()
    for b, p in meta.groupby(["size_bin_index", "size_bin"], sort=True):
        print(
            f"  bin{int(b[0])}: {b[1]} | worms={len(p)} "
            f"meanN={p.population_n.mean():.1f}"
        )

    all_metrics = []
    file_audit = []
    signatures: dict[tuple[int, int, str], tuple] = {}

    for fold in args.folds:
        for seed in args.seeds:
            for method, base in METHODS.items():
                path = base / f"fold{fold}" / f"seed{seed}" / "test_candidate_scores.csv"
                frame_raw = load_long_scores(path)
                frame = aggregate_references(frame_raw)
                sig = query_signature(frame)
                signatures[(fold, seed, method)] = sig

                file_audit.append({
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "path": str(path.resolve()),
                    "sha256": sha256(path),
                    "raw_rows": len(frame_raw),
                    "aggregated_rows": len(frame),
                    "queries": len(sig),
                })

                for wm in compute_worm_metrics(frame, method, seed, fold):
                    all_metrics.append(wm.row())

            # Hard guard: exact same query universe.
            sig_m = signatures[(fold, seed, "MPRT")]
            sig_f = signatures[(fold, seed, "fDNC")]
            if sig_m != sig_f:
                sm, sf = set(sig_m), set(sig_f)
                raise RuntimeError(
                    f"QUERY SIGNATURE MISMATCH fold={fold} seed={seed}: "
                    f"MPRT={len(sig_m)} fDNC={len(sig_f)} "
                    f"MPRT-only={list(sorted(sm-sf))[:5]} "
                    f"fDNC-only={list(sorted(sf-sm))[:5]}"
                )
            print(
                f"fold{fold} seed{seed}: exact MPRT/fDNC query signature "
                f"Q={len(sig_m)} ✓"
            )

    metrics = pd.DataFrame(all_metrics)
    metrics = metrics.merge(
        meta[
            [
                "fold", "group_uid", "population_n", "raw_n",
                "unique_labeled_n", "size_bin_index", "size_bin",
            ]
        ],
        on=["fold", "group_uid"],
        how="left",
        validate="many_to_one",
    )

    if metrics["population_n"].isna().any():
        bad = metrics.loc[metrics["population_n"].isna(), ["fold", "group_uid"]].drop_duplicates()
        raise RuntimeError(f"Could not map score groups to current test NPZs:\n{bad}")

    metrics.to_csv(args.output_dir / "worm_metrics.csv", index=False)
    pd.DataFrame(file_audit).to_csv(args.output_dir / "score_file_audit.csv", index=False)

    summary_rows = []
    for method in METHODS:
        for bidx, part in metrics[metrics["method"] == method].groupby("size_bin_index", sort=True):
            out = pooled_metrics(part)
            lo, hi = cluster_bootstrap_top1(
                part,
                iterations=args.bootstrap,
                seed=args.bootstrap_seed + 1000 * int(bidx) + (0 if method == "MPRT" else 100),
            )
            summary_rows.append({
                "method": method,
                "size_bin_index": int(bidx),
                "size_bin": str(part["size_bin"].iloc[0]),
                **out,
                "top1_ci95_low": lo,
                "top1_ci95_high": hi,
            })

    summary = pd.DataFrame(summary_rows).sort_values(["size_bin_index", "method"])
    summary.to_csv(args.output_dir / "scaling_summary.csv", index=False)

    delta_rows = []
    for bidx in sorted(metrics["size_bin_index"].unique()):
        a = metrics[(metrics["method"] == "MPRT") & (metrics["size_bin_index"] == bidx)]
        b = metrics[(metrics["method"] == "fDNC") & (metrics["size_bin_index"] == bidx)]
        point, lo, hi = paired_bootstrap_delta(
            a, b, args.bootstrap, args.bootstrap_seed + 5000 + int(bidx)
        )
        delta_rows.append({
            "size_bin_index": int(bidx),
            "size_bin": str(a["size_bin"].iloc[0]),
            "mprt_minus_fdnc_top1": point,
            "ci95_low": lo,
            "ci95_high": hi,
            "worms": int(a["group_uid"].nunique()),
        })

    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(args.output_dir / "paired_delta_summary.csv", index=False)

    corr_rows = []
    # Average model-seed worm accuracy before correlation, preserving worm as unit.
    worm_seed_avg = (
        metrics.groupby(["method", "group_uid", "population_n"], as_index=False)
        .agg(top1=("top1", "mean"), hungarian=("hungarian", "mean"))
    )
    for method, part in worm_seed_avg.groupby("method"):
        for metric in ("top1", "hungarian"):
            rho, p = spearmanr(part["population_n"], part[metric], nan_policy="omit")
            corr_rows.append({
                "method": method,
                "metric": metric,
                "worms": len(part),
                "spearman_rho": float(rho),
                "p_value": float(p),
            })

    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(args.output_dir / "correlations.csv", index=False)

    protocol = {
        "protocol": "RLD natural population-size accuracy scaling current-v2",
        "dataset": str(CV_ROOT.resolve()),
        "folds": args.folds,
        "seeds": args.seeds,
        "methods": list(METHODS),
        "fdnc_variant": "current-grouped fine-tuned seed42, fixed outer-train geometry medoid",
        "mprt_variant": "final Static Atlas anchored_pure",
        "population_size": "original finite xyz rows in each clean held-out test worm",
        "binning": f"{args.bins} equal-count quantile bins over biological test worms",
        "no_corruption": True,
        "no_subsampling": True,
        "no_duplication": True,
        "no_retraining": True,
        "no_test_time_tuning": True,
        "query_signature_guard": "MPRT and fDNC must be exactly identical within every fold x seed",
        "primary_metric": "pooled out-of-fold Top-1 within natural-N bin",
        "ci": f"{args.bootstrap}-iteration biological query-worm cluster bootstrap",
        "paired_delta": "MPRT - fDNC, paired biological-worm bootstrap",
        "note": (
            "This analysis is distinct from missing-neuron robustness: every worm "
            "is left intact and is grouped only by its naturally observed population size."
        ),
        "score_files": file_audit,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 118)
    print("NATURAL-N ACCURACY SCALING")
    print("=" * 118)
    for bidx in sorted(summary["size_bin_index"].unique()):
        sub = summary[summary["size_bin_index"] == bidx]
        print(f"\n{sub['size_bin'].iloc[0]}")
        for row in sub.itertuples(index=False):
            print(
                f"  {row.method:<6s} worms={row.worms:2d} Q={row.queries:4d} "
                f"Top1={100*row.top1:6.2f}% "
                f"CI=[{100*row.top1_ci95_low:6.2f},{100*row.top1_ci95_high:6.2f}] "
                f"Top5={100*row.top5:6.2f}% "
                f"MRR={row.mrr:.4f} "
                f"Hung={100*row.hungarian:6.2f}%"
            )
        d = deltas[deltas["size_bin_index"] == bidx].iloc[0]
        print(
            f"  Δ MPRT-fDNC Top1 = {100*d.mprt_minus_fdnc_top1:+6.2f} pp "
            f"95% CI [{100*d.ci95_low:+6.2f},{100*d.ci95_high:+6.2f}]"
        )

    print()
    print("=" * 118)
    print("SPEARMAN NATURAL-N ASSOCIATION")
    print("=" * 118)
    for row in correlations.itertuples(index=False):
        print(
            f"{row.method:<6s} {row.metric:<10s} "
            f"rho={row.spearman_rho:+.3f} p={row.p_value:.4g} worms={row.worms}"
        )

    print()
    print("Outputs:", args.output_dir)
    print("DONE")


if __name__ == "__main__":
    main()
