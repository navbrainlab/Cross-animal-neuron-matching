#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.lib.benchmark_cv5x3_common as common

OUT = ROOT / "baselines/official/results"
RUN = ROOT / "baselines/official/runs/euclidean_locked"
OUT.mkdir(parents=True, exist_ok=True)
RUN.mkdir(parents=True, exist_ok=True)

DATASETS = ("atanas", "rld")
FOLDS = range(1, 6)
SEEDS = (1, 42, 123)


def worm_ids(records):
    return [str(r.worm_id) for r in records]


def get_xyz(record) -> np.ndarray:
    if not hasattr(record, "xyz"):
        raise AttributeError(
            f"record {getattr(record, 'worm_id', '?')} has no .xyz field; "
            "the locked Candidate-B loader is expected to expose worm-wise "
            "normalized XYZ."
        )

    xyz = record.xyz
    if torch.is_tensor(xyz):
        xyz = xyz.detach().cpu().numpy()
    xyz = np.asarray(xyz, dtype=np.float64)

    if xyz.ndim != 2 or xyz.shape[0] != len(record.labels) or xyz.shape[1] < 3:
        raise ValueError(
            f"{record.worm_id}: xyz shape={xyz.shape}, "
            f"labels={len(record.labels)}"
        )

    xyz = xyz[:, :3]
    if not np.isfinite(xyz).all():
        raise ValueError(f"{record.worm_id}: XYZ contains NaN/Inf")
    return xyz


def euclidean_score(a, b) -> np.ndarray:
    xa = get_xyz(a)
    xb = get_xyz(b)
    delta = xa[:, None, :] - xb[None, :, :]
    # Higher score is better, matching common.evaluate_score_matrix().
    return -np.sum(delta * delta, axis=-1)


all_queries = []
fold_rows = []
protocol_audit = []

for dataset in DATASETS:
    print("=" * 100)
    print(dataset.upper())
    print("=" * 100)

    for fold in FOLDS:
        # Deterministic baseline: use seed42 only for data loading, but first
        # verify that all three seed routes point to the same biological test set.
        bundles = {
            seed: common.split_bundle(dataset, fold, seed, "cpu")
            for seed in SEEDS
        }
        ids = {seed: worm_ids(bundle["test"]) for seed, bundle in bundles.items()}
        seed_invariant = ids[1] == ids[42] == ids[123]
        if not seed_invariant:
            raise RuntimeError(
                f"{dataset} fold={fold}: biological test worms differ across seeds: {ids}"
            )

        records = bundles[42]["test"]

        qdf = common.evaluate_pairwise(
            records,
            euclidean_score,
            dataset=dataset,
            fold=fold,
            seed=-1,
            method="Euclidean",
        )
        if qdf.empty:
            raise RuntimeError(f"{dataset} fold={fold}: zero eligible queries")

        run_dir = RUN / dataset / f"fold_{fold}"
        run_dir.mkdir(parents=True, exist_ok=True)
        qdf.to_csv(run_dir / "query_level.csv", index=False)

        metrics = {
            "dataset": dataset,
            "fold": int(fold),
            "queries": int(len(qdf)),
            "top1": float(qdf["top1"].mean()),
            "top3": float(qdf["top3"].mean()),
            "top5": float(qdf["top5"].mean()),
            "mrr": float(qdf["rr"].mean()),
            "hungarian_top1": float(qdf["hungarian_top1"].mean()),
            "test_worms": len(records),
            "seed_invariant_test_worms": True,
        }
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )

        fold_rows.append(metrics)
        all_queries.append(qdf)
        protocol_audit.append({
            "dataset": dataset,
            "fold": fold,
            "test_worms": ids[42],
            "seed_invariant_test_worms": seed_invariant,
        })

        print(
            f"fold={fold} test_worms={len(records):2d} queries={len(qdf):5d} | "
            f"Top1={100*metrics['top1']:6.2f}% | "
            f"Top5={100*metrics['top5']:6.2f}% | "
            f"Hungarian={100*metrics['hungarian_top1']:6.2f}% | "
            f"MRR={metrics['mrr']:.4f}"
        )

fold_df = pd.DataFrame(fold_rows)
query_df = pd.concat(all_queries, ignore_index=True)

fold_df.to_csv(OUT / "euclidean_locked_fold_metrics.csv", index=False)
query_df.to_csv(OUT / "euclidean_locked_query_level.csv", index=False)
(OUT / "euclidean_locked_protocol_audit.json").write_text(
    json.dumps(protocol_audit, indent=2) + "\n",
    encoding="utf-8",
)

summary_rows = []

print("\n" + "=" * 100)
print("FINAL EUCLIDEAN — PAPER STATISTICS")
print("Deterministic baseline; mean ± sample SD across 5 biological outer folds")
print("=" * 100)

for dataset in DATASETS:
    g = fold_df[fold_df["dataset"] == dataset].sort_values("fold")
    if len(g) != 5:
        raise RuntimeError(f"{dataset}: expected 5 folds, got {len(g)}")

    row = {"dataset": dataset, "folds": 5}
    for metric in ("top1", "top3", "top5", "hungarian_top1", "mrr"):
        x = g[metric].to_numpy(dtype=float)
        row[f"{metric}_mean"] = float(x.mean())
        row[f"{metric}_sd"] = float(x.std(ddof=1))
    summary_rows.append(row)

    print(f"\n{dataset.upper()}")
    print(
        f"Top1      = {100*row['top1_mean']:.2f}% ± "
        f"{100*row['top1_sd']:.2f}%"
    )
    print(
        f"Top5      = {100*row['top5_mean']:.2f}% ± "
        f"{100*row['top5_sd']:.2f}%"
    )
    print(
        f"Hungarian = {100*row['hungarian_top1_mean']:.2f}% ± "
        f"{100*row['hungarian_top1_sd']:.2f}%"
    )
    print(
        f"MRR       = {row['mrr_mean']:.4f} ± "
        f"{row['mrr_sd']:.4f}"
    )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUT / "euclidean_locked_paper_summary.csv", index=False)

print("\nSaved:")
print(OUT / "euclidean_locked_fold_metrics.csv")
print(OUT / "euclidean_locked_query_level.csv")
print(OUT / "euclidean_locked_paper_summary.csv")
print(OUT / "euclidean_locked_protocol_audit.json")
