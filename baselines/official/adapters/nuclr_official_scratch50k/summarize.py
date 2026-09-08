#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "baselines/official/runs/nuclr_official_scratch50k_cv5x3"
OUT = ROOT / "baselines/official/results"
OUT.mkdir(parents=True, exist_ok=True)

rows = []
missing = []
for ds in ("atanas", "rld"):
    for fold in range(1, 6):
        for seed in (1, 42, 123):
            base = RUN / ds / f"fold_{fold}" / f"seed_{seed}"
            p = base / "outer_test_medoid_template_v1" / "result.json"
            s = base / "selected" / "selection.json"
            if not p.is_file() or not s.is_file():
                missing.append(str(p))
                continue
            x = json.loads(p.read_text())
            sel = json.loads(s.read_text())
            m = x["metrics"]
            rows.append({
                "dataset": ds,
                "fold": fold,
                "seed": seed,
                "selected_global_step": int(x["selected_global_step"]),
                "final_global_step": int(x["final_global_step"]),
                "setup_steps_per_epoch": int(sel["setup_steps_per_epoch"]),
                "queries": int(m["queries"]),
                "top1": float(m["ranking_top1"]),
                "top3": float(m["top3"]),
                "top5": float(m["top5"]),
                "top10": float(m["top10"]),
                "hungarian": float(m["assignment_top1"]),
                "mrr": float(m["mrr"]),
                "mean_rank": float(m["mean_rank"]),
            })

if missing:
    print(f"Completed {len(rows)}/30")
    print("Missing:")
    for x in missing:
        print(" ", x)
    raise SystemExit(2)

df = pd.DataFrame(rows)
if not (df.final_global_step == 50000).all():
    bad = df[df.final_global_step != 50000]
    raise RuntimeError(f"Non-50k completed runs:\n{bad}")

# Exactly the manuscript convention: average the 3 seeds within each biological
# outer fold, then report mean ± sample SD across the 5 folds.
fold_df = (
    df.groupby(["dataset", "fold"], as_index=False)
      .agg(
          seeds=("seed", "nunique"),
          top1=("top1", "mean"),
          top3=("top3", "mean"),
          top5=("top5", "mean"),
          top10=("top10", "mean"),
          hungarian=("hungarian", "mean"),
          mrr=("mrr", "mean"),
          mean_rank=("mean_rank", "mean"),
      )
)

paper_rows = []
print("=" * 112)
print("FINAL NuCLR — OFFICIAL CALCIUM OBJECTIVE, RANDOM INIT, 50k OPTIMIZER STEPS")
print("3 seeds averaged within fold; mean ± sample SD across 5 biological outer folds")
print("=" * 112)
for ds in ("atanas", "rld"):
    g = fold_df[fold_df.dataset == ds].sort_values("fold")
    if len(g) != 5 or not (g.seeds == 3).all():
        raise RuntimeError(f"{ds}: incomplete fold/seed aggregation")
    out = {"dataset": ds}
    for metric in ("top1", "top3", "top5", "top10", "hungarian", "mrr", "mean_rank"):
        arr = g[metric].to_numpy(float)
        out[metric + "_mean"] = float(arr.mean())
        out[metric + "_sd"] = float(arr.std(ddof=1))
    paper_rows.append(out)

    print(f"\n{ds.upper()}")
    print(f"Top1      = {100*out['top1_mean']:.2f}% ± {100*out['top1_sd']:.2f}%")
    print(f"Top5      = {100*out['top5_mean']:.2f}% ± {100*out['top5_sd']:.2f}%")
    print(f"Hungarian = {100*out['hungarian_mean']:.2f}% ± {100*out['hungarian_sd']:.2f}%")
    print(f"MRR       = {out['mrr_mean']:.4f} ± {out['mrr_sd']:.4f}")
    d = df[df.dataset == ds]
    print(
        "Validation-selected checkpoint step: "
        f"median={float(d.selected_global_step.median()):.0f}, "
        f"range={int(d.selected_global_step.min())}-{int(d.selected_global_step.max())}"
    )

stem = "nuclr_official_scratch50k_medoid_template_v1"
df.to_csv(OUT / f"{stem}_all_runs.csv", index=False)
fold_df.to_csv(OUT / f"{stem}_fold_collapsed.csv", index=False)
pd.DataFrame(paper_rows).to_csv(OUT / f"{stem}_paper_summary.csv", index=False)
print("\nSaved under:", OUT)
