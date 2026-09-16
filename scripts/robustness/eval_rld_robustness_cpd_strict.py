#!/usr/bin/env python3
"""
Strict RLD controlled-corruption evaluation for CPD.

Key principle:
    DO NOT reimplement CPD or the identity-matching protocol here.
    Each corruption condition is evaluated by the exact clean benchmark runner:
        evaluate_train_reference_ensemble.py
    with:
        --method cpd
        --normalization zscore

Only the held-out test NPZs differ across corruption conditions.  The clean
outer-training split, train-only geometry medoid selection, CPD adapter,
identity vocabulary, tie policy, and metric implementation therefore remain
identical to the main benchmark.

For missing-neuron experiments, this wrapper additionally reports the
reference-covered diagnostic:
    coverage_vs_clean_eligible = surviving eligible queries / clean eligible queries
    effective_top1            = top1 * surviving eligible queries / clean eligible queries

Every fold first replays severity=0 and MUST exactly reproduce the saved clean
CPD benchmark before any nonzero corruption is evaluated.

The formal identity accuracy is rescored by
``summarize_rld_robustness_hierarchical.py`` on the canonical main-table query
cohort.  These native fields must not be used as the cross-method denominator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
BASE_RUNNER = ROOT / "scripts/fair_identity/evaluate_train_reference_ensemble.py"
CORR_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions"
CLEAN_ROOT = ROOT / "runs/fair_identity_medoid_template_v1/cpd/rld"
OUT_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v2/results/cpd"

COND_RE = re.compile(
    r"^(coord_noise|missing|outlier)_l([0-9]+(?:\.[0-9]+)?)_p([0-9]+)$"
)


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jdump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def discover_conditions(fold: int, kinds: set[str]) -> list[dict]:
    fold_root = CORR_ROOT / f"fold{fold}"
    if not fold_root.is_dir():
        raise FileNotFoundError(fold_root)

    rows = []
    for path in sorted(fold_root.iterdir()):
        if not path.is_dir():
            continue
        m = COND_RE.match(path.name)
        if not m:
            continue
        kind, sev_s, p_s = m.groups()
        if kind not in kinds:
            continue
        if not (path / "test").is_dir():
            raise FileNotFoundError(path / "test")
        rows.append(
            {
                "fold": fold,
                "kind": kind,
                "severity": float(sev_s),
                "perturbation_seed": int(p_s),
                "name": path.name,
                "root": path,
            }
        )

    order = {"coord_noise": 0, "missing": 1, "outlier": 2}
    rows.sort(
        key=lambda r: (
            order[r["kind"]],
            r["severity"],
            r["perturbation_seed"],
        )
    )
    return rows


def find_clean_condition(fold: int) -> dict:
    path = CORR_ROOT / f"fold{fold}" / "coord_noise_l0.00_p0"
    if not (path / "train").exists():
        raise FileNotFoundError(
            f"{path}/train is missing. The corruption benchmark must expose the "
            "unchanged clean outer-train split for exact medoid replay."
        )
    if not (path / "test").is_dir():
        raise FileNotFoundError(path / "test")
    return {
        "fold": fold,
        "kind": "coord_noise",
        "severity": 0.0,
        "perturbation_seed": 0,
        "name": path.name,
        "root": path,
    }


def saved_clean(fold: int) -> dict:
    path = CLEAN_ROOT / f"fold{fold}" / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    report = jload(path)
    try:
        metric = report["metrics"]["template_score"]
    except KeyError as exc:
        raise KeyError(f"{path}: missing metrics.template_score") from exc
    return {
        "path": path,
        "report": report,
        "metric": metric,
    }


def validate_training_split(condition_root: Path, saved_report: dict):
    """
    Lightweight structural check only.

    Corruption materialization renames files (e.g.
    train__<uid>__<hash>.npz), while the original clean report stores source
    basenames such as <uid>.npz.  Therefore basename equality is NOT a valid
    split-provenance check.

    The strict provenance guard is performed after replay:
      1) the original clean runner recomputes the train-only geometry medoid;
      2) replay template_uid must equal the saved clean template_uid;
      3) Q/Top1/Top5/MRR/Hungarian must match exactly.
    """
    train_dir = condition_root / "train"
    train_files = sorted(train_dir.glob("*.npz"))
    if not train_files:
        raise RuntimeError(f"No train NPZs in {train_dir}")

    expected_n = saved_report.get("training_animals")
    if expected_n is not None and len(train_files) != int(expected_n):
        raise RuntimeError(
            f"Training split size mismatch for {condition_root}: "
            f"materialized={len(train_files)} saved_clean={expected_n}. "
            "Refuse to evaluate a different outer-train split."
        )


def run_original_runner(row: dict, force: bool, workers: int = 1) -> dict:
    cond = row["name"]
    fold = row["fold"]
    dst = OUT_ROOT / f"fold{fold}" / cond
    raw_out = dst / "original_runner"
    cache_dir = dst / "score_cache"
    metrics_path = raw_out / "metrics.json"

    if metrics_path.is_file() and not force:
        return jload(metrics_path)

    raw_out.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-u",
        str(BASE_RUNNER),
        "--method",
        "cpd",
        "--method-label",
        "CPD",
        "--fold-root",
        str(row["root"]),
        "--normalization",
        "zscore",
        "--device",
        "cpu",
        "--workers",
        str(workers),
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(raw_out),
    ]

    print("\n" + "=" * 100, flush=True)
    print(
        f"[RUN] fold{fold} {cond} using ORIGINAL clean CPD runner",
        flush=True,
    )
    print(" ".join(cmd), flush=True)
    print("=" * 100, flush=True)

    subprocess.run(cmd, cwd=ROOT, check=True)

    if not metrics_path.is_file():
        raise RuntimeError(f"Original runner did not create {metrics_path}")
    return jload(metrics_path)


def metric_block(report: dict) -> dict:
    try:
        return report["metrics"]["template_score"]
    except KeyError as exc:
        raise KeyError("missing metrics.template_score") from exc


def clean_guard(fold: int, replay_report: dict, expected: dict):
    replay = metric_block(replay_report)
    saved = expected["metric"]

    keys = ("queries", "top1", "top5", "mrr", "hungarian_accuracy")
    diffs = {}
    ok = True
    for k in keys:
        a = replay[k]
        b = saved[k]
        if k == "queries":
            same = int(a) == int(b)
            diff = abs(int(a) - int(b))
        else:
            same = abs(float(a) - float(b)) <= 1e-12
            diff = abs(float(a) - float(b))
        diffs[k] = {"saved": b, "replay": a, "abs_diff": diff}
        ok &= same

    saved_template = expected["report"].get("template_selection", {}).get("template_uid")
    replay_template = replay_report.get("template_selection", {}).get("template_uid")
    if saved_template is not None or replay_template is not None:
        same_template = saved_template == replay_template
        ok &= same_template
    else:
        same_template = True

    print(
        f"[CLEAN GUARD {'EXACT' if ok else 'MISMATCH'}] fold{fold} "
        f"Q={replay['queries']} "
        f"Top1={100*replay['top1']:.4f}% "
        f"Top5={100*replay['top5']:.4f}% "
        f"MRR={replay['mrr']:.6f} "
        f"Hung={100*replay['hungarian_accuracy']:.4f}% "
        f"template={replay_template}",
        flush=True,
    )

    return {
        "status": "exact" if ok else "mismatch",
        "saved_metrics_path": str(expected["path"]),
        "saved_template_uid": saved_template,
        "replay_template_uid": replay_template,
        "template_exact": same_template,
        "diffs": diffs,
    }


def augment_result(row: dict, report: dict, clean_queries: int, guard=None) -> dict:
    m = metric_block(report)
    q = int(m["queries"])
    clean_q = int(clean_queries)

    # evaluate_identity_scores already restricts to identities representable by
    # the fixed medoid template.  Thus q is the number of surviving eligible
    # real queries under the exact clean benchmark protocol.
    coverage = q / clean_q if clean_q else float("nan")
    effective_top1 = float(m["top1"]) * q / clean_q if clean_q else float("nan")
    effective_top5 = float(m["top5"]) * q / clean_q if clean_q else float("nan")
    effective_hung = (
        float(m["hungarian_accuracy"]) * q / clean_q if clean_q else float("nan")
    )

    out = {
        "method": "CPD",
        "fold": int(row["fold"]),
        "condition": row["name"],
        "kind": row["kind"],
        "severity": float(row["severity"]),
        "perturbation_seed": int(row["perturbation_seed"]),
        "protocol": {
            "base_runner": str(BASE_RUNNER),
            "base_runner_method": "cpd",
            "base_runner_normalization": "zscore",
            "template_policy": "outer-training geometry medoid; recomputed by original runner",
            "retrained_on_corruption": False,
            "test_only_corruption": True,
            "condition_specific_score_cache": True,
            "eligible_query_definition":
                "exact evaluate_identity_scores query universe induced by fixed medoid vocabulary",
            "missing_coverage_denominator":
                "severity=0 eligible queries in the same biological fold",
            "outlier_note":
                "synthetic distractors enter CPD registration input; unsupervised/outlier labels are not eligible GT queries",
        },
        "original_runner_metrics": m,
        "clean_eligible_queries": clean_q,
        "queries": q,
        "top1": float(m["top1"]),
        "top5": float(m["top5"]),
        "mrr": float(m["mrr"]),
        "hungarian": float(m["hungarian_accuracy"]),
        "coverage_vs_clean_eligible": float(coverage),
        "effective_top1": float(effective_top1),
        "effective_top5": float(effective_top5),
        "effective_hungarian": float(effective_hung),
        "original_runner_output": str(
            OUT_ROOT / f"fold{row['fold']}" / row["name"] / "original_runner"
        ),
    }
    if guard is not None:
        out["clean_replay_guard"] = guard

    path = OUT_ROOT / f"fold{row['fold']}" / row["name"] / "robustness_metrics.json"
    jdump(path, out)

    print(
        f"[CPD] fold{row['fold']} {row['name']}: "
        f"Q={q}/{clean_q} "
        f"Top1={100*out['top1']:.2f}% "
        f"Top5={100*out['top5']:.2f}% "
        f"Hung={100*out['hungarian']:.2f}% "
        f"Coverage={100*out['coverage_vs_clean_eligible']:.2f}% "
        f"EffectiveTop1={100*out['effective_top1']:.2f}%",
        flush=True,
    )
    return out


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def summarize(results: list[dict]):
    flat = []
    for r in results:
        flat.append(
            {
                "method": "CPD",
                "fold": r["fold"],
                "kind": r["kind"],
                "severity": r["severity"],
                "perturbation_seed": r["perturbation_seed"],
                "queries": r["queries"],
                "clean_eligible_queries": r["clean_eligible_queries"],
                "top1": r["top1"],
                "top5": r["top5"],
                "mrr": r["mrr"],
                "hungarian": r["hungarian"],
                "coverage": r["coverage_vs_clean_eligible"],
                "effective_top1": r["effective_top1"],
                "effective_top5": r["effective_top5"],
                "effective_hungarian": r["effective_hungarian"],
            }
        )
    write_csv(OUT_ROOT / "cpd_all_cells.csv", flat)

    # Mean perturbation seeds inside each biological fold.
    metric_names = (
        "top1",
        "top5",
        "mrr",
        "hungarian",
        "coverage",
        "effective_top1",
        "effective_top5",
        "effective_hungarian",
    )
    groups = {}
    for r in flat:
        key = (r["kind"], r["severity"], r["fold"])
        groups.setdefault(key, []).append(r)

    fold_rows = []
    for (kind, sev, fold), items in sorted(groups.items()):
        row = {
            "method": "CPD",
            "kind": kind,
            "severity": sev,
            "fold": fold,
            "perturbation_seeds": len(items),
        }
        for k in metric_names:
            vals = np.asarray([float(x[k]) for x in items], dtype=np.float64)
            row[k] = float(vals.mean())
        fold_rows.append(row)
    write_csv(OUT_ROOT / "cpd_fold_level.csv", fold_rows)

    # Biological-fold macro mean ± sample SD.
    macro_groups = {}
    for r in fold_rows:
        key = (r["kind"], r["severity"])
        macro_groups.setdefault(key, []).append(r)

    macro_rows = []
    for (kind, sev), items in sorted(macro_groups.items()):
        row = {
            "method": "CPD",
            "kind": kind,
            "severity": sev,
            "folds": len(items),
        }
        for k in metric_names:
            vals = np.asarray([float(x[k]) for x in items], dtype=np.float64)
            row[f"{k}_mean"] = float(vals.mean())
            row[f"{k}_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        macro_rows.append(row)
    write_csv(OUT_ROOT / "cpd_macro_summary.csv", macro_rows)

    print("\n" + "=" * 100)
    print("CPD ROBUSTNESS MACRO SUMMARY")
    print("=" * 100)
    for r in macro_rows:
        print(
            f"{r['kind']:12s} level={r['severity']:.2f} "
            f"Top1={100*r['top1_mean']:.2f}±{100*r['top1_sd']:.2f}% "
            f"Hung={100*r['hungarian_mean']:.2f}±{100*r['hungarian_sd']:.2f}% "
            f"Cov={100*r['coverage_mean']:.2f}±{100*r['coverage_sd']:.2f}% "
            f"EffTop1={100*r['effective_top1_mean']:.2f}±{100*r['effective_top1_sd']:.2f}%"
        )


def main():
    global CORR_ROOT, OUT_ROOT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument(
        "--kinds",
        default="coord_noise,missing,outlier",
        help="comma-separated subset of coord_noise,missing,outlier",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="rerun conditions even if original_runner/metrics.json already exists",
    )
    ap.add_argument("--corruption-root", type=Path, default=CORR_ROOT)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    CORR_ROOT = args.corruption_root.resolve()
    OUT_ROOT = args.out_root.resolve()

    if not BASE_RUNNER.is_file():
        raise FileNotFoundError(BASE_RUNNER)

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    valid = {"coord_noise", "missing", "outlier"}
    if not kinds or not kinds <= valid:
        raise ValueError(f"--kinds must be subset of {sorted(valid)}, got {sorted(kinds)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for fold in folds:
        expected = saved_clean(fold)
        clean_row = find_clean_condition(fold)
        validate_training_split(clean_row["root"], expected["report"])

        # Always perform severity=0 replay first, even when coord_noise was not
        # requested, because every nonzero corruption depends on this guard.
        clean_report = run_original_runner(clean_row, force=args.force, workers=args.workers)
        guard = clean_guard(fold, clean_report, expected)
        clean_queries = int(expected["metric"]["queries"])
        clean_result = augment_result(
            clean_row, clean_report, clean_queries=clean_queries, guard=guard
        )

        if guard["status"] != "exact":
            raise RuntimeError(
                f"fold{fold}: severity=0 failed exact clean replay. "
                "STOP before evaluating any nonzero corruption."
            )

        results.append(clean_result)

        rows = discover_conditions(fold, kinds)
        for row in rows:
            if (
                row["kind"] == "coord_noise"
                and abs(row["severity"]) <= 1e-12
                and row["perturbation_seed"] == 0
            ):
                continue

            # A condition must use the same clean outer-train split.  This also
            # catches accidentally regenerated corruptions with a different CV split.
            validate_training_split(row["root"], expected["report"])

            report = run_original_runner(row, force=args.force, workers=args.workers)
            expected_template = expected["report"]["template_selection"]["template_uid"]
            observed_template = report["template_selection"]["template_uid"]
            if observed_template != expected_template:
                raise RuntimeError(
                    f"fold{fold} {row['name']}: reference changed from "
                    f"{expected_template} to {observed_template}"
                )
            result = augment_result(
                row, report, clean_queries=clean_queries, guard=None
            )
            results.append(result)

    summarize(results)

    print("\nCOMPLETE")
    print(f"results: {OUT_ROOT}")
    print(f"macro:   {OUT_ROOT / 'cpd_macro_summary.csv'}")


if __name__ == "__main__":
    main()
