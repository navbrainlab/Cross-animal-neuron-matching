#!/usr/bin/env python3
"""Paired RLD corruption replay for the seed-42 static identity Atlas."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORR = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions"
DEFAULT_OUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/results/ours_static_seed42"
RUNS = ROOT / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld"
EVALUATOR = ROOT / "scripts/mprt/evaluate_mprt_static_atlas.py"
PACKAGE = ROOT / "mprt_net_v1_1"
KINDS = {"coord_noise", "activity_noise", "missing", "outlier"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint(fold: int) -> Path:
    path = RUNS / f"fold{fold}/seed42/dynamic/low_rank_r8/best.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def saved_clean(fold: int):
    path = RUNS / f"fold{fold}/seed42/atlas_identity_test/metrics.json"
    report = read_json(path)
    return path, report["modes"]["static"]


def conditions(manifest: dict, fold: int, kinds: set[str], only_clean: bool = False):
    rows = [
        x for x in manifest["conditions"]
        if int(x["fold"]) == fold
        and str(x["kind"]) in kinds
        and (not only_clean or float(x["severity"]) == 0.0)
    ]
    order = {"coord_noise": 0, "activity_noise": 1, "missing": 2, "outlier": 3}
    return sorted(rows, key=lambda x: (
        order[str(x["kind"])], float(x["severity"]), int(x["perturbation_seed"])
    ))


def evaluate(row: dict, fold: int, ckpt: Path, out_root: Path, device: str, force: bool):
    name = Path(row["root"]).name
    dst = out_root / f"fold{fold}/{name}"
    metrics = dst / "metrics.json"
    queries = dst / "queries.csv"
    if metrics.is_file() and not force:
        return read_json(metrics)
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(EVALUATOR),
        "--package-root", str(PACKAGE),
        "--dataset-root", str(Path(row["root"])),
        "--split", "test",
        "--checkpoint", str(ckpt),
        "--device", device,
        "--dataset", "rld",
        "--fold", str(fold),
        "--seed", "42",
        "--variant", "static_identity_atlas_paired_robustness",
        "--output", str(metrics),
        "--query-output", str(queries),
    ]
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)
    return read_json(metrics)


def exact_clean_guard(observed: dict, expected: dict, fold: int):
    mapping = {
        "queries": "queries", "top1_real": "top1_real", "top5_real": "top5_real",
        "mrr_real": "mrr_real", "hungarian_accuracy": "hungarian_accuracy",
    }
    bad = {}
    # Ranking decisions are discrete and must reproduce exactly. MRR is a mean
    # of reciprocal ranks and can differ at the last few bits across CPU/GPU
    # reduction backends, so it alone receives a documented numerical tolerance.
    tolerances = {"mrr_real": 5e-8}
    for got, want in mapping.items():
        a, b = observed[got], expected[want]
        if (got == "queries" and int(a) != int(b)) or (
            got != "queries"
            and abs(float(a) - float(b)) > tolerances.get(got, 1e-12)
        ):
            bad[got] = {"observed": a, "expected": b}
    if bad:
        raise RuntimeError(f"fold{fold} Ours severity-zero mismatch: {bad}")
    print(f"[CLEAN GUARD PASSED] fold{fold} (MRR atol=5e-8)", flush=True)


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def summarize(rows: list[dict], out_root: Path):
    metrics = ("top1", "top5", "mrr", "hungarian", "coverage", "effective_top1")
    write_csv(out_root / "ours_all_cells.csv", rows)
    fold_rows = []
    for key in sorted({(r["kind"], r["severity"], r["fold"]) for r in rows}):
        part = [r for r in rows if (r["kind"], r["severity"], r["fold"]) == key]
        item = {"method": "Ours static Atlas", "kind": key[0], "severity": key[1],
                "fold": key[2], "perturbation_seeds": len(part)}
        for metric in metrics:
            item[metric] = float(np.mean([float(x[metric]) for x in part]))
        fold_rows.append(item)
    write_csv(out_root / "ours_fold_level.csv", fold_rows)
    macro = []
    for key in sorted({(r["kind"], r["severity"]) for r in fold_rows}):
        part = [r for r in fold_rows if (r["kind"], r["severity"]) == key]
        item = {"method": "Ours static Atlas", "kind": key[0], "severity": key[1],
                "folds": len(part)}
        for metric in metrics:
            values = np.asarray([float(x[metric]) for x in part])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        macro.append(item)
    write_csv(out_root / "ours_macro_summary.csv", macro)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_CORR / "MANIFEST.json")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--kinds", default="coord_noise,activity_noise,missing,outlier")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only-clean", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    if not kinds <= KINDS:
        raise ValueError(kinds)
    manifest = read_json(args.manifest)
    rows = []
    for fold in folds:
        clean_path, clean = saved_clean(fold)
        ckpt = checkpoint(fold)
        clean_q = int(clean["queries"])
        seen_clean = set()
        for condition in conditions(manifest, fold, kinds, args.only_clean):
            kind = str(condition["kind"]); severity = float(condition["severity"])
            result = evaluate(condition, fold, ckpt, args.out_root, args.device, args.force)
            if severity == 0.0:
                exact_clean_guard(result, clean, fold)
                seen_clean.add(kind)
            elif kind not in seen_clean:
                raise RuntimeError(f"fold{fold} {kind}: nonzero condition before clean guard")
            q = int(result["queries"]); coverage = q / clean_q if clean_q else float("nan")
            rows.append({
                "method": "Ours static Atlas", "fold": fold, "kind": kind,
                "severity": severity, "perturbation_seed": int(condition["perturbation_seed"]),
                "queries": q, "clean_queries": clean_q,
                "top1": float(result["top1_real"]), "top5": float(result["top5_real"]),
                "mrr": float(result["mrr_real"]),
                "hungarian": float(result["hungarian_accuracy"]),
                "coverage": coverage, "effective_top1": float(result["top1_real"]) * coverage,
                "checkpoint": str(ckpt.resolve()), "saved_clean": str(clean_path.resolve()),
                "corruption_root": str(Path(condition["root"]).resolve()),
            })
    summarize(rows, args.out_root)


if __name__ == "__main__":
    main()
