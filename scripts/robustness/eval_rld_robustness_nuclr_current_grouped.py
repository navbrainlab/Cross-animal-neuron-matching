#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
CURRENT_RUNNER = ROOT / "scripts/benchmarks/run_nuclr_official_scratch50k_current_rld.py"
CLEAN_ROOT = ROOT / "baselines/official/runs/nuclr_official_scratch50k_current_cv_seed42"
CORR_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions"
MANIFEST = CORR_ROOT / "MANIFEST.json"
OUT_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v2/results/nuclr"

METRIC_KEYS = (
    "queries", "ranking_top1", "top3", "top5",
    "top10", "mrr", "mean_rank", "assignment_top1",
)
KIND_ORDER = {"coord_noise": 0, "activity_noise": 1, "missing": 2, "outlier": 3}


def load_current_runner():
    if not CURRENT_RUNNER.is_file():
        raise FileNotFoundError(CURRENT_RUNNER)
    spec = importlib.util.spec_from_file_location(
        "nuclr_current_grouped_clean_runner", CURRENT_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {CURRENT_RUNNER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def parse_int_csv(text: str):
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    bad = [x for x in vals if x not in range(5)]
    if bad:
        raise ValueError(f"folds must be 0..4, got {bad}")
    return vals


def metric_diffs(a: dict, b: dict):
    out = {}
    for k in METRIC_KEYS:
        if k not in a or k not in b:
            continue
        av, bv = a[k], b[k]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            out[k] = abs(float(av) - float(bv))
    return out


def assert_same_metrics(a: dict, b: dict, name: str, tol: float = 1e-12):
    diffs = metric_diffs(a, b)
    bad = {k: v for k, v in diffs.items() if v > tol}
    if bad:
        raise RuntimeError(
            f"{name} FAILED exact replay.\n"
            f"metric diffs={bad}\nexpected={b}\nobserved={a}"
        )


def stable_uids(cur, paths):
    rows = {}
    for p in paths:
        uid, _, _, _ = cur.read_raw_meta(Path(p))
        if uid in rows:
            raise RuntimeError(f"duplicate test UID {uid}: {rows[uid]} and {p}")
        rows[uid] = Path(p)
    return rows


def corruption_rows(manifest: dict, folds: set[int], kinds: set[str]):
    rows = []
    for x in manifest["conditions"]:
        f = int(x["fold"])
        kind = str(x["kind"])
        if f not in folds or kind not in kinds:
            continue
        row = dict(x)
        row["fold"] = f
        row["kind"] = kind
        row["severity"] = float(x["severity"])
        row["perturbation_seed"] = int(x["perturbation_seed"])
        rows.append(row)
    rows.sort(
        key=lambda x: (
            int(x["fold"]),
            KIND_ORDER[x["kind"]],
            float(x["severity"]),
            int(x["perturbation_seed"]),
        )
    )
    if not rows:
        raise RuntimeError("No matching corruption rows in MANIFEST.json")
    return rows


def clear_activity_cache(r):
    candidates = [r]
    for name in ("common", "s1"):
        if hasattr(r, name):
            candidates.append(getattr(r, name))
    for obj in candidates:
        if hasattr(obj, "_ACTIVITY_CACHE"):
            getattr(obj, "_ACTIVITY_CACHE").clear()


def build_fold_context(cur, r, fold: int, seed: int, device: torch.device):
    train_paths = cur.split_paths("rld", fold, "train")
    val_paths = cur.split_paths("rld", fold, "val")
    clean_test_paths = cur.split_paths("rld", fold, "test")
    cur.assert_disjoint(train_paths, val_paths, clean_test_paths)

    label_map = cur.collect_label_map(train_paths, val_paths)
    train_common = cur.make_common(train_paths, label_map)
    train_records = r.make_nuclr_records(train_common, "train")

    train_geometry = []
    for record in train_common:
        xyz = record.xyz.detach().cpu().numpy()
        xyz = np.asarray(xyz, dtype=np.float64)[:, :3]
        train_geometry.append(
            (xyz - np.median(xyz, axis=0, keepdims=True)) / 200.0
        )
    medoid = r.select_geometry_medoid(
        [str(x.worm_id) for x in train_common], train_geometry
    )
    template_common = train_common[medoid.index]
    template_record = train_records[medoid.index]

    run_dir = CLEAN_ROOT / "rld" / f"fold_{fold}" / f"seed_{seed}"
    clean_result_path = run_dir / "outer_test_medoid_template_v1" / "result.json"
    if not clean_result_path.is_file():
        raise FileNotFoundError(clean_result_path)
    clean_result = read_json(clean_result_path)

    if int(clean_result.get("fold", -1)) != fold:
        raise RuntimeError(f"fold mismatch in {clean_result_path}")
    if int(clean_result.get("seed", -1)) != seed:
        raise RuntimeError(f"seed mismatch in {clean_result_path}")
    if int(clean_result.get("target_train_steps", -1)) != 50000:
        raise RuntimeError(f"not a 50k run: {clean_result_path}")
    if int(clean_result.get("final_global_step", -1)) != 50000:
        raise RuntimeError(f"training did not finish 50k: {clean_result_path}")

    expected_template = str(clean_result["protocol"]["reference_worm"])
    actual_template = str(template_common.worm_id)
    if expected_template != actual_template:
        raise RuntimeError(
            f"fold{fold}: template mismatch saved={expected_template}, "
            f"recomputed={actual_template}"
        )

    checkpoint = Path(clean_result["selected_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = r.s1.build_official_model(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"missing model_state_dict: {checkpoint}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return {
        "fold": fold,
        "seed": seed,
        "label_map": label_map,
        "template_common": template_common,
        "template_record": template_record,
        "template_worm": actual_template,
        "checkpoint": checkpoint,
        "selected_global_step": int(clean_result["selected_global_step"]),
        "clean_result": clean_result,
        "clean_test_uids": stable_uids(cur, clean_test_paths),
        "model": model,
    }


def audit_condition_inputs(cur, ctx, row):
    test_dir = Path(row["root"]) / "test"
    paths = sorted(test_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(test_dir)

    corrupt = stable_uids(cur, paths)
    clean_ids = set(ctx["clean_test_uids"])
    corrupt_ids = set(corrupt)
    if clean_ids != corrupt_ids:
        raise RuntimeError(
            f"fold{ctx['fold']} {row['kind']} sev={row['severity']} "
            f"p={row['perturbation_seed']}: test UID set changed; "
            f"missing={sorted(clean_ids-corrupt_ids)[:5]}, "
            f"extra={sorted(corrupt_ids-clean_ids)[:5]}"
        )

    synthetic = 0
    neurons = 0
    for p in paths:
        with np.load(p, allow_pickle=True) as z:
            for key in ("activity_raw", "xyz", "cell_id"):
                if key not in z.files:
                    raise KeyError(f"{p}: missing {key}")
            xyz = np.asarray(z["xyz"])
            activity = np.asarray(z["activity_raw"])
            labels = np.asarray(z["cell_id"]).reshape(-1)
            n = len(labels)
            neurons += n
            if xyz.shape[0] != n:
                raise RuntimeError(f"{p}: xyz rows != cell_id rows")
            if activity.ndim != 2:
                raise RuntimeError(f"{p}: activity_raw must be 2D")
            if activity.shape[0] != n and activity.shape[1] != n:
                raise RuntimeError(
                    f"{p}: activity_raw shape {activity.shape} incompatible with N={n}"
                )
            synthetic += sum(
                str(x).startswith("__OUTLIER_") for x in labels.tolist()
            )

    return paths, {
        "test_worms": len(paths),
        "total_neuron_rows": int(neurons),
        "synthetic_outlier_rows": int(synthetic),
    }


@torch.inference_mode()
def evaluate_condition(cur, r, ctx, row, device, out_root: Path):
    paths, input_audit = audit_condition_inputs(cur, ctx, row)

    # Frozen train+val label map; only test NPZs are replaced by corruption.
    test_common = cur.make_common(paths, ctx["label_map"])
    clear_activity_cache(r)

    # Exact official scratch50k activity preprocessing + inference path.
    test_records = r.make_nuclr_records(test_common, "test")
    metrics, query_rows = r.evaluate_model_against_template(
        ctx["model"],
        test_records,
        test_common,
        ctx["template_record"],
        ctx["template_common"],
        device,
        "rld",
        ctx["fold"],
        ctx["seed"],
        method=(
            "NuCLR (official scratch, 50k SSL, current grouped CV) "
            "— controlled robustness"
        ),
    )

    kind = str(row["kind"])
    sev = float(row["severity"])
    pseed = int(row["perturbation_seed"])
    name = f"{kind}_l{sev:.2f}_p{pseed}"

    dst = out_root / f"fold{ctx['fold']}" / name
    dst.mkdir(parents=True, exist_ok=True)
    query_rows.to_csv(dst / "query_level.csv", index=False)

    clean_q = int(ctx["clean_result"]["metrics"]["queries"])
    q = int(metrics["queries"])
    coverage = q / clean_q if clean_q else float("nan")
    effective_top1 = (
        float(metrics["ranking_top1"]) * q / clean_q
        if clean_q else float("nan")
    )

    result = {
        "dataset": "rld",
        "fold": int(ctx["fold"]),
        "seed": int(ctx["seed"]),
        "condition": name,
        "kind": kind,
        "severity": sev,
        "perturbation_seed": pseed,
        "method": "NuCLR official scratch50k current grouped CV",
        "selected_global_step": int(ctx["selected_global_step"]),
        "frozen_checkpoint": str(ctx["checkpoint"].resolve()),
        "frozen_reference_worm": str(ctx["template_worm"]),
        "metrics": metrics,
        "derived": {
            "clean_query_denominator": clean_q,
            "coverage_vs_clean_eligible": float(coverage),
            "effective_top1": float(effective_top1),
        },
        "input_audit": input_audit,
        "protocol": {
            "training_on_corruption": False,
            "checkpoint_selection": "clean validation only; frozen",
            "template": "exact clean outer-train geometry medoid; frozen",
            "test_only_corruption": True,
            "activity_preprocessing": "exact original scratch50k z-score path",
            "synthetic_distractors": (
                "remain in full activity/population input; "
                "not members of frozen train+val identity map"
            ),
        },
    }
    write_json(dst / "result.json", result)
    return result


def summarize(results, out_root: Path):
    metrics = ["ranking_top1", "top5", "mrr", "assignment_top1"]
    derived = ["coverage_vs_clean_eligible", "effective_top1"]

    grouped = defaultdict(list)
    for x in results:
        grouped[(x["kind"], float(x["severity"]), int(x["fold"]))].append(x)

    fold_rows = []
    for (kind, sev, fold), xs in sorted(
        grouped.items(),
        key=lambda kv: (KIND_ORDER[kv[0][0]], kv[0][1], kv[0][2]),
    ):
        row = {
            "kind": kind,
            "severity": sev,
            "fold": fold,
            "num_perturbation_seeds": len(xs),
        }
        for k in metrics:
            row[k] = float(np.mean([float(x["metrics"][k]) for x in xs]))
        for k in derived:
            row[k] = float(np.mean([float(x["derived"][k]) for x in xs]))
        fold_rows.append(row)

    grouped2 = defaultdict(list)
    for x in fold_rows:
        grouped2[(x["kind"], float(x["severity"]))].append(x)

    macro_rows = []
    for (kind, sev), cells in sorted(
        grouped2.items(),
        key=lambda kv: (KIND_ORDER[kv[0][0]], kv[0][1]),
    ):
        row = {"kind": kind, "severity": sev, "num_folds": len(cells)}
        for k in metrics + derived:
            vals = np.asarray([float(x[k]) for x in cells], dtype=float)
            row[k + "_mean"] = float(vals.mean())
            row[k + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        macro_rows.append(row)

    pd.DataFrame(fold_rows).to_csv(
        out_root / "nuclr_fold_level_summary.csv", index=False
    )
    pd.DataFrame(macro_rows).to_csv(
        out_root / "nuclr_macro_summary.csv", index=False
    )
    write_json(
        out_root / "SUMMARY.json",
        {
            "aggregation": (
                "mean perturbation seeds within biological fold, "
                "then unweighted mean ± sample SD across biological folds"
            ),
            "fold_level": fold_rows,
            "macro": macro_rows,
        },
    )

    labels = {
        "coord_noise": "Coordinate noise",
        "activity_noise": "Activity noise",
        "missing": "Missing neurons",
        "outlier": "Distractors",
    }
    lookup = {(row["kind"], float(row["severity"])): row for row in macro_rows}
    selected = [("Clean", lookup[("coord_noise", 0.0)])]
    for kind in ("coord_noise", "activity_noise", "missing", "outlier"):
        for sev in sorted(
            level for name, level in lookup if name == kind and level > 0
        ):
            selected.append((labels[kind], lookup[(kind, sev)]))

    def pct(row, key):
        return (
            f"{100 * row[key + '_mean']:.2f} ± "
            f"{100 * row[key + '_sd']:.2f}%"
        )

    lines = [
        "# NuCLR robustness — native grouped CV5 × seed42",
        "",
        "Each perturbation seed is averaged within biological fold first; values are the "
        "unweighted mean ± sample SD across the same five folds as the Main Benchmark. "
        "No shared-cohort rescoring is used.",
        "",
        "| Corruption | Severity | Top-1 ↑ | Hungarian ↑ | Coverage ↑ | Effective Top-1 ↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    export = []
    for label, row in selected:
        item = {
            "corruption": label,
            "severity": float(row["severity"]),
            "top1": pct(row, "ranking_top1"),
            "hungarian": pct(row, "assignment_top1"),
            "coverage": pct(row, "coverage_vs_clean_eligible"),
            "effective_top1": pct(row, "effective_top1"),
        }
        export.append(item)
        lines.append(
            f"| {label} | {item['severity']:.2f} | {item['top1']} | "
            f"{item['hungarian']} | {item['coverage']} | {item['effective_top1']} |"
        )
    lines.extend([
        "",
        "Coordinate noise leaves NuCLR unchanged because this baseline consumes activity only.",
        "",
    ])
    table = "\n".join(lines)
    (out_root / "NUCLR_TABLE.md").write_text(table, encoding="utf-8")
    pd.DataFrame(export).to_csv(out_root / "nuclr_formal_table.csv", index=False)
    formal_dir = ROOT / "runs/rld_robustness_cv5_seed42_v2/formal_native_cv5_nuclr"
    formal_dir.mkdir(parents=True, exist_ok=True)
    (formal_dir / "TABLE.md").write_text(table, encoding="utf-8")
    pd.DataFrame(export).to_csv(formal_dir / "summary.csv", index=False)

    print()
    print("=" * 100)
    print("NuCLR CURRENT-GROUPED ROBUSTNESS MACRO SUMMARY")
    print("=" * 100)
    for row in macro_rows:
        print(
            f"{row['kind']:<12s} "
            f"level={row['severity']:.2f} "
            f"Top1={100*row['ranking_top1_mean']:.2f}±"
            f"{100*row['ranking_top1_sd']:.2f}% "
            f"Top5={100*row['top5_mean']:.2f}±"
            f"{100*row['top5_sd']:.2f}% "
            f"MRR={row['mrr_mean']:.4f}±{row['mrr_sd']:.4f} "
            f"Hung={100*row['assignment_top1_mean']:.2f}±"
            f"{100*row['assignment_top1_sd']:.2f}% "
            f"Cov={100*row['coverage_vs_clean_eligible_mean']:.2f}±"
            f"{100*row['coverage_vs_clean_eligible_sd']:.2f}% "
            f"EffTop1={100*row['effective_top1_mean']:.2f}±"
            f"{100*row['effective_top1_sd']:.2f}%"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=42, choices=[42])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--kinds", default="coord_noise,activity_noise,missing,outlier")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--out-root", type=Path, default=OUT_ROOT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    folds = parse_int_csv(args.folds)
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    if not kinds or not kinds <= set(KIND_ORDER):
        raise ValueError(f"invalid kinds={sorted(kinds)}")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NuCLR official robustness requires CUDA")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("NuCLR official model requires BF16-capable CUDA")

    cur = load_current_runner()
    r = cur.import_original()
    manifest = read_json(args.manifest)
    rows = corruption_rows(manifest, set(folds), kinds)

    args.out_root.mkdir(parents=True, exist_ok=True)
    results = []

    print("=" * 100)
    print("NuCLR OFFICIAL SCRATCH50K — CURRENT RLD GROUPED CV — CONTROLLED ROBUSTNESS")
    print("=" * 100)

    for fold in folds:
        print()
        print("=" * 120)
        print(f"LOCKED CURRENT-GROUPED NUCLR CONTEXT — fold{fold} seed42")
        print("=" * 120)

        ctx = build_fold_context(cur, r, fold, args.seed, device)
        print("checkpoint:", ctx["checkpoint"])
        print("selected  :", ctx["selected_global_step"])
        print("template  :", ctx["template_worm"])
        print("clean     :", ctx["clean_result"]["metrics"])

        fold_rows = [x for x in rows if int(x["fold"]) == fold]
        fold_rows.sort(
            key=lambda x: (
                KIND_ORDER[x["kind"]],
                float(x["severity"]),
                int(x["perturbation_seed"]),
            )
        )
        seen_clean_guard = set()

        for row in fold_rows:
            kind = row["kind"]
            sev = float(row["severity"])
            ps = int(row["perturbation_seed"])
            name = f"{kind}_l{sev:.2f}_p{ps}"
            dst = args.out_root / f"fold{fold}" / name / "result.json"

            if dst.is_file() and not args.overwrite:
                result = read_json(dst)
                print("[REUSE]", f"fold{fold}", name)
            else:
                print("[EVAL ]", f"fold{fold}", name, flush=True)
                result = evaluate_condition(
                    cur, r, ctx, row, device, args.out_root
                )

            if sev == 0.0:
                assert_same_metrics(
                    result["metrics"],
                    ctx["clean_result"]["metrics"],
                    f"fold{fold} {kind} severity=0",
                )
                seen_clean_guard.add(kind)
                print(f"[CLEAN GUARD EXACT] fold{fold} {kind}", flush=True)
            elif kind not in seen_clean_guard:
                raise RuntimeError(
                    f"fold{fold} {kind}: nonzero corruption reached before "
                    "severity=0 exact guard"
                )

            if kind == "coord_noise":
                assert_same_metrics(
                    result["metrics"],
                    ctx["clean_result"]["metrics"],
                    f"fold{fold} coordinate invariance severity={sev} p={ps}",
                )
                print(
                    f"[COORD-INVARIANCE EXACT] fold{fold} {name}",
                    flush=True,
                )

            results.append(result)

        del ctx["model"]
        torch.cuda.empty_cache()

    summarize(results, args.out_root)

    print()
    print("COMPLETE")
    print("results:", args.out_root)
    print("macro:  ", args.out_root / "nuclr_macro_summary.csv")


if __name__ == "__main__":
    main()
