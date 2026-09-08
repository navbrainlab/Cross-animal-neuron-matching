#!/usr/bin/env python3
"""
CURRENT-GROUPED GeoTransformer robustness replay on RLD.

Uses:
  clean split: Data/Dunn_001623/cv5_grouped_v1/fold_{0..4}
  corruptions: runs/rld_robustness_cv5_seed42_v2/corruptions/fold{0..4}/...
  checkpoints:
    /home/ubuntu/klb/nuclr/geotransformer_official/
      current_grouped_selection/rld/fold{F}/seed42/best_checkpoint.txt

Protocol:
  - frozen validation-selected current-grouped seed42 checkpoint
  - fixed outer-train geometry medoid, reselected ONLY from the clean train split
  - test-only corruption
  - original GeoTransformer semantic model / evaluate_pair
  - independent original-clean replay first
  - every severity=0 corruption MUST exactly reproduce that clean replay
  - no corruption-specific tuning / retraining / template reselection

Run with physical GPU1:
  CUDA_VISIBLE_DEVICES=1 python -u eval_geotransformer_rld_robustness_current_grouped.py --device cuda:0
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
DATASET = "rld"
CV_ROOTS = {
    "atanas": ROOT / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": ROOT / "Data/Dunn_001623/cv5_grouped_v1",
}
CV_ROOT = CV_ROOTS[DATASET]
CORR_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions"
OUT_ROOT = (
    ROOT
    / "runs/rld_robustness_cv5_seed42_v2/results"
    / "geotransformer"
)

GEO_ROOT = Path("/home/ubuntu/klb/nuclr/geotransformer_official")
EXP_ROOT = GEO_ROOT / "experiments/geotransformer.rld.semantic"
SELECTION_ROOT = GEO_ROOT / "current_grouped_selection/rld"
OLD_HELPER = GEO_ROOT / "train_medoid_protocol/evaluate_cv5x3_train_medoid.py"

COND_RE = re.compile(r"^(coord_noise|missing|outlier)_l([0-9]+(?:\.[0-9]+)?)_p([0-9]+)$")


def parse_args():
    global DATASET, CV_ROOT, EXP_ROOT, SELECTION_ROOT, CORR_ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("atanas", "rld"), default="rld")
    p.add_argument("--folds", default="0,1,2,3,4")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--kinds", default="coord_noise,missing,outlier")
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument("--corruption-root", type=Path, default=CORR_ROOT)
    p.add_argument("--clean-only", action="store_true")
    args = p.parse_args()
    DATASET = args.dataset
    CV_ROOT = CV_ROOTS[DATASET]
    EXP_ROOT = GEO_ROOT / f"experiments/geotransformer.{DATASET}.semantic"
    SELECTION_ROOT = GEO_ROOT / f"current_grouped_selection/{DATASET}"
    CORR_ROOT = args.corruption_root.resolve()
    return args


def jdump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_folds(text: str):
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if any(x not in range(5) for x in vals):
        raise ValueError(vals)
    return vals


def parse_checkpoint(fold: int):
    sel = SELECTION_ROOT / f"fold{fold}/seed42/best_checkpoint.txt"
    if not sel.is_file():
        raise FileNotFoundError(sel)

    text = sel.read_text(encoding="utf-8")
    m_ck = re.search(r"^checkpoint=(.+)$", text, re.M)
    m_it = re.search(r"^iteration=(\d+)$", text, re.M)
    m_v1 = re.search(r"^validation_top1_percent=([0-9.eE+-]+)$", text, re.M)
    if not (m_ck and m_it and m_v1):
        raise RuntimeError(f"Malformed selection file: {sel}\n{text}")

    ckpt = Path(m_ck.group(1).strip()).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if "current_grouped" not in str(ckpt):
        raise RuntimeError(f"Refusing non-current-grouped checkpoint: {ckpt}")

    return {
        "selection_file": str(sel.resolve()),
        "checkpoint": ckpt,
        "iteration": int(m_it.group(1)),
        "validation_top1_percent": float(m_v1.group(1)),
        "checkpoint_sha256": sha256(ckpt),
    }


def discover_env_name():
    text = (EXP_ROOT / "config.py").read_text(encoding="utf-8")
    m = re.search(
        r'_C\.data\.dataset_root\s*=\s*os\.environ\.get\(\s*["\']([^"\']+)["\']',
        text,
        re.S,
    )
    if not m:
        raise RuntimeError("Cannot discover GeoTransformer dataset-root env variable")
    return m.group(1)


def import_geo():
    if str(GEO_ROOT) not in sys.path:
        sys.path.insert(0, str(GEO_ROOT))
    if str(EXP_ROOT) not in sys.path:
        sys.path.insert(0, str(EXP_ROOT))

    for name in ("config", "dataset", "model", "loss", "evaluate_semantic"):
        sys.modules.pop(name, None)

    cfgmod = importlib.import_module("config")
    dsmod = importlib.import_module("dataset")
    modelmod = importlib.import_module("model")
    evalmod = importlib.import_module("evaluate_semantic")

    if not OLD_HELPER.is_file():
        raise FileNotFoundError(OLD_HELPER)
    spec = importlib.util.spec_from_file_location("geo_medoid_helper", OLD_HELPER)
    helper = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(helper)

    for fn in ("make_fixed_template_loader", "evaluate_checkpoint"):
        if not hasattr(helper, fn):
            raise RuntimeError(f"{OLD_HELPER} lacks {fn}")

    return cfgmod, dsmod, modelmod, evalmod, helper


def canonical_uid(path: Path):
    parts = path.stem.split("__")
    if len(parts) >= 3:
        return parts[1]
    return path.stem


def normalized_cloud(dsmod, path: Path):
    with np.load(path, allow_pickle=False) as z:
        xyz = np.asarray(z["xyz"], dtype=np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return dsmod.normalize_cloud(xyz)


def select_train_medoid(clean_fold_root: Path, dsmod):
    # Reuse the repository's strict medoid implementation.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.lib.fair_identity_protocol import select_geometry_medoid

    train_files = sorted((clean_fold_root / "train").rglob("*.npz"))
    if not train_files:
        raise RuntimeError(f"No clean train files under {clean_fold_root}")

    uids = [canonical_uid(p) for p in train_files]
    clouds = [normalized_cloud(dsmod, p) for p in train_files]
    medoid = select_geometry_medoid(uids, clouds)

    # Need the index in dataset_module.build_dataset(..., 'train') order.
    return medoid, train_files


def make_mixed_root(base: Path, clean_fold_root: Path, test_dir: Path):
    """
    Build a lightweight fold root:
      train -> clean train
      val   -> clean val
      test  -> requested clean/corrupted test
    so the preserved fixed-template helper can be reused unchanged.
    """
    if base.exists() or base.is_symlink():
        if base.is_symlink() or base.is_file():
            base.unlink()
        else:
            shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)

    targets = {
        "train": clean_fold_root / "train",
        "val": clean_fold_root / "val",
        "test": test_dir,
    }
    for name, target in targets.items():
        if not target.is_dir():
            raise FileNotFoundError(target)
        os.symlink(str(target.resolve()), str(base / name), target_is_directory=True)


def dataset_file_index_by_uid(train_dataset, wanted_uid: str):
    hits = []
    for i, p in enumerate(train_dataset.files):
        if canonical_uid(Path(p)) == wanted_uid:
            hits.append(i)
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one train dataset entry for medoid {wanted_uid}, got {hits}"
        )
    return hits[0]


def metric_exact_guard(clean: dict, replay: dict, fold: int, condition: str):
    keys = (
        "queries", "top1", "top5", "mrr",
        "hungarian_accuracy", "candidate_coverage",
    )
    bad = []
    for key in keys:
        a = clean[key]
        b = replay[key]
        if key == "queries":
            ok = int(a) == int(b)
            diff = int(b) - int(a)
        else:
            diff = abs(float(a) - float(b))
            ok = diff <= 1e-12
        print(
            f"[CLEAN GUARD] fold{fold} {condition} {key}: "
            f"clean={a} replay={b} diff={diff}",
            flush=True,
        )
        if not ok:
            bad.append((key, a, b))
    if bad:
        raise RuntimeError(
            f"SEVERITY-0 CLEAN REPLAY FAILED fold={fold} condition={condition}: {bad}"
        )


def discover_conditions(fold: int, kinds: set[str]):
    root = CORR_ROOT / f"fold{fold}"
    if not root.is_dir():
        raise FileNotFoundError(root)
    rows = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        m = COND_RE.match(p.name)
        if not m:
            continue
        kind, sev, ps = m.groups()
        if kind not in kinds:
            continue
        test = p / "test"
        if not test.is_dir():
            raise FileNotFoundError(test)
        rows.append({
            "kind": kind,
            "severity": float(sev),
            "perturbation_seed": int(ps),
            "name": p.name,
            "test_dir": test,
        })
    order = {"coord_noise": 0, "missing": 1, "outlier": 2}
    rows.sort(key=lambda r: (order[r["kind"]], r["severity"], r["perturbation_seed"]))
    return rows


def evaluate_one(
    *,
    fold: int,
    test_dir: Path,
    condition_name: str,
    clean_fold_root: Path,
    medoid_uid: str,
    selection: dict,
    device: torch.device,
    env_name: str,
    cfgmod,
    dsmod,
    modelmod,
    evalmod,
    helper,
    scratch_root: Path,
):
    mixed = scratch_root / f"fold{fold}" / condition_name
    make_mixed_root(mixed, clean_fold_root, test_dir)

    os.environ[env_name] = str(mixed)
    os.environ["SEED"] = "42"
    os.environ["RUN_TAG"] = f"robust_current_grouped_fold{fold}_{condition_name}"

    cfg = cfgmod.make_cfg()
    cfg.data.dataset_root = str(mixed)
    if hasattr(cfg, "test") and hasattr(cfg.test, "num_workers"):
        cfg.test.num_workers = 0

    # Confirm clean train medoid maps to the same dataset entry.
    train_dataset = dsmod.build_dataset(cfg, "train")
    template_index = dataset_file_index_by_uid(train_dataset, medoid_uid)

    loader, neighbor_limits, excluded = helper.make_fixed_template_loader(
        cfg, dsmod, template_index
    )

    model = modelmod.create_model(cfg).to(device)
    evalmod.load_checkpoint(model, str(selection["checkpoint"]))
    model.eval()

    with torch.inference_mode():
        test_metrics = helper.evaluate_checkpoint(
            model,
            loader,
            evalmod.evaluate_pair,
            device,
        )

    del model
    torch.cuda.empty_cache()

    return {
        "dataset": DATASET,
        "robustness_fold": fold,
        "seed": 42,
        "condition": condition_name,
        "checkpoint": str(selection["checkpoint"]),
        "checkpoint_sha256": selection["checkpoint_sha256"],
        "selected_iteration": selection["iteration"],
        "validation_top1_percent_used_for_selection": selection["validation_top1_percent"],
        "template_uid": medoid_uid,
        "neighbor_limits": [int(x) for x in neighbor_limits],
        "excluded_no_shared_identity": excluded,
        "test": test_metrics,
        "protocol": {
            "training": "NONE; frozen current-grouped seed42 checkpoint",
            "checkpoint_selection": "clean validation only, already completed",
            "template_selection": "outer-clean-train geometry medoid only",
            "template_reselected_under_corruption": False,
            "test_only_corruption": True,
            "model": "original semantic GeoTransformer",
            "metric": f"unchanged experiments/geotransformer.{DATASET}.semantic/evaluate_semantic.py evaluate_pair",
        },
    }


def main():
    a = parse_args()
    folds = parse_folds(a.folds)
    kinds = {x.strip() for x in a.kinds.split(",") if x.strip()}
    allowed = {"coord_noise", "missing", "outlier"}
    if not kinds <= allowed:
        raise ValueError(kinds - allowed)

    device = torch.device(a.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    print("=" * 126)
    print(f"GEOTRANSFORMER CURRENT-GROUPED {DATASET.upper()} — LOCKED EVALUATION")
    print("=" * 126)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("logical device      :", device)
    print("GPU                 :", torch.cuda.get_device_name(device))
    print("folds               :", folds)
    print("kinds               :", sorted(kinds))
    print()

    env_name = discover_env_name()
    cfgmod, dsmod, modelmod, evalmod, helper = import_geo()

    out_root = a.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    scratch_root = out_root / "_mixed_fold_roots"

    manifest = []

    for fold in folds:
        print("\n" + "=" * 126)
        print(f"FOLD {fold}")
        print("=" * 126)

        clean_fold = CV_ROOT / f"fold_{fold}"
        selection = parse_checkpoint(fold)

        # Select medoid once from the ORIGINAL clean outer-train split.
        medoid, train_files = select_train_medoid(clean_fold, dsmod)
        medoid_uid = str(medoid.uid)
        print("checkpoint :", selection["checkpoint"])
        print("iteration  :", selection["iteration"])
        print("val Top1   :", selection["validation_top1_percent"])
        print("medoid UID :", medoid_uid)
        print("medoid mean distance:", medoid.mean_distance)

        clean_path = out_root / f"fold{fold}/clean_reference/result.json"
        if clean_path.is_file() and not a.overwrite:
            clean_result = jload(clean_path)
        else:
            clean_result = evaluate_one(
                fold=fold,
                test_dir=clean_fold / "test",
                condition_name="clean_reference",
                clean_fold_root=clean_fold,
                medoid_uid=medoid_uid,
                selection=selection,
                device=device,
                env_name=env_name,
                cfgmod=cfgmod,
                dsmod=dsmod,
                modelmod=modelmod,
                evalmod=evalmod,
                helper=helper,
                scratch_root=scratch_root,
            )
            clean_result["template_selection"] = {
                "selection_split": "outer_train_only",
                "template_uid": medoid_uid,
                "template_mean_distance": float(medoid.mean_distance),
                "uses_test": False,
            }
            jdump(clean_path, clean_result)

        print("[CLEAN]", clean_result["test"])

        manifest.append({
            "fold": fold,
            "condition": "clean_reference",
            "result": str(clean_path.resolve()),
        })

        if a.clean_only:
            continue

        for row in discover_conditions(fold, kinds):
            dst = out_root / f"fold{fold}/{row['name']}/result.json"

            if dst.is_file() and not a.overwrite:
                result = jload(dst)
            else:
                print(
                    f"\n[RUN] fold={fold} {row['name']} "
                    f"checkpoint=iter{selection['iteration']}",
                    flush=True,
                )
                result = evaluate_one(
                    fold=fold,
                    test_dir=row["test_dir"],
                    condition_name=row["name"],
                    clean_fold_root=clean_fold,
                    medoid_uid=medoid_uid,
                    selection=selection,
                    device=device,
                    env_name=env_name,
                    cfgmod=cfgmod,
                    dsmod=dsmod,
                    modelmod=modelmod,
                    evalmod=evalmod,
                    helper=helper,
                    scratch_root=scratch_root,
                )
                result.update({
                    "kind": row["kind"],
                    "severity": row["severity"],
                    "perturbation_seed": row["perturbation_seed"],
                    "corrupted_test_root": str(row["test_dir"].resolve()),
                    "clean_query_denominator": int(clean_result["test"]["queries"]),
                })

                q = int(result["test"]["queries"])
                clean_q = int(clean_result["test"]["queries"])
                result["derived"] = {
                    "surviving_evaluable_query_fraction": q / clean_q if clean_q else None,
                    "effective_top1_vs_clean_queries": (
                        float(result["test"]["top1"]) * q / clean_q
                        if clean_q else None
                    ),
                }
                jdump(dst, result)

            if math_is_zero(row["severity"]):
                metric_exact_guard(
                    clean_result["test"],
                    result["test"],
                    fold,
                    row["name"],
                )

            t = result["test"]
            print(
                f"[RESULT] fold={fold} {row['name']} "
                f"Q={t['queries']} "
                f"Top1={100*t['top1']:.2f}% "
                f"Top5={100*t['top5']:.2f}% "
                f"MRR={t['mrr']:.4f} "
                f"Hung={100*t['hungarian_accuracy']:.2f}% "
                f"Cov={100*t['candidate_coverage']:.2f}%",
                flush=True,
            )

            manifest.append({
                "fold": fold,
                "condition": row["name"],
                "kind": row["kind"],
                "severity": row["severity"],
                "perturbation_seed": row["perturbation_seed"],
                "result": str(dst.resolve()),
            })

    jdump(
        out_root / "run_manifest.json",
        {
            "method": f"GeoTransformer current-grouped {DATASET} seed42",
            "dataset": DATASET,
            "folds": folds,
            "kinds": sorted(kinds),
            "results": manifest,
        },
    )

    print("\nDONE:", out_root)


def math_is_zero(x: float) -> bool:
    return abs(float(x)) <= 1e-12


if __name__ == "__main__":
    main()
