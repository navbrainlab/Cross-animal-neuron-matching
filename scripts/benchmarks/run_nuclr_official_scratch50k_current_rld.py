#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
ORIG = ROOT / "benchmark_official/adapters/nuclr_official_scratch50k/run_one.py"
CV_ROOTS = {
    "atanas": ROOT / "Data/Atanas_SF_unified_000776/cv5_grouped_v1",
    "rld": ROOT / "Data/Dunn_001623/cv5_grouped_v1",
}
RUN_ROOT = ROOT / "benchmark_official/runs/nuclr_official_scratch50k_current_cv_seed42"


def import_original():
    # Historical project modules include non-package imports.
    search_dirs = [
        ROOT,
        ROOT / "engines",
        ROOT / "table1_clean",
        ROOT / "benchmark_official",
        ORIG.parent,
    ]
    legacy = list(ROOT.rglob("train_hyqurp_nuclr_quantum_crossmodal_v2.py"))
    if legacy:
        search_dirs.append(legacy[0].parent)
    for p in search_dirs:
        p = str(Path(p).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location(
        "nuclr_scratch50k_original_currentcv", ORIG
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {ORIG}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def scalar_text(z, key: str) -> str:
    if key not in z.files:
        return ""
    a = np.asarray(z[key]).reshape(-1)
    if not len(a):
        return ""
    x = a[0]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    return str(x)


def normalize_label(x) -> str:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    s = str(x).strip()
    return "" if s.lower() in {"", "nan", "none", "null", "-1"} else s


def read_raw_meta(path: Path):
    with np.load(path, allow_pickle=True) as z:
        if "activity_raw" not in z.files or "xyz" not in z.files or "cell_id" not in z.files:
            raise KeyError(f"{path}: requires activity_raw, xyz, cell_id")
        activity = np.asarray(z["activity_raw"])
        xyz = np.asarray(z["xyz"], dtype=np.float32)
        labels = np.asarray(z["cell_id"]).reshape(-1)

        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError(f"{path}: xyz shape {xyz.shape}")
        xyz = xyz[:, :3]
        n = len(labels)
        if xyz.shape[0] != n:
            raise ValueError(f"{path}: xyz={xyz.shape[0]} labels={n}")
        if activity.ndim != 2:
            raise ValueError(f"{path}: activity_raw shape {activity.shape}")
        if activity.shape[0] != n and activity.shape[1] == n:
            activity = activity.T
        if activity.shape[0] != n:
            raise ValueError(f"{path}: activity={activity.shape} labels={n}")

        if "labeled_mask" in z.files:
            mask = np.asarray(z["labeled_mask"], dtype=bool).reshape(-1)
        elif "clean_mask" in z.files:
            mask = np.asarray(z["clean_mask"], dtype=bool).reshape(-1)
        else:
            mask = np.asarray([bool(normalize_label(x)) for x in labels], dtype=bool)
        if len(mask) != n:
            raise ValueError(f"{path}: mask={len(mask)} labels={n}")

        uid = scalar_text(z, "recording_uid") or scalar_text(z, "worm_id") or path.stem

    return uid, xyz, labels, mask


def split_paths(dataset: str, fold: int, split: str):
    root = CV_ROOTS[dataset] / f"fold_{fold}" / split
    paths = sorted(root.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(root)
    return paths


def collect_label_map(train_paths, val_paths):
    values = set()
    for path in [*train_paths, *val_paths]:
        _, _, labels, mask = read_raw_meta(path)
        for x, keep in zip(labels, mask):
            token = normalize_label(x)
            if keep and token:
                values.add(token)
    return {token: i for i, token in enumerate(sorted(values))}


def make_common(paths, label_map):
    records = []
    for path in paths:
        uid, xyz, raw_labels, mask = read_raw_meta(path)
        labels = np.full(len(raw_labels), -1, dtype=np.int64)
        for i, (x, keep) in enumerate(zip(raw_labels, mask)):
            token = normalize_label(x)
            if keep and token in label_map:
                labels[i] = int(label_map[token])
        records.append(
            SimpleNamespace(
                worm_id=uid,
                source_path=str(path.resolve()),
                path=str(path.resolve()),
                embedding_path=str(path.resolve()),
                xyz=torch.from_numpy(np.asarray(xyz, dtype=np.float32)),
                labels=torch.from_numpy(labels),
            )
        )
    return records


def assert_disjoint(train, val, test):
    def ids(paths):
        return {read_raw_meta(p)[0] for p in paths}
    a, b, c = ids(train), ids(val), ids(test)
    overlap = {
        "train_val": sorted(a & b),
        "train_test": sorted(a & c),
        "val_test": sorted(b & c),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Split leakage: {overlap}")
    return overlap


def json_dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def audit_fold(dataset: str, fold: int):
    train = split_paths(dataset, fold, "train")
    val = split_paths(dataset, fold, "val")
    test = split_paths(dataset, fold, "test")
    overlap = assert_disjoint(train, val, test)
    label_map = collect_label_map(train, val)
    common = make_common(train + val + test, label_map)
    min_n = min(len(r.labels) for r in common)
    max_n = max(len(r.labels) for r in common)
    return {
        "fold": fold,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "label_universe_train_val": len(label_map),
        "neurons_min": min_n,
        "neurons_max": max_n,
        "overlap": overlap,
    }


def train_fold(r, dataset: str, fold: int, seed: int, device: torch.device, target_steps: int,
               val_every: int, save_every: int, resume: bool):
    cv_root = CV_ROOTS[dataset]
    train_paths = split_paths(dataset, fold, "train")
    val_paths = split_paths(dataset, fold, "val")
    test_paths = split_paths(dataset, fold, "test")
    assert_disjoint(train_paths, val_paths, test_paths)

    label_map = collect_label_map(train_paths, val_paths)
    train_common = make_common(train_paths, label_map)
    val_common = make_common(val_paths, label_map)

    train_records = r.make_nuclr_records(train_common, "train")
    val_records = r.make_nuclr_records(val_common, "val")

    # Exact original scratch50k medoid rule.
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

    out_dir = RUN_ROOT / dataset / f"fold_{fold}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_info = r.git_info(r.OFFICIAL_REPO)
    if repo_info["dirty"]:
        raise RuntimeError(f"Official NuCLR checkout dirty: {r.OFFICIAL_REPO}")

    protocol = {
        "dataset": dataset,
        "cv_root": str(cv_root.resolve()),
        "fold": fold,
        "seed": seed,
        "method": "NuCLR (official scratch, 50k SSL, current grouped CV)",
        "source_runner": str(ORIG.resolve()),
        "official_repo": repo_info,
        "split_policy": "exact Data/Dunn_001623/cv5_grouped_v1 fold",
        "train_ids": [str(x.worm_id) for x in train_common],
        "val_ids": [str(x.worm_id) for x in val_common],
        "test_opened_before_selection": False,
        "label_map_built_from": "train+val only",
        "training_loss_uses_identity_labels": False,
        "activity_preprocessing": "original scratch50k make_nuclr_records -> s1.normalize_traces(..., 'zscore')",
        "template_selection": {
            "split": "train only",
            "worm": str(template_common.worm_id),
            "index": int(medoid.index),
            "mean_distance": medoid.mean_distance,
            "rule": "original scratch50k select_geometry_medoid",
        },
        "target_train_steps": target_steps,
        "val_every_steps": val_every,
        "save_last_every_steps": save_every,
    }
    json_dump(out_dir / "current_cv_protocol.json", protocol)

    selected = r.train_one(
        dataset=dataset,
        fold=fold,
        seed=seed,
        train_records=train_records,
        val_records=val_records,
        val_common=val_common,
        device=device,
        out_dir=out_dir,
        target_steps=target_steps,
        val_every_steps=val_every,
        save_last_every_steps=save_every,
        official_repo=repo_info,
        resume=resume,
    )
    json_dump(out_dir / "selected" / "selection.json", selected)

    # Only now open the current-fold clean outer test.
    test_common = make_common(test_paths, label_map)
    test_records = r.make_nuclr_records(test_common, "test")

    model = r.s1.build_official_model(device)
    ckpt = torch.load(selected["best_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    metrics, query_rows = r.evaluate_model_against_template(
        model,
        test_records,
        test_common,
        template_record,
        template_common,
        device,
        dataset,
        fold,
        seed,
        method="NuCLR (official scratch, 50k SSL, current grouped CV)",
    )

    outer = out_dir / "outer_test_medoid_template_v1"
    outer.mkdir(parents=True, exist_ok=True)
    query_rows.to_csv(outer / "query_level.csv", index=False)
    result = {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "method": "NuCLR (official scratch, 50k SSL, current grouped CV)",
        "selected_global_step": int(selected["best_global_step"]),
        "selected_checkpoint": selected["best_checkpoint"],
        "target_train_steps": target_steps,
        "final_global_step": int(selected["final_global_step"]),
        "metrics": metrics,
        "protocol": {
            "cv_root": str(cv_root.resolve()),
            "reference_worm": str(template_common.worm_id),
            "test_opened_after_selection": True,
            "random_initialization": True,
            "external_pretraining": False,
            "zscore_activity": True,
            "cosine": True,
            "test_test_pairing": False,
        },
    }
    json_dump(outer / "result.json", result)
    print("\nCURRENT-CV CLEAN OUTER TEST")
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("atanas", "rld"), default="rld")
    ap.add_argument("--fold", type=int, choices=range(5))
    ap.add_argument("--seed", type=int, default=42, choices=[42])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target-steps", type=int, default=50000)
    ap.add_argument("--val-every-steps", type=int, default=1000)
    ap.add_argument("--save-last-every-steps", type=int, default=1000)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()

    r = import_original()

    if args.audit_only:
        rows = [audit_fold(args.dataset, f) for f in range(5)]
        print(json.dumps(rows, indent=2))
        return

    if args.fold is None:
        raise ValueError("--fold is required unless --audit-only")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)

    train_fold(
        r,
        args.dataset,
        args.fold,
        args.seed,
        device,
        args.target_steps,
        args.val_every_steps,
        args.save_last_every_steps,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
