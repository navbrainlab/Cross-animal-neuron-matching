#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
CV_ROOT = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
CORR_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v1/corruptions"
OUT_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v1/results"

FDNC_SOURCE = ROOT / "baselines/official/third_party/fdnc_official"
FDNC_EXPECTED_COMMIT = "19c678781cd11a17866af7b6348ac0096a168c06"
FDNC_MODEL_SHA256 = "ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9"

CPD_CLEAN_ROOT = ROOT / "runs/fair_identity_medoid_template_v1/cpd/rld"
FDNC_CLEAN_ROOT = ROOT / "runs/fair_identity_retest_v1/fdnc_official/rld"

INVALID = {"", "none", "nan", "null", "unknown", "?", "-1"}


def jload(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def jdump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
                    encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_label(x: Any) -> str:
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, bytes):
        x = x.decode("utf-8", errors="replace")
    s = str(x).strip()
    return "" if s.lower() in INVALID else s


def recording_uid(path: Path, z=None) -> str:
    close = False
    if z is None:
        z = np.load(path, allow_pickle=True)
        close = True
    try:
        for key in ("recording_uid", "worm_id"):
            if key in z.files:
                a = np.asarray(z[key]).reshape(-1)
                if len(a):
                    v = a[0]
                    if isinstance(v, np.generic):
                        v = v.item()
                    if isinstance(v, bytes):
                        v = v.decode("utf-8", errors="replace")
                    if str(v):
                        return str(v)
    finally:
        if close:
            z.close()
    parts = path.stem.split("__")
    return parts[1] if len(parts) >= 3 else path.stem


def load_worm(path: Path) -> dict:
    path = Path(path).resolve()
    with np.load(path, allow_pickle=True) as z:
        if "xyz" not in z.files or "cell_id" not in z.files:
            raise KeyError(f"{path}: requires xyz and cell_id")
        xyz = np.asarray(z["xyz"], dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError(f"{path}: bad xyz {xyz.shape}")
        xyz = xyz[:, :3]
        raw = np.asarray(z["cell_id"]).reshape(-1)
        if len(raw) != len(xyz):
            raise ValueError(f"{path}: xyz/cell_id mismatch")
        labels = [norm_label(x) for x in raw]
        if "labeled_mask" in z.files:
            mask = np.asarray(z["labeled_mask"], dtype=bool).reshape(-1)
        elif "clean_mask" in z.files:
            mask = np.asarray(z["clean_mask"], dtype=bool).reshape(-1)
        else:
            mask = np.asarray([bool(x) for x in labels], dtype=bool)
        if len(mask) != len(xyz):
            raise ValueError(f"{path}: mask mismatch")
        # Synthetic distractors are always model input but never GT.
        for i, lab in enumerate(labels):
            if lab.startswith("__OUTLIER_"):
                mask[i] = False
        if "roi_index" in z.files:
            roi = np.asarray(z["roi_index"]).reshape(-1)
            if len(roi) != len(xyz):
                roi = np.arange(len(xyz))
        else:
            roi = np.arange(len(xyz))
        uid = recording_uid(path, z)
    return dict(path=path, uid=uid, xyz=xyz, labels=labels, mask=mask, roi=roi)


def unique_map(worm: dict) -> dict[str, int]:
    pos = defaultdict(list)
    for i, (lab, keep) in enumerate(zip(worm["labels"], worm["mask"])):
        if keep and lab:
            pos[lab].append(i)
    return {lab: rows[0] for lab, rows in pos.items() if len(rows) == 1}


def normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    x = np.asarray(xyz, dtype=np.float32)
    return (x - np.median(x, axis=0, keepdims=True)) / 200.0


def chamfer(a: np.ndarray, b: np.ndarray) -> float:
    ta, tb = cKDTree(a), cKDTree(b)
    return 0.5 * (tb.query(a, k=1)[0].mean() + ta.query(b, k=1)[0].mean())


def find_template_from_saved_json(fold: int, train: list[dict]) -> dict | None:
    p = CPD_CLEAN_ROOT / f"fold{fold}" / "metrics.json"
    if not p.is_file():
        return None
    obj = jload(p)
    strings = []

    def walk(x, key=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if "template" in str(k).lower():
                    if isinstance(v, (str, int, float)):
                        strings.append(str(v))
                walk(v, str(k))
        elif isinstance(x, list):
            for v in x:
                walk(v, key)

    walk(obj)
    for token in strings:
        if ".npz" not in token:
            continue
        name = Path(token).name
        for w in train:
            if w["path"].name == name:
                return w
        # fall back to UID inside legacy-prefixed file names
        parts = Path(token).stem.split("__")
        uid = parts[1] if len(parts) >= 3 else Path(token).stem
        for w in train:
            if w["uid"] == uid:
                return w
    return None


def select_train_medoid(fold: int, train: list[dict]) -> dict:
    saved = find_template_from_saved_json(fold, train)
    if saved is not None:
        print(f"[TEMPLATE] fold{fold} from saved clean CPD: {saved['path'].name}")
        return saved

    xs = [normalize_xyz(w["xyz"]) for w in train]
    n = len(xs)
    sums = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = chamfer(xs[i], xs[j])
            sums[i] += d
            sums[j] += d
    mean = sums / max(n - 1, 1)
    idx = int(np.argmin(mean))
    print(f"[TEMPLATE] fold{fold} recomputed train-only Chamfer medoid: "
          f"{train[idx]['path'].name} mean={mean[idx]:.6f}")
    return train[idx]


def import_fdnc():
    if not FDNC_SOURCE.is_dir():
        raise FileNotFoundError(FDNC_SOURCE)
    import subprocess
    commit = subprocess.check_output(
        ["git", "-C", str(FDNC_SOURCE), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != FDNC_EXPECTED_COMMIT:
        raise RuntimeError(f"fDNC commit mismatch: {commit}")

    model_file = None
    for p in (FDNC_SOURCE / "model").rglob("*"):
        if p.is_file():
            try:
                if sha256(p) == FDNC_MODEL_SHA256:
                    model_file = p
                    break
            except OSError:
                pass
    if model_file is None:
        for p in FDNC_SOURCE.rglob("*"):
            if p.is_file() and p.stat().st_size < 1_000_000_000:
                try:
                    if sha256(p) == FDNC_MODEL_SHA256:
                        model_file = p
                        break
                except OSError:
                    pass
    if model_file is None:
        raise FileNotFoundError(
            f"No released fDNC model with sha256={FDNC_MODEL_SHA256}"
        )

    src = FDNC_SOURCE / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    spec = importlib.util.spec_from_file_location("fdnc_robust_official_model", src / "model.py")
    if spec is None or spec.loader is None:
        raise ImportError(src / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod, model_file


def build_fdnc(device: torch.device):
    mod, model_file = import_fdnc()
    model = mod.NIT_Registration(
        input_dim=3, n_hidden=128, n_layer=6,
        p_rotate=0, feat_trans=0, cuda=device.type == "cuda",
    )
    payload = torch.load(model_file, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device).eval()
    print(f"[fDNC] released checkpoint: {model_file}")
    return model, model_file


@torch.no_grad()
def fdnc_matrix(model, ref_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    _, out = model([ref_xyz, query_xyz], match_dict=None, ref_idx=0, mode="eval")
    return out["p_m"][1, :len(query_xyz), :len(ref_xyz)].detach().float().cpu().numpy()


def import_cpd():
    src = FDNC_SOURCE / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from cpd_rigid_sep import register_rigid
    from cpd_nonrigid_sep import register_nonrigid
    return register_rigid, register_nonrigid


def cpd_transform_scores(ref_xyz: np.ndarray, query_xyz: np.ndarray) -> np.ndarray:
    """
    Leakage-safe CPD replay.

    Original fDNC helper chooses original-vs-inverted orientation using GT
    match_dict distances. That is forbidden here. We instead choose orientation
    solely by unsupervised post-rigid symmetric Chamfer, then run non-rigid CPD.
    """
    register_rigid, register_nonrigid = import_cpd()
    w, lamb, beta = 0.1, 4e3, 0.25
    ref = normalize_xyz(ref_xyz).astype(np.float64)
    mov = normalize_xyz(query_xyz).astype(np.float64)

    candidates = []
    for reflected in (False, True):
        x = mov.copy()
        if reflected:
            x[:, :2] *= -1.0
        rigid, _, _, _ = register_rigid(ref, x, w=w, fix_scale=True)
        score = chamfer(np.asarray(rigid), ref)
        candidates.append((score, np.asarray(rigid)))
    candidates.sort(key=lambda t: t[0])
    mov_rigid = candidates[0][1]
    moved = register_nonrigid(ref, mov_rigid, w=w, lamb=lamb, beta=beta)
    moved = np.asarray(moved)
    d2 = np.sum((moved[:, None, :] - ref[None, :, :]) ** 2, axis=2)
    return -d2.astype(np.float64)


def score_to_identity(raw_scores: np.ndarray, template: dict):
    rmap = unique_map(template)
    identities = sorted(rmap)
    cols = np.asarray([rmap[x] for x in identities], dtype=np.int64)
    return identities, raw_scores[:, cols]


def metric_from_identity_scores(
    query: dict,
    identities: list[str],
    scores: np.ndarray,
    clean_denominator: int | None,
):
    qmap = unique_map(query)
    id_col = {lab: i for i, lab in enumerate(identities)}
    observed = len(qmap)
    covered = 0
    hit1 = hit5 = 0
    rr_sum = 0.0

    qrows = []
    for lab, qi in sorted(qmap.items()):
        col = id_col.get(lab)
        if col is None:
            qrows.append((lab, qi, None, None))
            continue
        covered += 1
        row = scores[qi]
        rank = 1 + int(np.sum(row > row[col]))
        hit1 += int(rank <= 1)
        hit5 += int(rank <= min(5, len(identities)))
        rr_sum += 1.0 / rank
        qrows.append((lab, qi, col, rank))

    # Hungarian over all observed query rows; GT-uncovered/unassigned count wrong.
    assignment_hits = 0
    if observed and identities:
        labs_rows = sorted(qmap.items())
        rows_idx = np.asarray([qi for _, qi in labs_rows], dtype=np.int64)
        mat = scores[rows_idx]
        rr, cc = linear_sum_assignment(-mat)
        assign = {int(r): int(c) for r, c in zip(rr, cc)}
        for local_row, (lab, qi) in enumerate(labs_rows):
            true_col = id_col.get(lab)
            if true_col is not None and assign.get(local_row, -1) == true_col:
                assignment_hits += 1

    denom = max(observed, 1)
    m = {
        "observed_queries": int(observed),
        "covered_queries": int(covered),
        "coverage": float(covered / denom),
        "top1_observed": float(hit1 / denom),
        "top5_observed": float(hit5 / denom),
        "mrr_observed": float(rr_sum / denom),
        "assignment_top1_observed": float(assignment_hits / denom),
        "top1_covered": float(hit1 / max(covered, 1)),
    }
    if clean_denominator is not None:
        m["effective_top1_vs_clean_queries"] = float(hit1 / max(clean_denominator, 1))
    return m, qrows


def aggregate_worm_metrics(items: list[dict], clean_denominator: int | None):
    keys = [
        "observed_queries", "covered_queries",
    ]
    observed = sum(x["observed_queries"] for x in items)
    covered = sum(x["covered_queries"] for x in items)

    # Recover hit-equivalents exactly enough from denominators.
    h1 = sum(x["top1_observed"] * x["observed_queries"] for x in items)
    h5 = sum(x["top5_observed"] * x["observed_queries"] for x in items)
    rr = sum(x["mrr_observed"] * x["observed_queries"] for x in items)
    ha = sum(x["assignment_top1_observed"] * x["observed_queries"] for x in items)
    denom = max(observed, 1)
    out = {
        "observed_queries": int(observed),
        "covered_queries": int(covered),
        "coverage": float(covered / denom),
        "top1_observed": float(h1 / denom),
        "top5_observed": float(h5 / denom),
        "mrr_observed": float(rr / denom),
        "assignment_top1_observed": float(ha / denom),
        "top1_covered": float(h1 / max(covered, 1)),
    }
    if clean_denominator is not None:
        out["effective_top1_vs_clean_queries"] = float(
            h1 / max(clean_denominator, 1)
        )
    return out


def flatten_numbers(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numbers(v, key))
    elif isinstance(obj, (int, float)) and math.isfinite(float(obj)):
        out[prefix.lower()] = float(obj)
    return out


def compare_saved_clean(method: str, fold: int, metrics: dict):
    root = CPD_CLEAN_ROOT if method == "cpd" else FDNC_CLEAN_ROOT
    path = root / f"fold{fold}" / "metrics.json"
    if not path.is_file():
        print(f"[CLEAN GUARD WARNING] no saved clean artifact: {path}")
        return {"status": "missing_saved_clean"}

    nums = flatten_numbers(jload(path))
    aliases = {
        "top1_observed": ("ranking_top1", "top1_real", "top1"),
        "top5_observed": ("ranking_top5", "top5_real", "top5"),
        "mrr_observed": ("mrr_real", "mrr"),
        "assignment_top1_observed": ("assignment_top1", "hungarian_accuracy",
                                     "hungarian_top1", "hungarian"),
    }
    found = {}
    for ours, suffixes in aliases.items():
        candidates = [
            (k, v) for k, v in nums.items()
            if any(k.endswith("." + s) or k == s for s in suffixes)
        ]
        if candidates:
            # Prefer the shortest path (= most top-level/final-looking).
            candidates.sort(key=lambda kv: (kv[0].count("."), len(kv[0])))
            found[ours] = candidates[0]

    if not found:
        print(f"[CLEAN GUARD WARNING] could not identify comparable metrics in {path}")
        return {"status": "unparsed", "path": str(path)}

    diffs = {}
    for ours, (key, expected) in found.items():
        diffs[ours] = {
            "saved_key": key,
            "saved": expected,
            "replay": float(metrics[ours]),
            "abs_diff": abs(expected - float(metrics[ours])),
        }
    max_diff = max(x["abs_diff"] for x in diffs.values())
    status = "exact" if max_diff <= 1e-12 else "mismatch"
    print(f"[CLEAN GUARD {status.upper()}] {method} fold{fold} maxdiff={max_diff:.3g}")
    return {"status": status, "path": str(path), "diffs": diffs}


def condition_rows(manifest: dict, fold: int, kinds: set[str]):
    rows = [
        x for x in manifest["conditions"]
        if int(x["fold"]) == fold and str(x["kind"]) in kinds
    ]
    order = {"coord_noise": 0, "missing": 1, "outlier": 2}
    rows.sort(key=lambda x: (
        order[str(x["kind"])],
        float(x["severity"]),
        int(x["perturbation_seed"]),
    ))
    return rows


def eval_one_method(
    method: str,
    fold: int,
    row: dict,
    train: list[dict],
    template: dict,
    model,
    model_file,
    out_root: Path,
    clean_denom: int | None,
):
    test_root = Path(row["root"]) / "test"
    tests = [load_worm(p) for p in sorted(test_root.glob("*.npz"))]
    if not tests:
        raise RuntimeError(f"No tests: {test_root}")

    template_ids = None
    worm_metrics = []
    csv_rows = []

    for q in tests:
        if method == "fdnc":
            raw = fdnc_matrix(
                model,
                normalize_xyz(template["xyz"]),
                normalize_xyz(q["xyz"]),
            )
        elif method == "cpd":
            raw = cpd_transform_scores(template["xyz"], q["xyz"])
        else:
            raise ValueError(method)

        identities, scores = score_to_identity(raw, template)
        if template_ids is None:
            template_ids = identities
        elif identities != template_ids:
            raise RuntimeError("Template identity universe changed")

        m, qrows = metric_from_identity_scores(q, identities, scores, clean_denom)
        worm_metrics.append(m)

        # Long score file for audit/recomputation.
        qmap = unique_map(q)
        for lab, qi in sorted(qmap.items()):
            rid = q["roi"][qi]
            try:
                rid = int(rid)
            except Exception:
                rid = int(qi)
            query_uid = f"{q['uid']}::{rid}"
            for ci, cand in enumerate(identities):
                csv_rows.append({
                    "query_uid": query_uid,
                    "group_uid": q["uid"],
                    "gt_label": lab,
                    "candidate_label": cand,
                    "score": float(scores[qi, ci]),
                    "assignment_score": float(scores[qi, ci]),
                    "reference_uid": template["uid"],
                })

    metrics = aggregate_worm_metrics(worm_metrics, clean_denom)
    kind = str(row["kind"])
    sev = float(row["severity"])
    ps = int(row["perturbation_seed"])
    cond = f"{kind}_l{sev:.2f}_p{ps}"
    dst = out_root / method / f"fold{fold}" / cond
    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(csv_rows).to_csv(dst / "candidate_scores.csv", index=False)

    result = {
        "method": method,
        "fold": fold,
        "kind": kind,
        "severity": sev,
        "perturbation_seed": ps,
        "reference_policy": "single clean outer-train geometry medoid",
        "reference_worm": template["uid"],
        "reference_file": str(template["path"]),
        "test_root": str(test_root.resolve()),
        "metrics": metrics,
        "protocol": {
            "retrained_on_corruption": False,
            "full_point_cloud_input": True,
            "synthetic_outliers_are_input_context_but_not_GT": True,
            "query_GT": "unique supervised surviving identities",
        },
    }
    if method == "fdnc":
        result["official_checkpoint"] = str(model_file)
        result["official_checkpoint_sha256"] = FDNC_MODEL_SHA256
        result["official_commit"] = FDNC_EXPECTED_COMMIT
        result["normalization"] = "per-worm median center / 200"
    else:
        result["cpd"] = {
            "w": 0.1,
            "lambda": 4000.0,
            "beta": 0.25,
            "orientation_choice": "unsupervised post-rigid symmetric Chamfer; NEVER GT",
            "score": "negative squared Euclidean distance after non-rigid CPD",
            "normalization": "per-worm median center / 200",
        }
    jdump(dst / "metrics.json", result)
    print(f"[{method.upper()}] fold{fold} {cond}: "
          f"Q={metrics['observed_queries']} "
          f"Top1={100*metrics['top1_observed']:.2f}% "
          f"Top5={100*metrics['top5_observed']:.2f}% "
          f"Cov={100*metrics['coverage']:.2f}%")
    return result


def summarize(results: list[dict], out_root: Path):
    rows = []
    for x in results:
        m = x["metrics"]
        rows.append({
            "method": x["method"],
            "fold": x["fold"],
            "kind": x["kind"],
            "severity": x["severity"],
            "perturbation_seed": x["perturbation_seed"],
            **m,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "cpd_fdnc_all_cells.csv", index=False)

    # p-seeds mean inside fold, then biological folds macro.
    metrics = [
        "top1_observed", "top5_observed", "mrr_observed",
        "assignment_top1_observed", "coverage", "top1_covered",
        "effective_top1_vs_clean_queries",
    ]
    existing = [x for x in metrics if x in df.columns]
    fold_df = (
        df.groupby(["method", "kind", "severity", "fold"], as_index=False)[existing]
        .mean()
    )
    fold_df.to_csv(out_root / "cpd_fdnc_fold_level.csv", index=False)

    agg_rows = []
    for keys, g in fold_df.groupby(["method", "kind", "severity"], sort=True):
        row = dict(method=keys[0], kind=keys[1], severity=keys[2], folds=len(g))
        for k in existing:
            vals = g[k].dropna().to_numpy(float)
            if len(vals):
                row[k + "_mean"] = float(vals.mean())
                row[k + "_sd"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg_rows.append(row)
    pd.DataFrame(agg_rows).to_csv(out_root / "cpd_fdnc_macro_summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="cpd")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--kinds", default="coord_noise,missing,outlier")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument(
        "--allow-clean-mismatch",
        action="store_true",
        help="Diagnostic only. Paper runs should leave this OFF.",
    )
    args = ap.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    if not set(methods) <= {"cpd", "fdnc"}:
        raise ValueError(methods)

    manifest = jload(CORR_ROOT / "MANIFEST.json")
    device = torch.device(args.device)
    fdnc_model = fdnc_file = None
    if "fdnc" in methods:
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        fdnc_model, fdnc_file = build_fdnc(device)

    results = []
    clean_denoms = {m: {} for m in methods}

    for fold in folds:
        train = [load_worm(p) for p in sorted((CV_ROOT / f"fold_{fold}" / "train").glob("*.npz"))]
        if not train:
            raise FileNotFoundError(CV_ROOT / f"fold_{fold}" / "train")
        template = select_train_medoid(fold, train)
        rows = condition_rows(manifest, fold, kinds)

        # Require clean severity=0 first when coord_noise is in requested kinds.
        clean_rows = [
            r for r in manifest["conditions"]
            if int(r["fold"]) == fold
            and str(r["kind"]) == "coord_noise"
            and abs(float(r["severity"])) < 1e-12
            and int(r["perturbation_seed"]) == 0
        ]
        if len(clean_rows) != 1:
            raise RuntimeError(f"fold{fold}: clean coord row count={len(clean_rows)}")

        for method in methods:
            clean = eval_one_method(
                method, fold, clean_rows[0], train, template,
                fdnc_model if method == "fdnc" else None,
                fdnc_file if method == "fdnc" else None,
                OUT_ROOT, None,
            )
            denom = int(clean["metrics"]["observed_queries"])
            clean_denoms[method][fold] = denom
            guard = compare_saved_clean(method, fold, clean["metrics"])
            clean["clean_replay_guard"] = guard
            clean_path = (
                OUT_ROOT / method / f"fold{fold}" /
                "coord_noise_l0.00_p0" / "metrics.json"
            )
            jdump(clean_path, clean)

            if guard.get("status") == "mismatch" and not args.allow_clean_mismatch:
                raise RuntimeError(
                    f"{method} fold{fold} severity=0 does not reproduce saved clean "
                    f"baseline. STOP before nonzero corruptions. See {clean_path}"
                )

            results.append(clean)

        for row in rows:
            if (str(row["kind"]) == "coord_noise"
                    and abs(float(row["severity"])) < 1e-12
                    and int(row["perturbation_seed"]) == 0):
                continue
            for method in methods:
                result = eval_one_method(
                    method, fold, row, train, template,
                    fdnc_model if method == "fdnc" else None,
                    fdnc_file if method == "fdnc" else None,
                    OUT_ROOT, clean_denoms[method][fold],
                )
                results.append(result)

    summarize(results, OUT_ROOT)
    print("\nCOMPLETE")
    print(OUT_ROOT / "cpd_fdnc_macro_summary.csv")


if __name__ == "__main__":
    main()
