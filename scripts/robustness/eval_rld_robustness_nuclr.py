#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
RUN_ONE = ROOT / "baselines/official/adapters/nuclr_official_scratch50k/run_one.py"
DEFAULT_MANIFEST = ROOT / "runs/rld_robustness_cv5_seed42_v1/corruptions/MANIFEST.json"
DEFAULT_CLEAN_ROOT = ROOT / "baselines/official/runs/nuclr_official_scratch50k_cv5x3"
DEFAULT_OUT = ROOT / "runs/rld_robustness_cv5_seed42_v1/results/nuclr_official_scratch50k_seed42"

METRIC_KEYS = (
    "queries",
    "ranking_top1",
    "top3",
    "top5",
    "top10",
    "mrr",
    "mean_rank",
    "assignment_top1",
)


def load_run_one():
    if not RUN_ONE.is_file():
        raise FileNotFoundError(RUN_ONE)
    # The original runner has its own local sys.path setup.  Importing the file
    # by path preserves all original functions/classes without copying model code.
    spec = importlib.util.spec_from_file_location("nuclr_official_scratch50k_run_one", RUN_ONE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RUN_ONE}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_folds(s: str):
    out = [int(x) for x in s.split(",") if x.strip()]
    bad = [x for x in out if x not in range(5)]
    if bad:
        raise ValueError(f"Robustness folds must be 0..4, got {bad}")
    return out


def metric_diff(a: dict, b: dict):
    diffs = {}
    for k in METRIC_KEYS:
        if k not in a or k not in b:
            continue
        av, bv = a[k], b[k]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            diffs[k] = float(abs(float(av) - float(bv)))
    return diffs


def assert_same_metrics(a: dict, b: dict, name: str, tol: float = 1e-12):
    diffs = metric_diff(a, b)
    bad = {k: v for k, v in diffs.items() if v > tol}
    if bad:
        raise RuntimeError(
            f"{name} did NOT reproduce exactly. metric diffs={bad}\n"
            f"A={a}\nB={b}"
        )


def collect_outlier_labels(test_paths):
    """
    The corruption generator inserts labels such as __OUTLIER_s0_0000.
    They must remain unmatched / unsupervised and must never become a new
    canonical identity.  Return those strings so the legacy loader can map
    them to -1 if it requires an explicit vocabulary entry.
    """
    out = set()
    for p in test_paths:
        with np.load(p, allow_pickle=True) as z:
            for key in ("cell_id", "cell_id_alt", "labels", "label"):
                if key not in z.files:
                    continue
                arr = np.asarray(z[key]).reshape(-1)
                for x in arr.tolist():
                    s = str(x)
                    if s.startswith("__OUTLIER_"):
                        out.add(s)
    return sorted(out)


def patched_label_map(label_to_int, test_paths):
    # Usually load_split already treats unknown/unlabeled cells safely.
    # This explicit patch only affects synthetic inserted-outlier labels.
    m = copy.deepcopy(label_to_int)
    labels = collect_outlier_labels(test_paths)
    if not labels:
        return m, 0

    # Expected type is a normal dict in benchmark_cv5x3_common.
    # Fail closed rather than silently changing evaluation semantics.
    if not hasattr(m, "__setitem__"):
        raise TypeError(
            f"label_to_int is not mutable ({type(m)}); cannot safely mark "
            "synthetic outliers as unmatched."
        )
    for s in labels:
        m[s] = -1
    return m, len(labels)



def _raw_worm_id(path: Path) -> str:
    path = Path(path)
    with np.load(path, allow_pickle=True) as z:
        for key in ("recording_uid", "worm_id"):
            if key in z.files:
                arr = np.asarray(z[key]).reshape(-1)
                if len(arr):
                    value = arr[0]
                    if isinstance(value, np.generic):
                        value = value.item()
                    if isinstance(value, bytes):
                        value = value.decode("utf-8", errors="replace")
                    value = str(value)
                    if value:
                        return value
    parts = path.stem.split("__")
    if len(parts) >= 3:
        return parts[1]
    return path.stem


def resolve_nuclr_fold_mapping(r, manifest: dict, robust_folds, seed: int):
    """
    Resolve robustness fold0..4 -> official NuCLR fold_1..fold_5 by exact
    OUTER-TEST worm-set identity.  Never assume a numeric offset.
    """
    robust_sets = {}
    for rf in robust_folds:
        candidates = [
            row for row in manifest["conditions"]
            if int(row["fold"]) == int(rf)
            and str(row["kind"]) == "coord_noise"
            and abs(float(row["severity"])) < 1e-12
            and int(row["perturbation_seed"]) == 0
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"Need exactly one clean coord_noise row for robustness fold{rf}, "
                f"found {len(candidates)}"
            )
        test_dir = Path(candidates[0]["root"]) / "test"
        paths = sorted(test_dir.glob("*.npz"))
        if not paths:
            raise RuntimeError(f"No clean corruption test files: {test_dir}")
        ids = {_raw_worm_id(p) for p in paths}
        if len(ids) != len(paths):
            raise RuntimeError(
                f"Duplicate recording_uid/worm_id in robustness fold{rf}"
            )
        robust_sets[int(rf)] = ids

    official_sets = {}
    for of in range(1, 6):
        cfg, locked_args, label_to_int, train_common, val_common = (
            r.locked_train_val_context("rld", of, seed)
        )
        test_paths = r.common.read_list(locked_args.test_list)
        test_common = r.common.legacy.load_split(
            test_paths,
            label_to_int,
            locked_args,
        )
        ids = {str(rec.worm_id) for rec in test_common}
        if len(ids) != len(test_common):
            raise RuntimeError(f"Duplicate official test worm_id in fold_{of}")
        official_sets[of] = ids

    print("\nOFFICIAL NUCLR FOLD MAPPING AUDIT")
    print("rows = robustness folds; columns = official NuCLR folds")
    header = "          " + " ".join([f"of{of:>2}" for of in range(1, 6)])
    print(header)
    for rf in robust_folds:
        cells = []
        a = robust_sets[int(rf)]
        for of in range(1, 6):
            b = official_sets[of]
            inter = len(a & b)
            union = len(a | b)
            jac = inter / union if union else 1.0
            cells.append(f"{inter:02d}/{len(a):02d}:{jac:.2f}")
        print(f"rob f{rf}: " + " ".join(cells))

    mapping = {}
    used = set()
    for rf in robust_folds:
        exact = [
            of for of in range(1, 6)
            if robust_sets[int(rf)] == official_sets[of]
        ]
        if len(exact) != 1:
            details = []
            a = robust_sets[int(rf)]
            for of in range(1, 6):
                b = official_sets[of]
                details.append({
                    "official_fold": of,
                    "robust_n": len(a),
                    "official_n": len(b),
                    "intersection": len(a & b),
                    "missing_from_official": sorted(a - b)[:5],
                    "extra_in_official": sorted(b - a)[:5],
                })
            raise RuntimeError(
                f"Could not uniquely exact-match robustness fold{rf} to an "
                f"official NuCLR fold. Details={details}"
            )
        of = exact[0]
        if of in used:
            raise RuntimeError(
                f"Official fold_{of} matched more than one robustness fold"
            )
        used.add(of)
        mapping[int(rf)] = int(of)

    print("LOCKED FOLD MAPPING:", mapping, flush=True)
    return mapping


def build_fold_context(r, robust_fold: int, nuclr_fold: int, seed: int, device: torch.device, clean_root: Path):

    cfg, locked_args, label_to_int, train_common, val_common = r.locked_train_val_context(
        "rld", nuclr_fold, seed
    )
    train_records = r.make_nuclr_records(train_common, "train")

    # Load the exact original prepared OUTER-TEST records used by the official
    # clean benchmark.  These records define the evaluation label universe
    # (including any clean/labeled filtering already applied upstream).
    #
    # This is evaluation metadata only; checkpoint and template selection were
    # already locked from train/val before test is opened.
    clean_test_paths = r.common.read_list(locked_args.test_list)
    clean_test_common = r.common.legacy.load_split(
        clean_test_paths,
        label_to_int,
        locked_args,
    )

    clean_test_by_worm_id = {}
    clean_test_by_source_name = {}
    for rec in clean_test_common:
        raw_source = r.common.source_npz(rec).resolve()

        worm_key = str(rec.worm_id)
        if worm_key in clean_test_by_worm_id:
            raise RuntimeError(f"Duplicate clean test worm_id: {worm_key}")
        clean_test_by_worm_id[worm_key] = {
            "record": rec,
            "raw_source": raw_source,
        }

        # Keep basename only as a secondary fallback because grouped-CV
        # materialized filenames can differ from the original raw source name.
        clean_test_by_source_name.setdefault(
            raw_source.name,
            {
                "record": rec,
                "raw_source": raw_source,
            },
        )

    # Reproduce the exact clean runner's train-only geometry medoid calculation.
    train_geometry = []
    for record in train_common:
        xyz = (
            record.xyz.detach().cpu().numpy()
            if torch.is_tensor(record.xyz)
            else np.asarray(record.xyz)
        )
        xyz = np.asarray(xyz, dtype=np.float64)[:, :3]
        train_geometry.append(
            (xyz - np.median(xyz, axis=0, keepdims=True)) / 200.0
        )

    medoid = r.select_geometry_medoid(
        [str(record.worm_id) for record in train_common], train_geometry
    )
    template_common = train_common[medoid.index]
    template_record = train_records[medoid.index]

    clean_dir = clean_root / "rld" / f"fold_{nuclr_fold}" / f"seed_{seed}"
    clean_result_path = clean_dir / "outer_test_medoid_template_v1" / "result.json"
    if not clean_result_path.is_file():
        raise FileNotFoundError(clean_result_path)
    clean_result = read_json(clean_result_path)

    if int(clean_result.get("target_train_steps", -1)) != 50000:
        raise RuntimeError(f"Not official 50k result: {clean_result_path}")
    if int(clean_result.get("final_global_step", -1)) != 50000:
        raise RuntimeError(f"Training did not finish 50k: {clean_result_path}")

    expected_template = str(clean_result["protocol"]["reference_worm"])
    actual_template = str(template_common.worm_id)
    if expected_template != actual_template:
        raise RuntimeError(
            f"Template mismatch robust_fold={robust_fold}: "
            f"clean result={expected_template}, recomputed={actual_template}"
        )

    checkpoint = Path(clean_result["selected_checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = r.s1.build_official_model(device)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise RuntimeError(f"Missing model_state_dict in {checkpoint}")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return {
        "nuclr_fold": nuclr_fold,
        "locked_args": locked_args,
        "label_to_int": label_to_int,
        "template_common": template_common,
        "template_record": template_record,
        "template_worm": actual_template,
        "checkpoint": checkpoint,
        "selected_global_step": int(clean_result["selected_global_step"]),
        "clean_result": clean_result,
        "clean_test_common": clean_test_common,
        "clean_test_by_worm_id": clean_test_by_worm_id,
        "clean_test_by_source_name": clean_test_by_source_name,
        "model": model,
    }



def _as_1d(a):
    return np.asarray(a).reshape(-1)


def _unique_nonnegative_index_map(values):
    """
    Return value -> row index when the key is unique among non-negative rows.
    Synthetic outliers use -1 for roi_index/aligned_table_id and are ignored.
    """
    vals = _as_1d(values)
    out = {}
    for i, v in enumerate(vals.tolist()):
        try:
            key = int(v)
        except Exception:
            return None
        if key < 0:
            continue
        if key in out:
            return None
        out[key] = i
    return out


def _labels_for_corruption(clean_rec, clean_raw_path, corrupt_path):
    """
    Propagate the EXACT prepared clean-test labels to the corrupted neuron rows.

    coord_noise:
        same neuron rows/order -> labels copied identically.

    missing:
        corruption generator subsets every neuron-axis array with a sorted keep
        index.  We recover surviving original rows by roi_index (preferred) or
        aligned_table_id.

    outlier:
        generator appends synthetic rows after all original rows.  Original
        prepared labels are copied for the prefix; appended rows are -1.

    This deliberately does NOT infer evaluation labels from raw cell_id.
    """
    clean_labels = clean_rec.labels.detach().cpu().numpy().astype(np.int64, copy=True)

    with np.load(clean_raw_path, allow_pickle=True) as z0:
        clean_payload = {k: np.asarray(z0[k]) for k in z0.files}

    with np.load(corrupt_path, allow_pickle=True) as z1:
        cur_payload = {k: np.asarray(z1[k]) for k in z1.files}

    if "activity_raw" not in cur_payload or "xyz" not in cur_payload:
        raise KeyError(f"{corrupt_path}: raw activity/xyz missing")

    clean_n = len(clean_labels)
    cur_xyz = np.asarray(cur_payload["xyz"])
    cur_n = int(cur_xyz.shape[0])

    if clean_n <= 0:
        raise RuntimeError(f"{clean_raw_path}: empty prepared clean labels")

    # Clean raw source and prepared record must refer to the exact same rows.
    source_n = None
    for key in ("xyz", "activity_raw", "cell_id"):
        if key in clean_payload:
            a = np.asarray(clean_payload[key])
            if a.ndim >= 1:
                if key == "activity_raw":
                    # Normally [N,T]; tolerate transposed activity.
                    if a.shape[0] == clean_n:
                        source_n = clean_n
                        break
                    if a.ndim == 2 and a.shape[1] == clean_n:
                        source_n = clean_n
                        break
                elif a.shape[0] == clean_n:
                    source_n = clean_n
                    break
    if source_n != clean_n:
        raise RuntimeError(
            f"Prepared clean labels ({clean_n}) do not align with raw source "
            f"{clean_raw_path}"
        )

    # Coordinate-noise clean/same-cardinality case, and outlier case:
    # the materializer preserves every original neuron in the original order.
    if cur_n >= clean_n:
        labels = np.full(cur_n, -1, dtype=np.int64)
        labels[:clean_n] = clean_labels

        # Strong audit: when roi_index exists, verify that the original prefix
        # is truly unchanged rather than merely assuming it.
        if "roi_index" in clean_payload and "roi_index" in cur_payload:
            a = _as_1d(clean_payload["roi_index"])
            b = _as_1d(cur_payload["roi_index"])
            if len(a) == clean_n and len(b) >= clean_n:
                if not np.array_equal(a, b[:clean_n]):
                    raise RuntimeError(
                        f"{corrupt_path}: original roi_index prefix changed"
                    )
        return labels

    # Missing-neuron case: map the surviving rows back to clean rows.
    for key in ("roi_index", "aligned_table_id"):
        if key not in clean_payload or key not in cur_payload:
            continue

        clean_vals = _as_1d(clean_payload[key])
        cur_vals = _as_1d(cur_payload[key])

        if len(clean_vals) != clean_n or len(cur_vals) != cur_n:
            continue

        clean_map = _unique_nonnegative_index_map(clean_vals)
        if clean_map is None:
            continue

        labels = np.full(cur_n, -1, dtype=np.int64)
        matched = 0
        for i, v in enumerate(cur_vals.tolist()):
            try:
                token = int(v)
            except Exception:
                continue
            if token < 0:
                continue
            j = clean_map.get(token)
            if j is not None:
                labels[i] = clean_labels[j]
                matched += 1

        if matched == cur_n:
            return labels

    # Final fallback: unique non-empty cell_id.  Only rows that can be matched
    # uniquely are propagated; everything else remains non-evaluable (-1).
    if "cell_id" in clean_payload and "cell_id" in cur_payload:
        clean_ids = [str(x) for x in _as_1d(clean_payload["cell_id"]).tolist()]
        cur_ids = [str(x) for x in _as_1d(cur_payload["cell_id"]).tolist()]

        positions = {}
        duplicate = set()
        for i, token in enumerate(clean_ids):
            if token in positions:
                duplicate.add(token)
            else:
                positions[token] = i
        for token in duplicate:
            positions.pop(token, None)

        labels = np.full(cur_n, -1, dtype=np.int64)
        matched = 0
        for i, token in enumerate(cur_ids):
            if token.startswith("__OUTLIER_"):
                continue
            j = positions.get(token)
            if j is not None:
                labels[i] = clean_labels[j]
                matched += 1

        if matched > 0:
            return labels

    raise RuntimeError(
        f"Could not propagate clean prepared labels to missing-neuron corruption: "
        f"{corrupt_path}"
    )


def raw_corruption_to_common_records(r, ctx, test_paths):
    """
    Build benchmark WormRecord objects from raw corruption NPZs while preserving
    the EXACT original official clean-test evaluation labels.
    """
    record_cls = type(ctx["template_common"])
    activity_dim = int(ctx["locked_args"].activity_dim)
    clean_by_worm = ctx["clean_test_by_worm_id"]
    clean_by_name = ctx["clean_test_by_source_name"]

    records = []
    synthetic_outlier_labels = 0

    for path in test_paths:
        path = Path(path).resolve()

        # Resolve the stable recording identity from the corruption NPZ.
        # The grouped-CV filename may be:
        #   test__20240902-17-33-24__9b2f6f70.npz
        # while the official prepared record may resolve source_npz() to a
        # different original basename. worm_id/recording_uid is the stable key.
        corrupt_worm_id = ""
        with np.load(path, allow_pickle=True) as z:
            for key in ("recording_uid", "worm_id"):
                if key in z.files:
                    arr = np.asarray(z[key]).reshape(-1)
                    if len(arr):
                        value = arr[0]
                        if isinstance(value, np.generic):
                            value = value.item()
                        if isinstance(value, bytes):
                            value = value.decode("utf-8", errors="replace")
                        corrupt_worm_id = str(value)
                        if corrupt_worm_id:
                            break

        if not corrupt_worm_id:
            parts = path.stem.split("__")
            if len(parts) >= 3:
                corrupt_worm_id = parts[1]
            else:
                corrupt_worm_id = path.stem

        clean_info = clean_by_worm.get(corrupt_worm_id)

        # Secondary fallback for datasets whose prepared worm_id convention
        # differs but whose raw source basename is preserved.
        if clean_info is None:
            clean_info = clean_by_name.get(path.name)

        if clean_info is None:
            sample_clean_ids = sorted(clean_by_worm)[:10]
            raise RuntimeError(
                "Corrupted test record cannot be matched to official clean test. "
                f"corrupt_file={path.name}, corrupt_worm_id={corrupt_worm_id!r}, "
                f"sample_clean_worm_ids={sample_clean_ids}"
            )

        clean_rec = clean_info["record"]
        clean_raw_path = clean_info["raw_source"]

        if str(clean_rec.worm_id) != corrupt_worm_id:
            # Filename fallback is allowed only when stable IDs cannot be read.
            # Print-worthy mismatch is still made explicit in the result path.
            pass

        labels = _labels_for_corruption(
            clean_rec,
            clean_raw_path,
            path,
        )

        with np.load(path, allow_pickle=True) as z:
            if "activity_raw" not in z.files:
                raise KeyError(f"{path}: missing activity_raw")
            if "xyz" not in z.files:
                raise KeyError(f"{path}: missing xyz")

            activity = np.asarray(z["activity_raw"])
            xyz = np.asarray(z["xyz"], dtype=np.float32)

            if xyz.ndim != 2 or xyz.shape[1] < 3:
                raise ValueError(f"{path}: xyz must be [N,>=3], got {xyz.shape}")
            xyz = np.asarray(xyz[:, :3], dtype=np.float32)

            n = len(labels)
            if xyz.shape[0] != n:
                raise ValueError(
                    f"{path}: xyz rows={xyz.shape[0]} != propagated labels={n}"
                )

            if activity.ndim != 2:
                raise ValueError(
                    f"{path}: activity_raw must be 2-D, got {activity.shape}"
                )
            if activity.shape[0] != n and activity.shape[1] == n:
                activity = activity.T
            if activity.shape[0] != n:
                raise ValueError(
                    f"{path}: activity rows={activity.shape[0]} != neurons={n}"
                )

            if "cell_id" in z.files:
                for value in np.asarray(z["cell_id"]).reshape(-1).tolist():
                    if str(value).startswith("__OUTLIER_"):
                        synthetic_outlier_labels += 1

        # Critical: preserve the exact official clean record worm_id so query
        # identity / source matching is identical to the original benchmark.
        worm_id = str(clean_rec.worm_id)

        dummy_nuclr = np.zeros((n, activity_dim), dtype=np.float32)

        records.append(
            record_cls(
                worm_id=worm_id,
                embedding_path=str(path),
                source_path=str(path),
                xyz=torch.from_numpy(xyz),
                nuclr_emb=torch.from_numpy(dummy_nuclr),
                labels=torch.from_numpy(labels),
            )
        )

    if len(records) != len(test_paths):
        raise RuntimeError("Raw corruption adapter dropped test records")

    return records, synthetic_outlier_labels


def evaluate_condition(r, ctx, robust_fold, seed, row, device, out_root):
    condition_root = Path(row["root"])
    test_dir = condition_root / "test"
    test_paths = sorted(test_dir.glob("*.npz"))
    if not test_paths:
        raise RuntimeError(f"No test NPZ files: {test_dir}")

    # Critically: only TEST is changed. Model/checkpoint/template all came
    # from the already locked clean run above. Corruption files are RAW RLD
    # NPZs, so adapt them directly instead of legacy.load_split(), which expects
    # prepared benchmark embedding NPZs containing `nuclr_emb` + `labels`.
    test_common, num_outlier_labels = raw_corruption_to_common_records(
        r, ctx, test_paths
    )

    # Ensure no stale raw-activity block can survive across corruption
    # conditions. (The cache key already contains source_path; clearing here is
    # an additional audit safeguard.)
    if hasattr(r.common, "_ACTIVITY_CACHE"):
        r.common._ACTIVITY_CACHE.clear()

    # This reads activity_raw through each record.source_path and performs the
    # original official z-score preprocessing before frozen NuCLR inference.
    test_records = r.make_nuclr_records(test_common, "test")

    metrics, query_rows = r.evaluate_model_against_template(
        ctx["model"],
        test_records,
        test_common,
        ctx["template_record"],
        ctx["template_common"],
        device,
        "rld",
        ctx["nuclr_fold"],
        seed,
        method="NuCLR (official scratch, 50k SSL) — RLD controlled robustness",
    )

    kind = str(row["kind"])
    severity = float(row["severity"])
    pseed = int(row["perturbation_seed"])
    condition = f"{kind}_l{severity:.2f}_p{pseed}"

    dst = out_root / f"fold{robust_fold}" / condition
    dst.mkdir(parents=True, exist_ok=True)
    query_rows.to_csv(dst / "query_level.csv", index=False)

    clean_q = int(ctx["clean_result"]["metrics"]["queries"])
    q = int(metrics["queries"])
    top1_correct_equiv = float(metrics["ranking_top1"]) * q

    result = {
        "dataset": "rld",
        "robustness_fold": int(robust_fold),
        "nuclr_outer_fold": int(ctx["nuclr_fold"]),
        "seed": int(seed),
        "kind": kind,
        "severity": severity,
        "perturbation_seed": pseed,
        "method": "NuCLR (official scratch, 50k SSL)",
        "frozen_checkpoint": str(ctx["checkpoint"].resolve()),
        "selected_global_step": int(ctx["selected_global_step"]),
        "final_global_step": 50000,
        "frozen_reference_worm": str(ctx["template_worm"]),
        "corrupted_test_root": str(test_dir.resolve()),
        "num_corrupted_test_worms": len(test_paths),
        "synthetic_outlier_labels_forced_unmatched": int(num_outlier_labels),
        "metrics": metrics,
        "derived": {
            "clean_query_denominator": clean_q,
            "surviving_evaluable_query_fraction": (q / clean_q) if clean_q else None,
            "effective_ranking_top1_vs_clean_queries": (
                top1_correct_equiv / clean_q if clean_q else None
            ),
        },
        "protocol": {
            "training": "NONE; frozen clean-run checkpoint",
            "checkpoint_selection": "unchanged validation-selected clean-run checkpoint",
            "template_selection": "NONE at robustness time; exact clean outer-train medoid is reused",
            "test_only_corruption": True,
            "activity_only_model": True,
            "outlier_labels_are_not_added_as_canonical_identities": True,
        },
    }
    write_json(dst / "result.json", result)

    return result


def summarize(results, out_root):
    # First average perturbation seeds within a biological fold, then average
    # biological folds.  This matches the intended robustness protocol.
    metric_names = [
        "ranking_top1", "top5", "mrr", "assignment_top1",
    ]
    derived_names = [
        "surviving_evaluable_query_fraction",
        "effective_ranking_top1_vs_clean_queries",
    ]

    fold_cells = []
    grouped = defaultdict(list)
    for x in results:
        grouped[(x["kind"], x["severity"], x["robustness_fold"])].append(x)

    for (kind, sev, fold), xs in sorted(grouped.items()):
        cell = {
            "kind": kind,
            "severity": sev,
            "fold": fold,
            "num_perturbation_seeds": len(xs),
        }
        for k in metric_names:
            vals = [float(x["metrics"][k]) for x in xs if k in x["metrics"]]
            if vals:
                cell[k] = float(np.mean(vals))
        for k in derived_names:
            vals = [
                float(x["derived"][k])
                for x in xs
                if x["derived"].get(k) is not None
            ]
            if vals:
                cell[k] = float(np.mean(vals))
        fold_cells.append(cell)

    macro = []
    grouped2 = defaultdict(list)
    for c in fold_cells:
        grouped2[(c["kind"], c["severity"])].append(c)

    for (kind, sev), cells in sorted(grouped2.items()):
        row = {
            "kind": kind,
            "severity": sev,
            "num_folds": len(cells),
        }
        for k in metric_names + derived_names:
            vals = [float(c[k]) for c in cells if k in c]
            if vals:
                row[k + "_mean"] = float(np.mean(vals))
                row[k + "_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        macro.append(row)

    pd.DataFrame(fold_cells).to_csv(out_root / "fold_level_summary.csv", index=False)
    pd.DataFrame(macro).to_csv(out_root / "macro_summary.csv", index=False)
    write_json(
        out_root / "SUMMARY.json",
        {
            "aggregation": "mean perturbation seeds within fold, then unweighted mean across biological folds",
            "fold_level": fold_cells,
            "macro": macro,
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--clean-run-root", type=Path, default=DEFAULT_CLEAN_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--kinds",
        default="coord_noise,missing,outlier",
        help="Comma-separated subset of coord_noise,missing,outlier",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.seed != 42:
        raise ValueError("This locked robustness script is intentionally seed42-only.")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("NuCLR robustness replay requires CUDA.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("NuCLR official model requires BF16-capable CUDA.")

    folds = parse_folds(args.folds)
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    allowed = {"coord_noise", "missing", "outlier"}
    if not kinds <= allowed:
        raise ValueError(f"Unknown kinds: {sorted(kinds - allowed)}")

    manifest = read_json(args.manifest)
    rows = [
        x for x in manifest["conditions"]
        if int(x["fold"]) in folds and str(x["kind"]) in kinds
    ]
    if not rows:
        raise RuntimeError("No matching corruption conditions in manifest")

    r = load_run_one()

    # Resolve fold correspondence from exact outer-test worm sets.  This is
    # intentionally done before any model is loaded.
    fold_mapping = resolve_nuclr_fold_mapping(
        r,
        manifest,
        folds,
        args.seed,
    )

    args.out_root.mkdir(parents=True, exist_ok=True)
    results = []

    for robust_fold in folds:
        nuclr_fold = fold_mapping[robust_fold]
        print("\n" + "=" * 120)
        print(f"LOCKED NUCLR CONTEXT robust_fold={robust_fold} -> official fold_{nuclr_fold}")
        print("=" * 120, flush=True)

        ctx = build_fold_context(
            r,
            robust_fold,
            nuclr_fold,
            args.seed,
            device,
            args.clean_run_root,
        )
        print("checkpoint:", ctx["checkpoint"])
        print("template  :", ctx["template_worm"])
        print("clean     :", ctx["clean_result"]["metrics"], flush=True)

        fold_rows = [x for x in rows if int(x["fold"]) == robust_fold]
        # Clean first, then increasing severity.
        fold_rows.sort(
            key=lambda x: (
                {"coord_noise": 0, "missing": 1, "outlier": 2}[str(x["kind"])],
                float(x["severity"]),
                int(x["perturbation_seed"]),
            )
        )

        clean_reference_by_kind = {}

        for row in fold_rows:
            kind = str(row["kind"])
            sev = float(row["severity"])
            ps = int(row["perturbation_seed"])
            condition = f"{kind}_l{sev:.2f}_p{ps}"
            dst = args.out_root / f"fold{robust_fold}" / condition / "result.json"

            if dst.is_file() and not args.overwrite:
                result = read_json(dst)
                print("[REUSE]", f"fold{robust_fold}", condition)
            else:
                print("[EVAL ]", f"fold{robust_fold}", condition, flush=True)
                result = evaluate_condition(
                    r, ctx, robust_fold, args.seed, row, device, args.out_root
                )

            # Severity-zero must exactly reproduce the original clean benchmark.
            if sev == 0.0:
                assert_same_metrics(
                    result["metrics"],
                    ctx["clean_result"]["metrics"],
                    f"fold{robust_fold} {kind} severity=0 clean replay",
                )
                clean_reference_by_kind[kind] = result["metrics"]
                print("[CLEAN EXACT ✓]", kind, result["metrics"], flush=True)

            # NuCLR is activity-only. Coordinate noise changes xyz only, so every
            # coordinate-noise result must be identical to the clean result.
            if kind == "coord_noise":
                assert_same_metrics(
                    result["metrics"],
                    ctx["clean_result"]["metrics"],
                    f"fold{robust_fold} coord_noise severity={sev} pseed={ps}",
                )
                print("[COORD-INVARIANCE ✓]", condition, flush=True)

            results.append(result)

        # Free the large fold model before loading the next fold.
        del ctx["model"]
        torch.cuda.empty_cache()

    summarize(results, args.out_root)

    print("\n" + "=" * 120)
    print("NUCLR RLD CONTROLLED ROBUSTNESS COMPLETE")
    print("=" * 120)
    print("Results :", args.out_root)
    print("Summary :", args.out_root / "macro_summary.csv")


if __name__ == "__main__":
    main()
