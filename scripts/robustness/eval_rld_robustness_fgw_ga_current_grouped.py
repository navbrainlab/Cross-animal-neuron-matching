#!/usr/bin/env python3
"""Evaluate a genuine geometry+activity FGW baseline on RLD activity noise.

This runner deliberately follows the already locked single-medoid benchmark
protocol:

* the shared, label-free outer-training geometry medoid is the reference;
* geometry normalization is fitted on outer training only;
* the FGW mixing coefficient is selected on outer validation only;
* outer test is first opened after the lock has been written;
* test scoring uses the complete canonical main-table query cohort;
* a canonical identity absent from the medoid remains in the denominator and
  receives zero credit;
* corruption draws are averaged within fold, then folds are equally weighted.

Unlike the historical ``Vanilla FGW`` runner (which uses geometry for both
FGW terms), this is genuinely multimodal.  The linear node cost M is squared
Euclidean geometry, while each within-recording structural cost C is the
Euclidean distance between simultaneously recorded, high-pass-filtered
activity traces.  POT's alpha weights the activity/GW term and (1-alpha)
weights the geometry/linear term.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
MEDOID_RUN = ROOT / "runs/mprt_v1_1_atlas_medoid_rld_cv5_test_seeds_42_v1"
CANONICAL = (
    ROOT / "runs/unified_main_benchmark_cv5_seed42_canonical_all_v1/"
    "canonical_query_manifest.csv"
)
CORRUPTIONS = ROOT / "runs/rld_robustness_cv5_seed42_v2/corruptions/MANIFEST.json"
MAIN_OUT = ROOT / "runs/unified_canonical_all_retest_v1/fgw_ga/rld"
ROBUST_OUT = ROOT / "runs/rld_robustness_cv5_seed42_v2/results/fgw_ga"

INVALID_IDS = {"", "nan", "none", "null", "unknown", "unk", "-1", "-1.0"}
QUERY_FIELDS = [
    "dataset", "fold", "seed", "method", "condition", "query_uid",
    "query_index", "identity", "reference_uid", "reference_covered",
    "rank", "top1", "top5", "rr", "hungarian_correct",
    "predicted_index", "predicted_identity", "alpha_activity",
]


@dataclass(frozen=True)
class Worm:
    path: Path
    uid: str
    xyz: np.ndarray
    activity: np.ndarray
    labels: np.ndarray
    supervised: np.ndarray
    sample_rate_hz: float


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_mask(data: np.lib.npyio.NpzFile, key: str, n: int) -> np.ndarray:
    if key not in data.files:
        return np.ones(n, dtype=bool)
    value = np.asarray(data[key], dtype=bool).reshape(-1)
    if value.shape != (n,):
        raise ValueError(f"{key}: expected {(n,)}, got {value.shape}")
    return value


def load_worm(path: Path, cutoff_hz: float) -> Worm:
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"], dtype=np.float64)
        activity = np.asarray(data["activity_raw"], dtype=np.float64)
        labels = np.asarray(data["cell_id"]).astype(str).reshape(-1)
        n = len(labels)
        if xyz.shape != (n, 3):
            raise ValueError(f"{path}: xyz shape {xyz.shape}")
        if activity.ndim != 2:
            raise ValueError(f"{path}: activity shape {activity.shape}")
        if activity.shape[0] != n and activity.shape[1] == n:
            activity = activity.T
        if activity.shape[0] != n:
            raise ValueError(f"{path}: activity shape {activity.shape}, N={n}")
        valid = np.isfinite(xyz).all(axis=1) & as_mask(data, "valid_xyz_mask", n)
        labeled = as_mask(data, "labeled_mask", n)
        certain = as_mask(data, "certain_mask", n)
        clean = as_mask(data, "clean_mask", n)
        valid_id = np.asarray([x.strip().lower() not in INVALID_IDS for x in labels])
        supervised = labeled & certain & clean & valid_id
        uid = (
            str(np.asarray(data["recording_uid"]).reshape(-1)[0])
            if "recording_uid" in data.files else path.stem
        )
        if "timestamps" in data.files:
            timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
            delta = np.diff(timestamps)
            delta = delta[np.isfinite(delta) & (delta > 0)]
            sample_rate = float(1.0 / np.median(delta)) if len(delta) else 4.0
        elif "sampling_rate_hz" in data.files:
            sample_rate = float(np.asarray(data["sampling_rate_hz"]).reshape(-1)[0])
        else:
            sample_rate = 4.0

    xyz = np.ascontiguousarray(xyz[valid])
    activity = np.ascontiguousarray(activity[valid])
    labels = labels[valid]
    supervised = supervised[valid]
    finite = np.isfinite(activity)
    if not finite.all():
        median = np.nanmedian(np.where(finite, activity, np.nan), axis=1)
        activity = np.where(finite, activity, np.nan_to_num(median)[:, None])
    if cutoff_hz > 0:
        nyquist = 0.5 * sample_rate
        if not cutoff_hz < nyquist:
            raise ValueError(f"{path}: cutoff={cutoff_hz} is not below Nyquist={nyquist}")
        sos = butter(1, cutoff_hz / nyquist, btype="highpass", output="sos")
        activity = sosfiltfilt(sos, activity, axis=1)
    return Worm(path, uid, xyz, np.ascontiguousarray(activity), labels, supervised, sample_rate)


def unique_map(worm: Worm) -> dict[str, int]:
    counts = Counter(
        str(label).strip() for label, keep in zip(worm.labels, worm.supervised) if keep
    )
    return {
        str(label).strip(): index
        for index, (label, keep) in enumerate(zip(worm.labels, worm.supervised))
        if keep and counts[str(label).strip()] == 1
    }


def split_paths(fold: int, split: str) -> list[Path]:
    paths = sorted((DATA_ROOT / f"fold_{fold}" / split).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"Empty split: fold{fold}/{split}")
    return paths


def uid_for(path: Path) -> str:
    with np.load(path, allow_pickle=False) as data:
        if "recording_uid" in data.files:
            return str(np.asarray(data["recording_uid"]).reshape(-1)[0])
    parts = path.stem.split("__")
    return parts[1] if len(parts) >= 3 else path.stem


def frozen_medoid(fold: int) -> tuple[str, Path]:
    summary = read_json(MEDOID_RUN / "summary.json")
    uid = str(summary["fold_results"][fold]["medoid"]["uid"])
    matches = [path for path in split_paths(fold, "train") if uid_for(path) == uid]
    if len(matches) != 1:
        raise RuntimeError(f"fold{fold}: frozen medoid {uid!r} has matches {matches}")
    return uid, matches[0]


def canonical_queries(fold: int) -> dict[str, list[tuple[int, str]]]:
    output: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for row in read_csv(CANONICAL):
        if row["dataset"] == "rld" and int(row["fold"]) == fold:
            output[row["uid"]].append((int(row["node_index"]), row["identity"]))
    if not output:
        raise RuntimeError(f"fold{fold}: no canonical RLD queries in {CANONICAL}")
    for rows in output.values():
        rows.sort()
    return dict(output)


def validation_queries(fold: int, worms: Iterable[Worm]) -> dict[str, list[tuple[int, str]]]:
    train_union: set[str] = set()
    for path in split_paths(fold, "train"):
        train_union.update(unique_map(load_worm(path, 0.01)))
    output = {}
    for worm in worms:
        output[worm.uid] = sorted(
            (index, identity)
            for identity, index in unique_map(worm).items()
            if identity in train_union
        )
    return output


def fit_geometry_scaler(train: Iterable[Worm]) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.concatenate([worm.xyz for worm in train], axis=0)
    mean, std = xyz.mean(axis=0), xyz.std(axis=0)
    return mean, np.where(std > 1e-8, std, 1.0)


def normalize_cost(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float64)
    maximum = float(np.max(cost))
    return cost / maximum if np.isfinite(maximum) and maximum > 1e-12 else cost


def activity_cost(activity: np.ndarray) -> np.ndarray:
    # RMS trace distance is independent of recording length.  It is computed
    # only within a recording, where time samples are synchronized.
    length = activity.shape[1]
    norms = np.einsum("it,it->i", activity, activity) / length
    squared = norms[:, None] + norms[None, :] - 2.0 * (activity @ activity.T) / length
    cost = np.sqrt(np.maximum(squared, 0.0))
    np.fill_diagonal(cost, 0.0)
    return normalize_cost(cost)


def fgw_plan(
    query: Worm,
    reference: Worm,
    mean: np.ndarray,
    std: np.ndarray,
    alpha_activity: float,
) -> np.ndarray:
    query_xyz = (query.xyz - mean) / std
    reference_xyz = (reference.xyz - mean) / std
    geometry = normalize_cost(cdist(query_xyz, reference_xyz, metric="sqeuclidean"))
    query_activity = activity_cost(query.activity)
    reference_activity = activity_cost(reference.activity)
    plan = ot.gromov.fused_gromov_wasserstein(
        geometry,
        query_activity,
        reference_activity,
        p=ot.unif(len(query.xyz)),
        q=ot.unif(len(reference.xyz)),
        loss_fun="square_loss",
        alpha=float(alpha_activity),
        armijo=False,
        symmetric=True,
        log=False,
        max_iter=10000,
        tol_rel=1e-9,
        tol_abs=1e-9,
    )
    plan = np.asarray(plan, dtype=np.float64)
    if plan.shape != (len(query.xyz), len(reference.xyz)) or not np.isfinite(plan).all():
        raise RuntimeError(f"Invalid FGW plan {plan.shape}")
    return plan


def evaluate(
    fold: int,
    condition: str,
    paths: Iterable[Path],
    wanted_by_uid: dict[str, list[tuple[int, str]]],
    reference: Worm,
    mean: np.ndarray,
    std: np.ndarray,
    alpha: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reference_map = unique_map(reference)
    inverse_reference = {index: identity for identity, index in reference_map.items()}
    rows: list[dict[str, Any]] = []
    worms_seen = 0
    for path in paths:
        worm = load_worm(path, 0.01)
        wanted = wanted_by_uid.get(worm.uid, [])
        if not wanted:
            continue
        worms_seen += 1
        query_map = unique_map(worm)
        for index, identity in wanted:
            if query_map.get(identity) != index:
                raise RuntimeError(
                    f"Canonical query unavailable: fold{fold}/{condition}/{worm.uid}/{index}/{identity}"
                )
        plan = fgw_plan(worm, reference, mean, std, alpha)
        assignment_rows, assignment_cols = linear_sum_assignment(-plan)
        assignment = {int(row): int(col) for row, col in zip(assignment_rows, assignment_cols)}
        for query_index, identity in wanted:
            scores = plan[query_index]
            prediction = int(np.argmax(scores))
            target = reference_map.get(identity)
            if target is None:
                rank, top1, top5, rr, hungarian = -1, 0, 0, 0.0, 0
            else:
                order = np.argsort(-scores, kind="stable")
                hit = np.flatnonzero(order == target)
                if len(hit) != 1:
                    raise RuntimeError("FGW target ranking failure")
                rank = int(hit[0]) + 1
                top1, top5, rr = int(rank == 1), int(rank <= min(5, len(scores))), 1.0 / rank
                hungarian = int(assignment.get(query_index, -1) == target)
            rows.append({
                "dataset": "rld", "fold": fold, "seed": "deterministic",
                "method": "FGW (G+A)", "condition": condition,
                "query_uid": worm.uid, "query_index": query_index,
                "identity": identity, "reference_uid": reference.uid,
                "reference_covered": int(target is not None), "rank": rank,
                "top1": top1, "top5": top5, "rr": rr,
                "hungarian_correct": hungarian, "predicted_index": prediction,
                "predicted_identity": inverse_reference.get(prediction, ""),
                "alpha_activity": alpha,
            })
    expected = sum(len(rows) for uid, rows in wanted_by_uid.items() if uid in {
        uid_for(path) for path in paths
    })
    if len(rows) != expected:
        raise RuntimeError(f"fold{fold}/{condition}: rows={len(rows)}, expected={expected}")
    q = len(rows)
    if q == 0:
        raise RuntimeError(f"fold{fold}/{condition}: no canonical queries")
    metrics = {
        "queries": q,
        "worms": worms_seen,
        "covered_queries": sum(int(row["reference_covered"]) for row in rows),
        "top1_correct": sum(int(row["top1"]) for row in rows),
        "top5_correct": sum(int(row["top5"]) for row in rows),
        "rr_sum": float(sum(float(row["rr"]) for row in rows)),
        "hungarian_correct": sum(int(row["hungarian_correct"]) for row in rows),
    }
    for key, numerator in (
        ("top1", "top1_correct"), ("top5", "top5_correct"),
        ("mrr", "rr_sum"), ("hungarian", "hungarian_correct"),
    ):
        metrics[key] = float(metrics[numerator] / q)
    metrics["coverage"] = 1.0
    metrics["effective_top1"] = metrics["top1"]
    covered = int(metrics["covered_queries"])
    metrics["reference_coverage"] = float(covered / q)
    metrics["covered_top1"] = float(metrics["top1_correct"] / max(covered, 1))
    metrics["covered_top5"] = float(metrics["top5_correct"] / max(covered, 1))
    metrics["covered_mrr"] = float(metrics["rr_sum"] / max(covered, 1))
    metrics["covered_hungarian"] = float(
        metrics["hungarian_correct"] / max(covered, 1)
    )
    return metrics, rows


def select_alpha(
    fold: int,
    val: list[Worm],
    queries: dict[str, list[tuple[int, str]]],
    reference: Worm,
    mean: np.ndarray,
    std: np.ndarray,
    alphas: list[float],
) -> tuple[float, list[dict[str, Any]]]:
    sweep = []
    paths = [worm.path for worm in val]
    for alpha in alphas:
        metrics, _ = evaluate(
            fold, "outer_validation", paths, queries, reference, mean, std, alpha
        )
        sweep.append({"alpha_activity": alpha, **metrics})
        print(
            f"[VAL] fold{fold} alpha_activity={alpha:.2f} "
            f"Top1={100 * metrics['top1']:.2f}% "
            f"Hung={100 * metrics['hungarian']:.2f}%",
            flush=True,
        )
    # Exact selection convention from the previous FGW runner.
    selected = max(sweep, key=lambda row: (row["hungarian"], row["top1"]))
    return float(selected["alpha_activity"]), sweep


def corruption_conditions(fold: int) -> list[dict[str, Any]]:
    manifest = read_json(CORRUPTIONS)
    rows = [
        row for row in manifest["conditions"]
        if int(row["fold"]) == fold and row["kind"] == "activity_noise"
    ]
    rows.sort(key=lambda row: (float(row["severity"]), int(row["perturbation_seed"])))
    expected = 1 + 5 * 3
    if len(rows) != expected:
        raise RuntimeError(f"fold{fold}: expected {expected} activity conditions, got {len(rows)}")
    return rows


def condition_queries(
    spec: dict[str, Any], canonical: dict[str, list[tuple[int, str]]]
) -> tuple[list[Path], dict[str, list[tuple[int, str]]]]:
    paths = []
    queries: dict[str, list[tuple[int, str]]] = {}
    for record in spec["files"]:
        path = Path(record["output"])
        paths.append(path)
        uid = str(record["recording_uid"])
        # New manifests store this projection explicitly.  Older materialized
        # manifests are reconstructed from the same kept-source-index contract.
        if "canonical_query_rows_after_corruption" in record:
            rows = [
                (int(item["row"]), str(item["cell_id"]))
                for item in record["canonical_query_rows_after_corruption"]
            ]
        else:
            source_to_current = {
                int(source): current
                for current, source in enumerate(record["kept_source_indices"])
            }
            rows = [
                (source_to_current[source], identity)
                for source, identity in canonical.get(uid, [])
                if source in source_to_current
            ]
        queries[uid] = sorted(rows)
    return paths, queries


def exact_clean_gate(main: dict[str, Any], replay: dict[str, Any]) -> None:
    integer = ("queries", "top1_correct", "top5_correct", "hungarian_correct")
    floating = ("rr_sum", "top1", "top5", "mrr", "hungarian")
    errors = {}
    for key in integer:
        if int(main[key]) != int(replay[key]):
            errors[key] = [main[key], replay[key]]
    for key in floating:
        if not np.isclose(float(main[key]), float(replay[key]), rtol=0.0, atol=1e-12):
            errors[key] = [main[key], replay[key]]
    if errors:
        raise RuntimeError(f"FGW (G+A) severity-zero clean gate failed: {errors}")


def summarize(cell_rows: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    fold_rows = []
    for fold in range(5):
        for severity in sorted({float(row["severity"]) for row in cell_rows}):
            cells = [
                row for row in cell_rows
                if int(row["fold"]) == fold and float(row["severity"]) == severity
            ]
            item = {
                "method": "FGW (G+A)", "kind": "activity_noise",
                "severity": severity, "fold": fold, "corruption_draws": len(cells),
            }
            for metric in ("queries", "top1", "top5", "mrr", "hungarian", "coverage", "effective_top1"):
                item[metric] = float(np.mean([float(row[metric]) for row in cells]))
            # Native single-reference accuracy used by the pre-existing
            # activity-noise table.  The complete-canonical metrics above are
            # retained as the stricter cross-method estimand.
            item["reference_coverage"] = float(np.mean([
                float(row["covered_queries"]) / float(row["queries"]) for row in cells
            ]))
            for metric, numerator in (
                ("covered_top1", "top1_correct"),
                ("covered_top5", "top5_correct"),
                ("covered_mrr", "rr_sum"),
                ("covered_hungarian", "hungarian_correct"),
            ):
                item[metric] = float(np.mean([
                    float(row[numerator]) / max(float(row["covered_queries"]), 1.0)
                    for row in cells
                ]))
            fold_rows.append(item)
    summary = []
    for severity in sorted({float(row["severity"]) for row in fold_rows}):
        cells = [row for row in fold_rows if float(row["severity"]) == severity]
        item = {"method": "FGW (G+A)", "kind": "activity_noise", "severity": severity, "folds": 5}
        for metric in (
            "top1", "top5", "mrr", "hungarian", "coverage", "effective_top1",
            "reference_coverage", "covered_top1", "covered_top5", "covered_mrr",
            "covered_hungarian",
        ):
            values = np.asarray([float(row[metric]) for row in cells])
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_sd"] = float(values.std(ddof=1))
        summary.append(item)
    write_csv(output / "fold_level.csv", fold_rows)
    write_csv(output / "summary.csv", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--alphas", default="0.25,0.50,0.75")
    parser.add_argument("--reuse-lock", action="store_true")
    args = parser.parse_args()
    folds = [int(value) for value in args.folds.split(",")]
    alphas = [float(value) for value in args.alphas.split(",")]
    if any(not 0.0 < alpha < 1.0 for alpha in alphas):
        raise ValueError("G+A alpha candidates must lie strictly between zero and one")

    all_cells = []
    for fold in folds:
        print(f"\n{'=' * 100}\nFGW (G+A) RLD FOLD {fold}\n{'=' * 100}", flush=True)
        main_dir = MAIN_OUT / f"fold{fold}"
        robust_dir = ROBUST_OUT / f"fold{fold}"
        lock_path = main_dir / "LOCKED_BEFORE_TEST.json"
        medoid_uid, medoid_path = frozen_medoid(fold)
        train = [load_worm(path, 0.01) for path in split_paths(fold, "train")]
        val = [load_worm(path, 0.01) for path in split_paths(fold, "val")]
        mean, std = fit_geometry_scaler(train)
        reference = load_worm(medoid_path, 0.01)
        val_queries = validation_queries(fold, val)

        if args.reuse_lock:
            lock = read_json(lock_path)
            if lock["status"] != "LOCKED_BEFORE_TEST" or int(lock["fold"]) != fold:
                raise RuntimeError(f"Invalid lock: {lock_path}")
            alpha = float(lock["selected_alpha_activity"])
            if not np.allclose(mean, np.asarray(lock["train_xyz_mean"]), atol=0, rtol=0):
                raise RuntimeError(f"fold{fold}: train mean changed")
            if not np.allclose(std, np.asarray(lock["train_xyz_std"]), atol=0, rtol=0):
                raise RuntimeError(f"fold{fold}: train std changed")
        else:
            alpha, sweep = select_alpha(fold, val, val_queries, reference, mean, std, alphas)
            lock = {
                "status": "LOCKED_BEFORE_TEST", "method": "FGW (G+A)",
                "dataset": "rld", "fold": fold, "deterministic": True,
                "reference_uid": medoid_uid, "reference_path": str(medoid_path.resolve()),
                "reference_selection": "shared label-free outer-training geometry medoid",
                "train_xyz_mean": mean.tolist(), "train_xyz_std": std.tolist(),
                "alpha_grid": alphas, "selected_alpha_activity": alpha,
                "selected_alpha_geometry": 1.0 - alpha,
                "selection_split": "outer validation only",
                "selection_metric": "validation Hungarian then Top1",
                "validation_sweep": sweep,
                "feature_cost_M": "normalized squared Euclidean train-zscored xyz",
                "structure_cost_C": "normalized within-recording RMS distance of 0.01-Hz high-pass activity",
                "pot_alpha_semantics": "alpha*activity_GW + (1-alpha)*geometry_linear",
                "test_opened_before_lock": False, "pot_version": str(ot.__version__),
            }
            write_json(lock_path, lock)

        # First test access: independently establish the clean main-benchmark cell.
        canonical = canonical_queries(fold)
        clean_metrics, clean_rows = evaluate(
            fold, "clean_main", split_paths(fold, "test"), canonical,
            reference, mean, std, alpha,
        )
        write_csv(main_dir / "query_level.csv", clean_rows, QUERY_FIELDS)
        clean_report = {
            **clean_metrics, "method": "FGW (G+A)", "dataset": "rld", "fold": fold,
            "model_seed": None, "deterministic": True,
            "query_protocol": "canonical_all; medoid-reference misses retained as errors",
            "reference_uid": medoid_uid, "selected_alpha_activity": alpha,
            "lock": str(lock_path.resolve()), "lock_sha256": sha256(lock_path),
        }
        write_json(main_dir / "test_metrics.json", clean_report)

        for spec in corruption_conditions(fold):
            severity = float(spec["severity"])
            draw = int(spec["perturbation_seed"])
            condition = Path(spec["root"]).name
            paths, queries = condition_queries(spec, canonical)
            metrics, rows = evaluate(
                fold, condition, paths, queries, reference, mean, std, alpha
            )
            if severity == 0.0:
                exact_clean_gate(clean_metrics, metrics)
            destination = robust_dir / condition
            write_csv(destination / "query_level.csv", rows, QUERY_FIELDS)
            report = {
                **metrics, "method": "FGW (G+A)", "dataset": "rld", "fold": fold,
                "kind": "activity_noise", "severity": severity,
                "perturbation_seed": draw, "condition": condition,
                "selected_alpha_activity": alpha, "reference_uid": medoid_uid,
                "clean_main_metrics": str((main_dir / "test_metrics.json").resolve()),
                "severity_zero_exact_main_replay": severity == 0.0,
            }
            write_json(destination / "metrics.json", report)
            all_cells.append(report)
            print(
                f"[TEST] fold{fold} {condition} Top1={100 * metrics['top1']:.2f}% "
                f"Hung={100 * metrics['hungarian']:.2f}%",
                flush=True,
            )

    if folds == list(range(5)):
        summary = summarize(all_cells, ROBUST_OUT)
        main_cells = [read_json(MAIN_OUT / f"fold{fold}/test_metrics.json") for fold in range(5)]
        main_summary = {"method": "FGW (G+A)", "dataset": "rld", "folds": 5}
        for metric in ("top1", "top5", "mrr", "hungarian"):
            values = np.asarray([float(row[metric]) for row in main_cells])
            main_summary[f"{metric}_mean"] = float(values.mean())
            main_summary[f"{metric}_sd"] = float(values.std(ddof=1))
        for metric, numerator in (
            ("covered_top1", "top1_correct"),
            ("covered_top5", "top5_correct"),
            ("covered_mrr", "rr_sum"),
            ("covered_hungarian", "hungarian_correct"),
        ):
            values = np.asarray([
                float(row[numerator]) / max(float(row["covered_queries"]), 1.0)
                for row in main_cells
            ])
            main_summary[f"{metric}_mean"] = float(values.mean())
            main_summary[f"{metric}_sd"] = float(values.std(ddof=1))
        write_json(MAIN_OUT / "summary.json", main_summary)
        write_json(ROBUST_OUT / "AUDIT.json", {
            "status": "passed", "method": "FGW (G+A)",
            "protocol": "RLD grouped CV5; deterministic; shared medoid and canonical query cohort",
            "activity_conditions": len(all_cells), "folds": 5,
            "severity_zero_exact_main_replay": True,
            "aggregation": "unweighted corruption-draw mean within fold; unweighted mean and sample SD over folds",
            "source": str(Path(__file__).resolve()), "source_sha256": sha256(Path(__file__)),
            "main_summary": main_summary, "robustness_summary": summary,
        })
        print(json.dumps({"main": main_summary, "activity": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
