#!/usr/bin/env python3
"""
Strict RLD controlled-corruption evaluation for the FINAL fDNC benchmark.

This replays the exact final clean protocol:
  * current RLD biological fold k (0..4) maps to official benchmark fold k+1;
  * use the already validation-selected fDNC seed42 checkpoint for that fold;
  * use the already locked OUTER-TRAIN geometry-medoid template recorded by the
    final benchmark;
  * do NOT retrain, retune, reselect a checkpoint, or reselect a template;
  * only the outer-test worms are replaced by materialized corrupted variants;
  * fDNC preprocessing and pair scoring are delegated to the same repository
    modules used by evaluate_train_reference_ensemble.py:
        baselines.atanas_locked.normalize_xyz
        engines.evaluate_atanas_fdnc_unified.load_checkpoint
        engines.train_atanas_fdnc_fold.score_pair

Metric semantics match the official pairwise fDNC evaluator:
  * a GT query is counted only when its identity occurs uniquely in both the
    query worm and the fixed template;
  * ranking is against ALL template neuron columns, including unlabeled neurons;
  * Hungarian is solved on the FULL query x template score matrix and evaluated
    only on the shared unique GT identities;
  * inserted synthetic distractor rows participate in fDNC context and Hungarian
    competition but are never GT queries.

Before any non-zero corruption for a biological fold, severity=0 must reproduce
the saved final clean fDNC result for seed42.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
CORR_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v1/corruptions"
OUT_ROOT = ROOT / "runs/rld_robustness_cv5_seed42_v1/results/fdnc_medoid_seed42_strict_v1"

FINAL_RECORD_ROOT = (
    ROOT
    / "experiments/mprt_population_relational_transport/"
      "medoid_template_cv5x3_20260825/records/fdnc/rld"
)

COND_RE = re.compile(
    r"^(coord_noise|missing|outlier)_l([0-9]+(?:\.[0-9]+)?)_p([0-9]+)$"
)

EXPECTED_OFFICIAL_REPO_COMMIT = "cbd2b1b2e623c22fe1921562afed1967fe58e139"
EXPECTED_RELEASED_PRETRAINED_SHA256 = (
    "ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9"
)

INVALID_LABELS = {
    "", "nan", "none", "null", "-1", "unknown", "unk", "?", "unlabeled",
    "unlabelled", "invalid",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"{path}: expected JSON object")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def norm_label(value: Any) -> Optional[str]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    if text.lower() in INVALID_LABELS:
        return None
    if text.startswith("__OUTLIER"):
        return None
    return text


def recording_uid(path: Path) -> str:
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
                    text = str(value).strip()
                    if text:
                        return text
    stem = path.stem
    parts = stem.split("__")
    if len(parts) >= 3:
        return parts[1]
    return stem


@dataclass
class Worm:
    path: Path
    uid: str
    xyz: np.ndarray
    labels: list[Optional[str]]
    mask_key: str

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])


def choose_mask(z: Any, n: int) -> tuple[np.ndarray, str]:
    """
    Prefer the exact clean-mask semantics used by the final fDNC benchmark.
    Fallbacks are accepted only so the script can audit materialized robustness
    files; severity=0 clean guard will reject any semantically wrong fallback.
    """
    for key in ("clean_mask", "labeled_mask", "certain_mask", "supervised_mask"):
        if key in z.files:
            mask = np.asarray(z[key], dtype=bool).reshape(-1)
            if len(mask) != n:
                raise ValueError(f"{key}: length {len(mask)} != xyz rows {n}")
            return mask, key
    # Last resort: all rows with a syntactically valid identity.
    return np.ones(n, dtype=bool), "<label-validity-fallback>"


def load_worm(path: Path) -> Worm:
    with np.load(path, allow_pickle=True) as z:
        if "xyz" not in z.files or "cell_id" not in z.files:
            raise KeyError(f"{path}: requires xyz and cell_id; keys={list(z.files)}")
        xyz = np.asarray(z["xyz"], dtype=np.float32)
        raw_labels = np.asarray(z["cell_id"]).reshape(-1)
        if xyz.ndim != 2 or xyz.shape[1] < 3:
            raise ValueError(f"{path}: xyz must be [N,>=3], got {xyz.shape}")
        xyz = xyz[:, :3]
        if len(raw_labels) != len(xyz):
            raise ValueError(
                f"{path}: xyz rows={len(xyz)} labels={len(raw_labels)}"
            )
        mask, mask_key = choose_mask(z, len(xyz))

    labels: list[Optional[str]] = []
    for value, keep in zip(raw_labels, mask):
        label = norm_label(value) if bool(keep) else None
        labels.append(label)

    if not np.isfinite(xyz).all():
        raise ValueError(f"{path}: xyz contains NaN/Inf")

    return Worm(
        path=path.resolve(),
        uid=recording_uid(path),
        xyz=xyz,
        labels=labels,
        mask_key=mask_key,
    )


def unique_identity_map(labels: list[Optional[str]]) -> dict[str, int]:
    positions: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        if label is not None:
            positions.setdefault(label, []).append(i)
    return {
        label: indices[0]
        for label, indices in positions.items()
        if len(indices) == 1
    }


def import_final_fdnc_modules():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from baselines.atanas_locked import normalize_xyz
    from engines import evaluate_atanas_fdnc_unified as fdnc_loader
    from scripts.lib import fdnc_scoring

    return normalize_xyz, fdnc_loader, fdnc_scoring


def official_fold(robust_fold: int) -> int:
    if robust_fold not in range(5):
        raise ValueError(f"robust fold must be 0..4, got {robust_fold}")
    return robust_fold + 1


def final_record_path(robust_fold: int) -> Path:
    f = official_fold(robust_fold)
    return (
        FINAL_RECORD_ROOT
        / f"fold_{f}/seed_42/outer_test_medoid_template_v1/result.json"
    )


def load_locked_context(robust_fold: int, device: torch.device):
    normalize_xyz, fdnc_loader, fdnc_train = import_final_fdnc_modules()

    result_path = final_record_path(robust_fold)
    saved = read_json(result_path)

    f = official_fold(robust_fold)
    if int(saved.get("fold", -1)) != f:
        raise RuntimeError(
            f"{result_path}: fold={saved.get('fold')} expected {f}"
        )
    if int(saved.get("seed", -1)) != 42:
        raise RuntimeError(f"{result_path}: not seed42")
    if saved.get("evaluation_protocol") != "outer_training_geometry_medoid_template_v1":
        raise RuntimeError(
            f"{result_path}: wrong evaluation protocol "
            f"{saved.get('evaluation_protocol')!r}"
        )
    if saved.get("selected_on") != "validation only":
        raise RuntimeError(f"{result_path}: checkpoint was not validation-selected")

    selection = saved.get("template_selection", {})
    if selection.get("selection_split") != "outer_train_only":
        raise RuntimeError(f"{result_path}: template not selected on outer train only")
    if bool(selection.get("uses_validation", True)):
        raise RuntimeError(f"{result_path}: template selection used validation")
    if bool(selection.get("uses_test", True)):
        raise RuntimeError(f"{result_path}: template selection used test")
    if bool(selection.get("uses_identity_labels", True)):
        raise RuntimeError(f"{result_path}: template selection used identity labels")

    checkpoint = Path(saved["checkpoint"]).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_digest = sha256(checkpoint)
    expected_digest = str(saved["checkpoint_sha256"])
    if checkpoint_digest != expected_digest:
        raise RuntimeError(
            f"Checkpoint SHA256 mismatch:\n"
            f"  actual   {checkpoint_digest}\n"
            f"  expected {expected_digest}\n"
            f"  file     {checkpoint}"
        )

    template_path = Path(selection["template_path"]).expanduser()
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    template = load_worm(template_path)
    expected_uid = str(selection["template_worm"])
    if template.uid != expected_uid:
        raise RuntimeError(
            f"Template UID mismatch: loaded={template.uid} saved={expected_uid}"
        )

    metadata = saved.get("training_metadata", {})
    repo_commit = str(metadata.get("official_repo_commit", ""))
    if repo_commit and repo_commit != EXPECTED_OFFICIAL_REPO_COMMIT:
        raise RuntimeError(
            f"Unexpected official fDNC training repo commit: {repo_commit}"
        )
    released_sha = str(metadata.get("official_pretrained_sha256", ""))
    if released_sha and released_sha != EXPECTED_RELEASED_PRETRAINED_SHA256:
        raise RuntimeError(
            f"Unexpected released initialization SHA256: {released_sha}"
        )
    if bool(metadata.get("outer_test_accessed_during_training", True)):
        raise RuntimeError(f"{result_path}: outer test was accessed during training")

    model = fdnc_loader.load_checkpoint(checkpoint, device, 128, 6)
    model.eval()

    print(f"[LOCK] robust fold{robust_fold} -> official fold_{f}", flush=True)
    print(f"[LOCK] clean result = {result_path}", flush=True)
    print(f"[LOCK] checkpoint   = {checkpoint}", flush=True)
    print(f"[LOCK] checkpoint sha256 = {checkpoint_digest}", flush=True)
    print(f"[LOCK] template     = {template.uid}", flush=True)
    print(f"[LOCK] template path= {template.path}", flush=True)
    print(f"[LOCK] template mask= {template.mask_key}", flush=True)

    return {
        "official_fold": f,
        "saved": saved,
        "saved_path": result_path,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint_digest,
        "template": template,
        "model": model,
        "normalize_xyz": normalize_xyz,
        "fdnc_train": fdnc_train,
    }


@torch.inference_mode()
def score_pair_exact(ctx: dict[str, Any], query: Worm, device: torch.device) -> np.ndarray:
    """
    Exact scorer used by evaluate_train_reference_ensemble.py for fDNC:
        normalize_xyz per worm
        score_pair(model, query_xyz, reference_xyz)
        second returned direction = query -> reference
        drop fDNC outlier column
    """
    normalize_xyz = ctx["normalize_xyz"]
    fdnc_train = ctx["fdnc_train"]
    template: Worm = ctx["template"]
    model = ctx["model"]

    query_xyz = torch.from_numpy(normalize_xyz(query.xyz)).to(
        device=device, dtype=torch.float32
    )
    reference_xyz = torch.from_numpy(normalize_xyz(template.xyz)).to(
        device=device, dtype=torch.float32
    )

    _, query_to_reference = fdnc_train.score_pair(
        model, query_xyz, reference_xyz
    )
    scores = query_to_reference[:, :-1].float().cpu().numpy()
    scores = np.asarray(scores, dtype=np.float64)

    expected = (query.n, template.n)
    if scores.shape != expected:
        raise RuntimeError(
            f"fDNC score shape {scores.shape} != expected {expected}"
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("fDNC produced NaN/Inf scores")
    return scores


def evaluate_pair(scores: np.ndarray, query: Worm, template: Worm) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Metric semantics copied from the official fDNC pair evaluator:
      rank against every template neuron;
      Hungarian on the full matrix;
      score GT only for identities unique in both worms.
    """
    qmap = unique_identity_map(query.labels)
    rmap = unique_identity_map(template.labels)
    common = sorted(set(qmap) & set(rmap))

    ranks: list[int] = []
    rows: list[dict[str, Any]] = []

    for identity in common:
        qi, ri = qmap[identity], rmap[identity]
        row = scores[qi]
        rank = 1 + int(np.sum(row > row[ri]))
        ranks.append(rank)
        rows.append({
            "query_uid": f"{query.uid}::{qi}",
            "group_uid": query.uid,
            "gt_label": identity,
            "query_row": int(qi),
            "template_row": int(ri),
            "rank": int(rank),
            "gt_score": float(row[ri]),
        })

    hits = 0
    assignment: dict[int, int] = {}
    if scores.size:
        ar, ac = linear_sum_assignment(-scores)
        assignment = {int(r): int(c) for r, c in zip(ar, ac)}

    for row in rows:
        qi = int(row["query_row"])
        pred = assignment.get(qi, -1)
        ok = (
            pred >= 0
            and pred < len(template.labels)
            and template.labels[pred] == row["gt_label"]
        )
        row["hungarian_correct"] = int(ok)
        row["hungarian_template_row"] = int(pred)
        hits += int(ok)

    arr = np.asarray(ranks, dtype=np.float64)
    result = {
        "queries": int(len(arr)),
        "ranking_top1": float(np.mean(arr <= 1)) if len(arr) else math.nan,
        "top3": float(np.mean(arr <= 3)) if len(arr) else math.nan,
        "top5": float(np.mean(arr <= 5)) if len(arr) else math.nan,
        "top10": float(np.mean(arr <= 10)) if len(arr) else math.nan,
        "mrr": float(np.mean(1.0 / arr)) if len(arr) else math.nan,
        "assignment_top1": float(hits / len(arr)) if len(arr) else math.nan,
        "correct_top1": int(np.sum(arr <= 1)) if len(arr) else 0,
        "correct_top5": int(np.sum(arr <= 5)) if len(arr) else 0,
        "hungarian_hits": int(hits),
    }
    return result, rows


def aggregate_pair_results(pair_results: list[dict[str, Any]]) -> dict[str, Any]:
    q = sum(int(x["queries"]) for x in pair_results)
    if q <= 0:
        return {
            "queries": 0,
            "ranking_top1": math.nan,
            "top3": math.nan,
            "top5": math.nan,
            "top10": math.nan,
            "mrr": math.nan,
            "assignment_top1": math.nan,
            "correct_top1": 0,
            "correct_top5": 0,
            "hungarian_hits": 0,
        }

    def weighted(key: str) -> float:
        return float(
            sum(float(x[key]) * int(x["queries"]) for x in pair_results)
            / q
        )

    return {
        "queries": q,
        "ranking_top1": weighted("ranking_top1"),
        "top3": weighted("top3"),
        "top5": weighted("top5"),
        "top10": weighted("top10"),
        "mrr": weighted("mrr"),
        "assignment_top1": weighted("assignment_top1"),
        "correct_top1": sum(int(x["correct_top1"]) for x in pair_results),
        "correct_top5": sum(int(x["correct_top5"]) for x in pair_results),
        "hungarian_hits": sum(int(x["hungarian_hits"]) for x in pair_results),
    }


def test_files(root: Path) -> list[Path]:
    files = sorted((root / "test").glob("*.npz"))
    if not files:
        files = sorted((root / "test").rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No test NPZs under {root / 'test'}")
    return files


def condition_rows(manifest: dict[str, Any], fold: int, kinds: set[str]) -> list[dict[str, Any]]:
    rows = [
        x
        for x in manifest["conditions"]
        if int(x["fold"]) == fold and str(x["kind"]) in kinds
    ]
    order = {"coord_noise": 0, "missing": 1, "outlier": 2}
    rows.sort(
        key=lambda x: (
            order[str(x["kind"])],
            float(x["severity"]),
            int(x["perturbation_seed"]),
        )
    )
    return rows


def clean_row_for_fold(manifest: dict[str, Any], fold: int) -> dict[str, Any]:
    candidates = [
        x
        for x in manifest["conditions"]
        if int(x["fold"]) == fold
        and str(x["kind"]) == "coord_noise"
        and abs(float(x["severity"])) <= 1e-15
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one fold{fold} coord_noise severity=0 condition; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def expected_test_uids_from_saved(saved: dict[str, Any]) -> set[str]:
    path = Path(saved["test_list"])
    if not path.is_file():
        raise FileNotFoundError(path)
    uids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        uids.add(recording_uid(Path(text)))
    return uids


def validate_clean_test_population(ctx: dict[str, Any], clean_root: Path) -> None:
    observed_files = test_files(clean_root)
    observed = {recording_uid(p) for p in observed_files}
    expected = expected_test_uids_from_saved(ctx["saved"])
    if observed != expected:
        raise RuntimeError(
            "Robustness severity=0 test population does not match the final "
            "fDNC outer-test list.\n"
            f"missing={sorted(expected - observed)}\n"
            f"extra={sorted(observed - expected)}"
        )
    print(
        f"[AUDIT] clean test UID set EXACT: {len(observed)} worms",
        flush=True,
    )


def canonical_unique_gt_count(clean_root: Path) -> int:
    total = 0
    for path in test_files(clean_root):
        worm = load_worm(path)
        total += len(unique_identity_map(worm.labels))
    return total


def evaluate_condition(
    ctx: dict[str, Any],
    robust_fold: int,
    row: dict[str, Any],
    device: torch.device,
    clean_shared_queries: int,
    canonical_clean_queries: int,
) -> dict[str, Any]:
    root = Path(row["root"])
    worms = [load_worm(p) for p in test_files(root)]

    pair_results: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    mask_keys = sorted({w.mask_key for w in worms})

    for i, worm in enumerate(worms, start=1):
        scores = score_pair_exact(ctx, worm, device)
        metrics, rows = evaluate_pair(scores, worm, ctx["template"])
        pair_results.append(metrics)
        query_rows.extend(rows)
        print(
            f"[fDNC] fold{robust_fold} "
            f"{row['kind']} l={float(row['severity']):.2f} "
            f"p={int(row['perturbation_seed'])} "
            f"worm={i:02d}/{len(worms):02d} {worm.uid} "
            f"Q={metrics['queries']} "
            f"Top1={100*metrics['ranking_top1']:.2f}%"
            if metrics["queries"] else
            f"[fDNC] fold{robust_fold} worm={i:02d}/{len(worms):02d} "
            f"{worm.uid} Q=0",
            flush=True,
        )

    metrics = aggregate_pair_results(pair_results)

    # Two coverage notions are intentionally reported:
    # 1) relative to fDNC's clean shared/template-covered query universe;
    # 2) relative to every unique clean GT query in the clean test worms.
    metrics["clean_shared_queries"] = int(clean_shared_queries)
    metrics["canonical_clean_queries"] = int(canonical_clean_queries)
    metrics["coverage_vs_clean_shared"] = (
        float(metrics["queries"] / clean_shared_queries)
        if clean_shared_queries > 0 else math.nan
    )
    metrics["candidate_coverage_vs_canonical"] = (
        float(metrics["queries"] / canonical_clean_queries)
        if canonical_clean_queries > 0 else math.nan
    )
    metrics["effective_top1_vs_clean_shared"] = (
        float(metrics["correct_top1"] / clean_shared_queries)
        if clean_shared_queries > 0 else math.nan
    )
    metrics["effective_top1_vs_canonical"] = (
        float(metrics["correct_top1"] / canonical_clean_queries)
        if canonical_clean_queries > 0 else math.nan
    )

    kind = str(row["kind"])
    severity = float(row["severity"])
    pseed = int(row["perturbation_seed"])
    cond = f"{kind}_l{severity:.2f}_p{pseed}"
    dst = OUT_ROOT / f"fold{robust_fold}" / cond
    dst.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(query_rows).to_csv(dst / "query_level.csv", index=False)

    result = {
        "method": "fDNC official fine-tuned — locked seed42",
        "robust_fold": robust_fold,
        "official_fold": ctx["official_fold"],
        "kind": kind,
        "severity": severity,
        "perturbation_seed": pseed,
        "checkpoint": str(ctx["checkpoint"]),
        "checkpoint_sha256": ctx["checkpoint_sha256"],
        "reference_worm": ctx["template"].uid,
        "reference_file": str(ctx["template"].path),
        "test_root": str(root.resolve()),
        "test_worms": len(worms),
        "mask_keys": mask_keys,
        "metrics": metrics,
        "protocol": {
            "model_seed": 42,
            "retrained_on_corruption": False,
            "checkpoint_reselected_on_corruption": False,
            "template_reselected_on_corruption": False,
            "reference_policy": "saved clean outer-train geometry medoid",
            "xyz_preprocessing": "same normalize_xyz as final medoid benchmark",
            "ranking_candidate_set": "all template neuron columns",
            "gt_query_policy": "identity unique in query and fixed template",
            "hungarian_input": "full query x template score matrix",
            "synthetic_outliers_are_input_context_but_not_gt": True,
        },
    }
    write_json(dst / "metrics.json", result)

    print(
        f"[RESULT] fold{robust_fold} {cond}: "
        f"Q={metrics['queries']}/{clean_shared_queries} "
        f"Top1={100*metrics['ranking_top1']:.2f}% "
        f"Top5={100*metrics['top5']:.2f}% "
        f"MRR={metrics['mrr']:.4f} "
        f"Hung={100*metrics['assignment_top1']:.2f}% "
        f"SharedCov={100*metrics['coverage_vs_clean_shared']:.2f}% "
        f"CanonicalCov={100*metrics['candidate_coverage_vs_canonical']:.2f}% "
        f"EffTop1Canon={100*metrics['effective_top1_vs_canonical']:.2f}%",
        flush=True,
    )
    return result


def clean_guard(result: dict[str, Any], ctx: dict[str, Any]) -> None:
    got = result["metrics"]
    saved = ctx["saved"]["metrics"]

    pairs = [
        ("queries", int(got["queries"]), int(saved["queries"]), 0.0),
        ("ranking_top1", float(got["ranking_top1"]), float(saved["ranking_top1"]), 1e-10),
        ("top3", float(got["top3"]), float(saved["top3"]), 1e-10),
        ("top5", float(got["top5"]), float(saved["top5"]), 1e-10),
        ("top10", float(got["top10"]), float(saved["top10"]), 1e-10),
        ("mrr", float(got["mrr"]), float(saved["mrr"]), 1e-10),
        ("assignment_top1", float(got["assignment_top1"]), float(saved["assignment_top1"]), 1e-10),
    ]

    bad = []
    for key, a, b, tol in pairs:
        diff = abs(float(a) - float(b))
        print(
            f"[CLEAN GUARD] {key:18s} reproduced={a} saved={b} diff={diff:.3e}",
            flush=True,
        )
        if diff > tol:
            bad.append((key, a, b, diff))

    if bad:
        raise RuntimeError(
            "fDNC severity=0 FAILED to reproduce the FINAL seed42 medoid "
            f"benchmark for official fold_{ctx['official_fold']}: {bad}"
        )

    print(
        f"[CLEAN GUARD EXACT] official fold_{ctx['official_fold']} "
        f"seed42 reproduced final medoid result",
        flush=True,
    )


def summarize(results: list[dict[str, Any]]) -> None:
    if not results:
        return

    rows = []
    for r in results:
        rows.append({
            "fold": r["robust_fold"],
            "official_fold": r["official_fold"],
            "kind": r["kind"],
            "severity": r["severity"],
            "perturbation_seed": r["perturbation_seed"],
            **r["metrics"],
        })
    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_ROOT / "fdnc_all_cells.csv", index=False)

    metric_cols = [
        "ranking_top1",
        "top5",
        "mrr",
        "assignment_top1",
        "coverage_vs_clean_shared",
        "candidate_coverage_vs_canonical",
        "effective_top1_vs_clean_shared",
        "effective_top1_vs_canonical",
    ]
    metric_cols = [x for x in metric_cols if x in df.columns]

    # perturbation seeds averaged within biological fold first
    fold_df = (
        df.groupby(["kind", "severity", "fold"], as_index=False)[metric_cols]
        .mean()
    )
    fold_df.to_csv(OUT_ROOT / "fdnc_fold_level.csv", index=False)

    summary_rows = []
    for (kind, severity), group in fold_df.groupby(["kind", "severity"]):
        row = {
            "kind": kind,
            "severity": float(severity),
            "biological_folds": int(group["fold"].nunique()),
        }
        for key in metric_cols:
            vals = group[key].astype(float).to_numpy()
            row[f"{key}_mean"] = float(np.mean(vals))
            row[f"{key}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values(["kind", "severity"])
    summary.to_csv(OUT_ROOT / "fdnc_macro_summary.csv", index=False)

    print("\n" + "=" * 108)
    print("fDNC FINAL-MEDOID ROBUSTNESS MACRO SUMMARY")
    print("=" * 108)
    for _, r in summary.iterrows():
        print(
            f"{r['kind']:12s} level={r['severity']:.2f} "
            f"Top1={100*r['ranking_top1_mean']:.2f}±{100*r['ranking_top1_sd']:.2f}% "
            f"Top5={100*r['top5_mean']:.2f}±{100*r['top5_sd']:.2f}% "
            f"Hung={100*r['assignment_top1_mean']:.2f}±{100*r['assignment_top1_sd']:.2f}% "
            f"SharedCov={100*r['coverage_vs_clean_shared_mean']:.2f}±{100*r['coverage_vs_clean_shared_sd']:.2f}% "
            f"CanonCov={100*r['candidate_coverage_vs_canonical_mean']:.2f}±"
            f"{100*r['candidate_coverage_vs_canonical_sd']:.2f}%",
            flush=True,
        )

    print(f"\nresults: {OUT_ROOT}")
    print(f"macro:   {OUT_ROOT / 'fdnc_macro_summary.csv'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--folds",
        default="0,1,2,3,4",
        help="robustness biological folds, comma separated, 0..4",
    )
    p.add_argument(
        "--kinds",
        default="coord_noise,missing,outlier",
        help="comma separated subset of coord_noise,missing,outlier",
    )
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    kinds = {x.strip() for x in args.kinds.split(",") if x.strip()}
    allowed = {"coord_noise", "missing", "outlier"}
    if not kinds or not kinds <= allowed:
        raise ValueError(f"--kinds must be subset of {sorted(allowed)}")
    if any(f not in range(5) for f in folds):
        raise ValueError("--folds must contain only 0..4")

    manifest_path = CORR_ROOT / "MANIFEST.json"
    if not manifest_path.is_file():
        # Some materializers used extensionless MANIFEST.
        alt = CORR_ROOT / "MANIFEST"
        if alt.is_file():
            manifest_path = alt
        else:
            raise FileNotFoundError(
                f"Neither {CORR_ROOT / 'MANIFEST.json'} nor {alt} exists"
            )
    manifest = read_json(manifest_path)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[MANIFEST] {manifest_path}", flush=True)
    print(f"[OUTPUT]   {OUT_ROOT}", flush=True)

    results: list[dict[str, Any]] = []

    for fold in folds:
        print("\n" + "=" * 108)
        print(f"ROBUSTNESS FOLD {fold}  -> FINAL fDNC OFFICIAL FOLD_{official_fold(fold)} SEED42")
        print("=" * 108)

        ctx = load_locked_context(fold, device)

        clean_row = clean_row_for_fold(manifest, fold)
        clean_root = Path(clean_row["root"])
        validate_clean_test_population(ctx, clean_root)
        canonical_clean_queries = canonical_unique_gt_count(clean_root)

        # First evaluate clean severity=0 and REQUIRE exact reproduction.
        clean = evaluate_condition(
            ctx=ctx,
            robust_fold=fold,
            row=clean_row,
            device=device,
            clean_shared_queries=int(ctx["saved"]["metrics"]["queries"]),
            canonical_clean_queries=canonical_clean_queries,
        )
        clean_guard(clean, ctx)
        results.append(clean)

        # Only after the exact guard may nonzero corruptions run.
        for row in condition_rows(manifest, fold, kinds):
            if (
                str(row["kind"]) == "coord_noise"
                and abs(float(row["severity"])) <= 1e-15
            ):
                continue
            result = evaluate_condition(
                ctx=ctx,
                robust_fold=fold,
                row=row,
                device=device,
                clean_shared_queries=int(ctx["saved"]["metrics"]["queries"]),
                canonical_clean_queries=canonical_clean_queries,
            )
            results.append(result)

        del ctx["model"]
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summarize(results)
    print("\nCOMPLETE", flush=True)


if __name__ == "__main__":
    main()
