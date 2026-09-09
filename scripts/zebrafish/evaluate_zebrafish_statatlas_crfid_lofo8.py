#!/usr/bin/env python3
"""Leakage-locked StatAtlas/CRF-ID pairwise adaptations for zebrafish LOFO8.

The zebrafish benchmark has pair-local longitudinal IDs, not a shared identity
vocabulary across fish.  Consequently the canonical cross-animal identity-atlas
protocol of StatAtlas and CRF-ID is undefined on this dataset.  This script uses
the scientifically meaningful pairwise adaptations:

* StatAtlas-pair: a query fragment is a one-observation local atlas.  A shared
  residual covariance is estimated from TRAIN pairs; test alignment is
  label-free trimmed ICP and matching is Gaussian/Mahalanobis scoring.
* CRF-ID-pair: bidirectional deformable CPD supplies the official registration
  style unary potential; relational angle/distance consistency refines scores.
  Relation scales are estimated from TRAIN pairs.

Run ``select`` first.  It reads TRAIN and VAL only and writes immutable-looking
parameter locks.  Run ``test`` separately; it verifies the locks before opening
TEST.  Cell IDs are passed only to training-statistic estimation (TRAIN), model
selection metrics (VAL), and final metrics (TEST), never to test scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pycpd import DeformableRegistration
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from mprt_net.data import PairIndex, WormCache, build_pair_targets


METHODS = ("statatlas_pair", "crfid_pair")
EXPECTED_TEST_QUERIES = {1: 3536, 2: 3954, 3: 768, 4: 1820, 5: 1464, 6: 938, 7: 2492, 8: 2184}
OFFICIAL_REFERENCES = {
    "statatlas": {
        "repository": "https://github.com/amin-nejat/stat-atlas",
        "pinned_commit": "29652ab83b6ff870b71a5969e25328fa8e488c90",
        "principles_used": ["affine/similarity atlas alignment", "Gaussian spatial likelihood"],
    },
    "crfid": {
        "repository": "https://github.com/shiveshc/CRF_Cell_ID",
        "pinned_commit": "1166cdb8b26ca1851011112b620e41b044012518",
        "principles_used": ["CPD registration unary", "relative-angle and distance edge consistency"],
        "runtime_note": "Python adaptation because the official implementation requires MATLAB/UGM",
    },
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_manifest(split_dir: Path) -> dict[str, Any]:
    paths = sorted(split_dir.glob("*.npz"))
    h = hashlib.sha256()
    rows = []
    for path in paths:
        digest = sha256_file(path)
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256": digest})
        h.update(path.name.encode())
        h.update(digest.encode())
    return {"split": split_dir.name, "files": len(rows), "sha256": h.hexdigest(), "entries": rows}


def source_hash() -> str:
    return sha256_file(Path(__file__).resolve())


def standardize_xyz(x: np.ndarray) -> np.ndarray:
    """Exact numpy equivalent of mprt_net.relations.standardize_xyz."""
    x = np.asarray(x, dtype=np.float64)
    center = np.median(x, axis=0)
    centered = x - center
    scale = np.sqrt(np.mean(centered * centered, axis=0))
    return centered / np.maximum(scale, 1e-5)


@dataclass
class Pair:
    uid_a: str
    uid_b: str
    a: np.ndarray
    b: np.ndarray
    row_target: np.ndarray
    col_target: np.ndarray


def load_pairs(fold_root: Path, split: str, cache: WormCache) -> list[Pair]:
    index = PairIndex(fold_root, split, min_shared=20)
    pairs = []
    for path_a, path_b in index.pairs:
        sample_a, sample_b, targets = build_pair_targets(cache.get(path_a), cache.get(path_b))
        pairs.append(Pair(
            uid_a=str(sample_a.uid),
            uid_b=str(sample_b.uid),
            a=standardize_xyz(sample_a.xyz.detach().cpu().numpy()),
            b=standardize_xyz(sample_b.xyz.detach().cpu().numpy()),
            row_target=targets.row_target.detach().cpu().numpy().astype(np.int64),
            col_target=targets.col_target.detach().cpu().numpy().astype(np.int64),
        ))
    return pairs


def matched_rows(pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    rows = np.flatnonzero((pair.row_target >= 0) & (pair.row_target < len(pair.b)))
    return pair.a[rows], pair.b[pair.row_target[rows]]


def fit_similarity(a: np.ndarray, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if len(a) < 3:
        return 1.0, np.eye(3), b.mean(0) - a.mean(0)
    ca, cb = a.mean(0), b.mean(0)
    aa, bb = a - ca, b - cb
    u, svals, vt = np.linalg.svd(aa.T @ bb, full_matrices=False)
    rotation = u @ vt  # Reflection is allowed; no identity is consulted.
    scale = float(svals.sum() / max(float(np.sum(aa * aa)), 1e-12))
    scale = scale if np.isfinite(scale) and scale > 1e-8 else 1.0
    translation = cb - scale * (ca @ rotation)
    return scale, rotation, translation


def fit_affine(a: np.ndarray, b: np.ndarray, ridge: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([a, np.ones(len(a))])
    reg = ridge * np.eye(4)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + reg, design.T @ b)
    return beta[:3], beta[3]


def pca_initializations(src: np.ndarray, tgt: np.ndarray):
    yield 1.0, np.eye(3), np.zeros(3)
    cs, ct = src.mean(0), tgt.mean(0)
    xs, xt = src - cs, tgt - ct
    _, _, vhs = np.linalg.svd(xs, full_matrices=False)
    _, _, vht = np.linalg.svd(xt, full_matrices=False)
    bs, bt = vhs.T, vht.T
    rms_s = math.sqrt(float(np.mean(np.sum(xs * xs, axis=1))))
    rms_t = math.sqrt(float(np.mean(np.sum(xt * xt, axis=1))))
    scale = rms_t / max(rms_s, 1e-12)
    for signs in product((-1.0, 1.0), repeat=3):
        rotation = bs @ np.diag(signs) @ bt.T
        yield scale, rotation, ct - scale * (cs @ rotation)


def label_free_icp(
    src: np.ndarray,
    tgt: np.ndarray,
    *,
    transform: str,
    trim: float,
    iterations: int,
) -> np.ndarray:
    """Nearest-neighbour ICP.  This function deliberately has no label input."""
    if transform == "native":
        return src.copy()
    tree = cKDTree(tgt)
    best_x, best_obj = None, float("inf")
    for scale, rotation, translation in pca_initializations(src, tgt):
        cur = scale * (src @ rotation) + translation
        for _ in range(iterations):
            dist, idx = tree.query(cur, k=1)
            keep_n = max(4, int(math.ceil(trim * len(cur))))
            keep = np.argsort(dist, kind="stable")[:keep_n]
            if transform == "similarity":
                ds, dr, dt = fit_similarity(cur[keep], tgt[idx[keep]])
                cur = ds * (cur @ dr) + dt
            elif transform == "affine":
                matrix, offset = fit_affine(cur[keep], tgt[idx[keep]])
                cur = cur @ matrix + offset
            else:
                raise ValueError(transform)
        dist, _ = tree.query(cur, k=1)
        keep_n = max(4, int(math.ceil(trim * len(cur))))
        obj = float(np.mean(np.sort(dist)[:keep_n] ** 2))
        if obj < best_obj:
            best_x, best_obj = cur.copy(), obj
    assert best_x is not None
    return best_x


def estimate_train_state(train_pairs: list[Pair]) -> dict[str, Any]:
    residuals = []
    relation_errors = []
    for pair in train_pairs:
        a, b = matched_rows(pair)
        if len(a) < 4:
            continue
        scale, rotation, translation = fit_similarity(a, b)
        aligned = scale * (a @ rotation) + translation
        residuals.extend((b - aligned).tolist())
        da = np.linalg.norm(a[:, None, :] - a[None, :, :], axis=2)
        db = np.linalg.norm(b[:, None, :] - b[None, :, :], axis=2)
        tri = np.triu_indices(len(a), 1)
        relation_errors.extend(np.abs(da[tri] - db[tri]).tolist())

    residuals_np = np.asarray(residuals, dtype=np.float64)
    if len(residuals_np) < 4:
        covariance = np.eye(3)
    else:
        covariance = np.cov(residuals_np, rowvar=False)
        base = max(float(np.trace(covariance) / 3.0), 1e-4)
        covariance = 0.8 * covariance + 0.2 * base * np.eye(3)
    rel = np.asarray(relation_errors, dtype=np.float64)
    relation_sigma = max(float(np.median(rel)) if len(rel) else 0.25, 0.05)
    return {
        "covariance": covariance.tolist(),
        "relation_sigma": relation_sigma,
        "train_pairs": len(train_pairs),
        "residual_vectors": int(len(residuals_np)),
        "relation_differences": int(len(rel)),
    }


def mahalanobis_score(x: np.ndarray, y: np.ndarray, covariance: np.ndarray, learned: bool) -> np.ndarray:
    inv = np.linalg.pinv(covariance) if learned else np.eye(3)
    delta = x[:, None, :] - y[None, :, :]
    return -np.einsum("ijk,kl,ijl->ij", delta, inv, delta)


def statatlas_scores(pair: Pair, state: dict[str, Any], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(state["covariance"], dtype=np.float64)
    a_to_b = label_free_icp(pair.a, pair.b, transform=cfg["transform"], trim=cfg["trim"], iterations=20)
    b_to_a = label_free_icp(pair.b, pair.a, transform=cfg["transform"], trim=cfg["trim"], iterations=20)
    ab = mahalanobis_score(a_to_b, pair.b, cov, cfg["covariance"] == "train")
    ba = mahalanobis_score(b_to_a, pair.a, cov, cfg["covariance"] == "train")
    return ab, ba


def cpd_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    reg = DeformableRegistration(
        X=target.astype(np.float64),
        Y=source.astype(np.float64),
        beta=1.0,
        alpha=3.0,
        max_iterations=100,
        tolerance=1e-5,
    )
    transformed, _ = reg.register()
    return np.asarray(transformed, dtype=np.float64)


def crf_refine(
    original_source: np.ndarray,
    aligned_source: np.ndarray,
    target: np.ndarray,
    *,
    relation_sigma: float,
    edge_weight: float,
    iterations: int = 4,
) -> np.ndarray:
    """Deterministic max-product-style CRF update with unique assignments."""
    d2 = np.sum((aligned_source[:, None, :] - target[None, :, :]) ** 2, axis=2)
    positive = d2[d2 > 0]
    unary_sigma2 = max(float(np.median(positive)) if len(positive) else 0.1, 1e-4)
    unary = -d2 / (2.0 * unary_sigma2)
    if edge_weight <= 0:
        return unary

    ds = np.linalg.norm(original_source[:, None, :] - original_source[None, :, :], axis=2)
    dt = np.linalg.norm(target[:, None, :] - target[None, :, :], axis=2)
    current = unary.copy()
    for _ in range(iterations):
        rows, cols = linear_sum_assignment(-current)
        mapping = {int(i): int(j) for i, j in zip(rows, cols)}
        relation = np.zeros_like(unary)
        for i in range(len(original_source)):
            support = [(k, mapping[k]) for k in mapping if k != i]
            if not support:
                continue
            ks = np.asarray([v[0] for v in support], dtype=np.int64)
            ls = np.asarray([v[1] for v in support], dtype=np.int64)
            src_vec = aligned_source[i] - aligned_source[ks]
            src_norm = np.linalg.norm(src_vec, axis=1)
            for j in range(len(target)):
                tgt_vec = target[j] - target[ls]
                tgt_norm = np.linalg.norm(tgt_vec, axis=1)
                denom = np.maximum(src_norm * tgt_norm, 1e-8)
                angle = np.sum(src_vec * tgt_vec, axis=1) / denom
                dist_error = ds[i, ks] - dt[j, ls]
                distance = np.exp(-0.5 * (dist_error / relation_sigma) ** 2)
                relation[i, j] = float(np.mean(0.5 * angle + 0.5 * distance))
        current = unary + float(edge_weight) * relation
    return current


def crfid_base(pair: Pair) -> dict[str, np.ndarray]:
    return {
        "a_to_b": cpd_align(pair.a, pair.b),
        "b_to_a": cpd_align(pair.b, pair.a),
    }


def crfid_scores(pair: Pair, state: dict[str, Any], cfg: dict[str, Any], base: dict[str, np.ndarray] | None = None):
    base = crfid_base(pair) if base is None else base
    ab = crf_refine(pair.a, base["a_to_b"], pair.b, relation_sigma=state["relation_sigma"], edge_weight=cfg["edge_weight"])
    ba = crf_refine(pair.b, base["b_to_a"], pair.a, relation_sigma=state["relation_sigma"], edge_weight=cfg["edge_weight"])
    return ab, ba


def directional(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | int]:
    valid = (targets >= 0) & (targets < scores.shape[1])
    scores_v, targets_v = scores[valid], targets[valid]
    if not len(targets_v):
        return {"queries": 0, "top1": 0, "top5": 0, "rr": 0.0}
    target_score = scores_v[np.arange(len(targets_v)), targets_v]
    rank = 1 + np.sum(scores_v > target_score[:, None], axis=1)
    return {"queries": len(rank), "top1": int(np.sum(rank <= 1)), "top5": int(np.sum(rank <= 5)), "rr": float(np.sum(1.0 / rank))}


def hungarian(scores: np.ndarray, row_target: np.ndarray, col_target: np.ndarray) -> tuple[int, int]:
    rows, cols = linear_sum_assignment(-scores)
    assignment = {int(i): int(j) for i, j in zip(rows, cols)}
    inverse = {j: i for i, j in assignment.items()}
    correct = total = 0
    for i, target in enumerate(row_target):
        if 0 <= target < scores.shape[1]:
            total += 1
            correct += int(assignment.get(i, -1) == int(target))
    for j, target in enumerate(col_target):
        if 0 <= target < scores.shape[0]:
            total += 1
            correct += int(inverse.get(j, -1) == int(target))
    return correct, total


def evaluate(pairs: list[Pair], score_fn) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = {"queries": 0, "top1": 0, "top5": 0, "rr": 0.0, "hc": 0, "hq": 0}
    pair_rows = []
    for idx, pair in enumerate(pairs, 1):
        ab, ba = score_fn(pair)
        left, right = directional(ab, pair.row_target), directional(ba, pair.col_target)
        hc, hq = hungarian(0.5 * (ab + ba.T), pair.row_target, pair.col_target)
        row = {
            "uid_a": pair.uid_a, "uid_b": pair.uid_b,
            "queries": int(left["queries"] + right["queries"]),
            "top1_correct": int(left["top1"] + right["top1"]),
            "top5_correct": int(left["top5"] + right["top5"]),
            "reciprocal_rank_sum": float(left["rr"] + right["rr"]),
            "hungarian_correct": int(hc), "hungarian_queries": int(hq),
        }
        pair_rows.append(row)
        for key, source in (("queries", "queries"), ("top1", "top1_correct"), ("top5", "top5_correct"), ("rr", "reciprocal_rank_sum"), ("hc", "hungarian_correct"), ("hq", "hungarian_queries")):
            totals[key] += row[source]
        print(f"  [{idx:02d}/{len(pairs):02d}] {pair.uid_a} <-> {pair.uid_b}", flush=True)
    q, hq = max(totals["queries"], 1), max(totals["hq"], 1)
    metrics = {
        "pairs": len(pairs), "queries": int(totals["queries"]),
        "top1_real": float(totals["top1"] / q), "top5_real": float(totals["top5"] / q),
        "mrr_real": float(totals["rr"] / q), "hungarian_queries": int(totals["hq"]),
        "hungarian_accuracy": float(totals["hc"] / hq),
    }
    return metrics, pair_rows


def candidate_grid(method: str) -> list[dict[str, Any]]:
    if method == "statatlas_pair":
        return [
            {"transform": transform, "trim": trim, "covariance": covariance}
            for transform in ("native", "similarity", "affine")
            for trim in ((1.0,) if transform == "native" else (0.6, 0.8, 1.0))
            for covariance in ("isotropic", "train")
        ]
    return [{"edge_weight": weight} for weight in (0.0, 0.005, 0.01, 0.025, 0.05, 0.1)]


def selection_key(metrics: dict[str, Any], cfg: dict[str, Any]) -> tuple[Any, ...]:
    # Predeclared lexicographic objective; cfg JSON makes exact ties deterministic.
    return (metrics["top1_real"], metrics["mrr_real"], metrics["hungarian_accuracy"], json.dumps(cfg, sort_keys=True))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def select_fold(args, fold: int, method: str) -> None:
    fold_root = args.data_root / f"fold_{fold}"
    cache = WormCache(activity_length=128)
    print(f"[SELECT] fold={fold} method={method}: opening TRAIN and VAL only", flush=True)
    train_pairs = load_pairs(fold_root, "train", cache)
    val_pairs = load_pairs(fold_root, "val", cache)
    state = estimate_train_state(train_pairs)
    grid_rows = []
    crf_bases = [crfid_base(pair) for pair in val_pairs] if method == "crfid_pair" else None
    for number, cfg in enumerate(candidate_grid(method), 1):
        print(f" candidate {number}/{len(candidate_grid(method))}: {cfg}", flush=True)
        if method == "statatlas_pair":
            fn = lambda pair, c=cfg: statatlas_scores(pair, state, c)
        else:
            base_by_id = {id(pair): base for pair, base in zip(val_pairs, crf_bases)}
            fn = lambda pair, c=cfg: crfid_scores(pair, state, c, base_by_id[id(pair)])
        metrics, _ = evaluate(val_pairs, fn)
        grid_rows.append({"config": cfg, "validation_metrics": metrics})
    winner = max(grid_rows, key=lambda row: selection_key(row["validation_metrics"], row["config"]))
    lock = {
        "schema": "zebrafish-pairwise-adaptation-lock-v1", "fold": fold, "method": method,
        "selection_split": "val", "test_opened_during_selection": False,
        "selected_config": winner["config"], "selected_validation_metrics": winner["validation_metrics"],
        "train_state": state, "candidate_results": grid_rows,
        "source_sha256": source_hash(),
        "train_manifest": split_manifest(fold_root / "train"),
        "val_manifest": split_manifest(fold_root / "val"),
        "identity_contract": "IDs are pair-local and are not pooled across fish",
        "official_references": OFFICIAL_REFERENCES,
    }
    path = args.run_root / "locks" / f"fold_{fold}_{method}.json"
    write_json(path, lock)
    print(f"[LOCKED] {path} -> {winner['config']}", flush=True)


def verify_lock(lock: dict[str, Any], fold_root: Path, fold: int, method: str) -> None:
    if lock.get("fold") != fold or lock.get("method") != method:
        raise RuntimeError("lock fold/method mismatch")
    if lock.get("selection_split") != "val" or lock.get("test_opened_during_selection") is not False:
        raise RuntimeError("lock leakage declaration is invalid")
    if lock.get("source_sha256") != source_hash():
        raise RuntimeError("source changed after selection; rerun select before test")
    for split in ("train", "val"):
        current = split_manifest(fold_root / split)["sha256"]
        if current != lock[f"{split}_manifest"]["sha256"]:
            raise RuntimeError(f"{split} data changed after selection")


def test_fold(args, fold: int, method: str) -> dict[str, Any]:
    fold_root = args.data_root / f"fold_{fold}"
    lock_path = args.run_root / "locks" / f"fold_{fold}_{method}.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"missing selection lock: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    verify_lock(lock, fold_root, fold, method)
    print(f"[TEST] fold={fold} method={method}: lock verified; opening TEST now", flush=True)
    test_pairs = load_pairs(fold_root, "test", WormCache(activity_length=128))
    cfg, state = lock["selected_config"], lock["train_state"]
    fn = (lambda pair: statatlas_scores(pair, state, cfg)) if method == "statatlas_pair" else (lambda pair: crfid_scores(pair, state, cfg))
    metrics, pair_rows = evaluate(test_pairs, fn)
    expected_q = EXPECTED_TEST_QUERIES[fold]
    if metrics["queries"] != expected_q or metrics["hungarian_queries"] != expected_q:
        raise RuntimeError(
            f"fold {fold}: locked denominator mismatch: metrics={metrics['queries']}/"
            f"{metrics['hungarian_queries']}, expected={expected_q}/{expected_q}"
        )
    result = {
        "schema": "zebrafish-pairwise-adaptation-result-v1", "fold": fold, "method": method,
        "display_name": "StatAtlas (pairwise adaptation)" if method == "statatlas_pair" else "CRF-ID (pairwise Python adaptation)",
        "protocol": "LOFO8; train statistics + validation selection + once-only locked test",
        "selected_config": cfg, "metrics": metrics,
        "lock_sha256": sha256_file(lock_path), "source_sha256": source_hash(),
        "test_manifest": split_manifest(fold_root / "test"),
        "test_labels_usage": "metrics only, after both directional score matrices are complete",
        "activity_used": False, "official_references": OFFICIAL_REFERENCES,
    }
    out_dir = args.run_root / "test" / f"fold_{fold}" / method
    write_json(out_dir / "metrics.json", result)
    write_jsonl(out_dir / "pairs.jsonl", pair_rows)
    return result


def aggregate(args, methods: list[str]) -> None:
    rows = []
    aggregate = {"schema": "zebrafish-pairwise-adaptation-aggregate-v1", "folds": args.folds, "aggregation": "unweighted mean and sample SD across held-out fish", "sample_sd_ddof": 1, "methods": {}}
    for method in methods:
        results = []
        for fold in args.folds:
            path = args.run_root / "test" / f"fold_{fold}" / method / "metrics.json"
            if not path.exists():
                raise FileNotFoundError(path)
            results.append(json.loads(path.read_text(encoding="utf-8")))
        summary = {}
        for metric in ("top1_real", "top5_real", "mrr_real", "hungarian_accuracy"):
            values = np.asarray([r["metrics"][metric] for r in results], dtype=float)
            summary[metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "fold_values": values.tolist()}
        display = results[0]["display_name"]
        aggregate["methods"][method] = {"display_name": display, "summary": summary, "fold_results": results}
        rows.append((display, summary))
    write_json(args.run_root / "aggregate.json", aggregate)
    lines = ["# Zebrafish StatAtlas / CRF-ID pairwise-adaptation results", "", "Unweighted mean ± sample SD across the eight held-out fish. Position only.", "", "| Method | Top-1 | Top-5 | MRR | Hungarian |", "|---|---:|---:|---:|---:|"]
    for display, s in rows:
        lines.append(f"| {display} | {100*s['top1_real']['mean']:.2f} ± {100*s['top1_real']['sd']:.2f}% | {100*s['top5_real']['mean']:.2f} ± {100*s['top5_real']['sd']:.2f}% | {s['mrr_real']['mean']:.4f} ± {s['mrr_real']['sd']:.4f} | {100*s['hungarian_accuracy']['mean']:.2f} ± {100*s['hungarian_accuracy']['sd']:.2f}% |")
    lines += ["", "These are pairwise adaptations, not canonical global-identity atlas runs: zebrafish IDs are local to each longitudinal pair.", ""]
    (args.run_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "test", "aggregate"))
    parser.add_argument("--data-root", type=Path, default=Path("Data/Zebrafish_MPRT_LOFO8_60m"))
    parser.add_argument("--run-root", type=Path, default=Path("runs/zebrafish_statatlas_crfid_pairwise_lofo8_v1"))
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    args.data_root, args.run_root = args.data_root.resolve(), args.run_root.resolve()
    if args.stage == "select":
        for fold in args.folds:
            for method in args.methods:
                select_fold(args, fold, method)
    elif args.stage == "test":
        for fold in args.folds:
            for method in args.methods:
                test_fold(args, fold, method)
    else:
        aggregate(args, args.methods)


if __name__ == "__main__":
    main()
