#!/usr/bin/env python3
"""
RLD natural population-size accuracy scaling — DIRECT FINAL-MODEL replay.

No candidate-score CSVs and no export_unified_candidate_scores.py are required.

Models
------
MPRT:
  runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/
    rld/foldF/seed42/static_atlas/anchored_pure.pt
  Exact production path:
    encode_population(query) -> match_encodings(query, static_atlas)

fDNC:
  runs/fdnc_current_grouped_cv_v2/
    rld/foldF/seed42/selected/best.pt
  Exact current grouped-CV scorer and fixed OUTER-TRAIN geometry medoid.

Protocol
--------
* clean held-out RLD test worms only
* original natural population size N; NO corruption/subsampling/deletion/duplication
* one locked model seed42
* exact same canonical query cohort for both methods:
    every unique supervised labeled neuron in each held-out worm
* if GT identity is absent from a method's candidate universe:
    it stays in the denominator and is incorrect (RR=0)
* also reports candidate coverage and covered-only Top-1, so ranking-vs-coverage
  effects can be separated
* 4 equal-count quantile bins over biological test worms by natural N
* primary system metric: pooled OOF Top-1 over canonical queries
* 20k biological-worm cluster bootstrap CI
* paired biological-worm bootstrap for MPRT - fDNC
* Spearman association between natural N and per-worm accuracy

Run:
  CUDA_VISIBLE_DEVICES=1 python -u analyze_rld_natural_population_accuracy_scaling_direct.py \
      --device cuda:0 --bootstrap 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
CV_ROOT = ROOT / "Data/Dunn_001623/cv5_grouped_v1"
MPRT_ROOT = ROOT / "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1/rld"
FDNC_ROOT = ROOT / "runs/fdnc_current_grouped_cv_v2/rld"
DEFAULT_OUT = ROOT / "runs/rld_natural_accuracy_scaling_direct_current_v1"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mprt_net_v1_1"))

from mprt_net.data import WormCache, split_files
from mprt_net.evaluate import load_checkpoint
from baselines.atanas_locked import normalize_xyz
from scripts.lib.fair_identity_protocol import select_geometry_medoid
from engines import evaluate_atanas_fdnc_unified as fdnc_loader
from engines import train_atanas_fdnc_fold as fdnc_train

INVALID_LABELS = {
    "", "nan", "none", "null", "-1",
    "unknown", "unk", "?", "unlabeled", "unlabelled",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20260826)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def norm_label(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    return "" if text.lower() in INVALID_LABELS else text


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_queries(sample):
    labels = [norm_label(x) for x in sample.cell_ids]
    mask = sample.supervised_mask.detach().cpu().numpy().astype(bool)
    counts = Counter(x for x, keep in zip(labels, mask) if keep and x)
    indices = [
        i for i, (label, keep) in enumerate(zip(labels, mask))
        if keep and label and counts[label] == 1
    ]
    signature = tuple((int(i), labels[i]) for i in indices)
    return indices, labels, signature


def candidate_unique_map(labels):
    counts = Counter(x for x in labels if x)
    return {
        x: i for i, x in enumerate(labels)
        if x and counts[x] == 1
    }


def natural_population_n(path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        xyz = np.asarray(data["xyz"])
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError(f"{path}: invalid xyz shape {xyz.shape}")
    return int(np.isfinite(xyz[:, :3]).all(axis=1).sum())


def score_metrics(
    query_indices,
    query_labels,
    candidate_labels,
    ranking_full,
    assignment_full,
):
    """
    Denominator = exact canonical query set.
    Missing GT candidate is an error, not an excluded query.
    """
    q = len(query_indices)
    cmap = candidate_unique_map(candidate_labels)

    if q == 0:
        return {
            "queries": 0,
            "covered_queries": 0,
            "top1_correct": 0,
            "top5_correct": 0,
            "rr_sum": 0.0,
            "hungarian_correct": 0,
            "top1": math.nan,
            "top5": math.nan,
            "mrr": math.nan,
            "hungarian": math.nan,
            "candidate_coverage": math.nan,
            "covered_only_top1": math.nan,
        }

    qidx = np.asarray(query_indices, dtype=np.int64)
    ranking = np.asarray(ranking_full, dtype=np.float64)[qidx]
    assignment = np.asarray(assignment_full, dtype=np.float64)[qidx]

    if ranking.shape[1] != len(candidate_labels):
        raise RuntimeError(
            f"ranking columns={ranking.shape[1]} candidate_labels={len(candidate_labels)}"
        )
    if assignment.shape != ranking.shape:
        raise RuntimeError(
            f"assignment shape={assignment.shape} ranking shape={ranking.shape}"
        )

    top1 = top5 = covered = 0
    rr_sum = 0.0

    for r, qi in enumerate(query_indices):
        gt = query_labels[qi]
        cj = cmap.get(gt)
        if cj is None:
            continue
        covered += 1
        row = ranking[r]
        target = float(row[cj])
        # Same optimistic tie policy as the MPRT benchmark.
        rank = 1 + int(np.sum(row > target))
        top1 += int(rank <= 1)
        top5 += int(rank <= min(5, len(row)))
        rr_sum += 1.0 / rank

    # Hungarian on all canonical query rows. A query whose GT is absent
    # from candidates cannot score as correct and remains in q denominator.
    hung = 0
    if assignment.size:
        finite = np.isfinite(assignment)
        if finite.any():
            floor = float(assignment[finite].min()) - 1e6
            mat = np.where(finite, assignment, floor)
            rr, cc = linear_sum_assignment(-mat)
            for r, c in zip(rr.tolist(), cc.tolist()):
                qi = query_indices[r]
                if candidate_labels[c] == query_labels[qi]:
                    hung += 1

    return {
        "queries": q,
        "covered_queries": covered,
        "top1_correct": top1,
        "top5_correct": top5,
        "rr_sum": rr_sum,
        "hungarian_correct": hung,
        "top1": top1 / q,
        "top5": top5 / q,
        "mrr": rr_sum / q,
        "hungarian": hung / q,
        "candidate_coverage": covered / q,
        "covered_only_top1": top1 / covered if covered else math.nan,
    }


@torch.inference_mode()
def evaluate_mprt_fold(fold, seed, device):
    ckpt = MPRT_ROOT / f"fold{fold}/seed{seed}/static_atlas/anchored_pure.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"MPRT checkpoint missing: {ckpt}")

    fold_root = CV_ROOT / f"fold_{fold}"
    test_paths = split_files(fold_root, "test")
    cache = WormCache(activity_length=512, max_items=max(32, len(test_paths)))

    model, payload = load_checkpoint(ckpt, device)
    model.eval()
    atlas = model.atlas_encoding()

    mapping = {norm_label(k): int(v) for k, v in payload["atlas_identity_to_slot"].items()}
    slot_to_label = {slot: label for label, slot in mapping.items()}
    candidate_labels = [
        slot_to_label.get(i, f"__UNMAPPED_SLOT_{i:04d}")
        for i in range(model.config.atlas_size)
    ]

    rows = []
    signatures = {}

    for path in test_paths:
        sample_cpu = cache.get(path)
        uid = str(sample_cpu.uid)
        indices, labels, signature = canonical_queries(sample_cpu)
        signatures[uid] = signature

        sample = sample_cpu.to(device)
        encoded = model.encode_population(sample)
        output = model.match_encodings(encoded, atlas)

        ranking = output.row_conditional[:, :-1].detach().float().cpu().numpy()
        assignment = output.plan[:-1, :-1].detach().float().cpu().numpy()

        if ranking.shape[0] != sample_cpu.num_nodes:
            raise RuntimeError(
                f"MPRT fold{fold} {uid}: ranking rows={ranking.shape[0]} "
                f"nodes={sample_cpu.num_nodes}"
            )

        metrics = score_metrics(
            indices, labels, candidate_labels, ranking, assignment
        )
        rows.append({
            "method": "MPRT",
            "fold": fold,
            "seed": seed,
            "group_uid": uid,
            "population_n": natural_population_n(Path(sample_cpu.source_path)),
            **metrics,
        })

    provenance = {
        "method": "MPRT",
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_sha256": sha256(ckpt),
        "candidate_universe": "final Static Atlas slots",
        "query_definition": "unique supervised labeled neurons in clean held-out worm",
    }

    del model, atlas
    torch.cuda.empty_cache()
    return rows, signatures, provenance


@torch.inference_mode()
def evaluate_fdnc_fold(fold, seed, device, mprt_signatures):
    ckpt = FDNC_ROOT / f"fold{fold}/seed{seed}/selected/best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"fDNC current checkpoint missing: {ckpt}")

    fold_root = CV_ROOT / f"fold_{fold}"
    train_paths = split_files(fold_root, "train")
    test_paths = split_files(fold_root, "test")
    cache = WormCache(
        activity_length=512,
        max_items=max(32, len(train_paths) + len(test_paths)),
    )
    train_samples = [cache.get(p) for p in train_paths]
    test_samples = [cache.get(p) for p in test_paths]

    # Exact current clean evaluator template policy.
    normalized_train_xyz = [
        normalize_xyz(s.xyz.cpu().numpy()) for s in train_samples
    ]
    medoid = select_geometry_medoid(
        [str(s.uid) for s in train_samples],
        normalized_train_xyz,
    )
    template = train_samples[medoid.index]

    model = fdnc_loader.load_checkpoint(ckpt, device, 128, 6)
    model.eval()

    cand_idx, template_labels, _ = canonical_queries(template)
    candidate_labels = [template_labels[i] for i in cand_idx]
    if not cand_idx:
        raise RuntimeError(f"fold{fold}: fDNC medoid has no candidate identities")

    ref_xyz = torch.from_numpy(
        normalize_xyz(template.xyz.cpu().numpy())
    ).to(device=device, dtype=torch.float32)

    rows = []
    seen = set()

    for sample_cpu in test_samples:
        uid = str(sample_cpu.uid)
        indices, labels, signature = canonical_queries(sample_cpu)

        if uid not in mprt_signatures:
            raise RuntimeError(f"fold{fold}: fDNC worm {uid} absent from MPRT")
        if signature != mprt_signatures[uid]:
            raise RuntimeError(
                f"QUERY SIGNATURE MISMATCH fold={fold} worm={uid}: "
                f"MPRT={len(mprt_signatures[uid])} fDNC={len(signature)}"
            )
        seen.add(uid)

        q_xyz = torch.from_numpy(
            normalize_xyz(sample_cpu.xyz.cpu().numpy())
        ).to(device=device, dtype=torch.float32)

        # score_pair(a, b) returns b->a first, a->b second.
        _, q_to_ref = fdnc_train.score_pair(model, q_xyz, ref_xyz)
        scores = q_to_ref[:, :-1].detach().float().cpu().numpy()

        expected_shape = (sample_cpu.num_nodes, template.num_nodes)
        if tuple(scores.shape) != expected_shape:
            raise RuntimeError(
                f"fDNC fold{fold} {uid}: scores={scores.shape} expected={expected_shape}"
            )

        restricted = scores[:, np.asarray(cand_idx, dtype=np.int64)]
        metrics = score_metrics(
            indices, labels, candidate_labels, restricted, restricted
        )
        rows.append({
            "method": "fDNC",
            "fold": fold,
            "seed": seed,
            "group_uid": uid,
            "population_n": natural_population_n(Path(sample_cpu.source_path)),
            **metrics,
        })

    if seen != set(mprt_signatures):
        raise RuntimeError(
            f"fold{fold}: biological worm set mismatch: "
            f"missing={sorted(set(mprt_signatures)-seen)[:10]}"
        )

    provenance = {
        "method": "fDNC",
        "checkpoint": str(ckpt.resolve()),
        "checkpoint_sha256": sha256(ckpt),
        "template_uid": str(template.uid),
        "template_mean_distance": float(medoid.mean_distance),
        "template_selection": "outer-train-only symmetric geometry medoid",
        "candidate_universe": "unique supervised identities in fixed medoid",
        "query_definition": "EXACT MPRT canonical query signature",
    }

    del model
    torch.cuda.empty_cache()
    return rows, provenance


def assign_quantile_bins(meta, bins):
    meta = meta.copy()
    # Deterministic equal-count grouping even when N ties occur.
    rank = meta["population_n"].rank(method="first")
    meta["size_bin_index"] = pd.qcut(rank, q=bins, labels=False).astype(int)

    label_map = {}
    for b, part in meta.groupby("size_bin_index", sort=True):
        lo = int(part["population_n"].min())
        hi = int(part["population_n"].max())
        med = float(part["population_n"].median())
        label_map[int(b)] = f"Q{int(b)+1}: N={lo}-{hi} (median {med:g})"

    meta["size_bin"] = meta["size_bin_index"].map(label_map)
    return meta


def pooled_metrics(part):
    q = int(part["queries"].sum())
    covered = int(part["covered_queries"].sum())
    return {
        "worms": int(part["group_uid"].nunique()),
        "queries": q,
        "covered_queries": covered,
        "top1": float(part["top1_correct"].sum() / q) if q else math.nan,
        "top5": float(part["top5_correct"].sum() / q) if q else math.nan,
        "mrr": float(part["rr_sum"].sum() / q) if q else math.nan,
        "hungarian": float(part["hungarian_correct"].sum() / q) if q else math.nan,
        "candidate_coverage": covered / q if q else math.nan,
        "covered_only_top1": (
            float(part["top1_correct"].sum() / covered)
            if covered else math.nan
        ),
    }


def bootstrap_top1(part, iterations, rng):
    by = (
        part.groupby("group_uid", as_index=False)[["top1_correct", "queries"]]
        .sum()
    )
    hits = by["top1_correct"].to_numpy(float)
    qs = by["queries"].to_numpy(float)
    n = len(by)
    vals = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        vals[i] = hits[idx].sum() / qs[idx].sum()
    return np.percentile(vals, [2.5, 97.5])


def paired_bootstrap_delta(mprt, fdnc, iterations, rng):
    a = mprt.set_index("group_uid")[["top1_correct", "queries"]]
    b = fdnc.set_index("group_uid")[["top1_correct", "queries"]]

    if set(a.index) != set(b.index):
        raise RuntimeError("Paired bootstrap worm sets differ")

    common = sorted(a.index)
    aq = a.loc[common, "queries"].to_numpy(float)
    bq = b.loc[common, "queries"].to_numpy(float)
    if not np.array_equal(aq, bq):
        raise RuntimeError("Paired per-worm query denominators differ")

    ah = a.loc[common, "top1_correct"].to_numpy(float)
    bh = b.loc[common, "top1_correct"].to_numpy(float)

    point = float((ah.sum() - bh.sum()) / aq.sum())

    n = len(common)
    vals = np.empty(iterations, dtype=float)
    for i in range(iterations):
        idx = rng.integers(0, n, size=n)
        vals[i] = (ah[idx].sum() - bh[idx].sum()) / aq[idx].sum()

    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.seed != 42:
        raise ValueError("Locked direct scaling analysis currently supports seed42 only.")
    if any(f not in range(5) for f in args.folds):
        raise ValueError(args.folds)
    if args.bins < 2:
        raise ValueError("--bins must be >=2")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    print("=" * 122)
    print("RLD NATURAL POPULATION-SIZE ACCURACY SCALING — DIRECT LOCKED CLEAN REPLAY")
    print("=" * 122)
    print("folds       :", args.folds)
    print("seed        :", args.seed)
    print("bins        :", args.bins)
    print("bootstrap   :", args.bootstrap)
    print("corruption  : NONE")
    print("CSV inputs  : NONE")
    print("GPU         :", torch.cuda.get_device_name(device))
    print()

    all_rows = []
    provenance = []

    for fold in args.folds:
        print("\n" + "=" * 122)
        print(f"FOLD {fold}")
        print("=" * 122)

        mprt_rows, signatures, mprt_prov = evaluate_mprt_fold(
            fold, args.seed, device
        )
        fdnc_rows, fdnc_prov = evaluate_fdnc_fold(
            fold, args.seed, device, signatures
        )

        m = pd.DataFrame(mprt_rows)
        f = pd.DataFrame(fdnc_rows)

        # Hard exact paired biological/query denominator guard.
        mm = m.set_index("group_uid")["queries"].sort_index()
        ff = f.set_index("group_uid")["queries"].sort_index()
        if not mm.index.equals(ff.index) or not np.array_equal(
            mm.to_numpy(), ff.to_numpy()
        ):
            raise RuntimeError(f"fold{fold}: paired query denominator mismatch")

        print(
            f"fold{fold}: exact MPRT/fDNC biological + query signature "
            f"worms={len(mm)} Q={int(mm.sum())} ✓"
        )

        pm = pooled_metrics(m)
        pf = pooled_metrics(f)
        print(
            f"  MPRT Top1={100*pm['top1']:.2f}% "
            f"Coverage={100*pm['candidate_coverage']:.2f}% "
            f"CoveredTop1={100*pm['covered_only_top1']:.2f}%"
        )
        print(
            f"  fDNC Top1={100*pf['top1']:.2f}% "
            f"Coverage={100*pf['candidate_coverage']:.2f}% "
            f"CoveredTop1={100*pf['covered_only_top1']:.2f}%"
        )

        all_rows.extend(mprt_rows)
        all_rows.extend(fdnc_rows)
        provenance.extend([mprt_prov, fdnc_prov])

    metrics = pd.DataFrame(all_rows)
    metrics.to_csv(args.output_dir / "worm_metrics_unbinned.csv", index=False)

    # Metadata must be method-independent.
    meta = (
        metrics[["fold", "group_uid", "population_n"]]
        .drop_duplicates()
        .copy()
    )
    dup = meta.groupby(["fold", "group_uid"])["population_n"].nunique()
    if (dup > 1).any():
        raise RuntimeError("Population N disagrees between methods")

    meta = meta.drop_duplicates(["fold", "group_uid"])
    meta = assign_quantile_bins(meta, args.bins)
    meta.to_csv(args.output_dir / "worm_metadata.csv", index=False)

    metrics = metrics.merge(
        meta,
        on=["fold", "group_uid", "population_n"],
        how="left",
        validate="many_to_one",
    )
    metrics.to_csv(args.output_dir / "worm_metrics.csv", index=False)

    print("\nN distribution:")
    print(meta["population_n"].describe().to_string())
    print()
    for b, part in meta.groupby(["size_bin_index", "size_bin"], sort=True):
        print(
            f"  bin{int(b[0])}: {b[1]} | worms={len(part)} "
            f"meanN={part.population_n.mean():.1f}"
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    summary_rows = []
    delta_rows = []

    print()
    print("=" * 122)
    print("NATURAL-N ACCURACY SCALING")
    print("=" * 122)

    for bidx in sorted(metrics["size_bin_index"].unique()):
        part_bin = metrics[metrics["size_bin_index"] == bidx]
        label = part_bin["size_bin"].iloc[0]
        print(f"\n{label}")

        method_parts = {}
        for method in ("MPRT", "fDNC"):
            part = part_bin[part_bin["method"] == method].copy()
            method_parts[method] = part
            p = pooled_metrics(part)
            lo, hi = bootstrap_top1(part, args.bootstrap, rng)
            row = {
                "size_bin_index": int(bidx),
                "size_bin": label,
                "method": method,
                **p,
                "top1_ci95_low": float(lo),
                "top1_ci95_high": float(hi),
            }
            summary_rows.append(row)

            print(
                f"  {method:<5s} worms={p['worms']:2d} Q={p['queries']:4d} "
                f"Top1={100*p['top1']:6.2f}% "
                f"CI=[{100*lo:6.2f},{100*hi:6.2f}] "
                f"Top5={100*p['top5']:6.2f}% "
                f"MRR={p['mrr']:.4f} "
                f"Hung={100*p['hungarian']:6.2f}% "
                f"CandCov={100*p['candidate_coverage']:6.2f}% "
                f"CoveredTop1={100*p['covered_only_top1']:6.2f}%"
            )

        point, lo, hi = paired_bootstrap_delta(
            method_parts["MPRT"],
            method_parts["fDNC"],
            args.bootstrap,
            rng,
        )
        delta_rows.append({
            "size_bin_index": int(bidx),
            "size_bin": label,
            "mprt_minus_fdnc_top1": point,
            "ci95_low": lo,
            "ci95_high": hi,
        })
        print(
            f"  Δ MPRT-fDNC Top1 = {100*point:+6.2f} pp "
            f"95% CI [{100*lo:+6.2f},{100*hi:+6.2f}]"
        )

    summary = pd.DataFrame(summary_rows)
    deltas = pd.DataFrame(delta_rows)
    summary.to_csv(args.output_dir / "scaling_summary.csv", index=False)
    deltas.to_csv(args.output_dir / "paired_delta_summary.csv", index=False)

    # Per-worm associations.
    corr_rows = []
    print()
    print("=" * 122)
    print("SPEARMAN NATURAL-N ASSOCIATION")
    print("=" * 122)

    for method in ("MPRT", "fDNC"):
        part = metrics[metrics["method"] == method].copy()
        for metric in ("top1", "hungarian", "candidate_coverage", "covered_only_top1"):
            x = part["population_n"].to_numpy(float)
            y = part[metric].to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(y)
            rho, pval = spearmanr(x[keep], y[keep])
            corr_rows.append({
                "method": method,
                "metric": metric,
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "worms": int(keep.sum()),
            })
            print(
                f"{method:<5s} {metric:<18s} "
                f"rho={rho:+.3f} p={pval:.4g} worms={int(keep.sum())}"
            )

    # Direct per-worm advantage association.
    a = metrics[metrics.method == "MPRT"].set_index(["fold", "group_uid"])
    b = metrics[metrics.method == "fDNC"].set_index(["fold", "group_uid"])
    common = a.index.intersection(b.index)
    delta_worm = (
        a.loc[common, "top1"].to_numpy(float)
        - b.loc[common, "top1"].to_numpy(float)
    )
    n_worm = a.loc[common, "population_n"].to_numpy(float)
    rho, pval = spearmanr(n_worm, delta_worm)
    corr_rows.append({
        "method": "MPRT-fDNC",
        "metric": "per_worm_top1_delta",
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "worms": int(len(common)),
    })
    print(
        f"{'ΔM-F':<5s} {'per_worm_top1_delta':<18s} "
        f"rho={rho:+.3f} p={pval:.4g} worms={len(common)}"
    )

    pd.DataFrame(corr_rows).to_csv(
        args.output_dir / "correlations.csv", index=False
    )

    protocol = {
        "protocol": "RLD natural population-size accuracy scaling direct current v1",
        "dataset": str(CV_ROOT.resolve()),
        "folds": args.folds,
        "seed": args.seed,
        "methods": {
            "MPRT": "final Static Atlas anchored_pure",
            "fDNC": "current-grouped fine-tuned selected/best.pt + fixed outer-train geometry medoid",
        },
        "population_size": "original finite xyz rows in clean held-out test worm",
        "binning": f"{args.bins} deterministic equal-count quantile bins over biological worms",
        "canonical_query_cohort": "unique supervised labeled neurons in each held-out worm; exact paired denominator",
        "missing_candidate_policy": "query retained; incorrect; RR=0",
        "candidate_coverage_reported": True,
        "covered_only_top1_reported": True,
        "no_corruption": True,
        "no_subsampling": True,
        "no_duplication": True,
        "no_retraining": True,
        "no_test_time_tuning": True,
        "primary_metric": "pooled OOF system Top-1 within natural-N bin",
        "ci": f"{args.bootstrap}-iteration biological query-worm cluster bootstrap",
        "paired_delta": "MPRT - fDNC, paired biological-worm bootstrap",
        "provenance": provenance,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("Outputs:", args.output_dir)
    print("DONE")


if __name__ == "__main__":
    main()
