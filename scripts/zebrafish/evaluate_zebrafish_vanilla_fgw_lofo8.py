#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

try:
    import ot
except ImportError as e:
    raise ImportError("POT is required: pip install POT") from e


EXPECTED_Q = {
    1: 3536,
    2: 3954,
    3: 768,
    4: 1820,
    5: 1464,
    6: 938,
    7: 2492,
    8: 2184,
}

ALPHAS = [0.0, 0.25, 0.50, 0.75, 1.0]

INVALID_IDS = {
    "", "-1", "nan", "none", "null",
    "na", "n/a", "unknown",
    "unlabeled", "unlabelled", "invalid",
}


def scalar_string(x):
    a = np.asarray(x)
    if a.ndim == 0:
        x = a.item()
    elif a.size == 1:
        x = a.reshape(-1)[0]
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def mask_or_true(z, key, n):
    if key not in z:
        return np.ones(n, dtype=bool)
    x = np.asarray(z[key], dtype=bool).reshape(-1)
    if len(x) != n:
        raise RuntimeError(f"{key}: {len(x)} != {n}")
    return x


def clean_id(x):
    s = str(x).strip()
    return "" if s.lower() in INVALID_IDS else s


def load_side(path):
    with np.load(path, allow_pickle=True) as z:
        xyz = np.asarray(z["xyz"], dtype=np.float64)
        ids = np.asarray(z["cell_id"]).reshape(-1)

        n = len(ids)
        if xyz.shape != (n, 3):
            raise RuntimeError(f"{path}: bad xyz {xyz.shape}")

        candidate = (
            np.isfinite(xyz).all(axis=1)
            & mask_or_true(z, "valid_xyz_mask", n)
        )

        supervised = (
            candidate
            & mask_or_true(z, "labeled_mask", n)
            & mask_or_true(z, "certain_mask", n)
            & mask_or_true(z, "clean_mask", n)
        )

        pair_id = scalar_string(z["pair_id"])
        side = scalar_string(z["side"]).lower()

    if side not in {"q", "r"}:
        raise RuntimeError(f"{path}: invalid side={side}")

    # Full candidate population enters FGW.
    return {
        "path": str(path),
        "pair_id": pair_id,
        "side": side,
        "xyz_raw": xyz[candidate],
        "ids": ids[candidate],
        "supervised": supervised[candidate],
    }


def load_pairs(fold_root, split):
    paths = sorted((fold_root / split).rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"No NPZ under {fold_root / split}")

    grouped = {}

    for path in paths:
        s = load_side(path)
        pid = s["pair_id"]
        grouped.setdefault(pid, {})

        if s["side"] in grouped[pid]:
            raise RuntimeError(f"duplicate {pid}/{s['side']}")

        grouped[pid][s["side"]] = s

    pairs = []

    for pid in sorted(grouped):
        g = grouped[pid]
        if set(g) != {"q", "r"}:
            raise RuntimeError(f"incomplete pair {pid}: {sorted(g)}")

        pairs.append({
            "pair_id": pid,
            "q": g["q"],
            "r": g["r"],
        })

    return pairs


def fit_train_norm(train_pairs):
    xyz = np.concatenate(
        [
            p[s]["xyz_raw"]
            for p in train_pairs
            for s in ("q", "r")
        ],
        axis=0,
    )

    mean = xyz.mean(axis=0)
    std = xyz.std(axis=0)
    std = np.maximum(std, 1e-6)

    return mean, std


def normalize_pairs(pairs, mean, std):
    out = []

    for p in pairs:
        q = dict(p["q"])
        r = dict(p["r"])

        q["xyz"] = (q["xyz_raw"] - mean) / std
        r["xyz"] = (r["xyz_raw"] - mean) / std

        out.append({
            "pair_id": p["pair_id"],
            "q": q,
            "r": r,
        })

    return out


def unique_supervised_map(side):
    valid_ids = [
        clean_id(cid)
        for cid, ok in zip(
            side["ids"],
            side["supervised"],
        )
        if ok and clean_id(cid)
    ]

    counts = Counter(valid_ids)

    result = {}

    for i, (cid, ok) in enumerate(
        zip(side["ids"], side["supervised"])
    ):
        cid = clean_id(cid)

        if ok and cid and counts[cid] == 1:
            result[cid] = i

    return result


def normalize_cost(x):
    x = np.asarray(x, dtype=np.float64)

    maximum = float(np.max(x))
    if maximum > 1e-12:
        x = x / maximum

    return x


def fgw_score(qxyz, rxyz, alpha):
    """
    Standard geometry-only fused GW.

    M  : cross-population absolute geometry cost
    Cq : query within-population geometry
    Cr : reference within-population geometry

    Output T is the POT transport coupling; larger = better match.
    """

    # Feature cost.
    M = cdist(
        qxyz,
        rxyz,
        metric="sqeuclidean",
    )
    M = normalize_cost(M)

    # Structural costs.
    Cq = cdist(
        qxyz,
        qxyz,
        metric="euclidean",
    )

    Cr = cdist(
        rxyz,
        rxyz,
        metric="euclidean",
    )

    Cq = normalize_cost(Cq)
    Cr = normalize_cost(Cr)

    nq = len(qxyz)
    nr = len(rxyz)

    p = np.full(
        nq,
        1.0 / nq,
        dtype=np.float64,
    )
    q = np.full(
        nr,
        1.0 / nr,
        dtype=np.float64,
    )

    T = ot.gromov.fused_gromov_wasserstein(
        M,
        Cq,
        Cr,
        p,
        q,
        loss_fun="square_loss",
        alpha=float(alpha),
        armijo=False,
        log=False,
        max_iter=200,
        tol_rel=1e-9,
        tol_abs=1e-9,
    )

    T = np.asarray(T, dtype=np.float64)

    if T.shape != (nq, nr):
        raise RuntimeError(
            f"FGW returned {T.shape}, expected {(nq, nr)}"
        )

    if not np.isfinite(T).all():
        raise RuntimeError("FGW produced non-finite coupling")

    return T


def directional(score, query_map, ref_map):
    common = sorted(set(query_map) & set(ref_map))

    q = c1 = c5 = 0
    rr = 0.0

    for cid in common:
        i = query_map[cid]
        j = ref_map[cid]

        gt = score[i, j]

        # Same optimistic-tie definition as MPRT benchmark.
        rank = 1 + int(np.sum(score[i] > gt))

        q += 1
        c1 += int(rank == 1)
        c5 += int(rank <= min(5, score.shape[1]))
        rr += 1.0 / rank

    return {
        "queries": q,
        "correct1": c1,
        "correct5": c5,
        "rr": rr,
        "common": common,
    }


def evaluate(pairs, alpha):
    total_q = 0
    c1 = c5 = 0
    rr = 0.0
    hung_correct = 0

    pair_rows = []

    for pair in pairs:
        qs = pair["q"]
        rs = pair["r"]

        # Exactly one transport plan for physical pair.
        T = fgw_score(
            qs["xyz"],
            rs["xyz"],
            alpha,
        )

        qm = unique_supervised_map(qs)
        rm = unique_supervised_map(rs)

        qr = directional(T, qm, rm)
        rq = directional(T.T, rm, qm)

        if qr["queries"] != rq["queries"]:
            raise RuntimeError(
                f"{pair['pair_id']}: directional Q mismatch"
            )

        pair_q = qr["queries"] + rq["queries"]

        pair_c1 = (
            qr["correct1"]
            + rq["correct1"]
        )

        pair_c5 = (
            qr["correct5"]
            + rq["correct5"]
        )

        pair_rr = qr["rr"] + rq["rr"]

        # One Hungarian assignment on the same T.
        rows, cols = linear_sum_assignment(-T)

        assignment = {
            int(i): int(j)
            for i, j in zip(rows, cols)
        }

        common = sorted(set(qm) & set(rm))

        one_dir_hits = sum(
            int(
                assignment.get(qm[cid], -1)
                == rm[cid]
            )
            for cid in common
        )

        # Count against bidirectional query denominator.
        pair_hung = 2 * one_dir_hits

        total_q += pair_q
        c1 += pair_c1
        c5 += pair_c5
        rr += pair_rr
        hung_correct += pair_hung

        pair_rows.append({
            "pair_id": pair["pair_id"],
            "q_candidates": len(qs["xyz"]),
            "r_candidates": len(rs["xyz"]),
            "queries": pair_q,
            "top1": pair_c1 / max(pair_q, 1),
            "top5": pair_c5 / max(pair_q, 1),
            "mrr": pair_rr / max(pair_q, 1),
            "hungarian": pair_hung / max(pair_q, 1),
        })

    if total_q == 0:
        raise RuntimeError("No valid FGW queries")

    return {
        "pairs": len(pairs),
        "queries": total_q,
        "top1": c1 / total_q,
        "top5": c5 / total_q,
        "mrr": rr / total_q,
        "hungarian": hung_correct / total_q,
        "pair_results": pair_rows,
    }


def choose_alpha(val_pairs):
    results = []

    for alpha in ALPHAS:
        m = evaluate(val_pairs, alpha)

        print(
            f"  alpha={alpha:.2f} "
            f"Q={m['queries']:4d} "
            f"Top1={100*m['top1']:6.2f}% "
            f"Top5={100*m['top5']:6.2f}% "
            f"MRR={m['mrr']:.4f} "
            f"Hung={100*m['hungarian']:6.2f}%"
        )

        results.append({
            "alpha": alpha,
            "metrics": m,
        })

    # Primary selection = val Top1.
    # Secondary only resolves exact ties.
    results.sort(
        key=lambda x: (
            -x["metrics"]["top1"],
            -x["metrics"]["mrr"],
            -x["metrics"]["hungarian"],
            abs(x["alpha"] - 0.5),
            x["alpha"],
        )
    )

    return results[0], results


def mean_sd(xs):
    x = np.asarray(xs, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/"
            "Data/Zebrafish_MPRT_LOFO8_60m"
        ),
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/home/ubuntu/klb/nuclr/nuclr/"
            "runs/zebrafish_vanilla_fgw_lofo8"
        ),
    )

    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    fold_results = []

    print("=" * 100)
    print("VANILLA FGW (POT OFFICIAL) — ZEBRAFISH LOFO8")
    print("=" * 100)

    for fold in range(1, 9):
        fold_root = args.data_root / f"fold_{fold}"
        out = args.out / f"fold_{fold}"
        out.mkdir(parents=True, exist_ok=True)

        # Test remains unread during normalization + alpha selection.
        train_raw = load_pairs(fold_root, "train")
        val_raw = load_pairs(fold_root, "val")

        mean, std = fit_train_norm(train_raw)

        train = normalize_pairs(
            train_raw,
            mean,
            std,
        )

        val = normalize_pairs(
            val_raw,
            mean,
            std,
        )

        print()
        print("-" * 100)
        print(
            f"FOLD {fold}: "
            f"train_pairs={len(train)} "
            f"val_pairs={len(val)} "
            f"test=UNREAD"
        )

        selected, val_grid = choose_alpha(val)
        alpha = float(selected["alpha"])

        lock = {
            "status": "LOCKED_BEFORE_TEST",
            "method": "Vanilla FGW (POT)",
            "fold": fold,
            "deterministic": True,
            "input": "geometry-only",
            "normalization": "train-only per-axis z-score",
            "feature_cost": "normalized squared Euclidean cross-population XYZ",
            "structure_cost": "normalized within-population Euclidean distance",
            "alpha_grid": ALPHAS,
            "selected_alpha": alpha,
            "selection_metric": "validation Top1; MRR/Hungarian exact-tie break",
            "POT_version": getattr(ot, "__version__", "unknown"),
            "val_grid": val_grid,
            "train_mean": mean.tolist(),
            "train_std": std.tolist(),
        }

        (
            out / "LOCKED_BEFORE_TEST.json"
        ).write_text(
            json.dumps(lock, indent=2) + "\n"
        )

        # --------------------------------------------------
        # FIRST TEST ACCESS
        # --------------------------------------------------
        test_raw = load_pairs(fold_root, "test")
        test = normalize_pairs(
            test_raw,
            mean,
            std,
        )

        metrics = evaluate(test, alpha)

        expected = EXPECTED_Q[fold]

        if metrics["queries"] != expected:
            raise RuntimeError(
                f"fold{fold}: QUERY AUDIT FAILED: "
                f"{metrics['queries']} != {expected}"
            )

        result = {
            "fold": fold,
            "selected_alpha": alpha,
            "expected_queries": expected,
            "query_audit_passed": True,
            "test": metrics,
        }

        (
            out / "summary.json"
        ).write_text(
            json.dumps(result, indent=2) + "\n"
        )

        fold_results.append(result)

        print(
            f"LOCKED TEST fold{fold}: "
            f"alpha={alpha:.2f} "
            f"pairs={metrics['pairs']:2d} "
            f"Q={metrics['queries']:4d} "
            f"Top1={100*metrics['top1']:6.2f}% "
            f"Top5={100*metrics['top5']:6.2f}% "
            f"MRR={metrics['mrr']:.4f} "
            f"Hung={100*metrics['hungarian']:6.2f}%"
        )

    aggregate = {
        "method": "Vanilla FGW (POT)",
        "protocol": "Zebrafish LOFO8",
        "deterministic": True,
        "folds": 8,
        "selected_alpha": [
            r["selected_alpha"]
            for r in fold_results
        ],
        "top1": mean_sd([
            r["test"]["top1"]
            for r in fold_results
        ]),
        "top5": mean_sd([
            r["test"]["top5"]
            for r in fold_results
        ]),
        "mrr": mean_sd([
            r["test"]["mrr"]
            for r in fold_results
        ]),
        "hungarian": mean_sd([
            r["test"]["hungarian"]
            for r in fold_results
        ]),
        "fold_results": fold_results,
    }

    (
        args.out / "fgw_lofo8_aggregate.json"
    ).write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )

    print()
    print("=" * 100)
    print("FINAL — MEAN ± SAMPLE SD ACROSS 8 HELD-OUT FISH")
    print("=" * 100)

    for key, name, pct in [
        ("top1", "Top-1", True),
        ("top5", "Top-5", True),
        ("mrr", "MRR", False),
        ("hungarian", "Hungarian", True),
    ]:
        m = aggregate[key]["mean"]
        s = aggregate[key]["sd"]

        if pct:
            print(
                f"{name:<10s}: "
                f"{100*m:.4f} ± {100*s:.4f}%"
            )
        else:
            print(
                f"{name:<10s}: "
                f"{m:.4f} ± {s:.4f}"
            )

    print()
    print(
        "selected alpha:",
        aggregate["selected_alpha"],
    )

    print(
        "summary:",
        args.out / "fgw_lofo8_aggregate.json",
    )

    print("AUDIT PASSED")


if __name__ == "__main__":
    main()
