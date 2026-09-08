#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

import numpy as np
import ot
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class Worm:
    path: Path
    xyz: np.ndarray
    labels: np.ndarray
    valid_label_mask: np.ndarray


def load_worm(path: Path) -> Worm:
    with np.load(path, allow_pickle=True) as z:
        xyz = np.asarray(z["xyz"], dtype=np.float64)
        labels = np.asarray(z["cell_id"]).reshape(-1).astype(str)

        # RLD strict supervision/evaluation protocol:
        # clean_mask is authoritative.
        if "clean_mask" in z.files:
            mask = np.asarray(z["clean_mask"], dtype=bool).reshape(-1)
        elif "labeled_mask" in z.files:
            mask = np.asarray(z["labeled_mask"], dtype=bool).reshape(-1)
        else:
            mask = np.ones(len(labels), dtype=bool)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise RuntimeError(f"{path}: xyz={xyz.shape}")

    if len(xyz) != len(labels) or len(mask) != len(labels):
        raise RuntimeError(f"{path}: xyz/label/mask mismatch")

    finite = np.isfinite(xyz).all(axis=1)

    xyz = xyz[finite]
    labels = labels[finite]
    mask = mask[finite]

    nonempty = np.asarray([
        x.strip() not in {"", "nan", "None", "-1"}
        for x in labels
    ])

    mask &= nonempty

    return Worm(
        path=path,
        xyz=xyz,
        labels=labels,
        valid_label_mask=mask,
    )


def unique_label_map(w: Worm):
    idx = np.flatnonzero(w.valid_label_mask)

    counts = Counter(str(w.labels[i]) for i in idx)

    return {
        str(w.labels[i]): int(i)
        for i in idx
        if counts[str(w.labels[i])] == 1
    }


def fit_scaler(train):
    x = np.concatenate(
        [w.xyz for w in train],
        axis=0,
    )

    mean = x.mean(axis=0)
    std = x.std(axis=0)

    std[std < 1e-8] = 1.0

    return mean, std


def norm_xyz(w, mean, std):
    return (
        (w.xyz - mean[None, :])
        / std[None, :]
    ).astype(np.float64)


def symmetric_chamfer(a, b):
    D = cdist(a, b, metric="sqeuclidean")

    return 0.5 * (
        D.min(axis=1).mean()
        +
        D.min(axis=0).mean()
    )


def select_train_medoid(train, mean, std):
    xs = [
        norm_xyz(w, mean, std)
        for w in train
    ]

    n = len(xs)

    means = []

    for i in range(n):
        d = []

        for j in range(n):
            if i == j:
                continue

            d.append(
                symmetric_chamfer(
                    xs[i],
                    xs[j],
                )
            )

        means.append(float(np.mean(d)))

    best = int(np.argmin(means))

    return train[best], means


def normalize_cost(C):
    C = np.asarray(C, dtype=np.float64)

    m = float(np.max(C))

    if np.isfinite(m) and m > 1e-12:
        C = C / m

    return C


def fgw_plan(
    xyz1,
    xyz2,
    alpha,
):
    # Node-feature cross-domain cost.
    M = cdist(
        xyz1,
        xyz2,
        metric="sqeuclidean",
    )

    # Within-graph structural costs.
    C1 = cdist(
        xyz1,
        xyz1,
        metric="sqeuclidean",
    )

    C2 = cdist(
        xyz2,
        xyz2,
        metric="sqeuclidean",
    )

    # Put linear and structural terms on comparable scales.
    M = normalize_cost(M)
    C1 = normalize_cost(C1)
    C2 = normalize_cost(C2)

    p = ot.unif(len(xyz1))
    q = ot.unif(len(xyz2))

    T = ot.gromov.fused_gromov_wasserstein(
        M,
        C1,
        C2,
        p=p,
        q=q,
        loss_fun="square_loss",
        alpha=float(alpha),
        armijo=False,
        symmetric=True,
        log=False,
        max_iter=10000,
        tol_rel=1e-9,
        tol_abs=1e-9,
    )

    T = np.asarray(T, dtype=np.float64)

    if T.shape != (len(xyz1), len(xyz2)):
        raise RuntimeError(
            f"Bad transport shape: {T.shape}"
        )

    if not np.isfinite(T).all():
        raise RuntimeError("Non-finite FGW plan")

    return T


def rank_of_gt(scores, gt):
    order = np.argsort(-scores, kind="stable")

    hit = np.flatnonzero(order == gt)

    if len(hit) != 1:
        raise RuntimeError("GT ranking failure")

    return int(hit[0]) + 1


def evaluate(
    worms,
    template,
    mean,
    std,
    alpha,
):
    ref_xyz = norm_xyz(
        template,
        mean,
        std,
    )

    ref_map = unique_label_map(template)

    total_valid = 0
    queries = 0

    correct1 = 0
    correct5 = 0
    reciprocal = 0.0

    hung_correct = 0

    for w in worms:
        q_xyz = norm_xyz(
            w,
            mean,
            std,
        )

        qmap = unique_label_map(w)

        total_valid += len(qmap)

        common = sorted(
            set(qmap)
            &
            set(ref_map)
        )

        if not common:
            continue

        T = fgw_plan(
            q_xyz,
            ref_xyz,
            alpha,
        )

        # Direct ranking.
        for label in common:
            qi = qmap[label]
            gi = ref_map[label]

            rank = rank_of_gt(
                T[qi],
                gi,
            )

            queries += 1
            correct1 += int(rank == 1)
            correct5 += int(rank <= 5)
            reciprocal += 1.0 / rank

        # Hungarian on full graph, including distractors.
        row, col = linear_sum_assignment(-T)

        assignment = {
            int(r): int(c)
            for r, c in zip(row, col)
        }

        for label in common:
            qi = qmap[label]
            gi = ref_map[label]

            hung_correct += int(
                assignment.get(qi, -1)
                == gi
            )

    if queries == 0:
        raise RuntimeError("No evaluable queries")

    return {
        "queries": int(queries),
        "valid_queries_before_coverage": int(total_valid),
        "top1": float(correct1 / queries),
        "top5": float(correct5 / queries),
        "mrr": float(reciprocal / queries),
        "hungarian": float(hung_correct / queries),
        "coverage": float(
            queries / max(total_valid, 1)
        ),
    }


def sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for x in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(x)

    return h.hexdigest()


def get_split(root, fold, split):
    d = (
        root
        / f"fold_{fold}"
        / split
    )

    paths = sorted(d.glob("*.npz"))

    if not paths:
        raise RuntimeError(
            f"No files: {d}"
        )

    return [
        load_worm(p)
        for p in paths
    ]


def fit_lock(args):
    # TEST IS NOT ACCESSED IN THIS FUNCTION.

    train = get_split(
        args.data_root,
        args.fold,
        "train",
    )

    val = get_split(
        args.data_root,
        args.fold,
        "val",
    )

    mean, std = fit_scaler(train)

    if args.template_json is not None:
        spec = json.loads(args.template_json.read_text())
        template_name = str(spec["template"])

        matches = [
            w for w in train
            if w.path.name == template_name
        ]

        if len(matches) != 1:
            raise RuntimeError(
                f"Canonical template {template_name!r} "
                f"not uniquely found in TRAIN: {len(matches)} matches"
            )

        template = matches[0]
        medoid_scores = None

        print(
            "[CANONICAL TEMPLATE] "
            f"{template.path.name}"
        )
    else:
        template, medoid_scores = (
            select_train_medoid(
                train,
                mean,
                std,
            )
        )

    print("=" * 90)
    print("POT OFFICIAL FGW — FIT/VAL ONLY")
    print("=" * 90)
    print("POT version :", ot.__version__)
    print("fold        :", args.fold)
    print("train worms :", len(train))
    print("val worms   :", len(val))
    print("template    :", template.path.name)
    print()

    rows = []

    for alpha in args.alphas:
        m = evaluate(
            val,
            template,
            mean,
            std,
            alpha,
        )

        rows.append(
            (alpha, m)
        )

        print(
            f"alpha={alpha:.2f} "
            f"Q={m['queries']:4d} "
            f"Top1={100*m['top1']:6.2f}% "
            f"Top5={100*m['top5']:6.2f}% "
            f"MRR={m['mrr']:.4f} "
            f"Hung={100*m['hungarian']:6.2f}% "
            f"Cov={100*m['coverage']:6.2f}%"
        )

    # Same lock convention as learned baselines:
    # validation Hungarian primary, Top1 tie-break.
    best_alpha, best = max(
        rows,
        key=lambda x: (
            x[1]["hungarian"],
            x[1]["top1"],
        ),
    )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    lock = {
        "status": "LOCKED_BEFORE_TEST",
        "method": "FGW (POT official)",
        "pot_version": str(ot.__version__),
        "data_root": str(args.data_root.resolve()),
        "fold": int(args.fold),
        "split_counts_before_test": {
            "train": len(train),
            "val": len(val),
        },
        "alpha_grid": [
            float(x)
            for x in args.alphas
        ],
        "selected_alpha": float(best_alpha),
        "selection_metric":
            "validation Hungarian then Top1",
        "val_metrics": best,
        "template": template.path.name,
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "structure_cost":
            "normalized squared Euclidean",
        "feature_cost":
            "normalized squared Euclidean XYZ",
        "loss_fun": "square_loss",
        "armijo": False,
        "max_iter": 10000,
    }

    p = args.out / "LOCKED_BEFORE_TEST.json"

    p.write_text(
        json.dumps(lock, indent=2)
        + "\n"
    )

    print()
    print(
        f"[BEST] alpha={best_alpha:.2f} "
        f"VAL Top1={100*best['top1']:.2f}% "
        f"Hung={100*best['hungarian']:.2f}%"
    )
    print("[LOCK] TEST WAS NOT READ")
    print("lock:", p)


def locked_test(args):
    lock_path = (
        args.out
        / "LOCKED_BEFORE_TEST.json"
    )

    if not lock_path.is_file():
        raise RuntimeError(
            f"Missing lock: {lock_path}"
        )

    lock = json.loads(
        lock_path.read_text()
    )

    if lock["status"] != "LOCKED_BEFORE_TEST":
        raise RuntimeError("Invalid lock")

    if int(lock["fold"]) != args.fold:
        raise RuntimeError("Fold mismatch")

    if Path(lock.get("data_root", "")).resolve() != args.data_root.resolve():
        raise RuntimeError("Data-root mismatch")

    mean = np.asarray(
        lock["train_mean"],
        dtype=np.float64,
    )

    std = np.asarray(
        lock["train_std"],
        dtype=np.float64,
    )

    alpha = float(
        lock["selected_alpha"]
    )

    # Locate template inside TRAIN only.
    train_dir = (
        args.data_root
        / f"fold_{args.fold}"
        / "train"
    )

    template_path = (
        train_dir
        / lock["template"]
    )

    if not template_path.is_file():
        found = list(
            train_dir.rglob(
                lock["template"]
            )
        )

        if len(found) != 1:
            raise RuntimeError(
                "Cannot uniquely locate "
                "locked train template"
            )

        template_path = found[0]

    template = load_worm(
        template_path
    )

    print("=" * 90)
    print("LOCK CONFIRMED — NOW READING TEST")
    print("=" * 90)
    print(
        f"fold={args.fold} "
        f"alpha={alpha:.2f} "
        f"template={template.path.name}"
    )

    # FIRST TEST ACCESS.
    test = get_split(
        args.data_root,
        args.fold,
        "test",
    )

    m = evaluate(
        test,
        template,
        mean,
        std,
        alpha,
    )

    result = {
        **m,
        "method": "FGW (POT official)",
        "pot_version": str(ot.__version__),
        "data_root": str(args.data_root.resolve()),
        "fold": int(args.fold),
        "split_counts": {
            **lock["split_counts_before_test"],
            "test": len(test),
        },
        "alpha": alpha,
        "template": template.path.name,
        "lock_sha256": sha256(lock_path),
    }

    out = args.out / "test_metrics.json"

    out.write_text(
        json.dumps(
            result,
            indent=2,
        ) + "\n"
    )

    print()
    print("=" * 90)
    print(
        f"FGW (POT OFFICIAL) — "
        f"RLD FOLD{args.fold} "
        f"— LOCKED TEST"
    )
    print("=" * 90)
    print(f"Queries     : {m['queries']}")
    print(f"Top-1       : {100*m['top1']:.2f}%")
    print(f"Top-5       : {100*m['top5']:.2f}%")
    print(f"MRR         : {m['mrr']:.4f}")
    print(f"Hungarian   : {100*m['hungarian']:.2f}%")
    print(f"Coverage    : {100*m['coverage']:.2f}%")
    print(f"alpha       : {alpha:.2f}")
    print("=" * 90)
    print("metrics:", out)


def main():
    parser = argparse.ArgumentParser()

    sub = parser.add_subparsers(
        dest="command",
        required=True,
    )

    for name in ["fit-lock", "locked-test"]:
        p = sub.add_parser(name)

        p.add_argument(
            "--data-root",
            type=Path,
            required=True,
        )

        p.add_argument(
            "--fold",
            type=int,
            required=True,
        )

        p.add_argument(
            "--out",
            type=Path,
            required=True,
        )

        if name == "fit-lock":
            p.add_argument(
                "--alphas",
                type=float,
                nargs="+",
                default=[
                    0.25,
                    0.50,
                    0.75,
                ],
            )

            p.add_argument(
                "--template-json",
                type=Path,
                default=None,
            )

    args = parser.parse_args()

    if args.command == "fit-lock":
        fit_lock(args)
    else:
        locked_test(args)


if __name__ == "__main__":
    main()
