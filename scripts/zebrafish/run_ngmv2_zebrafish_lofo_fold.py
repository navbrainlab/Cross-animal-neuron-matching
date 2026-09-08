#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Reuse the already-audited NGM-v2 adaptation.
from scripts.benchmarks import run_ngmv2_atanas_fold as base


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

INVALID_IDS = {
    "",
    "-1",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "unknown",
    "unlabeled",
    "unlabelled",
    "invalid",
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_string(x) -> str:
    a = np.asarray(x)

    if a.ndim == 0:
        x = a.item()
    elif a.size == 1:
        x = a.reshape(-1)[0]

    if isinstance(x, bytes):
        x = x.decode()

    return str(x)


def mask_or_true(z, key: str, n: int):
    if key not in z:
        return np.ones(n, dtype=bool)

    x = np.asarray(
        z[key],
        dtype=bool,
    ).reshape(-1)

    if len(x) != n:
        raise RuntimeError(
            f"{key}: {len(x)} != {n}"
        )

    return x


def load_side(path: Path) -> dict:
    with np.load(
        path,
        allow_pickle=True,
    ) as z:

        xyz = np.asarray(
            z["xyz"],
            dtype=np.float32,
        )

        cell_id = np.asarray(
            z["cell_id"]
        ).reshape(-1)

        n = len(cell_id)

        if xyz.shape != (n, 3):
            raise RuntimeError(
                f"{path}: bad xyz shape "
                f"{xyz.shape}, N={n}"
            )

        finite = np.isfinite(
            xyz
        ).all(axis=1)

        valid_xyz = mask_or_true(
            z,
            "valid_xyz_mask",
            n,
        )

        labeled = mask_or_true(
            z,
            "labeled_mask",
            n,
        )

        certain = mask_or_true(
            z,
            "certain_mask",
            n,
        )

        clean = mask_or_true(
            z,
            "clean_mask",
            n,
        )

        candidate = (
            finite
            & valid_xyz
        )

        supervised = (
            candidate
            & labeled
            & certain
            & clean
        )

        pair_id = scalar_string(
            z["pair_id"]
        )

        side = scalar_string(
            z["side"]
        ).lower()

    if side not in {"q", "r"}:
        raise RuntimeError(
            f"{path}: side={side}"
        )

    # Full candidate population remains in graph.
    xyz = xyz[candidate]
    cell_id = cell_id[candidate]
    supervised = supervised[candidate]

    return {
        "path": str(path.resolve()),
        "name": path.name,
        "pair_id": pair_id,
        "side": side,
        "xyz_raw": xyz.astype(
            np.float32
        ),
        "xyz": None,
        "cell_id": cell_id,
        "mask": supervised.astype(bool),
    }


def load_physical_pairs(
    fold_root: Path,
    split: str,
):
    root = fold_root / split

    paths = sorted(
        root.rglob("*.npz")
    )

    if not paths:
        raise RuntimeError(
            f"No NPZ under {root}"
        )

    grouped = {}

    for path in paths:
        x = load_side(path)
        pid = x["pair_id"]
        side = x["side"]

        grouped.setdefault(
            pid,
            {},
        )

        if side in grouped[pid]:
            raise RuntimeError(
                f"duplicate {pid}/{side}"
            )

        grouped[pid][side] = x

    pairs = []

    for pid in sorted(grouped):
        g = grouped[pid]

        if set(g) != {"q", "r"}:
            raise RuntimeError(
                f"incomplete pair {pid}: "
                f"{sorted(g)}"
            )

        pairs.append({
            "pair_id": pid,
            "q": g["q"],
            "r": g["r"],
        })

    return pairs


def fit_train_norm(pairs):
    arrays = []

    for p in pairs:
        arrays.append(
            p["q"]["xyz_raw"]
        )
        arrays.append(
            p["r"]["xyz_raw"]
        )

    x = np.concatenate(
        arrays,
        axis=0,
    ).astype(np.float64)

    mean = x.mean(axis=0)
    std = x.std(axis=0)

    std = np.maximum(
        std,
        1e-6,
    )

    return (
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def normalize_side(
    side,
    mean,
    std,
):
    x = dict(side)

    x["xyz"] = (
        (
            side["xyz_raw"]
            - mean[None]
        )
        / std[None]
    ).astype(np.float32)

    return x


def normalize_pairs(
    pairs,
    mean,
    std,
):
    out = []

    for p in pairs:
        out.append({
            "pair_id": p["pair_id"],
            "q": normalize_side(
                p["q"],
                mean,
                std,
            ),
            "r": normalize_side(
                p["r"],
                mean,
                std,
            ),
        })

    return out


def clean_id(value) -> str:
    x = str(value).strip()

    if x.lower() in INVALID_IDS:
        return ""

    return x


def unique_supervised_map(side):
    ids = []

    for cid, ok in zip(
        side["cell_id"],
        side["mask"],
    ):
        cid = clean_id(cid)

        if ok and cid:
            ids.append(cid)

    counts = Counter(ids)

    result = {}

    for i, (cid, ok) in enumerate(
        zip(
            side["cell_id"],
            side["mask"],
        )
    ):
        cid = clean_id(cid)

        if (
            ok
            and cid
            and counts[cid] == 1
        ):
            result[cid] = i

    return result


def training_intersection(
    side1,
    side2,
    min_common,
):
    """
    Preserve old NGM-v2 training protocol:
    train only on common unique supervised IDs.

    Evaluation is full candidate graph.
    """

    m1 = unique_supervised_map(
        side1
    )

    m2 = unique_supervised_map(
        side2
    )

    common = sorted(
        set(m1)
        & set(m2)
    )

    if len(common) < min_common:
        return None

    idx1 = np.asarray(
        [m1[x] for x in common],
        dtype=np.int64,
    )

    idx2 = np.asarray(
        [m2[x] for x in common],
        dtype=np.int64,
    )

    xyz1 = side1["xyz"][idx1]
    xyz2 = side2["xyz"][idx2]

    gt = np.eye(
        len(common),
        dtype=np.float32,
    )

    return (
        xyz1,
        xyz2,
        gt,
        common,
    )


def build_training_orientations(
    pairs,
    min_common,
):
    """
    Include q->r and r->q so training matches
    bidirectional evaluation.
    """

    usable = []

    for pi, p in enumerate(pairs):

        for a, b, direction in [
            ("q", "r", "q2r"),
            ("r", "q", "r2q"),
        ]:
            item = training_intersection(
                p[a],
                p[b],
                min_common,
            )

            if item is not None:
                usable.append(
                    (
                        pi,
                        a,
                        b,
                        direction,
                    )
                )

    return usable


def parse_original_scheduler():
    """
    Read exact MultiStepLR settings from the existing
    audited Atanas NGM-v2 runner, instead of guessing them.
    """

    text = Path(
        base.__file__
    ).read_text(
        encoding="utf-8"
    )

    pos = text.find(
        "MultiStepLR"
    )

    if pos < 0:
        raise RuntimeError(
            "Cannot find MultiStepLR "
            "in original runner"
        )

    block = text[
        pos : pos + 1200
    ]

    mm = re.search(
        r"milestones\s*=\s*\["
        r"([^\]]+)\]",
        block,
        re.S,
    )

    gm = re.search(
        r"gamma\s*=\s*"
        r"([0-9eE.+\-]+)",
        block,
    )

    if mm is None or gm is None:
        raise RuntimeError(
            "Could not parse original "
            "scheduler settings.\n"
            + block
        )

    milestones = [
        int(x)
        for x in re.findall(
            r"\d+",
            mm.group(1),
        )
    ]

    gamma = float(
        gm.group(1)
    )

    if not milestones:
        raise RuntimeError(
            "Empty scheduler milestones"
        )

    return milestones, gamma


def directional_metrics(
    score,
    query_map,
    ref_map,
):
    common = sorted(
        set(query_map)
        & set(ref_map)
    )

    q = 0
    c1 = 0
    c5 = 0
    rr = 0.0

    for cid in common:
        i = query_map[cid]
        j = ref_map[cid]

        gt_score = float(
            score[i, j]
        )

        # Locked optimistic-tie rank.
        rank = (
            1
            + int(
                np.sum(
                    score[i]
                    > gt_score
                )
            )
        )

        q += 1
        c1 += int(rank == 1)
        c5 += int(
            rank
            <= min(
                5,
                score.shape[1],
            )
        )
        rr += 1.0 / rank

    return {
        "queries": q,
        "top1_correct": c1,
        "top5_correct": c5,
        "rr_sum": rr,
        "common": common,
    }


@torch.no_grad()
def evaluate_pairs(
    model,
    pairs,
    device,
):
    """
    One NGM-v2 Sinkhorn matrix per physical pair.

    Same S is used for:
      q -> r ranking
      r -> q ranking using S.T
      one Hungarian assignment
    """

    model.eval()

    total_q = 0
    c1 = 0
    c5 = 0
    rr = 0.0
    hc = 0

    pair_results = []

    for pair_index, p in enumerate(
        pairs
    ):
        qside = p["q"]
        rside = p["r"]

        graph_q = base.make_graph(
            qside["xyz"],
            device,
        )

        graph_r = base.make_graph(
            rside["xyz"],
            device,
        )

        ds = model(
            graph_q,
            graph_r,
        )[0]

        score = (
            ds.detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        if score.shape != (
            len(qside["xyz"]),
            len(rside["xyz"]),
        ):
            raise RuntimeError(
                f"{p['pair_id']}: "
                f"score={score.shape}, "
                f"expected="
                f"{len(qside['xyz'])}x"
                f"{len(rside['xyz'])}"
            )

        qmap = unique_supervised_map(
            qside
        )

        rmap = unique_supervised_map(
            rside
        )

        qr = directional_metrics(
            score,
            qmap,
            rmap,
        )

        rq = directional_metrics(
            score.T,
            rmap,
            qmap,
        )

        if qr["queries"] != rq["queries"]:
            raise RuntimeError(
                f"{p['pair_id']}: "
                "directional query mismatch "
                f"{qr['queries']} vs "
                f"{rq['queries']}"
            )

        pair_q = (
            qr["queries"]
            + rq["queries"]
        )

        pair_c1 = (
            qr["top1_correct"]
            + rq["top1_correct"]
        )

        pair_c5 = (
            qr["top5_correct"]
            + rq["top5_correct"]
        )

        pair_rr = (
            qr["rr_sum"]
            + rq["rr_sum"]
        )

        # One physical Hungarian assignment.
        row_ind, col_ind = (
            linear_sum_assignment(
                -score
            )
        )

        assignment = {
            int(i): int(j)
            for i, j in zip(
                row_ind,
                col_ind,
            )
        }

        pair_hc_one_direction = 0

        common = sorted(
            set(qmap)
            & set(rmap)
        )

        for cid in common:
            qi = qmap[cid]
            ri = rmap[cid]

            pair_hc_one_direction += int(
                assignment.get(
                    qi,
                    -1,
                )
                == ri
            )

        # Same physical correct correspondence
        # contributes to q->r and r->q benchmark queries.
        pair_hc = (
            2
            * pair_hc_one_direction
        )

        total_q += pair_q
        c1 += pair_c1
        c5 += pair_c5
        rr += pair_rr
        hc += pair_hc

        pair_results.append({
            "pair_index": pair_index,
            "pair_id":
                p["pair_id"],
            "q_candidates":
                len(qside["xyz"]),
            "r_candidates":
                len(rside["xyz"]),
            "queries":
                pair_q,
            "q_to_r_queries":
                qr["queries"],
            "r_to_q_queries":
                rq["queries"],
            "top1":
                pair_c1
                / max(pair_q, 1),
            "top5":
                pair_c5
                / max(pair_q, 1),
            "mrr":
                pair_rr
                / max(pair_q, 1),
            "hungarian":
                pair_hc
                / max(pair_q, 1),
        })

    if total_q <= 0:
        raise RuntimeError(
            "No evaluation queries"
        )

    return {
        "pairs": len(pairs),
        "queries": total_q,
        "top1":
            c1 / total_q,
        "top5":
            c5 / total_q,
        "mrr":
            rr / total_q,
        "hungarian":
            hc / total_q,
        "coverage": 1.0,
        "pair_results":
            pair_results,
    }


def print_metrics(
    tag,
    m,
):
    print(
        f"{tag:<5s} "
        f"pairs={m['pairs']:2d} "
        f"Q={m['queries']:4d} "
        f"Top1="
        f"{100*m['top1']:6.2f}% "
        f"Top5="
        f"{100*m['top5']:6.2f}% "
        f"MRR={m['mrr']:.4f} "
        f"Hung="
        f"{100*m['hungarian']:6.2f}%"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 9),
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--device",
        default="cuda:0",
    )

    ap.add_argument(
        "--feature-dim",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--pairs-per-epoch",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--min-common",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=2e-3,
    )

    ap.add_argument(
        "--skip-test",
        action="store_true",
    )

    ap.add_argument(
        "--out",
        type=Path,
        required=True,
    )

    args = ap.parse_args()

    seed_everything(
        args.seed
    )

    device = torch.device(
        args.device
    )

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    fold_root = (
        args.data_root
        / f"fold_{args.fold}"
    )

    # =========================================================
    # PRE-TEST ONLY
    # =========================================================

    train_raw = (
        load_physical_pairs(
            fold_root,
            "train",
        )
    )

    val_raw = (
        load_physical_pairs(
            fold_root,
            "val",
        )
    )

    print("=" * 94)
    print(
        f"NGM-v2 — ZEBRAFISH "
        f"LOFO FOLD{args.fold} "
        f"SEED{args.seed}"
    )
    print("=" * 94)

    print(
        "train physical pairs:",
        len(train_raw),
    )

    print(
        "val physical pairs  :",
        len(val_raw),
    )

    print(
        "test                : UNREAD"
    )

    mean, std = fit_train_norm(
        train_raw
    )

    np.savez(
        args.out
        / "train_normalization.npz",
        mean=mean,
        std=std,
    )

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

    usable = (
        build_training_orientations(
            train,
            args.min_common,
        )
    )

    if not usable:
        raise RuntimeError(
            "No usable training pairs"
        )

    min_candidates = min(
        len(p[s]["xyz"])
        for p in train + val
        for s in ("q", "r")
    )

    print(
        "usable ordered train pairs:",
        len(usable),
    )

    print(
        "minimum candidate N:",
        min_candidates,
    )

    milestones, gamma = (
        parse_original_scheduler()
    )

    print(
        "feature_dim:",
        args.feature_dim,
    )

    print(
        "epochs:",
        args.epochs,
    )

    print(
        "pairs_per_epoch:",
        args.pairs_per_epoch,
    )

    print(
        "lr:",
        args.lr,
    )

    print(
        "scheduler milestones:",
        milestones,
    )

    print(
        "scheduler gamma:",
        gamma,
    )

    model = base.AtanasNGMv2(
        feature_dim=args.feature_dim
    ).to(device)

    criterion = (
        base.PermutationLoss()
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )
    )

    print(
        "parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    protocol = {
        "method":
            "NGM-v2 official core adapted",
        "dataset":
            "Zebrafish longitudinal LOFO8",
        "fold":
            args.fold,
        "seed":
            args.seed,
        "input":
            "geometry-only",
        "node_feature":
            "XYZ -> MLP",
        "graph":
            "3D Delaunay symmetric",
        "training_filter":
            "intersection of unique supervised identities",
        "training_pairs":
            "real physical q/r pairs, both orientations",
        "evaluation":
            "full candidate q/r graphs",
        "normalization":
            "train-only per-axis z-score",
        "feature_dim":
            args.feature_dim,
        "epochs":
            args.epochs,
        "pairs_per_epoch":
            args.pairs_per_epoch,
        "min_common":
            args.min_common,
        "lr":
            args.lr,
        "scheduler_milestones":
            milestones,
        "scheduler_gamma":
            gamma,
        "criterion":
            "PermutationLoss",
        "optimizer":
            "Adam",
        "grad_clip":
            5.0,
        "checkpoint_selection":
            "highest validation Top-1; exact tie -> earlier epoch",
        "ranking":
            "pre-Hungarian NGM-v2 Sinkhorn soft assignment",
        "hungarian":
            "one assignment per physical pair; counted bidirectionally",
    }

    (
        args.out
        / "protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
        ) + "\n"
    )

    best_top1 = -1.0
    best_epoch = -1

    best_path = (
        args.out
        / "best.pt"
    )

    history = []

    # =========================================================
    # TRAIN + VAL
    # =========================================================

    for epoch in range(
        args.epochs
    ):
        model.train()

        rng = random.Random(
            args.seed
            + epoch
        )

        order = list(
            usable
        )

        rng.shuffle(order)

        if (
            args.pairs_per_epoch
            > 0
        ):
            if (
                len(order)
                >= args.pairs_per_epoch
            ):
                order = order[
                    : args.pairs_per_epoch
                ]
            else:
                order = [
                    rng.choice(
                        usable
                    )
                    for _ in range(
                        args.pairs_per_epoch
                    )
                ]

        losses = []

        for (
            pair_index,
            side_a,
            side_b,
            direction,
        ) in order:

            pair = train[
                pair_index
            ]

            item = (
                training_intersection(
                    pair[side_a],
                    pair[side_b],
                    args.min_common,
                )
            )

            if item is None:
                continue

            (
                xyz1,
                xyz2,
                gt,
                common,
            ) = item

            graph1 = base.make_graph(
                xyz1,
                device,
            )

            graph2 = base.make_graph(
                xyz2,
                device,
            )

            gt_tensor = (
                torch.as_tensor(
                    gt,
                    dtype=torch.float32,
                    device=device,
                )[None]
            )

            ns1 = torch.tensor(
                [len(xyz1)],
                dtype=torch.long,
                device=device,
            )

            ns2 = torch.tensor(
                [len(xyz2)],
                dtype=torch.long,
                device=device,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            ds = model(
                graph1,
                graph2,
            )

            # Exact old runner objective.
            loss = criterion(
                ds,
                gt_tensor,
                ns1,
                ns2,
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"non-finite loss "
                    f"epoch={epoch+1} "
                    f"pair="
                    f"{pair['pair_id']} "
                    f"direction="
                    f"{direction}"
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            losses.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

        if not losses:
            raise RuntimeError(
                "No training loss"
            )

        val_metrics = (
            evaluate_pairs(
                model,
                val,
                device,
            )
        )

        lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"[E{epoch+1:02d}/"
            f"{args.epochs:02d}] "
            f"loss="
            f"{np.mean(losses):.5f} "
            f"lr={lr:.2e} "
            f"VAL Q="
            f"{val_metrics['queries']} "
            f"Top1="
            f"{100*val_metrics['top1']:.2f}% "
            f"Top5="
            f"{100*val_metrics['top5']:.2f}% "
            f"MRR="
            f"{val_metrics['mrr']:.4f} "
            f"Hung="
            f"{100*val_metrics['hungarian']:.2f}%"
        )

        history.append({
            "epoch":
                epoch + 1,
            "loss":
                float(
                    np.mean(losses)
                ),
            "lr":
                float(lr),
            "val":
                val_metrics,
        })

        # Unified Zebrafish learned-baseline protocol.
        # Exact tie keeps earlier checkpoint.
        if (
            val_metrics["top1"]
            > best_top1 + 1e-12
        ):
            best_top1 = (
                val_metrics["top1"]
            )

            best_epoch = (
                epoch + 1
            )

            torch.save({
                "epoch":
                    best_epoch,
                "model_state":
                    model.state_dict(),
                "val":
                    val_metrics,
                "fold":
                    args.fold,
                "seed":
                    args.seed,
                "feature_dim":
                    args.feature_dim,
                "protocol":
                    protocol,
            }, best_path)

        scheduler.step()

    (
        args.out
        / "history.json"
    ).write_text(
        json.dumps(
            history,
            indent=2,
        ) + "\n"
    )

    payload = torch.load(
        best_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        payload["model_state"]
    )

    val_final = evaluate_pairs(
        model,
        val,
        device,
    )

    print()
    print("=" * 94)
    print(
        "TRAINING FINISHED — "
        "TEST STILL UNREAD"
    )
    print("=" * 94)

    print(
        "best epoch    :",
        best_epoch,
    )

    print(
        "best val Top1 :",
        f"{100*best_top1:.2f}%"
    )

    print_metrics(
        "VAL",
        val_final,
    )

    lock = {
        **protocol,
        "status":
            "LOCKED_BEFORE_TEST",
        "best_epoch":
            best_epoch,
        "best_val_top1":
            best_top1,
        "test_accessed_during_training":
            False,
    }

    (
        args.out
        / "LOCKED_BEFORE_TEST.json"
    ).write_text(
        json.dumps(
            lock,
            indent=2,
        ) + "\n"
    )

    if args.skip_test:
        print(
            "[LOCK] --skip-test: "
            "TEST WAS NOT READ"
        )
        return

    # =========================================================
    # FIRST TEST ACCESS
    # =========================================================

    test_raw = (
        load_physical_pairs(
            fold_root,
            "test",
        )
    )

    test = normalize_pairs(
        test_raw,
        mean,
        std,
    )

    test_metrics = (
        evaluate_pairs(
            model,
            test,
            device,
        )
    )

    expected = EXPECTED_Q[
        args.fold
    ]

    if (
        test_metrics["queries"]
        != expected
    ):
        raise RuntimeError(
            "LOCKED QUERY AUDIT FAILED: "
            f"fold{args.fold}: "
            f"{test_metrics['queries']} "
            f"!= {expected}"
        )

    summary = {
        "method":
            "NGM-v2 official core adapted",
        "fold":
            args.fold,
        "seed":
            args.seed,
        "best_epoch":
            best_epoch,
        "validation":
            val_final,
        "test":
            test_metrics,
        "expected_queries":
            expected,
        "query_audit_passed":
            True,
        "protocol":
            lock,
    }

    (
        args.out
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n"
    )

    print()
    print_metrics(
        "TEST",
        test_metrics,
    )

    print()
    print("=" * 94)
    print(
        f"NGM-v2 ZEBRAFISH "
        f"FOLD{args.fold} "
        "— LOCKED TEST"
    )
    print("=" * 94)

    print(
        "Physical pairs :",
        test_metrics["pairs"],
    )

    print(
        "Queries        :",
        test_metrics["queries"],
    )

    print(
        "Top-1          :",
        f"{100*test_metrics['top1']:.2f}%"
    )

    print(
        "Top-5          :",
        f"{100*test_metrics['top5']:.2f}%"
    )

    print(
        "MRR            :",
        f"{test_metrics['mrr']:.4f}"
    )

    print(
        "Hungarian      :",
        f"{100*test_metrics['hungarian']:.2f}%"
    )

    print(
        "Saved:",
        args.out
        / "summary.json"
    )


if __name__ == "__main__":
    main()
