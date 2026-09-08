#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from utils.config import cfg, cfg_from_file
from utils.loss_func import PermLoss

# Reuse the already-audited official-RGM adapter pieces.
import run_rgm_atanas_single_template as base


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
    if key in z:
        x = np.asarray(z[key], dtype=bool).reshape(-1)
        if len(x) != n:
            raise RuntimeError(
                f"{key}: {len(x)} != {n}"
            )
        return x

    return np.ones(n, dtype=bool)


def load_side(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as z:

        xyz = np.asarray(
            z["xyz"],
            dtype=np.float32,
        )

        raw_ids = np.asarray(
            z["cell_id"]
        ).reshape(-1)

        n = len(raw_ids)

        if xyz.shape != (n, 3):
            raise RuntimeError(
                f"{path}: xyz={xyz.shape}, "
                f"cell_id={raw_ids.shape}"
            )

        finite = np.isfinite(xyz).all(axis=1)

        if "valid_xyz_mask" in z:
            valid_xyz = np.asarray(
                z["valid_xyz_mask"],
                dtype=bool,
            ).reshape(-1)
        else:
            valid_xyz = np.ones(n, dtype=bool)

        candidate = finite & valid_xyz

        labeled = mask_or_true(
            z, "labeled_mask", n
        )
        certain = mask_or_true(
            z, "certain_mask", n
        )
        clean = mask_or_true(
            z, "clean_mask", n
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
            f"{path}: invalid side={side}"
        )

    # Candidate population stays intact.
    xyz = xyz[candidate]
    ids = raw_ids[candidate]
    mask = supervised[candidate]

    return {
        "path": str(path.resolve()),
        "name": path.name,
        "pair_id": pair_id,
        "side": side,

        # base.fit_train_norm() expects xyz_raw.
        "xyz_raw": xyz.astype(np.float32),

        # Filled after train-only normalization.
        "xyz": None,

        "cell_id": ids,
        "mask": mask.astype(bool),
    }


def load_physical_pairs(root: Path, split: str):
    split_root = root / split

    files = sorted(
        split_root.rglob("*.npz")
    )

    if not files:
        raise RuntimeError(
            f"No NPZ under {split_root}"
        )

    groups = {}

    for path in files:
        w = load_side(path)

        pid = w["pair_id"]
        side = w["side"]

        groups.setdefault(pid, {})

        if side in groups[pid]:
            raise RuntimeError(
                f"duplicate {pid}/{side}"
            )

        groups[pid][side] = w

    pairs = []

    for pid in sorted(groups):
        g = groups[pid]

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


def fit_train_normalization(train_pairs):
    sides = []

    for p in train_pairs:
        sides += [p["q"], p["r"]]

    # Exact normalization implementation from the
    # previously audited Atanas RGM runner.
    return base.fit_train_norm(sides)


def normalize_pairs(pairs, mu, sd):
    out = []

    for p in pairs:
        q, r = base.normalize(
            [p["q"], p["r"]],
            mu,
            sd,
        )

        out.append({
            "pair_id": p["pair_id"],
            "q": q,
            "r": r,
        })

    return out


def ordered_training_pairs(physical_pairs):
    """
    Both directions are training examples so temporal
    direction is not privileged.
    """
    out = []

    for p in physical_pairs:
        out.append(
            (p["q"], p["r"], p["pair_id"], "q2r")
        )
        out.append(
            (p["r"], p["q"], p["pair_id"], "r2q")
        )

    return out


def train_one_pair(
    model,
    criterion,
    src,
    tgt,
    device,
):
    x = base.make_pair(
        src,
        tgt,
        device,
    )

    if not x["queries"]:
        return None

    s, in_src, in_ref = model(
        x["P1"],
        x["P2"],
        x["A1"],
        x["A2"],
        x["n1"],
        x["n2"],
    )

    # Preserve exact official crop-RGM loss behavior.
    if cfg.PGM.USEINLIERRATE:
        s_loss = (
            in_src
            * s
            * in_ref.transpose(
                2, 1
            ).contiguous()
        )
    else:
        s_loss = s

    loss = criterion(
        s_loss,
        x["gt"],
        x["n1"],
        x["n2"],
    )

    return loss


def directional_metrics(score, gt):
    """
    Unified benchmark rank semantics.

    score: [Nquery, Ncand]
    gt   : bool [Nquery, Ncand]
    """

    valid = gt.any(axis=1)

    top1 = 0
    top5 = 0
    rr = 0.0
    q = int(valid.sum())

    for i in np.flatnonzero(valid):

        gt_cols = np.flatnonzero(gt[i])

        if len(gt_cols) == 0:
            continue

        # Usually exactly one under strict partial GT.
        gt_scores = score[
            i,
            gt_cols,
        ]

        best_gt_score = float(
            np.max(gt_scores)
        )

        # Same optimistic tie convention as locked benchmark:
        # rank = 1 + number strictly greater than GT.
        rank = 1 + int(
            np.sum(
                score[i] > best_gt_score
            )
        )

        if rank == 1:
            top1 += 1

        if rank <= 5:
            top5 += 1

        rr += 1.0 / rank

    return {
        "queries": q,
        "top1_correct": top1,
        "top5_correct": top5,
        "rr_sum": rr,
    }


@torch.no_grad()
def evaluate_physical_pairs(
    model,
    physical_pairs,
    device,
):
    """
    One model evaluation per physical q/r pair.

    Same S(q,r) is used for:
      q -> r ranking,
      r -> q ranking via S^T,
      one Hungarian assignment.
    """
    model.eval()

    total_q = 0
    c1 = 0
    c5 = 0
    rr = 0.0
    hc = 0

    rows = []

    for pair_index, p in enumerate(
        physical_pairs
    ):
        qside = p["q"]
        rside = p["r"]

        x = base.make_pair(
            qside,
            rside,
            device,
        )

        s, in_src, in_ref = model(
            x["P1"],
            x["P2"],
            x["A1"],
            x["A2"],
            x["n1"],
            x["n2"],
        )

        # PRE-HUNGARIAN RGM Sinkhorn matrix.
        score = (
            s[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        gt = (
            x["gt"][0]
            .detach()
            .cpu()
            .numpy()
            > 0.5
        )

        # q -> r
        a = directional_metrics(
            score,
            gt,
        )

        # r -> q, using SAME score matrix.
        b = directional_metrics(
            score.T,
            gt.T,
        )

        pair_q = (
            a["queries"]
            + b["queries"]
        )

        pair_c1 = (
            a["top1_correct"]
            + b["top1_correct"]
        )

        pair_c5 = (
            a["top5_correct"]
            + b["top5_correct"]
        )

        pair_rr = (
            a["rr_sum"]
            + b["rr_sum"]
        )

        # Unified Hungarian on same soft matrix.
        row_ind, col_ind = (
            linear_sum_assignment(
                -score
            )
        )

        valid_rows = gt.any(axis=1)
        valid_cols = gt.any(axis=0)

        pair_hc = 0

        for i, j in zip(
            row_ind,
            col_ind,
        ):
            if not gt[i, j]:
                continue

            # Count both directional benchmark queries.
            if valid_rows[i]:
                pair_hc += 1

            if valid_cols[j]:
                pair_hc += 1

        total_q += pair_q
        c1 += pair_c1
        c5 += pair_c5
        rr += pair_rr
        hc += pair_hc

        row = {
            "pair_index": pair_index,
            "pair_id": p["pair_id"],
            "queries": pair_q,
            "q_to_r_queries":
                a["queries"],
            "r_to_q_queries":
                b["queries"],
            "top1":
                pair_c1 / pair_q,
            "top5":
                pair_c5 / pair_q,
            "mrr":
                pair_rr / pair_q,
            "hungarian":
                pair_hc / pair_q,
        }

        rows.append(row)

    if total_q == 0:
        raise RuntimeError(
            "No evaluation queries."
        )

    return {
        "pairs": len(physical_pairs),
        "queries": total_q,
        "top1": c1 / total_q,
        "top5": c5 / total_q,
        "mrr": rr / total_q,
        "hungarian": hc / total_q,
        "pair_results": rows,
    }


def print_metrics(tag, m):
    print(
        f"{tag:<5s} "
        f"pairs={m['pairs']:2d} "
        f"Q={m['queries']:4d} "
        f"Top1={100*m['top1']:6.2f}% "
        f"Top5={100*m['top5']:6.2f}% "
        f"MRR={m['mrr']:.4f} "
        f"Hung={100*m['hungarian']:6.2f}%"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fold",
        type=int,
        required=True,
        choices=range(1, 9),
    )

    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--run-root",
        type=Path,
        required=True,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--epochs",
        type=int,
        default=200,
    )

    ap.add_argument(
        "--skip-test",
        action="store_true",
    )

    args = ap.parse_args()

    base.seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    torch.cuda.set_device(args.gpu)
    device = torch.device(
        f"cuda:{args.gpu}"
    )

    repo = Path(__file__).resolve().parent

    official_cfg = (
        repo
        / "experiments"
        / "train_RGM_Seen_Crop_modelnet40_transformer.yaml"
    )

    cfg_from_file(
        str(official_cfg)
    )

    cfg.GPUS = [args.gpu]

    # Import AFTER official config is loaded.
    from models.Net import Net

    run_root = args.run_root
    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # PRE-TEST: train + validation only.
    # ----------------------------------------------------------
    train_raw = load_physical_pairs(
        args.data_root,
        "train",
    )

    val_raw = load_physical_pairs(
        args.data_root,
        "val",
    )

    print("=" * 90)
    print(
        f"OFFICIAL RGM — ZEBRAFISH "
        f"LOFO FOLD{args.fold} SEED{args.seed}"
    )
    print("=" * 90)
    print(
        "train physical pairs:",
        len(train_raw),
    )
    print(
        "val physical pairs  :",
        len(val_raw),
    )
    print(
        "test               : UNREAD"
    )
    print(
        "official config     :",
        official_cfg,
    )
    print(
        "neighbors           :",
        cfg.PGM.NEIGHBORSNUM,
    )
    print(
        "features            :",
        cfg.PGM.FEATURES,
    )

    # ----------------------------------------------------------
    # TRAIN-ONLY normalization.
    # ----------------------------------------------------------
    mu, sd = fit_train_normalization(
        train_raw
    )

    np.savez(
        run_root
        / "train_normalization.npz",
        mean=mu,
        std=sd,
    )

    train_pairs = normalize_pairs(
        train_raw,
        mu,
        sd,
    )

    val_pairs = normalize_pairs(
        val_raw,
        mu,
        sd,
    )

    train_ordered = (
        ordered_training_pairs(
            train_pairs
        )
    )

    print(
        "training ordered pairs:",
        len(train_ordered),
    )

    # Preflight min N for DGCNN kNN.
    min_n = min(
        len(p[s]["xyz"])
        for p in train_pairs + val_pairs
        for s in ("q", "r")
    )

    print(
        "minimum candidate N:",
        min_n,
    )

    if min_n < cfg.PGM.NEIGHBORSNUM:
        raise RuntimeError(
            f"minimum N={min_n} < "
            f"RGM k={cfg.PGM.NEIGHBORSNUM}"
        )

    model = Net().to(device)

    print(
        "RGM parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}"
    )

    criterion = PermLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.TRAIN.LR,
        momentum=cfg.TRAIN.MOMENTUM,
        nesterov=True,
    )

    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(
                cfg.TRAIN.LR_STEP
            ),
            gamma=cfg.TRAIN.LR_DECAY,
        )
    )

    best_top1 = -1.0
    best_epoch = -1

    history = []

    # ----------------------------------------------------------
    # TRAIN + VAL.
    # ----------------------------------------------------------
    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        order = np.random.RandomState(
            args.seed + epoch
        ).permutation(
            len(train_ordered)
        )

        losses = []

        for ii in order:
            src, tgt, pid, direction = (
                train_ordered[int(ii)]
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = train_one_pair(
                model,
                criterion,
                src,
                tgt,
                device,
            )

            if loss is None:
                continue

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss "
                    f"epoch={epoch} "
                    f"pair={pid} "
                    f"direction={direction}"
                )

            loss.backward()
            optimizer.step()

            losses.append(
                float(loss.item())
            )

        if not losses:
            raise RuntimeError(
                "No training losses generated."
            )

        val = evaluate_physical_pairs(
            model,
            val_pairs,
            device,
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"loss={np.mean(losses):.6f} "
            f"lr={lr:.2e} "
            f"val Top1={100*val['top1']:.2f}% "
            f"Top5={100*val['top5']:.2f}% "
            f"MRR={val['mrr']:.4f} "
            f"Hung={100*val['hungarian']:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "loss":
                float(np.mean(losses)),
            "lr": float(lr),
            "val_top1": val["top1"],
            "val_top5": val["top5"],
            "val_mrr": val["mrr"],
            "val_hungarian":
                val["hungarian"],
        })

        # Primary benchmark metric:
        # highest validation Top-1.
        # Exact tie keeps EARLIER epoch.
        if (
            val["top1"]
            > best_top1 + 1e-12
        ):
            best_top1 = val["top1"]
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "model_state":
                    model.state_dict(),
                "val": val,
                "fold": args.fold,
                "seed": args.seed,
            }, run_root / "best.pt")

        scheduler.step()

    (
        run_root / "history.json"
    ).write_text(
        json.dumps(
            history,
            indent=2,
        ) + "\n"
    )

    print()
    print("=" * 90)
    print(
        "TRAINING FINISHED — "
        "TEST STILL UNREAD"
    )
    print("=" * 90)
    print(
        "best epoch    :",
        best_epoch,
    )
    print(
        "best val Top1 :",
        f"{100*best_top1:.2f}%"
    )

    # ----------------------------------------------------------
    # Protocol lock BEFORE first test read.
    # ----------------------------------------------------------
    lock = {
        "method": "RGM",
        "implementation":
            "official fukexue/RGM",
        "task":
            "Zebrafish longitudinal pair-local matching",
        "fold": args.fold,
        "seed": args.seed,
        "geometry_only": True,
        "model_input": "xyz",
        "train_physical_pairs":
            len(train_pairs),
        "val_physical_pairs":
            len(val_pairs),
        "train_directions":
            "q->r and r->q",
        "normalization":
            "global train-only per-axis z-score",
        "official_config":
            str(official_cfg),
        "checkpoint_selection":
            "highest validation Top-1; exact tie -> earlier epoch",
        "ranking_scores":
            "pre-Hungarian RGM Sinkhorn soft assignment",
        "hungarian":
            "scipy linear_sum_assignment on same soft score matrix",
        "test_accessed_during_training":
            False,
        "best_epoch":
            best_epoch,
    }

    (
        run_root
        / "LOCKED_BEFORE_TEST.json"
    ).write_text(
        json.dumps(
            lock,
            indent=2,
        ) + "\n"
    )

    if args.skip_test:
        print(
            "SKIP TEST requested."
        )
        return

    # ----------------------------------------------------------
    # FIRST TEST ACCESS.
    # ----------------------------------------------------------
    test_raw = load_physical_pairs(
        args.data_root,
        "test",
    )

    test_pairs = normalize_pairs(
        test_raw,
        mu,
        sd,
    )

    ckpt = torch.load(
        run_root / "best.pt",
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        ckpt["model_state"]
    )

    val_final = evaluate_physical_pairs(
        model,
        val_pairs,
        device,
    )

    test_final = evaluate_physical_pairs(
        model,
        test_pairs,
        device,
    )

    print()
    print_metrics(
        "VAL",
        val_final,
    )
    print_metrics(
        "TEST",
        test_final,
    )

    expected = EXPECTED_Q[
        args.fold
    ]

    if (
        test_final["queries"]
        != expected
    ):
        raise RuntimeError(
            "LOCKED QUERY AUDIT FAILED: "
            f"{test_final['queries']} "
            f"!= {expected}"
        )

    summary = {
        "method": "RGM",
        "fold": args.fold,
        "seed": args.seed,
        "best_epoch":
            best_epoch,
        "validation":
            val_final,
        "test":
            test_final,
        "expected_queries":
            expected,
        "query_audit_passed":
            True,
        "protocol":
            lock,
    }

    (
        run_root / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ) + "\n"
    )

    print()
    print("=" * 90)
    print(
        f"RGM ZEBRAFISH FOLD{args.fold} "
        "— LOCKED TEST"
    )
    print("=" * 90)
    print(
        "Physical pairs :",
        test_final["pairs"],
    )
    print(
        "Queries        :",
        test_final["queries"],
    )
    print(
        "Top-1          :",
        f"{100*test_final['top1']:.2f}%"
    )
    print(
        "Top-5          :",
        f"{100*test_final['top5']:.2f}%"
    )
    print(
        "MRR            :",
        f"{test_final['mrr']:.4f}"
    )
    print(
        "Hungarian      :",
        f"{100*test_final['hungarian']:.2f}%"
    )
    print(
        "Saved:",
        run_root / "summary.json"
    )


if __name__ == "__main__":
    main()
