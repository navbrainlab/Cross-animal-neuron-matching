import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from config import make_cfg
from dataset import test_data_loader
from model import create_model
from loss import build_dense_scores
from evaluate_semantic import load_checkpoint
from scripts.zebrafish.query_record_io import QUERY_COLUMNS, records_from_score_matrix


def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def first_int(x):
    if torch.is_tensor(x):
        return int(x.reshape(-1)[0].item())
    if isinstance(x, np.ndarray):
        return int(x.reshape(-1)[0])
    if isinstance(x, (list, tuple)):
        return first_int(x[0])
    return int(x)


def directional(scores, gt):
    """
    scores: [Nquery, Ncand]
    gt:     [Nquery, Ncand]
    """
    valid_q = gt.any(dim=1)
    valid_score = scores > -1e8

    nq = int(valid_q.sum().item())

    if nq == 0:
        return {
            "queries": 0,
            "top1": 0,
            "top5": 0,
            "rr_sum": 0.0,
            "covered": 0,
        }

    # Coverage: correct candidate was actually exposed
    gt_seen = torch.logical_and(
        valid_score,
        gt,
    ).any(dim=1)

    # ---------------- Top-1 ----------------
    pred_score, pred = scores.max(dim=1)

    top1_ok = gt[
        torch.arange(scores.shape[0], device=scores.device),
        pred,
    ]

    # Important: all -1e9 must not accidentally count.
    top1_ok = (
        top1_ok
        & valid_q
        & (pred_score > -1e8)
    )

    # ---------------- Top-5 ----------------
    k = min(5, scores.shape[1])

    topk_score, topk_idx = torch.topk(
        scores,
        k=k,
        dim=1,
    )

    topk_gt = torch.gather(
        gt,
        1,
        topk_idx,
    )

    top5_ok = torch.logical_and(
        topk_gt,
        topk_score > -1e8,
    ).any(dim=1)

    top5_ok = top5_ok & valid_q

    # ---------------- MRR ----------------
    minus_inf = torch.full_like(
        scores,
        -1e9,
    )

    correct_scores = torch.where(
        gt,
        scores,
        minus_inf,
    )

    best_gt_score = correct_scores.max(dim=1).values
    seen = best_gt_score > -1e8

    ranks = 1 + (
        scores > best_gt_score[:, None]
    ).sum(dim=1)

    rr = torch.zeros(
        scores.shape[0],
        dtype=torch.float32,
        device=scores.device,
    )

    use = valid_q & seen
    rr[use] = 1.0 / ranks[use].float()

    return {
        "queries": nq,
        "top1": int(top1_ok.sum().item()),
        "top5": int(top5_ok.sum().item()),
        "rr_sum": float(rr[valid_q].sum().item()),
        "covered": int(gt_seen[valid_q].sum().item()),
    }


def bidirectional_pair_metrics(dense, ref_ids, src_ids):
    gt = (
        (ref_ids[:, None] == src_ids[None, :])
        & (ref_ids[:, None] >= 0)
        & (src_ids[None, :] >= 0)
    )

    # q/ref -> r/src
    a = directional(dense, gt)

    # r/src -> q/ref
    b = directional(
        dense.transpose(0, 1),
        gt.transpose(0, 1),
    )

    # ------------------------------------------------------------
    # One Hungarian solve on the physical q/r matrix.
    # Count correctness in BOTH directions.
    # Invalid (-1e9) GeoTransformer cells cannot count as correct.
    # ------------------------------------------------------------
    score_np = (
        dense.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    gt_np = gt.detach().cpu().numpy()
    valid_np = score_np > -1e8

    row_ind, col_ind = linear_sum_assignment(
        -score_np
    )

    valid_ref = gt.any(dim=1).detach().cpu().numpy()
    valid_src = gt.any(dim=0).detach().cpu().numpy()

    hung_correct = 0

    for r, c in zip(row_ind, col_ind):
        if not valid_np[r, c]:
            continue

        if not gt_np[r, c]:
            continue

        if valid_ref[r]:
            hung_correct += 1

        if valid_src[c]:
            hung_correct += 1

    return {
        "queries": a["queries"] + b["queries"],
        "top1_correct": a["top1"] + b["top1"],
        "top5_correct": a["top5"] + b["top5"],
        "rr_sum": a["rr_sum"] + b["rr_sum"],
        "covered": a["covered"] + b["covered"],
        "hungarian_correct": hung_correct,

        "q_to_r_queries": a["queries"],
        "r_to_q_queries": b["queries"],
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--checkpoint",
        required=True,
    )
    ap.add_argument(
        "--output",
        required=True,
    )
    ap.add_argument(
        "--device",
        default="cuda",
    )
    ap.add_argument(
        "--expected-pairs",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--expected-queries",
        type=int,
        default=3536,
    )
    ap.add_argument("--query-output", type=Path, default=None)
    ap.add_argument("--fold", type=int, choices=range(1, 9), default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)

    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cpu":
        # The pinned upstream revision constructs several temporary tensors
        # with a literal ``.cuda()``.  For CPU-only archival replay, redirect
        # only that no-argument convenience call; model/data placement still
        # follows the explicit ``device`` above.
        torch.Tensor.cuda = lambda self, *unused_args, **unused_kwargs: self
        def _cpu_index_select(data, index, dim):
            selected = data.index_select(dim, index.reshape(-1))
            if index.ndim > 1:
                shape = data.shape[:dim] + index.shape + data.shape[dim + 1:]
                selected = selected.reshape(*shape)
            return selected
        import geotransformer.modules.kpconv.kpconv as _kpconv
        import geotransformer.modules.kpconv.functional as _kpfunctional
        import geotransformer.modules.ops.pointcloud_partition as _partition
        import geotransformer.modules.registration.matching as _matching
        for _module in (_kpconv, _kpfunctional, _partition, _matching):
            if hasattr(_module, "index_select"):
                _module.index_select = _cpu_index_select

    cfg = make_cfg()
    cfg.test.num_workers = args.num_workers

    loader_result = test_data_loader(cfg)

    # Be compatible with either loader or (loader, neighbor_limits).
    if isinstance(loader_result, tuple):
        loader = loader_result[0]
    else:
        loader = loader_result

    model = create_model(cfg).to(device)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    total = {
        "pairs": 0,
        "queries": 0,
        "top1_correct": 0,
        "top5_correct": 0,
        "rr_sum": 0.0,
        "hungarian_correct": 0,
        "covered": 0,
    }

    pair_rows = []
    query_rows = []

    with torch.no_grad():

        for ordered_index, data_dict in enumerate(loader):

            # Dataset was constructed:
            # 0=q->r, 1=r->q, 2=q->r, 3=r->q, ...
            #
            # Prefer explicit pair_index when retained by collate.
            if "pair_index" in data_dict:
                pidx = first_int(
                    data_dict["pair_index"]
                )
            else:
                pidx = ordered_index

            # Only evaluate ONE dense matrix per physical pair.
            if pidx % 2 == 1:
                continue

            data_dict = move_to_device(
                data_dict,
                device,
            )

            output = model(data_dict)

            dense = build_dense_scores(output)

            ref_ids = output["ref_ids"]
            src_ids = output["src_ids"]

            result = bidirectional_pair_metrics(
                dense,
                ref_ids,
                src_ids,
            )
            if args.query_output:
                if args.fold is None:
                    raise ValueError("--fold is required with --query-output")
                gt = (
                    (ref_ids[:, None] == src_ids[None, :])
                    & (ref_ids[:, None] >= 0)
                    & (src_ids[None, :] >= 0)
                )
                row_target = torch.where(
                    gt.any(dim=1), gt.float().argmax(dim=1), -torch.ones(gt.shape[0], device=gt.device, dtype=torch.long)
                )
                col_target = torch.where(
                    gt.any(dim=0), gt.float().argmax(dim=0), -torch.ones(gt.shape[1], device=gt.device, dtype=torch.long)
                )
                query_rows.extend(records_from_score_matrix(
                    method="geotransformer", fold=args.fold, seed=args.seed,
                    pair_index=total["pairs"],
                    pair_id=Path(data_dict["ref_name"][0] if isinstance(data_dict["ref_name"], list) else data_dict["ref_name"]).stem.rsplit("__q", 1)[0],
                    score=dense.detach().cpu().numpy(),
                    valid_score=(dense > -1e8).detach().cpu().numpy(),
                    q_uid=Path(data_dict["ref_name"][0] if isinstance(data_dict["ref_name"], list) else data_dict["ref_name"]).stem,
                    r_uid=Path(data_dict["src_name"][0] if isinstance(data_dict["src_name"], list) else data_dict["src_name"]).stem,
                    q_ids=ref_ids.detach().cpu().numpy(),
                    r_ids=src_ids.detach().cpu().numpy(),
                    row_target=row_target.detach().cpu().numpy(),
                    col_target=col_target.detach().cpu().numpy(),
                    top1_from_prediction=True,
                ))

            total["pairs"] += 1

            for key in [
                "queries",
                "top1_correct",
                "top5_correct",
                "hungarian_correct",
                "covered",
            ]:
                total[key] += result[key]

            total["rr_sum"] += result["rr_sum"]

            row = {
                "physical_pair_index": total["pairs"] - 1,
                **result,
            }

            pair_rows.append(row)

            print(
                f"pair={total['pairs']:02d} "
                f"Q={result['queries']:4d} "
                f"Top1={100*result['top1_correct']/result['queries']:6.2f}% "
                f"Top5={100*result['top5_correct']/result['queries']:6.2f}% "
                f"MRR={result['rr_sum']/result['queries']:.4f} "
                f"Hung={100*result['hungarian_correct']/result['queries']:6.2f}% "
                f"Cov={100*result['covered']/result['queries']:6.2f}%"
            )

    if total["pairs"] != args.expected_pairs:
        raise RuntimeError(
            f"PAIR AUDIT FAILED: "
            f"{total['pairs']} != {args.expected_pairs}"
        )

    if total["queries"] != args.expected_queries:
        raise RuntimeError(
            f"QUERY AUDIT FAILED: "
            f"{total['queries']} != {args.expected_queries}"
        )

    q = total["queries"]

    summary = {
        "method": "GeoTransformer",
        "protocol": (
            "Zebrafish LOFO; pair-local longitudinal; "
            "bidirectional ranking; one Hungarian per physical pair"
        ),
        "checkpoint": str(
            Path(args.checkpoint).resolve()
        ),
        "pairs": total["pairs"],
        "queries": q,
        "top1": total["top1_correct"] / q,
        "top5": total["top5_correct"] / q,
        "mrr": total["rr_sum"] / q,
        "hungarian_accuracy":
            total["hungarian_correct"] / q,
        "coverage":
            total["covered"] / q,
        "pair_results": pair_rows,
    }

    print()
    print("=" * 80)
    print("ZEBRAFISH GEOTRANSFORMER FOLD1 — LOCKED TEST")
    print("=" * 80)
    print(f"Physical pairs : {summary['pairs']}")
    print(f"Queries        : {summary['queries']}")
    print(f"Top-1          : {100*summary['top1']:.2f}%")
    print(f"Top-5          : {100*summary['top5']:.2f}%")
    print(f"MRR            : {summary['mrr']:.4f}")
    print(
        f"Hungarian      : "
        f"{100*summary['hungarian_accuracy']:.2f}%"
    )
    print(
        f"Coverage       : "
        f"{100*summary['coverage']:.2f}%"
    )

    out = Path(args.output)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )
    if args.query_output:
        args.query_output.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if args.query_output.suffix == ".gz" else open
        with opener(args.query_output, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=QUERY_COLUMNS)
            writer.writeheader()
            writer.writerows(query_rows)

    print("Saved:", out)


if __name__ == "__main__":
    main()
