import os
import argparse

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from config import make_cfg
from dataset import train_valid_data_loader, test_data_loader
from model import create_model
from loss import build_dense_scores
from geotransformer.utils.torch import to_cuda


def load_checkpoint(model, path):
    print(f"[checkpoint] {path}")

    ckpt = torch.load(
        path,
        map_location="cpu",
    )

    state = None

    if isinstance(ckpt, dict):
        for key in [
            "model",
            "state_dict",
            "model_state_dict",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                state = ckpt[key]
                print(f"[checkpoint] state key = {key}")
                break

    if state is None:
        if isinstance(ckpt, dict):
            # fallback: checkpoint itself may be state_dict
            if all(
                isinstance(k, str)
                for k in ckpt.keys()
            ):
                state = ckpt

    if state is None:
        raise RuntimeError(
            f"Cannot find model state dict. "
            f"Checkpoint keys: {list(ckpt.keys())}"
        )

    # DDP compatibility
    clean_state = {}

    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        clean_state[k] = v

    result = model.load_state_dict(
        clean_state,
        strict=False,
    )

    print(
        "[checkpoint] missing =",
        len(result.missing_keys),
    )
    print(
        "[checkpoint] unexpected =",
        len(result.unexpected_keys),
    )

    if result.missing_keys:
        print("missing keys:", result.missing_keys[:20])

    if result.unexpected_keys:
        print(
            "unexpected keys:",
            result.unexpected_keys[:20],
        )


def evaluate_pair(output_dict):

    dense = build_dense_scores(
        output_dict
    )

    ref_ids = output_dict["ref_ids"]
    src_ids = output_dict["src_ids"]

    # [Nr, Ns]
    gt_map = (
        ref_ids[:, None]
        == src_ids[None, :]
    )

    gt_map = torch.logical_and(
        gt_map,
        ref_ids[:, None] >= 0,
    )

    gt_map = torch.logical_and(
        gt_map,
        src_ids[None, :] >= 0,
    )

    # valid query = this reference neuron has
    # at least one same-ID neuron in source worm
    valid_q = gt_map.any(dim=1)

    q_indices = torch.where(valid_q)[0]

    num_queries = int(valid_q.sum().item())

    if num_queries == 0:
        return None

    # --------------------------------------------------
    # Candidate coverage
    #
    # Was the true identity ever exposed to the
    # fine-level score matrix after coarse selection?
    # --------------------------------------------------
    valid_score = dense > -1e8

    gt_candidate = torch.logical_and(
        valid_score,
        gt_map,
    ).any(dim=1)

    covered = int(
        gt_candidate[valid_q]
        .sum()
        .item()
    )

    # --------------------------------------------------
    # Top-1
    # --------------------------------------------------
    pred1 = dense.argmax(dim=1)

    top1_correct = gt_map[
        torch.arange(
            dense.shape[0],
            device=dense.device,
        ),
        pred1,
    ]

    top1_correct = torch.logical_and(
        top1_correct,
        valid_q,
    )

    n_top1 = int(
        top1_correct.sum().item()
    )

    # --------------------------------------------------
    # Top-5
    # --------------------------------------------------
    k = min(
        5,
        dense.shape[1],
    )

    topk = torch.topk(
        dense,
        k=k,
        dim=1,
    ).indices

    top5_correct = torch.gather(
        gt_map,
        1,
        topk,
    ).any(dim=1)

    # a row with no real candidate must not
    # accidentally count
    row_has_candidate = valid_score.any(dim=1)

    top5_correct = torch.logical_and(
        top5_correct,
        row_has_candidate,
    )

    top5_correct = torch.logical_and(
        top5_correct,
        valid_q,
    )

    n_top5 = int(
        top5_correct.sum().item()
    )

    # --------------------------------------------------
    # MRR
    #
    # Support duplicate IDs safely:
    # use the highest-ranked correct target.
    # --------------------------------------------------
    minus_inf = torch.full_like(
        dense,
        -1e9,
    )

    correct_scores = torch.where(
        gt_map,
        dense,
        minus_inf,
    )

    best_gt_score = correct_scores.max(
        dim=1
    ).values

    gt_seen = best_gt_score > -1e8

    ranks = 1 + (
        dense
        > best_gt_score[:, None]
    ).sum(dim=1)

    reciprocal_rank = torch.zeros(
        dense.shape[0],
        dtype=dense.dtype,
        device=dense.device,
    )

    mrr_valid = torch.logical_and(
        valid_q,
        gt_seen,
    )

    reciprocal_rank[mrr_valid] = (
        1.0
        / ranks[mrr_valid].float()
    )

    rr_sum = float(
        reciprocal_rank[valid_q]
        .sum()
        .item()
    )

    # --------------------------------------------------
    # Hungarian / one-to-one assignment
    # --------------------------------------------------
    score_np = (
        dense.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )

    # linear_sum_assignment minimises cost.
    # -score means maximum similarity assignment.
    row_ind, col_ind = linear_sum_assignment(
        -score_np
    )

    assignment = np.full(
        dense.shape[0],
        -1,
        dtype=np.int64,
    )

    assignment[row_ind] = col_ind

    gt_np = (
        gt_map.detach()
        .cpu()
        .numpy()
    )

    valid_np = (
        valid_q.detach()
        .cpu()
        .numpy()
    )

    hungarian_correct = 0

    for r in np.where(valid_np)[0]:

        c = assignment[r]

        if c >= 0 and gt_np[r, c]:
            hungarian_correct += 1

    return {
        "queries": num_queries,
        "top1_correct": n_top1,
        "top5_correct": n_top5,
        "rr_sum": rr_sum,
        "hungarian_correct": hungarian_correct,
        "covered": covered,
    }


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )

    args = parser.parse_args()

    cfg = make_cfg()

    print("dataset:", cfg.data.dataset_root)
    print("seed:", cfg.seed)
    print("split:", args.split)

    if args.split == "val":

        _, loader, neighbor_limits = \
            train_valid_data_loader(
                cfg,
                False,
            )

    else:

        loader, neighbor_limits = \
            test_data_loader(cfg)

    print(
        "neighbor_limits:",
        neighbor_limits,
    )

    model = create_model(cfg).cuda()

    load_checkpoint(
        model,
        args.checkpoint,
    )

    model.eval()

    total_q = 0
    total_top1 = 0
    total_top5 = 0
    total_rr = 0.0
    total_hung = 0
    total_covered = 0

    pair_results = []

    pbar = tqdm(
        loader,
        total=len(loader),
    )

    for pair_idx, data_dict in enumerate(pbar):

        ref_name = data_dict.get(
            "ref_name",
            f"ref_{pair_idx}",
        )

        src_name = data_dict.get(
            "src_name",
            f"src_{pair_idx}",
        )

        data_dict = to_cuda(
            data_dict
        )

        output_dict = model(
            data_dict
        )

        result = evaluate_pair(
            output_dict
        )

        if result is None:
            continue

        q = result["queries"]

        pair_top1 = (
            result["top1_correct"] / q
        )

        pair_top5 = (
            result["top5_correct"] / q
        )

        pair_mrr = (
            result["rr_sum"] / q
        )

        pair_hung = (
            result["hungarian_correct"] / q
        )

        pair_cov = (
            result["covered"] / q
        )

        pair_results.append(
            (
                ref_name,
                src_name,
                q,
                pair_top1,
                pair_top5,
                pair_mrr,
                pair_hung,
                pair_cov,
            )
        )

        total_q += q

        total_top1 += (
            result["top1_correct"]
        )

        total_top5 += (
            result["top5_correct"]
        )

        total_rr += (
            result["rr_sum"]
        )

        total_hung += (
            result["hungarian_correct"]
        )

        total_covered += (
            result["covered"]
        )

        pbar.set_postfix(
            {
                "Top1":
                    f"{total_top1 / total_q:.3f}",
                "Top5":
                    f"{total_top5 / total_q:.3f}",
                "MRR":
                    f"{total_rr / total_q:.3f}",
                "Hung":
                    f"{total_hung / total_q:.3f}",
                "Cov":
                    f"{total_covered / total_q:.3f}",
            }
        )

    print()
    print("=" * 80)

    print(
        f"GeoTransformer Semantic | {args.split}"
    )

    print("=" * 80)

    print(
        f"Pairs               : {len(pair_results)}"
    )

    print(
        f"Queries             : {total_q}"
    )

    print(
        f"Top-1               : "
        f"{100 * total_top1 / total_q:.2f}%"
    )

    print(
        f"Top-5               : "
        f"{100 * total_top5 / total_q:.2f}%"
    )

    print(
        f"MRR                 : "
        f"{total_rr / total_q:.4f}"
    )

    print(
        f"Hungarian Accuracy  : "
        f"{100 * total_hung / total_q:.2f}%"
    )

    print(
        f"GT Candidate Coverage: "
        f"{100 * total_covered / total_q:.2f}%"
    )

    print("=" * 80)

    print("\nPer-pair results:")

    for (
        ref_name,
        src_name,
        q,
        top1,
        top5,
        mrr,
        hung,
        cov,
    ) in pair_results:

        print(
            f"{ref_name} -> {src_name} | "
            f"Q={q:4d} | "
            f"T1={100*top1:6.2f}% | "
            f"T5={100*top5:6.2f}% | "
            f"MRR={mrr:.4f} | "
            f"Hung={100*hung:6.2f}% | "
            f"Cov={100*cov:6.2f}%"
        )


if __name__ == "__main__":
    main()
