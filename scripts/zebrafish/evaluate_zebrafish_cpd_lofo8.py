#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pycpd import DeformableRegistration
from scipy.optimize import linear_sum_assignment

from mprt_net.data import PairIndex, WormCache, build_pair_targets
from mprt_net.relations import standardize_xyz


def directional(scores: torch.Tensor, targets: torch.Tensor):
    valid = (targets >= 0) & (targets < scores.shape[1])

    if not bool(valid.any()):
        return {
            "queries": 0,
            "top1": 0,
            "top5": 0,
            "rr": 0.0,
        }

    scores = scores[valid]
    targets = targets[valid]

    target_scores = scores.gather(1, targets[:, None])

    # EXACT same optimistic-tie rank as existing MPRT/Euclidean evaluation.
    rank = 1 + (scores > target_scores).sum(dim=1)

    return {
        "queries": int(rank.numel()),
        "top1": int((rank <= 1).sum()),
        "top5": int((rank <= 5).sum()),
        "rr": float((1.0 / rank.float()).sum()),
    }


def hungarian(
    scores: torch.Tensor,
    row_target: torch.Tensor,
    col_target: torch.Tensor,
):
    rows, cols = linear_sum_assignment(
        -scores.detach().cpu().numpy()
    )

    assignment = {
        int(row): int(col)
        for row, col in zip(rows, cols)
    }
    inverse = {
        col: row
        for row, col in assignment.items()
    }

    num_a, num_b = scores.shape

    correct = 0
    total = 0

    for row, target in enumerate(row_target.tolist()):
        if 0 <= target < num_b:
            total += 1
            correct += int(
                assignment.get(row, -1) == int(target)
            )

    for col, target in enumerate(col_target.tolist()):
        if 0 <= target < num_a:
            total += 1
            correct += int(
                inverse.get(col, -1) == int(target)
            )

    return correct, total


def register_deformable(
    source: torch.Tensor,
    target: torch.Tensor,
    beta: float,
    alpha: float,
    max_iterations: int,
    tolerance: float,
):
    """
    Transform SOURCE into TARGET coordinates.

    pycpd convention:
        X = fixed target
        Y = moving source
    """

    x = target.detach().cpu().numpy().astype(np.float64)
    y = source.detach().cpu().numpy().astype(np.float64)

    reg = DeformableRegistration(
        X=x,
        Y=y,
        beta=beta,
        alpha=alpha,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    transformed, _ = reg.register()

    return torch.from_numpy(
        transformed.astype(np.float32)
    )


def evaluate_fold(
    dataset_root: Path,
    beta: float,
    alpha: float,
    max_iterations: int,
    tolerance: float,
):
    index = PairIndex(
        dataset_root,
        "test",
        min_shared=20,
    )

    cache = WormCache(activity_length=128)

    totals = {
        "queries": 0,
        "top1": 0,
        "top5": 0,
        "rr": 0.0,
        "hc": 0,
        "hq": 0,
    }

    pair_rows = []

    for pair_idx, (path_a, path_b) in enumerate(
        index.pairs, 1
    ):
        sample_a, sample_b, targets = build_pair_targets(
            cache.get(path_a),
            cache.get(path_b),
        )

        # ----------------------------------------------------------
        # Locked manuscript XYZ preprocessing.
        # ----------------------------------------------------------
        xyz_a = standardize_xyz(sample_a.xyz)
        xyz_b = standardize_xyz(sample_b.xyz)

        # ----------------------------------------------------------
        # CPD A -> B
        # ----------------------------------------------------------
        transformed_a = register_deformable(
            source=xyz_a,
            target=xyz_b,
            beta=beta,
            alpha=alpha,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

        scores_ab = -torch.cdist(
            transformed_a,
            xyz_b,
        ).square()

        # ----------------------------------------------------------
        # CPD B -> A
        # ----------------------------------------------------------
        transformed_b = register_deformable(
            source=xyz_b,
            target=xyz_a,
            beta=beta,
            alpha=alpha,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

        scores_ba = -torch.cdist(
            transformed_b,
            xyz_a,
        ).square()

        # Directional metrics use the corresponding CPD direction.
        left = directional(
            scores_ab,
            targets.row_target,
        )

        right = directional(
            scores_ba,
            targets.col_target,
        )

        # ----------------------------------------------------------
        # One symmetric score matrix for Hungarian.
        #
        # Both directions are already standardized using the same
        # locked preprocessing and expressed as negative squared
        # distance, so averaging is well-defined.
        # ----------------------------------------------------------
        scores_h = 0.5 * (
            scores_ab + scores_ba.transpose(0, 1)
        )

        hc, hq = hungarian(
            scores_h,
            targets.row_target,
            targets.col_target,
        )

        pair = {
            "uid_a": sample_a.uid,
            "uid_b": sample_b.uid,
            "queries": int(
                left["queries"] + right["queries"]
            ),
            "top1_correct": int(
                left["top1"] + right["top1"]
            ),
            "top5_correct": int(
                left["top5"] + right["top5"]
            ),
            "reciprocal_rank_sum": float(
                left["rr"] + right["rr"]
            ),
            "hungarian_correct": int(hc),
            "hungarian_queries": int(hq),
        }

        pair_rows.append(pair)

        totals["queries"] += pair["queries"]
        totals["top1"] += pair["top1_correct"]
        totals["top5"] += pair["top5_correct"]
        totals["rr"] += pair["reciprocal_rank_sum"]
        totals["hc"] += hc
        totals["hq"] += hq

        print(
            f"[{pair_idx:02d}/{len(index.pairs):02d}] "
            f"{path_a.name} <-> {path_b.name} "
            f"Q={pair['queries']}",
            flush=True,
        )

    q = max(int(totals["queries"]), 1)
    hq = max(int(totals["hq"]), 1)

    return {
        "method": "Deformable CPD",
        "dataset_root": str(dataset_root.resolve()),
        "split": "test",
        "pairs": len(pair_rows),
        "queries": int(totals["queries"]),
        "top1_real": float(totals["top1"] / q),
        "top5_real": float(totals["top5"] / q),
        "mrr_real": float(totals["rr"] / q),
        "hungarian_queries": int(totals["hq"]),
        "hungarian_accuracy": float(totals["hc"] / hq),
        "cpd": {
            "type": "deformable",
            "beta": beta,
            "alpha": alpha,
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "xyz_preprocessing":
                "mprt_net.relations.standardize_xyz",
            "directional_registration":
                "independent A->B and B->A",
            "hungarian_score":
                "mean of bidirectional negative squared distances",
        },
    }, pair_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "Data/Zebrafish_MPRT_LOFO8_60m"
        ),
    )

    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(
            "runs/mprt_v1_1/"
            "zebrafish_lofo8_seed42"
        ),
    )

    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
    )

    args = parser.parse_args()

    all_rows = []

    for fold in args.folds:

        print()
        print("=" * 72)
        print(f"ZEBRAFISH CPD — FOLD {fold}")
        print("=" * 72)

        dataset_root = (
            args.data_root / f"fold_{fold}"
        )

        baseline_dir = (
            args.run_root
            / "baselines"
            / f"fold_{fold}"
        )

        result, pair_rows = evaluate_fold(
            dataset_root,
            beta=args.beta,
            alpha=args.alpha,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
        )

        baseline_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output = baseline_dir / "cpd.json"
        output.write_text(
            json.dumps(result, indent=2) + "\n"
        )

        pair_output = (
            baseline_dir / "cpd_pairs.jsonl"
        )

        with pair_output.open("w") as handle:
            for row in pair_rows:
                handle.write(
                    json.dumps(row) + "\n"
                )

        print()
        print(
            f"[DONE] fold={fold} "
            f"Q={result['queries']} "
            f"Top1={100*result['top1_real']:.2f}% "
            f"Top5={100*result['top5_real']:.2f}% "
            f"MRR={result['mrr_real']:.4f} "
            f"Hung={100*result['hungarian_accuracy']:.2f}%"
        )

        all_rows.append({
            "fold": fold,
            **{
                k: result[k]
                for k in [
                    "queries",
                    "top1_real",
                    "top5_real",
                    "mrr_real",
                    "hungarian_accuracy",
                ]
            },
        })

    # --------------------------------------------------------------
    # Aggregate across held-out fish
    # --------------------------------------------------------------
    if len(all_rows) > 1:

        print()
        print("=" * 72)
        print(
            "CPD SUMMARY — MEAN ± SD ACROSS HELD-OUT FISH"
        )
        print("=" * 72)

        for metric in [
            "top1_real",
            "top5_real",
            "mrr_real",
            "hungarian_accuracy",
        ]:
            values = np.asarray(
                [r[metric] for r in all_rows],
                dtype=float,
            )

            mean = values.mean()
            sd = values.std(ddof=1)

            if metric == "mrr_real":
                print(
                    f"{metric:20s}: "
                    f"{mean:.4f} ± {sd:.4f}"
                )
            else:
                print(
                    f"{metric:20s}: "
                    f"{100*mean:.2f} ± "
                    f"{100*sd:.2f}%"
                )


if __name__ == "__main__":
    main()
