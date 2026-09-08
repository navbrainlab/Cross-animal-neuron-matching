#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
PKG = ROOT / "neurid"

DATASETS = {
    "atanas": (
        ROOT /
        "Data/Atanas_SF_unified_000776/cv5_grouped_v1"
    ),
    "rld": (
        ROOT /
        "Data/Dunn_001623/cv5_grouped_v1"
    ),
}

RUN_ROOT = (
    ROOT /
    "runs/mprt_v1_1_dynamic_residual_atlas_cv5x3_v1"
)

OUT_ROOT = (
    ROOT /
    "runs/mprt_static_repr_only_ablation_cv5x3_v1"
)

SEEDS = [1, 42, 123]
FOLDS = range(5)


def target_slots(sample, identity_to_slot):
    from mprt_net.data import unique_identity_map

    targets = torch.full(
        (sample.num_nodes,),
        -1,
        dtype=torch.long,
        device=sample.xyz.device,
    )

    for identity, node_index in unique_identity_map(sample).items():
        slot = identity_to_slot.get(str(identity))
        if slot is not None:
            targets[node_index] = int(slot)

    return targets


@torch.inference_mode()
def repr_only_match(model, query, atlas):
    """
    Representation-only unary matching.

    IMPORTANT:
      * query representation unchanged
      * atlas representation unchanged
      * geometry/activity information ALREADY encoded into Z is retained
      * explicit coordinate term is absent
      * relation transport is retained exactly
      * Sinkhorn/dustbin parameters are retained exactly
    """

    from mprt_net.sinkhorn import contracted_relation_cost

    qa = F.normalize(query.nodes, dim=-1)
    kb = F.normalize(atlas.nodes, dim=-1)

    # ----------------------------------------------------------
    # REPR-ONLY unary:
    #
    #   similarity(z_i, z*_k) / tau_Z
    #
    # NO explicit ||x_i - x*_k||^2 term.
    # ----------------------------------------------------------
    unary_logits = (
        qa @ kb.transpose(0, 1)
    ) / model.unary_temperature

    # Respect static-atlas support if present.
    support = getattr(atlas, "support", None)

    if support is not None:
        valid_atlas = support > 0

        unary_logits = unary_logits.masked_fill(
            ~valid_atlas[None, :],
            -1.0e4,
        )
    else:
        valid_atlas = None

    final_logits = unary_logits

    solved = model._solve(final_logits)

    relation_costs = []

    if model.config.use_relation_transport:

        for _ in range(model.config.transport_steps):

            relation_cost = contracted_relation_cost(
                query.relations,
                atlas.relations,
                solved.plan[:-1, :-1],
            )

            structural_logits = (
                -relation_cost /
                model.relation_temperature
            )

            if valid_atlas is not None:
                structural_logits = (
                    structural_logits.masked_fill(
                        ~valid_atlas[None, :],
                        -1.0e4,
                    )
                )

            relation_costs.append(relation_cost)

            final_logits = (
                unary_logits
                + model.structural_weight
                * structural_logits
            )

            if valid_atlas is not None:
                final_logits = final_logits.masked_fill(
                    ~valid_atlas[None, :],
                    -1.0e4,
                )

            solved = model._solve(final_logits)

    nq, na = unary_logits.shape

    row_conditional = (
        solved.plan[:nq, :]
        / solved.mu[:nq, None]
    )

    col_conditional = (
        solved.plan[:, :na]
        / solved.nu[None, :na]
    ).transpose(0, 1)

    # Use the current production output object as template.
    # This keeps compatibility if MPRTOutput has extra fields.
    full_template = model.match_encodings(query, atlas)

    return dataclasses.replace(
        full_template,
        plan=solved.plan,
        log_plan=solved.log_plan,
        row_conditional=row_conditional,
        col_conditional=col_conditional,
        unary_logits=unary_logits,
        final_logits=final_logits,
        relation_costs=tuple(relation_costs),
    )


def query_metrics(output, targets):
    probs = output.row_conditional.detach()

    real = probs[:, :-1]

    valid = (
        (targets >= 0)
        & (targets < real.shape[1])
    )

    idx = torch.nonzero(
        valid,
        as_tuple=False,
    ).flatten()

    rows = {}

    if idx.numel() == 0:
        return rows

    rr = real.index_select(0, idx)
    pp = probs.index_select(0, idx)
    tt = targets.index_select(0, idx)

    ts = rr.gather(1, tt[:, None])

    rank = (
        1
        + (rr > ts).sum(dim=1)
    )

    pred = rr.argmax(dim=1)

    dust = (
        pp[:, -1]
        > rr.amax(dim=1)
    )

    # Hungarian on full graph.
    ri, ci = linear_sum_assignment(
        -output.plan[:-1, :-1]
        .detach()
        .cpu()
        .numpy()
    )

    assignment = {
        int(r): int(c)
        for r, c in zip(ri, ci)
    }

    for off, node in enumerate(
        idx.detach().cpu().tolist()
    ):
        target = int(tt[off])

        rows[int(node)] = {
            "target": target,
            "pred": int(pred[off]),
            "rank": int(rank[off]),
            "top1": int(rank[off] <= 1),
            "top5": int(
                rank[off]
                <= min(5, real.shape[1])
            ),
            "rr": float(
                1.0 / float(rank[off])
            ),
            "dustbin_top1": int(dust[off]),
            "hungarian": int(
                assignment.get(
                    int(node), -1
                )
                == target
            ),
        }

    return rows


def summarize(rows):
    if not rows:
        return {
            "queries": 0,
            "top1": float("nan"),
            "top5": float("nan"),
            "mrr": float("nan"),
            "hungarian": float("nan"),
            "dustbin_top1": float("nan"),
        }

    return {
        "queries": len(rows),
        "top1": np.mean(
            [r["top1"] for r in rows]
        ),
        "top5": np.mean(
            [r["top5"] for r in rows]
        ),
        "mrr": np.mean(
            [r["rr"] for r in rows]
        ),
        "hungarian": np.mean(
            [r["hungarian"] for r in rows]
        ),
        "dustbin_top1": np.mean(
            [r["dustbin_top1"] for r in rows]
        ),
    }


@torch.inference_mode()
def run_cell(dataset, fold, seed, device):
    from mprt_net.data import (
        WormCache,
        split_files,
    )
    from mprt_net.evaluate import load_checkpoint

    data_root = (
        DATASETS[dataset]
        / f"fold_{fold}"
    )

    checkpoint = (
        RUN_ROOT
        / dataset
        / f"fold{fold}"
        / f"seed{seed}"
        / "static_atlas"
        / "anchored_pure.pt"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model, payload = load_checkpoint(
        checkpoint,
        device,
    )

    model.eval()

    if not model.atlas_is_initialized:
        raise RuntimeError(
            f"No initialized atlas: {checkpoint}"
        )

    mapping = payload.get(
        "atlas_identity_to_slot"
    )

    if not isinstance(mapping, dict):
        raise RuntimeError(
            "Missing atlas_identity_to_slot"
        )

    identity_to_slot = {
        str(k): int(v)
        for k, v in mapping.items()
    }

    atlas = model.atlas_encoding()

    files = split_files(
        data_root,
        "test",
    )

    cache = WormCache(
        activity_length=512,
        max_items=2,
    )

    full_rows = []
    repr_rows = []
    query_csv = []

    unary_abs_diffs = []
    unary_max_diffs = []

    for path in files:

        sample_cpu = cache.get(path)
        sample = sample_cpu.to(device)

        query = model.encode_population(sample)

        # Current production matcher.
        full = model.match_encodings(
            query,
            atlas,
        )

        # Repr-only intervention.
        repr_only = repr_only_match(
            model,
            query,
            atlas,
        )

        targets = target_slots(
            sample,
            identity_to_slot,
        )

        a = query_metrics(
            full,
            targets,
        )

        b = query_metrics(
            repr_only,
            targets,
        )

        # Hard guard: exact same query universe.
        if set(a) != set(b):
            raise RuntimeError(
                f"Query mismatch {sample_cpu.uid}"
            )

        # ------------------------------------------------------
        # Critical audit:
        # If this is exactly zero, production unary is already
        # representation-only.
        # ------------------------------------------------------
        ud = (
            full.unary_logits
            - repr_only.unary_logits
        ).abs()

        unary_abs_diffs.append(
            float(ud.mean().cpu())
        )
        unary_max_diffs.append(
            float(ud.max().cpu())
        )

        for node in sorted(a):

            ra = a[node]
            rb = b[node]

            full_rows.append(ra)
            repr_rows.append(rb)

            query_csv.append({
                "dataset": dataset,
                "fold": fold,
                "seed": seed,
                "worm": sample_cpu.uid,
                "node": node,
                "full_top1": ra["top1"],
                "repr_top1": rb["top1"],
                "full_rank": ra["rank"],
                "repr_rank": rb["rank"],
                "full_hungarian": ra["hungarian"],
                "repr_hungarian": rb["hungarian"],
            })

    fa = summarize(full_rows)
    rb = summarize(repr_rows)

    if fa["queries"] != rb["queries"]:
        raise RuntimeError(
            "Full/repr query denominator mismatch"
        )

    result = {
        "dataset": dataset,
        "fold": fold,
        "seed": seed,
        "split": "test",
        "checkpoint": str(checkpoint),
        "queries": fa["queries"],
        "full": fa,
        "repr_only": rb,
        "delta_repr_minus_full": {
            "top1": (
                rb["top1"]
                - fa["top1"]
            ),
            "top5": (
                rb["top5"]
                - fa["top5"]
            ),
            "mrr": (
                rb["mrr"]
                - fa["mrr"]
            ),
            "hungarian": (
                rb["hungarian"]
                - fa["hungarian"]
            ),
        },
        "unary_audit": {
            "mean_abs_difference": float(
                np.mean(unary_abs_diffs)
            ),
            "max_abs_difference": float(
                np.max(unary_max_diffs)
            ),
            "interpretation": (
                "zero means production unary "
                "was already representation-only"
            ),
        },
    }

    return result, query_csv


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--datasets",
        nargs="+",
        default=["atanas", "rld"],
        choices=["atanas", "rld"],
    )

    p.add_argument(
        "--device",
        default="cuda",
    )

    args = p.parse_args()

    sys.path.insert(
        0,
        str(PKG),
    )

    device = torch.device(args.device)

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for dataset in args.datasets:

        for fold in FOLDS:

            for seed in SEEDS:

                print()
                print("=" * 100)
                print(
                    dataset.upper(),
                    f"fold{fold}",
                    f"seed{seed}",
                )
                print("=" * 100)

                result, rows = run_cell(
                    dataset,
                    fold,
                    seed,
                    device,
                )

                out = (
                    OUT_ROOT
                    / dataset
                    / f"fold{fold}"
                    / f"seed{seed}"
                )

                out.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                (
                    out / "result.json"
                ).write_text(
                    json.dumps(
                        result,
                        indent=2,
                    )
                )

                with (
                    out / "queries.csv"
                ).open(
                    "w",
                    newline="",
                ) as f:

                    writer = csv.DictWriter(
                        f,
                        fieldnames=list(
                            rows[0].keys()
                        ),
                    )

                    writer.writeheader()
                    writer.writerows(rows)

                all_results.append(result)

                f = result["full"]
                r = result["repr_only"]
                d = result[
                    "delta_repr_minus_full"
                ]

                print(
                    f"Q={result['queries']} "
                    f"Full={100*f['top1']:.2f}% "
                    f"ReprOnly={100*r['top1']:.2f}% "
                    f"Δ={100*d['top1']:+.2f}pp "
                    f"HungΔ={100*d['hungarian']:+.2f}pp "
                    f"UnaryDiff="
                    f"{result['unary_audit']['max_abs_difference']:.6g}"
                )

    (
        OUT_ROOT / "all_results.json"
    ).write_text(
        json.dumps(
            all_results,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
