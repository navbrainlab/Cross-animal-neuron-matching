#!/usr/bin/env python3

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def load_stage1_module(path: Path):
    name = "_zebrafish_nuclr_stage1"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def scalar_str(x):
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.item()
    elif x.size == 1:
        x = x.reshape(-1)[0]
    if isinstance(x, bytes):
        x = x.decode()
    return str(x)


def valid_identity(x):
    s = str(x).strip()
    return s not in {"", "nan", "None", "none", "-1", "NA", "N/A"}


def load_record(path: Path):
    with np.load(path, allow_pickle=True) as d:
        activity = np.asarray(d["activity_raw"], dtype=np.float32)
        ids = np.asarray(d["cell_id"]).reshape(-1)

        if activity.ndim != 2:
            raise ValueError(f"{path}: activity shape={activity.shape}")

        if activity.shape[0] != len(ids) and activity.shape[1] == len(ids):
            activity = activity.T

        if activity.shape[0] != len(ids):
            raise ValueError(
                f"{path}: activity={activity.shape}, cell_id={ids.shape}"
            )

        n = len(ids)

        # Same candidate population convention as the locked MPRT data:
        # retain finite/valid XYZ population nodes.
        if "valid_xyz_mask" in d:
            candidate = np.asarray(d["valid_xyz_mask"], dtype=bool).reshape(-1)
        elif "xyz" in d:
            xyz = np.asarray(d["xyz"])
            candidate = np.isfinite(xyz).all(axis=1)
        else:
            candidate = np.ones(n, dtype=bool)

        labeled = (
            np.asarray(d["labeled_mask"], dtype=bool).reshape(-1)
            if "labeled_mask" in d else np.ones(n, dtype=bool)
        )
        certain = (
            np.asarray(d["certain_mask"], dtype=bool).reshape(-1)
            if "certain_mask" in d else np.ones(n, dtype=bool)
        )
        clean = (
            np.asarray(d["clean_mask"], dtype=bool).reshape(-1)
            if "clean_mask" in d else np.ones(n, dtype=bool)
        )

        id_ok = np.asarray([valid_identity(x) for x in ids], dtype=bool)

        supervised = candidate & labeled & certain & clean & id_ok

        pair_id = scalar_str(d["pair_id"]) if "pair_id" in d else path.stem
        if "side" in d:
            side = scalar_str(d["side"]).lower()
        elif "__q" in path.stem:
            side = "q"
        elif "__r" in path.stem:
            side = "r"
        else:
            raise ValueError(f"{path}: cannot determine q/r side")

    # Candidate nodes remain in the score matrix.
    activity = activity[candidate]
    ids = ids[candidate]
    supervised = supervised[candidate]

    return {
        "path": path,
        "pair_id": pair_id,
        "side": side,
        "activity": activity,
        "ids": np.asarray([str(x) for x in ids]),
        "supervised": supervised,
    }


def load_pairs(root: Path):
    grouped = {}

    for path in sorted(root.glob("*.npz")):
        rec = load_record(path)
        p = grouped.setdefault(rec["pair_id"], {})
        if rec["side"] in p:
            raise RuntimeError(
                f"Duplicate side {rec['side']} for pair {rec['pair_id']}"
            )
        p[rec["side"]] = rec

    pairs = []
    for pair_id, sides in sorted(grouped.items()):
        if set(sides) != {"q", "r"}:
            raise RuntimeError(
                f"Incomplete pair {pair_id}: sides={list(sides)}"
            )
        pairs.append((sides["q"], sides["r"]))

    if not pairs:
        raise RuntimeError(f"No q/r pairs found under {root}")

    return pairs


def build_targets(a, b):
    ids_a = a["ids"]
    ids_b = b["ids"]
    sup_a = a["supervised"]
    sup_b = b["supervised"]

    map_a = {}
    map_b = {}

    for i, x in enumerate(ids_a):
        if not sup_a[i]:
            continue
        if x in map_a:
            raise RuntimeError(f"Duplicate supervised ID in q: {x}")
        map_a[x] = i

    for j, x in enumerate(ids_b):
        if not sup_b[j]:
            continue
        if x in map_b:
            raise RuntimeError(f"Duplicate supervised ID in r: {x}")
        map_b[x] = j

    common = sorted(set(map_a) & set(map_b))

    row_targets = [(map_a[x], map_b[x]) for x in common]
    col_targets = [(map_b[x], map_a[x]) for x in common]

    return row_targets, col_targets


def directional_ranks(scores, targets):
    ranks = []

    for i, j in targets:
        gt = scores[i, j]
        # Locked MPRT convention: optimistic tie handling.
        rank = 1 + int(np.sum(scores[i] > gt))
        ranks.append(rank)

    return ranks


def metrics_from_ranks(ranks):
    x = np.asarray(ranks, dtype=np.int64)
    return {
        "queries": int(len(x)),
        "top1": float(np.mean(x <= 1)),
        "top3": float(np.mean(x <= 3)),
        "top5": float(np.mean(x <= 5)),
        "mrr": float(np.mean(1.0 / x)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--stage1-script", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-queries", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args_cli = ap.parse_args()

    device = torch.device(args_cli.device)

    stage1 = load_stage1_module(args_cli.stage1_script)

    # ------------------------------------------------------------------
    # xFormers compatibility shim
    #
    # Official NuCLR calls:
    #   BlockDiagonalMask.from_seqlens(..., device=device)
    #
    # Some xFormers releases removed the explicit `device` keyword.
    # Keep the official NuCLR source untouched and adapt only the API
    # boundary in this evaluator.
    # ------------------------------------------------------------------
    import inspect
    import xformers.ops as xops

    _orig_from_seqlens = xops.fmha.BlockDiagonalMask.from_seqlens

    try:
        _from_seqlens_params = inspect.signature(
            _orig_from_seqlens
        ).parameters
    except (TypeError, ValueError):
        _from_seqlens_params = {}

    if "device" not in _from_seqlens_params:
        def _from_seqlens_compat(*args, **kwargs):
            kwargs.pop("device", None)
            return _orig_from_seqlens(*args, **kwargs)

        xops.fmha.BlockDiagonalMask.from_seqlens = _from_seqlens_compat
        print(
            "xFormers compatibility: "
            "ignoring unsupported device= in "
            "BlockDiagonalMask.from_seqlens"
        )

    ckpt = torch.load(
        args_cli.checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    train_args = ckpt["args"]
    import argparse as _argparse
    model_args = _argparse.Namespace(**train_args)

    model = stage1.build_official_model(model_args, device)
    stage1.load_model_state(model, ckpt)
    model.eval()

    print("checkpoint :", args_cli.checkpoint)
    print("epoch      :", ckpt.get("epoch"))
    print("global_step:", ckpt.get("global_step"))
    print("embedding  : backbone output (projector NOT used)")
    print("similarity : cosine")

    @torch.inference_mode()
    def encode(activity: np.ndarray):
        if activity.shape[1] != model_args.window_frames:
            raise RuntimeError(
                f"Expected {model_args.window_frames} frames, "
                f"got {activity.shape[1]}"
            )

        traces = stage1.normalize_traces(
            activity,
            model_args.normalization,
        )

        record = stage1.WormRecord(
            path=Path("locked_eval.npz"),
            worm_name="locked_eval",
            traces=traces,
        )

        # Reuse the exact official input packing code.
        # min_unit_fraction=1.0 disables unit dropout for evaluation.
        prepared = stage1.prepare_two_view_batch(
            [record],
            model_args.window_frames,
            model_args.patch_size,
            -1,
            1,
            0.0,
            model_args.min_overlap_frames,
            1.0,
            np.random.default_rng(0),
            device,
        )

        with stage1.autocast_context(device, model_args.precision):
            y = model(
                bins=prepared.bins1,
                unit_seqlen=prepared.unit_seqlen1,
            )

        if isinstance(y, (tuple, list)):
            if len(y) != 1:
                raise RuntimeError(
                    f"Unexpected NuCLR tuple/list output length={len(y)}"
                )
            y = y[0]

        if not torch.is_tensor(y):
            raise RuntimeError(f"Unexpected NuCLR output type: {type(y)}")

        y = y.float()

        if y.ndim != 2:
            raise RuntimeError(
                f"Expected NuCLR neuron embeddings [N,D], got {tuple(y.shape)}"
            )

        if y.shape[0] != activity.shape[0]:
            raise RuntimeError(
                f"Embedding neuron count mismatch: "
                f"input={activity.shape[0]}, output={y.shape[0]}"
            )

        return F.normalize(y, p=2, dim=-1).cpu().numpy()


    pairs = load_pairs(args_cli.test_root)

    all_ranks = []
    hung_correct = 0
    hung_queries = 0
    pair_rows = []

    for pair_idx, (q, r) in enumerate(pairs):
        zq = encode(q["activity"])
        zr = encode(r["activity"])

        scores = zq @ zr.T

        row_targets, col_targets = build_targets(q, r)

        row_ranks = directional_ranks(scores, row_targets)
        col_ranks = directional_ranks(scores.T, col_targets)

        ranks = row_ranks + col_ranks
        all_ranks.extend(ranks)

        # One global one-to-one assignment on the same score matrix.
        rr, cc = linear_sum_assignment(-scores)
        assignment = {int(i): int(j) for i, j in zip(rr, cc)}
        reverse_assignment = {int(j): int(i) for i, j in zip(rr, cc)}

        pair_correct = 0
        pair_queries = 0

        for i, j in row_targets:
            pair_correct += int(assignment.get(i, -1) == j)
            pair_queries += 1

        for j, i in col_targets:
            pair_correct += int(reverse_assignment.get(j, -1) == i)
            pair_queries += 1

        hung_correct += pair_correct
        hung_queries += pair_queries

        pm = metrics_from_ranks(ranks)

        pair_rows.append({
            "pair_id": q["pair_id"],
            "q_file": q["path"].name,
            "r_file": r["path"].name,
            "n_q": int(len(q["ids"])),
            "n_r": int(len(r["ids"])),
            "queries": pm["queries"],
            "top1": pm["top1"],
            "top5": pm["top5"],
            "mrr": pm["mrr"],
            "hungarian_accuracy":
                float(pair_correct / pair_queries)
                if pair_queries else float("nan"),
        })

        print(
            f"[{pair_idx+1:02d}/{len(pairs):02d}] "
            f"{q['pair_id']} "
            f"N={len(q['ids'])}x{len(r['ids'])} "
            f"Q={pm['queries']} "
            f"Top1={100*pm['top1']:.2f}% "
            f"Hung={100*pair_rows[-1]['hungarian_accuracy']:.2f}%"
        )

    final = metrics_from_ranks(all_ranks)
    final["hungarian_queries"] = int(hung_queries)
    final["hungarian_accuracy"] = float(hung_correct / hung_queries)
    final["pairs"] = len(pairs)
    final["method"] = "NuCLR official calcium SSL backbone cosine"
    final["checkpoint"] = str(args_cli.checkpoint)
    final["checkpoint_epoch"] = ckpt.get("epoch")
    final["checkpoint_global_step"] = ckpt.get("global_step")
    final["embedding"] = "NuclrV2gaCa2 backbone output"
    final["projector_used"] = False
    final["similarity"] = "cosine"
    final["pair_metrics"] = pair_rows

    if args_cli.expected_queries is not None:
        if final["queries"] != args_cli.expected_queries:
            raise RuntimeError(
                f"LOCKED QUERY MISMATCH: "
                f"NuCLR={final['queries']} "
                f"expected={args_cli.expected_queries}"
            )

        if hung_queries != args_cli.expected_queries:
            raise RuntimeError(
                f"HUNGARIAN QUERY MISMATCH: "
                f"{hung_queries} vs {args_cli.expected_queries}"
            )

    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    args_cli.output.write_text(
        json.dumps(final, indent=2) + "\n"
    )

    print()
    print("=" * 100)
    print("ZEBRAFISH LOFO FOLD 1 — NuCLR LOCKED TEST")
    print("=" * 100)
    print(f"pairs    : {final['pairs']}")
    print(f"queries  : {final['queries']}")
    print(f"Top1     : {100*final['top1']:.4f}%")
    print(f"Top3     : {100*final['top3']:.4f}%")
    print(f"Top5     : {100*final['top5']:.4f}%")
    print(f"MRR      : {final['mrr']:.4f}")
    print(f"Hungarian: {100*final['hungarian_accuracy']:.4f}%")
    print(f"summary  : {args_cli.output}")


if __name__ == "__main__":
    main()
