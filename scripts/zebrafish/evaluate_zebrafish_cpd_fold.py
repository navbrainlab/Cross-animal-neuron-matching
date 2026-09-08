#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from pycpd import DeformableRegistration

from mprt_net.data import PairIndex, WormCache, build_pair_targets
from mprt_net.metrics import MetricTotals, pair_metrics


def norm_xyz(x, mode):
    x = x.float()

    if mode == "raw":
        return x

    xc = x - x.mean(0, keepdim=True)

    if mode == "zscore":
        return xc / x.std(0, unbiased=False, keepdim=True).clamp_min(1e-8)

    if mode == "zscore_sample":
        return xc / x.std(0, unbiased=True, keepdim=True).clamp_min(1e-8)

    if mode == "global_std":
        return xc / xc.std(unbiased=False).clamp_min(1e-8)

    if mode == "rms":
        return xc / torch.sqrt(xc.square().mean()).clamp_min(1e-8)

    raise ValueError(mode)


def make_output(score_ab, score_ba):
    """
    score_ab: A -> B, shape [Na,Nb]
    score_ba: B -> A, shape [Nb,Na]
    """

    # pair_metrics only uses ordering for Top-k / MRR.
    # Add a dustbin score lower than every real candidate.
    dust_a = score_ab.amin(1, keepdim=True) - 1.0
    dust_b = score_ba.amin(1, keepdim=True) - 1.0

    row = torch.cat([score_ab, dust_a], 1)
    col = torch.cat([score_ba, dust_b], 1)

    # One symmetric score matrix for the shared Hungarian evaluation.
    joint = 0.5 * (score_ab + score_ba.T)

    plan = torch.zeros(
        (joint.shape[0] + 1, joint.shape[1] + 1),
        dtype=torch.float32,
    )
    plan[:-1, :-1] = joint

    return SimpleNamespace(
        row_conditional=row,
        col_conditional=col,
        plan=plan,
    )


def euclidean_output(a, b, mode):
    xa = norm_xyz(a.xyz, mode)
    xb = norm_xyz(b.xyz, mode)

    d = torch.cdist(xa, xb)

    return make_output(-d, -d.T)


def cpd_transform(source, target, beta, alpha, max_iter):
    reg = DeformableRegistration(
        X=target.astype(np.float64),
        Y=source.astype(np.float64),
        beta=beta,
        alpha=alpha,
        max_iterations=max_iter,
        tolerance=1e-5,
    )
    transformed, _ = reg.register()
    return transformed.astype(np.float32)


def cpd_output(a, b, mode, beta, alpha, max_iter):
    xa = norm_xyz(a.xyz, mode).cpu().numpy()
    xb = norm_xyz(b.xyz, mode).cpu().numpy()

    # A -> B
    ta = cpd_transform(xa, xb, beta, alpha, max_iter)
    dab = torch.cdist(
        torch.from_numpy(ta),
        torch.from_numpy(xb.astype(np.float32)),
    )

    # B -> A
    tb = cpd_transform(xb, xa, beta, alpha, max_iter)
    dba = torch.cdist(
        torch.from_numpy(tb),
        torch.from_numpy(xa.astype(np.float32)),
    )

    # Scale each direction without changing ranks.
    sa = dab[dab > 0].median().clamp_min(1e-8)
    sb = dba[dba > 0].median().clamp_min(1e-8)

    return make_output(-dab / sa, -dba / sb)


def evaluate(root, method, mode, args):
    index = PairIndex(root, "test", min_shared=20)
    cache = WormCache(activity_length=128)
    totals = MetricTotals()

    print("pairs =", len(index.pairs))

    for i, (pa, pb) in enumerate(index.pairs, 1):
        base_a = cache.get(pa)
        base_b = cache.get(pb)

        a, b, targets = build_pair_targets(base_a, base_b)

        if method == "euclidean":
            out = euclidean_output(a, b, mode)
        else:
            out = cpd_output(
                a, b, mode,
                args.beta,
                args.alpha,
                args.max_iter,
            )

        totals.update(pair_metrics(out, targets))

        print(
            f"[{i:02d}/{len(index.pairs):02d}] "
            f"{pa.name} <-> {pb.name}",
            flush=True,
        )

    return totals.compute()


def metric_diff(x, y):
    keys = [
        "top1_real",
        "top5_real",
        "mrr_real",
        "hungarian_accuracy",
    ]
    return max(abs(float(x[k]) - float(y[k])) for k in keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--max-iter", type=int, default=100)
    args = p.parse_args()

    data_root = Path("Data/Zebrafish_MPRT_LOFO8_60m")
    run_root = Path("runs/mprt_v1_1/zebrafish_lofo8_seed42")

    root = data_root / f"fold_{args.fold}"

    ref_path = (
        run_root / "baselines"
        / f"fold_{args.fold}"
        / "euclidean.json"
    )

    ref = json.loads(ref_path.read_text())

    print("=" * 70)
    print("STEP 1: RECOVER LOCKED EUCLIDEAN NORMALIZATION")
    print("=" * 70)

    modes = [
        "zscore",
        "zscore_sample",
        "global_std",
        "rms",
        "raw",
    ]

    candidates = []

    for mode in modes:
        print("\nTesting:", mode)

        m = evaluate(root, "euclidean", mode, args)
        diff = metric_diff(m, ref)

        candidates.append((diff, mode, m))

        print(
            f"{mode}: "
            f"Q={m['queries']} "
            f"Top1={100*m['top1_real']:.4f}% "
            f"Top5={100*m['top5_real']:.4f}% "
            f"MRR={m['mrr_real']:.6f} "
            f"Hung={100*m['hungarian_accuracy']:.4f}% "
            f"diff={diff:.3e}"
        )

    candidates.sort(key=lambda x: x[0])
    diff, mode, eu = candidates[0]

    print("\n" + "=" * 70)
    print("BEST NORMALIZATION =", mode)
    print("MAX METRIC DIFF    =", diff)
    print("=" * 70)

    if eu["queries"] != ref["queries"]:
        raise RuntimeError(
            f"Query mismatch: {eu['queries']} vs {ref['queries']}"
        )

    if diff > 1e-7:
        raise RuntimeError(
            "Existing Euclidean baseline was NOT exactly reproduced. "
            "Stop here and send this output to ChatGPT."
        )

    print("\nEuclidean protocol reproduced exactly: PASS")

    print("\n" + "=" * 70)
    print("STEP 2: CPD")
    print("=" * 70)

    result = evaluate(root, "cpd", mode, args)

    if result["queries"] != ref["queries"]:
        raise RuntimeError(
            f"CPD query mismatch: "
            f"{result['queries']} vs {ref['queries']}"
        )

    out = (
        run_root / "baselines"
        / f"fold_{args.fold}"
        / "cpd.json"
    )

    payload = {
        "method": "Deformable CPD",
        "fold": args.fold,
        "split": "test",
        "xyz_normalization": mode,
        "cpd_beta": args.beta,
        "cpd_alpha": args.alpha,
        "cpd_max_iter": args.max_iter,
        **result,
    }

    out.write_text(json.dumps(payload, indent=2) + "\n")

    print("\n" + "=" * 70)
    print("CPD RESULT")
    print("=" * 70)
    print("Queries   :", result["queries"])
    print(f"Top-1     : {100*result['top1_real']:.2f}%")
    print(f"Top-5     : {100*result['top5_real']:.2f}%")
    print(f"MRR       : {result['mrr_real']:.4f}")
    print(f"Hungarian : {100*result['hungarian_accuracy']:.2f}%")
    print("Saved     :", out)


if __name__ == "__main__":
    main()
