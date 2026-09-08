from pathlib import Path
import json
import numpy as np
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment

RUN_ROOT = Path(
    "runs/crfid_official_atanas_cv5_v1"
)


def mstr(x):
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, np.str_):
        return str(x).strip()
    if isinstance(x, bytes):
        return x.decode().strip()
    if isinstance(x, np.ndarray):
        x = np.squeeze(x)
        if x.ndim == 0:
            return mstr(x.item())
        if x.dtype.kind in ("U", "S"):
            return "".join(
                str(v) for v in x.reshape(-1)
            ).strip()
        if x.size == 1:
            return mstr(x.item())
    return str(x).strip()


def canonical_id(x):
    return str(x).strip().rstrip("?")


def evaluate_worm(sidecar_path, output_path):
    side = np.load(
        sidecar_path,
        allow_pickle=True,
    )

    source = Path(
        str(side["source"].item())
    )

    orig_idx = np.asarray(
        side["orig_idx"],
        dtype=int,
    )

    z = np.load(
        source,
        allow_pickle=True,
    )

    gt_all = np.asarray(
        z["cell_id"]
    ).astype(str)

    labeled_all = np.asarray(
        z["labeled_mask"],
        dtype=bool,
    )

    out = loadmat(
        output_path,
        squeeze_me=True,
        struct_as_record=False,
    )

    B = np.asarray(
        out["conserved_nodeBel"],
        dtype=float,
    )

    candidates = [
        mstr(x)
        for x in np.asarray(
            out["Neuron_head"]
        ).reshape(-1)
    ]

    assert B.ndim == 2
    assert B.shape[0] == len(orig_idx)
    assert B.shape[1] == len(candidates)
    assert B.shape[1] == 178
    assert np.isfinite(B).all()

    row_sum = B.sum(axis=1)

    if not np.allclose(
        row_sum,
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            f"{output_path}: belief rows not normalized"
        )

    eval_local = np.where(
        labeled_all[orig_idx]
    )[0]

    gt = np.asarray([
        canonical_id(x)
        for x in gt_all[
            orig_idx[eval_local]
        ]
    ])

    scores = B[eval_local]
    Q = len(gt)

    state_to_idx = {
        name: j
        for j, name in enumerate(candidates)
    }

    covered = np.asarray([
        y in state_to_idx
        for y in gt
    ], dtype=bool)

    ranking = np.argsort(
        -scores,
        axis=1,
    )

    top1_correct = np.zeros(
        Q, dtype=bool
    )
    top5_correct = np.zeros(
        Q, dtype=bool
    )
    rr = np.zeros(
        Q, dtype=float
    )

    for i, y in enumerate(gt):
        if y not in state_to_idx:
            continue

        j = state_to_idx[y]

        pos = int(
            np.where(
                ranking[i] == j
            )[0][0]
        )

        top1_correct[i] = (
            pos == 0
        )

        top5_correct[i] = (
            pos < 5
        )

        rr[i] = (
            1.0 / (pos + 1)
        )

    # Unified Hungarian over ALL query neurons.
    #
    # IMPORTANT:
    # labeled_mask is used only for evaluation, never to decide
    # which query neurons participate in the assignment.
    rows_all, cols_all = linear_sum_assignment(-B)

    assigned_state = np.full(
        B.shape[0],
        -1,
        dtype=int,
    )

    assigned_state[rows_all] = cols_all

    hung_correct = np.zeros(
        Q,
        dtype=bool,
    )

    for i, node_row in enumerate(eval_local):
        y = gt[i]

        if y not in state_to_idx:
            continue

        c = assigned_state[node_row]

        if c < 0:
            continue

        hung_correct[i] = (
            candidates[c] == y
        )

    # Official CRF duplicate-resolution assignment,
    # diagnostic only.
    official_correct = np.zeros(
        Q,
        dtype=bool,
    )

    if "node_label" in out:
        node_label = np.asarray(
            out["node_label"]
        )

        if node_label.ndim == 1:
            node_label = node_label.reshape(-1, 1)

        if (
            node_label.shape[0] != B.shape[0]
            and
            node_label.shape[1] == B.shape[0]
        ):
            node_label = node_label.T

        if node_label.shape[0] == B.shape[0]:
            labels = node_label[
                eval_local, 0
            ].astype(int)

            for i, state1 in enumerate(labels):
                if state1 <= 0:
                    continue

                j = state1 - 1

                if (
                    j < len(candidates)
                    and
                    gt[i] in state_to_idx
                ):
                    official_correct[i] = (
                        candidates[j] == gt[i]
                    )

    return {
        "source": str(source),
        "Q": int(Q),
        "covered": int(covered.sum()),
        "top1_correct": int(
            top1_correct.sum()
        ),
        "top5_correct": int(
            top5_correct.sum()
        ),
        "rr_sum": float(rr.sum()),
        "hungarian_correct": int(
            hung_correct.sum()
        ),
        "official_correct": int(
            official_correct.sum()
        ),
        "missing_gt": sorted(
            set(gt[~covered])
        ),
    }


def fold_metrics(results):
    Q = sum(x["Q"] for x in results)

    covered = sum(
        x["covered"] for x in results
    )

    top1 = sum(
        x["top1_correct"]
        for x in results
    )

    top5 = sum(
        x["top5_correct"]
        for x in results
    )

    rr = sum(
        x["rr_sum"]
        for x in results
    )

    hung = sum(
        x["hungarian_correct"]
        for x in results
    )

    official = sum(
        x["official_correct"]
        for x in results
    )

    return {
        "Q": Q,
        "top1": top1 / Q,
        "top5": top5 / Q,
        "mrr": rr / Q,
        "hungarian": hung / Q,
        "coverage": covered / Q,
        "covered_top1": (
            top1 / covered
            if covered
            else float("nan")
        ),
        "covered_top5": (
            top5 / covered
            if covered
            else float("nan")
        ),
        "official_duplicate_accuracy": (
            official / Q
        ),
    }


def fmt_pct(x):
    return f"{100*x:.2f}%"


def main():
    all_folds = []

    print("=" * 110)
    print(
        "CRF_ID OFFICIAL CALCIUM + "
        "TRAIN-ONLY FRAME CALIBRATION — ATANAS CV5"
    )
    print("=" * 110)

    for fold in range(5):
        test_dir = (
            RUN_ROOT /
            f"fold{fold}" /
            "test"
        )

        sidecars = sorted(
            test_dir.glob(
                "worm_*_sidecar.npz"
            )
        )

        if not sidecars:
            raise RuntimeError(
                f"No sidecars: {test_dir}"
            )

        worm_results = []

        for side in sidecars:
            output = Path(
                str(side).replace(
                    "_sidecar.npz",
                    "_output.mat",
                )
            )

            if not output.exists():
                raise FileNotFoundError(
                    output
                )

            r = evaluate_worm(
                side,
                output,
            )

            worm_results.append(r)

        fm = fold_metrics(
            worm_results
        )

        fm["fold"] = fold
        fm["n_worms"] = len(worm_results)

        all_folds.append(fm)

        with open(
            test_dir.parent /
            "fold_metrics.json",
            "w",
        ) as f:
            json.dump(
                {
                    "fold_metrics": fm,
                    "worms": worm_results,
                },
                f,
                indent=2,
            )

        print(
            f"fold{fold}: "
            f"worms={fm['n_worms']:2d} "
            f"Q={fm['Q']:4d}  "
            f"Top1={fmt_pct(fm['top1']):>7}  "
            f"Top5={fmt_pct(fm['top5']):>7}  "
            f"MRR={fm['mrr']:.4f}  "
            f"Hung={fmt_pct(fm['hungarian']):>7}  "
            f"Cov={fmt_pct(fm['coverage']):>7}"
        )

    keys = [
        "top1",
        "top5",
        "mrr",
        "hungarian",
        "coverage",
        "covered_top1",
        "covered_top5",
        "official_duplicate_accuracy",
    ]

    summary = {}

    for key in keys:
        vals = np.asarray([
            f[key]
            for f in all_folds
        ], dtype=float)

        summary[key] = {
            "mean": float(
                vals.mean()
            ),
            "sample_sd": float(
                vals.std(ddof=1)
            ),
            "fold_values": vals.tolist(),
        }

    summary["n_folds"] = 5
    summary["aggregation"] = (
        "query-weighted within biological fold; "
        "mean and sample SD across 5 biological folds"
    )
    summary["protocol"] = (
        "Official MultiCellCalciumImaging CRF inference; "
        "uniform unary; angle-only pairwise relation; "
        "UGM LBP; outer-fold-train-only O(3) "
        "coordinate-frame calibration; no test labels "
        "used for inference/frame calibration."
    )

    with open(
        RUN_ROOT / "summary.json",
        "w",
    ) as f:
        json.dump(
            {
                "summary": summary,
                "folds": all_folds,
            },
            f,
            indent=2,
        )

    print()
    print("=" * 110)
    print(
        "FINAL — MEAN ± SAMPLE SD "
        "ACROSS BIOLOGICAL FOLDS"
    )
    print("=" * 110)

    for key in [
        "top1",
        "top5",
        "mrr",
        "hungarian",
        "coverage",
        "covered_top1",
        "covered_top5",
        "official_duplicate_accuracy",
    ]:
        m = summary[key]["mean"]
        sd = summary[key]["sample_sd"]

        if key == "mrr":
            print(
                f"{key:30s}: "
                f"{m:.4f} ± {sd:.4f}"
            )
        else:
            print(
                f"{key:30s}: "
                f"{100*m:.2f} ± {100*sd:.2f}%"
            )

    print("=" * 110)
    print(
        "saved:",
        RUN_ROOT / "summary.json"
    )


if __name__ == "__main__":
    main()
