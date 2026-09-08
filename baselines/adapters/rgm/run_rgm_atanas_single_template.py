#!/usr/bin/env python3
from collections import Counter
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from utils.config import cfg, cfg_from_file
from utils.loss_func import PermLoss
from utils.hungarian import hungarian


INVALID_IDS = {"", "nan", "None", "NONE", "-1"}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_split(root, split):
    worms = []

    for p in sorted((Path(root) / split).glob("*.npz")):
        d = np.load(p, allow_pickle=True)

        xyz = np.asarray(d["xyz"], dtype=np.float32)
        cid = np.asarray(d["cell_id"]).astype(str)

        # Unified benchmark supervision-mask priority:
        # clean_mask > labeled_mask > valid non-empty identity.
        if "clean_mask" in d:
            mask = np.asarray(d["clean_mask"], dtype=bool)
        elif "labeled_mask" in d:
            mask = np.asarray(d["labeled_mask"], dtype=bool)
        else:
            mask = np.array(
                [x not in INVALID_IDS for x in cid],
                dtype=bool,
            )

        good = np.isfinite(xyz).all(axis=1)

        worms.append({
            "name": p.name,
            "path": str(p),
            "xyz_raw": xyz[good],
            "cell_id": cid[good],
            "mask": mask[good],
        })

    if not worms:
        raise RuntimeError(f"No .npz files in {root}/{split}")

    return worms


def fit_train_norm(train):
    xyz = np.concatenate(
        [w["xyz_raw"] for w in train],
        axis=0,
    )

    mu = xyz.mean(0).astype(np.float32)
    sd = xyz.std(0).astype(np.float32)
    sd = np.maximum(sd, 1e-6)

    return mu, sd


def normalize(worms, mu, sd):
    out = []

    for w in worms:
        q = dict(w)
        q["xyz"] = (
            (w["xyz_raw"] - mu[None]) /
            sd[None]
        ).astype(np.float32)
        out.append(q)

    return out


def chamfer(a, b):
    ta = cKDTree(a)
    tb = cKDTree(b)

    ab = tb.query(a, k=1)[0].mean()
    ba = ta.query(b, k=1)[0].mean()

    return 0.5 * (ab + ba)


def choose_medoid(train):
    n = len(train)
    D = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = chamfer(
                train[i]["xyz"],
                train[j]["xyz"],
            )
            D[i, j] = d
            D[j, i] = d

    mean_d = D.sum(1) / max(n - 1, 1)
    idx = int(np.argmin(mean_d))

    print("\n===== TRAIN-ONLY MEDOID =====")
    for i, w in enumerate(train):
        flag = "  <-- TEMPLATE" if i == idx else ""
        print(
            f"{w['name']:28s} "
            f"{mean_d[i]:.6f}{flag}"
        )

    print("=============================\n")

    return idx, D, mean_d


def template_id_map(template):
    """
    Strict one-to-one template identity map.

    An identity can define a correspondence only when it occurs
    exactly once within the fixed template.
    """
    valid_ids = []

    for cid, m in zip(
        template["cell_id"],
        template["mask"],
    ):
        cid = str(cid)

        if m and cid not in INVALID_IDS:
            valid_ids.append(cid)

    counts = Counter(valid_ids)

    result = {}

    for j, (cid, m) in enumerate(
        zip(template["cell_id"],
            template["mask"])
    ):
        cid = str(cid)

        if (
            m
            and cid not in INVALID_IDS
            and counts[cid] == 1
        ):
            result[cid] = j

    return result, counts


def make_pair(src, template, device):
    """
    Construct strict partial-permutation GT.

    A positive correspondence is created only when the identity:
      1. occurs exactly once in the source;
      2. occurs exactly once in the template;
      3. exists in both.

    Duplicate/ambiguous source identities remain as RGM input
    points, but receive no positive GT correspondence.
    """

    xs = src["xyz"]
    xt = template["xyz"]

    ns = len(xs)
    nt = len(xt)

    if ns < cfg.PGM.NEIGHBORSNUM:
        raise RuntimeError(
            f"{src['name']}: N={ns} < "
            f"k={cfg.PGM.NEIGHBORSNUM}"
        )

    if nt < cfg.PGM.NEIGHBORSNUM:
        raise RuntimeError(
            f"template N={nt} < "
            f"k={cfg.PGM.NEIGHBORSNUM}"
        )

    P1 = (
        torch.from_numpy(xs)
        [None]
        .float()
        .to(device)
    )

    P2 = (
        torch.from_numpy(xt)
        [None]
        .float()
        .to(device)
    )

    A1 = torch.ones(
        (1, ns, ns),
        dtype=torch.float32,
        device=device,
    )

    A2 = torch.ones(
        (1, nt, nt),
        dtype=torch.float32,
        device=device,
    )

    A1 -= torch.eye(
        ns,
        device=device,
    )[None]

    A2 -= torch.eye(
        nt,
        device=device,
    )[None]

    # --------------------------------------------------------
    # Source identity counts
    # --------------------------------------------------------
    source_valid_ids = []

    for cid, m in zip(
        src["cell_id"],
        src["mask"],
    ):
        cid = str(cid)

        if m and cid not in INVALID_IDS:
            source_valid_ids.append(cid)

    source_counts = Counter(
        source_valid_ids
    )

    # --------------------------------------------------------
    # Template identity counts
    # --------------------------------------------------------
    tmap, template_counts = (
        template_id_map(template)
    )

    gt_np = np.zeros(
        (ns, nt),
        dtype=np.float32,
    )

    queries = []

    # Unique labeled identity count, matching the previous
    # set-based preflight accounting.
    labeled_source_unique = len(
        source_counts
    )

    # An identity is evaluable only if it is unique within
    # the source specimen.
    evaluable_source = sum(
        count == 1
        for count in source_counts.values()
    )

    ambiguous_source_ids = {
        cid: int(count)
        for cid, count
        in source_counts.items()
        if count > 1
    }

    # --------------------------------------------------------
    # Strict one-to-one GT
    # --------------------------------------------------------
    for i, (cid, m) in enumerate(
        zip(
            src["cell_id"],
            src["mask"],
        )
    ):
        cid = str(cid)

        if (
            not m
            or cid in INVALID_IDS
        ):
            continue

        # Ambiguous in source -> no unique GT.
        if source_counts[cid] != 1:
            continue

        # Ambiguous or absent in template -> no unique GT.
        if template_counts.get(cid, 0) != 1:
            continue

        if cid not in tmap:
            continue

        j = tmap[cid]

        gt_np[i, j] = 1.0
        queries.append((i, j))

    # --------------------------------------------------------
    # Strict partial-permutation assertions
    # --------------------------------------------------------
    row_sum = gt_np.sum(axis=1)
    col_sum = gt_np.sum(axis=0)

    if np.any(row_sum > 1.0 + 1e-6):
        raise RuntimeError(
            f"{src['name']}: "
            "GT contains a row with >1 match"
        )

    if np.any(col_sum > 1.0 + 1e-6):
        raise RuntimeError(
            f"{src['name']}: "
            "GT contains a column with >1 match"
        )

    if int(gt_np.sum()) != len(queries):
        raise RuntimeError(
            f"{src['name']}: "
            f"GT/query mismatch: "
            f"{int(gt_np.sum())} "
            f"vs {len(queries)}"
        )

    gt = (
        torch.from_numpy(gt_np)
        [None]
        .to(device)
    )

    n1 = torch.tensor(
        [ns],
        dtype=torch.long,
    )

    n2 = torch.tensor(
        [nt],
        dtype=torch.long,
    )

    return {
        "P1": P1,
        "P2": P2,
        "A1": A1,
        "A2": A2,
        "gt": gt,
        "n1": n1,
        "n2": n2,
        "queries": queries,

        "labeled_source":
            labeled_source_unique,

        "evaluable_source":
            evaluable_source,

        "ambiguous_source_ids":
            ambiguous_source_ids,
    }


def evaluate(model, worms, template, device):
    model.eval()

    Q = 0
    labeled_total = 0

    c1 = 0
    c5 = 0
    rr = 0.0
    hc = 0

    pair_metrics = []

    for src in worms:
        x = make_pair(
            src,
            template,
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

        score = s[0].detach().cpu().numpy()

        # Official RGM partial Hungarian
        H = hungarian(
            s,
            x["n1"],
            x["n2"],
            in_src,
            in_ref,
        )[0].detach().cpu().numpy()

        pq = 0
        pc1 = 0
        pc5 = 0
        prr = 0.0
        ph = 0

        for i, jgt in x["queries"]:
            order = np.argsort(-score[i])

            rank = int(
                np.where(order == jgt)[0][0]
            ) + 1

            pq += 1

            if rank == 1:
                pc1 += 1

            if rank <= 5:
                pc5 += 1

            prr += 1.0 / rank

            assigned = np.flatnonzero(
                H[i] > 0.5
            )

            if (
                len(assigned) == 1
                and int(assigned[0]) == jgt
            ):
                ph += 1

        Q += pq
        labeled_total += x["evaluable_source"]

        c1 += pc1
        c5 += pc5
        rr += prr
        hc += ph

        pair_metrics.append({
            "worm": src["name"],
            "queries": pq,
            "labeled_source_unique": x["labeled_source"],
            "evaluable_source": x["evaluable_source"],
            "ambiguous_source_ids": x["ambiguous_source_ids"],
            "coverage": (
                pq / x["evaluable_source"]
                if x["evaluable_source"] else None
            ),
            "top1": pc1 / pq if pq else None,
            "top5": pc5 / pq if pq else None,
            "mrr": prr / pq if pq else None,
            "hungarian": ph / pq if pq else None,
        })

    if Q == 0:
        raise RuntimeError(
            "No valid GT queries against template."
        )

    return {
        "queries": int(Q),
        "labeled_source": int(labeled_total),
        "coverage": float(Q / labeled_total),
        "top1": float(c1 / Q),
        "top5": float(c5 / Q),
        "mrr": float(rr / Q),
        "hungarian": float(hc / Q),
        "pairs": pair_metrics,
    }


def print_metric(tag, m):
    print(
        f"{tag:<6s} "
        f"Q={m['queries']:4d} "
        f"Cov={100*m['coverage']:6.2f}% "
        f"Top1={100*m['top1']:6.2f}% "
        f"Top5={100*m['top5']:6.2f}% "
        f"MRR={m['mrr']:.4f} "
        f"Hung={100*m['hungarian']:6.2f}%"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", required=True)
    ap.add_argument("--run-root", required=True)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu", type=int, default=0)

    ap.add_argument("--epochs", type=int, default=200)

    args = ap.parse_args()

    seed_all(args.seed)

    repo = Path(__file__).resolve().parent

    official_cfg = (
        repo /
        "experiments" /
        "train_RGM_Seen_Crop_modelnet40_transformer.yaml"
    )

    if not official_cfg.exists():
        raise RuntimeError(
            f"Official RGM config missing: {official_cfg}"
        )

    # Exact official partial/crop RGM architecture/training config
    cfg_from_file(str(official_cfg))

    # No normals are supplied by Atanas and Net uses xyz/gxyz.
    cfg.PGM.NORMALS = False

    device = torch.device(
        f"cuda:{args.gpu}"
        if torch.cuda.is_available()
        else "cpu"
    )

    data_root = Path(args.data_root).resolve()
    run_root = Path(args.run_root).resolve()

    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("OFFICIAL RGM — ATANAS SINGLE TRAIN TEMPLATE")
    print("=" * 78)
    print("data root :", data_root)
    print("run root  :", run_root)
    print("seed      :", args.seed)
    print("device    :", device)
    print("epochs    :", args.epochs)
    print("config    :", official_cfg)
    print("=" * 78)

    # ==========================================================
    # IMPORTANT: TEST IS NOT READ HERE
    # ==========================================================

    train_raw = load_split(
        data_root,
        "train",
    )

    val_raw = load_split(
        data_root,
        "val",
    )

    print(
        f"PRE-TEST: train={len(train_raw)} "
        f"val={len(val_raw)}"
    )

    mu, sd = fit_train_norm(train_raw)

    np.savez(
        run_root / "train_normalization.npz",
        mu=mu,
        sd=sd,
    )

    train = normalize(
        train_raw,
        mu,
        sd,
    )

    val = normalize(
        val_raw,
        mu,
        sd,
    )

    medoid_idx, D, mean_d = choose_medoid(
        train
    )

    template = train[medoid_idx]

    np.save(
        run_root / "train_chamfer.npy",
        D,
    )

    with open(
        run_root / "template.json",
        "w",
    ) as f:
        json.dump({
            "template": template["name"],
            "selection": (
                "train-only symmetric Chamfer medoid"
            ),
            "mean_chamfer": float(
                mean_d[medoid_idx]
            ),
            "seed_independent": True,
        }, f, indent=2)

    print(
        "LOCKED TEMPLATE:",
        template["name"],
    )

    # Import after official config is loaded
    from models.Net import Net

    model = Net().to(device)

    nparams = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "RGM parameters:",
        f"{nparams:,}",
    )

    criterion = PermLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.TRAIN.LR,
        momentum=cfg.TRAIN.MOMENTUM,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg.TRAIN.LR_STEP),
        gamma=cfg.TRAIN.LR_DECAY,
    )

    sources = [
        w
        for i, w in enumerate(train)
        if i != medoid_idx
    ]

    print(
        "training source worms:",
        len(sources),
    )

    best_hung = -1.0
    best_top1 = -1.0
    best_epoch = -1

    history = []

    # ==========================================================
    # TRAIN + VAL ONLY
    # ==========================================================

    for epoch in range(1, args.epochs + 1):

        model.train()

        order = np.random.RandomState(
            args.seed + epoch
        ).permutation(
            len(sources)
        )

        losses = []

        for ii in order:
            src = sources[int(ii)]

            x = make_pair(
                src,
                template,
                device,
            )

            if not x["queries"]:
                continue

            optimizer.zero_grad(
                set_to_none=True
            )

            s, in_src, in_ref = model(
                x["P1"],
                x["P2"],
                x["A1"],
                x["A2"],
                x["n1"],
                x["n2"],
            )

            # Exact official crop RGM loss behavior.
            if cfg.PGM.USEINLIERRATE:
                s_loss = (
                    in_src *
                    s *
                    in_ref.transpose(2, 1).contiguous()
                )
            else:
                s_loss = s

            loss = criterion(
                s_loss,
                x["gt"],
                x["n1"],
                x["n2"],
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"non-finite loss: "
                    f"epoch={epoch}, "
                    f"worm={src['name']}"
                )

            loss.backward()
            optimizer.step()

            losses.append(
                float(loss.item())
            )

        val_m = evaluate(
            model,
            val,
            template,
            device,
        )

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"[{epoch:03d}/{args.epochs}] "
            f"loss={np.mean(losses):.6f} "
            f"lr={lr:.2e} "
            f"val Top1={100*val_m['top1']:.2f}% "
            f"Top5={100*val_m['top5']:.2f}% "
            f"Hung={100*val_m['hungarian']:.2f}%"
        )

        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "lr": float(lr),
            "val_top1": val_m["top1"],
            "val_top5": val_m["top5"],
            "val_mrr": val_m["mrr"],
            "val_hungarian": val_m["hungarian"],
        })

        # Official RGM selects using matching accuracy.
        # Here that corresponds to valid-query Hungarian recall.
        better = (
            val_m["hungarian"] > best_hung
            or (
                math.isclose(
                    val_m["hungarian"],
                    best_hung,
                    abs_tol=1e-12,
                )
                and val_m["top1"] > best_top1
            )
        )

        if better:
            best_hung = val_m["hungarian"]
            best_top1 = val_m["top1"]
            best_epoch = epoch

            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val": val_m,
                "template": template["name"],
            }, run_root / "best.pt")

        scheduler.step()

    with open(
        run_root / "history.json",
        "w",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
        )

    print("\n" + "=" * 78)
    print("TRAINING FINISHED — TEST STILL UNREAD")
    print("=" * 78)
    print("best epoch :", best_epoch)
    print("best val Top1:",
          f"{100*best_top1:.2f}%")
    print("best val Hung :",
          f"{100*best_hung:.2f}%")

    # ==========================================================
    # LOCK EVERYTHING BEFORE FIRST TEST ACCESS
    # ==========================================================

    lock = {
        "method": "RGM",
        "implementation": "official fukexue/RGM",
        "protocol": "Atanas fixed train-only medoid template",
        "seed": args.seed,
        "template": template["name"],
        "template_selection": (
            "train-only symmetric Chamfer medoid"
        ),
        "normalization": (
            "global train-only per-axis z-score"
        ),
        "official_config": str(official_cfg),
        "best_epoch": best_epoch,
        "selection_metric": (
            "validation Hungarian accuracy; "
            "Top-1 tie-break"
        ),
        "test_accessed_during_training": False,
    }

    with open(
        run_root / "LOCKED_BEFORE_TEST.json",
        "w",
    ) as f:
        json.dump(
            lock,
            f,
            indent=2,
        )

    print(
        "LOCK:",
        run_root / "LOCKED_BEFORE_TEST.json"
    )

    # ==========================================================
    # FIRST TEST ACCESS
    # ==========================================================

    ckpt = torch.load(
        run_root / "best.pt",
        map_location=device,
    )

    model.load_state_dict(
        ckpt["model_state"]
    )

    model.eval()

    print("\n>>> TEST UNLOCKED NOW <<<")

    test_raw = load_split(
        data_root,
        "test",
    )

    test = normalize(
        test_raw,
        mu,
        sd,
    )

    val_final = evaluate(
        model,
        val,
        template,
        device,
    )

    test_final = evaluate(
        model,
        test,
        template,
        device,
    )

    print("\n" + "=" * 78)
    print("ATANAS OFFICIAL RGM LOCKED RESULT")
    print("=" * 78)
    print("template:", template["name"])

    print_metric(
        "VAL",
        val_final,
    )

    print_metric(
        "TEST",
        test_final,
    )

    print("=" * 78)

    result = {
        **lock,
        "test_unlocked": True,
        "val": val_final,
        "test": test_final,
    }

    with open(
        run_root / "test_metrics.json",
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    print(
        "metrics:",
        run_root / "test_metrics.json"
    )


if __name__ == "__main__":
    main()