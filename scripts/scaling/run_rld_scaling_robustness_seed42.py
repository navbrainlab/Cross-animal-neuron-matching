#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path("/home/ubuntu/klb/nuclr/nuclr")
PACKAGE = REPO / "mprt_net_v1_1"
DATA = REPO / "Data/Dunn_001623/date_disjoint_full95_v1"
SOURCE_RUN = REPO / "runs/mprt_v1_1/rld/seed42/full"
FULL_STATIC = REPO / "runs/mprt_v1_1_anchored_atlas/rld/seed42/anchored_pure.pt"
FULL_DYNAMIC = REPO / "runs/mprt_v1_1_dynamic_residual_atlas/rld/seed42/low_rank_r8/best.pt"
OUT = REPO / "runs/mprt_v1_1_rld_scaling_robustness_seed42_v1"

MODEL_SEED = 42
SUBSET_SEED = 20260825
DEFAULT_SIZES = (5, 10, 20, 40, 67)
PERTURB_SEEDS = (0, 1, 2)

# Current dynamic-residual-atlas defaults used by the existing CV runner.
DYN = dict(
    epochs=30,
    pairs_per_epoch=128,
    activity_length=512,
    synthetic_drop_probability=0.05,
    learning_rate=5e-4,
    pair_loss_weight=0.5,
    magnitude_weight=0.02,
    smoothness_weight=0.05,
    distortion_weight=0.05,
    rank=8,
    coordinate_scale=0.15,
    node_scale=0.10,
    relation_scale=0.10,
    early_stopping_patience=8,
)

ACTIVITY_LENGTH = 512
MIN_SHARED = 2
BLEND_WEIGHT = 0.30
GATE_TEMPERATURE = 0.05


def stable_seed(*parts: Any) -> int:
    s = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(s).digest()[:8], "little") % (2**32)


def norm_label(v: Any) -> str:
    if isinstance(v, bytes):
        v = v.decode("utf-8", errors="replace")
    return str(v).strip()


def unique_supervised_indices(cell_ids: Iterable[Any], mask: np.ndarray) -> list[int]:
    labels = [norm_label(x) for x in cell_ids]
    counts = Counter(labels[i] for i, ok in enumerate(mask) if ok and labels[i])
    return [
        i for i, ok in enumerate(mask)
        if ok and labels[i] and counts[labels[i]] == 1
    ]


def run(cmd: list[str], *, cwd: Path, gpu: str, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("\n[RUN]", " ".join(map(str, cmd)), flush=True)
    with log.open("w", encoding="utf-8") as f:
        import subprocess
        p = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
    if rc:
        raise SystemExit(rc)


def symlink_split(src_files: list[Path], dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    existing = list(dst.glob("*.npz"))
    if existing:
        got = {p.name for p in existing}
        want = {p.name for p in src_files}
        if got != want:
            raise RuntimeError(f"Existing split mismatch: {dst}")
        return
    for src in src_files:
        os.symlink(src.resolve(), dst / src.name)


def nested_stratified_order(files: list[Path]) -> list[Path]:
    """Date-balanced deterministic order. One fixed subset realization, not a model seed."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        # RLD stems are date-time-like; grouping by first 8 chars balances recording dates.
        key = p.stem[:8]
        groups[key].append(p)

    rng = np.random.default_rng(SUBSET_SEED)
    keys = sorted(groups)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(groups[k])

    order: list[Path] = []
    depth = 0
    while True:
        added = False
        cycle = keys.copy()
        rng.shuffle(cycle)
        for k in cycle:
            if depth < len(groups[k]):
                order.append(groups[k][depth])
                added = True
        if not added:
            break
        depth += 1
    assert len(order) == len(files)
    return order


def prepare_subset(n: int) -> Path:
    train = sorted((DATA / "train").glob("*.npz"))
    val = sorted((DATA / "val").glob("*.npz"))
    test = sorted((DATA / "test").glob("*.npz"))
    if len(train) != 67:
        raise RuntimeError(f"Expected 67 RLD train worms, found {len(train)}")
    if n > len(train):
        raise ValueError(n)

    selected = nested_stratified_order(train)[:n]
    root = OUT / "data_scaling" / f"n{n}"
    symlink_split(selected, root / "train")
    symlink_split(val, root / "val")
    symlink_split(test, root / "test")
    (root / "subset.json").write_text(
        json.dumps(
            {
                "n_train": n,
                "model_seed": MODEL_SEED,
                "subset_seed": SUBSET_SEED,
                "selection": "nested_date_balanced_round_robin",
                "train_files": [p.name for p in selected],
                "val_files": [p.name for p in val],
                "test_files": [p.name for p in test],
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    return root


def train_pairwise(n: int, root: Path, gpu: str) -> Path:
    if n == 67:
        ckpt = SOURCE_RUN / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        print(f"[REUSE n67 pairwise] {ckpt}")
        return ckpt

    sys.path.insert(0, str(PACKAGE))
    from mprt_net.experiments import orchestration as util

    parser = util.train_parser(PACKAGE)
    source = util.complete_arguments(parser, util.source_arguments(SOURCE_RUN))
    run_dir = OUT / "scaling" / f"n{n}" / "pairwise" / "full"
    ckpt = run_dir / "best.pt"
    if ckpt.is_file():
        print(f"[REUSE pairwise n={n}] {ckpt}")
        return ckpt
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Partial pairwise output exists: {run_dir}")

    values = dict(source)
    values.update(
        dataset_root=str(root),
        output_dir=str(run_dir),
        seed=MODEL_SEED,
        variant="full",
        cycle_weight=0.0,
        atlas_weight=0.0,
        atlas_blend_weight=0.0,
        device="cuda",
        allow_existing_output=False,
    )
    cmd = util.command_from_values(parser, values)
    util.run_command(cmd, cwd=PACKAGE, gpu=gpu, log_path=run_dir / "train.log")
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    return ckpt


def build_static(n: int, root: Path, pairwise: Path, gpu: str) -> Path:
    if n == 67:
        if not FULL_STATIC.is_file():
            raise FileNotFoundError(FULL_STATIC)
        print(f"[REUSE n67 static atlas] {FULL_STATIC}")
        return FULL_STATIC

    atlas_dir = OUT / "scaling" / f"n{n}" / "static_atlas"
    pure = atlas_dir / "anchored_pure.pt"
    gated = atlas_dir / "anchored_gated_b030.pt"
    if pure.is_file():
        print(f"[REUSE static n={n}] {pure}")
        return pure
    if atlas_dir.exists() and any(atlas_dir.iterdir()):
        raise RuntimeError(f"Partial static-atlas output exists: {atlas_dir}")

    cmd = [
        sys.executable, "-u", "-m", "mprt_net.build_anchored_atlas",
        "--dataset-root", str(root),
        "--split", "train",
        "--checkpoint", str(pairwise),
        "--output", str(gated),
        "--pure-output", str(pure),
        "--activity-length", str(ACTIVITY_LENGTH),
        "--blend-weight", str(BLEND_WEIGHT),
        "--gate-temperature", str(GATE_TEMPERATURE),
        "--device", "cuda",
    ]
    run(cmd, cwd=PACKAGE, gpu=gpu, log=atlas_dir / "build.log")
    if not pure.is_file():
        raise FileNotFoundError(pure)
    return pure


def train_dynamic(n: int, root: Path, static: Path, gpu: str) -> Path:
    # Full-data endpoint is the already validated final model.
    if n == 67:
        if not FULL_DYNAMIC.is_file():
            raise FileNotFoundError(FULL_DYNAMIC)
        print(f"[REUSE n67 dynamic atlas] {FULL_DYNAMIC}")
        return FULL_DYNAMIC

    run_dir = OUT / "scaling" / f"n{n}" / "dynamic" / f"low_rank_r{DYN['rank']}"
    ckpt = run_dir / "best.pt"

    if ckpt.is_file():
        print(f"[REUSE dynamic n={n}] {ckpt}")
        return ckpt

    # Refuse to silently continue a partially optimized model.
    progress = [
        run_dir / "args.json",
        run_dir / "history.jsonl",
        run_dir / "last.pt",
        run_dir / "best.pt",
    ]
    if any(x.exists() for x in progress):
        raise RuntimeError(f"Partial dynamic optimization output exists: {run_dir}")

    # A CLI/preflight-only failed attempt may leave only train.log.
    stale_log = run_dir / "train.log"
    if stale_log.is_file():
        archive = run_dir / "train_cli_failed_previous.log"
        stale_log.replace(archive)
        print(f"[ARCHIVE preflight-only log] {archive}")

    values = {
        "dataset_root": str(root),
        "static_atlas_checkpoint": str(static),
        "output_dir": str(run_dir),
        "seed": MODEL_SEED,
        **DYN,
        "device": "cuda",
    }

    cmd = [sys.executable, "-u", "-m", "mprt_net.train_dynamic_atlas"]
    for key, value in values.items():
        cmd.extend(["--" + key.replace("_", "-"), str(value)])

    run(
        cmd,
        cwd=PACKAGE,
        gpu=gpu,
        log=run_dir / "train.log",
    )

    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    return ckpt


def load_model(checkpoint: Path, device: torch.device):
    if str(PACKAGE) not in sys.path:
        sys.path.insert(0, str(PACKAGE))
    from mprt_net.evaluate import load_checkpoint
    model, payload = load_checkpoint(checkpoint, device)
    model.eval()
    mapping = {norm_label(k): int(v) for k, v in payload["atlas_identity_to_slot"].items()}
    return model, payload, mapping


def canonical_queries(device: torch.device) -> dict[str, set[str]]:
    """Fixed query universe from the full 67-worm atlas."""
    if str(PACKAGE) not in sys.path:
        sys.path.insert(0, str(PACKAGE))
    from mprt_net.data import WormCache, split_files

    _, _, full_mapping = load_model(FULL_STATIC, device)
    files = split_files(DATA, "test")
    cache = WormCache(activity_length=ACTIVITY_LENGTH, max_items=max(32, len(files)))
    out: dict[str, set[str]] = {}
    for p in files:
        s = cache.get(p)
        labels = [norm_label(x) for x in s.cell_ids]
        mask = s.supervised_mask.detach().cpu().numpy().astype(bool)
        indices = unique_supervised_indices(labels, mask)
        out[s.uid] = {labels[i] for i in indices if labels[i] in full_mapping}
    return out


def score_checkpoint(
    checkpoint: Path,
    dataset_root: Path,
    *,
    dynamic: bool,
    device: torch.device,
    canonical: dict[str, set[str]],
    transform=None,
) -> dict[str, Any]:
    if str(PACKAGE) not in sys.path:
        sys.path.insert(0, str(PACKAGE))
    from mprt_net.data import WormCache, split_files

    model, _, mapping = load_model(checkpoint, device)
    atlas = model.atlas_encoding()
    files = split_files(dataset_root, "test")
    cache = WormCache(activity_length=ACTIVITY_LENGTH, max_items=max(32, len(files)))

    total_canonical = sum(len(x) for x in canonical.values())
    observed = top1 = top5 = hung = 0
    rr = 0.0
    missing_candidate = 0
    missing_observation = 0

    with torch.no_grad():
        for p in files:
            s_cpu = cache.get(p)
            if transform is not None:
                s_cpu = transform(s_cpu)
            s = s_cpu.to(device)
            enc = model.encode_population(s)
            if dynamic:
                dyn = model.deform_atlas(enc, atlas)
                output = model.match_encodings(enc, dyn.encoding)
            else:
                output = model.match_encodings(enc, atlas)

            ranking = output.row_conditional[:, :-1].detach().float().cpu().numpy()
            assignment = output.plan[:-1, :-1].detach().float().cpu().numpy()
            labels = [norm_label(x) for x in s_cpu.cell_ids]
            mask = s_cpu.supervised_mask.detach().cpu().numpy().astype(bool)

            # Current-label -> unique row.
            counts = Counter(labels[i] for i, ok in enumerate(mask) if ok)
            label_to_row = {
                labels[i]: i
                for i, ok in enumerate(mask)
                if ok and counts[labels[i]] == 1
            }

            if assignment.size:
                rows, cols = linear_sum_assignment(-assignment)
                row_to_col = dict(zip(rows.tolist(), cols.tolist()))
            else:
                row_to_col = {}

            for gt in canonical.get(s_cpu.uid, set()):
                if gt not in label_to_row:
                    missing_observation += 1
                    continue
                observed += 1
                r = label_to_row[gt]
                if gt not in mapping:
                    missing_candidate += 1
                    continue
                c = mapping[gt]
                scores = ranking[r]
                rank = 1 + int(np.sum(scores > scores[c]))
                top1 += int(rank == 1)
                top5 += int(rank <= min(5, len(scores)))
                rr += 1.0 / rank
                hung += int(row_to_col.get(r, -1) == c)

    denom = max(observed, 1)
    coverage = observed / max(total_canonical, 1)
    return {
        "canonical_queries": total_canonical,
        "observed_queries": observed,
        "coverage": coverage,
        "top1_observed": top1 / denom,
        "top5_observed": top5 / denom,
        "mrr_observed": rr / denom,
        "hungarian_observed": hung / denom,
        "top1_effective": top1 / max(total_canonical, 1),
        "hungarian_effective": hung / max(total_canonical, 1),
        "missing_observation": missing_observation,
        "missing_candidate": missing_candidate,
        "checkpoint": str(checkpoint),
    }


def make_transform(kind: str, severity: float, perturb_seed: int):
    if str(PACKAGE) not in sys.path:
        sys.path.insert(0, str(PACKAGE))
    from mprt_net.data import WormSample

    def transform(sample):
        rng = np.random.default_rng(
            stable_seed(kind, f"{severity:.8f}", perturb_seed, sample.uid)
        )
        xyz = sample.xyz.detach().cpu().clone()
        activity = sample.activity.detach().cpu().clone()
        mask = sample.supervised_mask.detach().cpu().clone()
        ids = list(sample.cell_ids)
        n = xyz.shape[0]

        if severity <= 0:
            return sample

        if kind == "coord_noise":
            if n > 1:
                d = torch.pdist(xyz.float())
                scale = float(d.median()) if d.numel() else 1.0
            else:
                scale = 1.0
            noise = torch.from_numpy(
                rng.normal(0.0, severity * max(scale, 1e-6), size=xyz.shape).astype(np.float32)
            )
            xyz = xyz + noise

        elif kind == "missing":
            keep_n = max(2, int(round(n * (1.0 - severity))))
            keep = np.sort(rng.choice(n, size=keep_n, replace=False))
            idx = torch.as_tensor(keep, dtype=torch.long)
            xyz = xyz.index_select(0, idx)
            activity = activity.index_select(0, idx)
            mask = mask.index_select(0, idx)
            ids = [ids[i] for i in keep.tolist()]

        elif kind == "outlier":
            add_n = max(1, int(round(n * severity)))
            if n > 1:
                d = torch.pdist(xyz.float())
                dmed = float(d.median()) if d.numel() else 1.0
            else:
                dmed = 1.0

            lo = xyz.min(0).values.numpy()
            hi = xyz.max(0).values.numpy()
            pad = 0.25 * max(dmed, 1e-6)
            new_xyz = rng.uniform(lo - pad, hi + pad, size=(add_n, 3)).astype(np.float32)

            base_idx = rng.integers(0, n, size=add_n)
            base_act = activity[torch.as_tensor(base_idx, dtype=torch.long)].numpy()
            act_std = activity.float().std(dim=0, unbiased=False).numpy()
            act_noise = rng.normal(
                0.0, 0.25 * np.maximum(act_std, 1e-6),
                size=base_act.shape,
            ).astype(np.float32)
            new_act = base_act.astype(np.float32) + act_noise

            xyz = torch.cat([xyz, torch.from_numpy(new_xyz)], 0)
            activity = torch.cat([activity, torch.from_numpy(new_act)], 0)
            mask = torch.cat([mask, torch.zeros(add_n, dtype=torch.bool)], 0)
            ids += [f"__OUTLIER_s{perturb_seed}_{i:04d}" for i in range(add_n)]

        else:
            raise ValueError(kind)

        return WormSample(
            uid=sample.uid,
            xyz=xyz,
            activity=activity,
            cell_ids=tuple(ids),
            supervised_mask=mask,
            source_path=sample.source_path,
        )

    return transform


def scaling(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canonical = canonical_queries(device)
    rows = []

    for n in args.sizes:
        root = prepare_subset(n)
        pairwise = train_pairwise(n, root, args.gpu)
        static = build_static(n, root, pairwise, args.gpu)
        dynamic = train_dynamic(n, root, static, args.gpu)
        metrics = score_checkpoint(
            dynamic, root, dynamic=True, device=device, canonical=canonical
        )
        row = {"n_train": n, "model_seed": MODEL_SEED, **metrics}
        rows.append(row)
        print(
            f"[SCALING] N={n:2d} "
            f"Top1={100*metrics['top1_observed']:.2f}% "
            f"EffTop1={100*metrics['top1_effective']:.2f}% "
            f"Hung={100*metrics['hungarian_observed']:.2f}% "
            f"Coverage={100*metrics['coverage']:.2f}%"
        )

    result = {
        "protocol": "RLD scaling; one model seed; nested date-balanced train subsets",
        "model_seed": MODEL_SEED,
        "subset_seed": SUBSET_SEED,
        "sizes": list(args.sizes),
        "rows": rows,
    }
    path = OUT / "scaling" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("\nSaved:", path)


def robustness(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    canonical = canonical_queries(device)
    checkpoint = FULL_DYNAMIC
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing existing RLD seed42 dynamic checkpoint: {checkpoint}"
        )
    print(f"[REUSE robustness checkpoint] {checkpoint}")

    grids = {
        "coord_noise": (0.0, 0.02, 0.05, 0.10, 0.20),
        "missing": (0.0, 0.10, 0.20, 0.30, 0.40, 0.50),
        "outlier": (0.0, 0.10, 0.20, 0.30, 0.40, 0.50),
    }
    all_rows = []

    for kind, levels in grids.items():
        for severity in levels:
            seed_rows = []
            for pseed in PERTURB_SEEDS:
                metrics = score_checkpoint(
                    checkpoint,
                    DATA,
                    dynamic=True,
                    device=device,
                    canonical=canonical,
                    transform=make_transform(kind, severity, pseed),
                )
                row = {
                    "kind": kind,
                    "severity": severity,
                    "model_seed": MODEL_SEED,
                    "perturbation_seed": pseed,
                    **metrics,
                }
                seed_rows.append(row)
                all_rows.append(row)

            def ms(key):
                vals = np.asarray([r[key] for r in seed_rows], dtype=float)
                return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

            t1, t1sd = ms("top1_observed")
            et1, et1sd = ms("top1_effective")
            hu, husd = ms("hungarian_observed")
            cov, covsd = ms("coverage")
            print(
                f"[ROBUST] {kind:11s} level={severity:>4.2f} "
                f"Top1={100*t1:6.2f}±{100*t1sd:4.2f}% "
                f"Eff={100*et1:6.2f}±{100*et1sd:4.2f}% "
                f"Hung={100*hu:6.2f}±{100*husd:4.2f}% "
                f"Cov={100*cov:6.2f}±{100*covsd:4.2f}%"
            )

    summary = {
        "protocol": "RLD controlled robustness; model seed42 x perturbation seeds 0,1,2",
        "model_seed": MODEL_SEED,
        "perturbation_seeds": list(PERTURB_SEEDS),
        "checkpoint": str(checkpoint),
        "grids": {k: list(v) for k, v in grids.items()},
        "rows": all_rows,
    }
    path = OUT / "robustness" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print("\nSaved:", path)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("scaling")
    a.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    a.add_argument("--gpu", default="0")
    a.set_defaults(func=scaling)

    b = sub.add_parser("robustness")
    b.add_argument("--gpu", default="0")
    b.set_defaults(func=robustness)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
