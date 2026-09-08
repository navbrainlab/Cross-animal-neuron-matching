#!/usr/bin/env python3
"""Fine-tune the AUTHORS' OFFICIAL fDNC NIT_Registration on one locked fold/seed.

Important:
- Official source under benchmark_official/third_party/fdnc_official is never edited.
- Initialization must be the authors' official model/model.bin.
- This TRAINING CLI accepts train/val protocol files only. It has NO test-list argument.
- Model selection uses validation direct Top-1 only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

ROOT = Path("/home/ubuntu/klb/nuclr/nuclr")
BENCH = ROOT / "benchmark_official"
OFFICIAL = BENCH / "third_party" / "fdnc_official"
OFFICIAL_MODEL_PY = OFFICIAL / "src" / "model.py"
OFFICIAL_PRETRAINED = OFFICIAL / "model" / "model.bin"
PROTOCOLS = BENCH / "protocols"

INVALID = {"", "-1", "none", "nan", "null", "unknown", "unk", "?", "unlabeled", "unlabelled"}


@dataclass
class Worm:
    path: Path
    worm_id: str
    xyz: np.ndarray               # normalized [N,3], float32
    labels: list[Optional[str]]   # clean canonical ID or None

    @property
    def n(self) -> int:
        return int(self.xyz.shape[0])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_official_module():
    if not OFFICIAL_MODEL_PY.is_file():
        raise FileNotFoundError(OFFICIAL_MODEL_PY)
    spec = importlib.util.spec_from_file_location("fdnc_official_model_finetune", OFFICIAL_MODEL_PY)
    if spec is None or spec.loader is None:
        raise ImportError(OFFICIAL_MODEL_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_official_model(module, checkpoint: Path, device: torch.device):
    payload = load_torch(checkpoint)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise KeyError(f"{checkpoint}: expected official checkpoint dict with state_dict")
    model = module.NIT_Registration(
        input_dim=3,
        n_hidden=128,
        n_layer=6,
        cuda=(device.type == "cuda"),
        p_rotate=False,
        feat_trans=False,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device)


def clean_text(value: Any) -> Optional[str]:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value).strip()
    return None if text.lower() in INVALID else text


def read_protocol(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [Path(x.strip()).resolve() for x in path.read_text().splitlines() if x.strip()]
    if not rows:
        raise RuntimeError(f"Empty protocol: {path}")
    for p in rows:
        if not p.is_file():
            raise FileNotFoundError(p)
    return rows


def preprocess_xyz(xyz: np.ndarray, scale: float) -> np.ndarray:
    # Locked choice for unified Atanas/RLD: input XYZ already interpreted as micrometers.
    x = np.asarray(xyz, dtype=np.float32).copy()
    x -= np.median(x, axis=0, keepdims=True)
    x /= float(scale)
    return x.astype(np.float32)


def load_worm(path: Path, mask_key: str, scale: float) -> Worm:
    with np.load(path, allow_pickle=True) as d:
        xyz = np.asarray(d["xyz"], dtype=np.float32)[:, :3]
        raw = np.asarray(d["cell_id"]).reshape(-1)
        mask = (
            np.asarray(d[mask_key], dtype=bool).reshape(-1)
            if mask_key in d.files
            else np.ones(len(raw), dtype=bool)
        )
        stored_id = None
        for key in ("worm_id", "worm_name", "name"):
            if key in d.files:
                stored_id = clean_text(np.asarray(d[key]).reshape(-1)[0])
                if stored_id:
                    break
    if len(xyz) != len(raw) or len(raw) != len(mask):
        raise ValueError(f"{path}: xyz/cell_id/mask mismatch")
    if not np.isfinite(xyz).all():
        raise ValueError(f"{path}: xyz has NaN/Inf")
    labels = [clean_text(v) if bool(k) else None for v, k in zip(raw, mask)]
    return Worm(path.resolve(), stored_id or path.stem, preprocess_xyz(xyz, scale), labels)


def load_split(protocol: Path, mask_key: str, scale: float) -> list[Worm]:
    records = [load_worm(p, mask_key, scale) for p in read_protocol(protocol)]
    ids = [r.worm_id for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"Duplicate worm IDs in {protocol}: {ids}")
    return records


def unique_label_map(labels: Sequence[Optional[str]]) -> dict[str, int]:
    pos: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        if lab is not None:
            pos.setdefault(lab, []).append(i)
    return {lab: rows[0] for lab, rows in pos.items() if len(rows) == 1}


def supervised_unique_indices(w: Worm) -> set[int]:
    return set(unique_label_map(w.labels).values())


def match_dict_for_reference(reference: Worm, moving: Worm) -> dict[Any, np.ndarray]:
    """Build exactly the keys expected by official NIT_Registration.forward()."""
    ref_map = unique_label_map(reference.labels)
    mov_map = unique_label_map(moving.labels)

    # Reference self-supervision, mirroring the official batch convention.
    ref_match = np.asarray([[idx, idx] for idx in sorted(ref_map.values())], dtype=np.int64)
    if ref_match.size == 0:
        ref_match = np.empty((0, 2), dtype=np.int64)

    ref_unique = set(ref_map.values())
    ref_unlabel = np.asarray([i for i in range(reference.n) if i not in ref_unique], dtype=np.int64)

    # Moving -> reference supervision.
    shared = sorted(set(ref_map) & set(mov_map))
    mov_match = np.asarray([[mov_map[lab], ref_map[lab]] for lab in shared], dtype=np.int64)
    if mov_match.size == 0:
        mov_match = np.empty((0, 2), dtype=np.int64)

    mov_unique = set(mov_map.values())
    matched_mov = {mov_map[lab] for lab in shared}
    # A clean uniquely labelled neuron whose identity is absent in the reference
    # is a known outlier for this pair.
    mov_outlier = np.asarray(sorted(mov_unique - matched_mov), dtype=np.int64)
    # Non-clean or duplicate-labelled ROIs remain unlabeled, never forced to outlier.
    mov_unlabel = np.asarray([i for i in range(moving.n) if i not in mov_unique], dtype=np.int64)

    return {
        0: ref_match,
        1: mov_match,
        "outlier_0": np.empty((0,), dtype=np.int64),
        "outlier_1": mov_outlier,
        "unlabel_0": ref_unlabel,
        "unlabel_1": mov_unlabel,
    }


def configure_trainable(model: nn.Module, last_n: int) -> list[nn.Parameter]:
    if last_n < 1 or last_n > len(model.model.layers):
        raise ValueError(f"unfreeze_last_n must be 1..{len(model.model.layers)}")
    for p in model.parameters():
        p.requires_grad_(False)

    for layer in model.model.layers[-last_n:]:
        for p in layer.parameters():
            p.requires_grad_(True)

    if getattr(model.model, "norm", None) is not None:
        for p in model.model.norm.parameters():
            p.requires_grad_(True)

    for p in model.fc_outlier.parameters():
        p.requires_grad_(True)

    return [p for p in model.parameters() if p.requires_grad]


def set_train_mode(model: nn.Module, last_n: int) -> None:
    # Keep frozen PointTransFeat / early Transformer blocks in eval mode.
    model.eval()
    for layer in model.model.layers[-last_n:]:
        layer.train()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm.train()
    model.fc_outlier.train()


def augment(xyz: np.ndarray, scale_jitter: float, coordinate_jitter: float) -> np.ndarray:
    scale = 1.0 + np.random.randn() * scale_jitter
    noise = np.random.randn(*xyz.shape).astype(np.float32) * coordinate_jitter
    return (xyz * scale + noise).astype(np.float32)


def autocast_context(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=(device.type == "cuda" and precision == "bf16"),
    )


def l2sp_penalty(model: nn.Module, anchors: dict[str, torch.Tensor]) -> torch.Tensor:
    named = dict(model.named_parameters())
    pieces = [(named[k] - v).float().square().mean() for k, v in anchors.items()]
    return torch.stack(pieces).mean() if pieces else torch.tensor(0.0, device=next(model.parameters()).device)


def train_direction(
    model: nn.Module,
    reference: Worm,
    moving: Worm,
    args: argparse.Namespace,
    anchors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    match_dict = match_dict_for_reference(reference, moving)
    shared = int(len(match_dict[1]))
    if shared < args.minimum_common:
        return next(model.parameters()).sum() * 0.0, {"shared": shared, "skip": 1.0}

    ref = augment(reference.xyz, args.scale_jitter, args.coordinate_jitter)
    mov = augment(moving.xyz, args.scale_jitter, args.coordinate_jitter)

    loss_dict, _ = model([ref, mov], match_dict=match_dict, ref_idx=0, mode="train")
    match_loss = loss_dict["loss"] / max(int(loss_dict["num"]), 1)
    entropy_loss = loss_dict["loss_entropy"] / max(int(loss_dict["num_unlabel"]), 1)
    l2sp = l2sp_penalty(model, anchors)
    total = match_loss + args.entropy_weight * entropy_loss + args.l2sp_weight * l2sp
    return total, {
        "shared": shared,
        "skip": 0.0,
        "match_loss": float(match_loss.detach().cpu()),
        "entropy_loss": float(entropy_loss.detach().cpu()),
        "l2sp": float(l2sp.detach().cpu()),
    }


@torch.no_grad()
def official_scores(model: nn.Module, reference: Worm, query: Worm, device: torch.device, precision: str) -> np.ndarray:
    model.eval()
    with autocast_context(device, precision):
        _, output = model([reference.xyz, query.xyz], match_dict=None, ref_idx=0, mode="eval")
    p_m = output["p_m"][1].detach().float().cpu().numpy()
    p_m = p_m[:query.n, : reference.n + 1]
    return np.asarray(p_m[:, : reference.n], dtype=np.float64)


def rank_of_gt(row: np.ndarray, gt: int) -> int:
    return 1 + int(np.sum(row > row[gt]))


def eval_direction(scores: np.ndarray, query: Worm, reference: Worm) -> list[dict[str, Any]]:
    qmap = unique_label_map(query.labels)
    rmap = unique_label_map(reference.labels)
    common = sorted(set(qmap) & set(rmap))
    if not common:
        return []

    rr, cc = linear_sum_assignment(-scores)
    hung = {int(r): int(c) for r, c in zip(rr, cc)}
    rows = []
    for lab in common:
        qi, gi = qmap[lab], rmap[lab]
        rank = rank_of_gt(scores[qi], gi)
        rows.append({
            "query_worm": query.worm_id,
            "reference_worm": reference.worm_id,
            "query_row": qi,
            "identity": lab,
            "rank": rank,
            "top1": float(rank == 1),
            "top3": float(rank <= 3),
            "top5": float(rank <= 5),
            "top10": float(rank <= 10),
            "rr": 1.0 / rank,
            "hungarian_top1": float(hung.get(qi, -1) == gi),
        })
    return rows


@torch.no_grad()
def evaluate(model: nn.Module, records: Sequence[Worm], device: torch.device, precision: str) -> dict[str, Any]:
    rows = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            a, b = records[i], records[j]
            s_ba = official_scores(model, a, b, device, precision)
            s_ab = official_scores(model, b, a, device, precision)
            rows.extend(eval_direction(s_ba, b, a))
            rows.extend(eval_direction(s_ab, a, b))
    if not rows:
        raise RuntimeError("No validation queries")
    return {
        "queries": len(rows),
        "ranking_top1": float(np.mean([r["top1"] for r in rows])),
        "top3": float(np.mean([r["top3"] for r in rows])),
        "top5": float(np.mean([r["top5"] for r in rows])),
        "top10": float(np.mean([r["top10"] for r in rows])),
        "mrr": float(np.mean([r["rr"] for r in rows])),
        "assignment_top1": float(np.mean([r["hungarian_top1"] for r in rows])),
    }


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_metric: float, args, metadata):
    torch.save({
        "format": "fdnc_official_source_finetune_v1",
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "metadata": metadata,
    }, path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["atanas", "rld"], required=True)
    p.add_argument("--fold", type=int, choices=[1,2,3,4,5], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--train-list", type=Path, default=None)
    p.add_argument("--val-list", type=Path, default=None)
    p.add_argument("--pretrained", type=Path, default=OFFICIAL_PRETRAINED)
    p.add_argument("--expected-pretrained-sha256", default="ab529eb6a886cb6ab3f199b7aaa4e49b82562dc280b3ac736ed61e25d5138ec9")
    p.add_argument("--save-dir", type=Path, required=True)
    p.add_argument("--mask-key", default="clean_mask")
    p.add_argument("--coordinate-scale", type=float, default=200.0)

    p.add_argument("--unfreeze-last-n", type=int, default=2)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--pairs-per-epoch", type=int, default=200)
    p.add_argument("--minimum-common", type=int, default=8)
    p.add_argument("--backbone-lr", type=float, default=5e-6)
    p.add_argument("--outlier-lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--entropy-weight", type=float, default=0.1)
    p.add_argument("--l2sp-weight", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--scale-jitter", type=float, default=0.05)
    p.add_argument("--coordinate-jitter", type=float, default=0.01)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    if args.train_list is None:
        args.train_list = PROTOCOLS / args.dataset / f"fold_{args.fold}" / f"seed_{args.seed}" / "train.txt"
    if args.val_list is None:
        args.val_list = PROTOCOLS / args.dataset / f"fold_{args.fold}" / f"seed_{args.seed}" / "val.txt"

    # No outer-test path exists anywhere in the training CLI.
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    observed_sha = sha256(args.pretrained)
    if observed_sha != args.expected_pretrained_sha256.lower():
        raise AssertionError(
            f"Official pretrained SHA mismatch: {observed_sha} != {args.expected_pretrained_sha256}"
        )

    train_records = load_split(args.train_list, args.mask_key, args.coordinate_scale)
    val_records = load_split(args.val_list, args.mask_key, args.coordinate_scale)

    train_ids = {r.worm_id for r in train_records}
    val_ids = {r.worm_id for r in val_records}
    if train_ids & val_ids:
        raise RuntimeError(f"Train/val overlap: {sorted(train_ids & val_ids)}")

    module = load_official_module()
    model = build_official_model(module, args.pretrained, device)
    trainable = configure_trainable(model, args.unfreeze_last_n)

    anchors = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    backbone_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("fc_outlier.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr, "base_lr": args.backbone_lr},
            {"params": list(model.fc_outlier.parameters()), "lr": args.outlier_lr, "base_lr": args.outlier_lr},
        ],
        weight_decay=args.weight_decay,
    )

    official_commit = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
    ).strip()

    metadata = {
        "official_repo": str(OFFICIAL.resolve()),
        "official_repo_commit": official_commit,
        "official_source_modified": False,
        "official_pretrained": str(args.pretrained.resolve()),
        "official_pretrained_sha256": observed_sha,
        "architecture": "NIT_Registration(input_dim=3,n_hidden=128,n_layer=6,p_rotate=False,feat_trans=False)",
        "train_list": str(args.train_list.resolve()),
        "val_list": str(args.val_list.resolve()),
        "outer_test_accessed_during_training": False,
        "train_worms": [r.worm_id for r in train_records],
        "val_worms": [r.worm_id for r in val_records],
        "selection_metric": "validation direct ranking_top1",
        "trainable_parameters": int(sum(p.numel() for p in trainable)),
    }
    (args.save_dir / "config.json").write_text(
        json.dumps({
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "metadata": metadata,
        }, indent=2) + "\n"
    )

    initial_val = evaluate(model, val_records, device, args.precision)
    (args.save_dir / "initial_validation.json").write_text(json.dumps(initial_val, indent=2) + "\n")

    pairs = [(i,j) for i in range(len(train_records)) for j in range(i+1, len(train_records))]
    best_metric = -1.0
    best_mrr = -1.0
    best_epoch = -1
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        warm = min(1.0, epoch / max(args.warmup_epochs, 1))
        for g in optimizer.param_groups:
            g["lr"] = g["base_lr"] * warm

        set_train_mode(model, args.unfreeze_last_n)
        random.shuffle(pairs)
        selected = pairs[: min(args.pairs_per_epoch, len(pairs))]

        epoch_losses = []
        used_directions = 0
        skipped_directions = 0

        for ia, ib in selected:
            a, b = train_records[ia], train_records[ib]
            optimizer.zero_grad(set_to_none=True)

            with autocast_context(device, args.precision):
                loss_ab, info_ab = train_direction(model, a, b, args, anchors)
                loss_ba, info_ba = train_direction(model, b, a, args, anchors)
                valid_losses = []
                if not info_ab["skip"]:
                    valid_losses.append(loss_ab)
                    used_directions += 1
                else:
                    skipped_directions += 1
                if not info_ba["skip"]:
                    valid_losses.append(loss_ba)
                    used_directions += 1
                else:
                    skipped_directions += 1

                if not valid_losses:
                    continue
                total = torch.stack(valid_losses).mean()

            total.backward()
            nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step()
            epoch_losses.append(float(total.detach().cpu()))

        validation = evaluate(model, val_records, device, args.precision)
        metric = float(validation["ranking_top1"])
        mrr = float(validation["mrr"])

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else math.nan,
            "train_pair_updates": len(epoch_losses),
            "used_directions": used_directions,
            "skipped_directions": skipped_directions,
            "val_top1": metric,
            "val_top3": validation["top3"],
            "val_top5": validation["top5"],
            "val_mrr": mrr,
            "val_hungarian": validation["assignment_top1"],
        }
        history.append(row)

        with (args.save_dir / "history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)

        # Primary selection Top-1; MRR only breaks exact Top-1 ties.
        better = (metric > best_metric + 1e-12) or (
            abs(metric - best_metric) <= 1e-12 and mrr > best_mrr
        )
        if better:
            best_metric = metric
            best_mrr = mrr
            best_epoch = epoch
            stale = 0
            save_checkpoint(args.save_dir / "best.pt", model, optimizer, epoch, best_metric, args, metadata)
            (args.save_dir / "best_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        else:
            stale += 1

        print(
            f"Epoch {epoch:03d} "
            f"loss={row['train_loss']:.6f} "
            f"val_top1={100*metric:.2f}% "
            f"val_mrr={mrr:.4f} "
            f"best={100*best_metric:.2f}%@{best_epoch}"
        )

        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    if best_epoch < 1:
        raise RuntimeError("No best checkpoint selected")

    ckpt = load_torch(args.save_dir / "best.pt")
    model.load_state_dict(ckpt["model_state"], strict=True)
    final_val = evaluate(model, val_records, device, args.precision)
    summary = {
        "dataset": args.dataset,
        "fold": args.fold,
        "seed": args.seed,
        "best_epoch": int(best_epoch),
        "best_validation": final_val,
        "test_data_accessed": False,
        "official_pretrained_sha256": observed_sha,
        "official_repo_commit": official_commit,
    }
    (args.save_dir / "final_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
