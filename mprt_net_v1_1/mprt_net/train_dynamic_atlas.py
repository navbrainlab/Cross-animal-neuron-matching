from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig
from .data import PairIndex, WormCache, build_pair_targets, iter_pairs
from .dynamic_atlas import LeaveOneAnimalOutAtlasBank, build_atlas_targets
from .evaluate import evaluate_model
from .losses import symmetric_focal_matching_loss, symmetric_focal_probability_loss
from .model import MPRTNet


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _seed(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _dynamic_parameters(model: MPRTNet) -> list[torch.nn.Parameter]:
    if model.dynamic_atlas_adapter is None:
        raise RuntimeError("Dynamic atlas adapter was not constructed")
    return [parameter for parameter in model.dynamic_atlas_adapter.parameters()]


def _save_checkpoint(
    *,
    source: dict[str, Any],
    model: MPRTNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_top1_real: float,
    args: argparse.Namespace,
    output: Path,
) -> None:
    checkpoint = dict(source)
    checkpoint["model_state"] = model.state_dict()
    checkpoint["model_config"] = model.config.to_dict()
    checkpoint["optimizer_state"] = optimizer.state_dict()
    checkpoint["epoch"] = epoch
    checkpoint["best_top1_real"] = best_top1_real
    checkpoint["selection_metric"] = "top1_real"
    checkpoint["dynamic_atlas_train_args"] = vars(args)
    checkpoint["dynamic_atlas"] = {
        "method": args.conditioner,
        "encoder_frozen": True,
        "atlas_frozen": True,
        "cross_fitting": "leave-one-training-animal-out",
        "atlas_build_split": "train",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a cross-fitted dynamic residual relational atlas"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--static-atlas-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pairs-per-epoch", type=int, default=128)
    parser.add_argument("--val-max-pairs", type=int, default=0)
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--synthetic-drop-probability", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--pair-loss-weight", type=float, default=0.5)
    parser.add_argument("--magnitude-weight", type=float, default=0.02)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
    parser.add_argument("--distortion-weight", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--dynamic-hidden-dim", type=int, default=64)
    parser.add_argument("--coordinate-scale", type=float, default=0.15)
    parser.add_argument("--node-scale", type=float, default=0.10)
    parser.add_argument("--relation-scale", type=float, default=0.10)
    parser.add_argument("--smooth-k", type=int, default=8)
    parser.add_argument(
        "--conditioner",
        choices=("global_pool", "atlas_cross_attention"),
        default="global_pool",
    )
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--geometry-sigma", type=float, default=1.0)
    parser.add_argument("--geometry-weight", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-existing-output", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rank < 1 or args.dynamic_hidden_dim < 1 or args.smooth_k < 1:
        raise ValueError("Rank, hidden dimension, and smooth-k must be positive")
    if args.conditioner == "atlas_cross_attention":
        if args.attention_heads < 1:
            raise ValueError("--attention-heads must be positive")
        if args.dynamic_hidden_dim % args.attention_heads != 0:
            raise ValueError(
                "Dynamic hidden dimension must be divisible by attention heads"
            )
    if args.geometry_sigma <= 0.0:
        raise ValueError("--geometry-sigma must be positive")
    if args.geometry_weight < 0.0:
        raise ValueError("--geometry-weight must be non-negative")
    for name in (
        "pair_loss_weight",
        "magnitude_weight",
        "smoothness_weight",
        "distortion_weight",
        "coordinate_scale",
        "node_scale",
        "relation_scale",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0.0 <= args.synthetic_drop_probability < 1.0:
        raise ValueError("--synthetic-drop-probability must be in [0, 1)")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")

    output_dir = Path(args.output_dir)
    protected = [
        output_dir / "best.pt",
        output_dir / "last.pt",
        output_dir / "history.jsonl",
    ]
    if not args.allow_existing_output and any(path.exists() for path in protected):
        raise FileExistsError(f"Dynamic atlas output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    generator = _seed(args.seed)
    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    source_path = Path(args.static_atlas_checkpoint)
    source = torch.load(source_path, map_location=device)
    identity_to_slot = source.get("atlas_identity_to_slot")
    if not isinstance(identity_to_slot, dict) or not identity_to_slot:
        raise ValueError("Static checkpoint has no anchored atlas identity map")
    identity_to_slot = {str(key): int(value) for key, value in identity_to_slot.items()}
    source_config = ModelConfig.from_dict(source["model_config"])
    if source_config.atlas_size != len(identity_to_slot):
        raise ValueError("Atlas size and identity map disagree")
    if source_config.dynamic_atlas_enabled:
        raise ValueError("The source checkpoint must be a frozen static atlas")

    config_values = source_config.to_dict()
    config_values.update(
        dynamic_atlas_enabled=True,
        dynamic_atlas_rank=args.rank,
        dynamic_atlas_hidden_dim=args.dynamic_hidden_dim,
        dynamic_atlas_coordinate_scale=args.coordinate_scale,
        dynamic_atlas_node_scale=args.node_scale,
        dynamic_atlas_relation_scale=args.relation_scale,
        dynamic_atlas_smooth_k=args.smooth_k,
        dynamic_atlas_conditioner=args.conditioner,
        dynamic_atlas_attention_heads=args.attention_heads,
        dynamic_atlas_geometry_sigma=args.geometry_sigma,
        dynamic_atlas_geometry_weight=args.geometry_weight,
        atlas_blend_weight=1.0,
        atlas_confidence_gating=False,
    )
    model = MPRTNet(ModelConfig.from_dict(config_values)).to(device)
    incompatible = model.load_state_dict(source["model_state"], strict=False)
    allowed_missing = (
        "atlas_xyz",
        "atlas_relation_support",
        "dynamic_atlas_adapter.",
    )
    unexpected_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_missing)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Incompatible static checkpoint: missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    # Do not retain a second GPU copy of the frozen source model/optimizer.
    source.pop("model_state", None)
    source.pop("optimizer_state", None)

    train_index = PairIndex(args.dataset_root, "train", min_shared=args.min_shared)
    val_index = PairIndex(args.dataset_root, "val", min_shared=args.min_shared)
    cache = WormCache(
        activity_length=args.activity_length,
        max_items=max(48, len(train_index.files) + len(val_index.files)),
    )
    bank = LeaveOneAnimalOutAtlasBank.build(
        model=model,
        files=train_index.files,
        cache=cache,
        identity_to_slot=identity_to_slot,
        device=device,
    )
    full_atlas = bank.prototype()
    model.initialize_atlas_from_prototypes(
        full_atlas.nodes,
        full_atlas.relations,
        full_atlas.support,
        coordinates=full_atlas.coordinates,
        relation_support=full_atlas.relation_support,
    )
    del full_atlas
    model.set_atlas_blend(1.0)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = _dynamic_parameters(model)
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(
        f"device={device} train_pairs={len(train_index.pairs)} "
        f"val_pairs={len(val_index.pairs)} atlas_size={model.config.atlas_size} "
        f"trainable_parameters={sum(parameter.numel() for parameter in trainable)}",
        flush=True,
    )
    history_path = output_dir / "history.jsonl"
    best_top1_real = -1.0
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        # The frozen encoder must stay deterministic.  The residual adapter
        # has no dropout or running statistics, but mark it trainable for
        # future-compatible modules.
        model.eval()
        model.dynamic_atlas_adapter.train()
        objective_sum = 0.0
        atlas_loss_sum = 0.0
        pair_loss_sum = 0.0
        magnitude_sum = 0.0
        smoothness_sum = 0.0
        distortion_sum = 0.0
        updates = 0
        query_sum = 0

        for base_a, base_b in iter_pairs(
            train_index, cache, generator, limit=args.pairs_per_epoch
        ):
            sample_a, sample_b, pair_targets = build_pair_targets(
                base_a,
                base_b,
                synthetic_drop_probability=args.synthetic_drop_probability,
                generator=generator,
            )
            sample_a = sample_a.to(device)
            sample_b = sample_b.to(device)
            pair_targets = pair_targets.to(device)
            if (
                pair_targets.num_direct_matches
                + pair_targets.num_synthetic_unmatched
                == 0
            ):
                continue
            with torch.no_grad():
                encoding_a = model.encode_population(sample_a)
                encoding_b = model.encode_population(sample_b)
                atlas_a = bank.prototype(exclude_source=sample_a.source_path)
                atlas_b = bank.prototype(exclude_source=sample_b.source_path)
                targets_a = build_atlas_targets(
                    sample_a, identity_to_slot, atlas_a.support
                )
                targets_b = build_atlas_targets(
                    sample_b, identity_to_slot, atlas_b.support
                )
            if targets_a.num_direct_matches + targets_b.num_direct_matches == 0:
                continue

            optimizer.zero_grad(set_to_none=True)
            dynamic_a = model.deform_atlas(encoding_a, atlas_a)
            dynamic_b = model.deform_atlas(encoding_b, atlas_b)
            alignment_a = model.match_encodings(encoding_a, dynamic_a.encoding)
            alignment_b = model.match_encodings(encoding_b, dynamic_b.encoding)
            alignment_losses = []
            alignment_queries = 0
            if targets_a.num_direct_matches > 0:
                loss_a = symmetric_focal_matching_loss(
                    alignment_a, targets_a, gamma=args.focal_gamma
                )
                alignment_losses.append(loss_a.total)
                alignment_queries += loss_a.num_queries
            if targets_b.num_direct_matches > 0:
                loss_b = symmetric_focal_matching_loss(
                    alignment_b, targets_b, gamma=args.focal_gamma
                )
                alignment_losses.append(loss_b.total)
                alignment_queries += loss_b.num_queries
            atlas_loss = torch.stack(alignment_losses).mean()

            atlas_row, atlas_column = model._atlas_induced_transitions(
                alignment_a, alignment_b
            )
            pair_loss = symmetric_focal_probability_loss(
                atlas_row,
                atlas_column,
                pair_targets,
                gamma=args.focal_gamma,
            ).total
            regularization_a = model.dynamic_atlas_adapter.regularization(
                atlas_a, dynamic_a
            )
            regularization_b = model.dynamic_atlas_adapter.regularization(
                atlas_b, dynamic_b
            )
            regularization = {
                key: 0.5 * (regularization_a[key] + regularization_b[key])
                for key in regularization_a
            }
            objective = (
                atlas_loss
                + args.pair_loss_weight * pair_loss
                + args.magnitude_weight * regularization["magnitude"]
                + args.smoothness_weight * regularization["smoothness"]
                + args.distortion_weight * regularization["distortion"]
            )
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(
                    f"Non-finite dynamic atlas loss for {sample_a.uid}/{sample_b.uid}"
                )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step()

            queries = alignment_queries
            objective_sum += float(objective.detach()) * queries
            atlas_loss_sum += float(atlas_loss.detach()) * queries
            pair_loss_sum += float(pair_loss.detach()) * queries
            magnitude_sum += float(regularization["magnitude"].detach())
            smoothness_sum += float(regularization["smoothness"].detach())
            distortion_sum += float(regularization["distortion"].detach())
            query_sum += queries
            updates += 1

        validation = evaluate_model(
            model,
            val_index,
            cache,
            device,
            max_pairs=args.val_max_pairs if args.val_max_pairs > 0 else None,
        )
        record = {
            "epoch": epoch,
            "train_objective": objective_sum / max(query_sum, 1),
            "train_atlas_loss": atlas_loss_sum / max(query_sum, 1),
            "train_pair_loss": pair_loss_sum / max(query_sum, 1),
            "magnitude": magnitude_sum / max(updates, 1),
            "smoothness": smoothness_sum / max(updates, 1),
            "distortion": distortion_sum / max(updates, 1),
            "updates": updates,
            "train_queries": query_sum,
            "seconds": time.time() - started,
            "val": validation,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch:03d} objective={record['train_objective']:.5f} "
            f"atlas={record['train_atlas_loss']:.5f} "
            f"pair={record['train_pair_loss']:.5f} "
            f"mag={record['magnitude']:.6f} "
            f"smooth={record['smoothness']:.6f} "
            f"distort={record['distortion']:.6f} "
            f"val_top1_real={100.0 * float(validation['top1_real']):.2f}% "
            f"seconds={record['seconds']:.1f}",
            flush=True,
        )

        current = float(validation["top1_real"])
        if current > best_top1_real:
            best_top1_real = current
            best_epoch = epoch
            _save_checkpoint(
                source=source,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_top1_real=best_top1_real,
                args=args,
                output=output_dir / "best.pt",
            )
        _save_checkpoint(
            source=source,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_top1_real=best_top1_real,
            args=args,
            output=output_dir / "last.pt",
        )
        if (
            args.early_stopping_patience > 0
            and epoch - best_epoch >= args.early_stopping_patience
        ):
            print(
                f"early_stop epoch={epoch} best_epoch={best_epoch} "
                f"patience={args.early_stopping_patience}",
                flush=True,
            )
            break

    print(
        f"best_val_top1_real={100.0 * best_top1_real:.2f}% "
        f"checkpoint={output_dir / 'best.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
