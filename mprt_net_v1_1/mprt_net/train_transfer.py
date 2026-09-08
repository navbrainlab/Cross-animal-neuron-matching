from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data import PairIndex, WormCache, WormSample, build_pair_targets, iter_pairs
from .evaluate import evaluate_model, load_checkpoint as load_model_checkpoint
from .losses import (
    six_way_cycle_consistency_loss,
    symmetric_focal_matching_loss,
    symmetric_focal_probability_loss,
)
from .model import MPRTNet


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def cycle_weight_for_epoch(
    base_weight: float,
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
) -> float:
    """Return a zero-warmup, linear-ramp cycle coefficient."""

    if base_weight <= 0.0 or epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return base_weight
    progress = min(1.0, (epoch - warmup_epochs) / float(ramp_epochs))
    return base_weight * progress


def choose_third_population(
    index: PairIndex,
    cache: WormCache,
    sample_a_source: str,
    sample_b_source: str,
    generator: torch.Generator,
) -> WormSample:
    """Deterministically sample a third, distinct training population."""

    excluded = {
        str(Path(sample_a_source).resolve()),
        str(Path(sample_b_source).resolve()),
    }
    candidates = [path for path in index.files if str(path.resolve()) not in excluded]
    if not candidates:
        raise ValueError("Cycle consistency requires at least three training animals")
    chosen = int(torch.randint(len(candidates), (1,), generator=generator).item())
    return cache.get(candidates[chosen])


def config_for_variant(args: argparse.Namespace) -> ModelConfig:
    values = dict(
        hidden_dim=args.hidden_dim,
        edge_dim=args.edge_dim,
        relation_dim=args.relation_dim,
        activity_channels=args.activity_channels,
        num_heads=args.num_heads,
        population_layers=args.population_layers,
        dropout=args.dropout,
        sinkhorn_iterations=args.sinkhorn_iterations,
        transport_steps=args.transport_steps,
        structural_weight=args.structural_weight,
        atlas_size=args.atlas_size,
        atlas_momentum=args.atlas_momentum,
        atlas_min_support=args.atlas_min_support,
        atlas_blend_weight=args.atlas_blend_weight,
        hard_knn_k=args.hard_knn_k,
    )
    if args.variant == "geometry_only":
        values.update(use_geometry=True, use_activity=False)
    elif args.variant == "activity_only":
        values.update(use_geometry=False, use_activity=True)
    elif args.variant == "no_population":
        values.update(use_population_encoder=False)
    elif args.variant == "no_transport":
        values.update(use_relation_transport=False)
    elif args.variant == "node_only":
        values.update(use_population_encoder=False, use_relation_transport=False)
    elif args.variant == "hard_knn":
        values.update(relation_graph_mode="hard_knn")
    elif args.variant == "geometry_relations":
        values.update(relation_graph_mode="geometry_only")
    elif args.variant == "activity_relations":
        values.update(relation_graph_mode="activity_only")
    elif args.variant == "no_edge_conditioning":
        values.update(use_edge_conditioning=False)
    return ModelConfig(**values)


def save_checkpoint(
    path: Path,
    model: MPRTNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_top1_real: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "epoch": epoch,
            "best_top1_real": best_top1_real,
            "selection_metric": "top1_real",
            "train_args": vars(args),
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MPRT-Net on one dataset")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=[
        "full", "geometry_only", "activity_only", "no_population", "no_transport",
        "node_only", "hard_knn", "geometry_relations", "activity_relations",
        "no_edge_conditioning",
    ], default="full")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pairs-per-epoch", type=int, default=128)
    parser.add_argument("--val-max-pairs", type=int, default=0)
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--min-shared", type=int, default=2)
    parser.add_argument("--synthetic-drop-probability", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--cycle-weight",
        type=float,
        default=0.0,
        help="Maximum weight of six-direction multi-animal cycle consistency.",
    )
    parser.add_argument(
        "--cycle-warmup-epochs",
        type=int,
        default=5,
        help="Number of supervised-only epochs before enabling cycle loss.",
    )
    parser.add_argument(
        "--cycle-ramp-epochs",
        type=int,
        default=10,
        help="Epochs used to linearly ramp cycle loss to --cycle-weight.",
    )
    parser.add_argument(
        "--atlas-weight",
        type=float,
        default=0.0,
        help="Maximum supervised weight of matching induced through the atlas.",
    )
    parser.add_argument(
        "--atlas-blend-weight",
        type=float,
        default=0.0,
        help="Maximum atlas probability blended into train and inference matching.",
    )
    parser.add_argument(
        "--atlas-warmup-epochs",
        type=int,
        default=2,
        help="Pairwise-only epochs before initializing the shared atlas.",
    )
    parser.add_argument(
        "--atlas-ramp-epochs",
        type=int,
        default=3,
        help="Epochs used to ramp atlas supervision and inference blending.",
    )
    parser.add_argument(
        "--atlas-size",
        type=int,
        default=0,
        help="Atlas slots; zero selects the largest training population size.",
    )
    parser.add_argument("--atlas-momentum", type=float, default=0.999)
    parser.add_argument("--atlas-min-support", type=float, default=0.05)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many non-improving epochs; zero disables stopping.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help=(
            "Initialize MPRT network weights from another dataset checkpoint. "
            "Optimizer/epoch/atlas state are NOT restored."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Allow writing into an existing run directory (disabled by default).",
    )

    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--edge-dim", type=int, default=48)
    parser.add_argument("--relation-dim", type=int, default=8)
    parser.add_argument("--activity-channels", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--population-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--sinkhorn-iterations", type=int, default=20)
    parser.add_argument("--transport-steps", type=int, default=2)
    parser.add_argument("--structural-weight", type=float, default=1.0)
    parser.add_argument(
        "--hard-knn-k",
        type=int,
        default=16,
        help="Number of non-self neighbors in the hard-KNN relation ablation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cycle_weight < 0.0:
        raise ValueError("--cycle-weight must be non-negative")
    if args.cycle_warmup_epochs < 0 or args.cycle_ramp_epochs < 0:
        raise ValueError("Cycle warmup and ramp epochs must be non-negative")
    if args.atlas_weight < 0.0:
        raise ValueError("--atlas-weight must be non-negative")
    if not 0.0 <= args.atlas_blend_weight <= 1.0:
        raise ValueError("--atlas-blend-weight must be in [0, 1]")
    if args.atlas_warmup_epochs < 0 or args.atlas_ramp_epochs < 0:
        raise ValueError("Atlas warmup and ramp epochs must be non-negative")
    if args.atlas_size < 0:
        raise ValueError("--atlas-size must be non-negative")
    if not 0.0 <= args.atlas_momentum < 1.0:
        raise ValueError("--atlas-momentum must be in [0, 1)")
    if not 0.0 <= args.atlas_min_support <= 1.0:
        raise ValueError("--atlas-min-support must be in [0, 1]")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.hard_knn_k < 1:
        raise ValueError("--hard-knn-k must be positive")
    atlas_enabled = args.atlas_weight > 0.0 or args.atlas_blend_weight > 0.0
    if atlas_enabled and args.cycle_weight > 0.0:
        raise ValueError("Run atlas and cycle objectives as separate controlled experiments")
    output_dir = Path(args.output_dir)
    existing_run_files = [
        output_dir / "history.jsonl",
        output_dir / "best.pt",
        output_dir / "last.pt",
    ]
    if not args.allow_existing_output and any(path.exists() for path in existing_run_files):
        raise FileExistsError(
            f"Run output already exists in {output_dir}. Use a new directory or pass "
            "--allow-existing-output deliberately."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = seed_everything(args.seed)
    cycle_generator = torch.Generator(device="cpu")
    cycle_generator.manual_seed(args.seed + 1_000_003)
    device = choose_device(args.device)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    train_index = PairIndex(args.dataset_root, "train", min_shared=args.min_shared)
    val_index = PairIndex(args.dataset_root, "val", min_shared=args.min_shared)
    cache = WormCache(activity_length=args.activity_length, max_items=48)
    atlas_reference = None
    if atlas_enabled:
        atlas_reference = max(
            (cache.get(path) for path in train_index.files),
            key=lambda sample: sample.num_nodes,
        )
        if args.atlas_size == 0:
            args.atlas_size = atlas_reference.num_nodes
        if args.atlas_size > atlas_reference.num_nodes:
            raise ValueError(
                f"atlas_size={args.atlas_size} exceeds the largest training "
                f"population ({atlas_reference.num_nodes})"
            )
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    model = MPRTNet(config_for_variant(args)).to(device)

    # ------------------------------------------------------------
    # Cross-dataset transfer initialization.
    # Load only architecture-compatible neural-network weights.
    # Do NOT transfer source atlas state, optimizer state, or epoch.
    # ------------------------------------------------------------
    if args.init_checkpoint is not None:
        init_path = Path(args.init_checkpoint)

        if not init_path.is_file():
            raise FileNotFoundError(init_path)

        source_model, source_payload = load_model_checkpoint(
            init_path,
            device,
        )

        source_state = source_model.state_dict()
        target_state = model.state_dict()

        def is_atlas_key(name: str) -> bool:
            return (
                name.startswith("atlas_")
                or ".atlas_" in name
            )

        compatible = {}

        for key, value in source_state.items():
            if is_atlas_key(key):
                continue

            if key not in target_state:
                continue

            if tuple(value.shape) != tuple(
                target_state[key].shape
            ):
                continue

            compatible[key] = value

        target_core = [
            key
            for key in target_state
            if not is_atlas_key(key)
        ]

        missing_core = [
            key
            for key in target_core
            if key not in compatible
        ]

        if missing_core:
            raise RuntimeError(
                "Atanas -> RLD initialization is not "
                "architecture-compatible. Missing core keys: "
                + str(missing_core[:30])
            )

        result = model.load_state_dict(
            compatible,
            strict=False,
        )

        invalid_missing = [
            key
            for key in result.missing_keys
            if not is_atlas_key(key)
        ]

        if invalid_missing:
            raise RuntimeError(
                "Unexpected missing non-atlas keys: "
                + str(invalid_missing[:30])
            )

        print("=" * 90)
        print("TRANSFER INITIALIZATION")
        print("=" * 90)
        print("source checkpoint :", init_path)
        print(
            "source epoch      :",
            source_payload.get("epoch"),
        )
        print(
            "loaded tensors    :",
            len(compatible),
        )
        print(
            "target core       :",
            len(target_core),
        )
        print("optimizer restored: NO")
        print("epoch restored    : NO")
        print("source atlas      : NO")
        print("=" * 90)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    if args.cycle_weight > 0.0 and len(train_index.files) < 3:
        raise ValueError("--cycle-weight > 0 requires at least three training animals")

    print(
        f"device={device} train_pairs={len(train_index.pairs)} "
        f"val_pairs={len(val_index.pairs)} variant={args.variant} "
        f"atlas_size={args.atlas_size if atlas_enabled else 0}"
    )
    best_top1_real = -1.0
    best_epoch = 0
    history_path = output_dir / "history.jsonl"
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        model.train()
        loss_sum = 0.0
        objective_sum = 0.0
        cycle_loss_sum = 0.0
        atlas_loss_sum = 0.0
        query_sum = 0
        updates = 0
        cycle_updates = 0
        atlas_updates = 0
        if (
            atlas_enabled
            and epoch > args.atlas_warmup_epochs
            and not model.atlas_is_initialized
        ):
            model.eval()
            with torch.no_grad():
                reference_encoding = model.encode_population(atlas_reference.to(device))
            model.initialize_atlas(reference_encoding)
            model.train()
        cycle_weight = cycle_weight_for_epoch(
            args.cycle_weight,
            epoch,
            args.cycle_warmup_epochs,
            args.cycle_ramp_epochs,
        )
        atlas_weight = cycle_weight_for_epoch(
            args.atlas_weight,
            epoch,
            args.atlas_warmup_epochs,
            args.atlas_ramp_epochs,
        )
        atlas_blend = cycle_weight_for_epoch(
            args.atlas_blend_weight,
            epoch,
            args.atlas_warmup_epochs,
            args.atlas_ramp_epochs,
        )
        model.set_atlas_blend(atlas_blend)
        for base_a, base_b in iter_pairs(
            train_index, cache, generator, limit=args.pairs_per_epoch
        ):
            sample_a, sample_b, targets = build_pair_targets(
                base_a,
                base_b,
                synthetic_drop_probability=args.synthetic_drop_probability,
                generator=generator,
            )
            if targets.num_direct_matches + targets.num_synthetic_unmatched == 0:
                continue
            sample_a = sample_a.to(device)
            sample_b = sample_b.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            atlas_match = None
            atlas_breakdown = None
            if model.atlas_is_initialized and (atlas_weight > 0.0 or atlas_blend > 0.0):
                encoding_a = model.encode_population(sample_a)
                encoding_b = model.encode_population(sample_b)
                atlas_match = model.match_with_atlas_encodings(
                    encoding_a,
                    encoding_b,
                    blend=atlas_blend,
                )
                output = atlas_match.output
                atlas_breakdown = symmetric_focal_probability_loss(
                    atlas_match.atlas_row_conditional,
                    atlas_match.atlas_col_conditional,
                    targets,
                    gamma=args.focal_gamma,
                )
                cycle = None
            elif cycle_weight > 0.0:
                sample_c = choose_third_population(
                    train_index,
                    cache,
                    sample_a.source_path,
                    sample_b.source_path,
                    cycle_generator,
                ).to(device)
                encoding_a = model.encode_population(sample_a)
                encoding_b = model.encode_population(sample_b)
                encoding_c = model.encode_population(sample_c)
                output = model.match_encodings(encoding_a, encoding_b)
                output_bc = model.match_encodings(encoding_b, encoding_c)
                output_ac = model.match_encodings(encoding_a, encoding_c)
                cycle = six_way_cycle_consistency_loss(output, output_bc, output_ac)
            else:
                output = model(sample_a, sample_b)
                cycle = None
            breakdown = symmetric_focal_matching_loss(
                output, targets, gamma=args.focal_gamma
            )
            objective = breakdown.total
            if cycle is not None:
                objective = objective + cycle_weight * cycle.total
            if atlas_breakdown is not None:
                objective = objective + atlas_weight * atlas_breakdown.total
            if not bool(torch.isfinite(objective)):
                raise FloatingPointError(
                    f"Non-finite loss for pair {sample_a.uid} / {sample_b.uid}"
                )
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            if atlas_match is not None:
                model.update_atlas(encoding_a, atlas_match.alignment_a)
                model.update_atlas(encoding_b, atlas_match.alignment_b)

            loss_sum += float(breakdown.total.detach()) * breakdown.num_queries
            objective_sum += float(objective.detach()) * breakdown.num_queries
            if cycle is not None:
                cycle_loss_sum += float(cycle.total.detach())
                cycle_updates += 1
            if atlas_breakdown is not None:
                atlas_loss_sum += float(atlas_breakdown.total.detach())
                atlas_updates += 1
            query_sum += breakdown.num_queries
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
            "train_loss": loss_sum / max(query_sum, 1),
            "train_objective": objective_sum / max(query_sum, 1),
            "train_cycle_loss": cycle_loss_sum / max(cycle_updates, 1),
            "cycle_weight": cycle_weight,
            "cycle_updates": cycle_updates,
            "train_atlas_loss": atlas_loss_sum / max(atlas_updates, 1),
            "atlas_weight": atlas_weight,
            "atlas_blend": atlas_blend,
            "atlas_updates": atlas_updates,
            "atlas_total_updates": (
                int(model.atlas_update_count.item()) if model.has_atlas else 0
            ),
            "atlas_mean_support": (
                float(model.atlas_support.mean())
                if model.atlas_is_initialized
                else 0.0
            ),
            "train_queries": query_sum,
            "updates": updates,
            "seconds": time.time() - started,
            "val": validation,
            "selection_metric": "top1_real",
            "unary_temperature": float(model.unary_temperature.detach()),
            "relation_temperature": float(model.relation_temperature.detach()),
            "structural_weight": float(model.structural_weight.detach()),
            "deletion_logit": float(model.deletion_logit.detach()),
            "insertion_logit": float(model.insertion_logit.detach()),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch:03d} loss={record['train_loss']:.5f} "
            f"cycle={record['train_cycle_loss']:.5f}@{cycle_weight:.4f} "
            f"atlas={record['train_atlas_loss']:.5f}@{atlas_weight:.4f} "
            f"blend={atlas_blend:.3f} "
            f"val_top1_real={100.0 * float(validation['top1_real']):.2f}% "
            f"val_top5_real={100.0 * float(validation['top5_real']):.2f}% "
            f"mrr_real={float(validation['mrr_real']):.4f} "
            f"top1_dustbin={100.0 * float(validation['top1_with_dustbin']):.2f}% "
            f"seconds={record['seconds']:.1f}"
        )

        current_top1_real = float(validation["top1_real"])
        if current_top1_real > best_top1_real:
            best_top1_real = current_top1_real
            best_epoch = epoch
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                best_top1_real,
                args,
            )
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            best_top1_real,
            args,
        )
        earliest_stop = max(
            1,
            args.atlas_warmup_epochs + args.atlas_ramp_epochs
            if atlas_enabled
            else 1,
        )
        if (
            args.early_stopping_patience > 0
            and epoch >= earliest_stop
            and epoch - best_epoch >= args.early_stopping_patience
        ):
            print(
                f"early_stop epoch={epoch} best_epoch={best_epoch} "
                f"patience={args.early_stopping_patience}"
            )
            break

    print(
        f"best_val_top1_real={100.0 * best_top1_real:.2f}% "
        f"checkpoint={output_dir / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
