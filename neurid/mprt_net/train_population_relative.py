from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .config import ModelConfig
from .data import WormCache, split_files, unique_identity_map
from .dynamic_atlas import (
    LeaveOneAnimalOutAtlasBank,
    build_atlas_targets,
    build_open_set_episode,
    canonical_source,
)
from .losses import symmetric_focal_matching_loss
from .model import MPRTNet


def seed_all(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def device_from(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else (
        "cpu" if value == "auto" else value
    ))


def identity_slots(files: list[Path], cache: WormCache) -> dict[str, int]:
    identities = sorted({
        identity
        for path in files
        for identity in unique_identity_map(cache.get(path))
    })
    if not identities:
        raise RuntimeError("No supervised train identities")
    return {identity: slot for slot, identity in enumerate(identities)}


@torch.no_grad()
def validate(
    model: MPRTNet,
    train_files: list[Path],
    val_files: list[Path],
    cache: WormCache,
    identity_to_slot: dict[str, int],
    device: torch.device,
    max_recordings: int,
) -> dict[str, float | int]:
    bank = LeaveOneAnimalOutAtlasBank.build(
        model=model,
        files=train_files,
        cache=cache,
        identity_to_slot=identity_to_slot,
        device=device,
    )
    atlas = bank.prototype()
    model.eval()
    totals = {"queries": 0, "top1": 0, "top5": 0, "rr": 0.0}
    files = val_files[:max_recordings] if max_recordings > 0 else val_files
    for path in files:
        sample = cache.get(path).to(device)
        targets = build_atlas_targets(sample, identity_to_slot, atlas.support).to(device)
        output = model.match_encodings(model.encode_population(sample), atlas)
        valid = (targets.row_target >= 0) & (targets.row_target < atlas.nodes.shape[0])
        rows = valid.nonzero(as_tuple=False).flatten()
        if not rows.numel():
            continue
        truth = targets.row_target[rows]
        scores = output.row_conditional[rows, :-1]
        order = scores.argsort(dim=1, descending=True, stable=True)
        ranks = (order == truth[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        totals["queries"] += int(rows.numel())
        totals["top1"] += int((ranks <= 1).sum())
        totals["top5"] += int((ranks <= 5).sum())
        totals["rr"] += float((1.0 / ranks.float()).sum())
    q = max(int(totals["queries"]), 1)
    return {
        "recordings": len(files),
        "queries": int(totals["queries"]),
        "top1_real": int(totals["top1"]) / q,
        "top5_real": int(totals["top5"]) / q,
        "mrr_real": float(totals["rr"]) / q,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the population-relative quadratic matcher with LOO atlas episodes"
    )
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--episodes-per-epoch", type=int, default=128)
    p.add_argument("--activity-length", type=int, default=512)
    p.add_argument("--support-drop-probability", type=float, default=0.05)
    p.add_argument("--query-drop-probability", type=float, default=0.05)
    p.add_argument("--relation-reliability-lambda", type=float, default=1.0)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--edge-dim", type=int, default=48)
    p.add_argument("--relation-dim", type=int, default=8)
    p.add_argument("--activity-channels", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--population-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--sinkhorn-iterations", type=int, default=20)
    p.add_argument("--transport-steps", type=int, default=2)
    p.add_argument("--structural-weight", type=float, default=1.0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gradient-clip", type=float, default=1.0)
    p.add_argument("--val-max-recordings", type=int, default=0)
    p.add_argument("--early-stopping-patience", type=int, default=12)
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-existing-output", action="store_true")
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume exactly from output-dir/last.pt and append to history.jsonl.",
    )
    return p


def main() -> None:
    args = parser().parse_args()
    for name in ("support_drop_probability", "query_drop_probability"):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")
    if args.relation_reliability_lambda < 0.0:
        raise ValueError("relation_reliability_lambda must be non-negative")
    output = Path(args.output_dir)
    protected = [output / "best.pt", output / "last.pt", output / "history.jsonl"]
    if args.resume and not (output / "last.pt").is_file():
        raise FileNotFoundError(f"--resume requires {output / 'last.pt'}")
    if not args.resume and not args.allow_existing_output and any(path.exists() for path in protected):
        raise FileExistsError(f"Refusing to overwrite existing attempt: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "args.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    generator = seed_all(args.seed)
    device = device_from(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    train_files = split_files(args.dataset_root, "train")
    val_files = split_files(args.dataset_root, "val")
    cache = WormCache(
        activity_length=args.activity_length,
        max_items=len(train_files) + len(val_files),
    )
    identity_to_slot = identity_slots(train_files, cache)
    config = ModelConfig(
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
        relation_objective="population_relative_quadratic",
        atlas_relation_reliability_lambda=args.relation_reliability_lambda,
    )
    model = MPRTNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    history_path = output / "history.jsonl"
    best = -1.0
    best_epoch = 0
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(output / "last.pt", map_location=device)
        saved_config = ModelConfig.from_dict(checkpoint["model_config"])
        if saved_config != config:
            raise ValueError("Resume checkpoint model_config differs from requested config")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best = float(checkpoint.get("best_top1_real", -1.0))
        best_path = output / "best.pt"
        best_epoch = (
            int(torch.load(best_path, map_location="cpu")["epoch"])
            if best_path.is_file() else int(checkpoint["epoch"])
        )
        rng_state = checkpoint.get("rng_state")
        if rng_state is not None:
            random.setstate(rng_state["python"])
            np.random.set_state(rng_state["numpy"])
            torch.set_rng_state(rng_state["torch"].cpu())
            generator.set_state(rng_state["episode_generator"].cpu())
            if torch.cuda.is_available() and rng_state.get("cuda") is not None:
                torch.cuda.set_rng_state_all([state.cpu() for state in rng_state["cuda"]])
        else:
            print(
                "warning: legacy partial checkpoint has no RNG snapshot; "
                "resume remains deterministic but is not bitwise-equivalent "
                "to an uninterrupted run",
                flush=True,
            )
        print(
            f"resume last_epoch={checkpoint['epoch']} start_epoch={start_epoch} "
            f"best_epoch={best_epoch} best_top1={best:.6f}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        # Refresh detached support prototypes once per epoch so they track the
        # evolving encoder without retaining a graph across episodes.
        bank = LeaveOneAnimalOutAtlasBank.build(
            model=model,
            files=train_files,
            cache=cache,
            identity_to_slot=identity_to_slot,
            device=device,
        )
        eligible_query_files = [
            path for path in train_files
            if canonical_source(path) in bank.contributions
        ]
        excluded_query_files = len(train_files) - len(eligible_query_files)
        if not eligible_query_files:
            raise RuntimeError("No train recordings contribute supervised atlas identities")
        model.train()
        loss_sum = 0.0
        query_sum = 0
        direct_sum = 0
        unmatched_sum = 0
        objective_steps = 0
        objective_decreases = 0
        for _ in range(args.episodes_per_epoch):
            index = int(torch.randint(len(eligible_query_files), (), generator=generator))
            base = cache.get(eligible_query_files[index])
            query, atlas, targets, episode = build_open_set_episode(
                base,
                bank,
                identity_to_slot,
                support_drop_probability=args.support_drop_probability,
                query_drop_probability=args.query_drop_probability,
                generator=generator,
            )
            if targets.num_direct_matches + targets.num_synthetic_unmatched == 0:
                continue
            query = query.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model.match_encodings(model.encode_population(query), atlas)
            breakdown = symmetric_focal_matching_loss(
                result, targets, gamma=args.focal_gamma
            )
            if not bool(torch.isfinite(breakdown.total)):
                raise FloatingPointError(f"Non-finite loss in episode {base.uid}")
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            optimizer.step()
            loss_sum += float(breakdown.total.detach()) * breakdown.num_queries
            query_sum += breakdown.num_queries
            direct_sum += episode["direct_matches"]
            unmatched_sum += episode["synthetic_unmatched"]
            objectives = [float(value.detach()) for value in result.quadratic_objectives]
            objective_steps += max(len(objectives) - 1, 0)
            objective_decreases += sum(
                later <= earlier + 1e-8
                for earlier, later in zip(objectives, objectives[1:])
            )

        validation = validate(
            model, train_files, val_files, cache, identity_to_slot, device,
            args.val_max_recordings,
        )
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / max(query_sum, 1),
            "train_supervised_queries": query_sum,
            "train_direct_matches": direct_sum,
            "train_synthetic_unmatched": unmatched_sum,
            "eligible_episode_recordings": len(eligible_query_files),
            "excluded_zero_supervision_recordings": excluded_query_files,
            "quadratic_objective_nonincrease_fraction": (
                objective_decreases / max(objective_steps, 1)
            ),
            "validation": validation,
            "seconds": time.time() - started,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "model_config": model.config.to_dict(),
            "atlas_identity_to_slot": identity_to_slot,
            "epoch": epoch,
            "best_top1_real": max(best, float(validation["top1_real"])),
            "selection_metric": "episodic_atlas_val_top1_real",
            "train_args": vars(args),
            "method": "population-relative partial quadratic matching",
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "episode_generator": generator.get_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        torch.save(checkpoint, output / "last.pt")
        score = float(validation["top1_real"])
        if score > best:
            best, best_epoch = score, epoch
            torch.save(checkpoint, output / "best.pt")
        print(json.dumps(record), flush=True)
        if args.early_stopping_patience and epoch - best_epoch >= args.early_stopping_patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    completion = {
        "status": "complete",
        "best_epoch": best_epoch,
        "best_val_top1_real": best,
        "output": str(output),
    }
    (output / "complete.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
