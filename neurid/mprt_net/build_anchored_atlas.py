from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .data import WormCache, split_files, unique_identity_map
from .model import MPRTNet


_ATLAS_BUFFER_KEYS = {
    "atlas_nodes",
    "atlas_relations",
    "atlas_support",
    "atlas_initialized_flag",
    "atlas_update_count",
    "atlas_blend",
}


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _save_checkpoint(
    source: dict[str, Any],
    model: MPRTNet,
    identity_to_slot: dict[str, int],
    output: Path,
    build_metadata: dict[str, Any],
) -> None:
    checkpoint = copy.deepcopy(source)
    checkpoint.pop("optimizer_state", None)
    checkpoint["model_state"] = model.state_dict()
    checkpoint["model_config"] = model.config.to_dict()
    checkpoint["atlas_identity_to_slot"] = identity_to_slot
    checkpoint["atlas_build"] = build_metadata
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)


@torch.no_grad()
def build_anchored_atlas(
    model: MPRTNet,
    files: list[Path],
    cache: WormCache,
    identity_to_slot: dict[str, int],
    device: torch.device,
) -> dict[str, float | int]:
    size = len(identity_to_slot)
    node_sum = torch.zeros(size, model.config.hidden_dim, device=device)
    node_count = torch.zeros(size, device=device)
    relation_sum = torch.zeros(
        size, size, model.config.relation_dim, device=device
    )
    relation_count = torch.zeros(size, size, device=device)

    model.eval()
    for number, path in enumerate(files, start=1):
        sample = cache.get(path)
        identity_map = unique_identity_map(sample)
        identities = sorted(identity_map)
        if not identities:
            continue
        node_indices = torch.tensor(
            [identity_map[identity] for identity in identities],
            dtype=torch.long,
            device=device,
        )
        slots = torch.tensor(
            [identity_to_slot[identity] for identity in identities],
            dtype=torch.long,
            device=device,
        )
        encoding = model.encode_population(sample.to(device))
        anchored_nodes = F.normalize(
            encoding.nodes.index_select(0, node_indices), dim=-1
        )
        node_sum.index_add_(0, slots, anchored_nodes)
        node_count.index_add_(0, slots, torch.ones_like(slots, dtype=node_sum.dtype))

        anchored_relations = encoding.relations.index_select(
            0, node_indices
        ).index_select(1, node_indices)
        flat_slots = (slots[:, None] * size + slots[None, :]).reshape(-1)
        relation_sum.view(-1, model.config.relation_dim).index_add_(
            0,
            flat_slots,
            anchored_relations.reshape(-1, model.config.relation_dim),
        )
        relation_count.view(-1).index_add_(
            0,
            flat_slots,
            torch.ones_like(flat_slots, dtype=relation_count.dtype),
        )
        print(
            f"atlas_build recording={number:03d}/{len(files):03d} "
            f"uid={sample.uid} anchors={len(identities)}"
        )

    if bool((node_count == 0).any()):
        missing = int((node_count == 0).sum())
        raise RuntimeError(f"{missing} atlas identities received no observations")
    node_prototypes = node_sum / node_count[:, None]
    relation_prototypes = relation_sum / relation_count[..., None].clamp_min(1.0)
    relation_mask = relation_count > 0
    reliability_lambda = model.config.atlas_relation_reliability_lambda
    relation_support = (
        relation_count / (relation_count + reliability_lambda)
        if reliability_lambda > 0.0
        else relation_mask.to(relation_count.dtype)
    )
    relation_prototypes = relation_prototypes * relation_mask[..., None]
    support = node_count / node_count.max().clamp_min(1.0)
    model.initialize_atlas_from_prototypes(
        node_prototypes,
        relation_prototypes,
        support,
        relation_support=relation_support,
        relation_count=relation_count,
    )
    return {
        "recordings": len(files),
        "atlas_size": size,
        "node_observations": int(node_count.sum()),
        "min_identity_observations": int(node_count.min()),
        "max_identity_observations": int(node_count.max()),
        "relation_coverage": float((relation_count > 0).float().mean()),
        "relation_reliability_lambda": reliability_lambda,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a training-label-anchored shared relational atlas"
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pure-output", default=None)
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--activity-length", type=int, default=512)
    parser.add_argument("--blend-weight", type=float, default=0.30)
    parser.add_argument("--gate-temperature", type=float, default=0.05)
    parser.add_argument(
        "--relation-mask",
        action="store_true",
        help="Mask atlas identity pairs that never co-occurred in outer-train.",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not 0.0 <= args.blend_weight <= 1.0:
        raise ValueError("--blend-weight must be in [0, 1]")
    if args.gate_temperature <= 0.0:
        raise ValueError("--gate-temperature must be positive")
    requested_outputs = [Path(args.output)]
    if args.pure_output:
        requested_outputs.append(Path(args.pure_output))
    existing_outputs = [path for path in requested_outputs if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "Refusing to overwrite anchored atlas checkpoint(s): "
            + ", ".join(str(path) for path in existing_outputs)
        )

    device = _device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    source = torch.load(args.checkpoint, map_location=device)
    source_config = ModelConfig.from_dict(source["model_config"])
    if source_config.atlas_size != 0:
        raise ValueError("The source checkpoint must be the atlas-free baseline")

    files = split_files(args.dataset_root, args.split)
    cache = WormCache(activity_length=args.activity_length, max_items=max(48, len(files)))
    identities = sorted(
        {
            identity
            for path in files
            for identity in unique_identity_map(cache.get(path))
        }
    )
    if not identities:
        raise ValueError("No unique supervised identities were found in the train split")
    identity_to_slot = {identity: slot for slot, identity in enumerate(identities)}

    config_values = source_config.to_dict()
    config_values.update(
        atlas_size=len(identities),
        atlas_blend_weight=args.blend_weight,
        atlas_confidence_gating=True,
        atlas_gate_temperature=args.gate_temperature,
        atlas_relation_masking=(
            bool(args.relation_mask)
            or bool(source_config.atlas_relation_masking)
        ),
    )
    model = MPRTNet(ModelConfig.from_dict(config_values)).to(device)
    incompatible = model.load_state_dict(source["model_state"], strict=False)
    expected_missing = set(_ATLAS_BUFFER_KEYS)
    if model.atlas_relation_support is not None:
        expected_missing.add("atlas_relation_support")
    if model.atlas_relation_count is not None:
        expected_missing.add("atlas_relation_count")
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(f"Unexpected missing state keys: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected source state keys: {incompatible.unexpected_keys}")

    metadata = build_anchored_atlas(
        model,
        files,
        cache,
        identity_to_slot,
        device,
    )
    metadata.update(
        {
            "source_checkpoint": str(Path(args.checkpoint).resolve()),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "split": args.split,
            "blend_weight": args.blend_weight,
            "confidence_gating": True,
            "gate_temperature": args.gate_temperature,
        }
    )
    model.set_atlas_blend(args.blend_weight)
    output = Path(args.output)
    _save_checkpoint(source, model, identity_to_slot, output, metadata)
    print(f"saved_anchored_atlas={output}")
    print(metadata)

    if args.pure_output:
        pure_model = copy.deepcopy(model)
        pure_model.config.atlas_confidence_gating = False
        pure_model.config.atlas_blend_weight = 1.0
        pure_model.set_atlas_blend(1.0)
        pure_metadata = dict(metadata)
        pure_metadata.update(blend_weight=1.0, confidence_gating=False)
        pure_output = Path(args.pure_output)
        _save_checkpoint(
            source,
            pure_model,
            identity_to_slot,
            pure_output,
            pure_metadata,
        )
        print(f"saved_pure_anchored_atlas={pure_output}")


if __name__ == "__main__":
    main()
