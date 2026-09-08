from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .data import PairTargets, WormCache, WormSample, unique_identity_map
from .model import MPRTNet, PopulationEncoding


def canonical_source(path: str | Path) -> str:
    return str(Path(path).resolve())


@dataclass
class AtlasContribution:
    slots: torch.Tensor
    nodes: torch.Tensor
    coordinates: torch.Tensor
    relations: torch.Tensor


class LeaveOneAnimalOutAtlasBank:
    """Training-only sufficient statistics for cross-fitted atlas prototypes."""

    def __init__(
        self,
        *,
        identity_to_slot: Mapping[str, int],
        node_sum: torch.Tensor,
        coordinate_sum: torch.Tensor,
        node_count: torch.Tensor,
        relation_sum: torch.Tensor,
        relation_count: torch.Tensor,
        contributions: Mapping[str, AtlasContribution],
        relation_reliability_lambda: float = 0.0,
    ) -> None:
        self.identity_to_slot = dict(identity_to_slot)
        self.node_sum = node_sum
        self.coordinate_sum = coordinate_sum
        self.node_count = node_count
        self.relation_sum = relation_sum
        self.relation_count = relation_count
        self.contributions = dict(contributions)
        self.relation_reliability_lambda = float(relation_reliability_lambda)

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        model: MPRTNet,
        files: Sequence[Path],
        cache: WormCache,
        identity_to_slot: Mapping[str, int],
        device: torch.device,
    ) -> "LeaveOneAnimalOutAtlasBank":
        size = len(identity_to_slot)
        hidden = model.config.hidden_dim
        relation_dim = model.config.relation_dim
        node_sum = torch.zeros(size, hidden, device=device)
        coordinate_sum = torch.zeros(size, 3, device=device)
        node_count = torch.zeros(size, device=device)
        relation_sum = torch.zeros(size, size, relation_dim, device=device)
        relation_count = torch.zeros(size, size, device=device)
        contributions: dict[str, AtlasContribution] = {}

        model.eval()
        for number, path in enumerate(files, start=1):
            sample = cache.get(path)
            identity_map = unique_identity_map(sample)
            identities = sorted(set(identity_map).intersection(identity_to_slot))
            if not identities:
                continue
            indices = torch.tensor(
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
            if encoding.coordinates is None:
                raise RuntimeError("The population encoder did not return coordinates")
            nodes = F.normalize(encoding.nodes.index_select(0, indices), dim=-1)
            coordinates = encoding.coordinates.index_select(0, indices)
            relations = encoding.relations.index_select(0, indices).index_select(
                1, indices
            )

            node_sum.index_add_(0, slots, nodes)
            coordinate_sum.index_add_(0, slots, coordinates)
            node_count.index_add_(0, slots, torch.ones_like(slots, dtype=node_sum.dtype))
            flat_slots = (slots[:, None] * size + slots[None, :]).reshape(-1)
            relation_sum.view(-1, relation_dim).index_add_(
                0, flat_slots, relations.reshape(-1, relation_dim)
            )
            relation_count.view(-1).index_add_(
                0,
                flat_slots,
                torch.ones_like(flat_slots, dtype=relation_count.dtype),
            )
            source = canonical_source(sample.source_path)
            if source in contributions:
                raise ValueError(f"Duplicate training source: {source}")
            contributions[source] = AtlasContribution(
                slots=slots,
                nodes=nodes,
                coordinates=coordinates,
                relations=relations,
            )
            print(
                f"dynamic_atlas_bank recording={number:03d}/{len(files):03d} "
                f"uid={sample.uid} anchors={len(identities)}",
                flush=True,
            )

        if bool((node_count == 0).any()):
            missing = int((node_count == 0).sum())
            raise RuntimeError(f"{missing} atlas identities received no observations")
        return cls(
            identity_to_slot=identity_to_slot,
            node_sum=node_sum,
            coordinate_sum=coordinate_sum,
            node_count=node_count,
            relation_sum=relation_sum,
            relation_count=relation_count,
            contributions=contributions,
            relation_reliability_lambda=model.config.atlas_relation_reliability_lambda,
        )

    def prototype(
        self,
        exclude_source: str | Path | None = None,
        drop_slots: torch.Tensor | None = None,
    ) -> PopulationEncoding:
        node_sum = self.node_sum.clone()
        coordinate_sum = self.coordinate_sum.clone()
        node_count = self.node_count.clone()
        relation_sum = self.relation_sum.clone()
        relation_count = self.relation_count.clone()

        if exclude_source is not None:
            source = canonical_source(exclude_source)
            if source not in self.contributions:
                raise KeyError(f"No atlas contribution for training source: {source}")
            contribution = self.contributions[source]
            slots = contribution.slots
            node_sum.index_add_(0, slots, -contribution.nodes)
            coordinate_sum.index_add_(0, slots, -contribution.coordinates)
            node_count.index_add_(
                0, slots, -torch.ones_like(slots, dtype=node_count.dtype)
            )
            size = node_count.shape[0]
            flat_slots = (slots[:, None] * size + slots[None, :]).reshape(-1)
            relation_sum.view(-1, relation_sum.shape[-1]).index_add_(
                0,
                flat_slots,
                -contribution.relations.reshape(-1, relation_sum.shape[-1]),
            )
            relation_count.view(-1).index_add_(
                0,
                flat_slots,
                -torch.ones_like(flat_slots, dtype=relation_count.dtype),
            )

        if drop_slots is not None:
            drop_slots = drop_slots.to(device=node_count.device, dtype=torch.long)
            if drop_slots.numel():
                node_sum.index_fill_(0, drop_slots, 0.0)
                coordinate_sum.index_fill_(0, drop_slots, 0.0)
                node_count.index_fill_(0, drop_slots, 0.0)
                relation_sum.index_fill_(0, drop_slots, 0.0)
                relation_sum.index_fill_(1, drop_slots, 0.0)
                relation_count.index_fill_(0, drop_slots, 0.0)
                relation_count.index_fill_(1, drop_slots, 0.0)

        node_count = node_count.clamp_min(0.0)
        relation_count = relation_count.clamp_min(0.0)
        valid = node_count > 0
        nodes = node_sum / node_count[:, None].clamp_min(1.0)
        nodes = F.normalize(nodes, dim=-1) * valid[:, None]
        coordinates = coordinate_sum / node_count[:, None].clamp_min(1.0)
        coordinates = coordinates * valid[:, None]
        relations = relation_sum / relation_count[..., None].clamp_min(1.0)
        relation_mask = relation_count > 0
        reliability_lambda = self.relation_reliability_lambda
        relation_support = (
            relation_count / (relation_count + reliability_lambda)
            if reliability_lambda > 0.0
            else relation_mask.to(relation_count.dtype)
        )
        relations = relations * relation_mask[..., None]
        support = node_count / node_count.max().clamp_min(1.0)
        empty = relations.new_empty(relations.shape[0], relations.shape[1], 0)
        return PopulationEncoding(
            nodes=nodes,
            relations=relations,
            geometry_relations=empty,
            activity_relations=empty,
            coordinates=coordinates,
            support=support,
            relation_support=relation_support.to(relations.dtype),
            relation_count=relation_count,
        )


def build_atlas_targets(
    sample: WormSample,
    identity_to_slot: Mapping[str, int],
    atlas_support: torch.Tensor,
) -> PairTargets:
    """Create sparse node-to-identity-slot supervision for one animal."""

    atlas_size = len(identity_to_slot)
    if atlas_support.shape != (atlas_size,):
        raise ValueError("atlas_support must contain one value per identity slot")
    row_target = torch.full(
        (sample.num_nodes,), -1, dtype=torch.long, device=sample.xyz.device
    )
    col_target = torch.full(
        (atlas_size,), -1, dtype=torch.long, device=sample.xyz.device
    )
    identity_map = unique_identity_map(sample)
    matched = 0
    for identity, node_index in identity_map.items():
        slot = identity_to_slot.get(identity)
        if slot is None or not bool(atlas_support[slot] > 0):
            continue
        row_target[node_index] = slot
        col_target[slot] = node_index
        matched += 1
    return PairTargets(
        row_target=row_target,
        col_target=col_target,
        num_direct_matches=matched,
        num_synthetic_unmatched=0,
    )


def build_open_set_episode(
    sample: WormSample,
    bank: LeaveOneAnimalOutAtlasBank,
    identity_to_slot: Mapping[str, int],
    *,
    support_drop_probability: float,
    query_drop_probability: float,
    generator: torch.Generator,
) -> tuple[WormSample, PopulationEncoding, PairTargets, dict[str, int]]:
    """Build one semantically correct query-to-support-atlas episode.

    Identities removed from every support contribution become query-side
    unknown targets.  Query neurons removed after support construction become
    atlas-side missing targets.  Naturally unobserved identities remain
    ignored because their absence is not experimentally controlled.
    """

    for name, probability in (
        ("support_drop_probability", support_drop_probability),
        ("query_drop_probability", query_drop_probability),
    ):
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"{name} must be in [0, 1)")

    identity_map = unique_identity_map(sample)
    base_atlas = bank.prototype(exclude_source=sample.source_path)
    candidates = [
        identity for identity in sorted(identity_map)
        if identity in identity_to_slot
        and bool(base_atlas.support[identity_to_slot[identity]] > 0)
    ]
    support_dropped = {
        identity for identity in candidates
        if bool(torch.rand((), generator=generator) < support_drop_probability)
    }
    remaining = [identity for identity in candidates if identity not in support_dropped]
    query_dropped = {
        identity for identity in remaining
        if bool(torch.rand((), generator=generator) < query_drop_probability)
    }
    # Keep at least one controlled direct match whenever possible.
    if candidates and len(support_dropped | query_dropped) == len(candidates):
        query_dropped.discard(candidates[0])
        support_dropped.discard(candidates[0])
    minimum_query_nodes = min(2, sample.num_nodes)
    kept_query_nodes = sample.num_nodes - len(query_dropped)
    if kept_query_nodes < minimum_query_nodes:
        for identity in candidates:
            if identity in query_dropped:
                query_dropped.remove(identity)
                kept_query_nodes += 1
                if kept_query_nodes >= minimum_query_nodes:
                    break

    drop_slots = torch.tensor(
        [identity_to_slot[identity] for identity in sorted(support_dropped)],
        dtype=torch.long,
        device=base_atlas.nodes.device,
    )
    atlas = bank.prototype(
        exclude_source=sample.source_path,
        drop_slots=drop_slots,
    )
    keep = torch.ones(sample.num_nodes, dtype=torch.bool)
    for identity in query_dropped:
        keep[identity_map[identity]] = False
    if int(keep.sum()) < min(2, sample.num_nodes):
        raise RuntimeError("Open-set augmentation removed too many query neurons")
    query = sample.subset(keep)
    query_map = unique_identity_map(query)
    atlas_size = len(identity_to_slot)
    row_target = torch.full((query.num_nodes,), -1, dtype=torch.long)
    col_target = torch.full((atlas_size,), -1, dtype=torch.long)
    direct = 0
    unmatched = 0
    for identity in candidates:
        slot = identity_to_slot[identity]
        if identity in support_dropped:
            if identity in query_map:
                row_target[query_map[identity]] = atlas_size
                unmatched += 1
        elif identity in query_dropped:
            col_target[slot] = query.num_nodes
            unmatched += 1
        elif identity in query_map and bool(atlas.support[slot] > 0):
            row_target[query_map[identity]] = slot
            col_target[slot] = query_map[identity]
            direct += 1

    return query, atlas, PairTargets(
        row_target=row_target,
        col_target=col_target,
        num_direct_matches=direct,
        num_synthetic_unmatched=unmatched,
    ), {
        "support_identities_dropped": len(support_dropped),
        "query_neurons_dropped": len(query_dropped),
        "direct_matches": direct,
        "synthetic_unmatched": unmatched,
    }
