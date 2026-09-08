from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .config import ModelConfig
from .data import WormSample
from .layers import PopulationRelationEncoder
from .relations import (
    ActivityNodeEncoder,
    GeometryNodeEncoder,
    MultimodalRelationBuilder,
    SymmetricResidualFusion,
    standardize_xyz,
)
from .sinkhorn import SinkhornResult, augmented_sinkhorn, contracted_relation_cost
from .transitions import compose_partial_transitions, normalize_partial_transition


def _inverse_softplus(value: float) -> float:
    if value <= 0:
        return -20.0
    return math.log(math.expm1(value))


@dataclass
class PopulationEncoding:
    nodes: torch.Tensor
    relations: torch.Tensor
    geometry_relations: torch.Tensor
    activity_relations: torch.Tensor
    coordinates: torch.Tensor | None = None
    support: torch.Tensor | None = None
    relation_support: torch.Tensor | None = None
    relation_count: torch.Tensor | None = None


@dataclass
class DynamicAtlasEncoding:
    encoding: PopulationEncoding
    coefficients: torch.Tensor
    coordinate_residual: torch.Tensor
    node_residual: torch.Tensor
    relation_residual: torch.Tensor
    attention_weights: torch.Tensor | None = None


@dataclass
class MPRTOutput:
    plan: torch.Tensor
    log_plan: torch.Tensor
    row_conditional: torch.Tensor
    col_conditional: torch.Tensor
    unary_logits: torch.Tensor
    final_logits: torch.Tensor
    relation_costs: tuple[torch.Tensor, ...]
    encoding_a: PopulationEncoding
    encoding_b: PopulationEncoding
    mu: torch.Tensor
    nu: torch.Tensor
    quadratic_objectives: tuple[torch.Tensor, ...] = ()


@dataclass
class AtlasMatchOutput:
    output: MPRTOutput
    direct_output: MPRTOutput
    alignment_a: MPRTOutput
    alignment_b: MPRTOutput
    atlas_row_conditional: torch.Tensor
    atlas_col_conditional: torch.Tensor
    dynamic_atlas_a: DynamicAtlasEncoding | None = None
    dynamic_atlas_b: DynamicAtlasEncoding | None = None


class DynamicResidualAtlas(nn.Module):
    """Bounded low-rank deformation of fixed identity-anchored atlas slots.

    ``global_pool`` is the original DeepSets conditioner and is retained for
    old-checkpoint compatibility. ``atlas_cross_attention`` makes every atlas
    slot query the input animal and combines feature attention with a soft
    standardized-coordinate prior.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        pose_input_dim = config.hidden_dim + 7
        hidden = config.dynamic_atlas_hidden_dim
        rank = config.dynamic_atlas_rank
        size = config.atlas_size
        self.conditioner = config.dynamic_atlas_conditioner

        if self.conditioner == "global_pool":
            # Exact original names and shapes preserve old checkpoints.
            self.pose_nodes = nn.Sequential(
                nn.Linear(pose_input_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.pose_coefficients = nn.Linear(2 * hidden, rank)
        elif self.conditioner == "atlas_cross_attention":
            heads = config.dynamic_atlas_attention_heads
            self.attention_heads = heads
            self.head_dim = hidden // heads
            self.atlas_queries = nn.Sequential(
                nn.Linear(pose_input_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
            self.condition_keys = nn.Sequential(
                nn.Linear(pose_input_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
            self.condition_values = nn.Sequential(
                nn.Linear(pose_input_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
            self.slot_context = nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            self.pose_coefficients = nn.Linear(hidden, rank)
        else:  # ModelConfig normally rejects this first.
            raise ValueError(f"Unknown dynamic atlas conditioner={self.conditioner!r}")
        # A zero coefficient head makes the initial dynamic model exactly the
        # frozen static atlas, while nonzero random bases preserve gradients.
        nn.init.zeros_(self.pose_coefficients.weight)
        nn.init.zeros_(self.pose_coefficients.bias)
        self.coordinate_basis = nn.Parameter(0.02 * torch.randn(rank, size, 3))
        self.node_basis = nn.Parameter(
            0.02 * torch.randn(rank, size, config.hidden_dim)
        )
        self.geometry_to_relation = nn.Sequential(
            nn.Linear(7, hidden),
            nn.SiLU(),
            nn.Linear(hidden, config.relation_dim),
        )

    @staticmethod
    def _edge_geometry(coordinates: torch.Tensor) -> torch.Tensor:
        displacement = coordinates[:, None, :] - coordinates[None, :, :]
        distance = torch.linalg.vector_norm(displacement, dim=-1, keepdim=True)
        unit = displacement / distance.clamp_min(1e-6)
        return torch.cat([displacement, unit, distance], dim=-1)

    @staticmethod
    def _node_input(encoding: PopulationEncoding) -> torch.Tensor:
        if encoding.coordinates is None:
            raise ValueError("Dynamic atlas conditioning requires node coordinates")
        coordinates = encoding.coordinates
        radius = coordinates.square().sum(dim=-1, keepdim=True).sqrt()
        return torch.cat(
            [
                F.normalize(encoding.nodes, dim=-1),
                coordinates,
                coordinates.square(),
                radius,
            ],
            dim=-1,
        )

    def _global_coefficients(
        self,
        condition: PopulationEncoding,
    ) -> tuple[torch.Tensor, None]:
        per_node = self.pose_nodes(self._node_input(condition))
        pooled = torch.cat(
            [per_node.mean(dim=0), per_node.max(dim=0).values], dim=0
        )
        return torch.tanh(self.pose_coefficients(pooled)), None

    def _slot_coefficients(
        self,
        condition: PopulationEncoding,
        atlas: PopulationEncoding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition_input = self._node_input(condition)
        atlas_input = self._node_input(atlas)
        num_slots = atlas_input.shape[0]
        num_nodes = condition_input.shape[0]
        if num_nodes == 0:
            raise ValueError("Dynamic atlas cannot attend to an empty population")

        queries = self.atlas_queries(atlas_input)
        keys = self.condition_keys(condition_input)
        values = self.condition_values(condition_input)
        queries_by_head = queries.view(
            num_slots, self.attention_heads, self.head_dim
        )
        keys_by_head = keys.view(num_nodes, self.attention_heads, self.head_dim)
        values_by_head = values.view(
            num_nodes, self.attention_heads, self.head_dim
        )
        logits = torch.einsum(
            "shd,nhd->hsn", queries_by_head, keys_by_head
        ) / math.sqrt(float(self.head_dim))

        squared_distance = torch.cdist(
            atlas.coordinates, condition.coordinates
        ).square()
        sigma = self.config.dynamic_atlas_geometry_sigma
        geometry_bias = -squared_distance / (2.0 * sigma * sigma)
        logits = logits + (
            self.config.dynamic_atlas_geometry_weight * geometry_bias[None, :, :]
        )
        attention = F.softmax(logits, dim=-1)
        dropped_attention = F.dropout(
            attention, p=self.config.dropout, training=self.training
        )
        context_by_head = torch.einsum(
            "hsn,nhd->shd", dropped_attention, values_by_head
        )
        context = context_by_head.reshape(num_slots, -1)
        slot_features = self.slot_context(
            torch.cat([queries, context, context - queries], dim=-1)
        )
        coefficients = torch.tanh(self.pose_coefficients(slot_features))
        if atlas.support is not None:
            valid = (atlas.support > 0).to(coefficients.dtype)
            coefficients = coefficients * valid[:, None]
        return coefficients, attention.mean(dim=0)

    def forward(
        self,
        condition: PopulationEncoding,
        atlas: PopulationEncoding,
    ) -> DynamicAtlasEncoding:
        if atlas.coordinates is None:
            raise ValueError("Dynamic atlas requires frozen atlas coordinates")
        if self.conditioner == "global_pool":
            coefficients, attention_weights = self._global_coefficients(condition)
        else:
            coefficients, attention_weights = self._slot_coefficients(condition, atlas)
        rank = float(self.config.dynamic_atlas_rank)
        coefficient_equation = "r,rsd->sd" if coefficients.ndim == 1 else "sr,rsd->sd"
        node_equation = "r,rsh->sh" if coefficients.ndim == 1 else "sr,rsh->sh"
        coordinate_residual = (
            self.config.dynamic_atlas_coordinate_scale
            * torch.einsum(
                coefficient_equation,
                coefficients,
                torch.tanh(self.coordinate_basis),
            )
            / rank
        )
        node_residual = (
            self.config.dynamic_atlas_node_scale
            * torch.einsum(
                node_equation,
                coefficients,
                torch.tanh(self.node_basis),
            )
            / rank
        )
        dynamic_coordinates = atlas.coordinates + coordinate_residual
        dynamic_nodes = F.normalize(atlas.nodes + node_residual, dim=-1)

        static_geometry = self._edge_geometry(atlas.coordinates)
        dynamic_geometry = self._edge_geometry(dynamic_coordinates)
        relation_residual = self.config.dynamic_atlas_relation_scale * torch.tanh(
            self.geometry_to_relation(dynamic_geometry)
            - self.geometry_to_relation(static_geometry)
        )
        if atlas.relation_support is not None:
            relation_residual = relation_residual * atlas.relation_support[..., None]
        elif atlas.support is not None:
            pair_support = (atlas.support > 0)[:, None] & (atlas.support > 0)[None, :]
            relation_residual = relation_residual * pair_support[..., None]
        dynamic_relations = atlas.relations + relation_residual
        encoding = PopulationEncoding(
            nodes=dynamic_nodes,
            relations=dynamic_relations,
            geometry_relations=dynamic_geometry,
            activity_relations=atlas.activity_relations,
            coordinates=dynamic_coordinates,
            support=atlas.support,
            relation_support=atlas.relation_support,
            relation_count=atlas.relation_count,
        )
        return DynamicAtlasEncoding(
            encoding=encoding,
            coefficients=coefficients,
            coordinate_residual=coordinate_residual,
            node_residual=node_residual,
            relation_residual=relation_residual,
            attention_weights=attention_weights,
        )

    def regularization(
        self,
        atlas: PopulationEncoding,
        dynamic: DynamicAtlasEncoding,
    ) -> dict[str, torch.Tensor]:
        if atlas.coordinates is None:
            raise ValueError("Dynamic atlas regularization requires coordinates")
        coordinate = dynamic.coordinate_residual
        magnitude = (
            coordinate.square().mean()
            + dynamic.node_residual.square().mean()
            + dynamic.relation_residual.square().mean()
        )
        size = coordinate.shape[0]
        if size < 2:
            zero = magnitude * 0.0
            return {"magnitude": magnitude, "smoothness": zero, "distortion": zero}

        k = min(self.config.dynamic_atlas_smooth_k, size - 1)
        static_distance = torch.cdist(atlas.coordinates, atlas.coordinates)
        invalid = torch.eye(size, dtype=torch.bool, device=static_distance.device)
        if atlas.support is not None:
            valid = atlas.support > 0
            invalid = invalid | ~valid[:, None] | ~valid[None, :]
        neighbor_distance = static_distance.masked_fill(invalid, torch.inf)
        neighbors = neighbor_distance.topk(k=k, dim=1, largest=False).indices
        rows = torch.arange(size, device=coordinate.device)[:, None].expand(-1, k)
        selected_valid = torch.isfinite(neighbor_distance[rows, neighbors])
        residual_difference = coordinate[rows] - coordinate[neighbors]
        squared_difference = residual_difference.square().sum(dim=-1)
        smoothness = (
            squared_difference[selected_valid].mean()
            if bool(selected_valid.any())
            else magnitude * 0.0
        )

        dynamic_distance = torch.cdist(
            dynamic.encoding.coordinates, dynamic.encoding.coordinates
        )
        edge_change = dynamic_distance[rows, neighbors] - static_distance[rows, neighbors]
        distortion = (
            edge_change[selected_valid].square().mean()
            if bool(selected_valid.any())
            else magnitude * 0.0
        )
        return {
            "magnitude": magnitude,
            "smoothness": smoothness,
            "distortion": distortion,
        }


class IndependentPopulationEncoder(nn.Module):
    """Shared encoder applied independently to each animal."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.geometry_nodes = GeometryNodeEncoder(config.hidden_dim, config.dropout)
        self.activity_nodes = ActivityNodeEncoder(
            config.hidden_dim, config.activity_channels, config.dropout
        )
        self.node_fusion = SymmetricResidualFusion(config.hidden_dim, config.dropout)
        self.relation_builder = MultimodalRelationBuilder(
            edge_dim=config.edge_dim,
            rbf_scales=config.rbf_scales,
            dropout=config.dropout,
            use_geometry=config.use_geometry,
            use_activity=config.use_activity,
        )
        self.population = PopulationRelationEncoder(
            hidden_dim=config.hidden_dim,
            edge_dim=config.edge_dim,
            relation_dim=config.relation_dim,
            num_heads=config.num_heads,
            num_layers=config.population_layers if config.use_population_encoder else 0,
            dropout=config.dropout,
        )

    def _controlled_relation_graph(
        self,
        xyz: torch.Tensor,
        multimodal: torch.Tensor,
        geometry: torch.Tensor,
        activity: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.config.relation_graph_mode
        if mode == "geometry_only":
            return geometry
        if mode == "activity_only":
            return activity
        if mode == "soft":
            return multimodal
        # Directed KNN support is deliberately binary. Self-relations remain
        # available, while exactly K non-self neighbors retain their learned
        # multimodal relation values.
        num_nodes = xyz.shape[0]
        k = min(self.config.hard_knn_k, max(num_nodes - 1, 1))
        distances = torch.cdist(xyz, xyz)
        distances.fill_diagonal_(torch.inf)
        neighbors = distances.topk(k=k, dim=1, largest=False).indices
        support = torch.zeros(
            num_nodes, num_nodes, dtype=torch.bool, device=xyz.device
        )
        support.scatter_(1, neighbors, True)
        support.fill_diagonal_(True)
        return multimodal * support[..., None].to(multimodal.dtype)

    def forward(self, sample: WormSample) -> PopulationEncoding:
        xyz = standardize_xyz(sample.xyz)
        if self.config.use_geometry:
            geometry_nodes = self.geometry_nodes(xyz)
        else:
            geometry_nodes = xyz.new_zeros(sample.num_nodes, self.config.hidden_dim)
        if self.config.use_activity:
            activity_nodes = self.activity_nodes(sample.activity)
        else:
            activity_nodes = xyz.new_zeros(sample.num_nodes, self.config.hidden_dim)

        if self.config.use_geometry and self.config.use_activity:
            nodes = self.node_fusion(geometry_nodes, activity_nodes)
        elif self.config.use_geometry:
            nodes = geometry_nodes
        else:
            nodes = activity_nodes

        relations, geometry_relations, activity_relations = self.relation_builder(
            xyz, sample.activity
        )
        relations = self._controlled_relation_graph(
            xyz,
            relations,
            geometry_relations,
            activity_relations,
        )
        if self.config.use_edge_conditioning:
            nodes, relation_field = self.population(nodes, relations)
        else:
            # The zero-relation pass is an edge-independent population
            # Transformer. A second pass supplies the same projected relation
            # field to FGW, so this ablation removes edge conditioning without
            # simultaneously deleting the relational-transport stage.
            nodes_without_edges, _ = self.population(nodes, torch.zeros_like(relations))
            _, relation_field = self.population(nodes, relations)
            nodes = nodes_without_edges
        return PopulationEncoding(
            nodes=nodes,
            relations=relation_field,
            geometry_relations=geometry_relations,
            activity_relations=activity_relations,
            coordinates=xyz,
        )


class MPRTNet(nn.Module):
    """Multimodal Population-Relational Transport Network.

    Identity is formed independently inside each animal.  Cross-animal
    interaction begins only when the unary assignment and relational transport
    are computed.
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = IndependentPopulationEncoder(self.config)

        self.raw_unary_temperature = nn.Parameter(
            torch.tensor(
                _inverse_softplus(
                    self.config.unary_temperature_init - self.config.min_temperature
                ),
                dtype=torch.float32,
            )
        )
        self.raw_relation_temperature = nn.Parameter(
            torch.tensor(
                _inverse_softplus(
                    self.config.relation_temperature_init - self.config.min_temperature
                ),
                dtype=torch.float32,
            )
        )
        self.raw_structural_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(self.config.structural_weight), dtype=torch.float32)
        )
        self.deletion_logit = nn.Parameter(
            torch.tensor(self.config.dustbin_logit_init, dtype=torch.float32)
        )
        self.insertion_logit = nn.Parameter(
            torch.tensor(self.config.dustbin_logit_init, dtype=torch.float32)
        )

        if self.config.atlas_size > 0:
            size = self.config.atlas_size
            self.register_buffer(
                "atlas_nodes", torch.zeros(size, self.config.hidden_dim)
            )
            self.register_buffer(
                "atlas_relations",
                torch.zeros(size, size, self.config.relation_dim),
            )
            self.register_buffer("atlas_support", torch.zeros(size))
            self.register_buffer(
                "atlas_initialized_flag", torch.tensor(False, dtype=torch.bool)
            )
            self.register_buffer("atlas_update_count", torch.zeros((), dtype=torch.long))
            self.register_buffer("atlas_blend", torch.zeros(()))
            if self.config.dynamic_atlas_enabled:
                self.register_buffer("atlas_xyz", torch.zeros(size, 3))
            else:
                self.atlas_xyz = None
            if (
                self.config.dynamic_atlas_enabled
                or self.config.atlas_relation_masking
                or self.config.relation_objective == "population_relative_quadratic"
            ):
                self.register_buffer(
                    "atlas_relation_support", torch.zeros(size, size)
                )
            else:
                self.atlas_relation_support = None
            if self.config.relation_objective == "population_relative_quadratic":
                self.register_buffer("atlas_relation_count", torch.zeros(size, size))
            else:
                self.atlas_relation_count = None
            if self.config.dynamic_atlas_enabled:
                self.dynamic_atlas_adapter = DynamicResidualAtlas(self.config)
            else:
                self.dynamic_atlas_adapter = None
        else:
            self.atlas_nodes = None
            self.atlas_relations = None
            self.atlas_support = None
            self.atlas_initialized_flag = None
            self.atlas_update_count = None
            self.atlas_blend = None
            self.atlas_xyz = None
            self.atlas_relation_support = None
            self.atlas_relation_count = None
            self.dynamic_atlas_adapter = None

    @property
    def unary_temperature(self) -> torch.Tensor:
        return self.config.min_temperature + F.softplus(self.raw_unary_temperature)

    @property
    def relation_temperature(self) -> torch.Tensor:
        return self.config.min_temperature + F.softplus(self.raw_relation_temperature)

    @property
    def structural_weight(self) -> torch.Tensor:
        return F.softplus(self.raw_structural_weight)

    @property
    def has_atlas(self) -> bool:
        return self.config.atlas_size > 0

    @property
    def atlas_is_initialized(self) -> bool:
        return self.has_atlas and bool(self.atlas_initialized_flag.item())

    def set_atlas_blend(self, value: float) -> None:
        if not self.has_atlas:
            if value != 0.0:
                raise ValueError("Cannot set atlas blend when atlas_size is zero")
            return
        self.atlas_blend.fill_(min(max(float(value), 0.0), 1.0))

    def atlas_encoding(self) -> PopulationEncoding:
        if not self.atlas_is_initialized:
            raise RuntimeError("The shared relational atlas has not been initialized")
        empty_relations = self.atlas_relations.new_empty(
            self.config.atlas_size, self.config.atlas_size, 0
        )
        return PopulationEncoding(
            nodes=self.atlas_nodes,
            relations=self.atlas_relations,
            geometry_relations=empty_relations,
            activity_relations=empty_relations,
            coordinates=self.atlas_xyz,
            support=self.atlas_support,
            relation_support=self.atlas_relation_support,
            relation_count=self.atlas_relation_count,
        )

    def deform_atlas(
        self,
        condition: PopulationEncoding,
        atlas: PopulationEncoding | None = None,
    ) -> DynamicAtlasEncoding:
        if not self.config.dynamic_atlas_enabled or self.dynamic_atlas_adapter is None:
            raise RuntimeError("The dynamic residual atlas is disabled")
        base = self.atlas_encoding() if atlas is None else atlas
        return self.dynamic_atlas_adapter(condition, base)

    @torch.no_grad()
    def initialize_atlas(self, encoding: PopulationEncoding) -> None:
        """Bootstrap atlas slots from one high-coverage training animal."""

        if not self.has_atlas:
            raise RuntimeError("atlas_size must be positive before initialization")
        size = self.config.atlas_size
        if encoding.nodes.shape[0] < size:
            raise ValueError(
                f"Reference population has {encoding.nodes.shape[0]} nodes, "
                f"fewer than atlas_size={size}"
            )
        self.atlas_nodes.copy_(F.normalize(encoding.nodes[:size].detach(), dim=-1))
        self.atlas_relations.copy_(encoding.relations[:size, :size].detach())
        self.atlas_support.fill_(1.0)
        if self.config.dynamic_atlas_enabled:
            if encoding.coordinates is None:
                raise ValueError("Dynamic atlas initialization requires coordinates")
            self.atlas_xyz.copy_(encoding.coordinates[:size].detach())
        if self.atlas_relation_support is not None:
            self.atlas_relation_support.fill_(1.0)
        if self.atlas_relation_count is not None:
            self.atlas_relation_count.fill_(1.0)
        self.atlas_initialized_flag.fill_(True)
        self.atlas_update_count.zero_()

    @torch.no_grad()
    def initialize_atlas_from_prototypes(
        self,
        nodes: torch.Tensor,
        relations: torch.Tensor,
        support: torch.Tensor | None = None,
        coordinates: torch.Tensor | None = None,
        relation_support: torch.Tensor | None = None,
        relation_count: torch.Tensor | None = None,
    ) -> None:
        """Initialize fixed-semantic atlas slots from training-only anchors."""

        if not self.has_atlas:
            raise RuntimeError("atlas_size must be positive before initialization")
        expected_nodes = (self.config.atlas_size, self.config.hidden_dim)
        expected_relations = (
            self.config.atlas_size,
            self.config.atlas_size,
            self.config.relation_dim,
        )
        if nodes.shape != expected_nodes:
            raise ValueError(
                f"Atlas node prototypes must have shape {expected_nodes}, "
                f"got {tuple(nodes.shape)}"
            )
        if relations.shape != expected_relations:
            raise ValueError(
                f"Atlas relation prototypes must have shape {expected_relations}, "
                f"got {tuple(relations.shape)}"
            )
        if support is None:
            support = nodes.new_ones(self.config.atlas_size)
        if support.shape != (self.config.atlas_size,):
            raise ValueError("Atlas support must have one value per slot")
        if not (
            torch.isfinite(nodes).all()
            and torch.isfinite(relations).all()
            and torch.isfinite(support).all()
        ):
            raise ValueError("Atlas prototypes and support must be finite")

        self.atlas_nodes.copy_(F.normalize(nodes.detach().to(self.atlas_nodes), dim=-1))
        self.atlas_relations.copy_(relations.detach().to(self.atlas_relations))
        self.atlas_support.copy_(support.detach().to(self.atlas_support))
        if self.config.dynamic_atlas_enabled:
            if coordinates is None or coordinates.shape != (self.config.atlas_size, 3):
                raise ValueError("Dynamic atlas coordinates must have shape [atlas_size, 3]")
            if not torch.isfinite(coordinates).all():
                raise ValueError("Dynamic atlas coordinates must be finite")
            self.atlas_xyz.copy_(coordinates.detach().to(self.atlas_xyz))
        if self.atlas_relation_support is not None:
            if relation_support is None:
                relation_support = (support > 0)[:, None] & (support > 0)[None, :]
            if relation_support.shape != (
                self.config.atlas_size,
                self.config.atlas_size,
            ):
                raise ValueError("Atlas relation support must have shape [atlas_size, atlas_size]")
            if not torch.isfinite(relation_support).all():
                raise ValueError("Atlas relation support must be finite")
            self.atlas_relation_support.copy_(
                relation_support.detach().to(self.atlas_relation_support)
            )
        if self.atlas_relation_count is not None:
            if relation_count is None:
                relation_count = (
                    relation_support > 0
                    if relation_support is not None
                    else ((support > 0)[:, None] & (support > 0)[None, :])
                ).to(support.dtype)
            expected_count = (self.config.atlas_size, self.config.atlas_size)
            if relation_count.shape != expected_count:
                raise ValueError("Atlas relation count must have shape [atlas_size, atlas_size]")
            if not torch.isfinite(relation_count).all() or bool((relation_count < 0).any()):
                raise ValueError("Atlas relation count must be finite and non-negative")
            self.atlas_relation_count.copy_(
                relation_count.detach().to(self.atlas_relation_count)
            )
        self.atlas_initialized_flag.fill_(True)
        self.atlas_update_count.zero_()

    @torch.no_grad()
    def update_atlas(
        self,
        encoding: PopulationEncoding,
        alignment: MPRTOutput,
        eps: float = 1e-8,
    ) -> None:
        """Slowly aggregate one aligned animal into node and relation prototypes."""

        if not self.atlas_is_initialized:
            raise RuntimeError("Cannot update an uninitialized atlas")
        atlas_to_worm = normalize_partial_transition(alignment.col_conditional, eps)
        real = atlas_to_worm[:, :-1]
        support = real.sum(dim=1)
        weights = real / support[:, None].clamp_min(eps)
        valid = support >= self.config.atlas_min_support

        candidate_nodes = weights @ F.normalize(encoding.nodes.detach(), dim=-1)
        node_alpha = ((1.0 - self.config.atlas_momentum) * support).clamp(0.0, 1.0)
        mixed_nodes = (
            (1.0 - node_alpha[:, None]) * self.atlas_nodes
            + node_alpha[:, None] * candidate_nodes
        )
        mixed_nodes = F.normalize(mixed_nodes, dim=-1)
        self.atlas_nodes.copy_(
            torch.where(valid[:, None], mixed_nodes, self.atlas_nodes)
        )

        relation_channels = encoding.relations.detach().permute(2, 0, 1)
        expanded_weights = weights.unsqueeze(0).expand(
            relation_channels.shape[0], -1, -1
        )
        candidate_relations = torch.bmm(
            torch.bmm(expanded_weights, relation_channels),
            expanded_weights.transpose(1, 2),
        ).permute(1, 2, 0)
        pair_support = support[:, None] * support[None, :]
        relation_alpha = (
            (1.0 - self.config.atlas_momentum) * pair_support
        ).clamp(0.0, 1.0)
        valid_pairs = valid[:, None] & valid[None, :]
        mixed_relations = (
            (1.0 - relation_alpha[..., None]) * self.atlas_relations
            + relation_alpha[..., None] * candidate_relations
        )
        self.atlas_relations.copy_(
            torch.where(valid_pairs[..., None], mixed_relations, self.atlas_relations)
        )
        self.atlas_support.mul_(self.config.atlas_momentum).add_(
            support, alpha=1.0 - self.config.atlas_momentum
        )
        self.atlas_update_count.add_(1)

    def _solve(self, logits: torch.Tensor) -> SinkhornResult:
        return augmented_sinkhorn(
            logits,
            deletion_logit=self.deletion_logit,
            insertion_logit=self.insertion_logit,
            iterations=self.config.sinkhorn_iterations,
        )

    def encode_population(self, sample: WormSample) -> PopulationEncoding:
        """Encode one animal without introducing cross-animal information."""

        return self.encoder(sample)

    def match_encodings(
        self,
        encoding_a: PopulationEncoding,
        encoding_b: PopulationEncoding,
    ) -> MPRTOutput:
        """Solve one pairwise match from independently computed encodings.

        Keeping this operation separate lets multi-animal objectives reuse each
        population encoding.  It does not add parameters or alter checkpoint
        compatibility.
        """

        nodes_a = F.normalize(encoding_a.nodes, dim=-1)
        nodes_b = F.normalize(encoding_b.nodes, dim=-1)
        unary_logits = (nodes_a @ nodes_b.transpose(0, 1)) / self.unary_temperature
        valid_pairs = None
        if encoding_a.support is not None or encoding_b.support is not None:
            valid_a = (
                encoding_a.support > 0
                if encoding_a.support is not None
                else torch.ones(nodes_a.shape[0], dtype=torch.bool, device=nodes_a.device)
            )
            valid_b = (
                encoding_b.support > 0
                if encoding_b.support is not None
                else torch.ones(nodes_b.shape[0], dtype=torch.bool, device=nodes_b.device)
            )
            valid_pairs = valid_a[:, None] & valid_b[None, :]
            unary_logits = unary_logits.masked_fill(~valid_pairs, -1.0e4)

        final_logits = unary_logits
        solved = self._solve(final_logits)
        node_augmented_logits = solved.augmented_logits
        relation_costs: list[torch.Tensor] = []
        quadratic_objectives: list[torch.Tensor] = []
        if self.config.use_relation_transport:
            for _ in range(self.config.transport_steps):
                relation_cost = contracted_relation_cost(
                    encoding_a.relations,
                    encoding_b.relations,
                    solved.plan[:-1, :-1],
                    normalize_plan=(
                        self.config.relation_objective
                        == "legacy_normalized_directed"
                    ),
                    symmetric_directed=(
                        self.config.relation_objective in {
                            "raw_symmetric",
                            "population_relative_quadratic",
                        }
                    ),
                    support_a=(
                        encoding_a.relation_support
                        if self.config.atlas_relation_masking
                        or self.config.relation_objective
                        == "population_relative_quadratic"
                        else None
                    ),
                    support_b=(
                        encoding_b.relation_support
                        if self.config.atlas_relation_masking
                        or self.config.relation_objective
                        == "population_relative_quadratic"
                        else None
                    ),
                )
                relation_costs.append(relation_cost)
                if self.config.relation_objective == "population_relative_quadratic":
                    real_plan = solved.plan[:-1, :-1]
                    node_term = -(node_augmented_logits * solved.plan).sum()
                    relation_term = (
                        0.5
                        * self.structural_weight
                        / self.relation_temperature
                        * (real_plan * relation_cost).sum()
                    )
                    entropy = (
                        solved.plan
                        * (solved.log_plan - 1.0)
                    ).sum()
                    quadratic_objectives.append(node_term + relation_term + entropy)
                structural_logits = -relation_cost / self.relation_temperature
                final_logits = unary_logits + self.structural_weight * structural_logits
                if valid_pairs is not None:
                    final_logits = final_logits.masked_fill(~valid_pairs, -1.0e4)
                solved = self._solve(final_logits)

            if self.config.relation_objective == "population_relative_quadratic":
                final_cost = contracted_relation_cost(
                    encoding_a.relations,
                    encoding_b.relations,
                    solved.plan[:-1, :-1],
                    normalize_plan=False,
                    symmetric_directed=True,
                    support_a=encoding_a.relation_support,
                    support_b=encoding_b.relation_support,
                )
                real_plan = solved.plan[:-1, :-1]
                quadratic_objectives.append(
                    -(node_augmented_logits * solved.plan).sum()
                    + 0.5
                    * self.structural_weight
                    / self.relation_temperature
                    * (real_plan * final_cost).sum()
                    + (solved.plan * (solved.log_plan - 1.0)).sum()
                )

        num_a, num_b = unary_logits.shape
        row_conditional = solved.plan[:num_a, :] / solved.mu[:num_a, None]
        col_conditional = (
            solved.plan[:, :num_b] / solved.nu[None, :num_b]
        ).transpose(0, 1)
        return MPRTOutput(
            plan=solved.plan,
            log_plan=solved.log_plan,
            row_conditional=row_conditional,
            col_conditional=col_conditional,
            unary_logits=unary_logits,
            final_logits=final_logits,
            relation_costs=tuple(relation_costs),
            encoding_a=encoding_a,
            encoding_b=encoding_b,
            mu=solved.mu,
            nu=solved.nu,
            quadratic_objectives=tuple(quadratic_objectives),
        )

    @staticmethod
    def _atlas_induced_transitions(
        alignment_a: MPRTOutput,
        alignment_b: MPRTOutput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_to_atlas = normalize_partial_transition(alignment_a.row_conditional)
        atlas_to_a = normalize_partial_transition(alignment_a.col_conditional)
        b_to_atlas = normalize_partial_transition(alignment_b.row_conditional)
        atlas_to_b = normalize_partial_transition(alignment_b.col_conditional)
        row = compose_partial_transitions(a_to_atlas, atlas_to_b)
        column = compose_partial_transitions(b_to_atlas, atlas_to_a)
        return row, column

    def _fuse_atlas_output(
        self,
        direct: MPRTOutput,
        atlas_row: torch.Tensor,
        atlas_column: torch.Tensor,
        blend: torch.Tensor,
    ) -> MPRTOutput:
        if self.config.atlas_confidence_gating:
            row_advantage = (
                self._transition_confidence(atlas_row)
                - self._transition_confidence(direct.row_conditional)
            ) / self.config.atlas_gate_temperature
            column_advantage = (
                self._transition_confidence(atlas_column)
                - self._transition_confidence(direct.col_conditional)
            ) / self.config.atlas_gate_temperature
            row_gate = (blend * torch.sigmoid(row_advantage)).detach()[:, None]
            column_gate = (blend * torch.sigmoid(column_advantage)).detach()[:, None]
        else:
            row_gate = blend
            column_gate = blend
        row = (1.0 - row_gate) * direct.row_conditional + row_gate * atlas_row
        column = (
            (1.0 - column_gate) * direct.col_conditional
            + column_gate * atlas_column
        )
        row = normalize_partial_transition(row)
        column = normalize_partial_transition(column)

        # Ranking uses the two directed probabilities; Hungarian decoding uses
        # their symmetric real-node score.  Other transport diagnostics remain
        # those of the direct relational solve.
        real_score = 0.5 * (row[:, :-1] + column[:, :-1].transpose(0, 1))
        top = torch.cat([real_score, row[:, -1:]], dim=1)
        plan = torch.cat([top, direct.plan[-1:, :]], dim=0)
        return MPRTOutput(
            plan=plan,
            log_plan=plan.clamp_min(1e-12).log(),
            row_conditional=row,
            col_conditional=column,
            unary_logits=direct.unary_logits,
            final_logits=direct.final_logits,
            relation_costs=direct.relation_costs,
            encoding_a=direct.encoding_a,
            encoding_b=direct.encoding_b,
            mu=direct.mu,
            nu=direct.nu,
            quadratic_objectives=direct.quadratic_objectives,
        )

    @staticmethod
    def _transition_confidence(probabilities: torch.Tensor) -> torch.Tensor:
        """Non-dustbin Top-1 margin used for conservative atlas correction."""

        real = probabilities[:, :-1]
        if real.shape[1] == 1:
            margin = real[:, 0]
        else:
            top_two = torch.topk(real, k=2, dim=1).values
            margin = top_two[:, 0] - top_two[:, 1]
        survival = 1.0 - probabilities[:, -1]
        return margin.clamp_min(0.0) * survival.clamp(0.0, 1.0)

    def match_with_atlas_encodings(
        self,
        encoding_a: PopulationEncoding,
        encoding_b: PopulationEncoding,
        blend: float | torch.Tensor | None = None,
    ) -> AtlasMatchOutput:
        if not self.atlas_is_initialized:
            raise RuntimeError("The shared relational atlas has not been initialized")
        direct = self.match_encodings(encoding_a, encoding_b)
        atlas = self.atlas_encoding()
        dynamic_a = self.deform_atlas(encoding_a, atlas) if self.config.dynamic_atlas_enabled else None
        dynamic_b = self.deform_atlas(encoding_b, atlas) if self.config.dynamic_atlas_enabled else None
        atlas_a = dynamic_a.encoding if dynamic_a is not None else atlas
        atlas_b = dynamic_b.encoding if dynamic_b is not None else atlas
        alignment_a = self.match_encodings(encoding_a, atlas_a)
        alignment_b = self.match_encodings(encoding_b, atlas_b)
        atlas_row, atlas_column = self._atlas_induced_transitions(
            alignment_a, alignment_b
        )
        if blend is None:
            blend_tensor = self.atlas_blend.to(direct.plan)
        else:
            blend_tensor = direct.plan.new_tensor(blend)
        fused = self._fuse_atlas_output(
            direct,
            atlas_row,
            atlas_column,
            blend_tensor.clamp(0.0, 1.0),
        )
        return AtlasMatchOutput(
            output=fused,
            direct_output=direct,
            alignment_a=alignment_a,
            alignment_b=alignment_b,
            atlas_row_conditional=atlas_row,
            atlas_col_conditional=atlas_column,
            dynamic_atlas_a=dynamic_a,
            dynamic_atlas_b=dynamic_b,
        )

    def forward(self, sample_a: WormSample, sample_b: WormSample) -> MPRTOutput:
        encoding_a = self.encode_population(sample_a)
        encoding_b = self.encode_population(sample_b)
        if self.atlas_is_initialized and float(self.atlas_blend) > 0.0:
            return self.match_with_atlas_encodings(encoding_a, encoding_b).output
        return self.match_encodings(encoding_a, encoding_b)
