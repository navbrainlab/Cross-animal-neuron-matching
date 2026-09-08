from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for one NeuRID model.

    The defaults are deliberately moderate for 70--160 neurons per animal.
    They are starting values, not claimed optima.
    """

    hidden_dim: int = 96
    edge_dim: int = 48
    relation_dim: int = 8
    activity_channels: int = 32
    num_heads: int = 4
    population_layers: int = 2
    dropout: float = 0.10
    rbf_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)

    sinkhorn_iterations: int = 20
    transport_steps: int = 2
    structural_weight: float = 1.0
    unary_temperature_init: float = 0.10
    relation_temperature_init: float = 0.50
    min_temperature: float = 0.01
    dustbin_logit_init: float = -1.0

    # Optional cross-animal latent relational atlas.  Atlas tensors are EMA
    # buffers rather than freely optimized parameters.
    atlas_size: int = 0
    atlas_momentum: float = 0.999
    atlas_min_support: float = 0.05
    atlas_blend_weight: float = 0.0
    atlas_confidence_gating: bool = False
    atlas_gate_temperature: float = 0.05

    # Optional input-conditioned, low-rank deformation of a frozen anchored
    # atlas.  The static identity slots remain fixed; only this bounded
    # residual adapter is optimized during stage-two training.
    dynamic_atlas_enabled: bool = False
    dynamic_atlas_rank: int = 8
    dynamic_atlas_hidden_dim: int = 64
    dynamic_atlas_coordinate_scale: float = 0.15
    dynamic_atlas_node_scale: float = 0.10
    dynamic_atlas_relation_scale: float = 0.10
    dynamic_atlas_smooth_k: int = 8
    dynamic_atlas_conditioner: str = "global_pool"
    dynamic_atlas_attention_heads: int = 4
    dynamic_atlas_geometry_sigma: float = 1.0
    dynamic_atlas_geometry_weight: float = 1.0

    use_geometry: bool = True
    use_activity: bool = True
    use_population_encoder: bool = True
    use_relation_transport: bool = True
    # The relation objective modes form a cumulative ablation ladder:
    # legacy normalized/directed -> raw mass/directed -> raw mass/symmetric.
    # ``population_relative_quadratic`` additionally enables objective tracing
    # and is the named full formulation.
    relation_objective: str = "legacy_normalized_directed"
    atlas_relation_masking: bool = False
    atlas_relation_reliability_lambda: float = 0.0
    # Controlled relation-graph ablations. ``hard_knn`` keeps the learned
    # relation values only on directed KNN support; geometry/activity modes
    # retain multimodal node features while ablating one relation modality.
    relation_graph_mode: str = "soft"
    hard_knn_k: int = 16
    use_edge_conditioning: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not self.use_geometry and not self.use_activity:
            raise ValueError("At least one modality must be enabled")
        if self.transport_steps < 0:
            raise ValueError("transport_steps must be non-negative")
        if self.atlas_size < 0:
            raise ValueError("atlas_size must be non-negative")
        if not 0.0 <= self.atlas_momentum < 1.0:
            raise ValueError("atlas_momentum must be in [0, 1)")
        if not 0.0 <= self.atlas_min_support <= 1.0:
            raise ValueError("atlas_min_support must be in [0, 1]")
        if not 0.0 <= self.atlas_blend_weight <= 1.0:
            raise ValueError("atlas_blend_weight must be in [0, 1]")
        if self.atlas_gate_temperature <= 0.0:
            raise ValueError("atlas_gate_temperature must be positive")
        if self.dynamic_atlas_enabled and self.atlas_size <= 0:
            raise ValueError("dynamic_atlas_enabled requires atlas_size > 0")
        if self.dynamic_atlas_rank < 1:
            raise ValueError("dynamic_atlas_rank must be positive")
        if self.dynamic_atlas_hidden_dim < 1:
            raise ValueError("dynamic_atlas_hidden_dim must be positive")
        if self.dynamic_atlas_coordinate_scale < 0.0:
            raise ValueError("dynamic_atlas_coordinate_scale must be non-negative")
        if self.dynamic_atlas_node_scale < 0.0:
            raise ValueError("dynamic_atlas_node_scale must be non-negative")
        if self.dynamic_atlas_relation_scale < 0.0:
            raise ValueError("dynamic_atlas_relation_scale must be non-negative")
        if self.dynamic_atlas_smooth_k < 1:
            raise ValueError("dynamic_atlas_smooth_k must be positive")
        if self.dynamic_atlas_conditioner not in {
            "global_pool",
            "atlas_cross_attention",
        }:
            raise ValueError(
                "dynamic_atlas_conditioner must be 'global_pool' or "
                "'atlas_cross_attention'"
            )
        if self.dynamic_atlas_conditioner == "atlas_cross_attention":
            if self.dynamic_atlas_attention_heads < 1:
                raise ValueError("dynamic_atlas_attention_heads must be positive")
            if (
                self.dynamic_atlas_hidden_dim
                % self.dynamic_atlas_attention_heads
                != 0
            ):
                raise ValueError(
                    "dynamic_atlas_hidden_dim must be divisible by "
                    "dynamic_atlas_attention_heads"
                )
        if self.dynamic_atlas_geometry_sigma <= 0.0:
            raise ValueError("dynamic_atlas_geometry_sigma must be positive")
        if self.dynamic_atlas_geometry_weight < 0.0:
            raise ValueError("dynamic_atlas_geometry_weight must be non-negative")
        if self.relation_graph_mode not in {
            "soft",
            "hard_knn",
            "geometry_only",
            "activity_only",
        }:
            raise ValueError(f"Unknown relation_graph_mode={self.relation_graph_mode!r}")
        if self.hard_knn_k < 1:
            raise ValueError("hard_knn_k must be positive")
        if self.relation_objective not in {
            "legacy_normalized_directed",
            "raw_directed",
            "raw_symmetric",
            "population_relative_quadratic",
        }:
            raise ValueError(f"Unknown relation_objective={self.relation_objective!r}")
        if self.atlas_relation_reliability_lambda < 0.0:
            raise ValueError("atlas_relation_reliability_lambda must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        values = dict(values)
        if "rbf_scales" in values:
            values["rbf_scales"] = tuple(values["rbf_scales"])
        return cls(**values)
