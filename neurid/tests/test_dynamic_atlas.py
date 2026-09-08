from pathlib import Path

import pytest
import torch

from mprt_net.config import ModelConfig
from mprt_net.data import WormSample
from mprt_net.dynamic_atlas import (
    AtlasContribution,
    LeaveOneAnimalOutAtlasBank,
    build_atlas_targets,
    build_open_set_episode,
    canonical_source,
)
from mprt_net.model import MPRTNet


def _sample(uid: str, size: int = 5) -> WormSample:
    return WormSample(
        uid=uid,
        xyz=torch.randn(size, 3),
        activity=torch.randn(size, 32),
        cell_ids=tuple(f"N{i}" for i in range(size)),
        supervised_mask=torch.ones(size, dtype=torch.bool),
        source_path=f"{uid}.npz",
    )


def _config(conditioner: str = "atlas_cross_attention") -> ModelConfig:
    return ModelConfig(
        hidden_dim=32,
        edge_dim=16,
        relation_dim=4,
        activity_channels=8,
        num_heads=4,
        population_layers=1,
        transport_steps=1,
        sinkhorn_iterations=20,
        dropout=0.0,
        atlas_size=5,
        atlas_blend_weight=1.0,
        dynamic_atlas_enabled=True,
        dynamic_atlas_rank=4,
        dynamic_atlas_hidden_dim=16,
        dynamic_atlas_coordinate_scale=0.15,
        dynamic_atlas_conditioner=conditioner,
    )


def test_dynamic_atlas_is_static_at_initialization_and_has_gradients():
    torch.manual_seed(301)
    config = _config()
    model = MPRTNet(config)
    nodes = torch.randn(config.atlas_size, config.hidden_dim)
    relations = torch.randn(
        config.atlas_size, config.atlas_size, config.relation_dim
    )
    coordinates = torch.randn(config.atlas_size, 3)
    support = torch.ones(config.atlas_size)
    relation_support = torch.ones(config.atlas_size, config.atlas_size)
    model.initialize_atlas_from_prototypes(
        nodes,
        relations,
        support,
        coordinates=coordinates,
        relation_support=relation_support,
    )
    condition = model.encode_population(_sample("condition"))
    static = model.atlas_encoding()
    dynamic = model.deform_atlas(condition, static)

    torch.testing.assert_close(dynamic.coefficients, torch.zeros(5, 4))
    torch.testing.assert_close(
        dynamic.attention_weights.sum(dim=-1), torch.ones(5)
    )
    torch.testing.assert_close(dynamic.encoding.nodes, static.nodes)
    torch.testing.assert_close(dynamic.encoding.coordinates, static.coordinates)
    torch.testing.assert_close(dynamic.encoding.relations, static.relations)

    probe = torch.randn_like(dynamic.encoding.nodes)
    loss = (dynamic.encoding.nodes * probe).sum()
    loss.backward()
    gradient = model.dynamic_atlas_adapter.pose_coefficients.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert float(gradient.abs().sum()) > 0.0


def test_dynamic_residual_is_bounded_and_forward_is_finite():
    torch.manual_seed(302)
    config = _config()
    model = MPRTNet(config)
    model.initialize_atlas_from_prototypes(
        torch.randn(config.atlas_size, config.hidden_dim),
        torch.randn(config.atlas_size, config.atlas_size, config.relation_dim),
        torch.ones(config.atlas_size),
        coordinates=torch.randn(config.atlas_size, 3),
        relation_support=torch.ones(config.atlas_size, config.atlas_size),
    )
    with torch.no_grad():
        model.dynamic_atlas_adapter.pose_coefficients.bias.fill_(4.0)
    condition = model.encode_population(_sample("condition"))
    dynamic = model.deform_atlas(condition)
    assert float(dynamic.coordinate_residual.abs().max()) <= (
        config.dynamic_atlas_coordinate_scale + 1e-6
    )
    model.set_atlas_blend(1.0)
    output = model(_sample("a", 4), _sample("b", 6))
    assert output.plan.shape == (5, 7)
    assert torch.isfinite(output.plan).all()


def test_global_pool_conditioner_keeps_original_coefficient_shape():
    torch.manual_seed(303)
    config = _config("global_pool")
    model = MPRTNet(config)
    model.initialize_atlas_from_prototypes(
        torch.randn(config.atlas_size, config.hidden_dim),
        torch.randn(config.atlas_size, config.atlas_size, config.relation_dim),
        torch.ones(config.atlas_size),
        coordinates=torch.randn(config.atlas_size, 3),
        relation_support=torch.ones(config.atlas_size, config.atlas_size),
    )
    dynamic = model.deform_atlas(model.encode_population(_sample("global")))
    assert dynamic.coefficients.shape == (config.dynamic_atlas_rank,)
    assert dynamic.attention_weights is None


def test_leave_one_animal_out_prototype_removes_its_contribution(tmp_path: Path):
    device = torch.device("cpu")
    source_a = canonical_source(tmp_path / "a.npz")
    source_b = canonical_source(tmp_path / "b.npz")
    slots = torch.tensor([0, 1])
    nodes_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    nodes_b = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    coordinates_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    coordinates_b = 2.0 * coordinates_a
    relations_a = torch.ones(2, 2, 1)
    relations_b = 3.0 * torch.ones(2, 2, 1)
    bank = LeaveOneAnimalOutAtlasBank(
        identity_to_slot={"N0": 0, "N1": 1},
        node_sum=nodes_a + nodes_b,
        coordinate_sum=coordinates_a + coordinates_b,
        node_count=2.0 * torch.ones(2, device=device),
        relation_sum=relations_a + relations_b,
        relation_count=2.0 * torch.ones(2, 2, device=device),
        contributions={
            source_a: AtlasContribution(
                slots=slots,
                nodes=nodes_a,
                coordinates=coordinates_a,
                relations=relations_a,
            ),
            source_b: AtlasContribution(
                slots=slots,
                nodes=nodes_b,
                coordinates=coordinates_b,
                relations=relations_b,
            ),
        },
    )
    excluded = bank.prototype(exclude_source=source_a)
    torch.testing.assert_close(excluded.nodes, nodes_b)
    torch.testing.assert_close(excluded.coordinates, coordinates_b)
    torch.testing.assert_close(excluded.relations, relations_b)
    torch.testing.assert_close(excluded.support, torch.ones(2))


def test_atlas_targets_mask_unsupported_slots():
    sample = _sample("targets", 4)
    targets = build_atlas_targets(
        sample,
        {"N0": 0, "N1": 1, "N2": 2, "N3": 3},
        torch.tensor([1.0, 0.0, 1.0, 1.0]),
    )
    assert targets.num_direct_matches == 3
    assert targets.row_target.tolist() == [0, -1, 2, 3]
    assert targets.col_target.tolist() == [0, -1, 2, 3]


def _two_source_bank(tmp_path: Path, reliability_lambda: float = 1.0):
    source_a = canonical_source(tmp_path / "a.npz")
    source_b = canonical_source(tmp_path / "b.npz")
    slots = torch.arange(5)
    nodes_a = torch.eye(5)
    nodes_b = torch.eye(5)
    coordinates = torch.randn(5, 3)
    relations = torch.ones(5, 5, 1)
    return LeaveOneAnimalOutAtlasBank(
        identity_to_slot={f"N{i}": i for i in range(5)},
        node_sum=nodes_a + nodes_b,
        coordinate_sum=2.0 * coordinates,
        node_count=2.0 * torch.ones(5),
        relation_sum=2.0 * relations,
        relation_count=2.0 * torch.ones(5, 5),
        contributions={
            source_a: AtlasContribution(slots, nodes_a, coordinates, relations),
            source_b: AtlasContribution(slots, nodes_b, coordinates, relations),
        },
        relation_reliability_lambda=reliability_lambda,
    )


def test_relation_support_uses_count_reliability_and_drop_mask(tmp_path: Path):
    bank = _two_source_bank(tmp_path)
    full = bank.prototype()
    torch.testing.assert_close(full.relation_support, torch.full((5, 5), 2.0 / 3.0))
    torch.testing.assert_close(full.relation_count, 2.0 * torch.ones(5, 5))
    dropped = bank.prototype(drop_slots=torch.tensor([2]))
    assert not bool(dropped.relation_support[2].any())
    assert not bool(dropped.relation_support[:, 2].any())


def test_open_set_episode_has_semantic_query_unknown_targets(tmp_path: Path):
    bank = _two_source_bank(tmp_path)
    query = _sample("a")
    query.source_path = str(tmp_path / "a.npz")
    kept, atlas, targets, audit = build_open_set_episode(
        query,
        bank,
        {f"N{i}": i for i in range(5)},
        support_drop_probability=0.999999,
        query_drop_probability=0.0,
        generator=torch.Generator().manual_seed(0),
    )
    assert kept.num_nodes == query.num_nodes
    assert targets.num_direct_matches >= 1
    assert targets.num_synthetic_unmatched >= 1
    assert int((targets.row_target == atlas.nodes.shape[0]).sum()) == audit[
        "support_identities_dropped"
    ]


def test_open_set_episode_has_semantic_atlas_missing_targets(tmp_path: Path):
    bank = _two_source_bank(tmp_path)
    query = _sample("a")
    query.source_path = str(tmp_path / "a.npz")
    kept, atlas, targets, audit = build_open_set_episode(
        query,
        bank,
        {f"N{i}": i for i in range(5)},
        support_drop_probability=0.0,
        query_drop_probability=0.999999,
        generator=torch.Generator().manual_seed(0),
    )
    assert kept.num_nodes < query.num_nodes
    assert int((targets.col_target == kept.num_nodes).sum()) == audit[
        "query_neurons_dropped"
    ]


def test_dynamic_atlas_requires_identity_slots():
    with pytest.raises(ValueError):
        ModelConfig(dynamic_atlas_enabled=True, atlas_size=0)
