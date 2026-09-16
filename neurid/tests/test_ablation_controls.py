import pytest
import torch

from mprt_net.config import ModelConfig
from mprt_net.data import WormSample
from mprt_net.model import MPRTNet


def sample(uid: str, n: int = 7) -> WormSample:
    return WormSample(
        uid=uid,
        xyz=torch.randn(n, 3),
        activity=torch.randn(n, 48),
        cell_ids=tuple(f"N{i}" for i in range(n)),
        supervised_mask=torch.ones(n, dtype=torch.bool),
        source_path=f"{uid}.npz",
    )


@pytest.mark.parametrize(
    "values",
    [
        {"relation_objective": "raw_directed"},
        {"relation_objective": "raw_symmetric"},
        {"relation_graph_mode": "hard_knn", "hard_knn_k": 3},
        {"relation_graph_mode": "geometry_only"},
        {"relation_graph_mode": "activity_only"},
        {"use_edge_conditioning": False},
    ],
)
def test_controlled_relation_variants_are_finite(values):
    torch.manual_seed(101)
    config = ModelConfig(
        hidden_dim=32,
        edge_dim=16,
        relation_dim=4,
        activity_channels=8,
        num_heads=4,
        population_layers=1,
        transport_steps=1,
        sinkhorn_iterations=20,
        dropout=0.0,
        **values,
    )
    output = MPRTNet(config)(sample("a"), sample("b", 8))
    assert output.plan.shape == (8, 9)
    assert torch.isfinite(output.plan).all()


def test_no_population_keeps_relation_transport_active():
    """The (0, 1) ablation removes population attention, not relation transport."""

    torch.manual_seed(102)
    config = ModelConfig(
        hidden_dim=32,
        edge_dim=16,
        relation_dim=4,
        activity_channels=8,
        num_heads=4,
        population_layers=2,
        transport_steps=2,
        sinkhorn_iterations=20,
        dropout=0.0,
        use_population_encoder=False,
        use_relation_transport=True,
    )
    model = MPRTNet(config)
    output = model(sample("no-pop-a"), sample("no-pop-b", 8))

    assert len(model.encoder.population.layers) == 0
    assert output.encoding_a.relations.shape == (7, 7, config.relation_dim)
    assert output.encoding_b.relations.shape == (8, 8, config.relation_dim)
    assert len(output.relation_costs) == config.transport_steps
    assert all(torch.isfinite(cost).all() for cost in output.relation_costs)
    assert not torch.equal(output.final_logits, output.unary_logits)


def test_invalid_relation_graph_mode_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(relation_graph_mode="not-a-mode")


def test_atlas_mask_ablation_registers_relation_support():
    model = MPRTNet(ModelConfig(atlas_size=5, atlas_relation_masking=True))
    assert model.atlas_relation_support is not None
