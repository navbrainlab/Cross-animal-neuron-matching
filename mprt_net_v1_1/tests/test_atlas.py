import torch

from mprt_net.config import ModelConfig
from mprt_net.data import WormSample, build_pair_targets
from mprt_net.losses import (
    symmetric_focal_matching_loss,
    symmetric_focal_probability_loss,
)
from mprt_net.model import MPRTNet


def _sample(uid: str, size: int) -> WormSample:
    return WormSample(
        uid=uid,
        xyz=torch.randn(size, 3),
        activity=torch.randn(size, 32),
        cell_ids=tuple(f"N{i}" for i in range(size)),
        supervised_mask=torch.ones(size, dtype=torch.bool),
        source_path=f"{uid}.npz",
    )


def _config() -> ModelConfig:
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
        atlas_momentum=0.99,
        atlas_min_support=0.0,
        atlas_blend_weight=0.25,
    )


def test_ema_atlas_forward_backward_update_and_state_roundtrip():
    torch.manual_seed(42)
    model = MPRTNet(_config())
    assert model.has_atlas
    assert not model.atlas_is_initialized
    assert "atlas_nodes" not in dict(model.named_parameters())

    reference_encoding = model.encode_population(_sample("reference", 6))
    model.initialize_atlas(reference_encoding)
    model.set_atlas_blend(0.25)
    assert model.atlas_is_initialized

    sample_a, sample_b, targets = build_pair_targets(
        _sample("a", 4), _sample("b", 5)
    )
    encoding_a = model.encode_population(sample_a)
    encoding_b = model.encode_population(sample_b)
    atlas_match = model.match_with_atlas_encodings(encoding_a, encoding_b)

    assert atlas_match.atlas_row_conditional.shape == (4, 6)
    assert atlas_match.atlas_col_conditional.shape == (5, 5)
    torch.testing.assert_close(
        atlas_match.atlas_row_conditional.sum(dim=1),
        torch.ones(4),
        atol=2e-5,
        rtol=2e-5,
    )
    torch.testing.assert_close(
        atlas_match.atlas_col_conditional.sum(dim=1),
        torch.ones(5),
        atol=2e-5,
        rtol=2e-5,
    )

    pair_loss = symmetric_focal_matching_loss(atlas_match.output, targets).total
    atlas_loss = symmetric_focal_probability_loss(
        atlas_match.atlas_row_conditional,
        atlas_match.atlas_col_conditional,
        targets,
    ).total
    objective = pair_loss + 0.1 * atlas_loss
    objective.backward()
    assert torch.isfinite(objective)
    assert model.raw_relation_temperature.grad is not None

    previous_nodes = model.atlas_nodes.clone()
    model.update_atlas(encoding_a, atlas_match.alignment_a)
    model.update_atlas(encoding_b, atlas_match.alignment_b)
    assert int(model.atlas_update_count) == 2
    assert torch.isfinite(model.atlas_nodes).all()
    assert torch.isfinite(model.atlas_relations).all()
    assert not torch.equal(previous_nodes, model.atlas_nodes)

    restored = MPRTNet(_config())
    restored.load_state_dict(model.state_dict())
    assert restored.atlas_is_initialized
    torch.testing.assert_close(restored.atlas_nodes, model.atlas_nodes)
    torch.testing.assert_close(restored.atlas_relations, model.atlas_relations)
    torch.testing.assert_close(restored.atlas_blend, model.atlas_blend)
    restored_output = restored(sample_a, sample_b)
    assert restored_output.plan.shape == (5, 6)
    assert torch.isfinite(restored_output.plan).all()


def test_atlas_disabled_model_keeps_legacy_state_shape():
    model = MPRTNet(ModelConfig())
    assert not model.has_atlas
    assert not model.atlas_is_initialized
    assert not any(key.startswith("atlas_") for key in model.state_dict())


def test_anchored_prototype_initialization_and_confidence_gating():
    config = _config()
    config.atlas_confidence_gating = True
    config.atlas_gate_temperature = 0.05
    model = MPRTNet(config)
    nodes = torch.randn(config.atlas_size, config.hidden_dim)
    relations = torch.randn(
        config.atlas_size,
        config.atlas_size,
        config.relation_dim,
    )
    support = torch.linspace(0.2, 1.0, config.atlas_size)
    model.initialize_atlas_from_prototypes(nodes, relations, support)
    model.set_atlas_blend(0.3)
    assert model.atlas_is_initialized
    torch.testing.assert_close(model.atlas_support, support)
    torch.testing.assert_close(
        model.atlas_nodes.norm(dim=1),
        torch.ones(config.atlas_size),
        atol=1e-6,
        rtol=1e-6,
    )

    uncertain = torch.tensor([[0.45, 0.40, 0.15]])
    confident = torch.tensor([[0.90, 0.05, 0.05]])
    uncertain_score = model._transition_confidence(uncertain)
    confident_score = model._transition_confidence(confident)
    assert float(confident_score) > float(uncertain_score)

    output = model(_sample("anchored-a", 4), _sample("anchored-b", 5))
    assert output.plan.shape == (5, 6)
    assert torch.isfinite(output.plan).all()
