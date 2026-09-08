import torch

from mprt_net.config import ModelConfig
from mprt_net.data import WormSample, build_pair_targets
from mprt_net.losses import symmetric_focal_matching_loss
from mprt_net.model import MPRTNet


def sample(uid: str, n: int, t: int, shift: int = 0) -> WormSample:
    ids = tuple(f"N{i + shift}" for i in range(n))
    return WormSample(
        uid=uid,
        xyz=torch.randn(n, 3),
        activity=torch.randn(n, t),
        cell_ids=ids,
        supervised_mask=torch.ones(n, dtype=torch.bool),
        source_path=f"{uid}.npz",
    )


def test_forward_backward_and_shapes():
    torch.manual_seed(3)
    a, b, targets = build_pair_targets(sample("a", 7, 64), sample("b", 8, 64))
    config = ModelConfig(
        hidden_dim=32,
        edge_dim=16,
        relation_dim=4,
        activity_channels=8,
        num_heads=4,
        population_layers=1,
        transport_steps=1,
        sinkhorn_iterations=30,
        dropout=0.0,
    )
    model = MPRTNet(config)
    output = model(a, b)
    assert output.plan.shape == (8, 9)
    assert torch.isfinite(output.plan).all()
    loss = symmetric_focal_matching_loss(output, targets).total
    loss.backward()
    assert torch.isfinite(loss)
    assert model.raw_relation_temperature.grad is not None


def test_synthetic_dropout_only_labels_known_absences():
    torch.manual_seed(4)
    generator = torch.Generator().manual_seed(4)
    a, b, targets = build_pair_targets(
        sample("a", 10, 32),
        sample("b", 10, 32),
        synthetic_drop_probability=0.25,
        generator=generator,
    )
    assert targets.num_direct_matches + targets.num_synthetic_unmatched <= 10
    assert ((targets.row_target == b.num_nodes) | (targets.row_target < b.num_nodes)).all()
    assert ((targets.col_target == a.num_nodes) | (targets.col_target < a.num_nodes)).all()


def test_population_relative_match_reports_quadratic_objective_trace():
    torch.manual_seed(41)
    config = ModelConfig(
        hidden_dim=32,
        edge_dim=16,
        relation_dim=4,
        activity_channels=8,
        num_heads=4,
        population_layers=1,
        transport_steps=2,
        sinkhorn_iterations=30,
        dropout=0.0,
        relation_objective="population_relative_quadratic",
    )
    output = MPRTNet(config)(sample("qa", 6, 32), sample("qb", 7, 32))
    assert len(output.quadratic_objectives) == config.transport_steps + 1
    assert all(torch.isfinite(value) for value in output.quadratic_objectives)
