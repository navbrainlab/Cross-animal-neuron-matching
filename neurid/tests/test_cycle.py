from types import SimpleNamespace

import torch

from mprt_net.config import ModelConfig
from mprt_net.data import WormSample
from mprt_net.losses import (
    compose_partial_transitions,
    six_way_cycle_consistency_loss,
)
from mprt_net.model import MPRTNet
from mprt_net.train import cycle_weight_for_epoch


def _identity_output(size: int):
    transition = torch.cat([torch.eye(size), torch.zeros(size, 1)], dim=1)
    return SimpleNamespace(
        row_conditional=transition,
        col_conditional=transition.clone(),
    )


def _sample(uid: str, size: int) -> WormSample:
    return WormSample(
        uid=uid,
        xyz=torch.randn(size, 3),
        activity=torch.randn(size, 32),
        cell_ids=tuple(f"N{i}" for i in range(size)),
        supervised_mask=torch.ones(size, dtype=torch.bool),
        source_path=f"{uid}.npz",
    )


def test_dustbin_is_absorbing_during_composition():
    # The only source node is rejected on the first hop.  Even though the
    # second match maps its real source to a real target, no mass may reappear.
    first = torch.tensor([[0.0, 1.0]])
    second = torch.tensor([[1.0, 0.0]])
    composed = compose_partial_transitions(first, second)
    torch.testing.assert_close(composed, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(composed.sum(dim=1), torch.ones(1))


def test_consistent_identity_matches_have_zero_six_way_cycle_loss():
    output_ab = _identity_output(3)
    output_bc = _identity_output(3)
    output_ac = _identity_output(3)
    loss = six_way_cycle_consistency_loss(output_ab, output_bc, output_ac)
    torch.testing.assert_close(loss.total, torch.tensor(0.0), atol=1e-7, rtol=0.0)
    assert len(loss.directions) == 6


def test_inconsistent_direct_match_has_positive_cycle_loss_and_gradients():
    output_ab = _identity_output(2)
    output_bc = _identity_output(2)
    logits = torch.tensor(
        [[-2.0, 2.0, -4.0], [2.0, -2.0, -4.0]], requires_grad=True
    )
    swapped = torch.softmax(logits, dim=1)
    output_ac = SimpleNamespace(
        row_conditional=swapped,
        col_conditional=swapped,
    )
    loss = six_way_cycle_consistency_loss(output_ab, output_bc, output_ac).total
    assert float(loss.detach()) > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_three_population_model_cycle_backward():
    torch.manual_seed(42)
    model = MPRTNet(
        ModelConfig(
            hidden_dim=32,
            edge_dim=16,
            relation_dim=4,
            activity_channels=8,
            num_heads=4,
            population_layers=1,
            transport_steps=1,
            sinkhorn_iterations=20,
            dropout=0.0,
        )
    )
    encoding_a = model.encode_population(_sample("a", 4))
    encoding_b = model.encode_population(_sample("b", 5))
    encoding_c = model.encode_population(_sample("c", 6))
    output_ab = model.match_encodings(encoding_a, encoding_b)
    output_bc = model.match_encodings(encoding_b, encoding_c)
    output_ac = model.match_encodings(encoding_a, encoding_c)
    loss = six_way_cycle_consistency_loss(output_ab, output_bc, output_ac).total
    assert torch.isfinite(loss)
    loss.backward()
    assert model.raw_relation_temperature.grad is not None
    assert torch.isfinite(model.raw_relation_temperature.grad)


def test_cycle_weight_schedule():
    assert cycle_weight_for_epoch(0.02, 5, 5, 10) == 0.0
    assert cycle_weight_for_epoch(0.02, 6, 5, 10) == 0.002
    assert cycle_weight_for_epoch(0.02, 15, 5, 10) == 0.02
    assert cycle_weight_for_epoch(0.02, 20, 5, 10) == 0.02
