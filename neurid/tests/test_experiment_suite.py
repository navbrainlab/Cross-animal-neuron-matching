import torch

from mprt_net.experiments.complexity import explicit_relation_cost
from mprt_net.experiments.dustbin_robustness import solve_transport
from mprt_net.sinkhorn import contracted_relation_cost


def test_all_transport_controls_satisfy_their_marginals():
    torch.manual_seed(201)
    logits = torch.randn(4, 6)
    for mode in ("ordinary", "uniform_dustbin", "capacity_dustbin"):
        result = solve_transport(
            logits, torch.tensor(-1.0), torch.tensor(-0.5), 80, mode
        )
        torch.testing.assert_close(
            result.plan.sum(dim=1), result.mu, atol=3e-5, rtol=3e-5
        )
        torch.testing.assert_close(
            result.plan.sum(dim=0), result.nu, atol=3e-5, rtol=3e-5
        )


def test_materialized_and_contracted_costs_agree():
    torch.manual_seed(202)
    relation_a = torch.randn(4, 4, 3)
    relation_b = torch.randn(5, 5, 3)
    plan = torch.rand(4, 5)
    expected = explicit_relation_cost(relation_a, relation_b, plan)
    actual = contracted_relation_cost(relation_a, relation_b, plan)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
