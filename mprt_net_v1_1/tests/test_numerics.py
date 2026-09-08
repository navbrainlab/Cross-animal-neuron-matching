import torch

from mprt_net.sinkhorn import (
    augmented_sinkhorn,
    brute_force_relation_cost,
    contracted_relation_cost,
)


def test_augmented_sinkhorn_marginals():
    torch.manual_seed(1)
    result = augmented_sinkhorn(
        torch.randn(4, 6), torch.tensor(-1.0), torch.tensor(-0.5), iterations=60
    )
    torch.testing.assert_close(result.plan.sum(1), result.mu, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(result.plan.sum(0), result.nu, atol=2e-5, rtol=2e-5)


def test_contracted_relation_cost_matches_brute_force():
    torch.manual_seed(2)
    relation_a = torch.randn(3, 3, 4)
    relation_b = torch.randn(5, 5, 4)
    plan = torch.rand(3, 5)
    expected = brute_force_relation_cost(relation_a, relation_b, plan)
    actual = contracted_relation_cost(relation_a, relation_b, plan)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_population_relative_cost_is_symmetric_directed_unnormalized_and_masked():
    torch.manual_seed(22)
    relation_a = torch.randn(3, 3, 2)
    relation_b = torch.randn(4, 4, 2)
    plan = torch.rand(3, 4)
    support_a = torch.rand(3, 3)
    support_b = torch.rand(4, 4)
    expected = brute_force_relation_cost(
        relation_a,
        relation_b,
        plan,
        normalize_plan=False,
        symmetric_directed=True,
        support_a=support_a,
        support_b=support_b,
    )
    actual = contracted_relation_cost(
        relation_a,
        relation_b,
        plan,
        normalize_plan=False,
        symmetric_directed=True,
        support_a=support_a,
        support_b=support_b,
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    half_mass = contracted_relation_cost(
        relation_a,
        relation_b,
        0.5 * plan,
        normalize_plan=False,
        symmetric_directed=True,
        support_a=support_a,
        support_b=support_b,
    )
    torch.testing.assert_close(half_mass, 0.5 * actual, atol=2e-5, rtol=2e-5)
