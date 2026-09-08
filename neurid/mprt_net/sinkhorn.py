from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SinkhornResult:
    plan: torch.Tensor
    log_plan: torch.Tensor
    augmented_logits: torch.Tensor
    mu: torch.Tensor
    nu: torch.Tensor


def log_sinkhorn(
    logits: torch.Tensor,
    log_mu: torch.Tensor,
    log_nu: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Log-domain Sinkhorn with prescribed row and column marginals."""

    if logits.shape != (log_mu.numel(), log_nu.numel()):
        raise ValueError(
            f"Shape mismatch: logits={tuple(logits.shape)}, "
            f"mu={tuple(log_mu.shape)}, nu={tuple(log_nu.shape)}"
        )
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(logits + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(logits + u.unsqueeze(1), dim=0)
    return logits + u.unsqueeze(1) + v.unsqueeze(0)


def augmented_sinkhorn(
    real_logits: torch.Tensor,
    deletion_logit: torch.Tensor,
    insertion_logit: torch.Tensor,
    iterations: int = 20,
) -> SinkhornResult:
    """Capacity-correct partial matching with one dustbin row and column.

    A single dustbin cell can absorb multiple nodes because its marginal mass
    equals the number of nodes on the opposite side, following the augmented
    optimal-transport construction used for partial bipartite matching.
    """

    if real_logits.ndim != 2:
        raise ValueError("real_logits must be a matrix")
    num_a, num_b = real_logits.shape
    if num_a < 1 or num_b < 1:
        raise ValueError("Both populations must contain at least one node")

    deletion = deletion_logit.to(real_logits).expand(num_a, 1)
    insertion = insertion_logit.to(real_logits).expand(1, num_b)
    corner = real_logits.new_zeros(1, 1)
    augmented = torch.cat(
        [
            torch.cat([real_logits, deletion], dim=1),
            torch.cat([insertion, corner], dim=1),
        ],
        dim=0,
    )

    normalizer = float(num_a + num_b)
    mu = real_logits.new_ones(num_a + 1)
    nu = real_logits.new_ones(num_b + 1)
    mu[-1] = float(num_b)
    nu[-1] = float(num_a)
    mu = mu / normalizer
    nu = nu / normalizer

    log_plan = log_sinkhorn(augmented, mu.log(), nu.log(), iterations)
    return SinkhornResult(
        plan=log_plan.exp(),
        log_plan=log_plan,
        augmented_logits=augmented,
        mu=mu,
        nu=nu,
    )


def contracted_relation_cost(
    relation_a: torch.Tensor,
    relation_b: torch.Tensor,
    real_plan: torch.Tensor,
    eps: float = 1e-8,
    *,
    normalize_plan: bool = True,
    symmetric_directed: bool = False,
    support_a: torch.Tensor | None = None,
    support_b: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Compute the exact plan-conditioned squared relation discrepancy.

    .. math::

        D_{ij}(P)=\sum_{kl}P_{kl}C_{ij,kl}.

    In legacy mode, ``P`` is normalized to ``bar P`` and only outgoing
    directed relations are compared.  ``symmetric_directed=True`` uses

    .. math::

        C_{ij,kl}=\tfrac12(\|U^A_{ik}-U^B_{jl}\|^2
        +\|U^A_{ki}-U^B_{lj}\|^2),

    which preserves direction while satisfying ``C_ij,kl = C_kl,ij``.

    The squared distance is expanded into two norm terms and a bilinear
    contraction.  This is O(d N^3) compute and O(d N^2) memory; it never
    materializes an O(N^4) tensor.
    """

    if relation_a.ndim != 3 or relation_b.ndim != 3:
        raise ValueError("Relations must have shape [N, N, D]")
    num_a, num_a_2, dimension = relation_a.shape
    num_b, num_b_2, dimension_b = relation_b.shape
    if num_a != num_a_2 or num_b != num_b_2 or dimension != dimension_b:
        raise ValueError("Relation fields must be square and share feature dimension")
    if real_plan.shape != (num_a, num_b):
        raise ValueError(
            f"Plan shape {tuple(real_plan.shape)} does not match {(num_a, num_b)}"
        )

    plan = real_plan
    if normalize_plan:
        plan = plan / plan.sum().clamp_min(eps)

    # Preserve the original implementation exactly for all legacy checkpoints.
    if not symmetric_directed and support_a is None and support_b is None:
        row_mass = plan.sum(dim=1)
        col_mass = plan.sum(dim=0)
        first = relation_a.square().sum(dim=-1) @ row_mass
        second = relation_b.square().sum(dim=-1) @ col_mass
        a_channels = relation_a.permute(2, 0, 1)
        b_channels_t = relation_b.permute(2, 1, 0)
        plan_channels = plan.unsqueeze(0).expand(dimension, -1, -1)
        cross = torch.bmm(
            torch.bmm(a_channels, plan_channels), b_channels_t
        ).sum(dim=0)
        return (first[:, None] + second[None, :] - 2.0 * cross).clamp_min(0.0)

    def directional(
        rel_a: torch.Tensor,
        rel_b: torch.Tensor,
        mask_a: torch.Tensor | None,
        mask_b: torch.Tensor | None,
    ) -> torch.Tensor:
        sa = (
            torch.ones(rel_a.shape[:2], dtype=rel_a.dtype, device=rel_a.device)
            if mask_a is None else mask_a.to(rel_a)
        )
        sb = (
            torch.ones(rel_b.shape[:2], dtype=rel_b.dtype, device=rel_b.device)
            if mask_b is None else mask_b.to(rel_b)
        )
        if sa.shape != rel_a.shape[:2] or sb.shape != rel_b.shape[:2]:
            raise ValueError("Relation support must match the corresponding relation matrix")
        a_norm = rel_a.square().sum(dim=-1) * sa
        b_norm = rel_b.square().sum(dim=-1) * sb
        first = (a_norm @ plan) @ sb.transpose(0, 1)
        second = (sa @ plan) @ b_norm.transpose(0, 1)
        cross = rel_a.new_zeros(rel_a.shape[0], rel_b.shape[0])
        for channel in range(dimension):
            weighted_a = rel_a[..., channel] * sa
            weighted_b = rel_b[..., channel] * sb
            cross = cross + (weighted_a @ plan) @ weighted_b.transpose(0, 1)
        return (first + second - 2.0 * cross).clamp_min(0.0)

    outgoing = directional(relation_a, relation_b, support_a, support_b)
    if not symmetric_directed:
        return outgoing
    incoming = directional(
        relation_a.transpose(0, 1),
        relation_b.transpose(0, 1),
        None if support_a is None else support_a.transpose(0, 1),
        None if support_b is None else support_b.transpose(0, 1),
    )
    return 0.5 * (outgoing + incoming)


def brute_force_relation_cost(
    relation_a: torch.Tensor,
    relation_b: torch.Tensor,
    real_plan: torch.Tensor,
    *,
    normalize_plan: bool = True,
    symmetric_directed: bool = False,
    support_a: torch.Tensor | None = None,
    support_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """Small-tensor reference implementation used only by tests."""

    plan = real_plan / real_plan.sum().clamp_min(1e-8) if normalize_plan else real_plan
    num_a, num_b = real_plan.shape
    output = real_plan.new_zeros(num_a, num_b)
    for i in range(num_a):
        for j in range(num_b):
            total = real_plan.new_zeros(())
            for k in range(num_a):
                for l in range(num_b):
                    outgoing_weight = plan[k, l]
                    if support_a is not None:
                        outgoing_weight = outgoing_weight * support_a[i, k]
                    if support_b is not None:
                        outgoing_weight = outgoing_weight * support_b[j, l]
                    outgoing = (relation_a[i, k] - relation_b[j, l]).square().sum()
                    if symmetric_directed:
                        incoming_weight = plan[k, l]
                        if support_a is not None:
                            incoming_weight = incoming_weight * support_a[k, i]
                        if support_b is not None:
                            incoming_weight = incoming_weight * support_b[l, j]
                        incoming = (relation_a[k, i] - relation_b[l, j]).square().sum()
                        total = total + 0.5 * (
                            outgoing_weight * outgoing + incoming_weight * incoming
                        )
                    else:
                        total = total + outgoing_weight * outgoing
            output[i, j] = total
    return output
