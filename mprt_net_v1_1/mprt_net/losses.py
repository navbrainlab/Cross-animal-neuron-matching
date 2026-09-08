from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import PairTargets
from .model import MPRTOutput
from .transitions import compose_partial_transitions, normalize_partial_transition


@dataclass
class LossBreakdown:
    total: torch.Tensor
    row: torch.Tensor
    column: torch.Tensor
    num_queries: int


@dataclass
class CycleLossBreakdown:
    total: torch.Tensor
    directions: tuple[torch.Tensor, ...]


def _categorical_focal(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    eps: float,
) -> tuple[torch.Tensor | None, int]:
    valid = targets >= 0
    count = int(valid.sum())
    if count == 0:
        return None, 0
    selected = probabilities[valid]
    target = targets[valid]
    p = selected.gather(1, target[:, None]).squeeze(1).clamp(min=eps, max=1.0)
    return -((1.0 - p).pow(gamma) * p.log()), count


def symmetric_focal_matching_loss(
    output: MPRTOutput,
    targets: PairTargets,
    gamma: float = 2.0,
    eps: float = 1e-8,
) -> LossBreakdown:
    """One final structured matching objective, symmetric in both animals.

    No geometry-only or activity-only auxiliary labels are used.  Gradients
    reach the multimodal relation field through the final transport plan.
    Set ``gamma=0`` for ordinary categorical cross-entropy.
    """

    return symmetric_focal_probability_loss(
        output.row_conditional,
        output.col_conditional,
        targets,
        gamma=gamma,
        eps=eps,
    )


def symmetric_focal_probability_loss(
    row_probabilities: torch.Tensor,
    col_probabilities: torch.Tensor,
    targets: PairTargets,
    gamma: float = 2.0,
    eps: float = 1e-8,
) -> LossBreakdown:
    """Apply the matching loss to any pair of directed partial transitions."""

    row_values, num_row = _categorical_focal(
        row_probabilities, targets.row_target, gamma, eps
    )
    col_values, num_col = _categorical_focal(
        col_probabilities, targets.col_target, gamma, eps
    )
    pieces = [piece for piece in (row_values, col_values) if piece is not None]
    if not pieces:
        raise ValueError("The pair contains no supervised row or column queries")
    total = torch.cat(pieces).mean()
    zero = total.detach() * 0.0
    row = row_values.mean() if row_values is not None else zero
    column = col_values.mean() if col_values is not None else zero
    return LossBreakdown(
        total=total,
        row=row,
        column=column,
        num_queries=num_row + num_col,
    )


def _directional_transition(
    output: MPRTOutput,
    reverse: bool,
    eps: float,
) -> torch.Tensor:
    transition = output.col_conditional if reverse else output.row_conditional
    return normalize_partial_transition(transition, eps)


def _directional_cycle_loss(
    first: torch.Tensor,
    second: torch.Tensor,
    direct: torch.Tensor,
) -> torch.Tensor:
    composed = compose_partial_transitions(first, second)
    if composed.shape != direct.shape:
        raise ValueError(
            f"Composed transition {tuple(composed.shape)} does not match "
            f"direct transition {tuple(direct.shape)}"
        )
    # Frobenius-squared discrepancy normalized by the number of source nodes.
    # Unlike element-wise MSE, its scale does not vanish as candidate count grows.
    return (composed - direct).square().sum(dim=1).mean()


def six_way_cycle_consistency_loss(
    output_ab: MPRTOutput,
    output_bc: MPRTOutput,
    output_ac: MPRTOutput,
    eps: float = 1e-8,
) -> CycleLossBreakdown:
    """Cycle consistency for all six directed paths through three animals.

    The three pairwise solves provide A<->B, B<->C, and A<->C transitions.
    Both orientations are used, and every two-hop path is compared with its
    corresponding direct match.
    """

    ab = _directional_transition(output_ab, reverse=False, eps=eps)
    ba = _directional_transition(output_ab, reverse=True, eps=eps)
    bc = _directional_transition(output_bc, reverse=False, eps=eps)
    cb = _directional_transition(output_bc, reverse=True, eps=eps)
    ac = _directional_transition(output_ac, reverse=False, eps=eps)
    ca = _directional_transition(output_ac, reverse=True, eps=eps)

    directions = (
        _directional_cycle_loss(ab, bc, ac),
        _directional_cycle_loss(ac, cb, ab),
        _directional_cycle_loss(ba, ac, bc),
        _directional_cycle_loss(bc, ca, ba),
        _directional_cycle_loss(ca, ab, cb),
        _directional_cycle_loss(cb, ba, ca),
    )
    return CycleLossBreakdown(total=torch.stack(directions).mean(), directions=directions)
