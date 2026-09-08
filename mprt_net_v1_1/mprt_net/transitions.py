from __future__ import annotations

import torch


def normalize_partial_transition(
    transition: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Normalize a real-source to real-target-plus-dustbin transition."""

    if transition.ndim != 2 or transition.shape[1] < 2:
        raise ValueError("A partial transition must have shape [N, M + 1]")
    return transition / transition.sum(dim=1, keepdim=True).clamp_min(eps)


def compose_partial_transitions(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    """Compose two partial transitions using an absorbing dustbin state."""

    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("Partial transitions must be matrices")
    if first.shape[1] != second.shape[0] + 1:
        raise ValueError(
            f"Incompatible transitions: {tuple(first.shape)} then "
            f"{tuple(second.shape)}"
        )
    absorbing_row = torch.cat(
        [
            second.new_zeros(1, second.shape[1] - 1),
            second.new_ones(1, 1),
        ],
        dim=1,
    )
    return first @ torch.cat([second, absorbing_row], dim=0)
