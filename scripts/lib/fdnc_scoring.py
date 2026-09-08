"""Minimal fDNC pair scoring shared by benchmark evaluators."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def point_features(model: Any, xyz: Tensor) -> Tensor:
    """Encode one point cloud with the official fDNC point feature network."""
    return model.point_f(xyz.T.unsqueeze(0)).transpose(1, 2).squeeze(0)


def directional_logits(model: Any, reference: Tensor, target: Tensor) -> Tensor:
    """Return target-to-reference logits plus the official outlier column."""
    n_reference = reference.shape[0]
    sequence = torch.cat([reference + 1.0, target], dim=0)
    encoded = model.model(sequence.unsqueeze(1)).squeeze(1)
    reference_embedding = encoded[:n_reference]
    target_embedding = encoded[n_reference:]
    similarity = target_embedding @ reference_embedding.T
    return torch.cat([similarity, model.fc_outlier(target_embedding)], dim=1)


def score_pair(model: Any, a_xyz: Tensor, b_xyz: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``b -> a`` and ``a -> b`` directional score matrices."""
    a_features = point_features(model, a_xyz)
    b_features = point_features(model, b_xyz)
    return (
        directional_logits(model, a_features, b_features),
        directional_logits(model, b_features, a_features),
    )
