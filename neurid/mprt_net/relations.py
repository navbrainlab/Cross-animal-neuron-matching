from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def standardize_xyz(xyz: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-animal translation and anisotropic scale normalization."""

    center = xyz.median(dim=0).values
    centered = xyz - center
    scale = centered.square().mean(dim=0).sqrt().clamp_min(eps)
    return centered / scale


def standardize_activity(activity: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = activity.mean(dim=-1, keepdim=True)
    std = activity.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (activity - mean) / std


def pairwise_activity_features(activity: torch.Tensor) -> torch.Tensor:
    """Identity-free functional population relations.

    Correlation is computed only within an animal.  Recordings need not be
    temporally aligned across animals.
    """

    z = standardize_activity(activity)
    t = max(int(z.shape[-1]), 1)
    corr = (z @ z.transpose(0, 1)) / float(t)
    corr = corr.clamp(-1.0, 1.0)

    if z.shape[-1] > 1:
        dz = z[:, 1:] - z[:, :-1]
        dz = standardize_activity(dz)
        dcorr = (dz @ dz.transpose(0, 1)) / float(dz.shape[-1])
        dcorr = dcorr.clamp(-1.0, 1.0)
    else:
        dcorr = torch.zeros_like(corr)

    return torch.stack(
        [corr, dcorr, corr.clamp_min(0.0), (-corr).clamp_min(0.0)], dim=-1
    )


class GeometryNodeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        radius = xyz_normalized.norm(dim=-1, keepdim=True)
        return self.network(torch.cat([xyz_normalized, radius], dim=-1))


class ActivityNodeEncoder(nn.Module):
    def __init__(self, hidden_dim: int, channels: int, dropout: float):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.stats = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * channels + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, activity: torch.Tensor) -> torch.Tensor:
        mean = activity.mean(dim=-1)
        std = activity.std(dim=-1, unbiased=False).clamp_min(1e-5)
        q25, q50, q75 = torch.quantile(
            activity, torch.tensor([0.25, 0.50, 0.75], device=activity.device), dim=-1
        ).unbind(dim=0)
        stats = torch.stack([mean, std, q25, q50, q75], dim=-1)

        z = standardize_activity(activity)
        derivative = F.pad(z[:, 1:] - z[:, :-1], (1, 0))
        temporal = self.temporal(torch.stack([z, derivative], dim=1))
        pooled = torch.cat([temporal.mean(dim=-1), temporal.amax(dim=-1)], dim=-1)
        return self.output(torch.cat([pooled, self.stats(stats)], dim=-1))


class SymmetricResidualFusion(nn.Module):
    """Learned symmetric two-modality fusion with an explicit interaction residual."""

    def __init__(self, dimension: int, dropout: float):
        super().__init__()
        self.weight = nn.Sequential(
            nn.Linear(3 * dimension, dimension),
            nn.GELU(),
            nn.Linear(dimension, 2),
        )
        self.residual = nn.Sequential(
            nn.Linear(3 * dimension, 2 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension, dimension),
        )
        # Start close to a stable convex combination.
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, geometry: torch.Tensor, activity: torch.Tensor) -> torch.Tensor:
        joint = torch.cat([geometry, activity, geometry * activity], dim=-1)
        weights = self.weight(joint).softmax(dim=-1)
        base = weights[..., :1] * geometry + weights[..., 1:] * activity
        return self.norm(base + self.residual(joint))


class MultimodalRelationBuilder(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        rbf_scales: tuple[float, ...],
        dropout: float,
        use_geometry: bool,
        use_activity: bool,
    ):
        super().__init__()
        self.use_geometry = use_geometry
        self.use_activity = use_activity
        self.register_buffer("rbf_scales", torch.tensor(rbf_scales, dtype=torch.float32))
        geometry_input_dim = 4 + len(rbf_scales)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(geometry_input_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )
        self.activity_encoder = nn.Sequential(
            nn.Linear(4, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
            nn.LayerNorm(edge_dim),
        )
        self.fusion = SymmetricResidualFusion(edge_dim, dropout)

    def geometry_features(self, xyz_normalized: torch.Tensor) -> torch.Tensor:
        delta = xyz_normalized[:, None, :] - xyz_normalized[None, :, :]
        distance = delta.square().sum(dim=-1, keepdim=True).sqrt()
        scales = self.rbf_scales.to(dtype=xyz_normalized.dtype).view(1, 1, -1)
        rbf = torch.exp(-0.5 * distance.square() / scales.square().clamp_min(1e-6))
        return torch.cat([delta, distance, rbf], dim=-1)

    def forward(
        self, xyz_normalized: torch.Tensor, activity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n = xyz_normalized.shape[0]
        shape = (n, n, self.geometry_encoder[-1].normalized_shape[0])

        if self.use_geometry:
            geometry = self.geometry_encoder(self.geometry_features(xyz_normalized))
        else:
            geometry = xyz_normalized.new_zeros(shape)

        if self.use_activity:
            activity_edge = self.activity_encoder(pairwise_activity_features(activity))
        else:
            activity_edge = xyz_normalized.new_zeros(shape)

        if self.use_geometry and self.use_activity:
            fused = self.fusion(geometry, activity_edge)
        elif self.use_geometry:
            fused = geometry
        else:
            fused = activity_edge
        return fused, geometry, activity_edge
