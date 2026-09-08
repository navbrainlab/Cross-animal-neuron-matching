from __future__ import annotations

import torch
from torch import nn


class RelationAttentionLayer(nn.Module):
    """Self-attention confined to one animal and conditioned on relation edges."""

    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.edge_bias = nn.Linear(edge_dim, num_heads, bias=False)
        self.edge_value = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, nodes: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        n = nodes.shape[0]
        normalized = self.pre_norm(nodes)
        q, k, v = self.qkv(normalized).chunk(3, dim=-1)
        q = q.view(n, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(n, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(n, self.num_heads, self.head_dim).transpose(0, 1)

        logits = torch.einsum("hid,hjd->hij", q, k) * self.scale
        logits = logits + self.edge_bias(relations).permute(2, 0, 1)
        attention = logits.softmax(dim=-1)
        attention = self.dropout(attention)

        edge_values = self.edge_value(relations)
        edge_values = edge_values.view(n, n, self.num_heads, self.head_dim)
        edge_values = edge_values.permute(2, 0, 1, 3)
        messages = v[:, None, :, :] + edge_values
        update = (attention[..., None] * messages).sum(dim=2)
        update = update.transpose(0, 1).reshape(n, -1)
        nodes = nodes + self.dropout(self.output(update))
        nodes = nodes + self.feed_forward(self.ff_norm(nodes))
        return nodes


class PopulationRelationEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        relation_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelationAttentionLayer(hidden_dim, edge_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.relation_projector = nn.Sequential(
            nn.Linear(edge_dim + 2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, relation_dim),
            nn.Tanh(),
        )

    def forward(
        self, nodes: torch.Tensor, relations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            nodes = layer(nodes, relations)
        nodes = self.node_norm(nodes)

        difference = nodes[:, None, :] - nodes[None, :, :]
        product = nodes[:, None, :] * nodes[None, :, :]
        relation_field = self.relation_projector(
            torch.cat([relations, difference, product], dim=-1)
        )
        return nodes, relation_field
