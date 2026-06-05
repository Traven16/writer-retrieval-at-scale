from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import Decoder


class NetVLAD(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        ghost_clusters: int,
        dim: int,
    ) -> None:
        super().__init__()
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if ghost_clusters < 0:
            raise ValueError("ghost_clusters must be non-negative")
        self.num_clusters = num_clusters
        self.ghost_clusters = ghost_clusters
        total_clusters = num_clusters + ghost_clusters
        self.assignment = nn.Linear(dim, total_clusters)
        self.centroids = nn.Parameter(torch.randn(total_clusters, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, N, D), got {x.shape}")
        assignments = self.assignment(x)
        assignments = F.softmax(assignments, dim=-1)
        residuals = x.unsqueeze(2) - self.centroids.unsqueeze(0).unsqueeze(0)
        weighted = assignments.unsqueeze(-1) * residuals
        vlad = weighted.sum(dim=1)
        #vlad = vlad[:, :self.num_clusters, :]
        return vlad


class NetVLADTransformerDecoder(Decoder):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_clusters: int,
        ghost_clusters: int,
    ) -> None:
        super().__init__()
        self.decoder_type = "netvlad_transformer"
        self.netvlad = NetVLAD(
            num_clusters=num_clusters+ghost_clusters,
            ghost_clusters=0,
            dim=d_model,
        )
        self.ghost_clusters = ghost_clusters
        self.num_clusters = num_clusters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_clusters + ghost_clusters, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.cls_dim = d_model

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vlad_tokens = self.netvlad(embeddings)
        tokens = vlad_tokens + self.pos_embed
        decoded = self.transformer(tokens)
        out = decoded[:, :-self.ghost_clusters, :] if self.ghost_clusters > 0 else decoded
        
        out = out.reshape(out.size(0), -1)
        out = F.normalize(out, p=2, dim=1)
        return out, decoded

    def get_embeddings(
        self,
        cls_embed: torch.Tensor,
        decoded: torch.Tensor,
        feature_type: str,
        include_queries: bool,
    ) -> torch.Tensor:
        if feature_type == "query":
            raise ValueError("feature_type='query' is not supported for netvlad_transformer")
        if include_queries:
            raise ValueError("include_queries=True is not supported for netvlad_transformer")
        return cls_embed
