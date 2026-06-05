from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import Decoder


class GhostVLAD(nn.Module):
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
        self.dim = dim
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
        vlad = vlad[:, : self.num_clusters, :]
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.reshape(vlad.size(0), -1)
        vlad = F.normalize(vlad, p=2, dim=1)
        return vlad


class GhostVLADDecoder(Decoder):
    def __init__(
        self,
        d_model: int,
        ghostvlad_clusters: int,
        ghostvlad_ghost_clusters: int,
    ) -> None:
        super().__init__()
        self.decoder_type = "ghostvlad"
        self.decoder = GhostVLAD(
            num_clusters=ghostvlad_clusters,
            ghost_clusters=ghostvlad_ghost_clusters,
            dim=d_model,
        )
        self.cls_dim = ghostvlad_clusters * d_model

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings of shape (B, N, D), got {embeddings.shape}")
        cls_embed = self.decoder(embeddings)
        decoded = embeddings
        return cls_embed, decoded

    def get_embeddings(
        self,
        cls_embed: torch.Tensor,
        decoded: torch.Tensor,
        feature_type: str,
        include_queries: bool,
    ) -> torch.Tensor:
        if feature_type == "query":
            raise ValueError("feature_type='query' is not supported for ghostvlad")
        return cls_embed
