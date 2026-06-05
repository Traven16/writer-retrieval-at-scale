from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .base import Decoder


class XVLAD(nn.Module):
    def __init__(
        self,
        num_clusters: int,
        ghost_clusters: int,
        dim: int,
        nhead: int = 1,
        d_head: int = 0,
        no_intra_norm: bool = False,
    ) -> None:
        super().__init__()
        if num_clusters <= 0:
            raise ValueError("num_clusters must be positive")
        if ghost_clusters < 0:
            raise ValueError("ghost_clusters must be non-negative")
        if nhead <= 0:
            raise ValueError("nhead must be positive")
        if d_head > 0:
            if nhead * d_head != dim:
                raise ValueError(
                    f"nhead*d_head must equal dim ({nhead}*{d_head} != {dim})"
                )
            head_dim = d_head
        else:
            if dim % nhead != 0:
                raise ValueError(f"dim ({dim}) must be divisible by nhead ({nhead})")
            head_dim = dim // nhead
        self.num_clusters = num_clusters
        self.ghost_clusters = ghost_clusters
        self.dim = dim
        self.nhead = nhead
        self.head_dim = head_dim
        self.no_intra_norm = no_intra_norm
        total_clusters = num_clusters + ghost_clusters

        self.head_projections = nn.ModuleList(
            [nn.Linear(dim, self.head_dim) for _ in range(nhead)]
        )
        self.assignment_heads = nn.ModuleList(
            [nn.Linear(self.head_dim, total_clusters) for _ in range(nhead)]
        )
        self.last_head_pooled_keep: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected (B, N, D), got {x.shape}")

        pooled_heads = []
        for head_index in range(self.nhead):
            projected = self.head_projections[head_index](x)  # (B, N, d_head)
            assignments = self.assignment_heads[head_index](projected)
            assignments = F.softmax(assignments, dim=-1)  # (B, N, C+G)
            weighted = assignments.unsqueeze(-1) * projected.unsqueeze(2)
            pooled = weighted.sum(dim=1)  # (B, C+G, d_head)
            pooled = pooled[:, : self.num_clusters, :]  # drop ghost clusters
            pooled_heads.append(pooled)

        self.last_head_pooled_keep = torch.stack(pooled_heads, dim=1)  # (B, H, C, d_head)
        pooled_all = torch.cat(pooled_heads, dim=2)  # (B, C, D)
        if not self.no_intra_norm:
            pooled_all = F.normalize(pooled_all, p=2, dim=2)
        pooled_all = pooled_all.reshape(pooled_all.size(0), -1)
        pooled_all = F.normalize(pooled_all, p=2, dim=1)
        return pooled_all


class XVLADDecoder(Decoder):
    def __init__(
        self,
        d_model: int,
        ghostvlad_clusters: int,
        ghostvlad_ghost_clusters: int,
        nhead: int = 1,
        d_head: int = 0,
        xvlad_no_intra_norm: bool = False,
    ) -> None:
        super().__init__()
        self.decoder_type = "xvlad"
        self.decoder = XVLAD(
            num_clusters=ghostvlad_clusters,
            ghost_clusters=ghostvlad_ghost_clusters,
            dim=d_model,
            nhead=nhead,
            d_head=d_head,
            no_intra_norm=xvlad_no_intra_norm,
        )
        self.cls_dim = ghostvlad_clusters * d_model
        self.last_head_pooled_keep: Optional[torch.Tensor] = None

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings of shape (B, N, D), got {embeddings.shape}")
        cls_embed = self.decoder(embeddings)
        self.last_head_pooled_keep = self.decoder.last_head_pooled_keep
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
            raise ValueError("feature_type='query' is not supported for xvlad")
        return cls_embed
