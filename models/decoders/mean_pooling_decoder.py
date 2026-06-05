from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .base import Decoder


class MeanPoolingDecoder(Decoder):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.decoder_type = "mean_pooling"
        self.cls_dim = d_model

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings of shape (B, N, D), got {embeddings.shape}")
        cls_embed = embeddings.mean(dim=1)
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
            raise ValueError("feature_type='query' is not supported for mean_pooling")
        if include_queries:
            query_embed = decoded.reshape(decoded.size(0), -1)
            return torch.cat([cls_embed, query_embed], dim=1)
        return cls_embed
