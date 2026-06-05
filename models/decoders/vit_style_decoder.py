from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from .base import Decoder


class ViTStyleDecoder(Decoder):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.decoder_type = "vit_style"
        self.cls_dim = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings of shape (B, N, D), got {embeddings.shape}")
        batch_size = embeddings.size(0)
        cls_tok = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tok, embeddings], dim=1)
        decoded = self.decoder(tokens)
        cls_embed = decoded[:, 0, :]
        return cls_embed, decoded

    def get_embeddings(
        self,
        cls_embed: torch.Tensor,
        decoded: torch.Tensor,
        feature_type: str,
        include_queries: bool,
    ) -> torch.Tensor:
        if feature_type == "query":
            raise ValueError("feature_type='query' is not supported for vit_style")
        if include_queries:
            query_embed = decoded[:, 1:, :].reshape(decoded.size(0), -1)
            return torch.cat([cls_embed, query_embed], dim=1)
        return cls_embed
