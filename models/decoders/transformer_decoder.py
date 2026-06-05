from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.utils import checkpoint

from .base import Decoder


class TransformerDecoder(Decoder):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_queries: int,
        decoder_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.decoder_type = "transformer_decoder"
        self.cls_dim = d_model
        self.decoder_checkpoint = decoder_checkpoint
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries + 1, d_model))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings of shape (B, N, D), got {embeddings.shape}")
        batch_size = embeddings.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        # if queries.shape[1] > 16:
        #     raise AssertionError("num_queries is not small; check your config!")
        if self.decoder_checkpoint and self.training:
            output = queries
            for layer in self.decoder.layers:
                output = checkpoint.checkpoint(
                    layer, output, embeddings, use_reentrant=False
                )
            if self.decoder.norm is not None:
                output = self.decoder.norm(output)
            decoded = output
        else:
            decoded = self.decoder(queries, embeddings)
        cls_embed = decoded[:, 0, :]
        return cls_embed, decoded

    def get_embeddings(
        self,
        cls_embed: torch.Tensor,
        decoded: torch.Tensor,
        feature_type: str,
        include_queries: bool,
    ) -> torch.Tensor:
        return super().get_embeddings(cls_embed, decoded, feature_type, include_queries)
