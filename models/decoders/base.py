from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


class Decoder(nn.Module):
    decoder_type: str
    cls_dim: int

    def __init__(self) -> None:
        super().__init__()
        self.query_tokens: Optional[nn.Parameter] = None
        self.cls_token: Optional[nn.Parameter] = None

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def get_embeddings(
        self,
        cls_embed: torch.Tensor,
        decoded: torch.Tensor,
        feature_type: str,
        include_queries: bool,
    ) -> torch.Tensor:
        if feature_type == "query":
            if decoded.size(1) <= 1:
                raise ValueError("Decoder output has no query tokens for feature_type='query'")
            return decoded[:, 1:, :].reshape(decoded.size(0), -1)
        if include_queries and decoded.size(1) > 1:
            query_embed = decoded[:, 1:, :].reshape(decoded.size(0), -1)
            return torch.cat([cls_embed, query_embed], dim=1)
        return cls_embed
