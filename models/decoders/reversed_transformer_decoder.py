from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import checkpoint

from .base import Decoder


class _ReversedCrossAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError("Expected (B, T, D) tensors for query/key/value")
        batch_size, q_len, _ = query.shape
        k_len = key.size(1)
        q = self.q_proj(query).view(batch_size, q_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, k_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, k_len, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-2)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        return self.out_proj(out)


class ReversedTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.cross_attn = _ReversedCrossAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if tgt.dim() != 3 or memory.dim() != 3:
            raise ValueError("Expected (B, T, D) tensors for tgt/memory")
        tgt2 = self.self_attn(tgt, tgt, tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout3(tgt2))
        return tgt


class ReversedTransformerDecoder(Decoder):
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
        self.decoder_type = "reversed_transformer_decoder"
        self.cls_dim = d_model
        self.decoder_checkpoint = decoder_checkpoint
        self.layers = nn.ModuleList(
            [
                ReversedTransformerDecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries + 1, d_model))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if embeddings.dim() != 3:
            raise ValueError(f"Expected embeddings of shape (B, N, D), got {embeddings.shape}")
        batch_size = embeddings.size(0)
        queries = self.query_tokens.expand(batch_size, -1, -1)
        if queries.shape[1] > 16:
            raise AssertionError("num_queries is not small; check your config!")
        output = queries
        if self.decoder_checkpoint and self.training:
            for layer in self.layers:
                output = checkpoint.checkpoint(layer, output, embeddings, use_reentrant=False)
        else:
            for layer in self.layers:
                output = layer(output, embeddings)
        decoded = self.norm(output)
        cls_embed = decoded[:, 0, :]
        return cls_embed, decoded
