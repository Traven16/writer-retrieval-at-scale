from __future__ import annotations

from dataclasses import dataclass

from .base import Decoder
from .xvlad_decoder import XVLADDecoder
from .ghostvlad_decoder import GhostVLADDecoder
from .mean_pooling_decoder import MeanPoolingDecoder
from .patch_triplet_decoder import PatchTripletDecoder
from .netvlad_transformer_decoder import NetVLADTransformerDecoder
from .reversed_transformer_decoder import ReversedTransformerDecoder
from .transformer_decoder import TransformerDecoder
from .vit_style_decoder import ViTStyleDecoder


@dataclass(frozen=True)
class DecoderConfig:
    decoder_type: str
    d_model: int
    nhead: int
    d_head: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    num_queries: int
    ghostvlad_clusters: int
    ghostvlad_ghost_clusters: int
    xvlad_no_intra_norm: bool
    decoder_checkpoint: bool


def build_decoder(config: DecoderConfig) -> Decoder:
    if config.decoder_type == "transformer_decoder":
        return TransformerDecoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_queries=config.num_queries,
            decoder_checkpoint=config.decoder_checkpoint,
        )
    if config.decoder_type == "reversed_transformer_decoder":
        return ReversedTransformerDecoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_queries=config.num_queries,
            decoder_checkpoint=config.decoder_checkpoint,
        )
    if config.decoder_type == "vit_style":
        return ViTStyleDecoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        )
    if config.decoder_type == "ghostvlad":
        return GhostVLADDecoder(
            d_model=config.d_model,
            ghostvlad_clusters=config.ghostvlad_clusters,
            ghostvlad_ghost_clusters=config.ghostvlad_ghost_clusters,
        )
    if config.decoder_type == "xvlad":
        return XVLADDecoder(
            d_model=config.d_model,
            ghostvlad_clusters=config.ghostvlad_clusters,
            ghostvlad_ghost_clusters=config.ghostvlad_ghost_clusters,
            nhead=config.nhead,
            d_head=config.d_head,
            xvlad_no_intra_norm=config.xvlad_no_intra_norm,
        )
    if config.decoder_type == "mean_pooling":
        return MeanPoolingDecoder(d_model=config.d_model)
    if config.decoder_type == "patch_triplet":
        return PatchTripletDecoder(d_model=config.d_model)
    if config.decoder_type == "netvlad_transformer":
        return NetVLADTransformerDecoder(
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            num_clusters=config.ghostvlad_clusters,
            ghost_clusters=config.ghostvlad_ghost_clusters,
        )
    raise ValueError(f"Unknown decoder_type: {config.decoder_type}")

__all__ = [
    "Decoder",
    "DecoderConfig",
    "XVLADDecoder",
    "GhostVLADDecoder",
    "MeanPoolingDecoder",
    "PatchTripletDecoder",
    "NetVLADTransformerDecoder",
    "ReversedTransformerDecoder",
    "TransformerDecoder",
    "ViTStyleDecoder",
    "build_decoder",
]
