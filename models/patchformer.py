from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from models.decoders import DecoderConfig, build_decoder
from models.encoders import EncoderConfig, build_encoder

import torch.nn.functional as F


@dataclass(frozen=True)
class PatchformerConfig:
    num_classes: int
    num_patches: int
    num_queries: int
    patch_size: int
    in_channels: int
    # Explicit dimensions.
    d_encoder: int
    d_out: int
    n_head: int
    d_head: int
    # Backward-compatible aliases (kept for old configs/checkpoints/logs).
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    encoder_name: str
    encoder_pretrained: bool
    encoder_pretrained_path: str
    encoder_return_patch_sequence: bool
    encoder_freeze: bool
    encoder_trainable_last_layers: int
    decoder_type: str
    ghostvlad_clusters: int
    ghostvlad_ghost_clusters: int
    xvlad_no_intra_norm: bool
    embedding_whitening_head: bool
    decoder_checkpoint: bool
    resnet_3x3_stem: bool
    no_classifier: bool


class Patchformer(nn.Module):
    def __init__(self, config: PatchformerConfig) -> None:
        super().__init__()
        self.config = config

        self.num_patches = config.num_patches
        self.patch_size = config.patch_size
        self.decoder_type = config.decoder_type
        self.encoder_return_patch_sequence = config.encoder_return_patch_sequence

        self.patch_embedding = build_encoder(
            EncoderConfig(
                name=config.encoder_name,
                pretrained=config.encoder_pretrained,
                pretrained_path=config.encoder_pretrained_path,
                in_channels=config.in_channels,
                patch_size=config.patch_size,
                resnet_3x3_stem=config.resnet_3x3_stem,
                freeze=config.encoder_freeze,
                trainable_last_layers=config.encoder_trainable_last_layers,
            )
        )
        self._apply_encoder_trainability(
            freeze=config.encoder_freeze,
            trainable_last_layers=config.encoder_trainable_last_layers,
        )
        out_features = getattr(self.patch_embedding, "num_features", config.d_encoder)
        self.backbone_dropout = nn.Dropout(p=config.dropout)
        if out_features == config.d_encoder:
            self.patch_proj = nn.Identity()
        else:
            self.patch_proj = nn.Linear(out_features, config.d_encoder)
        if config.d_encoder == config.d_out:
            self.output_proj = nn.Identity()
        else:
            self.output_proj = nn.Linear(config.d_encoder, config.d_out)
        if config.num_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_encoder,
                nhead=config.n_head,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
            )
            self.patch_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=config.num_encoder_layers
            )
        else:
            self.patch_encoder = None

        self.decoder = build_decoder(
            DecoderConfig(
                decoder_type=config.decoder_type,
                d_model=config.d_out,
                nhead=config.n_head,
                d_head=config.d_head,
                num_decoder_layers=config.num_decoder_layers,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                num_queries=config.num_queries,
                ghostvlad_clusters=config.ghostvlad_clusters,
                ghostvlad_ghost_clusters=config.ghostvlad_ghost_clusters,
                xvlad_no_intra_norm=config.xvlad_no_intra_norm,
                decoder_checkpoint=config.decoder_checkpoint,
            )
        )
        cls_dim = self.decoder.cls_dim
        self.query_tokens = getattr(self.decoder, "query_tokens", None)
        self.cls_token = getattr(self.decoder, "cls_token", None)
        self.embedding_whitening = None
        if config.embedding_whitening_head:
            whitening = nn.Linear(cls_dim, cls_dim, bias=False)
            nn.init.eye_(whitening.weight)
            self.embedding_whitening = nn.Sequential(
                nn.LayerNorm(cls_dim),
                whitening,
            )

        self.no_classifier = config.no_classifier
        self.classifier = None if config.no_classifier else nn.Linear(
            cls_dim, config.num_classes
        )
        self._full_page_debug_printed = False
        self._non_patchsize_debug_printed = False

        self._reset_classifier()

    def _iter_encoder_trainable_tail(self, trainable_last_layers: int):
        if trainable_last_layers <= 0:
            return []
        encoder = self.patch_embedding
        if hasattr(encoder, "blocks"):
            return list(encoder.blocks)[-trainable_last_layers:]
        if hasattr(encoder, "stages"):
            blocks = []
            for stage in encoder.stages:
                if hasattr(stage, "blocks"):
                    blocks.extend(list(stage.blocks))
                else:
                    blocks.extend(list(stage.children()))
            if blocks:
                return blocks[-trainable_last_layers:]
        children = list(encoder.children())
        return children[-trainable_last_layers:]

    def _apply_encoder_trainability(
        self,
        freeze: bool,
        trainable_last_layers: int,
    ) -> None:
        if not freeze:
            return
        for param in self.patch_embedding.parameters():
            param.requires_grad = False
        trainable_modules = self._iter_encoder_trainable_tail(trainable_last_layers)
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True
        for name in ("norm", "fc_norm", "head"):
            module = getattr(self.patch_embedding, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = trainable_last_layers > 0
        trainable = sum(p.numel() for p in self.patch_embedding.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.patch_embedding.parameters())
        print(
            "Encoder trainability | "
            f"frozen={freeze} | trainable_last_layers={trainable_last_layers} | "
            f"trainable_params={trainable}/{total}"
        )

    def set_encoder_training(self, enabled: bool) -> None:
        if not enabled:
            for param in self.patch_embedding.parameters():
                param.requires_grad = False
            return
        trainable_last_layers = int(getattr(self.config, "encoder_trainable_last_layers", 0) or 0)
        if getattr(self.config, "encoder_freeze", False):
            self._apply_encoder_trainability(
                freeze=True,
                trainable_last_layers=trainable_last_layers,
            )
            return
        for param in self.patch_embedding.parameters():
            param.requires_grad = True

    def _reset_classifier(self) -> None:
        if self.classifier is not None:
            nn.init.trunc_normal_(self.classifier.weight, std=0.02)
            nn.init.zeros_(self.classifier.bias)

    def _apply_embedding_whitening(self, cls_embed: torch.Tensor) -> torch.Tensor:
        if self.embedding_whitening is None:
            return cls_embed
        return self.embedding_whitening(cls_embed)

    def _run_backbone(self, images: torch.Tensor) -> torch.Tensor:
        if self.encoder_return_patch_sequence and hasattr(self.patch_embedding, "forward_features"):
            features = self.patch_embedding.forward_features(images)
        else:
            features = self.patch_embedding(images)
        if isinstance(features, dict):
            if "x_norm_patchtokens" in features:
                features = features["x_norm_patchtokens"]
            elif "x_prenorm" in features:
                features = features["x_prenorm"]
            elif "x" in features:
                features = features["x"]
        if isinstance(features, (list, tuple)):
            features = features[-1]
        return features

    def _encode_full_spatial(
        self,
        images: torch.Tensor,
        pre_pool_stride: int = 1,
        debug_once: bool = False,
    ) -> torch.Tensor:
        if images.dim() != 4:
            raise ValueError(f"Expected images of shape (B, C, H, W), got {images.shape}")
        batch_size, channels, height, width = images.shape
        if (
            self.patch_size > 0
            and (height != self.patch_size or width != self.patch_size)
            and height % self.patch_size == 0
            and width % self.patch_size == 0
        ):
            grid_h = height // self.patch_size
            grid_w = width // self.patch_size
            windows = (
                images.unfold(2, self.patch_size, self.patch_size)
                .unfold(3, self.patch_size, self.patch_size)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(batch_size * grid_h * grid_w, channels, self.patch_size, self.patch_size)
            )
            features = self._run_backbone(windows)
            if features.dim() == 4:
                if pre_pool_stride > 1:
                    features = F.max_pool2d(
                        features,
                        kernel_size=pre_pool_stride,
                        stride=pre_pool_stride,
                    )
                features = features.flatten(2).transpose(1, 2)
            elif features.dim() == 3:
                if features.size(1) > 1 and hasattr(self.patch_embedding, "cls_token"):
                    features = features[:, 1:, :]
            else:
                raise ValueError(
                    f"Unexpected encoder output shape for windowed full-spatial mode: {features.shape}"
                )
            features = self.backbone_dropout(features)
            embeddings = self.output_proj(self.patch_proj(features))
            embeddings = embeddings.reshape(batch_size, grid_h * grid_w * embeddings.size(1), -1)
            if debug_once and not self._full_page_debug_printed:
                print(
                    "Full-page encoder debug | "
                    f"input {tuple(images.shape)} | windows {grid_h}x{grid_w} "
                    f"window_size {self.patch_size} | token_seq {tuple(embeddings.shape)}"
                )
                self._full_page_debug_printed = True
            if self.patch_encoder is not None:
                embeddings = self.patch_encoder(embeddings)
            return embeddings
        features = self._run_backbone(images)
        if features.dim() == 4:
            if pre_pool_stride > 1:
                features = F.max_pool2d(
                    features,
                    kernel_size=pre_pool_stride,
                    stride=pre_pool_stride,
                )
            features = features.flatten(2).transpose(1, 2)
        elif features.dim() == 3:
            if features.size(1) > 1 and hasattr(self.patch_embedding, "cls_token"):
                features = features[:, 1:, :]
        else:
            raise ValueError(
                f"Unexpected encoder output shape for full-spatial mode: {features.shape}"
            )
        features = self.backbone_dropout(features)
        embeddings = self.output_proj(self.patch_proj(features))
        if debug_once and not self._full_page_debug_printed:
            print(
                "Full-page encoder debug | "
                f"input {tuple(images.shape)} | "
                f"token_seq {tuple(embeddings.shape)}"
            )
            self._full_page_debug_printed = True
        if self.patch_encoder is not None:
            embeddings = self.patch_encoder(embeddings)
        return embeddings

    def _encode_patches(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dim() != 5:
            raise ValueError(f"Expected patches of shape (B, N, C, H, W), got {patches.shape}")
        batch_size, num_patches, _, height, width = patches.shape
        if height != self.patch_size or width != self.patch_size:
            if not self._non_patchsize_debug_printed:
                print(
                    "Patchformer input debug | "
                    f"received spatial {(height, width)} with N={num_patches} "
                    f"(configured patch_size={self.patch_size}); using full-spatial/window tokens (no GAP)."
                )
                self._non_patchsize_debug_printed = True
            patches = patches.view(batch_size * num_patches, patches.size(2), height, width)
            embeddings = self._encode_full_spatial(patches, pre_pool_stride=1, debug_once=False)
            return embeddings.reshape(batch_size, num_patches * embeddings.size(1), -1)
        # if num_patches > self.num_patches:
        #     if self.decoder_type == "transformer_decoder" and self.query_tokens is not None:
        #         print(
        #             f"Warning: num_patches ({num_patches}) exceeds model capacity "
        #             f"({self.num_patches}); continuing anyway."
        #         )
        patches = patches.view(batch_size * num_patches, patches.size(2), height, width)
        features = self._run_backbone(patches)

        if self.encoder_return_patch_sequence:
            if features.dim() == 4:
                # CNN-like fallback: flatten spatial map as token sequence.
                features = features.flatten(2).transpose(1, 2)
            elif features.dim() == 3:
                # Typical ViT forward_features returns [CLS]+patch tokens.
                if features.size(1) > 1 and hasattr(self.patch_embedding, "cls_token"):
                    features = features[:, 1:, :]
            else:
                raise ValueError(
                    f"Expected sequence features (B,T,D) or map features (B,D,H,W), got {features.shape}"
                )
            features = self.backbone_dropout(features)
            features = self.patch_proj(features)
            features = self.output_proj(features)
            embeddings = features.reshape(batch_size, num_patches * features.size(1), -1)
        else:
            if features.dim() == 4:
                features = features.mean(dim=(2, 3))
            elif features.dim() == 3:
                features = features[:, 0, :]
            features = self.backbone_dropout(features)
            features = features.view(batch_size * num_patches, -1)
            embeddings = self.patch_proj(features)
            embeddings = self.output_proj(embeddings).view(batch_size, num_patches, -1)
        #embeddings = nn.Identity()(features).view(batch_size, num_patches, -1)
        if self.patch_encoder is not None:
            embeddings = self.patch_encoder(embeddings)
        return embeddings

    def encode_embeddings(
        self,
        patches: torch.Tensor,
        feature_type: str = "cls",
        include_queries: bool = False,
    ) -> torch.Tensor:
        embeddings = self._encode_patches(patches)
        cls_embed, decoded = self.decoder(embeddings)
        cls_embed = self._apply_embedding_whitening(cls_embed)
        return self.decoder.get_embeddings(cls_embed, decoded, feature_type, include_queries)

    def encode_full_image_embeddings(
        self,
        images: torch.Tensor,
        feature_type: str = "cls",
        include_queries: bool = False,
        pre_pool_stride: int = 1,
    ) -> torch.Tensor:
        """Encode full-page images (B,C,H,W) into retrieval embeddings.

        This bypasses patch sampling and uses the full spatial feature map as
        a token sequence for the decoder.
        """
        embeddings = self._encode_full_spatial(
            images,
            pre_pool_stride=pre_pool_stride,
            debug_once=True,
        )
        cls_embed, decoded = self.decoder(embeddings)
        cls_embed = self._apply_embedding_whitening(cls_embed)
        return self.decoder.get_embeddings(cls_embed, decoded, feature_type, include_queries)

    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: Tensor of shape (B, N, C, H, W) with H=W=patch_size.
        Returns:
            logits: (B, num_classes)
            cls_embed: (B, d_model)
            decoded: (B, num_queries + 1, d_model)
        """
        embeddings = self._encode_patches(patches)
        cls_embed, decoded = self.decoder(embeddings)
        cls_embed = self._apply_embedding_whitening(cls_embed)
        logits = self.classifier(cls_embed) if self.classifier is not None else None
        return logits, cls_embed, decoded


class ArcFaceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        margin: float = 0.5,
        scale: float = 64.0,
        mlp_dim: int = 0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.mlp = None
        if mlp_dim and mlp_dim > 0:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, mlp_dim),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_dim, embedding_dim),
            )
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mlp is not None:
            embeddings = self.mlp(embeddings)
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)

        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_logit = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(dtype=cosine.dtype)
        logits = cosine * (1.0 - one_hot) + target_logit * one_hot
        logits = logits * self.scale
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    def raw_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.mlp is not None:
            embeddings = self.mlp(embeddings)
        embeddings = embeddings.to(dtype=self.weight.dtype)
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        return F.linear(embeddings, weight) * self.scale
