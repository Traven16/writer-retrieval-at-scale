from __future__ import annotations

from typing import Tuple
import warnings
import os

import timm
from timm.models.vision_transformer import VisionTransformer
from torch import nn
import torch
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from .base import EncoderConfig
from . import resnets as custom_resnets
from . import vision_transformer as custom_vit


_VIT_OVERRIDE = {
    "vit_small_patch4_32": (384, 6, 4),
    "vit_tiny_patch4_32": (192, 3, 4),
    "vit_tiny_patch8_32": (192, 3, 8),
}

_CUSTOM_VIT = {
    "custom_vit_tiny": (custom_vit.vit_tiny, 4),
    "custom_vit_small": (custom_vit.vit_small, 16),
    "custom_vit_base": (custom_vit.vit_base, 8),
    "custom_vit_large": (custom_vit.vit_large, 16),
}

def _load_pretrained_weights(encoder: nn.Module, path: str) -> None:
    if not path:
        return
    if not os.path.exists(path):
        raise FileNotFoundError(f"Encoder pretrained path not found: {path}")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if "teacher" in state:
            state = state["teacher"]
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    if isinstance(state, dict):
        prefixes = ("module.", "backbone.", "student.backbone.", "teacher.backbone.")
        normalized = {}
        for key, value in state.items():
            if not isinstance(key, str):
                normalized[key] = value
                continue
            new_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                        changed = True
            normalized[new_key] = value
        state = normalized

    current = encoder.state_dict()
    filtered = {}
    skipped_mismatch = []
    for key, value in state.items():
        if key not in current:
            filtered[key] = value
            continue
        target = current[key]
        if not hasattr(value, "shape") or not hasattr(target, "shape"):
            filtered[key] = value
            continue
        if value.shape == target.shape:
            filtered[key] = value
            continue
        if (
            key == "patch_embed.proj.weight"
            and value.ndim == 4
            and target.ndim == 4
            and value.shape[0] == target.shape[0]
            and value.shape[2:] == target.shape[2:]
        ):
            if value.shape[1] == 1 and target.shape[1] == 3:
                filtered[key] = value.repeat(1, 3, 1, 1) / 3.0
                continue
            if value.shape[1] == 3 and target.shape[1] == 1:
                filtered[key] = value.mean(dim=1, keepdim=True)
                continue
        skipped_mismatch.append(key)

    missing, unexpected = encoder.load_state_dict(filtered, strict=False)
    if missing or unexpected:
        print(
            f"[encoder] loaded pretrained weights from {path} | "
            f"missing {len(missing)} unexpected {len(unexpected)} "
            f"skipped_mismatch {len(skipped_mismatch)}"
        )


def _build_override_vit(config: EncoderConfig) -> nn.Module:
    embed_dim, num_heads, patch_size_override = _VIT_OVERRIDE[config.name]
    encoder = VisionTransformer(
        img_size=config.patch_size,
        patch_size=patch_size_override,
        in_chans=config.in_channels,
        num_classes=0,
        embed_dim=embed_dim,
        depth=12,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
    )
    if hasattr(encoder, "reset_classifier"):
        encoder.reset_classifier(0, "")
    return encoder


def _build_custom_vit(config: EncoderConfig) -> nn.Module:
    ctor, patch_size = _CUSTOM_VIT[config.name]
    patch_size = min(patch_size, config.patch_size)
    encoder = ctor(
        patch_size=patch_size,
        img_size=[224],
        in_chans=config.in_channels,
        num_classes=0,
        return_all_tokens=False,
    )
    encoder.num_features = encoder.embed_dim
    return encoder


def build_encoder(config: EncoderConfig) -> nn.Module:
    if config.name.startswith("custom_resnet"):
        if config.pretrained:
            warnings.warn("custom_resnet does not support pretrained weights; ignoring.")
        if config.resnet_3x3_stem:
            warnings.warn("custom_resnet does not support resnet_3x3_stem; ignoring.")
        name = config.name.replace("custom_", "")
        if not hasattr(custom_resnets, name):
            raise ValueError(f"Unknown custom resnet name: {config.name}")
        encoder = getattr(custom_resnets, name)()
        if config.in_channels != 3 and hasattr(encoder, "conv1"):
            encoder.conv1 = nn.Conv2d(
                config.in_channels,
                encoder.conv1.out_channels,
                kernel_size=encoder.conv1.kernel_size,
                stride=encoder.conv1.stride,
                padding=encoder.conv1.padding,
                bias=False,
            )
        encoder.num_features = encoder.layer3[-1].bn2.num_features
        _load_pretrained_weights(encoder, config.pretrained_path)
        return encoder
    if config.name.startswith("torchvision_resnet"):
        resnet_map = {
            "torchvision_resnet18": (resnet18, ResNet18_Weights.DEFAULT),
            "torchvision_resnet34": (resnet34, ResNet34_Weights.DEFAULT),
            "torchvision_resnet50": (resnet50, ResNet50_Weights.DEFAULT),
            "torchvision_resnet101": (resnet101, ResNet101_Weights.DEFAULT),
            "torchvision_resnet152": (resnet152, ResNet152_Weights.DEFAULT),
        }
        if config.name not in resnet_map:
            raise ValueError(f"Unknown torchvision resnet name: {config.name}")
        ctor, weights = resnet_map[config.name]
        encoder = ctor(weights=weights if config.pretrained else None)
        if config.resnet_3x3_stem and hasattr(encoder, "conv1"):
            encoder.conv1 = nn.Conv2d(
                config.in_channels,
                encoder.conv1.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            if hasattr(encoder, "maxpool"):
                encoder.maxpool = nn.Identity()
        elif config.in_channels != 3 and hasattr(encoder, "conv1"):
            encoder.conv1 = nn.Conv2d(
                config.in_channels,
                encoder.conv1.out_channels,
                kernel_size=encoder.conv1.kernel_size,
                stride=encoder.conv1.stride,
                padding=encoder.conv1.padding,
                bias=False,
            )
        out_features = encoder.fc.in_features
        encoder.fc = nn.Identity()
        encoder.num_features = out_features
        _load_pretrained_weights(encoder, config.pretrained_path)
        return encoder

    if config.name in _VIT_OVERRIDE:
        encoder = _build_override_vit(config)
        _load_pretrained_weights(encoder, config.pretrained_path)
        return encoder

    if config.name in _CUSTOM_VIT:
        encoder = _build_custom_vit(config)
        _load_pretrained_weights(encoder, config.pretrained_path)
        return encoder

    encoder_kwargs = {
        "pretrained": config.pretrained,
        "num_classes": 0,
        "global_pool": "",
        "in_chans": config.in_channels,
    }
    if "vit" in config.name:
        encoder_kwargs["img_size"] = config.patch_size
    encoder = timm.create_model(config.name, **encoder_kwargs)
    if config.resnet_3x3_stem and config.name.startswith("resnet") and hasattr(encoder, "conv1"):
        encoder.conv1 = nn.Conv2d(
            config.in_channels,
            encoder.conv1.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        if hasattr(encoder, "maxpool"):
            encoder.maxpool = nn.Identity()
    _load_pretrained_weights(encoder, config.pretrained_path)
    return encoder
