from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn


class Encoder(Protocol):
    num_features: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...


@dataclass(frozen=True)
class EncoderConfig:
    name: str
    pretrained: bool
    pretrained_path: str
    in_channels: int
    patch_size: int
    resnet_3x3_stem: bool
    freeze: bool
    trainable_last_layers: int
