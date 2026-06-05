from __future__ import annotations

import random
import torch
from typing import Tuple


def _flatten_draws(
    patches: torch.Tensor, labels: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if patches.dim() == 6:
        batch_size, draws, num_patches, channels, height, width = patches.shape
        patches = patches.view(batch_size * draws, num_patches, channels, height, width)
        labels = labels.repeat_interleave(draws)
        return patches, labels, draws
    return patches, labels, 1


def _split_patches_per_batch(
    patches: torch.Tensor,
    labels: torch.Tensor,
    split_options: list[int],
    apply_split: bool,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if patches.dim() != 5:
        raise ValueError(f"Expected patches (B, P, C, H, W), got {patches.shape}")
    if not apply_split or not split_options:
        return patches, labels, 1
    split = random.choice(split_options)
    split = max(1, int(split))
    if split == 1:
        return patches, labels, 1
    num_patches = patches.size(1)
    if num_patches == 1:
        # Single-image/sequence mode: no split possible.
        return patches, labels, 1
    if num_patches % split != 0:
        raise ValueError(
            f"num_patches ({num_patches}) must be divisible by split ({split})"
        )
    per_sample = num_patches // split
    patches = patches.view(
        patches.size(0) * split,
        per_sample,
        patches.size(2),
        patches.size(3),
        patches.size(4),
    )
    labels = labels.repeat_interleave(split)
    return patches, labels, split
