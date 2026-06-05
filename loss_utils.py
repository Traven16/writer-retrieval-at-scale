from __future__ import annotations

import torch
from torch.nn import functional as F


def _triplet_stats(
    valid_mask: torch.Tensor,
    active_mask: torch.Tensor,
) -> dict:
    total = int(valid_mask.numel())
    valid_count = int(valid_mask.sum().item())
    active_count = int((active_mask & valid_mask).sum().item())
    return {
        "active_triplet_ratio": (active_count / valid_count) if valid_count > 0 else 0.0,
        "valid_triplet_ratio": (valid_count / total) if total > 0 else 0.0,
        "active_triplet_count": active_count,
        "valid_triplet_count": valid_count,
        "triplet_anchor_count": total,
    }


def _triplet_embeddings(
    cls_embed: torch.Tensor, decoded: torch.Tensor, model
) -> torch.Tensor:
    if model.decoder_type in {"ghostvlad", "netvlad_transformer"}:
        return cls_embed
    if model.decoder_type == "patch_triplet":
        return decoded
    if decoded is not None and decoded.dim() == 3:
        return cls_embed
    return cls_embed


def head_cross_covariance_loss(
    head_outputs: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize between-head redundancy via batch-level cross-covariance."""
    if head_outputs is None:
        raise ValueError("head_outputs must not be None")
    if head_outputs.dim() == 4:
        batch_size, num_heads, num_clusters, head_dim = head_outputs.shape
        head_outputs = head_outputs.reshape(batch_size, num_heads, num_clusters * head_dim)
    elif head_outputs.dim() != 3:
        raise ValueError(f"Expected head outputs of shape (B,H,D) or (B,H,C,D), got {head_outputs.shape}")

    batch_size, num_heads, _ = head_outputs.shape
    if batch_size < 2 or num_heads < 2:
        return head_outputs.new_zeros(())

    centered = head_outputs - head_outputs.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    normalized = centered / std
    scale = float(max(batch_size - 1, 1))

    total = head_outputs.new_zeros(())
    pairs = 0
    for i in range(num_heads):
        zi = normalized[:, i, :]
        for j in range(i + 1, num_heads):
            zj = normalized[:, j, :]
            cross_cov = (zi.transpose(0, 1) @ zj) / scale
            total = total + cross_cov.pow(2).mean()
            pairs += 1
    if pairs == 0:
        return head_outputs.new_zeros(())
    return total / pairs


def embedding_covariance_loss(
    embeddings: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize final embedding correlation off-diagonals."""
    if embeddings.dim() != 2 or embeddings.size(0) < 2:
        return embeddings.new_zeros(())
    embeddings = F.normalize(embeddings.float(), dim=1)
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True, unbiased=False).clamp_min(eps)
    normalized = centered / std
    cov = (normalized.transpose(0, 1) @ normalized) / float(max(embeddings.size(0) - 1, 1))
    off_diag = cov - torch.diag_embed(torch.diagonal(cov))
    return off_diag.pow(2).mean()


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    if embeddings.size(0) < 2:
        zero = torch.tensor(0.0, device=embeddings.device)
        if return_stats:
            return zero, _triplet_stats(
                torch.zeros(embeddings.size(0), dtype=torch.bool, device=embeddings.device),
                torch.zeros(embeddings.size(0), dtype=torch.bool, device=embeddings.device),
            )
        return zero
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.to(dtype=torch.long)
    distances = torch.cdist(embeddings, embeddings, p=2)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = labels_equal & ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    neg_mask = ~labels_equal
    pos_dist = distances.masked_fill(~pos_mask, -float("inf"))
    neg_dist = distances.masked_fill(~neg_mask, float("inf"))
    hardest_pos = pos_dist.max(dim=1).values
    hardest_neg = neg_dist.min(dim=1).values
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    active = (hardest_pos - hardest_neg + margin) > 0
    if not valid.any():
        zero = torch.tensor(0.0, device=embeddings.device)
        if return_stats:
            return zero, _triplet_stats(valid, active)
        return zero
    loss = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    loss_mean = loss.mean()
    if return_stats:
        return loss_mean, _triplet_stats(valid, active)
    return loss_mean


def batch_hard_triplet_soft_margin_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    if embeddings.size(0) < 2:
        zero = torch.tensor(0.0, device=embeddings.device)
        if return_stats:
            return zero, _triplet_stats(
                torch.zeros(embeddings.size(0), dtype=torch.bool, device=embeddings.device),
                torch.zeros(embeddings.size(0), dtype=torch.bool, device=embeddings.device),
            )
        return zero
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.to(dtype=torch.long)
    distances = torch.cdist(embeddings, embeddings, p=2)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = labels_equal & ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    neg_mask = ~labels_equal
    pos_dist = distances.masked_fill(~pos_mask, -float("inf"))
    neg_dist = distances.masked_fill(~neg_mask, float("inf"))
    hardest_pos = pos_dist.max(dim=1).values
    hardest_neg = neg_dist.min(dim=1).values
    valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    # Soft-margin always contributes for valid anchors.
    active = valid
    if not valid.any():
        zero = torch.tensor(0.0, device=embeddings.device)
        if return_stats:
            return zero, _triplet_stats(valid, active)
        return zero
    loss = F.softplus(hardest_pos[valid] - hardest_neg[valid])
    loss_mean = loss.mean()
    if return_stats:
        return loss_mean, _triplet_stats(valid, active)
    return loss_mean


def batch_semi_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    soft_margin: bool,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    if embeddings.size(0) < 2:
        zero = torch.tensor(0.0, device=embeddings.device)
        if return_stats:
            return zero, _triplet_stats(
                torch.zeros(embeddings.size(0), dtype=torch.bool, device=embeddings.device),
                torch.zeros(embeddings.size(0), dtype=torch.bool, device=embeddings.device),
            )
        return zero
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.to(dtype=torch.long)
    distances = torch.cdist(embeddings, embeddings, p=2)
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = labels_equal & ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    neg_mask = ~labels_equal
    pos_dist = distances.masked_fill(~pos_mask, -float("inf"))
    pos_dist = pos_dist.max(dim=1).values
    neg_dist = distances.masked_fill(~neg_mask, float("inf"))
    lower = pos_dist.unsqueeze(1)
    upper = pos_dist.unsqueeze(1) + margin
    semi_mask = (neg_dist > lower) & (neg_dist < upper)
    semi_dist = neg_dist.masked_fill(~semi_mask, float("inf"))
    semi_hard_neg = semi_dist.min(dim=1).values
    valid_base = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    valid = torch.isfinite(semi_hard_neg)
    # A selected semi-hard negative is always active for hinge margin.
    active = valid
    if not valid.any():
        zero = torch.tensor(0.0, device=embeddings.device)
        if return_stats:
            return zero, _triplet_stats(valid_base, active)
        return zero
    if soft_margin:
        loss = F.softplus(pos_dist[valid] - semi_hard_neg[valid])
    else:
        loss = F.relu(pos_dist[valid] - semi_hard_neg[valid] + margin)
    loss_mean = loss.mean()
    if return_stats:
        return loss_mean, _triplet_stats(valid_base, active)
    return loss_mean
