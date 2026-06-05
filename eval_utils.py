from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, List

import math
import os
import random
import shutil
import time
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

from batch_utils import _split_patches_per_batch
from data.dataset import PatchDataset, normalize_patch_tensor
from loss_utils import (
    _triplet_embeddings,
    batch_hard_triplet_loss,
    batch_hard_triplet_soft_margin_loss,
    batch_semi_hard_triplet_loss,
)
from visualize_utils import render_decoder_cross_attention_image


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def accuracy_topk(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    if logits.numel() == 0:
        return 0.0
    k = min(k, logits.size(1))
    _, topk_idx = torch.topk(logits, k=k, dim=1)
    correct = topk_idx.eq(labels.view(-1, 1))
    return correct.any(dim=1).float().mean().item()


def knn_accuracy(
    support_embeddings: torch.Tensor,
    support_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_labels: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
) -> float:
    support_embeddings = F.normalize(support_embeddings, dim=1)
    query_embeddings = F.normalize(query_embeddings, dim=1)
    support_labels = support_labels.to(dtype=torch.long)
    query_labels = query_labels.to(dtype=torch.long)

    correct = 0
    total = query_embeddings.size(0)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = query_embeddings[start:end]
        sims = torch.matmul(chunk, support_embeddings.T)
        _, indices = torch.topk(sims, k=k, dim=1)
        neighbor_labels = support_labels[indices]
        preds = torch.mode(neighbor_labels, dim=1).values
        correct += (preds == query_labels[start:end]).sum().item()

    return correct / max(1, total)


def knn_accuracy_leave_one_out(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    chunk_size: int = 1024,
) -> float:
    total = embeddings.size(0)
    if total < 2:
        return 0.0
    k = min(k, total - 1)
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.to(dtype=torch.long)
    correct = 0

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        chunk = embeddings[start:end]
        sims = torch.matmul(chunk, embeddings.T)
        row_idx = torch.arange(end - start)
        col_idx = torch.arange(start, end)
        sims[row_idx, col_idx] = -float("inf")
        _, indices = torch.topk(sims, k=k, dim=1)
        neighbor_labels = labels[indices]
        preds = torch.mode(neighbor_labels, dim=1).values
        correct += (preds == labels[start:end]).sum().item()

    return correct / max(1, total)


def retrieval_metrics_leave_one_out(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 512,
    query_mask: Optional[torch.Tensor] = None,
    percent_thresholds: Tuple[float, ...] = (0.1, 0.2, 0.5),
    return_not_within_top10: bool = False,
    show_progress: bool = False,
    progress_desc: str = "retrieval",
) -> Tuple[float, float, Dict[str, float], List[int]]:
    total = embeddings.size(0)
    if total < 2:
        stats = {
            "queries_total": 0,
            "queries_with_relevant": 0,
            "queries_without_relevant": 0,
        }
        for pct in percent_thresholds:
            stats[f"closest_not_within_top{int(pct * 100)}pct"] = 0
        return 0.0, 0.0, stats, []
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.to(dtype=torch.long)
    label_counts = torch.bincount(labels)
    valid_query_mask = label_counts[labels] > 1
    singleton_count = (label_counts[labels] == 1).sum().item()
    if query_mask is not None:
        query_mask = query_mask.to(dtype=torch.bool)
        valid_query_mask = valid_query_mask & query_mask
    query_indices = torch.nonzero(valid_query_mask, as_tuple=False).squeeze(1)
    if query_indices.numel() == 0:
        stats = {
            "queries_total": 0,
            "queries_with_relevant": 0,
            "queries_without_relevant": 0,
        }
        for pct in percent_thresholds:
            stats[f"closest_not_within_top{int(pct * 100)}pct"] = 0
        return 0.0, 0.0, stats, []

    top1_correct = 0
    ap_sum = 0.0
    ap_count = 0
    queries_with_relevant = 0
    gallery_size = total - 1
    threshold_ranks = [
        max(1, math.ceil(pct * gallery_size)) for pct in percent_thresholds
    ]
    not_within_counts = [0 for _ in percent_thresholds]
    top10_threshold = max(1, math.ceil(0.1 * gallery_size))
    not_within_top10_indices: List[int] = []

    starts = list(range(0, query_indices.numel(), chunk_size))
    iterator = starts
    if show_progress:
        iterator = tqdm(starts, desc=progress_desc, leave=False)
    for start in iterator:
        end = min(start + chunk_size, query_indices.numel())
        chunk_indices = query_indices[start:end]
        chunk = embeddings[chunk_indices]
        sims = torch.matmul(chunk, embeddings.T)
        row_idx = torch.arange(end - start)
        sims[row_idx, chunk_indices] = -float("inf")
        indices = torch.argsort(sims, dim=1, descending=True)

        query_labels = labels[chunk_indices].unsqueeze(1)
        ranked_labels = labels[indices]
        self_mask = indices.eq(chunk_indices.unsqueeze(1))
        matches = ranked_labels.eq(query_labels) & ~self_mask
        top1_correct += matches[:, 0].sum().item()

        match_counts = matches.sum(dim=1)
        valid = match_counts > 0
        if valid.any():
            valid_matches = matches[valid]
            matches_f = valid_matches.float()
            cumsum = matches_f.cumsum(dim=1)
            ranks = torch.arange(1, matches_f.size(1) + 1).float().unsqueeze(0)
            precision = cumsum / ranks
            ap = (precision * matches_f).sum(dim=1) / match_counts[valid].float()
            ap_sum += ap.sum().item()
            ap_count += ap.numel()

            first_match_rank = valid_matches.float().argmax(dim=1) + 1
            for idx, threshold in enumerate(threshold_ranks):
                not_within_counts[idx] += (first_match_rank > threshold).sum().item()
            queries_with_relevant += valid.sum().item()
            if return_not_within_top10:
                not_within_top10 = first_match_rank > top10_threshold
                if not_within_top10.any():
                    not_within_top10_indices.extend(
                        chunk_indices[valid][not_within_top10].tolist()
                    )

    queries_total = query_indices.numel()
    top1 = top1_correct / max(1, queries_total)
    mean_ap = ap_sum / max(1, ap_count)
    stats = {
        "queries_total": queries_total,
        "queries_with_relevant": queries_with_relevant,
        "queries_without_relevant": queries_total - queries_with_relevant,
        "singletons": singleton_count,
    }
    for pct, count in zip(percent_thresholds, not_within_counts):
        stats[f"closest_not_within_top{int(pct * 100)}pct"] = count
    return top1, mean_ap, stats, not_within_top10_indices


def retrieval_map_progressive_gallery(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    writer_chunk_size: int = 1000,
    min_writers: int = 1000,
    query_mask: Optional[torch.Tensor] = None,
    shuffle_writers: bool = False,
    seed: int = 0,
    show_progress: bool = False,
    ap_hist_bins: int = 0,
    ap_hist_full_only: bool = False,
    print_each: bool = False,
) -> List[Dict[str, float]]:
    """Evaluate retrieval as writer count grows in fixed-size chunks."""
    labels = labels.to(dtype=torch.long)
    unique_writers = torch.unique(labels)
    if unique_writers.numel() == 0:
        return []
    if shuffle_writers:
        g = torch.Generator(device=unique_writers.device)
        g.manual_seed(int(seed))
        perm = torch.randperm(unique_writers.numel(), generator=g, device=unique_writers.device)
        unique_writers = unique_writers[perm]
    else:
        unique_writers, _ = torch.sort(unique_writers)

    total_writers = int(unique_writers.numel())
    chunk = max(1, int(writer_chunk_size))
    start = max(1, int(min_writers))
    results: List[Dict[str, float]] = []

    writer_counts = list(range(start, total_writers + 1, chunk))
    iterator = writer_counts
    if show_progress:
        iterator = tqdm(writer_counts, desc="gallery growth", leave=False)

    def _attach_ap_hist(
        row: Dict[str, float],
        emb_subset: torch.Tensor,
        lbl_subset: torch.Tensor,
        qmask_subset: Optional[torch.Tensor],
        is_full_set: bool,
    ) -> None:
        if ap_hist_bins <= 0:
            return
        if ap_hist_full_only and not is_full_set:
            return
        _, ap_scores = retrieval_ap_per_query_leave_one_out(
            emb_subset,
            lbl_subset,
            query_mask=qmask_subset,
        )
        if ap_scores.numel() > 0:
            hist = torch.histc(ap_scores, bins=int(ap_hist_bins), min=0.0, max=1.0)
            row["ap_hist"] = [int(v) for v in hist.tolist()]
        else:
            row["ap_hist"] = [0 for _ in range(int(ap_hist_bins))]

    for writer_count in iterator:
        selected = unique_writers[:writer_count]
        subset_mask = torch.isin(labels, selected)
        emb_subset = embeddings[subset_mask]
        lbl_subset = labels[subset_mask]
        if query_mask is not None:
            qmask_subset = query_mask[subset_mask]
        else:
            qmask_subset = None
        top1, m_ap, stats, _ = retrieval_metrics_leave_one_out(
            emb_subset,
            lbl_subset,
            query_mask=qmask_subset,
            return_not_within_top10=False,
            show_progress=False,
        )
        row = {
            "writers": float(writer_count),
            "samples": float(emb_subset.size(0)),
            "top1": float(top1),
            "map": float(m_ap),
            "queries_total": float(stats.get("queries_total", 0)),
            "queries_with_relevant": float(stats.get("queries_with_relevant", 0)),
        }
        _attach_ap_hist(
            row,
            emb_subset,
            lbl_subset,
            qmask_subset,
            is_full_set=(int(writer_count) >= total_writers),
        )
        results.append(row)
        if print_each:
            print(
                "Gallery growth | "
                f"writers {int(row['writers'])} | "
                f"samples {int(row['samples'])} | "
                f"Top-1 {row['top1']:.4f} mAP {row['map']:.4f}"
            )
            if ap_hist_bins > 0 and "ap_hist" in row:
                print(
                    "Gallery growth AP hist | "
                    f"writers {int(row['writers'])} | bins {row['ap_hist']}"
                )
    if not results or int(results[-1]["writers"]) != total_writers:
        selected = unique_writers
        subset_mask = torch.isin(labels, selected)
        emb_subset = embeddings[subset_mask]
        lbl_subset = labels[subset_mask]
        qmask_subset = query_mask[subset_mask] if query_mask is not None else None
        top1, m_ap, stats, _ = retrieval_metrics_leave_one_out(
            emb_subset,
            lbl_subset,
            query_mask=qmask_subset,
            return_not_within_top10=False,
        )
        row = {
            "writers": float(total_writers),
            "samples": float(emb_subset.size(0)),
            "top1": float(top1),
            "map": float(m_ap),
            "queries_total": float(stats.get("queries_total", 0)),
            "queries_with_relevant": float(stats.get("queries_with_relevant", 0)),
        }
        _attach_ap_hist(
            row,
            emb_subset,
            lbl_subset,
            qmask_subset,
            is_full_set=True,
        )
        results.append(row)
        if print_each:
            print(
                "Gallery growth | "
                f"writers {int(row['writers'])} | "
                f"samples {int(row['samples'])} | "
                f"Top-1 {row['top1']:.4f} mAP {row['map']:.4f}"
            )
            if ap_hist_bins > 0 and "ap_hist" in row:
                print(
                    "Gallery growth AP hist | "
                    f"writers {int(row['writers'])} | bins {row['ap_hist']}"
                )
    return results


def retrieval_ap_per_query_leave_one_out(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 512,
    query_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return valid query indices and their leave-one-out AP scores."""
    total = embeddings.size(0)
    if total < 2:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.to(dtype=torch.long)
    label_counts = torch.bincount(labels)
    valid_query_mask = label_counts[labels] > 1
    if query_mask is not None:
        valid_query_mask = valid_query_mask & query_mask.to(dtype=torch.bool)
    query_indices = torch.nonzero(valid_query_mask, as_tuple=False).squeeze(1)
    if query_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)

    ap_scores = torch.zeros(query_indices.numel(), dtype=torch.float32)
    for start in range(0, query_indices.numel(), chunk_size):
        end = min(start + chunk_size, query_indices.numel())
        chunk_indices = query_indices[start:end]
        chunk = embeddings[chunk_indices]
        sims = torch.matmul(chunk, embeddings.T)
        row_idx = torch.arange(end - start)
        sims[row_idx, chunk_indices] = -float("inf")
        ranked_indices = torch.argsort(sims, dim=1, descending=True)
        query_labels = labels[chunk_indices].unsqueeze(1)
        ranked_labels = labels[ranked_indices]
        self_mask = ranked_indices.eq(chunk_indices.unsqueeze(1))
        matches = ranked_labels.eq(query_labels) & ~self_mask
        matches_f = matches.float()
        cumsum = matches_f.cumsum(dim=1)
        ranks = torch.arange(1, matches_f.size(1) + 1, dtype=torch.float32).unsqueeze(0)
        precision = cumsum / ranks
        match_counts = matches.sum(dim=1).float().clamp(min=1.0)
        ap = (precision * matches_f).sum(dim=1) / match_counts
        ap_scores[start:end] = ap.cpu()
    return query_indices.cpu(), ap_scores


def pca_whiten_embeddings(
    embeddings: torch.Tensor,
    eps: float = 1e-5,
    output_dim: int = 256,
) -> torch.Tensor:
    if embeddings.numel() == 0:
        return embeddings
    x = embeddings.float()
    mean = x.mean(dim=0, keepdim=True)
    x_centered = x - mean
    denom = max(1, x_centered.size(0) - 1)
    cov = (x_centered.T @ x_centered) / denom
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = torch.clamp(eigvals, min=0.0)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if output_dim > 0 and eigvecs.size(1) > output_dim:
        eigvals = eigvals[:output_dim]
        eigvecs = eigvecs[:, :output_dim]
    inv_sqrt = torch.diag(1.0 / torch.sqrt(eigvals + eps))
    whitening = eigvecs @ inv_sqrt
    return x_centered @ whitening



def pca_whiten_fit(
    embeddings: torch.Tensor,
    eps: float = 1e-5,
    output_dim: int = 256,
    chunk_size: int = 4096,
    save_cov_path: str | None = None,
):
    if embeddings.numel() == 0:
        raise ValueError("Cannot fit PCA whitening on empty embeddings")
    if embeddings.dim() != 2:
        raise ValueError(f"Expected 2D tensor [n_samples, d_model], got {embeddings.shape}")

    # Keep original dtype/device for returned params
    x = embeddings if embeddings.is_floating_point() else embeddings.float()
    n_samples, d_model = x.shape
    denom = max(1, n_samples - 1)

    # Mean in float64 for better numerical stability
    mean64 = x.mean(dim=0, keepdim=True, dtype=torch.float64)

    # Covariance accumulation in float64
    cov64 = torch.zeros((d_model, d_model), device=x.device, dtype=torch.float64)

    if n_samples <= chunk_size:
        xc = x.to(torch.float64) - mean64
        cov64 = xc.T @ xc
    else:
        iterator = tqdm(
            range(0, n_samples, chunk_size),
            desc="pca_whiten: covariance",
            unit="chunk",
            mininterval=5,
        )
        for start in iterator:
            end = min(start + chunk_size, n_samples)
            chunk = x[start:end].to(torch.float64) - mean64
            cov64 += chunk.T @ chunk

    cov64 = cov64 / denom

    # Move to CPU float64 for robust eigendecomposition
    cov_cpu = cov64.cpu()
    cov_cpu = 0.5 * (cov_cpu + cov_cpu.T)  # enforce symmetry

    if save_cov_path:
        torch.save(cov_cpu, save_cov_path)
        print(f"pca_whiten_fit | saved covariance to {save_cov_path}")

    try:
        eigvals, eigvecs = torch.linalg.eigh(cov_cpu)  # ascending
    except RuntimeError as exc:
        print(f"pca_whiten_fit | eigh failed ({exc}); falling back to SVD")
        u, s, _ = torch.linalg.svd(cov_cpu, full_matrices=False)
        eigvecs = u
        eigvals = s

    eigvals = torch.clamp(eigvals, min=0.0)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if output_dim is not None and output_dim > 0:
        k = min(output_dim, eigvecs.size(1))
        eigvals = eigvals[:k]
        eigvecs = eigvecs[:, :k]

    # whitening = V * diag(1 / sqrt(lambda + eps)) without materializing diag
    scales = torch.rsqrt(eigvals + eps)  # [k]
    whitening = eigvecs * scales.unsqueeze(0)  # [d_model, k]

    whitening = whitening.to(dtype=x.dtype, device=x.device)
    mean = mean64.to(dtype=x.dtype, device=x.device)
    return mean, whitening


def pca_whiten_apply(embeddings: torch.Tensor, mean: torch.Tensor, whitening: torch.Tensor) -> torch.Tensor:
    x = embeddings.float()
    mean = mean.to(dtype=x.dtype, device=x.device)
    whitening = whitening.to(dtype=x.dtype, device=x.device)
    x_centered = x - mean
    return x_centered @ whitening


def evaluate(
    model,
    arcface,
    loader,
    device: torch.device,
    ce_weight: float,
    arcface_weight: float,
    triplet_weight: float,
    triplet_margin: float,
    triplet_soft_margin: bool,
    triplet_semi_hard: bool,
    log_interval: int,
    split_options: list[int],
    writer=None,
    tb_step: Optional[int] = None,
    log_decoder_attention: bool = False,
    annotate_decoder_attention: bool = False,
) -> Dict[str, float]:
    model.eval()
    mean, std = _loader_norm_stats(loader)
    running_loss = 0.0
    running_arcface = 0.0
    running_triplet = 0.0
    running_ce = 0.0
    running_acc = 0.0
    running_top5 = 0.0
    running_top10 = 0.0
    running_top50 = 0.0
    start_time = time.perf_counter()

    with torch.no_grad():
        for step, (patches, labels) in enumerate(loader, start=1):
            patches, labels, _ = _split_patches_per_batch(
                patches, labels, split_options, True
            )
            patches = patches.to(device)
            patches_raw = patches
            patches = normalize_patch_tensor(patches_raw, mean, std)
            labels = labels.to(device)

            if (
                log_decoder_attention
                and writer is not None
                and tb_step is not None
                and step == 1
            ):
                attn_img = render_decoder_cross_attention_image(
                    model,
                    patches_raw,
                    sample_index=0,
                    annotate=annotate_decoder_attention,
                    mean=mean,
                    std=std,
                )
                if attn_img is not None:
                    writer.add_image(
                        "val/decoder_cross_attention_sample0",
                        attn_img,
                        tb_step,
                    )

            logits, cls_embed, decoded = model(patches)
            ce_loss = torch.tensor(0.0, device=device)
            if logits is not None and ce_weight > 0.0:
                ce_loss = F.cross_entropy(logits, labels)
            arcface_loss = torch.tensor(0.0, device=device)
            arcface_logits = None
            if arcface_weight > 0.0:
                arcface_loss, arcface_logits = arcface(cls_embed, labels)
            triplet_loss = torch.tensor(0.0, device=device)
            if triplet_weight > 0.0:
                triplet_embed = _triplet_embeddings(cls_embed, decoded, model)
                if triplet_semi_hard:
                    triplet_loss = batch_semi_hard_triplet_loss(
                        triplet_embed,
                        labels,
                        triplet_margin,
                        triplet_soft_margin,
                    )
                elif triplet_soft_margin:
                    triplet_loss = batch_hard_triplet_soft_margin_loss(
                        triplet_embed, labels
                    )
                else:
                    triplet_loss = batch_hard_triplet_loss(
                        triplet_embed, labels, triplet_margin
                    )
            loss = (
                ce_weight * ce_loss
                + arcface_weight * arcface_loss
                + triplet_weight * triplet_loss
            )

            metric_logits = logits
            if ce_weight == 0.0 and arcface_weight > 0.0:
                metric_logits = arcface.raw_logits(cls_embed)
            acc = accuracy_from_logits(metric_logits, labels) if metric_logits is not None else 0.0
            if metric_logits is not None:
                top5 = accuracy_topk(metric_logits, labels, k=5)
                top10 = accuracy_topk(metric_logits, labels, k=10)
                top50 = accuracy_topk(metric_logits, labels, k=50)
            else:
                top5 = 0.0
                top10 = 0.0
                top50 = 0.0
            running_loss += loss.item()
            running_arcface += arcface_loss.item()
            running_triplet += triplet_loss.item()
            running_ce += ce_loss.item()
            running_acc += acc
            running_top5 += top5
            running_top10 += top10
            running_top50 += top50
            if log_interval > 0 and step % log_interval == 0:
                avg_loss = running_loss / step
                avg_arcface = running_arcface / step
                avg_triplet = running_triplet / step
                avg_ce = running_ce / step
                avg_acc = running_acc / step
                avg_top5 = running_top5 / step
                avg_top10 = running_top10 / step
                avg_top50 = running_top50 / step
                elapsed = time.perf_counter() - start_time
                steps_per_sec = step / max(elapsed, 1e-6)
                remaining = len(loader) - step
                eta_sec = remaining / max(steps_per_sec, 1e-6)
                mem_msg = ""
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                    mem_msg = f" | MEM: {mem_alloc:.0f} / {mem_reserved:.0f}MB"
                loss_msg = f"loss: {avg_loss:.2f}"
                top_msg = (
                    f"top1: {avg_acc * 100:.2f} top10: {avg_top10 * 100:.2f} top50: {avg_top50 * 100:.2f}"
                )
                weighted_ce = avg_ce * ce_weight
                weighted_arcface = avg_arcface * arcface_weight
                weighted_triplet = avg_triplet * triplet_weight
                ce_msg = f" | CE: {weighted_ce:.2f}" if ce_weight > 0.0 else ""
                arc_msg = f" | ArcFace: {weighted_arcface:.2f}" if arcface_weight > 0.0 else ""
                trip_msg = f" | Triplet: {weighted_triplet:.2f}" if triplet_weight > 0.0 else ""
                print(
                    f"[val] {step}/{len(loader)} | "
                    f"{loss_msg} {top_msg}"
                    f"{ce_msg}{arc_msg}{trip_msg} | "
                    f"ETA: {eta_sec/60:.1f}m"
                    f"{mem_msg}"
                )

    metrics = {
        "loss": running_loss / max(1, len(loader)),
        "arcface_loss": running_arcface / max(1, len(loader)),
        "triplet_loss": running_triplet / max(1, len(loader)),
        "acc": running_acc / max(1, len(loader)),
        "top5": running_top5 / max(1, len(loader)),
        "top10": running_top10 / max(1, len(loader)),
        "top50": running_top50 / max(1, len(loader)),
    }
    return metrics


def _maybe_tqdm(iterable, show_progress: bool, **kwargs):
    if show_progress:
        return tqdm(iterable, **kwargs)
    return iterable


def _loader_norm_stats(loader) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = getattr(loader, "dataset", None)
    if dataset is not None and hasattr(dataset, "mean") and hasattr(dataset, "std"):
        return dataset.mean, dataset.std
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    return mean, std


def extract_embeddings_(
    model,
    loader,
    device: torch.device,
    use_amp: bool,
    feature_type: str,
    include_queries: bool,
    split_options: list[int],
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    mean, std = _loader_norm_stats(loader)
    all_embeddings = []
    all_labels = []
    total_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (patches, labels) in enumerate(_maybe_tqdm(
            loader, show_progress, desc="extract", unit="batch"
        ), start=1):
            patches, labels, _ = _split_patches_per_batch(
                patches, labels, split_options, False
            )
            patches = patches.to(device)
            patches = normalize_patch_tensor(patches, mean, std)
            with torch.amp.autocast("cuda", enabled=use_amp):
                embeddings = model.encode_embeddings(
                    patches, feature_type=feature_type, include_queries=include_queries
                )
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
            if progress_callback is not None:
                progress_callback(batch_idx, total_batches)
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def extract_embeddings_with_splits(
    model,
    loader,
    device: torch.device,
    use_amp: bool,
    feature_type: str,
    include_queries: bool,
    split_options: list[int],
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    mean, std = _loader_norm_stats(loader)
    all_embeddings = []
    all_labels = []
    unique_splits = sorted({int(s) for s in split_options if int(s) > 0})
    if not unique_splits:
        return extract_embeddings_(
            model,
            loader,
            device,
            use_amp,
            feature_type,
            include_queries,
            split_options,
            show_progress,
        )
    with torch.no_grad():
        for patches, labels in _maybe_tqdm(
            loader, show_progress, mininterval=5, desc="extract", unit="batch"
        ):
            batch_size = patches.size(0)
            patches = patches.to(device)
            embed_sum = None
            for split in unique_splits:
                split_patches, _, applied = _split_patches_per_batch(
                    patches, labels, [split], True
                )
                split_patches = normalize_patch_tensor(split_patches, mean, std)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    embeddings = model.encode_embeddings(
                        split_patches,
                        feature_type=feature_type,
                        include_queries=include_queries,
                    )
                if applied > 1:
                    embeddings = embeddings.view(batch_size, applied, -1).mean(dim=1)
                embed_sum = embeddings if embed_sum is None else embed_sum + embeddings
            embeddings = embed_sum / max(1, len(unique_splits))
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def extract_embeddings_for_split(
    model,
    loader,
    device: torch.device,
    use_amp: bool,
    feature_type: str,
    include_queries: bool,
    split: int,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    mean, std = _loader_norm_stats(loader)
    all_embeddings = []
    all_labels = []
    split = max(1, int(split))
    with torch.no_grad():
        for patches, labels in _maybe_tqdm(
            loader, show_progress, mininterval=5, desc="extract", unit="batch"
        ):
            batch_size = patches.size(0)
            patches = patches.to(device)
            split_patches, _, applied = _split_patches_per_batch(
                patches, labels, [split], True
            )
            split_patches = normalize_patch_tensor(split_patches, mean, std)
            with torch.amp.autocast("cuda", enabled=use_amp):
                embeddings = model.encode_embeddings(
                    split_patches,
                    feature_type=feature_type,
                    include_queries=include_queries,
                )
            if applied > 1:
                embeddings = embeddings.view(batch_size, applied, -1).mean(dim=1)
            all_embeddings.append(embeddings.detach().cpu())
            all_labels.append(labels.detach().cpu())
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def extract_embeddings(
    model,
    loader,
    device: torch.device,
    use_amp: bool,
    feature_type: str,
    show_progress: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    mean, std = _loader_norm_stats(loader)
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for patches, labels in _maybe_tqdm(
            loader, show_progress, desc="extract", unit="batch"
        ):
            patches = patches.to(device)
            patches = normalize_patch_tensor(patches, mean, std)
            with torch.amp.autocast("cuda", enabled=use_amp):
                _, cls_embed, decoded = model(patches)
            if feature_type == "query":
                query_embed = decoded[:, 1:, :].reshape(decoded.size(0), -1)
                all_embeddings.append(query_embed.detach().cpu())
            else:
                all_embeddings.append(cls_embed.detach().cpu())
            all_labels.append(labels.detach().cpu())
    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def extract_embeddings_full_page(
    model,
    dataset: PatchDataset,
    device: torch.device,
    use_amp: bool,
    feature_type: str,
    include_queries: bool,
    show_progress: bool = True,
    val_gray: bool = False,
    resize_height: int = 0,
    resize_width: int = 0,
    batch_size: int = 1,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    cache_dir: str = "",
    cache_prefix: str = "val",
    cache_chunk_size: int = 0,
    recompute_cache: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    print(
        f"extract_full_page | samples={len(dataset.samples)} | "
        f"feature_type={feature_type} | include_queries={include_queries}"
    )
    if mean is None:
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    if std is None:
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)
    printed_first = False
    batch_size = max(1, int(batch_size))

    def _parse_chunk_name(name: str) -> Optional[Tuple[int, int]]:
        if not (name.startswith("chunk_") and name.endswith(".pt")):
            return None
        body = name[len("chunk_") : -len(".pt")]
        parts = body.split("_")
        if len(parts) != 2:
            return None
        try:
            start = int(parts[0])
            end = int(parts[1])
        except ValueError:
            return None
        if start < 0 or end <= start:
            return None
        return start, end

    def _load_chunks(chunk_paths: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_list: list[torch.Tensor] = []
        lbl_list: list[torch.Tensor] = []
        for path in chunk_paths:
            payload = torch.load(path, map_location="cpu")
            embeddings = payload.get("embeddings")
            labels = payload.get("labels")
            if embeddings is None or labels is None:
                continue
            if embeddings.numel() == 0:
                continue
            emb_list.append(embeddings)
            lbl_list.append(labels)
        if not emb_list:
            raise RuntimeError("No full-page embeddings could be extracted.")
        return torch.cat(emb_list, dim=0), torch.cat(lbl_list, dim=0)

    total = len(dataset.samples)
    resume_from = 0
    chunk_paths: list[str] = []
    chunk_dir = ""
    cache_chunk_size = max(0, int(cache_chunk_size))
    if cache_dir and cache_chunk_size > 0:
        chunk_dir = os.path.join(cache_dir, f"{cache_prefix}_emb_chunks")
        if recompute_cache and os.path.isdir(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)
        chunks: list[Tuple[int, int, str]] = []
        for fname in os.listdir(chunk_dir):
            parsed = _parse_chunk_name(fname)
            if parsed is None:
                continue
            start, end = parsed
            chunks.append((start, end, os.path.join(chunk_dir, fname)))
        chunks.sort(key=lambda item: item[0])
        expected = 0
        for start, end, path in chunks:
            if start != expected:
                break
            chunk_paths.append(path)
            expected = end
        resume_from = min(expected, total)
        if resume_from > 0:
            print(f"extract_full_page | resume {resume_from}/{total} from {chunk_dir}")
            if progress_callback is not None:
                progress_callback(resume_from, total)
        if resume_from >= total:
            # All indices processed; reuse cached chunks.
            return _load_chunks(chunk_paths)

    iterator = range(resume_from, total)
    if show_progress:
        iterator = tqdm(iterator, desc="extract_full_page", unit="img")

    batch_imgs: list[torch.Tensor] = []
    batch_lbls: list[int] = []
    batch_indices: list[int] = []

    chunk_start = resume_from
    chunk_embs: list[torch.Tensor] = []
    chunk_labels: list[int] = []
    chunk_kept_indices: list[int] = []

    def _flush_batch() -> None:
        nonlocal printed_first
        if not batch_imgs:
            return
        # Ensure all batch items have the same spatial size; pad to max if needed.
        heights = [t.size(-2) for t in batch_imgs]
        widths = [t.size(-1) for t in batch_imgs]
        max_h = max(heights)
        max_w = max(widths)
        padded = []
        for t in batch_imgs:
            pad_h = max_h - t.size(-2)
            pad_w = max_w - t.size(-1)
            if pad_h > 0 or pad_w > 0:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
            padded.append(t)
        images = torch.cat(padded, dim=0).to(device)
        images = normalize_patch_tensor(images, mean, std)
        with torch.amp.autocast("cuda", enabled=use_amp):
            embedding = model.encode_full_image_embeddings(
                images,
                feature_type=feature_type,
                include_queries=include_queries,
                pre_pool_stride=1,
            )
        if not printed_first:
            first_path = _resolve_dataset_image_path(dataset, dataset.samples[batch_indices[0]])
            print(
                "extract_full_page first batch | "
                f"path={first_path} | image_tensor={tuple(images.shape)} | "
                f"embedding={tuple(embedding.shape)}"
            )
            printed_first = True
        chunk_embs.append(embedding.detach().cpu())
        chunk_labels.extend(batch_lbls)
        chunk_kept_indices.extend(batch_indices)
        batch_imgs.clear()
        batch_lbls.clear()
        batch_indices.clear()

    def _flush_chunk(end_idx_exclusive: int) -> None:
        nonlocal chunk_start
        if not chunk_dir:
            return
        os.makedirs(chunk_dir, exist_ok=True)
        emb = torch.cat(chunk_embs, dim=0) if chunk_embs else torch.empty((0, 0))
        labels = torch.tensor(chunk_labels, dtype=torch.long)
        indices = torch.tensor(chunk_kept_indices, dtype=torch.int32)
        out_path = os.path.join(chunk_dir, f"chunk_{chunk_start:08d}_{end_idx_exclusive:08d}.pt")
        torch.save(
            {"start": int(chunk_start), "end": int(end_idx_exclusive), "indices": indices, "labels": labels, "embeddings": emb},
            out_path,
        )
        chunk_paths.append(out_path)
        chunk_start = end_idx_exclusive
        chunk_embs.clear()
        chunk_labels.clear()
        chunk_kept_indices.clear()

    with torch.no_grad():
        for idx in iterator:
            image_path = _resolve_dataset_image_path(dataset, dataset.samples[idx])
            if not os.path.exists(image_path):
                if chunk_dir and cache_chunk_size > 0 and (idx + 1 - chunk_start) >= cache_chunk_size:
                    _flush_batch()
                    _flush_chunk(idx + 1)
                if progress_callback is not None:
                    progress_callback(idx + 1, total)
                continue
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                if chunk_dir and cache_chunk_size > 0 and (idx + 1 - chunk_start) >= cache_chunk_size:
                    _flush_batch()
                    _flush_chunk(idx + 1)
                if progress_callback is not None:
                    progress_callback(idx + 1, total)
                continue
            if resize_height > 0 and resize_width > 0:
                resampling = Image.Resampling if hasattr(Image, "Resampling") else Image
                image = image.resize((int(resize_width), int(resize_height)), resampling.BILINEAR)
            image_np = np.asarray(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().unsqueeze(0)
            if val_gray:
                gray = image_tensor.mean(dim=1, keepdim=True)
                image_tensor = gray.repeat(1, 3, 1, 1)
            batch_imgs.append(image_tensor)
            batch_lbls.append(int(dataset.samples[idx][2]))
            batch_indices.append(idx)
            if len(batch_imgs) >= batch_size:
                _flush_batch()
            if chunk_dir and cache_chunk_size > 0 and (idx + 1 - chunk_start) >= cache_chunk_size:
                _flush_batch()
                _flush_chunk(idx + 1)
            if progress_callback is not None:
                progress_callback(idx + 1, total)

        _flush_batch()
        if chunk_dir and cache_chunk_size > 0 and chunk_start < total:
            _flush_chunk(total)

    if chunk_dir and cache_chunk_size > 0:
        return _load_chunks(chunk_paths)

    # Non-cached fallback: concatenate in-memory.
    if not chunk_embs:
        raise RuntimeError("No full-page embeddings could be extracted.")
    embeddings = torch.cat(chunk_embs, dim=0)
    labels = torch.tensor(chunk_labels, dtype=torch.long)
    return embeddings, labels


def visualize_full_page_keep_heatmaps(
    model,
    loader,
    dataset: PatchDataset,
    device: torch.device,
    output_dir: str,
    use_amp: bool,
    val_gray: bool = False,
    pre_pool_stride: int = 1,
    resize_height: int = 0,
    resize_width: int = 0,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
) -> int:
    """Visualize X-VLAD per-head assignment maps (R=keep, G=ghost1, B=ghost2)."""
    if loader is None:
        raise ValueError("Loader is required for full-page keep heatmap visualization")
    if getattr(model, "decoder_type", "") != "xvlad":
        raise ValueError("Full-page keep heatmap visualization currently supports decoder_type=xvlad only")
    xvlad = getattr(model.decoder, "decoder", None)
    if xvlad is None or not hasattr(xvlad, "assignment_heads") or not hasattr(xvlad, "head_projections"):
        raise ValueError("X-VLAD assignment heads are not available")

    os.makedirs(output_dir, exist_ok=True)
    first_batch = next(iter(loader))
    batch_size = int(first_batch[0].size(0))
    if batch_size <= 0:
        return 0

    if mean is None:
        mean = getattr(dataset, "mean", None)
    if std is None:
        std = getattr(dataset, "std", None)
    if mean is None:
        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1)
    if std is None:
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1)

    image_tensors: list[torch.Tensor] = []
    image_np_list: list[np.ndarray] = []
    image_paths: list[str] = []
    max_count = min(batch_size, len(dataset.samples))
    for idx in range(max_count):
        image_path = _resolve_dataset_image_path(dataset, dataset.samples[idx])
        if not os.path.exists(image_path):
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue
        if resize_height > 0 and resize_width > 0:
            resampling = Image.Resampling if hasattr(Image, "Resampling") else Image
            image = image.resize((int(resize_width), int(resize_height)), resampling.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().unsqueeze(0)
        if val_gray:
            gray = image_tensor.mean(dim=1, keepdim=True)
            image_tensor = gray.repeat(1, 3, 1, 1)
        image_tensors.append(image_tensor)
        image_np_list.append(image_np)
        image_paths.append(image_path)

    if not image_tensors:
        return 0

    saved = 0
    num_heads = len(xvlad.assignment_heads)
    map_debug = "unknown"
    norm_color = np.array([1.0, 0.82, 0.05], dtype=np.float32)
    keep_color = np.array([0.0, 0.42, 0.0], dtype=np.float32)
    reject_color = np.array([0.92, 0.08, 0.08], dtype=np.float32)

    def _robust_unit(x: np.ndarray, q: float = 99.0) -> np.ndarray:
        hi = float(np.percentile(x, q))
        if hi <= 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return np.clip(x / hi, 0.0, 1.0).astype(np.float32)

    print(
        f"Full-page keep heatmap debug | preparing to write overlays for "
        f"{len(image_tensors)} images to {output_dir}"
    )
    model.eval()
    with torch.no_grad():
        for b, image_tensor in enumerate(
            tqdm(image_tensors, desc="full_page_keep_heatmaps", unit="img")
        ):
            image_np = image_np_list[b]
            img_h, img_w = image_np.shape[:2]
            stem = os.path.splitext(os.path.basename(image_paths[b]))[0]
            sample_dir = os.path.join(output_dir, f"{b:02d}_{stem}")
            os.makedirs(sample_dir, exist_ok=True)

            image_tensor = image_tensor.to(device)
            image_tensor = normalize_patch_tensor(image_tensor, mean, std)

            with torch.amp.autocast("cuda", enabled=use_amp):
                if (
                    getattr(model, "patch_size", 0) > 0
                    and image_tensor.size(-2) != model.patch_size
                    and image_tensor.size(-1) != model.patch_size
                    and image_tensor.size(-2) % model.patch_size == 0
                    and image_tensor.size(-1) % model.patch_size == 0
                ):
                    grid_h = image_tensor.size(-2) // model.patch_size
                    grid_w = image_tensor.size(-1) // model.patch_size
                    windows = (
                        image_tensor.unfold(2, model.patch_size, model.patch_size)
                        .unfold(3, model.patch_size, model.patch_size)
                        .permute(0, 2, 3, 1, 4, 5)
                        .reshape(grid_h * grid_w, image_tensor.size(1), model.patch_size, model.patch_size)
                    )
                    features = model._run_backbone(windows)
                    if features.dim() == 4:
                        if pre_pool_stride > 1:
                            features = F.max_pool2d(
                                features,
                                kernel_size=pre_pool_stride,
                                stride=pre_pool_stride,
                            )
                        local_h, local_w = int(features.size(-2)), int(features.size(-1))
                        tokens = features.flatten(2).transpose(1, 2)
                    elif features.dim() == 3:
                        if features.size(1) > 1 and hasattr(model.patch_embedding, "cls_token"):
                            features = features[:, 1:, :]
                        tokens = features
                        token_count = int(tokens.size(1))
                        local_h = int(round(math.sqrt(token_count)))
                        local_w = local_h
                        if local_h * local_w != token_count:
                            local_h = 1
                            local_w = token_count
                    else:
                        raise ValueError(f"Unexpected encoder output shape: {features.shape}")
                    tokens = model.backbone_dropout(tokens)
                    tokens = model.patch_proj(tokens)
                    window_embeddings = model.output_proj(tokens)
                    embeddings = (
                        window_embeddings.view(grid_h, grid_w, local_h, local_w, -1)
                        .permute(0, 2, 1, 3, 4)
                        .reshape(1, grid_h * local_h * grid_w * local_w, -1)
                    )
                    map_h = grid_h * local_h
                    map_w = grid_w * local_w
                    if getattr(model, "patch_encoder", None) is not None:
                        embeddings = model.patch_encoder(embeddings)
                else:
                    features = model._run_backbone(image_tensor)
                    if features.dim() == 4:
                        if pre_pool_stride > 1:
                            features = F.max_pool2d(
                                features,
                                kernel_size=pre_pool_stride,
                                stride=pre_pool_stride,
                            )
                        map_h, map_w = int(features.size(-2)), int(features.size(-1))
                        tokens = features.flatten(2).transpose(1, 2)
                    elif features.dim() == 3:
                        if features.size(1) > 1 and hasattr(model.patch_embedding, "cls_token"):
                            tokens = features[:, 1:, :]
                        else:
                            tokens = features
                        token_count = int(tokens.size(1))
                        map_h = int(round(math.sqrt(token_count)))
                        map_w = map_h
                        if map_h * map_w != token_count:
                            map_h = 1
                            map_w = token_count
                    else:
                        raise ValueError(f"Unexpected encoder output shape: {features.shape}")

                    tokens = model.backbone_dropout(tokens)
                    tokens = model.patch_proj(tokens)
                    embeddings = model.output_proj(tokens)
                token_norm = torch.norm(embeddings, p=2, dim=2)  # (1, N)
                token_norm = token_norm / token_norm.max(dim=1, keepdim=True).values.clamp(min=1e-8)

                num_clusters = int(getattr(xvlad, "num_clusters", 1))
                ghost_clusters = int(getattr(xvlad, "ghost_clusters", 0))
                per_head_keep_reject = []
                per_head_keep_projnorm = []
                for head_index in range(len(xvlad.assignment_heads)):
                    projected = xvlad.head_projections[head_index](embeddings)
                    assignments = torch.softmax(
                        xvlad.assignment_heads[head_index](projected), dim=-1
                    )
                    keep = assignments[:, :, :num_clusters].sum(dim=2)  # (1, N)
                    if ghost_clusters > 0 and assignments.size(-1) > num_clusters:
                        reject = assignments[:, :, num_clusters:].sum(dim=2)
                    else:
                        reject = (1.0 - keep).clamp(min=0.0, max=1.0)
                    keep_reject_scores = (
                        torch.stack([keep, reject], dim=-1) * token_norm.unsqueeze(-1)
                    )
                    per_head_keep_reject.append(keep_reject_scores)
                    projected_norm = torch.norm(projected, p=2, dim=2)
                    projected_norm = projected_norm / projected_norm.max(dim=1, keepdim=True).values.clamp(min=1e-8)
                    keep_projnorm = keep * projected_norm
                    per_head_keep_projnorm.append(keep_projnorm)
                per_head_scores = (
                    torch.stack(per_head_keep_reject, dim=1).detach().cpu().squeeze(0)
                )  # (H, N, 2)
                per_head_keep_projnorm_scores = (
                    torch.stack(per_head_keep_projnorm, dim=1).detach().cpu().squeeze(0)
                )  # (H, N)
                token_norm_cpu = token_norm.detach().cpu().squeeze(0)  # (N,)

            map_debug = f"{map_h}x{map_w}"
            panel_norm_raw = None
            panel_norm_overlay = None
            panel_assign_raw = None
            panel_assign_overlay = None
            panel_keep_raw = None
            panel_keep_overlay = None
            if token_norm_cpu.numel() == map_h * map_w:
                norm_small = token_norm_cpu.view(map_h, map_w).unsqueeze(0).unsqueeze(0)
                norm_up = F.interpolate(
                    norm_small,
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()
                norm_min = float(norm_up.min())
                norm_max = float(norm_up.max())
                if norm_max > norm_min:
                    norm_vis = (norm_up - norm_min) / (norm_max - norm_min)
                else:
                    norm_vis = np.zeros_like(norm_up, dtype=np.float32)
                norm_rgb = np.zeros_like(image_np)
                norm_rgb[..., 0] = norm_color[0] * norm_vis
                norm_rgb[..., 1] = norm_color[1] * norm_vis
                norm_rgb[..., 2] = norm_color[2] * norm_vis
                norm_alpha = 0.55 * norm_vis[..., None]
                norm_overlay = image_np * (1.0 - norm_alpha) + norm_rgb * norm_alpha
                norm_overlay = np.clip(norm_overlay, 0.0, 1.0)
                norm_img = Image.fromarray((norm_vis * 255.0).astype(np.uint8), mode="L")
                norm_overlay_img = Image.fromarray((norm_overlay * 255.0).astype(np.uint8), mode="RGB")
                norm_img.save(os.path.join(sample_dir, "token_norm_heat.png"))
                norm_overlay_img.save(os.path.join(sample_dir, "token_norm_overlay.png"))
                panel_norm_raw = norm_rgb
                panel_norm_overlay = norm_overlay

            # Aggregate over heads for a compact summary view.
            if per_head_scores.size(0) > 0:
                agg_keep_reject_vec = per_head_scores.mean(dim=0)  # (N, 2)
                if agg_keep_reject_vec.size(0) == map_h * map_w:
                    agg_kr_small = agg_keep_reject_vec.view(map_h, map_w, 2).permute(2, 0, 1).unsqueeze(0)
                    agg_kr_up = F.interpolate(
                        agg_kr_small,
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0).permute(1, 2, 0).numpy()
                    agg_kr_up = np.clip(agg_kr_up, 0.0, 1.0)
                    agg_keep_vis = _robust_unit(agg_kr_up[..., 0], q=99.0)
                    agg_reject_vis = _robust_unit(agg_kr_up[..., 1], q=99.0)
                    agg_keep_vis = np.clip(1.25 * agg_keep_vis, 0.0, 1.0)
                    agg_heat_rgb = (
                        agg_keep_vis[..., None] * keep_color[None, None, :]
                        + agg_reject_vis[..., None] * reject_color[None, None, :]
                    )
                    agg_heat_rgb = np.clip(agg_heat_rgb, 0.0, 1.0)
                    agg_strength = np.maximum(agg_keep_vis, agg_reject_vis)[..., None]
                    agg_alpha = np.clip(0.1 + 0.55 * np.sqrt(agg_strength) + 0.2 * agg_keep_vis[..., None], 0.0, 0.85)
                    agg_overlay = image_np * (1.0 - agg_alpha) + agg_heat_rgb * agg_alpha
                    agg_overlay = np.clip(agg_overlay, 0.0, 1.0)
                    agg_img = Image.fromarray((agg_heat_rgb * 255.0).astype(np.uint8), mode="RGB")
                    agg_overlay_img = Image.fromarray((agg_overlay * 255.0).astype(np.uint8), mode="RGB")
                    agg_img.save(os.path.join(sample_dir, "heads_aggregate_assignment_rgb.png"))
                    agg_overlay_img.save(os.path.join(sample_dir, "heads_aggregate_assignment_overlay.png"))
                    panel_assign_raw = agg_heat_rgb
                    panel_assign_overlay = agg_overlay
                    keep_only_rgb = agg_keep_vis[..., None] * keep_color[None, None, :]
                    keep_only_rgb = np.clip(keep_only_rgb, 0.0, 1.0)
                    keep_only_alpha = np.clip(0.1 + 0.65 * np.sqrt(agg_keep_vis[..., None]), 0.0, 0.85)
                    keep_only_overlay = image_np * (1.0 - keep_only_alpha) + keep_only_rgb * keep_only_alpha
                    keep_only_overlay = np.clip(keep_only_overlay, 0.0, 1.0)
                    panel_keep_raw = keep_only_rgb
                    panel_keep_overlay = keep_only_overlay

            if per_head_keep_projnorm_scores.size(0) > 0:
                agg_keep_proj_vec = per_head_keep_projnorm_scores.mean(dim=0)  # (N,)
                if agg_keep_proj_vec.numel() == map_h * map_w:
                    agg_keep_proj_small = agg_keep_proj_vec.view(map_h, map_w).unsqueeze(0).unsqueeze(0)
                    agg_keep_proj_up = F.interpolate(
                        agg_keep_proj_small,
                        size=(img_h, img_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze().numpy()
                    akp_min = float(agg_keep_proj_up.min())
                    akp_max = float(agg_keep_proj_up.max())
                    if akp_max > akp_min:
                        agg_keep_proj_vis = (agg_keep_proj_up - akp_min) / (akp_max - akp_min)
                    else:
                        agg_keep_proj_vis = np.zeros_like(agg_keep_proj_up, dtype=np.float32)
                    agg_keep_proj_rgb = np.zeros_like(image_np)
                    agg_keep_proj_rgb[..., 1] = 0.42 * agg_keep_proj_vis
                    agg_keep_proj_alpha = np.clip(
                        0.1 + 0.65 * np.sqrt(agg_keep_proj_vis[..., None]), 0.0, 0.85
                    )
                    agg_keep_proj_overlay = (
                        image_np * (1.0 - agg_keep_proj_alpha) + agg_keep_proj_rgb * agg_keep_proj_alpha
                    )
                    agg_keep_proj_overlay = np.clip(agg_keep_proj_overlay, 0.0, 1.0)
                    agg_keep_proj_img = Image.fromarray((agg_keep_proj_vis * 255.0).astype(np.uint8), mode="L")
                    agg_keep_proj_overlay_img = Image.fromarray(
                        (agg_keep_proj_overlay * 255.0).astype(np.uint8), mode="RGB"
                    )
                    agg_keep_proj_img.save(os.path.join(sample_dir, "heads_aggregate_keep_projnorm_heat.png"))
                    agg_keep_proj_overlay_img.save(
                        os.path.join(sample_dir, "heads_aggregate_keep_projnorm_overlay.png")
                    )

            # One joint overview panel (2 rows x 4 columns):
            # top: blank | norm raw | assignment*norm raw | keep*norm raw
            # bottom: image | norm overlay | assignment*norm overlay | keep*norm overlay
            if (
                panel_norm_raw is not None
                and panel_norm_overlay is not None
                and panel_assign_raw is not None
                and panel_assign_overlay is not None
                and panel_keep_raw is not None
                and panel_keep_overlay is not None
            ):
                blank = np.ones_like(image_np, dtype=np.float32)
                top_tiles = [blank, panel_norm_raw, panel_assign_raw, panel_keep_raw]
                bottom_tiles = [image_np, panel_norm_overlay, panel_assign_overlay, panel_keep_overlay]
                top_row = np.concatenate(top_tiles, axis=1)
                bottom_row = np.concatenate(bottom_tiles, axis=1)
                panel = np.concatenate([top_row, bottom_row], axis=0)
                panel = np.clip(panel, 0.0, 1.0)
                panel_img = Image.fromarray((panel * 255.0).astype(np.uint8), mode="RGB")

                tile_w = int(image_np.shape[1])
                panel_w = int(panel_img.width)
                header_h = max(150, int(round(0.12 * tile_w)))
                full_panel = Image.new("RGB", (panel_w, panel_img.height + header_h), color=(255, 255, 255))
                full_panel.paste(panel_img, (0, header_h))
                draw = ImageDraw.Draw(full_panel)
                font_size = max(72, int(round(0.07 * tile_w)))
                try:
                    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
                except Exception:
                    font = ImageFont.load_default()
                headers = [
                    "Image",
                    "Patch Norm",
                    "Assignment (aggregate)",
                    "Keep (aggregate)",
                ]
                for col, title in enumerate(headers):
                    x0 = col * tile_w
                    x1 = x0 + tile_w
                    cx = (x0 + x1) // 2
                    cy = header_h // 2
                    bbox = draw.textbbox((0, 0), title, font=font)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    draw.text((cx - tw // 2, cy - th // 2), title, fill=(20, 20, 20), font=font)

                panel_img = full_panel
                panel_img.save(os.path.join(sample_dir, "summary_panel.jpg"), quality=95)
            for h in range(per_head_scores.size(0)):
                vec = per_head_scores[h]
                if vec.size(0) != map_h * map_w:
                    continue
                kr_small = vec.view(map_h, map_w, 2).permute(2, 0, 1).unsqueeze(0)  # (1,2,h,w)
                kr_up = F.interpolate(
                    kr_small,
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).permute(1, 2, 0).numpy()
                kr_up = np.clip(kr_up, 0.0, 1.0)
                keep_vis = _robust_unit(kr_up[..., 0], q=99.0)
                reject_vis = _robust_unit(kr_up[..., 1], q=99.0)
                keep_vis = np.clip(1.25 * keep_vis, 0.0, 1.0)
                heat_rgb = (
                    keep_vis[..., None] * keep_color[None, None, :]
                    + reject_vis[..., None] * reject_color[None, None, :]
                )
                heat_rgb = np.clip(heat_rgb, 0.0, 1.0)
                strength = np.maximum(keep_vis, reject_vis)[..., None]
                alpha = np.clip(0.1 + 0.55 * np.sqrt(strength) + 0.2 * keep_vis[..., None], 0.0, 0.85)
                overlay = image_np * (1.0 - alpha) + heat_rgb * alpha
                overlay = np.clip(overlay, 0.0, 1.0)

                heat_img = Image.fromarray((heat_rgb * 255.0).astype(np.uint8), mode="RGB")
                overlay_img = Image.fromarray((overlay * 255.0).astype(np.uint8), mode="RGB")
                heat_img.save(os.path.join(sample_dir, f"head_{h:02d}_assignment_rgb.png"))
                overlay_img.save(os.path.join(sample_dir, f"head_{h:02d}_overlay.png"))

                keep_proj_vec = per_head_keep_projnorm_scores[h]
                if keep_proj_vec.numel() != map_h * map_w:
                    continue
                keep_proj_small = keep_proj_vec.view(map_h, map_w).unsqueeze(0).unsqueeze(0)
                keep_proj_up = F.interpolate(
                    keep_proj_small,
                    size=(img_h, img_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()
                kp_min = float(keep_proj_up.min())
                kp_max = float(keep_proj_up.max())
                if kp_max > kp_min:
                    keep_proj_vis = (keep_proj_up - kp_min) / (kp_max - kp_min)
                else:
                    keep_proj_vis = np.zeros_like(keep_proj_up, dtype=np.float32)
                keep_proj_rgb = np.zeros_like(image_np)
                keep_proj_rgb[..., 1] = 0.42 * keep_proj_vis
                keep_proj_alpha = np.clip(0.1 + 0.65 * np.sqrt(keep_proj_vis[..., None]), 0.0, 0.85)
                keep_proj_overlay = image_np * (1.0 - keep_proj_alpha) + keep_proj_rgb * keep_proj_alpha
                keep_proj_overlay = np.clip(keep_proj_overlay, 0.0, 1.0)
                keep_proj_img = Image.fromarray((keep_proj_vis * 255.0).astype(np.uint8), mode="L")
                keep_proj_overlay_img = Image.fromarray((keep_proj_overlay * 255.0).astype(np.uint8), mode="RGB")
                keep_proj_img.save(os.path.join(sample_dir, f"head_{h:02d}_keep_projnorm_heat.png"))
                keep_proj_overlay_img.save(os.path.join(sample_dir, f"head_{h:02d}_keep_projnorm_overlay.png"))
                saved += 1
    print(
        f"Full-page keep heatmap debug | batch={len(image_tensors)} | "
        f"heads={num_heads} | token_map={map_debug} | saved={saved} overlays to {output_dir}"
    )
    return saved


def _query_mask_excluding_labels(
    labels: torch.Tensor, exclude_labels: list[int]
) -> Optional[torch.Tensor]:
    if not exclude_labels:
        return None
    exclude = torch.tensor(exclude_labels, device=labels.device, dtype=labels.dtype)
    return ~((labels.unsqueeze(1) == exclude.unsqueeze(0)).any(dim=1))


def _resolve_dataset_image_path(dataset: PatchDataset, sample: Tuple[str, str, int]) -> str:
    image_path = sample[0]
    if dataset.class_to_idx:
        return image_path
    if os.path.isabs(image_path) or not dataset.image_root:
        return image_path
    return os.path.join(dataset.image_root, image_path)


def _copy_indices_to_dir(
    indices: list[int], dataset: PatchDataset, output_dir: str
) -> int:
    if not output_dir or not indices:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    copied = 0
    for idx in indices:
        if idx < 0 or idx >= len(dataset.samples):
            continue
        image_path = _resolve_dataset_image_path(dataset, dataset.samples[idx])
        if not os.path.exists(image_path):
            continue
        base = os.path.basename(image_path)
        dest = os.path.join(output_dir, f"{idx:06d}_{base}")
        shutil.copy2(image_path, dest)
        copied += 1
    return copied


def _fit_image_to_tile(image: Image.Image, tile_width: int, tile_height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(tile_width / max(1, image.width), tile_height / max(1, image.height))
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resampling = Image.Resampling if hasattr(Image, "Resampling") else Image
    resized = image.resize((new_w, new_h), resampling.BICUBIC)
    tile = Image.new("RGB", (tile_width, tile_height), color=(245, 245, 245))
    x = (tile_width - new_w) // 2
    y = (tile_height - new_h) // 2
    tile.paste(resized, (x, y))
    return tile


def _draw_border(canvas: Image.Image, color: tuple[int, int, int], border: int) -> Image.Image:
    out = canvas.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for i in range(border):
        draw.rectangle((i, i, w - 1 - i, h - 1 - i), outline=color)
    return out


def _idx_to_writer_name(dataset: PatchDataset, label: int) -> str:
    if dataset.class_to_idx:
        idx_to_class = {idx: name for name, idx in dataset.class_to_idx.items()}
        return idx_to_class.get(int(label), str(label))
    return str(label)


def visualize_retrieval_topk_hits(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    dataset: PatchDataset,
    output_dir: str,
    k: int = 5,
    tile_width: int = 1000,
    tile_height: int = 1500,
    border_size: int = 10,
    group_size: int = 50,
    ap_threshold: float = 0.6,
) -> int:
    if embeddings.numel() == 0 or labels.numel() == 0:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    embeddings = F.normalize(embeddings, dim=1)
    labels = labels.to(dtype=torch.long)
    total = embeddings.size(0)
    k = max(1, min(int(k), max(1, total - 1)))
    gap = 12
    text_height = 260
    writer_cache = {int(i): _idx_to_writer_name(dataset, int(i)) for i in labels.unique().tolist()}
    saved = 0

    label_counts = torch.bincount(labels)
    valid_query_mask = label_counts[labels] > 1
    valid_queries = torch.nonzero(valid_query_mask, as_tuple=False).squeeze(1)
    if valid_queries.numel() == 0:
        return 0

    # Compute AP for all valid queries (leave-one-out).
    ap_scores = torch.zeros(total, dtype=torch.float32)
    chunk_size = 256
    for start in range(0, valid_queries.numel(), chunk_size):
        end = min(start + chunk_size, valid_queries.numel())
        chunk_indices = valid_queries[start:end]
        sims = torch.matmul(embeddings[chunk_indices], embeddings.T)
        row_idx = torch.arange(end - start)
        sims[row_idx, chunk_indices] = -float("inf")
        ranked = torch.argsort(sims, dim=1, descending=True)
        query_labels = labels[chunk_indices].unsqueeze(1)
        ranked_labels = labels[ranked]
        matches = ranked_labels.eq(query_labels)
        matches_f = matches.float()
        cumsum = matches_f.cumsum(dim=1)
        ranks = torch.arange(1, matches_f.size(1) + 1, dtype=torch.float32).unsqueeze(0)
        precision = cumsum / ranks
        match_counts = matches.sum(dim=1).float()
        ap = (precision * matches_f).sum(dim=1) / match_counts.clamp(min=1.0)
        ap_scores[chunk_indices] = ap

    valid_list = valid_queries.tolist()
    random_count = min(group_size, len(valid_list))
    random_group = random.sample(valid_list, k=random_count)
    below_thresh = [idx for idx in valid_list if float(ap_scores[idx].item()) < ap_threshold]
    below_thresh = sorted(below_thresh, key=lambda idx: float(ap_scores[idx].item()))
    below_count = min(group_size, len(below_thresh))
    below_group = below_thresh[:below_count]
    lowest_group = sorted(valid_list, key=lambda idx: float(ap_scores[idx].item()))[:random_count]

    selected_queries: list[tuple[str, int]] = (
        [("random", idx) for idx in random_group]
        + [("ap_lt_60", idx) for idx in below_group]
        + [("lowest_ap", idx) for idx in lowest_group]
    )

    for group_name, query_idx in selected_queries:
        query_label = int(labels[query_idx].item())
        sims = torch.matmul(embeddings[query_idx : query_idx + 1], embeddings.T).squeeze(0)
        sims[query_idx] = -float("inf")
        ranked_all = torch.argsort(sims, descending=True)
        relevant_ranks = (
            torch.nonzero(
                (labels[ranked_all] == query_label) & ranked_all.ne(query_idx),
                as_tuple=False,
            ).squeeze(1)
            + 1
        ).tolist()
        top_vals, top_idx = torch.topk(sims, k=k, largest=True)
        query_sample = dataset.samples[query_idx]
        query_path = _resolve_dataset_image_path(dataset, query_sample)
        if not os.path.exists(query_path):
            continue

        try:
            query_img = Image.open(query_path).convert("RGB")
        except Exception:
            continue
        tiles: list[Image.Image] = []
        query_tile = _fit_image_to_tile(query_img, tile_width, tile_height)
        query_tile = _draw_border(query_tile, (50, 120, 255), border_size)
        tiles.append(query_tile)

        hit_lines: list[str] = []
        for rank, (neighbor_idx, score) in enumerate(zip(top_idx.tolist(), top_vals.tolist()), start=1):
            if neighbor_idx < 0 or neighbor_idx >= total:
                continue
            hit_sample = dataset.samples[neighbor_idx]
            hit_path = _resolve_dataset_image_path(dataset, hit_sample)
            if not os.path.exists(hit_path):
                continue
            try:
                hit_img = Image.open(hit_path).convert("RGB")
            except Exception:
                continue
            hit_label = int(labels[neighbor_idx].item())
            correct = hit_label == query_label
            border_color = (25, 180, 60) if correct else (220, 30, 30)
            hit_tile = _fit_image_to_tile(hit_img, tile_width, tile_height)
            hit_tile = _draw_border(hit_tile, border_color, border_size)
            tiles.append(hit_tile)
            hit_lines.append(
                f"#{rank}: idx={neighbor_idx} sim={score:.4f} writer={writer_cache.get(hit_label, str(hit_label))} "
                f"{'correct' if correct else 'wrong'} file={os.path.basename(hit_path)}"
            )

        if len(tiles) <= 1:
            continue

        canvas_w = len(tiles) * tile_width + (len(tiles) - 1) * gap
        canvas_h = tile_height + text_height
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
        for i, tile in enumerate(tiles):
            x = i * (tile_width + gap)
            canvas.paste(tile, (x, 0))

        draw = ImageDraw.Draw(canvas)
        query_writer = writer_cache.get(query_label, str(query_label))
        query_ap = float(ap_scores[query_idx].item())
        query_text = (
            f"group={group_name} | query idx={query_idx} ap={query_ap:.4f} writer={query_writer} file={os.path.basename(query_path)} | "
            f"neighbors={len(tiles) - 1}"
        )
        relevant_preview = ", ".join(str(r) for r in relevant_ranks[:40]) if relevant_ranks else "none"
        draw.text((8, tile_height + 8), query_text, fill=(0, 0, 0))
        draw.text((8, tile_height + 34), f"relevant_ranks: [{relevant_preview}]", fill=(0, 0, 0))
        y = tile_height + 64
        for line in hit_lines:
            draw.text((8, y), line, fill=(0, 0, 0))
            y += 24
            if y >= canvas_h - 20:
                break

        out_name = (
            f"{group_name}_{query_idx:06d}_{os.path.splitext(os.path.basename(query_path))[0]}_top{k}.png"
        )
        canvas.save(os.path.join(output_dir, out_name))
        saved += 1
    return saved
