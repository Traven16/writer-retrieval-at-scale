from __future__ import annotations

import time
import math
import random
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import utils as vutils
from torchvision.transforms import functional as TF

from batch_utils import _split_patches_per_batch
from data.dataset import normalize_patch_tensor
from eval_utils import accuracy_from_logits, accuracy_topk
from loss_utils import (
    _triplet_embeddings,
    batch_hard_triplet_loss,
    batch_hard_triplet_soft_margin_loss,
    batch_semi_hard_triplet_loss,
    embedding_covariance_loss,
    head_cross_covariance_loss,
)


def _linear_schedule(step: int, start: float, end: float, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return end
    progress = min(max(step, 0), warmup_steps) / warmup_steps
    return start + (end - start) * progress


def _cosine_schedule(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    warmup_start: float,
) -> float:
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step <= warmup_steps:
        return _linear_schedule(step, warmup_start, base_lr, warmup_steps)
    t = max(step - warmup_steps, 0)
    t_max = max(total_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t / t_max))
    return min_lr + (base_lr - min_lr) * cosine


def _scheduled_value(
    step: int,
    schedule: str,
    start: float,
    end: float,
    warmup_steps: int,
) -> float:
    if schedule == "linear":
        return _linear_schedule(step, start, end, warmup_steps)
    return end


def _warmup_cosine_decay_value(
    step: int,
    start: float,
    peak: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    if warmup_steps > 0 and step <= warmup_steps:
        return _linear_schedule(step, start, peak, warmup_steps)
    if decay_steps <= 0:
        return peak
    t = max(step - warmup_steps, 0)
    if t >= decay_steps:
        return 0.0
    return 0.5 * peak * (1.0 + math.cos(math.pi * (t / max(decay_steps, 1))))


def set_classifier_only_training(model, enabled: bool) -> None:
    for param in model.parameters():
        param.requires_grad = not enabled
    for param in model.classifier.parameters():
        param.requires_grad = True


def set_backbone_training(model, enabled: bool) -> None:
    if hasattr(model, "set_encoder_training"):
        model.set_encoder_training(enabled)
        return
    for param in model.patch_embedding.parameters():
        param.requires_grad = enabled


def _sample_rrc_params(
    height: int,
    width: int,
    scale: Tuple[float, float] = (0.8, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> Tuple[int, int, int, int]:
    area = float(height * width)
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = area * float(torch.empty(1).uniform_(scale[0], scale[1]).item())
        aspect_ratio = math.exp(float(torch.empty(1).uniform_(log_ratio[0], log_ratio[1]).item()))
        h = int(round(math.sqrt(target_area / aspect_ratio)))
        w = int(round(math.sqrt(target_area * aspect_ratio)))
        if 0 < h <= height and 0 < w <= width:
            i = int(torch.randint(0, height - h + 1, (1,)).item())
            j = int(torch.randint(0, width - w + 1, (1,)).item())
            return i, j, h, w
    in_ratio = width / float(height)
    if in_ratio < ratio[0]:
        w = width
        h = int(round(w / ratio[0]))
    elif in_ratio > ratio[1]:
        h = height
        w = int(round(h * ratio[1]))
    else:
        h = height
        w = width
    i = max((height - h) // 2, 0)
    j = max((width - w) // 2, 0)
    return i, j, h, w


def _apply_sequence_patch_augment(
    patches: torch.Tensor,
    crop_size: int,
    strong: bool = False,
) -> torch.Tensor:
    """Apply patch augmentation with shared params per sequence (per sample in batch)."""
    if patches.dim() != 5:
        return patches
    bsz, _, _, h, w = patches.shape
    out = patches
    target_h = crop_size if crop_size > 0 else h
    target_w = crop_size if crop_size > 0 else w
    target_h = int(target_h)
    target_w = int(target_w)
    if target_h != h or target_w != w:
        target_h = h
        target_w = w

    for b in range(bsz):
        seq = out[b]  # (P,C,H,W)
        if strong:
            i, j, hh, ww = _sample_rrc_params(h, w, scale=(0.55, 1.0), ratio=(0.6, 1.6))
        else:
            i, j, hh, ww = _sample_rrc_params(h, w, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
        seq = TF.resized_crop(
            seq,
            i,
            j,
            hh,
            ww,
            size=[target_h, target_w],
            interpolation=TF.InterpolationMode.BILINEAR,
            antialias=True,
        )
        flip_p = 0.2 if strong else 0.1
        if torch.rand(1).item() < flip_p:
            seq = torch.flip(seq, dims=[-1])  # horizontal
        if torch.rand(1).item() < flip_p:
            seq = torch.flip(seq, dims=[-2])  # vertical

        if strong:
            if torch.rand(1).item() < 0.9:
                brightness = 1.0 + float(torch.empty(1).uniform_(-0.5, 0.5).item())
                contrast = 1.0 + float(torch.empty(1).uniform_(-0.5, 0.5).item())
                saturation = 1.0 + float(torch.empty(1).uniform_(-0.5, 0.5).item())
                hue = float(torch.empty(1).uniform_(-0.15, 0.15).item())
                seq = TF.adjust_brightness(seq, brightness)
                seq = TF.adjust_contrast(seq, contrast)
                seq = TF.adjust_saturation(seq, saturation)
                seq = TF.adjust_hue(seq, hue)
            if torch.rand(1).item() < 0.3:
                sigma = float(torch.empty(1).uniform_(0.1, 1.8).item())
                seq = TF.gaussian_blur(seq, kernel_size=[3, 3], sigma=[sigma, sigma])
        else:
            brightness = 1.0 + float(torch.empty(1).uniform_(-0.3, 0.3).item())
            contrast = 1.0 + float(torch.empty(1).uniform_(-0.3, 0.3).item())
            saturation = 1.0 + float(torch.empty(1).uniform_(-0.3, 0.3).item())
            hue = float(torch.empty(1).uniform_(-0.1, 0.1).item())
            seq = TF.adjust_brightness(seq, brightness)
            seq = TF.adjust_contrast(seq, contrast)
            seq = TF.adjust_saturation(seq, saturation)
            seq = TF.adjust_hue(seq, hue)
        gray_p = 0.2 if strong else 0.05
        if torch.rand(1).item() < gray_p:
            gray = seq.mean(dim=1, keepdim=True)
            seq = gray.repeat(1, seq.size(1), 1, 1)
        out[b] = seq.clamp(0.0, 1.0)
    return out


def train_one_epoch(
    model: nn.Module,
    arcface: nn.Module,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    train_cfg,
    total_train_steps: int,
    writer,
    global_step: int,
    epoch: int,
    epochs: int,
    mean: torch.Tensor,
    std: torch.Tensor,
    repeat_per_epoch: int,
    split_options: list[int],
    log_batch_images: bool = True,
) -> Tuple[dict, int]:
    model.train()
    running_loss = 0.0
    running_arcface = 0.0
    running_triplet = 0.0
    running_covariance_reg = 0.0
    running_acc = 0.0
    start_time = time.perf_counter()
    last_log_time = start_time
    interval_loss = 0.0
    interval_arcface = 0.0
    interval_triplet = 0.0
    interval_acc = 0.0
    interval_steps = 0
    interval_ce = 0.0
    interval_top1 = 0.0
    interval_top10 = 0.0
    interval_data_time = 0.0
    interval_compute_time = 0.0
    interval_triplet_active_ratio = 0.0
    interval_triplet_valid_ratio = 0.0
    interval_covariance_reg = 0.0
    interval_head_decor = 0.0

    total_steps = len(loader) * max(1, repeat_per_epoch)
    step = 0
    prev_step_end = time.perf_counter()
    mean_device = mean.to(device)
    std_device = std.to(device)
    for _ in range(max(1, repeat_per_epoch)):
        for patches, labels in loader:
            step_start = time.perf_counter()
            interval_data_time += step_start - prev_step_end
            patches, labels, _ = _split_patches_per_batch(
                patches, labels, split_options, True
            )
            if train_cfg.sequence_patch_augment_after_split and train_cfg.augment_patches:
                patches = _apply_sequence_patch_augment(
                    patches,
                    crop_size=train_cfg.sequence_patch_crop_size,
                    strong=train_cfg.strong_patch_augment,
                )
            step += 1
            step_idx = global_step + 1
            if train_cfg.lr_schedule == "cosine":
                lr = _cosine_schedule(
                    step_idx,
                    total_train_steps,
                    train_cfg.lr,
                    train_cfg.lr_min,
                    train_cfg.lr_warmup_steps,
                    train_cfg.lr_warmup_start,
                )
            else:
                lr = train_cfg.lr
            for group in optimizer.param_groups:
                group["lr"] = lr
            triplet_weight = _scheduled_value(
                step_idx,
                train_cfg.triplet_weight_schedule,
                train_cfg.triplet_weight_start,
                train_cfg.triplet_weight,
                train_cfg.triplet_weight_warmup_steps,
            )
            if not train_cfg.triplet_loss:
                triplet_weight = 0.0
            triplet_margin = _scheduled_value(
                step_idx,
                train_cfg.triplet_margin_schedule,
                train_cfg.triplet_margin_start,
                train_cfg.triplet_margin,
                train_cfg.triplet_margin_warmup_steps,
            )
            arcface_weight = _scheduled_value(
                step_idx,
                train_cfg.arcface_weight_schedule,
                train_cfg.arcface_weight_start,
                train_cfg.arcface_weight,
                train_cfg.arcface_weight_warmup_steps,
            )
            head_decor_weight = _warmup_cosine_decay_value(
                step_idx,
                train_cfg.head_decor_weight_start,
                train_cfg.head_decor_weight,
                train_cfg.head_decor_warmup_steps,
                train_cfg.head_decor_decay_steps,
            )

            patches = patches.to(device)
            patches_raw = patches
            patches = normalize_patch_tensor(patches_raw, mean_device, std_device)
            labels = labels.to(device)
            if step == 1 and log_batch_images:
                denorm = patches_raw.clamp(0.0, 1.0)

                max_samples = min(4, denorm.size(0))
                max_patches = min(8, denorm.size(1))
                sample_grid = vutils.make_grid(
                    denorm[0, :max_patches],
                    nrow=4,
                    normalize=False,
                )
                writer.add_image("train/patches_sample0", sample_grid, epoch)

                batch_patches = denorm[:max_samples, :max_patches].reshape(
                    max_samples * max_patches, denorm.size(2), denorm.size(3), denorm.size(4)
                )
                batch_grid = vutils.make_grid(
                    batch_patches,
                    nrow=max_patches,
                    normalize=False,
                )
                writer.add_image("train/patches_batch", batch_grid, epoch)

            optimizer.zero_grad(set_to_none=True)

            compute_start = time.perf_counter()
            with torch.amp.autocast('cuda', enabled=train_cfg.use_amp):
                logits, cls_embed, decoded = model(patches)
                ce_loss = torch.tensor(0.0, device=device)
                if logits is not None and train_cfg.ce_weight > 0.0:
                    ce_loss = F.cross_entropy(logits, labels)
                arcface_loss = torch.tensor(0.0, device=device)
                arcface_logits = None
                if arcface_weight > 0.0:
                    arcface_loss, arcface_logits = arcface(cls_embed, labels)
                triplet_loss = torch.tensor(0.0, device=device)
                triplet_stats = None
                if triplet_weight > 0.0:
                    triplet_embed = _triplet_embeddings(cls_embed, decoded, model)
                    triplet_labels = labels
                    if triplet_embed.dim() == 3:
                        batch_size, num_patches, dim = triplet_embed.shape
                        triplet_embed = triplet_embed.reshape(batch_size * num_patches, dim)
                        triplet_labels = labels.repeat_interleave(num_patches)
                    if train_cfg.triplet_semi_hard:
                        triplet_loss, triplet_stats = batch_semi_hard_triplet_loss(
                            triplet_embed,
                            triplet_labels,
                            triplet_margin,
                            train_cfg.triplet_soft_margin,
                            return_stats=True,
                        )
                    elif train_cfg.triplet_soft_margin:
                        triplet_loss, triplet_stats = batch_hard_triplet_soft_margin_loss(
                            triplet_embed,
                            triplet_labels,
                            return_stats=True,
                        )
                    else:
                        triplet_loss, triplet_stats = batch_hard_triplet_loss(
                            triplet_embed,
                            triplet_labels,
                            triplet_margin,
                            return_stats=True,
                        )
                head_decor_loss = torch.tensor(0.0, device=device)
                if head_decor_weight > 0.0:
                    head_outputs = getattr(model.decoder, "last_head_pooled_keep", None)
                    if head_outputs is not None:
                        head_decor_loss = head_cross_covariance_loss(head_outputs)
                covariance_reg_loss = torch.tensor(0.0, device=device)
                if train_cfg.covariance_reg_weight > 0.0:
                    covariance_reg_loss = embedding_covariance_loss(cls_embed)
                loss = (
                    train_cfg.ce_weight * ce_loss
                    + arcface_weight * arcface_loss
                    + triplet_weight * triplet_loss
                    + train_cfg.covariance_reg_weight * covariance_reg_loss
                    + head_decor_weight * head_decor_loss
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            compute_end = time.perf_counter()
            interval_compute_time += compute_end - compute_start

            metric_logits = logits
            if train_cfg.ce_weight == 0.0 and arcface_weight > 0.0:
                metric_logits = arcface.raw_logits(cls_embed)
            acc = accuracy_from_logits(metric_logits, labels) if metric_logits is not None else 0.0
            running_loss += loss.item()
            running_arcface += arcface_loss.item()
            running_triplet += triplet_loss.item()
            running_covariance_reg += covariance_reg_loss.item()
            running_acc += acc
            interval_loss += loss.item()
            interval_arcface += arcface_loss.item()
            interval_triplet += triplet_loss.item()
            interval_acc += acc
            interval_steps += 1

            interval_ce += ce_loss.item()
            interval_covariance_reg += covariance_reg_loss.item()
            interval_head_decor += head_decor_loss.item()
            if metric_logits is not None:
                interval_top1 += acc
                interval_top10 += accuracy_topk(metric_logits, labels, k=10)
            if triplet_stats is not None:
                interval_triplet_active_ratio += float(triplet_stats["active_triplet_ratio"])
                interval_triplet_valid_ratio += float(triplet_stats["valid_triplet_ratio"])
            if step % train_cfg.log_interval == 1:
                if train_cfg.debug_arcface:
                    logits_min = None if metric_logits is None else metric_logits.min().item()
                    logits_max = None if metric_logits is None else metric_logits.max().item()
                    logits_nan = False if metric_logits is None else torch.isnan(metric_logits).any().item()
                    preds_unique = (
                        0
                        if metric_logits is None
                        else torch.unique(torch.argmax(metric_logits, dim=1)).numel()
                    )
                    print(
                        f"[debug] labels min/max {labels.min().item()}/{labels.max().item()} | "
                        f"logits min/max {logits_min}/{logits_max} | "
                        f"nan {logits_nan} | preds_unique {preds_unique}"
                    )
                interval_avg_loss = interval_loss / max(1, interval_steps)
                interval_avg_arcface = interval_arcface / max(1, interval_steps)
                interval_avg_triplet = interval_triplet / max(1, interval_steps)
                interval_avg_ce = interval_ce / max(1, interval_steps)
                interval_avg_top1 = interval_top1 / max(1, interval_steps)
                interval_avg_top10 = interval_top10 / max(1, interval_steps)
                interval_avg_data = interval_data_time / max(1, interval_steps)
                interval_avg_compute = interval_compute_time / max(1, interval_steps)
                interval_avg_triplet_active = interval_triplet_active_ratio / max(1, interval_steps)
                interval_avg_triplet_valid = interval_triplet_valid_ratio / max(1, interval_steps)
                interval_avg_covariance_reg = interval_covariance_reg / max(1, interval_steps)
                interval_avg_head_decor = interval_head_decor / max(1, interval_steps)
                weighted_ce = interval_avg_ce * train_cfg.ce_weight
                weighted_arcface = interval_avg_arcface * arcface_weight
                weighted_triplet = interval_avg_triplet * triplet_weight
                weighted_covariance_reg = interval_avg_covariance_reg * train_cfg.covariance_reg_weight
                weighted_head_decor = interval_avg_head_decor * head_decor_weight
                writer.add_scalar("train/loss", interval_avg_loss, global_step)
                if metric_logits is not None:
                    writer.add_scalar("train/acc", interval_avg_top1, global_step)
                if arcface_weight > 0.0:
                    writer.add_scalar("train/arcface_loss", interval_avg_arcface, global_step)
                if triplet_weight > 0.0:
                    writer.add_scalar("train/triplet_loss", interval_avg_triplet, global_step)
                    writer.add_scalar(
                        "train/triplet_active_ratio", interval_avg_triplet_active, global_step
                    )
                    writer.add_scalar(
                        "train/triplet_valid_ratio", interval_avg_triplet_valid, global_step
                    )
                if train_cfg.covariance_reg_weight > 0.0:
                    writer.add_scalar("train/covariance_reg_loss", interval_avg_covariance_reg, global_step)
                    writer.add_scalar("train/covariance_reg_weighted", weighted_covariance_reg, global_step)
                writer.add_scalar("train/covariance_reg_weight", train_cfg.covariance_reg_weight, global_step)
                if head_decor_weight > 0.0:
                    writer.add_scalar("train/head_decor_loss", interval_avg_head_decor, global_step)
                writer.add_scalar("train/head_decor_weight", head_decor_weight, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                writer.add_scalar("train/arcface_weight", arcface_weight, global_step)
                writer.add_scalar("train/triplet_weight", triplet_weight, global_step)
                writer.add_scalar("train/triplet_margin", triplet_margin, global_step)
                writer.add_scalar("train/data_time", interval_avg_data, global_step)
                writer.add_scalar("train/compute_time", interval_avg_compute, global_step)
                elapsed = time.perf_counter() - start_time
                steps_per_sec = step / max(elapsed, 1e-6)
                remaining = total_steps - step
                eta_sec = remaining / max(steps_per_sec, 1e-6)
                interval_time = time.perf_counter() - last_log_time
                last_log_time = time.perf_counter()
                mem_msg = ""
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
                    mem_reserved = torch.cuda.memory_reserved(device) / (1024**2)
                    mem_msg = f" | MEM: {mem_alloc:.0f} / {mem_reserved:.0f}MB"
                loss_msg = f"loss: {interval_avg_loss:.2f}"
                top_msg = f"top1: {interval_avg_top1 * 100:.2f} top10: {interval_avg_top10 * 100:.2f}"
                ce_msg = f" | CE: {weighted_ce:.2f}" if train_cfg.ce_weight > 0.0 else ""
                arc_msg = (
                    f" | ArcFace: {weighted_arcface:.2f}" if arcface_weight > 0.0 else ""
                )
                trip_msg = (
                    f" | Triplet: {weighted_triplet:.2f}" if triplet_weight > 0.0 else ""
                )
                cov_msg = (
                    f" | CovReg: {weighted_covariance_reg:.4f}"
                    if train_cfg.covariance_reg_weight > 0.0
                    else ""
                )
                decor_msg = (
                    f" | HeadDecor: {weighted_head_decor:.4f}" if head_decor_weight > 0.0 else ""
                )
                triplet_ratio_msg = (
                    f" | triplet active {interval_avg_triplet_active * 100:.1f}% "
                    f"valid {interval_avg_triplet_valid * 100:.1f}%"
                    if triplet_weight > 0.0
                    else ""
                )
                timing_msg = f" | data {interval_avg_data:.3f}s compute {interval_avg_compute:.3f}s"
                print(
                    f"[{epoch}/{epochs}] {step}/{total_steps} | "
                    f"{loss_msg} {top_msg}"
                    f"{ce_msg}{arc_msg}{trip_msg}{cov_msg}{decor_msg}{triplet_ratio_msg}{timing_msg} | "
                    f"ETA: {eta_sec/60:.1f}m"
                    f"{mem_msg}"
                )
                interval_loss = 0.0
                interval_arcface = 0.0
                interval_triplet = 0.0
                interval_acc = 0.0
                interval_ce = 0.0
                interval_top1 = 0.0
                interval_top10 = 0.0
                interval_data_time = 0.0
                interval_compute_time = 0.0
                interval_triplet_active_ratio = 0.0
                interval_triplet_valid_ratio = 0.0
                interval_covariance_reg = 0.0
                interval_head_decor = 0.0
                interval_steps = 0
            prev_step_end = time.perf_counter()
            global_step += 1

    metrics = {
        "loss": running_loss / max(1, total_steps),
        "arcface_loss": running_arcface / max(1, total_steps),
        "triplet_loss": running_triplet / max(1, total_steps),
        "covariance_reg_loss": running_covariance_reg / max(1, total_steps),
        "acc": running_acc / max(1, total_steps),
    }
    return metrics, global_step
