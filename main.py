import argparse
import json
import os
import random
import time
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import functional as TF

from data.loader import build_datasets, build_loaders, summarize_dataset
from eval_utils import (
    evaluate,
    extract_embeddings,
    extract_embeddings_,
    extract_embeddings_full_page,
    extract_embeddings_for_split,
    extract_embeddings_with_splits,
    knn_accuracy,
    knn_accuracy_leave_one_out,
    _copy_indices_to_dir,
    _query_mask_excluding_labels,
    retrieval_metrics_leave_one_out,
    retrieval_map_progressive_gallery,
    visualize_retrieval_topk_hits,
    visualize_full_page_keep_heatmaps,
    pca_whiten_embeddings,
    pca_whiten_fit,
    pca_whiten_apply,
)
from train_loop import (
    train_one_epoch,
    set_classifier_only_training,
    set_backbone_training,
    _scheduled_value,
)
from models.patchformer import Patchformer, ArcFaceLoss, PatchformerConfig
from reranking.sgr import sgr_reranking
from visualize_utils import (
    visualize_retrieval_attention,
    render_decoder_cross_attention_image,
)


@dataclass
class DataConfig:
    train_csv: str
    val_csv: str
    train_root: str
    val_root: str
    train_mask_root: str
    val_mask_root: str
    image_ext: list[str]
    mask_ext: list[str]
    patch_ext: list[str]
    class_split_char: str
    class_id_index: int
    val_class_split_char: str
    val_class_id_index: int
    class_id_use_folder: bool
    train_use_precomputed_patches: bool
    val_use_precomputed_patches: bool
    train_patches_root: str
    val_patches_root: str
    cache_packed_patches_in_memory: bool
    split_patches_to_samples: list[int]
    num_patches_eval: int
    batch_size: int
    eval_batch_size: int
    balanced_batch: bool
    samples_per_class: int
    num_workers: int
    pin_memory: bool
    seed: int
    no_augment: bool
    augment_patches: bool
    augment_otsu: bool
    augment_morphology: bool
    binarized_in_memory: bool
    binarized_cache_on_gpu: bool
    binarized_cache_workers: int
    debug_dataset: bool
    debug_first_batch: bool
    val_gray: bool
    full_image_input: bool
    full_image_height: int
    full_image_width: int
    full_image_pad_to_size: bool
    full_image_resize_longest_side_first: bool
    sequence_patch_augment_in_dataset: bool
    sequence_patch_split_per_sample: int
    sequence_patch_crop_size: int
    strong_patch_augment: bool
    page_rrc_scale_min: float
    page_rrc_scale_max: float
    train_writer_fraction: float
    train_writer_subset_seed: int
    input_normalization: str


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    lr_schedule: str
    lr_warmup_steps: int
    lr_warmup_start: float
    lr_min: float
    weight_decay: float
    use_amp: bool
    arcface_weight: float
    arcface_margin: float
    arcface_scale: float
    arcface_mlp_dim: int
    arcface_weight_schedule: str
    arcface_weight_warmup_steps: int
    arcface_weight_start: float
    triplet_loss: bool
    triplet_weight: float
    triplet_margin: float
    triplet_soft_margin: bool
    triplet_semi_hard: bool
    triplet_weight_schedule: str
    triplet_weight_warmup_steps: int
    triplet_weight_start: float
    triplet_margin_schedule: str
    triplet_margin_warmup_steps: int
    triplet_margin_start: float
    covariance_reg_weight: float
    head_decor_weight: float
    head_decor_weight_start: float
    head_decor_warmup_steps: int
    head_decor_decay_steps: int
    ce_weight: float
    train_only_classifier_epochs: int
    freeze_backbone_epochs: int
    repeat_per_epoch: int
    val_every: int
    checkpoint_every: int
    log_interval: int
    insert_weights_on_mismatch: bool
    validate_before_first_epoch: bool
    debug_arcface: bool
    augment_patches: bool
    strong_patch_augment: bool
    sequence_patch_augment_after_split: bool
    sequence_patch_crop_size: int


@dataclass
class EvalConfig:
    eval_only: bool
    knn_eval: bool
    knn_k: int
    retrieval_eval: bool
    full_page_eval: bool
    val_features: str
    val_include_queries: bool
    sgr_reranking: bool
    sgr_k: int
    sgr_layers: int
    sgr_gamma: float
    retrieval_distractor_labels: list[int]
    copy_not_in_top10pct: str
    dump_tmp: str
    dump_embeddings_chunk_size: int
    recompute_embeddings: bool
    whiten_test: bool
    whiten_none: bool
    visualize_retrieval: bool
    visualize_full_page_keep_heatmaps: bool
    visualize_retrieval_topk_hits: bool
    retrieval_topk_viz_k: int
    retrieval_topk_viz_dir: str
    gallery_growth_eval: bool
    gallery_growth_writer_chunk: int
    gallery_growth_min_writers: int
    gallery_growth_shuffle_writers: bool
    gallery_growth_seed: int
    gallery_growth_csv_path: str
    gallery_growth_ap_hist: bool
    gallery_growth_ap_hist_bins: int
    gallery_growth_ap_hist_full_only: bool
    no_progress: bool
    checkpoint: str
    resume_weights: str


@dataclass
class RunConfig:
    run_name: str
    log_dir: str
    save_path: str
    results_dir: str


def _normalize_ext_list(exts: Optional[list[str]], default: str) -> list[str]:
    if not exts:
        exts = [default]
    normalized: list[str] = []
    for item in exts:
        parts = []
        if isinstance(item, str):
            parts = [p for p in item.split(",") if p]
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if not part.startswith("."):
                part = "." + part
            normalized.append(part.lower())
    return normalized or [default]


def _random_morph_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    kernel = torch.zeros((3, 3), device=device, dtype=dtype)
    kernel[1, 1] = 1.0
    rand = torch.rand((3, 3), device=device)
    kernel = torch.where(rand < 0.3, torch.ones_like(kernel), kernel)
    shift_y = int(torch.randint(-1, 2, (1,)).item())
    shift_x = int(torch.randint(-1, 2, (1,)).item())
    return torch.roll(kernel, shifts=(shift_y, shift_x), dims=(0, 1))


def _morph_op(tensor: torch.Tensor, kernel: torch.Tensor, op: str) -> torch.Tensor:
    if tensor.dim() != 3:
        raise ValueError("Expected tensor shape (C, H, W) for morphology")
    if kernel.shape != (3, 3):
        raise ValueError("Expected kernel shape (3, 3)")
    kernel_flat = kernel.reshape(-1).to(dtype=torch.bool)
    padded = torch.nn.functional.unfold(
        tensor.unsqueeze(0), kernel_size=3, padding=1
    )
    patches = padded.view(tensor.size(0), 9, -1)
    if op == "dilate":
        masked = patches.masked_fill(~kernel_flat.view(1, 9, 1), -float("inf"))
        reduced = masked.max(dim=1).values
    elif op == "erode":
        masked = patches.masked_fill(~kernel_flat.view(1, 9, 1), float("inf"))
        reduced = masked.min(dim=1).values
    else:
        raise ValueError(f"Unknown morph op: {op}")
    return reduced.view(tensor.size(0), tensor.size(1), tensor.size(2))


def _apply_morphology_augment(tensor: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        kernel = _random_morph_kernel(tensor.device, tensor.dtype)
        tensor = _morph_op(tensor, kernel, "erode")
    if torch.rand(1).item() < 0.5:
        kernel = _random_morph_kernel(tensor.device, tensor.dtype)
        tensor = _morph_op(tensor, kernel, "dilate")
    if torch.rand(1).item() < 0.25:
        sigma = float(torch.empty(1).uniform_(0.1, 1.5).item())
        tensor = TF.gaussian_blur(tensor, kernel_size=[3, 3], sigma=[sigma, sigma])
    return tensor


def build_configs(
    args: argparse.Namespace,
) -> Tuple[PatchformerConfig, DataConfig, TrainConfig, EvalConfig, RunConfig]:
    n_head = args.n_head if args.n_head > 0 else args.nhead
    d_out = args.d_out if args.d_out > 0 else args.d_model
    if args.d_head > 0:
        if d_out > 0 and d_out != n_head * args.d_head:
            raise ValueError(
                f"d_out ({d_out}) must equal n_head*d_head ({n_head * args.d_head})"
            )
        d_head = args.d_head
        d_out = n_head * d_head
    else:
        if d_out % n_head != 0:
            raise ValueError(f"d_out ({d_out}) must be divisible by n_head ({n_head})")
        d_head = d_out // n_head
    d_encoder = args.d_encoder if args.d_encoder > 0 else d_out
    if args.num_encoder_layers > 0 and d_encoder % n_head != 0:
        raise ValueError(
            f"d_encoder ({d_encoder}) must be divisible by n_head ({n_head}) when encoder is used"
        )
    model_cfg = PatchformerConfig(
        num_classes=args.num_classes,
        num_patches=args.num_patches,
        num_queries=args.num_queries,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        d_encoder=d_encoder,
        d_out=d_out,
        n_head=n_head,
        d_head=d_head,
        d_model=d_out,
        nhead=n_head,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        encoder_name=args.encoder_name,
        encoder_pretrained=args.encoder_pretrained,
        encoder_pretrained_path=args.encoder_pretrained_path,
        encoder_return_patch_sequence=args.encoder_return_patch_sequence,
        encoder_freeze=args.encoder_freeze,
        encoder_trainable_last_layers=args.encoder_trainable_last_layers,
        decoder_type=args.decoder_type,
        ghostvlad_clusters=args.ghostvlad_clusters,
        ghostvlad_ghost_clusters=args.ghostvlad_ghost_clusters,
        xvlad_no_intra_norm=args.xvlad_no_intra_norm,
        embedding_whitening_head=args.embedding_whitening_head,
        decoder_checkpoint=args.decoder_checkpoint,
        resnet_3x3_stem=args.resnet_3x3_stem,
        no_classifier=args.no_classifier,
    )
    data_cfg = DataConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        train_root=args.train_root,
        val_root=args.val_root,
        train_mask_root=args.train_mask_root,
        val_mask_root=args.val_mask_root,
        image_ext=_normalize_ext_list(args.image_ext, default=".jpg"),
        mask_ext=_normalize_ext_list(args.mask_ext, default=".png"),
        patch_ext=_normalize_ext_list(args.patch_ext, default=".png"),
        class_split_char=args.class_split_char,
        class_id_index=args.class_id_index,
        val_class_split_char=args.val_class_split_char,
        val_class_id_index=args.val_class_id_index,
        class_id_use_folder=args.class_id_use_folder,
        train_use_precomputed_patches=args.train_use_precomputed_patches,
        val_use_precomputed_patches=args.val_use_precomputed_patches,
        train_patches_root=args.train_patches_root,
        val_patches_root=args.val_patches_root,
        cache_packed_patches_in_memory=args.cache_packed_patches_in_memory,
        split_patches_to_samples=args.split_patches_to_samples,
        num_patches_eval=args.num_patches_eval,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        balanced_batch=args.balanced_batch,
        samples_per_class=args.samples_per_class,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        seed=args.seed,
        no_augment=args.no_augment,
        augment_patches=args.augment_patches,
        augment_otsu=args.augment_otsu,
        augment_morphology=args.augment_morphology,
        binarized_in_memory=args.binarized_in_memory,
        binarized_cache_on_gpu=args.binarized_cache_on_gpu,
        binarized_cache_workers=args.binarized_cache_workers,
        debug_dataset=args.debug_dataset,
        debug_first_batch=args.debug_first_batch,
        val_gray=args.val_gray,
        full_image_input=args.full_image_input,
        full_image_height=args.full_image_height,
        full_image_width=args.full_image_width,
        full_image_pad_to_size=args.full_image_pad_to_size,
        full_image_resize_longest_side_first=args.full_image_resize_longest_side_first,
        sequence_patch_augment_in_dataset=args.sequence_patch_augment_in_dataset,
        sequence_patch_split_per_sample=args.sequence_patch_split_per_sample,
        sequence_patch_crop_size=args.sequence_patch_crop_size,
        strong_patch_augment=args.strong_patch_augment,
        page_rrc_scale_min=args.page_rrc_scale_min,
        page_rrc_scale_max=args.page_rrc_scale_max,
        train_writer_fraction=args.train_writer_fraction,
        train_writer_subset_seed=args.train_writer_subset_seed,
        input_normalization=args.input_normalization,
    )
    if args.sequence_patch_augment_after_split and args.sequence_patch_augment_in_dataset:
        raise ValueError(
            "Use either --sequence-patch-augment-after-split or --sequence-patch-augment-in-dataset, not both."
        )
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_warmup_steps=args.lr_warmup_steps,
        lr_warmup_start=args.lr_warmup_start,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        use_amp=args.use_amp,
        arcface_weight=args.arcface_weight,
        arcface_margin=args.arcface_margin,
        arcface_scale=args.arcface_scale,
        arcface_mlp_dim=args.arcface_mlp_dim,
        arcface_weight_schedule=args.arcface_weight_schedule,
        arcface_weight_warmup_steps=args.arcface_weight_warmup_steps,
        arcface_weight_start=args.arcface_weight_start,
        triplet_loss=args.triplet_loss,
        triplet_weight=args.triplet_weight,
        triplet_margin=args.triplet_margin,
        triplet_soft_margin=args.triplet_soft_margin,
        triplet_semi_hard=args.triplet_semi_hard,
        triplet_weight_schedule=args.triplet_weight_schedule,
        triplet_weight_warmup_steps=args.triplet_weight_warmup_steps,
        triplet_weight_start=args.triplet_weight_start,
        triplet_margin_schedule=args.triplet_margin_schedule,
        triplet_margin_warmup_steps=args.triplet_margin_warmup_steps,
        triplet_margin_start=args.triplet_margin_start,
        covariance_reg_weight=args.covariance_reg_weight,
        head_decor_weight=args.head_decor_weight,
        head_decor_weight_start=args.head_decor_weight_start,
        head_decor_warmup_steps=args.head_decor_warmup_steps,
        head_decor_decay_steps=args.head_decor_decay_steps,
        ce_weight=args.ce_weight,
        train_only_classifier_epochs=args.train_only_classifier_epochs,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        repeat_per_epoch=args.repeat_per_epoch,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        log_interval=args.log_interval,
        insert_weights_on_mismatch=args.insert_weights_on_mismatch,
        validate_before_first_epoch=args.validate_before_first_epoch,
        debug_arcface=args.debug_arcface,
        augment_patches=args.augment_patches,
        strong_patch_augment=args.strong_patch_augment,
        sequence_patch_augment_after_split=args.sequence_patch_augment_after_split,
        sequence_patch_crop_size=args.sequence_patch_crop_size,
    )
    eval_cfg = EvalConfig(
        eval_only=args.eval_only,
        knn_eval=args.knn_eval,
        knn_k=args.knn_k,
        retrieval_eval=args.retrieval_eval,
        full_page_eval=args.full_page_eval,
        val_features=args.val_features,
        val_include_queries=args.val_include_queries,
        sgr_reranking=args.sgr_reranking,
        sgr_k=args.sgr_k,
        sgr_layers=args.sgr_layers,
        sgr_gamma=args.sgr_gamma,
        retrieval_distractor_labels=args.retrieval_distractor_labels,
        copy_not_in_top10pct=args.copy_not_in_top10pct,
        dump_tmp=args.dump_tmp,
        dump_embeddings_chunk_size=args.dump_embeddings_chunk_size,
        recompute_embeddings=args.recompute_embeddings,
        whiten_test=args.whiten_test,
        whiten_none=args.whiten_none,
        visualize_retrieval=args.visualize_retrieval,
        visualize_full_page_keep_heatmaps=args.visualize_full_page_keep_heatmaps,
        visualize_retrieval_topk_hits=args.visualize_retrieval_topk_hits,
        retrieval_topk_viz_k=args.retrieval_topk_viz_k,
        retrieval_topk_viz_dir=args.retrieval_topk_viz_dir,
        gallery_growth_eval=args.gallery_growth_eval,
        gallery_growth_writer_chunk=args.gallery_growth_writer_chunk,
        gallery_growth_min_writers=args.gallery_growth_min_writers,
        gallery_growth_shuffle_writers=args.gallery_growth_shuffle_writers,
        gallery_growth_seed=args.gallery_growth_seed,
        gallery_growth_csv_path=args.gallery_growth_csv_path,
        gallery_growth_ap_hist=args.gallery_growth_ap_hist,
        gallery_growth_ap_hist_bins=args.gallery_growth_ap_hist_bins,
        gallery_growth_ap_hist_full_only=args.gallery_growth_ap_hist_full_only,
        no_progress=args.no_progress,
        checkpoint=args.checkpoint,
        resume_weights=args.resume_weights,
    )
    run_cfg = RunConfig(
        run_name=args.run_name,
        log_dir=args.log_dir,
        save_path=args.save_path,
        results_dir=args.results_dir,
    )
    return model_cfg, data_cfg, train_cfg, eval_cfg, run_cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patchformer training")
    parser.add_argument("--train-csv", type=str, default="")
    parser.add_argument("--val-csv", type=str, default="")
    parser.add_argument("--train-root", type=str, default="")
    parser.add_argument("--val-root", type=str, default="")
    parser.add_argument("--train-mask-root", type=str, default="")
    parser.add_argument("--val-mask-root", type=str, default="")
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--num-patches", type=int, default=16)
    parser.add_argument("--num-patches-eval", type=int, default=256)
    parser.add_argument(
        "--split-patches-to-samples",
        type=int,
        nargs="+",
        default=[1],
        help="Split num_patches into N samples per batch (random choice). Must divide num_patches.",
    )
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument(
        "--d-encoder",
        type=int,
        default=0,
        help="Encoder token dim before decoder (0 = use d_out / d_model).",
    )
    parser.add_argument(
        "--d-out",
        type=int,
        default=0,
        help="Decoder input/output token dim (0 = use d_model).",
    )
    parser.add_argument(
        "--n-head",
        type=int,
        default=0,
        help="Explicit number of heads (0 = use --nhead).",
    )
    parser.add_argument(
        "--d-head",
        type=int,
        default=0,
        help="Per-head dim. Enforces n_head * d_head = d_out when set.",
    )
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=0)
    parser.add_argument("--num-decoder-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-name", type=str, default="resnet18")
    parser.add_argument("--encoder-pretrained", action="store_true")
    parser.add_argument("--encoder-pretrained-path", type=str, default="")
    parser.add_argument(
        "--encoder-freeze",
        action="store_true",
        help="Freeze the encoder backbone before training optional tail layers.",
    )
    parser.add_argument(
        "--encoder-trainable-last-layers",
        type=int,
        default=0,
        help="When --encoder-freeze is set, unfreeze this many final encoder blocks/layers.",
    )
    parser.add_argument(
        "--encoder-return-patch-sequence",
        action="store_true",
        help="Use encoder patch-token sequence output (e.g., ViT patch tokens) instead of per-image pooled features.",
    )
    parser.add_argument("--decoder-type", type=str, default="transformer_decoder")
    parser.add_argument(
        "--ghostvlad-clusters",
        "--xvlad-clusters",
        dest="ghostvlad_clusters",
        type=int,
        default=16,
        help="Number of retained aggregation clusters for GhostVLAD/X-VLAD.",
    )
    parser.add_argument(
        "--ghostvlad-ghost-clusters",
        "--xvlad-ghost-clusters",
        dest="ghostvlad_ghost_clusters",
        type=int,
        default=4,
        help="Number of discarded ghost clusters for GhostVLAD/X-VLAD.",
    )
    parser.add_argument(
        "--xvlad_no_intra_norm",
        "--xvlad-no-intra-norm",
        dest="xvlad_no_intra_norm",
        action="store_true",
        help="Disable X-VLAD intra-normalization across the concatenated head outputs.",
    )
    parser.add_argument(
        "--embedding-whitening-head",
        action="store_true",
        help="Apply LayerNorm plus identity-initialized linear whitening head to final embeddings.",
    )
    parser.add_argument("--decoder-checkpoint", action="store_true")
    parser.add_argument("--resnet-3x3-stem", action="store_true")
    parser.add_argument("--no-classifier", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="Override batch size for eval/validation. 0 uses --batch-size.",
    )
    parser.add_argument("--balanced-batch", action="store_true")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine"],
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-warmup-start", type=float, default=1e-7)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--arcface-weight", type=float, default=0.0)
    parser.add_argument("--arcface-margin", type=float, default=0.5)
    parser.add_argument("--arcface-scale", type=float, default=64.0)
    parser.add_argument("--arcface-mlp-dim", type=int, default=0)
    parser.add_argument(
        "--arcface-weight-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
    )
    parser.add_argument("--arcface-weight-warmup-steps", type=int, default=0)
    parser.add_argument("--arcface-weight-start", type=float, default=0.0)
    parser.add_argument("--triplet-loss", action="store_true")
    parser.add_argument("--triplet-weight", type=float, default=1.0)
    parser.add_argument("--triplet-margin", type=float, default=0.2)
    parser.add_argument("--triplet-soft-margin", action="store_true")
    parser.add_argument("--triplet-semi-hard", action="store_true")
    parser.add_argument(
        "--triplet-weight-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
    )
    parser.add_argument("--triplet-weight-warmup-steps", type=int, default=0)
    parser.add_argument("--triplet-weight-start", type=float, default=0.0)
    parser.add_argument(
        "--triplet-margin-schedule",
        type=str,
        default="constant",
        choices=["constant", "linear"],
    )
    parser.add_argument("--triplet-margin-warmup-steps", type=int, default=0)
    parser.add_argument("--triplet-margin-start", type=float, default=0.1)
    parser.add_argument(
        "--covariance-reg-weight",
        type=float,
        default=0.0,
        help="Weight for final embedding covariance off-diagonal regularization.",
    )
    parser.add_argument(
        "--head-decor-weight",
        type=float,
        default=0.0,
        help="Weight for between-head decorrelation loss (X-VLAD).",
    )
    parser.add_argument(
        "--head-decor-weight-start",
        type=float,
        default=0.0,
        help="Start value for head decorrelation warmup.",
    )
    parser.add_argument(
        "--head-decor-warmup-steps",
        type=int,
        default=0,
        help="Warmup steps for head decorrelation weight.",
    )
    parser.add_argument(
        "--head-decor-decay-steps",
        type=int,
        default=0,
        help="Cosine decay steps from peak head decorrelation weight to 0 after warmup (0 disables decay).",
    )
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze patch backbone (encoder CNN) for the first N epochs.",
    )
    parser.add_argument("--insert-weights-on-mismatch", action="store_true")
    parser.add_argument("--log-dir", type=str, default="runs/patchformer")
    parser.add_argument("--save-path", type=str, default="checkpoints/patchformer.pt")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save an additional periodic checkpoint every N epochs (0 disables).",
    )
    parser.add_argument("--knn-eval", action="store_true")
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--augment-patches", action="store_true")
    parser.add_argument(
        "--input-normalization",
        type=str,
        default="imagenet",
        choices=["imagenet", "none"],
        help="Input normalization before the encoder. Use 'none' for SSL checkpoints trained on raw [0,1] tensors.",
    )
    parser.add_argument(
        "--full-image-input",
        action="store_true",
        help="Load one resized image per sample (shape 1xCxHxW) instead of sampling multiple patch crops.",
    )
    parser.add_argument(
        "--full-image-height",
        type=int,
        default=0,
        help="Target full-image height used when --full-image-input is enabled (0 = use patch-size).",
    )
    parser.add_argument(
        "--full-image-width",
        type=int,
        default=0,
        help="Target full-image width used when --full-image-input is enabled (0 = use patch-size).",
    )
    parser.add_argument(
        "--full-image-pad-to-size",
        action="store_true",
        help=(
            "When --full-image-input is enabled, match target size by padding/"
            "center-cropping instead of bilinear resizing."
        ),
    )
    parser.add_argument(
        "--full-image-resize-longest-side-first",
        action="store_true",
        help=(
            "When used with --full-image-pad-to-size, first resize images so their "
            "longer side matches max(target_h,target_w), then pad to target size."
        ),
    )
    parser.add_argument(
        "--strong-patch-augment",
        action="store_true",
        help="Use stronger patch-level augmentations (document SSL-style).",
    )
    parser.add_argument(
        "--sequence-patch-augment-after-split",
        action="store_true",
        help="Apply patch augmentations after split_patches_to_samples with shared params per sequence.",
    )
    parser.add_argument(
        "--sequence-patch-augment-in-dataset",
        action="store_true",
        help="Apply sequence-level patch augmentation inside dataset workers (shared params per split chunk).",
    )
    parser.add_argument(
        "--sequence-patch-split-per-sample",
        type=int,
        default=1,
        help="Number of split chunks per sample for dataset-side sequence augment (e.g. 2).",
    )
    parser.add_argument(
        "--sequence-patch-crop-size",
        type=int,
        default=32,
        help="Target patch crop size for sequence-level RandomResizedCrop (defaults to 32).",
    )
    parser.add_argument(
        "--page-rrc-scale-min",
        type=float,
        default=0.5,
        help="Minimum scale for page-level RandomResizedCrop.",
    )
    parser.add_argument(
        "--page-rrc-scale-max",
        type=float,
        default=1.0,
        help="Maximum scale for page-level RandomResizedCrop.",
    )
    parser.add_argument("--augment-otsu", action="store_true")
    parser.add_argument("--augment-morphology", action="store_true")
    parser.add_argument(
        "--binarized-in-memory",
        action="store_true",
        help="Use in-memory binarized dataset (mask == image) for faster loading.",
    )
    parser.add_argument(
        "--binarized-cache-on-gpu",
        action="store_true",
        help="Cache binarized dataset on GPU (requires num_workers=0).",
    )
    parser.add_argument(
        "--binarized-cache-workers",
        type=int,
        default=0,
        help="Worker processes used to build the binarized cache (0 = sequential).",
    )
    parser.add_argument("--debug-dataset", action="store_true")
    parser.add_argument(
        "--debug-first-batch",
        action="store_true",
        help="If set, print timing and paths for the first few samples per worker.",
    )
    parser.add_argument("--val-gray", action="store_true")
    parser.add_argument("--class-split-char", type=str, default="_")
    parser.add_argument(
        "--class-id-index",
        type=int,
        default=0,
        help="Index of the token to use as class id when splitting filenames.",
    )
    parser.add_argument(
        "--class-id-use-folder",
        action="store_true",
        help="Include the parent folder path in the class id (folder names are not split).",
    )
    parser.add_argument(
        "--val-class-split-char",
        type=str,
        default="",
        help="Optional override for class split char when building val dataset.",
    )
    parser.add_argument(
        "--val-class-id-index",
        type=int,
        default=-1,
        help="Optional override for class id index when building val dataset.",
    )
    parser.add_argument(
        "--train-writer-fraction",
        type=float,
        default=1.0,
        help="Fraction of training writers/classes to keep. Applied at writer level, not page level.",
    )
    parser.add_argument(
        "--train-writer-subset-seed",
        type=int,
        default=0,
        help="Seed for the deterministic shuffled writer subset when --train-writer-fraction < 1.",
    )
    parser.add_argument("--image-ext", type=str, nargs="+", default=[".jpg"])
    parser.add_argument("--mask-ext", type=str, nargs="+", default=[".png"])
    parser.add_argument("--patch-ext", type=str, nargs="+", default=[".png"])
    parser.add_argument(
        "--train-use-precomputed-patches",
        action="store_true",
        help="Use packed patch tensors (*.pt) from --train-patches-root.",
    )
    parser.add_argument(
        "--val-use-precomputed-patches",
        action="store_true",
        help="Use packed patch tensors (*.pt) from --val-patches-root.",
    )
    parser.add_argument(
        "--train-patches-root",
        type=str,
        default="",
        help="Root directory with packed patch files produced by tools/pack_precomputed_patches.py.",
    )
    parser.add_argument(
        "--val-patches-root",
        type=str,
        default="",
        help="Root directory with packed patch files produced by tools/pack_precomputed_patches.py.",
    )
    parser.add_argument(
        "--cache-packed-patches-in-memory",
        action="store_true",
        help="Preload packed patch files into shared memory once before DataLoader workers start.",
    )
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--repeat-per-epoch", type=int, default=1)
    parser.add_argument("--retrieval-eval", action="store_true")
    parser.add_argument(
        "--full-page-eval",
        action="store_true",
        help="Use full-page encoder features for embedding extraction (bypass patch sampling).",
    )
    parser.add_argument("--val-features", type=str, default="cls", choices=["cls", "query"])
    parser.add_argument("--val-include-queries", action="store_true")
    parser.add_argument("--sgr-reranking", action="store_true")
    parser.add_argument("--sgr-k", type=int, default=2)
    parser.add_argument("--sgr-layers", type=int, default=1)
    parser.add_argument("--sgr-gamma", type=float, default=0.9)
    parser.add_argument(
        "--retrieval-distractor-labels",
        type=int,
        nargs="*",
        default=[],
        help="Labels to exclude from retrieval queries (distractors).",
    )
    parser.add_argument(
        "--copy-not-in-top10pct",
        type=str,
        default="",
        help="If set, copy query images whose closest relevant match is outside top 10%% to this directory.",
    )
    parser.add_argument(
        "--dump-tmp",
        type=str,
        default="",
        help="If set, cache train/val embeddings to this directory in eval-only mode.",
    )
    parser.add_argument(
        "--dump-embeddings-chunk-size",
        type=int,
        default=2048,
        help=(
            "When --dump-tmp is set, checkpoint full-page embedding extraction by saving "
            "chunk files of this many samples (0 disables chunking)."
        ),
    )
    parser.add_argument(
        "--recompute-embeddings",
        action="store_true",
        help="If set, recompute embeddings even if cached files exist.",
    )
    parser.add_argument(
        "--whiten-test",
        action="store_true",
        help="If set, fit whitening on test/val embeddings even when train embeddings exist.",
    )
    parser.add_argument(
        "--whiten-none",
        action="store_true",
        help="If set, skip whitening entirely and evaluate raw embeddings.",
    )
    parser.add_argument(
        "--visualize-retrieval",
        action="store_true",
        help="If set, visualize encoder/decoder attention for the first eval batch and exit.",
    )
    parser.add_argument(
        "--visualize-full-page-keep-heatmaps",
        action="store_true",
        help="If set, run first-batch full-page X-VLAD per-head keep heatmap overlays and exit.",
    )
    parser.add_argument(
        "--visualize-retrieval-topk-hits",
        action="store_true",
        help="If set in eval-only retrieval mode, save one query-vs-topk image per val sample.",
    )
    parser.add_argument(
        "--retrieval-topk-viz-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to visualize per query for --visualize-retrieval-topk-hits.",
    )
    parser.add_argument(
        "--retrieval-topk-viz-dir",
        type=str,
        default="",
        help="Optional output directory for --visualize-retrieval-topk-hits.",
    )
    parser.add_argument(
        "--gallery-growth-eval",
        action="store_true",
        help="Evaluate retrieval mAP/top1 as gallery writers are added in chunks.",
    )
    parser.add_argument(
        "--gallery-growth-writer-chunk",
        type=int,
        default=1000,
        help="Number of writers to add per gallery-growth step.",
    )
    parser.add_argument(
        "--gallery-growth-min-writers",
        type=int,
        default=1000,
        help="Starting number of writers for gallery-growth evaluation.",
    )
    parser.add_argument(
        "--gallery-growth-shuffle-writers",
        action="store_true",
        help="Shuffle writer order before progressive gallery evaluation.",
    )
    parser.add_argument(
        "--gallery-growth-seed",
        type=int,
        default=41023,
        help="Random seed for writer shuffling in gallery-growth evaluation.",
    )
    parser.add_argument(
        "--gallery-growth-csv-path",
        type=str,
        default="",
        help="Optional CSV output path for gallery-growth curve.",
    )
    parser.add_argument(
        "--gallery-growth-ap-hist",
        action="store_true",
        help="If set, print AP histograms at each gallery-growth increment.",
    )
    parser.add_argument(
        "--gallery-growth-ap-hist-bins",
        type=int,
        default=10,
        help="Number of AP histogram bins for --gallery-growth-ap-hist.",
    )
    parser.add_argument(
        "--gallery-growth-ap-hist-full-only",
        action="store_true",
        help="If set, compute AP histogram only for the full gallery set.",
    )
    parser.add_argument(
        "--no-numerical-text",
        action="store_true",
        help="If set, do not print numerical values on visualization plots.",
    )
    parser.add_argument(
        "--no-progress",
        "--no-progess",
        dest="no_progress",
        action="store_true",
        help="If set, disable tqdm progress bars during embedding extraction.",
    )
    parser.add_argument(
        "--validate-before-first-epoch",
        action="store_true",
        help="If set, run validation once before the first training epoch.",
    )
    parser.add_argument("--debug-arcface", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume-weights", type=str, default="")
    parser.add_argument("--train-only-classifier-epochs", type=int, default=0)
    return parser.parse_args()


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def print_model_summary(model: Patchformer) -> None:
    encoder_params = _count_params(model.patch_embedding) + _count_params(model.patch_proj)
    if model.patch_encoder is not None:
        encoder_params += _count_params(model.patch_encoder)
    decoder_params = _count_params(model.decoder)
    total_params = _count_params(model)
    if model.query_tokens is not None:
        decoder_params += model.query_tokens.numel()
        total_params += model.query_tokens.numel()
    if model.classifier is not None:
        decoder_params += _count_params(model.classifier)
        total_params += _count_params(model.classifier)
    print(
        "Model summary | "
        f"d_model {model.classifier.in_features if model.classifier is not None else 'none'} | "
        f"patch_size {model.patch_size} | "
        f"num_patches {model.num_patches} | "
        f"num_queries {model.query_tokens.size(1) - 1 if model.query_tokens is not None else 0}"
    )
    print(
        "Params | "
        f"encoder {encoder_params / 1e6:.3f}M | "
        f"decoder {decoder_params / 1e6:.3f}M | "
        f"total {total_params / 1e6:.3f}M"
    )


def build_run_name(user_name: str) -> str:
    if user_name:
        return user_name
    pets = [
        "lark",
        "otter",
        "puma",
        "stoat",
        "yak",
        "finch",
        "lynx",
        "orca",
        "bison",
        "swift",
    ]
    day = time.strftime("%Y%m%d-%H%M%S")
    pet = random.choice(pets)
    return f"{day}-{pet}"


def _format_run_path(template: str, run_name: str) -> str:
    if not template:
        return template
    return template.format_map({"run_name": run_name})


def _write_results(
    results_dir: str,
    run_name: str,
    summary: Dict[str, object],
    args: argparse.Namespace,
) -> None:
    if not results_dir:
        return
    output_dir = os.path.join(results_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "run_name": run_name,
        "summary": summary,
        "args": vars(args),
    }
    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _format_keys(keys: list[str]) -> str:
    if not keys:
        return "[]"
    return "[" + ", ".join(keys) + "]"


def _log_val_decoder_attention_sample(
    model,
    loader,
    device: torch.device,
    writer: SummaryWriter,
    step: int,
    annotate: bool,
    enabled: bool = True,
) -> None:
    if not enabled:
        return
    if loader is None:
        return
    try:
        batch = next(iter(loader))
        patches = batch[0].to(device)
        attn_img = render_decoder_cross_attention_image(
            model,
            patches,
            sample_index=0,
            annotate=annotate,
        )
        if attn_img is not None:
            writer.add_image("val/decoder_cross_attention_sample0", attn_img, step)
    except Exception as exc:
        print(f"Warning: failed to log decoder attention image ({exc})")


def _filter_state_dict(
    state: Dict[str, torch.Tensor],
    model: nn.Module,
    insert_on_mismatch: bool,
) -> Tuple[Dict[str, torch.Tensor], list[str], list[str], list[str]]:
    model_state = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    mismatched: list[str] = []
    inserted: list[str] = []
    for key, value in state.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        target = model_state[key]
        if target.shape != value.shape:
            if insert_on_mismatch and value.ndim == target.ndim:
                if all(s <= t for s, t in zip(value.shape, target.shape)):
                    new_value = target.clone()
                    slices = tuple(slice(0, s) for s in value.shape)
                    new_value[slices] = value.to(dtype=target.dtype)
                    filtered[key] = new_value
                    inserted.append(key)
                    continue
            mismatched.append(key)
            continue
        filtered[key] = value
    return filtered, unexpected, mismatched, inserted


def main() -> None:
    args = parse_args()
    model_cfg, data_cfg, train_cfg, eval_cfg, run_cfg = build_configs(args)
    if eval_cfg.eval_only and data_cfg.binarized_in_memory:
        print("Eval-only mode: disabling in-memory binarized page cache.")
        data_cfg.binarized_in_memory = False
        data_cfg.binarized_cache_on_gpu = False
        data_cfg.binarized_cache_workers = 0
    set_seed(data_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_csv = bool(data_cfg.train_csv and data_cfg.val_csv)
    use_folders = bool(data_cfg.train_root and data_cfg.val_root)
    eval_only = eval_cfg.eval_only
    if not (use_csv or use_folders or (eval_only and (data_cfg.val_csv or data_cfg.val_root))):
        raise ValueError("Provide --train-csv/--val-csv or --train-root/--val-root")

    page_transform = None
    train_transform = None
    if not data_cfg.no_augment:
        def _page_random_resized_crop(image, mask):
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image,
                scale=(data_cfg.page_rrc_scale_min, data_cfg.page_rrc_scale_max),
                ratio=(3 / 4, 4 / 3),
            )
            image = TF.resized_crop(
                image,
                i,
                j,
                h,
                w,
                size=[image.height, image.width],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            mask = TF.resized_crop(
                mask,
                i,
                j,
                h,
                w,
                size=[mask.height, mask.width],
                interpolation=TF.InterpolationMode.NEAREST,
            )
            return image, mask

        if data_cfg.augment_morphology:
            if data_cfg.augment_patches:
                page_transform = _page_random_resized_crop
                def train_transform(patch):
                    return _apply_morphology_augment(patch)
            else:
                def page_transform(image, mask):
                    image, mask = _page_random_resized_crop(image, mask)
                    image_tensor = TF.to_tensor(image)
                    image_tensor = _apply_morphology_augment(image_tensor)
                    image = TF.to_pil_image(image_tensor)
                    return image, mask
        elif data_cfg.augment_patches:
            page_transform = _page_random_resized_crop
            if (
                not train_cfg.sequence_patch_augment_after_split
                and not data_cfg.sequence_patch_augment_in_dataset
            ):
                if train_cfg.strong_patch_augment:
                    train_transform = transforms.Compose(
                        [
                            transforms.RandomResizedCrop(
                                size=32, scale=(0.55, 1.0), ratio=(0.6, 1.6)
                            ),
                            transforms.RandomHorizontalFlip(p=0.2),
                            transforms.RandomVerticalFlip(p=0.2),
                            transforms.RandomApply(
                                [
                                    transforms.ColorJitter(
                                        brightness=0.5,
                                        contrast=0.5,
                                        saturation=0.5,
                                        hue=0.15,
                                    )
                                ],
                                p=0.9,
                            ),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.8))], p=0.3),
                            transforms.RandomErasing(
                                p=0.25,
                                scale=(0.02, 0.25),
                                ratio=(0.3, 3.3),
                                value="random",
                                inplace=False,
                            ),
                        ]
                    )
                else:
                    train_transform = transforms.Compose(
                        [
                            transforms.RandomResizedCrop(
                                size=32, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
                            ),
                            transforms.RandomHorizontalFlip(p=0.1),
                            transforms.RandomVerticalFlip(p=0.1),
                            transforms.ColorJitter(
                                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                            ),
                            transforms.RandomGrayscale(p=0.05),
                        ]
                    )
        else:
            jitter = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
            )
            grayscale = transforms.RandomGrayscale(p=0.05)

            def page_transform(image, mask):
                image, mask = _page_random_resized_crop(image, mask)
                if random.random() < 0.2:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                if random.random() < 0.2:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)
                image = jitter(image)
                image = grayscale(image)
                return image, mask

    train_dataset, val_dataset = build_datasets(
        data_cfg, model_cfg, train_transform, page_transform, eval_only
    )

    if model_cfg.num_classes <= 0:
        if train_dataset is not None and train_dataset.class_to_idx:
            model_cfg = replace(model_cfg, num_classes=len(train_dataset.class_to_idx))
        elif val_dataset is not None and val_dataset.class_to_idx:
            model_cfg = replace(model_cfg, num_classes=len(val_dataset.class_to_idx))
        else:
            raise ValueError("num_classes must be set when using CSV inputs")
        args.num_classes = model_cfg.num_classes
    if train_dataset is not None:
        summarize_dataset("Train", train_dataset)
    if val_dataset is not None:
        summarize_dataset("Val", val_dataset)
    if data_cfg.input_normalization == "none":
        identity_mean = torch.zeros(1, 3, 1, 1, dtype=torch.float32)
        identity_std = torch.ones(1, 3, 1, 1, dtype=torch.float32)
        for dataset in (train_dataset, val_dataset):
            if dataset is not None:
                dataset.mean = identity_mean
                dataset.std = identity_std
        print("Input normalization | mode=none | using raw [0,1] tensors")
    else:
        print("Input normalization | mode=imagenet")

    if val_dataset is None:
        raise ValueError("Validation dataset could not be constructed")
    train_loader, val_loader = build_loaders(data_cfg, train_dataset, val_dataset)

    model = Patchformer(model_cfg).to(device)
    print_model_summary(model)
    print(
        "[debug-model] final_embedding_dim="
        f"{getattr(model.decoder, 'cls_dim', 'unknown')} "
        f"(decoder={model_cfg.decoder_type}, d_out={model_cfg.d_out}, "
        f"n_head={model_cfg.n_head}, d_head={model_cfg.d_head})"
    )

    def _extract_eval_embeddings_for_dataset(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
        if split == "val":
            dataset = val_dataset
            loader = val_loader
            gray = data_cfg.val_gray
        elif split == "train":
            dataset = train_dataset
            loader = train_loader
            gray = False
        else:
            raise ValueError(f"Unknown split: {split}")
        if dataset is None:
            raise ValueError(f"Dataset for split '{split}' is not available")
        if eval_cfg.full_page_eval:
            print(
                f"Eval embedding mode | split={split} | full_page_eval=True | "
                f"feature_type={eval_cfg.val_features}"
            )
            eval_bs = data_cfg.eval_batch_size if data_cfg.eval_batch_size > 0 else data_cfg.batch_size
            return extract_embeddings_full_page(
                model,
                dataset,
                device,
                train_cfg.use_amp,
                eval_cfg.val_features,
                eval_cfg.val_include_queries,
                show_progress=not eval_cfg.no_progress,
                val_gray=gray,
                resize_height=data_cfg.full_image_height,
                resize_width=data_cfg.full_image_width,
                batch_size=eval_bs,
                mean=getattr(dataset, "mean", None),
                std=getattr(dataset, "std", None),
                cache_dir=eval_cfg.dump_tmp,
                cache_prefix=split,
                cache_chunk_size=eval_cfg.dump_embeddings_chunk_size,
                recompute_cache=eval_cfg.recompute_embeddings,
            )
        if loader is None:
            raise ValueError(f"Loader for split '{split}' is not available")
        return extract_embeddings_(
            model,
            loader,
            device,
            train_cfg.use_amp,
            eval_cfg.val_features,
            eval_cfg.val_include_queries,
            data_cfg.split_patches_to_samples,
            show_progress=not eval_cfg.no_progress,
        )

    arcface_state = None
    resume_epoch = 0
    resume_global_step = 0
    resume_optimizer_state = None
    resume_scaler_state = None
    resume_best_val = None
    if eval_cfg.resume_weights or eval_cfg.checkpoint:
        print("Attempting to Resume Weights")
        ckpt_path = eval_cfg.resume_weights or eval_cfg.checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        load_state = state
        if eval_cfg.resume_weights:
            load_state = {k: v for k, v in state.items()}
        filtered, skipped_unexpected, skipped_mismatch, inserted = _filter_state_dict(
            load_state, model, train_cfg.insert_weights_on_mismatch
        )
        if eval_cfg.eval_only:
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(
                f"Loaded checkpoint (eval-only) from {ckpt_path} | "
                f"missing {len(missing)} {_format_keys(missing)} | "
                f"unexpected {len(unexpected)} {_format_keys(unexpected)} | "
                f"inserted {len(inserted)} {_format_keys(inserted)} | "
                f"skipped {len(skipped_unexpected)} unexpected "
                f"{_format_keys(skipped_unexpected)} | "
                f"skipped {len(skipped_mismatch)} mismatched "
                f"{_format_keys(skipped_mismatch)}"
            )
        else:
            tag = "resume weights" if eval_cfg.resume_weights else "checkpoint"
            ckpt_epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
            missing, unexpected = model.load_state_dict(filtered, strict=False)
            print(
                f"Loaded {tag} from {ckpt_path} (epoch {ckpt_epoch}) | "
                f"missing {len(missing)} {_format_keys(missing)} | "
                f"unexpected {len(unexpected)} {_format_keys(unexpected)} | "
                f"inserted {len(inserted)} {_format_keys(inserted)} | "
                f"skipped {len(skipped_unexpected)} unexpected "
                f"{_format_keys(skipped_unexpected)} | "
                f"skipped {len(skipped_mismatch)} mismatched "
                f"{_format_keys(skipped_mismatch)}"
            )
        if isinstance(ckpt, dict) and "arcface" in ckpt:
            arcface_state = ckpt["arcface"]
        if not eval_cfg.eval_only and eval_cfg.checkpoint and isinstance(ckpt, dict):
            resume_epoch = int(ckpt.get("epoch", 0) or 0)
            resume_global_step = int(ckpt.get("global_step", 0) or 0)
            resume_optimizer_state = ckpt.get("optimizer")
            resume_scaler_state = ckpt.get("scaler")
            if "val_acc" in ckpt and ckpt["val_acc"] is not None:
                resume_best_val = float(ckpt["val_acc"])

    arcface = ArcFaceLoss(
        num_classes=model_cfg.num_classes,
        embedding_dim=model.decoder.cls_dim,
        margin=train_cfg.arcface_margin,
        scale=train_cfg.arcface_scale,
        mlp_dim=train_cfg.arcface_mlp_dim,
    ).to(device)
    if arcface_state is not None:
        arcface_filtered, arcface_unexpected, arcface_mismatched, _ = _filter_state_dict(
            arcface_state, arcface, False
        )
        arcface_missing, arcface_unexpected_load = arcface.load_state_dict(
            arcface_filtered, strict=False
        )
        print(
            "Loaded arcface weights | "
            f"missing {len(arcface_missing)} {_format_keys(arcface_missing)} | "
            f"unexpected {len(arcface_unexpected_load)} {_format_keys(arcface_unexpected_load)} | "
            f"skipped {len(arcface_unexpected)} unexpected "
            f"{_format_keys(arcface_unexpected)} | "
            f"skipped {len(arcface_mismatched)} mismatched "
            f"{_format_keys(arcface_mismatched)}"
        )
    triplet_weight = train_cfg.triplet_weight if train_cfg.triplet_loss else 0.0

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(arcface.parameters()),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=train_cfg.use_amp)
    if resume_optimizer_state is not None:
        try:
            optimizer.load_state_dict(resume_optimizer_state)
            print("Loaded optimizer state from checkpoint")
        except Exception as exc:
            print(f"Warning: failed to load optimizer state from checkpoint: {exc}")
    if resume_scaler_state is not None:
        try:
            scaler.load_state_dict(resume_scaler_state)
            print("Loaded AMP scaler state from checkpoint")
        except Exception as exc:
            print(f"Warning: failed to load AMP scaler state from checkpoint: {exc}")

    run_name = build_run_name(run_cfg.run_name)
    args.log_dir = _format_run_path(run_cfg.log_dir, run_name)
    args.save_path = _format_run_path(run_cfg.save_path, run_name)
    args.results_dir = _format_run_path(run_cfg.results_dir, run_name)
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    hparams = {
        "batch_size": data_cfg.batch_size,
        "eval_batch_size": data_cfg.eval_batch_size or data_cfg.batch_size,
        "num_patches": model_cfg.num_patches,
        "num_patches_eval": data_cfg.num_patches_eval,
        "split_patches_to_samples": data_cfg.split_patches_to_samples,
        "d_encoder": model_cfg.d_encoder,
        "d_out": model_cfg.d_out,
        "n_head": model_cfg.n_head,
        "d_head": model_cfg.d_head,
        "num_encoder_layers": model_cfg.num_encoder_layers,
        "num_decoder_layers": model_cfg.num_decoder_layers,
        "encoder_name": model_cfg.encoder_name,
        "encoder_freeze": model_cfg.encoder_freeze,
        "encoder_trainable_last_layers": model_cfg.encoder_trainable_last_layers,
        "decoder_type": model_cfg.decoder_type,
        "embedding_whitening_head": model_cfg.embedding_whitening_head,
        "num_queries": model_cfg.num_queries,
        "lr": train_cfg.lr,
        "lr_schedule": train_cfg.lr_schedule,
        "weight_decay": train_cfg.weight_decay,
        "dropout": model_cfg.dropout,
        "arcface_weight": train_cfg.arcface_weight,
        "arcface_margin": train_cfg.arcface_margin,
        "arcface_scale": train_cfg.arcface_scale,
        "triplet_loss": train_cfg.triplet_loss,
        "triplet_weight": train_cfg.triplet_weight,
        "triplet_margin": train_cfg.triplet_margin,
        "triplet_semi_hard": train_cfg.triplet_semi_hard,
        "ce_weight": train_cfg.ce_weight,
        "covariance_reg_weight": train_cfg.covariance_reg_weight,
        "freeze_backbone_epochs": train_cfg.freeze_backbone_epochs,
        "sequence_patch_augment_after_split": train_cfg.sequence_patch_augment_after_split,
        "sequence_patch_crop_size": train_cfg.sequence_patch_crop_size,
        "strong_patch_augment": train_cfg.strong_patch_augment,
    }
    hparams_text = "\n".join(f"{k}: {v}" for k, v in hparams.items())
    writer.add_text("hparams", hparams_text, resume_global_step)

    best_val = 0.0
    best_metrics: Dict[str, float] = {}
    last_metrics: Dict[str, float] = {}
    global_step = resume_global_step
    start_epoch = 1
    if eval_cfg.checkpoint and resume_epoch > 0:
        start_epoch = resume_epoch + 1
        if resume_best_val is not None:
            best_val = resume_best_val
        if global_step <= 0 and train_loader is not None:
            steps_per_epoch = len(train_loader) * max(1, train_cfg.repeat_per_epoch)
            global_step = resume_epoch * steps_per_epoch
        print(
            f"Resuming training from checkpoint epoch {resume_epoch} -> starting at epoch {start_epoch} "
            f"(global_step={global_step})"
        )
    start_time = time.time()
    classifier_only_epochs = max(0, train_cfg.train_only_classifier_epochs)
    freeze_backbone_epochs = max(0, train_cfg.freeze_backbone_epochs)
    classifier_only_active = False
    backbone_frozen_active = False
    if classifier_only_epochs > 0 and not eval_cfg.eval_only:
        set_classifier_only_training(model, True)
        classifier_only_active = True
        print(f"Training classifier only for first {classifier_only_epochs} epochs")
    elif freeze_backbone_epochs > 0 and not eval_cfg.eval_only:
        set_backbone_training(model, False)
        backbone_frozen_active = True
        print(f"Freezing backbone for first {freeze_backbone_epochs} epochs")

    if eval_cfg.eval_only:
        if eval_cfg.visualize_full_page_keep_heatmaps:
            out_dir = os.path.join(args.results_dir, run_name, "full_page_keep_heatmaps")
            visualize_full_page_keep_heatmaps(
                model,
                val_loader,
                val_dataset,
                device,
                out_dir,
                use_amp=train_cfg.use_amp,
                val_gray=data_cfg.val_gray,
                resize_height=data_cfg.full_image_height,
                resize_width=data_cfg.full_image_width,
                mean=getattr(val_dataset, "mean", None),
                std=getattr(val_dataset, "std", None),
            )
            print(f"Saved full-page keep heatmap overlays to {out_dir}")
            writer.close()
            return
        if eval_cfg.visualize_retrieval:
            out_dir = os.path.join(args.results_dir, run_name, "visualizations")
            visualize_retrieval_attention(
                model,
                val_loader,
                device,
                out_dir,
                annotate=not args.no_numerical_text,
            )
            print(f"Saved retrieval visualizations to {out_dir}")
            writer.close()
            return
        eval_summary: Dict[str, float] = {}
        if eval_cfg.knn_eval or eval_cfg.retrieval_eval:
            def _load_tensor(path: str) -> Optional[torch.Tensor]:
                if not path or not os.path.exists(path):
                    return None
                print(f"Loading cached embeddings: {path}", flush=True)
                return torch.load(path, map_location="cpu")

            def _maybe_dump_tensor(path: str, tensor: Optional[torch.Tensor]) -> None:
                if not path or tensor is None:
                    return
                os.makedirs(os.path.dirname(path), exist_ok=True)
                torch.save(tensor, path)

            dump_dir = eval_cfg.dump_tmp
            val_emb_path = os.path.join(dump_dir, "val_emb.pt") if dump_dir else ""
            train_emb_path = os.path.join(dump_dir, "train_emb.pt") if dump_dir else ""
            train_emb: Optional[torch.Tensor] = None
            val_labels = torch.tensor(
                [label for _, _, label in val_dataset.samples], dtype=torch.long
            )
            train_labels = None
            if train_dataset is not None:
                train_labels = torch.tensor(
                    [label for _, _, label in train_dataset.samples], dtype=torch.long
                )

            val_emb = None if eval_cfg.recompute_embeddings else _load_tensor(val_emb_path)
            if val_emb is None:
                print("Extracting validation embeddings...", flush=True)
                val_emb, val_labels = _extract_eval_embeddings_for_dataset("val")
                _maybe_dump_tensor(val_emb_path, val_emb)
                print(f"Validation embeddings ready: {tuple(val_emb.shape)}", flush=True)
            else:
                print(f"Validation embeddings cache hit: {tuple(val_emb.shape)}", flush=True)

            if (
                train_loader is not None
                and eval_cfg.retrieval_eval
                and not eval_cfg.whiten_test
                and not eval_cfg.whiten_none
            ):
                train_emb = None if eval_cfg.recompute_embeddings else _load_tensor(train_emb_path)
                if train_emb is None:
                    print("Extracting train embeddings for whitening...", flush=True)
                    train_emb, train_labels = _extract_eval_embeddings_for_dataset("train")
                    _maybe_dump_tensor(train_emb_path, train_emb)
                    print(f"Train embeddings ready: {tuple(train_emb.shape)}", flush=True)
                else:
                    print(f"Train embeddings cache hit: {tuple(train_emb.shape)}", flush=True)
            _log_val_decoder_attention_sample(
                model,
                val_loader,
                device,
                writer,
                step=0,
                annotate=not args.no_numerical_text,
                enabled=not data_cfg.full_image_input,
            )
            if eval_cfg.knn_eval:
                knn_acc = knn_accuracy_leave_one_out(
                    val_emb, val_labels, k=eval_cfg.knn_k
                )
                eval_summary["knn_acc"] = knn_acc
                writer.add_scalar("val/knn_acc", knn_acc, 0)
                print(f"Eval-only | knn acc {knn_acc:.4f}")
            if eval_cfg.retrieval_eval:
                if eval_cfg.gallery_growth_eval:
                    query_mask = _query_mask_excluding_labels(
                        val_labels, eval_cfg.retrieval_distractor_labels
                    )
                    growth = retrieval_map_progressive_gallery(
                        embeddings=val_emb,
                        labels=val_labels,
                        writer_chunk_size=eval_cfg.gallery_growth_writer_chunk,
                        min_writers=eval_cfg.gallery_growth_min_writers,
                        query_mask=query_mask if eval_cfg.retrieval_distractor_labels else None,
                        shuffle_writers=eval_cfg.gallery_growth_shuffle_writers,
                        seed=eval_cfg.gallery_growth_seed,
                        show_progress=not eval_cfg.no_progress,
                        ap_hist_bins=(
                            eval_cfg.gallery_growth_ap_hist_bins
                            if eval_cfg.gallery_growth_ap_hist
                            else 0
                        ),
                        ap_hist_full_only=eval_cfg.gallery_growth_ap_hist_full_only,
                        print_each=True,
                    )
                    if growth:
                        csv_path = eval_cfg.gallery_growth_csv_path
                        if not csv_path:
                            csv_path = os.path.join(
                                args.results_dir, run_name, "gallery_growth.csv"
                            )
                        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                        with open(csv_path, "w", encoding="utf-8") as f:
                            f.write(
                                "writers,samples,top1,map,queries_total,queries_with_relevant\n"
                            )
                            for row in growth:
                                f.write(
                                    f"{int(row['writers'])},{int(row['samples'])},"
                                    f"{row['top1']:.6f},{row['map']:.6f},"
                                    f"{int(row['queries_total'])},{int(row['queries_with_relevant'])}\n"
                                )
                        print(f"Eval-only | wrote gallery-growth curve to {csv_path}")
                        eval_summary["gallery_growth_points"] = float(len(growth))
                        eval_summary["gallery_growth_last_map"] = float(growth[-1]["map"])
                        eval_summary["gallery_growth_last_top1"] = float(growth[-1]["top1"])
                    writer.close()
                    return
                retrieval_top1, retrieval_map, retrieval_stats, not_within_top10 = (
                    retrieval_metrics_leave_one_out(
                        val_emb,
                        val_labels,
                        return_not_within_top10=bool(eval_cfg.copy_not_in_top10pct),
                        show_progress=not eval_cfg.no_progress,
                        progress_desc="retrieval raw",
                    )
                )
                retrieval_top1_whitened = None
                retrieval_map_whitened = None
                if eval_cfg.whiten_none:
                    print("Skipping whitening (--whiten-none).", flush=True)
                elif eval_cfg.whiten_test:
                    print("Applying test-set whitening...", flush=True)
                    val_emb_whitened = pca_whiten_embeddings(val_emb)
                elif train_emb is not None:
                    print("Fitting whitening on train embeddings and applying to val...", flush=True)
                    whiten_mean, whiten_mat = pca_whiten_fit(train_emb)
                    val_emb_whitened = pca_whiten_apply(val_emb, whiten_mean, whiten_mat)
                else:
                    print("Train embeddings unavailable; falling back to test-set whitening...", flush=True)
                    val_emb_whitened = pca_whiten_embeddings(val_emb)
                val_emb_whitened_test = pca_whiten_embeddings(val_emb)
                val_emb_whitened_test_norm = F.normalize(val_emb_whitened_test, dim=1)
                if not eval_cfg.whiten_none:
                    retrieval_top1_whitened, retrieval_map_whitened, _, _ = (
                        retrieval_metrics_leave_one_out(
                            val_emb_whitened,
                            val_labels,
                            return_not_within_top10=False,
                            show_progress=not eval_cfg.no_progress,
                            progress_desc="retrieval whitened",
                        )
                    )
                sgr_top1 = None
                sgr_map = None
                sgr_stats = None
                sgr_not_within_top10 = None
                if eval_cfg.sgr_reranking:
                    reranked_emb, _ = sgr_reranking(
                        val_emb_whitened_test_norm,
                        k=eval_cfg.sgr_k,
                        layer=eval_cfg.sgr_layers,
                        gamma=eval_cfg.sgr_gamma,
                    )
                    sgr_top1, sgr_map, sgr_stats, sgr_not_within_top10 = (
                        retrieval_metrics_leave_one_out(
                            reranked_emb,
                            val_labels,
                            return_not_within_top10=bool(eval_cfg.copy_not_in_top10pct),
                            show_progress=not eval_cfg.no_progress,
                            progress_desc="retrieval sgr",
                        )
                    )
                    # for layer in (1, 2, 3):
                    #     for k in (1, 2, 3):
                    #         for gamma in range(1,10,2):
                    #             gamma = gamma / 100
                    #             sweep_emb, _ = sgr_reranking(
                    #                 val_emb_whitened_test_norm,
                    #                 k=k,
                    #                 layer=layer,
                    #                 gamma=gamma,
                    #             )
                    #             sweep_top1, sweep_map, sweep_stats, _ = (
                    #                 retrieval_metrics_leave_one_out(
                    #                     sweep_emb,
                    #                     val_labels,
                    #                     return_not_within_top10=False,
                    #                 )
                    #             )
                    #             print(
                    #                 f"Eval-only | SGR sweep k={k} layer={layer} gamma={gamma:.1f} | "
                    #                 f"top1 {sweep_top1:.4f} mAP {sweep_map:.4f} | "
                    #                 f"singletons {sweep_stats.get('singletons', 0)} | "
                    #                 f"not_in_top10 {sweep_stats.get('closest_not_within_top10pct', 0)} "
                    #                 f"not_in_top20 {sweep_stats.get('closest_not_within_top20pct', 0)} "
                    #                 f"not_in_top50 {sweep_stats.get('closest_not_within_top50pct', 0)}"
                    #             )
                metrics_logged_before_topk = False
                if eval_cfg.visualize_retrieval_topk_hits:
                    msg = (
                        f"Eval-only | top1 {retrieval_top1:.4f} mAP {retrieval_map:.4f}"
                    )
                    if retrieval_top1_whitened is not None and retrieval_map_whitened is not None:
                        msg += (
                            f" | whitened top1 {retrieval_top1_whitened:.4f} "
                            f"mAP {retrieval_map_whitened:.4f}"
                        )
                    msg += (
                        f" | singletons {retrieval_stats.get('singletons', 0)} | "
                        f"not_in_top10 {retrieval_stats.get('closest_not_within_top10pct', 0)} "
                        f"not_in_top20 {retrieval_stats.get('closest_not_within_top20pct', 0)} "
                        f"not_in_top50 {retrieval_stats.get('closest_not_within_top50pct', 0)}"
                    )
                    print(msg)
                    if eval_cfg.sgr_reranking and sgr_top1 is not None:
                        print(
                            f"Eval-only | SGR top1 {sgr_top1:.4f} mAP {sgr_map:.4f} | "
                            f"singletons {sgr_stats.get('singletons', 0)} | "
                            f"not_in_top10 {sgr_stats.get('closest_not_within_top10pct', 0)} "
                            f"not_in_top20 {sgr_stats.get('closest_not_within_top20pct', 0)} "
                            f"not_in_top50 {sgr_stats.get('closest_not_within_top50pct', 0)}"
                        )
                    metrics_logged_before_topk = True
                retrieval_top1_no_distractors = None
                retrieval_map_no_distractors = None
                sgr_top1_no_distractors = None
                sgr_map_no_distractors = None
                query_mask = _query_mask_excluding_labels(
                    val_labels, eval_cfg.retrieval_distractor_labels
                )
                if query_mask is not None and query_mask.any().item():
                    retrieval_top1_no_distractors, retrieval_map_no_distractors, _, _ = (
                        retrieval_metrics_leave_one_out(
                            val_emb,
                            val_labels,
                            query_mask=query_mask,
                            return_not_within_top10=False,
                            show_progress=not eval_cfg.no_progress,
                            progress_desc="retrieval raw no-distractors",
                        )
                    )
                    if eval_cfg.sgr_reranking and reranked_emb is not None:
                        sgr_top1_no_distractors, sgr_map_no_distractors, _, _ = (
                            retrieval_metrics_leave_one_out(
                                reranked_emb,
                                val_labels,
                                query_mask=query_mask,
                                return_not_within_top10=False,
                                show_progress=not eval_cfg.no_progress,
                                progress_desc="retrieval sgr no-distractors",
                            )
                        )
                if eval_cfg.copy_not_in_top10pct:
                    copy_indices = not_within_top10
                    if eval_cfg.sgr_reranking and sgr_not_within_top10 is not None:
                        copy_indices = sgr_not_within_top10
                    copied = _copy_indices_to_dir(
                        copy_indices, val_dataset, eval_cfg.copy_not_in_top10pct
                    )
                    print(
                        f"Eval-only | copied {copied} not_in_top10 images to "
                        f"{eval_cfg.copy_not_in_top10pct}"
                    )
                if eval_cfg.gallery_growth_eval:
                    growth = retrieval_map_progressive_gallery(
                        embeddings=val_emb,
                        labels=val_labels,
                        writer_chunk_size=eval_cfg.gallery_growth_writer_chunk,
                        min_writers=eval_cfg.gallery_growth_min_writers,
                        query_mask=query_mask if eval_cfg.retrieval_distractor_labels else None,
                        shuffle_writers=eval_cfg.gallery_growth_shuffle_writers,
                        seed=eval_cfg.gallery_growth_seed,
                        show_progress=not eval_cfg.no_progress,
                        ap_hist_bins=(
                            eval_cfg.gallery_growth_ap_hist_bins
                            if eval_cfg.gallery_growth_ap_hist
                            else 0
                        ),
                        ap_hist_full_only=eval_cfg.gallery_growth_ap_hist_full_only,
                    )
                    if growth:
                        csv_path = eval_cfg.gallery_growth_csv_path
                        if not csv_path:
                            csv_path = os.path.join(
                                args.results_dir, run_name, "gallery_growth.csv"
                            )
                        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                        with open(csv_path, "w", encoding="utf-8") as f:
                            f.write(
                                "writers,samples,top1,map,queries_total,queries_with_relevant\n"
                            )
                            for row in growth:
                                f.write(
                                    f"{int(row['writers'])},{int(row['samples'])},"
                                    f"{row['top1']:.6f},{row['map']:.6f},"
                                    f"{int(row['queries_total'])},{int(row['queries_with_relevant'])}\n"
                                )
                        print(f"Eval-only | wrote gallery-growth curve to {csv_path}")
                        for row in growth:
                            print(
                                "Gallery growth | "
                                f"writers {int(row['writers'])} | "
                                f"samples {int(row['samples'])} | "
                                f"top1 {row['top1']:.4f} mAP {row['map']:.4f}"
                            )
                            if eval_cfg.gallery_growth_ap_hist and "ap_hist" in row:
                                print(
                                    "Gallery growth AP hist | "
                                    f"writers {int(row['writers'])} | bins {row['ap_hist']}"
                                )
                        eval_summary["gallery_growth_points"] = float(len(growth))
                        eval_summary["gallery_growth_last_map"] = float(growth[-1]["map"])
                        eval_summary["gallery_growth_last_top1"] = float(growth[-1]["top1"])
                if eval_cfg.visualize_retrieval_topk_hits:
                    topk_viz_dir = eval_cfg.retrieval_topk_viz_dir
                    if not topk_viz_dir:
                        topk_viz_dir = os.path.join(
                            args.results_dir, run_name, "retrieval_topk_hits"
                        )
                    viz_count = visualize_retrieval_topk_hits(
                        embeddings=val_emb,
                        labels=val_labels,
                        dataset=val_dataset,
                        output_dir=topk_viz_dir,
                        k=max(1, eval_cfg.retrieval_topk_viz_k),
                    )
                    eval_summary["retrieval_topk_viz_count"] = float(viz_count)
                    print(
                        f"Eval-only | wrote {viz_count} retrieval top-k visualizations to {topk_viz_dir}"
                    )
                eval_summary.update(
                    {
                        "retrieval_top1": retrieval_top1,
                        "retrieval_map": retrieval_map,
                    }
                )
                if retrieval_top1_whitened is not None and retrieval_map_whitened is not None:
                    eval_summary["retrieval_top1_whitened"] = retrieval_top1_whitened
                    eval_summary["retrieval_map_whitened"] = retrieval_map_whitened
                for key, value in retrieval_stats.items():
                    eval_summary[f"retrieval_{key}"] = value
                if retrieval_top1_no_distractors is not None:
                    eval_summary["retrieval_top1_no_distractors"] = retrieval_top1_no_distractors
                if retrieval_map_no_distractors is not None:
                    eval_summary["retrieval_map_no_distractors"] = retrieval_map_no_distractors
                if sgr_top1 is not None:
                    eval_summary["retrieval_top1_sgr"] = sgr_top1
                if sgr_map is not None:
                    eval_summary["retrieval_map_sgr"] = sgr_map
                if sgr_stats is not None:
                    for key, value in sgr_stats.items():
                        eval_summary[f"retrieval_sgr_{key}"] = value
                if eval_cfg.sgr_reranking and sgr_top1_no_distractors is not None:
                    eval_summary["retrieval_top1_sgr_no_distractors"] = sgr_top1_no_distractors
                if eval_cfg.sgr_reranking and sgr_map_no_distractors is not None:
                    eval_summary["retrieval_map_sgr_no_distractors"] = sgr_map_no_distractors
                writer.add_scalar("val/retrieval_top1", retrieval_top1, 0)
                writer.add_scalar("val/retrieval_map", retrieval_map, 0)
                if retrieval_top1_whitened is not None:
                    writer.add_scalar("val/retrieval_top1_whitened", retrieval_top1_whitened, 0)
                if retrieval_map_whitened is not None:
                    writer.add_scalar("val/retrieval_map_whitened", retrieval_map_whitened, 0)
                if retrieval_top1_no_distractors is not None:
                    writer.add_scalar("val/retrieval_top1_no_distractors", retrieval_top1_no_distractors, 0)
                if retrieval_map_no_distractors is not None:
                    writer.add_scalar("val/retrieval_map_no_distractors", retrieval_map_no_distractors, 0)
                if sgr_top1 is not None:
                    writer.add_scalar("val/retrieval_top1_sgr", sgr_top1, 0)
                if sgr_map is not None:
                    writer.add_scalar("val/retrieval_map_sgr", sgr_map, 0)
                if eval_cfg.sgr_reranking and sgr_top1_no_distractors is not None:
                    writer.add_scalar("val/retrieval_top1_sgr_no_distractors", sgr_top1_no_distractors, 0)
                if eval_cfg.sgr_reranking and sgr_map_no_distractors is not None:
                    writer.add_scalar("val/retrieval_map_sgr_no_distractors", sgr_map_no_distractors, 0)
                if not metrics_logged_before_topk:
                    msg = (
                        f"Eval-only | top1 {retrieval_top1:.4f} mAP {retrieval_map:.4f}"
                    )
                    if retrieval_top1_whitened is not None and retrieval_map_whitened is not None:
                        msg += (
                            f" | whitened top1 {retrieval_top1_whitened:.4f} "
                            f"mAP {retrieval_map_whitened:.4f}"
                        )
                    msg += (
                        f" | singletons {retrieval_stats.get('singletons', 0)} | "
                        f"not_in_top10 {retrieval_stats.get('closest_not_within_top10pct', 0)} "
                        f"not_in_top20 {retrieval_stats.get('closest_not_within_top20pct', 0)} "
                        f"not_in_top50 {retrieval_stats.get('closest_not_within_top50pct', 0)}"
                    )
                    print(msg)
                    if eval_cfg.sgr_reranking and sgr_top1 is not None:
                        print(
                            f"Eval-only | SGR top1 {sgr_top1:.4f} mAP {sgr_map:.4f} | "
                            f"singletons {sgr_stats.get('singletons', 0)} | "
                            f"not_in_top10 {sgr_stats.get('closest_not_within_top10pct', 0)} "
                            f"not_in_top20 {sgr_stats.get('closest_not_within_top20pct', 0)} "
                            f"not_in_top50 {sgr_stats.get('closest_not_within_top50pct', 0)}"
                        )
        else:
            val_metrics = evaluate(
                model,
                arcface,
                val_loader,
                device,
                train_cfg.ce_weight,
                train_cfg.arcface_weight,
                triplet_weight,
                train_cfg.triplet_margin,
                train_cfg.triplet_soft_margin,
                train_cfg.triplet_semi_hard,
                train_cfg.log_interval,
                data_cfg.split_patches_to_samples,
                writer=writer,
                tb_step=0,
                log_decoder_attention=True,
                annotate_decoder_attention=not args.no_numerical_text,
            )
            eval_summary.update(val_metrics)
            writer.add_scalar("val/loss", val_metrics["loss"], 0)
            writer.add_scalar("val/acc", val_metrics["acc"], 0)
            if train_cfg.arcface_weight > 0.0:
                writer.add_scalar("val/arcface_loss", val_metrics["arcface_loss"], 0)
            if triplet_weight > 0.0:
                writer.add_scalar("val/triplet_loss", val_metrics["triplet_loss"], 0)
            print(
                f"Eval-only | val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
                f"top5 {val_metrics['top5']:.4f} top10 {val_metrics['top10']:.4f} "
                f"top50 {val_metrics['top50']:.4f}"
            )
        if eval_summary:
            _write_results(args.results_dir, run_name, eval_summary, args)
        writer.close()
        return

    total_train_steps = len(train_loader) * max(1, train_cfg.repeat_per_epoch) * train_cfg.epochs
    if start_epoch > train_cfg.epochs:
        print(
            f"Checkpoint already at epoch {resume_epoch}, which is >= configured epochs "
            f"({train_cfg.epochs}). Nothing to train."
        )
        if last_metrics or best_metrics:
            summary = {"best": best_metrics, "last": last_metrics}
            _write_results(args.results_dir, run_name, summary, args)
        writer.close()
        return
    if (
        train_cfg.validate_before_first_epoch
        and train_cfg.val_every > 0
        and start_epoch == 1
    ):
        arcface_weight_epoch = _scheduled_value(
            global_step,
            train_cfg.arcface_weight_schedule,
            train_cfg.arcface_weight_start,
            train_cfg.arcface_weight,
            train_cfg.arcface_weight_warmup_steps,
        )
        triplet_weight_epoch = _scheduled_value(
            global_step,
            train_cfg.triplet_weight_schedule,
            train_cfg.triplet_weight_start,
            train_cfg.triplet_weight,
            train_cfg.triplet_weight_warmup_steps,
        )
        triplet_margin_epoch = _scheduled_value(
            global_step,
            train_cfg.triplet_margin_schedule,
            train_cfg.triplet_margin_start,
            train_cfg.triplet_margin,
            train_cfg.triplet_margin_warmup_steps,
        )
        knn_acc = None
        retrieval_top1 = None
        retrieval_map = None
        retrieval_stats = None
        retrieval_top1_whitened = None
        retrieval_map_whitened = None
        retrieval_top1_no_distractors = None
        retrieval_map_no_distractors = None
        val_metrics = {
            "loss": float("nan"),
            "arcface_loss": float("nan"),
            "triplet_loss": float("nan"),
            "top5": float("nan"),
            "top10": float("nan"),
            "top50": float("nan"),
            "acc": 0.0,
        }
        if eval_cfg.knn_eval or eval_cfg.retrieval_eval:
            val_emb, val_labels = _extract_eval_embeddings_for_dataset("val")
            if eval_cfg.knn_eval:
                knn_acc = knn_accuracy_leave_one_out(val_emb, val_labels, k=eval_cfg.knn_k)
            if eval_cfg.retrieval_eval:
                retrieval_top1, retrieval_map, retrieval_stats, not_within_top10 = (
                    retrieval_metrics_leave_one_out(
                        val_emb,
                        val_labels,
                        return_not_within_top10=bool(eval_cfg.copy_not_in_top10pct),
                    )
                )
                if not eval_cfg.whiten_none:
                    val_emb_whitened = pca_whiten_embeddings(val_emb)
                    retrieval_top1_whitened, retrieval_map_whitened, _, _ = (
                        retrieval_metrics_leave_one_out(
                            val_emb_whitened, val_labels, return_not_within_top10=False
                        )
                    )
                query_mask = _query_mask_excluding_labels(
                    val_labels, eval_cfg.retrieval_distractor_labels
                )
                if query_mask is not None and query_mask.any().item():
                    retrieval_top1_no_distractors, retrieval_map_no_distractors, _, _ = (
                        retrieval_metrics_leave_one_out(
                            val_emb,
                            val_labels,
                            query_mask=query_mask,
                            return_not_within_top10=False,
                        )
                    )
                if eval_cfg.copy_not_in_top10pct:
                    output_dir = os.path.join(eval_cfg.copy_not_in_top10pct, "epoch_000")
                    copied = _copy_indices_to_dir(not_within_top10, val_dataset, output_dir)
                    print(
                        f"Epoch 000 | copied {copied} not_in_top10 images to {output_dir}"
                    )
        if not eval_cfg.knn_eval and not eval_cfg.retrieval_eval:
            val_metrics = evaluate(
                model,
                arcface,
                val_loader,
                device,
                train_cfg.ce_weight,
                arcface_weight_epoch,
                triplet_weight_epoch,
                triplet_margin_epoch,
                train_cfg.triplet_soft_margin,
                train_cfg.triplet_semi_hard,
                train_cfg.log_interval,
                data_cfg.split_patches_to_samples,
                writer=writer,
                tb_step=0,
                log_decoder_attention=True,
                annotate_decoder_attention=not args.no_numerical_text,
            )
        else:
            _log_val_decoder_attention_sample(
                model,
                val_loader,
                device,
                writer,
                step=0,
                annotate=not args.no_numerical_text,
                enabled=not data_cfg.full_image_input,
            )
        if not eval_cfg.knn_eval:
            writer.add_scalar("val/loss", val_metrics["loss"], 0)
            writer.add_scalar("val/acc", val_metrics["acc"], 0)
        if arcface_weight_epoch > 0.0:
            writer.add_scalar("val/arcface_loss", val_metrics["arcface_loss"], 0)
        if triplet_weight_epoch > 0.0:
            writer.add_scalar("val/triplet_loss", val_metrics["triplet_loss"], 0)
        if knn_acc is not None:
            writer.add_scalar("val/knn_acc", knn_acc, 0)
        if retrieval_top1 is not None:
            writer.add_scalar("val/retrieval_top1", retrieval_top1, 0)
        if retrieval_map is not None:
            writer.add_scalar("val/retrieval_map", retrieval_map, 0)
        if retrieval_top1_whitened is not None:
            writer.add_scalar("val/retrieval_top1_whitened", retrieval_top1_whitened, 0)
        if retrieval_map_whitened is not None:
            writer.add_scalar("val/retrieval_map_whitened", retrieval_map_whitened, 0)
        if retrieval_top1_no_distractors is not None:
            writer.add_scalar("val/retrieval_top1_no_distractors", retrieval_top1_no_distractors, 0)
        if retrieval_map_no_distractors is not None:
            writer.add_scalar("val/retrieval_map_no_distractors", retrieval_map_no_distractors, 0)
        if eval_cfg.retrieval_eval:
            msg = (
                f"Epoch 000 | "
                f"top1 {retrieval_top1:.4f} mAP {retrieval_map:.4f}"
            )
            if retrieval_top1_whitened is not None and retrieval_map_whitened is not None:
                msg += (
                    f" | whitened top1 {retrieval_top1_whitened:.4f} "
                    f"mAP {retrieval_map_whitened:.4f}"
                )
            msg += (
                f" | singletons {0 if retrieval_stats is None else retrieval_stats.get('singletons', 0)} | "
                f"not_in_top10 {0 if retrieval_stats is None else retrieval_stats.get('closest_not_within_top10pct', 0)} "
                f"not_in_top20 {0 if retrieval_stats is None else retrieval_stats.get('closest_not_within_top20pct', 0)} "
                f"not_in_top50 {0 if retrieval_stats is None else retrieval_stats.get('closest_not_within_top50pct', 0)}"
            )
            print(msg)
        elif eval_cfg.knn_eval:
            print(f"Epoch 000 | knn acc {knn_acc:.4f}")
        else:
            print(
                f"Epoch 000 | val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
                f"top5 {val_metrics['top5']:.4f} top10 {val_metrics['top10']:.4f} "
                f"top50 {val_metrics['top50']:.4f}"
            )
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        if classifier_only_epochs > 0:
            should_train_classifier_only = epoch <= classifier_only_epochs
            if should_train_classifier_only != classifier_only_active:
                set_classifier_only_training(model, should_train_classifier_only)
                classifier_only_active = should_train_classifier_only
                if classifier_only_active:
                    print(f"Epoch {epoch:03d} | classifier-only training enabled")
                else:
                    print(f"Epoch {epoch:03d} | full model training enabled")
        if not classifier_only_active and freeze_backbone_epochs > 0:
            should_freeze_backbone = epoch <= freeze_backbone_epochs
            if should_freeze_backbone != backbone_frozen_active:
                set_backbone_training(model, not should_freeze_backbone)
                backbone_frozen_active = should_freeze_backbone
                if backbone_frozen_active:
                    print(f"Epoch {epoch:03d} | backbone frozen")
                else:
                    print(f"Epoch {epoch:03d} | backbone unfrozen")
        train_metrics, global_step = train_one_epoch(
            model,
            arcface,
            train_loader,
            optimizer,
            scaler,
            device,
            train_cfg,
            total_train_steps,
            writer,
            global_step,
            epoch,
            train_cfg.epochs,
            train_dataset.mean,
            train_dataset.std,
            train_cfg.repeat_per_epoch,
            data_cfg.split_patches_to_samples,
            log_batch_images=not data_cfg.full_image_input,
        )

        knn_acc = None
        retrieval_top1 = None
        retrieval_map = None
        retrieval_stats = None
        retrieval_top1_whitened = None
        retrieval_map_whitened = None
        retrieval_top1_no_distractors = None
        retrieval_map_no_distractors = None
        val_metrics = {
            "loss": float("nan"),
            "arcface_loss": float("nan"),
            "triplet_loss": float("nan"),
            "top5": float("nan"),
            "top10": float("nan"),
            "top50": float("nan"),
            "acc": 0.0,
        }
        if train_cfg.val_every > 0 and epoch % train_cfg.val_every == 0:
            arcface_weight_epoch = _scheduled_value(
                global_step,
                train_cfg.arcface_weight_schedule,
                train_cfg.arcface_weight_start,
                train_cfg.arcface_weight,
                train_cfg.arcface_weight_warmup_steps,
            )
            triplet_weight_epoch = _scheduled_value(
                global_step,
                train_cfg.triplet_weight_schedule,
                train_cfg.triplet_weight_start,
                train_cfg.triplet_weight,
                train_cfg.triplet_weight_warmup_steps,
            )
            triplet_margin_epoch = _scheduled_value(
                global_step,
                train_cfg.triplet_margin_schedule,
                train_cfg.triplet_margin_start,
                train_cfg.triplet_margin,
                train_cfg.triplet_margin_warmup_steps,
            )
            if eval_cfg.knn_eval or eval_cfg.retrieval_eval:
                val_emb, val_labels = _extract_eval_embeddings_for_dataset("val")
                if eval_cfg.knn_eval:
                    knn_acc = knn_accuracy_leave_one_out(
                        val_emb, val_labels, k=eval_cfg.knn_k
                    )
                if eval_cfg.retrieval_eval:
                    retrieval_top1, retrieval_map, retrieval_stats, not_within_top10 = (
                        retrieval_metrics_leave_one_out(
                            val_emb,
                            val_labels,
                            return_not_within_top10=bool(eval_cfg.copy_not_in_top10pct),
                        )
                    )
                    if not eval_cfg.whiten_none:
                        val_emb_whitened = pca_whiten_embeddings(val_emb)
                        retrieval_top1_whitened, retrieval_map_whitened, _, _ = (
                            retrieval_metrics_leave_one_out(
                                val_emb_whitened, val_labels, return_not_within_top10=False
                            )
                        )
                    query_mask = _query_mask_excluding_labels(
                        val_labels, eval_cfg.retrieval_distractor_labels
                    )
                    if query_mask is not None and query_mask.any().item():
                        retrieval_top1_no_distractors, retrieval_map_no_distractors, _, _ = (
                            retrieval_metrics_leave_one_out(
                                val_emb,
                                val_labels,
                                query_mask=query_mask,
                                return_not_within_top10=False,
                            )
                        )
                    if eval_cfg.copy_not_in_top10pct:
                        output_dir = os.path.join(
                            eval_cfg.copy_not_in_top10pct, f"epoch_{epoch:03d}"
                        )
                        copied = _copy_indices_to_dir(
                            not_within_top10, val_dataset, output_dir
                        )
                        print(
                            f"Epoch {epoch:03d} | copied {copied} not_in_top10 images to "
                            f"{output_dir}"
                        )
            if not eval_cfg.knn_eval and not eval_cfg.retrieval_eval:
                val_metrics = evaluate(
                    model,
                    arcface,
                    val_loader,
                    device,
                    train_cfg.ce_weight,
                    arcface_weight_epoch,
                    triplet_weight_epoch,
                    triplet_margin_epoch,
                    train_cfg.triplet_soft_margin,
                    train_cfg.triplet_semi_hard,
                    train_cfg.log_interval,
                    data_cfg.split_patches_to_samples,
                    writer=writer,
                    tb_step=epoch,
                    log_decoder_attention=True,
                    annotate_decoder_attention=not args.no_numerical_text,
                )
            else:
                _log_val_decoder_attention_sample(
                    model,
                    val_loader,
                    device,
                    writer,
                    step=epoch,
                    annotate=not args.no_numerical_text,
                    enabled=not data_cfg.full_image_input,
                )

        if train_cfg.val_every > 0 and epoch % train_cfg.val_every == 0:
            if eval_cfg.retrieval_eval:
                last_metrics = {
                    "retrieval_top1": retrieval_top1,
                    "retrieval_map": retrieval_map,
                }
                if retrieval_top1_whitened is not None and retrieval_map_whitened is not None:
                    last_metrics["retrieval_top1_whitened"] = retrieval_top1_whitened
                    last_metrics["retrieval_map_whitened"] = retrieval_map_whitened
                if retrieval_stats is not None:
                    for key, value in retrieval_stats.items():
                        last_metrics[f"retrieval_{key}"] = value
                if retrieval_top1_no_distractors is not None:
                    last_metrics["retrieval_top1_no_distractors"] = retrieval_top1_no_distractors
                if retrieval_map_no_distractors is not None:
                    last_metrics["retrieval_map_no_distractors"] = retrieval_map_no_distractors
            elif eval_cfg.knn_eval:
                last_metrics = {"knn_acc": knn_acc}
            else:
                last_metrics = dict(val_metrics)
            if not eval_cfg.knn_eval:
                writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                writer.add_scalar("val/acc", val_metrics["acc"], epoch)
            if arcface_weight_epoch > 0.0:
                writer.add_scalar("val/arcface_loss", val_metrics["arcface_loss"], epoch)
            if triplet_weight_epoch > 0.0:
                writer.add_scalar("val/triplet_loss", val_metrics["triplet_loss"], epoch)
            if knn_acc is not None:
                writer.add_scalar("val/knn_acc", knn_acc, epoch)
            if retrieval_top1 is not None:
                writer.add_scalar("val/retrieval_top1", retrieval_top1, epoch)
            if retrieval_map is not None:
                writer.add_scalar("val/retrieval_map", retrieval_map, epoch)
            if retrieval_top1_whitened is not None:
                writer.add_scalar("val/retrieval_top1_whitened", retrieval_top1_whitened, epoch)
            if retrieval_map_whitened is not None:
                writer.add_scalar("val/retrieval_map_whitened", retrieval_map_whitened, epoch)
            if retrieval_top1_no_distractors is not None:
                writer.add_scalar("val/retrieval_top1_no_distractors", retrieval_top1_no_distractors, epoch)
            if retrieval_map_no_distractors is not None:
                writer.add_scalar("val/retrieval_map_no_distractors", retrieval_map_no_distractors, epoch)

        if train_cfg.val_every > 0 and epoch % train_cfg.val_every == 0:
            if retrieval_top1 is not None:
                current_score = retrieval_top1
            elif knn_acc is not None:
                current_score = knn_acc
            else:
                current_score = val_metrics["acc"]
            if current_score > best_val:
                best_val = current_score
                best_metrics = dict(last_metrics)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arcface": arcface.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_acc": best_val,
                    },
                    args.save_path,
                )
            if train_cfg.checkpoint_every > 0 and epoch % train_cfg.checkpoint_every == 0:
                save_root, save_ext = os.path.splitext(args.save_path)
                save_ext = save_ext or ".pt"
                interval_path = f"{save_root}_epoch{epoch:03d}{save_ext}"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "arcface": arcface.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_acc": current_score,
                        "best_val": best_val,
                    },
                    interval_path,
                )

        elapsed = time.time() - start_time
        if train_cfg.val_every > 0 and epoch % train_cfg.val_every == 0:
            if eval_cfg.retrieval_eval:
                msg = (
                    f"Epoch {epoch:03d} | "
                    f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
                    f"top1 {retrieval_top1:.4f} mAP {retrieval_map:.4f}"
                )
                if retrieval_top1_whitened is not None and retrieval_map_whitened is not None:
                    msg += (
                        f" | whitened top1 {retrieval_top1_whitened:.4f} "
                        f"mAP {retrieval_map_whitened:.4f}"
                    )
                msg += (
                    f" | singletons {0 if retrieval_stats is None else retrieval_stats.get('singletons', 0)} | "
                    f"not_in_top10 {0 if retrieval_stats is None else retrieval_stats.get('closest_not_within_top10pct', 0)} "
                    f"not_in_top20 {0 if retrieval_stats is None else retrieval_stats.get('closest_not_within_top20pct', 0)} "
                    f"not_in_top50 {0 if retrieval_stats is None else retrieval_stats.get('closest_not_within_top50pct', 0)} | "
                    f"elapsed {elapsed/60:.1f}m"
                )
                print(msg)
            elif eval_cfg.knn_eval:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
                    f"knn acc {knn_acc:.4f} | "
                    f"elapsed {elapsed/60:.1f}m"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | "
                    f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
                    f"val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.4f} | "
                    f"top5 {val_metrics['top5']:.4f} top10 {val_metrics['top10']:.4f} "
                    f"top50 {val_metrics['top50']:.4f} | "
                    f"elapsed {elapsed/60:.1f}m"
                )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.4f} | "
                f"val skipped | "
                f"elapsed {elapsed/60:.1f}m"
            )

    if last_metrics or best_metrics:
        summary = {"best": best_metrics, "last": last_metrics}
        _write_results(args.results_dir, run_name, summary, args)
    writer.close()


if __name__ == "__main__":
    main()
