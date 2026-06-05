#!/usr/bin/env python3
import argparse
import json
import os
import random
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as TF

from data.dataset import _index_folder, _resolve_mask_path


def _resolve_mask_with_fallback(mask_root: str, rel_path: str, mask_ext: Sequence[str]) -> str:
    mask_path = _resolve_mask_path(mask_root, rel_path, mask_ext)
    if os.path.exists(mask_path):
        return mask_path
    rel_dir = os.path.dirname(rel_path)
    base_name = os.path.basename(rel_path)
    parent_dir = os.path.dirname(rel_dir)
    if parent_dir and parent_dir != rel_dir:
        rel_alt = os.path.join(parent_dir, base_name)
        mask_alt = _resolve_mask_path(mask_root, rel_alt, mask_ext)
        if os.path.exists(mask_alt):
            return mask_alt
    return mask_path


def _normalize_exts(exts: Sequence[str] | str | None, default_ext: str) -> List[str]:
    if exts is None:
        values = [default_ext]
    elif isinstance(exts, str):
        values = [exts]
    else:
        values = list(exts)
    out = []
    for value in values:
        if not value:
            continue
        ext = value.lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        out.append(ext)
    return out or [default_ext]


def _load_data_cfg(config_path: str) -> Dict[str, object]:
    with open(config_path, "r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if "train_data" in loaded and isinstance(loaded["train_data"], dict):
        cfg = loaded["train_data"]
        return {
            "train_root": cfg.get("root", ""),
            "train_mask_root": cfg.get("mask_root", ""),
            "image_ext": _normalize_exts(cfg.get("image_ext"), ".jpg"),
            "mask_ext": _normalize_exts(cfg.get("mask_ext"), ".png"),
            "class_split_char": cfg.get("class_split_char", "_"),
            "class_id_index": int(cfg.get("class_id_index", 0)),
            "class_id_use_folder": bool(cfg.get("class_id_use_folder", False)),
            "augment_patches": bool(cfg.get("augment_patches", False)),
        }
    return {
        "train_root": loaded.get("train_root", ""),
        "train_mask_root": loaded.get("train_mask_root", ""),
        "image_ext": _normalize_exts(loaded.get("image_ext"), ".jpg"),
        "mask_ext": _normalize_exts(loaded.get("mask_ext"), ".png"),
        "class_split_char": loaded.get("class_split_char", "_"),
        "class_id_index": int(loaded.get("class_id_index", 0)),
        "class_id_use_folder": bool(loaded.get("class_id_use_folder", False)),
        "augment_patches": bool(loaded.get("augment_patches", False)),
    }


def _sample_coords(mask_np: np.ndarray, num_patches: int) -> np.ndarray:
    height, width = mask_np.shape
    coords = np.argwhere(mask_np >= 125)
    if coords.size == 0:
        ys = np.random.randint(0, height, size=num_patches)
        xs = np.random.randint(0, width, size=num_patches)
        return np.stack([ys, xs], axis=1)
    choice = np.random.choice(coords.shape[0], size=num_patches, replace=True)
    return coords[choice]


def _extract_patch(np_img: np.ndarray, y: int, x: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    extra = patch_size % 2
    padded = np.pad(
        np_img,
        ((half, half + extra), (half, half + extra), (0, 0)),
        mode="edge",
    )
    yy = y + half
    xx = x + half
    patch = padded[yy - half : yy + half + extra, xx - half : xx + half + extra, :]
    return patch


def _extract_mask_patch(mask_np: np.ndarray, y: int, x: int, patch_size: int) -> np.ndarray:
    half = patch_size // 2
    extra = patch_size % 2
    padded = np.pad(
        mask_np,
        ((half, half + extra), (half, half + extra)),
        mode="edge",
    )
    yy = y + half
    xx = x + half
    patch = padded[yy - half : yy + half + extra, xx - half : xx + half + extra]
    return patch


def _sample_balanced_batch(
    samples: List[Tuple[str, str, int]],
    writer_count: int,
    samples_per_writer: int,
) -> List[Tuple[str, str, int]]:
    label_to_samples: Dict[int, List[Tuple[str, str, int]]] = {}
    for sample in samples:
        label = sample[2]
        label_to_samples.setdefault(label, []).append(sample)

    eligible = [label for label, items in label_to_samples.items() if len(items) >= samples_per_writer]
    if len(eligible) < writer_count:
        raise ValueError(
            f"Not enough writers with >= {samples_per_writer} samples. "
            f"Need {writer_count}, have {len(eligible)}."
        )

    selected_labels = random.sample(eligible, k=writer_count)
    chosen: List[Tuple[str, str, int]] = []
    for label in selected_labels:
        chosen.extend(random.sample(label_to_samples[label], k=samples_per_writer))
    return chosen


def _build_patch_augment() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25
            ),
            transforms.RandomGrayscale(p=0.05),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a pseudo training batch with sampled patches.")
    parser.add_argument("--config", type=str, default="", help="Optional YAML config path.")
    parser.add_argument("--train-root", type=str, default="")
    parser.add_argument("--train-mask-root", type=str, default="")
    parser.add_argument("--image-ext", nargs="+", default=None)
    parser.add_argument("--mask-ext", nargs="+", default=None)
    parser.add_argument("--class-split-char", type=str, default=None)
    parser.add_argument("--class-id-index", type=int, default=None)
    parser.add_argument("--class-id-use-folder", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--samples-per-writer", type=int, default=2)
    parser.add_argument("--num-patches", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="viz/pseudo_batch")
    parser.add_argument("--augment-patches", action="store_true")
    args = parser.parse_args()

    cfg: Dict[str, object] = {}
    if args.config:
        cfg = _load_data_cfg(args.config)

    train_root = args.train_root or str(cfg.get("train_root", ""))
    train_mask_root = args.train_mask_root or str(cfg.get("train_mask_root", ""))
    if not train_root or not train_mask_root:
        raise ValueError("Need --train-root and --train-mask-root (or provide --config with these fields).")

    image_ext = _normalize_exts(args.image_ext, ".jpg")
    if args.image_ext is None and cfg.get("image_ext"):
        image_ext = _normalize_exts(cfg.get("image_ext"), ".jpg")
    mask_ext = _normalize_exts(args.mask_ext, ".png")
    if args.mask_ext is None and cfg.get("mask_ext"):
        mask_ext = _normalize_exts(cfg.get("mask_ext"), ".png")

    class_split_char = args.class_split_char if args.class_split_char is not None else str(
        cfg.get("class_split_char", "_")
    )
    class_id_index = args.class_id_index if args.class_id_index is not None else int(
        cfg.get("class_id_index", 0)
    )
    class_id_use_folder = bool(args.class_id_use_folder or cfg.get("class_id_use_folder", False))
    augment_patches = bool(args.augment_patches or cfg.get("augment_patches", False))

    if args.batch_size % args.samples_per_writer != 0:
        raise ValueError("--batch-size must be divisible by --samples-per-writer.")

    random.seed(args.seed)
    np.random.seed(args.seed)

    samples, class_to_idx = _index_folder(
        train_root,
        class_split_char,
        tuple(image_ext),
        class_id_index,
        class_id_use_folder,
    )
    valid_samples: List[Tuple[str, str, int]] = []
    for image_path, rel_path, label in samples:
        mask_path = _resolve_mask_with_fallback(train_mask_root, rel_path, mask_ext)
        if os.path.exists(mask_path):
            valid_samples.append((image_path, rel_path, label))
    if not valid_samples:
        raise ValueError("No samples with existing masks were found.")

    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    writers_per_batch = args.batch_size // args.samples_per_writer
    selected = _sample_balanced_batch(valid_samples, writers_per_batch, args.samples_per_writer)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = f"{args.out_dir}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    metadata = {
        "train_root": train_root,
        "train_mask_root": train_mask_root,
        "image_ext": image_ext,
        "mask_ext": mask_ext,
        "batch_size": args.batch_size,
        "samples_per_writer": args.samples_per_writer,
        "num_patches": args.num_patches,
        "patch_size": args.patch_size,
        "seed": args.seed,
        "augment_patches": augment_patches,
        "samples": [],
    }
    patch_aug = _build_patch_augment() if augment_patches else None

    for index, (image_path, rel_path, label) in enumerate(selected):
        mask_path = _resolve_mask_with_fallback(train_mask_root, rel_path, mask_ext)
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, resample=Image.NEAREST)
        image_np = np.array(image)
        mask_np = np.array(mask)

        coords = _sample_coords(mask_np, args.num_patches)
        draw_img = image.copy()
        draw = ImageDraw.Draw(draw_img)

        writer_id = idx_to_class.get(label, str(label)).replace("/", "_")
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        sample_name = f"{index:02d}_{writer_id}_{image_stem}"
        sample_dir = os.path.join(out_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "patches"), exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "mask_patches"), exist_ok=True)

        half = args.patch_size // 2
        extra = args.patch_size % 2
        for patch_idx, (y, x) in enumerate(coords):
            y0 = int(y - half)
            x0 = int(x - half)
            y1 = int(y + half + extra - 1)
            x1 = int(x + half + extra - 1)
            draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=2)

            patch_np = _extract_patch(image_np, int(y), int(x), args.patch_size)
            if patch_aug is not None:
                patch_tensor = TF.to_tensor(patch_np)
                patch_tensor = patch_aug(patch_tensor)
                patch_img = TF.to_pil_image(patch_tensor)
            else:
                patch_img = Image.fromarray(patch_np)
            patch_img.save(os.path.join(sample_dir, "patches", f"patch_{patch_idx:02d}.png"))

            mask_patch_np = _extract_mask_patch(mask_np, int(y), int(x), args.patch_size)
            mask_patch_img = Image.fromarray(mask_patch_np)
            mask_patch_img.save(os.path.join(sample_dir, "mask_patches", f"mask_patch_{patch_idx:02d}.png"))

        draw_img.save(os.path.join(sample_dir, "image_with_patches.png"))
        mask.save(os.path.join(sample_dir, "mask.png"))

        metadata["samples"].append(
            {
                "sample_name": sample_name,
                "image_path": image_path,
                "mask_path": mask_path,
                "writer_id": idx_to_class.get(label, str(label)),
                "label": int(label),
            }
        )

    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Saved pseudo batch visualization to {out_dir}")


if __name__ == "__main__":
    main()
