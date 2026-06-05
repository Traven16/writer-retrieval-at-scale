#!/usr/bin/env python3
import argparse
import os
import random
import multiprocessing as mp
import sys
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("OpenCV (cv2) is required for edges/SIFT masks. Install opencv-python.") from exc


def list_images(root: Path, exts: Iterable[str]) -> list[Path]:
    exts = {e.lower() for e in exts}
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
    return sorted(paths)


def _remove_largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    cleaned = mask.copy()
    cleaned[labels == largest] = 0
    return cleaned


def compute_edge_mask(
    img_gray: np.ndarray,
    low: int,
    high: int,
    dilate: int,
    remove_border: bool,
    edge_morph: bool,
) -> np.ndarray:
    edges = cv2.Canny(img_gray, low, high)
    if remove_border:
        edges = _remove_largest_component(edges)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
    if edge_morph:
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.erode(edges, kernel, iterations=2)
    return edges


def compute_sift_mask(
    img_gray: np.ndarray,
    max_keypoints: int,
    radius: int,
    dilate: int,
) -> np.ndarray:
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints = sift.detect(img_gray, None)
    mask = np.zeros_like(img_gray, dtype=np.uint8)
    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        cv2.circle(mask, (x, y), radius, 255, thickness=-1)
    if dilate > 0:
        kernel = np.ones((dilate, dilate), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask


def compute_sift_keypoints(img_gray: np.ndarray, max_keypoints: int) -> list[Tuple[int, int]]:
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints = sift.detect(img_gray, None)
    coords = []
    for kp in keypoints:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        coords.append((y, x))
    return coords


def sample_coords(mask: np.ndarray, n: int, rng: random.Random) -> list[Tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return [(rng.randrange(h), rng.randrange(w)) for _ in range(n)]
    idx = [rng.randrange(len(ys)) for _ in range(n)]
    return [(int(ys[i]), int(xs[i])) for i in idx]


def sample_coords_with_min_distance(
    mask: np.ndarray, min_dist: int, rng: random.Random
) -> list[Tuple[int, int]]:
    if min_dist <= 0:
        ys, xs = np.where(mask > 0)
        return [(int(ys[i]), int(xs[i])) for i in range(len(ys))]
    mask = mask.copy()
    coords = []
    h, w = mask.shape
    while True:
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            break
        idx = rng.randrange(len(ys))
        y = int(ys[idx])
        x = int(xs[idx])
        coords.append((y, x))
        y0 = max(0, y - min_dist)
        x0 = max(0, x - min_dist)
        y1 = min(h, y + min_dist + 1)
        x1 = min(w, x + min_dist + 1)
        mask[y0:y1, x0:x1] = 0
    return coords


def extract_patch(img: np.ndarray, center: Tuple[int, int], size: int) -> np.ndarray:
    h, w = img.shape[:2]
    half = size // 2
    y, x = center
    y0 = max(0, y - half)
    x0 = max(0, x - half)
    y1 = min(h, y0 + size)
    x1 = min(w, x0 + size)
    y0 = max(0, y1 - size)
    x0 = max(0, x1 - size)
    patch = img[y0:y1, x0:x1]
    if patch.shape[0] != size or patch.shape[1] != size:
        pad_y = size - patch.shape[0]
        pad_x = size - patch.shape[1]
        patch = np.pad(
            patch,
            ((0, pad_y), (0, pad_x), (0, 0)) if patch.ndim == 3 else ((0, pad_y), (0, pad_x)),
            mode="edge",
        )
    return patch


def _process_one(args: tuple) -> None:
    (
        img_path,
        input_dir,
        out_dir,
        num_patches,
        patch_size,
        mask_mode,
        mask_dir,
        seed,
        edge_low,
        edge_high,
        sift_max_keypoints,
        sift_radius,
        dilate,
        remove_border,
        edge_morph,
        patch_min_distance,
        output_ext,
        dry_run,
    ) = args
    rng = random.Random(seed)
    rel = img_path.relative_to(input_dir)
    stem = rel.stem
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    mask = None
    if mask_mode == "edges":
        mask = compute_edge_mask(gray, edge_low, edge_high, dilate, remove_border, edge_morph)
        if patch_min_distance > 0:
            coords = sample_coords_with_min_distance(mask, patch_min_distance, rng)
        else:
            coords = sample_coords(mask, num_patches, rng)
    elif mask_mode == "mask":
        if mask_dir is None:
            raise ValueError("mask_dir must be provided for mask mode")
        mask_path = mask_dir / rel
        if not mask_path.exists():
            # try without extension if needed
            candidates = list(mask_path.parent.glob(f"{mask_path.stem}.*"))
            if candidates:
                mask_path = candidates[0]
        mask_img = Image.open(mask_path).convert("L")
        if mask_img.size != img.size:
            mask_img = mask_img.resize(img.size, resample=Image.NEAREST)
        mask = (np.array(mask_img).astype(np.float32) / 255.0) >= 0.5
        mask = (mask.astype(np.uint8)) * 255
        if patch_min_distance > 0:
            coords = sample_coords_with_min_distance(mask, patch_min_distance, rng)
        else:
            coords = sample_coords(mask, num_patches, rng)
    else:
        if num_patches <= 0:
            coords = compute_sift_keypoints(gray, sift_max_keypoints)
        else:
            mask = compute_sift_mask(gray, sift_max_keypoints, sift_radius, dilate)
            coords = sample_coords(mask, num_patches, rng)
    out_subdir = out_dir / rel.parent / stem
    out_subdir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        if mask_mode == "sift" and num_patches <= 0:
            mask = np.zeros_like(gray, dtype=np.uint8)
            for (y, x) in coords:
                cv2.circle(mask, (int(x), int(y)), max(1, int(sift_radius)), 255, thickness=-1)
        if mask is None:
            return
        mask_img = Image.fromarray(mask)
        mask_img.save(out_subdir / f"{stem}_mask{output_ext}")
        return
    if not coords:
        return
    for idx, (y, x) in enumerate(coords):
        patch = extract_patch(img_np, (y, x), patch_size)
        patch_img = Image.fromarray(patch)
        patch_img.save(out_subdir / f"{stem}_{idx:04d}{output_ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract masked patches per image.")
    parser.add_argument("--input-dir", required=True, type=str)
    parser.add_argument("--out-dir", required=True, type=str)
    parser.add_argument("--num-patches", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--mask-mode", choices=["edges", "sift", "mask"], default="edges")
    parser.add_argument("--mask-dir", type=str, default="", help="Mask directory for mask mode.")
    parser.add_argument("--image-exts", type=str, nargs="+", default=[".jpg", ".png", ".jpeg", ".tif", ".tiff"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-low", type=int, default=50)
    parser.add_argument("--edge-high", type=int, default=150)
    parser.add_argument("--remove-border", action="store_true")
    parser.add_argument("--edge-morph", action="store_true")
    parser.add_argument(
        "--patch-min-distance",
        type=int,
        default=0,
        help="Minimum x/y distance between sampled patches (edges mode only).",
    )
    parser.add_argument("--sift-max-keypoints", type=int, default=500)
    parser.add_argument("--sift-radius", type=int, default=5)
    parser.add_argument("--dilate", type=int, default=3)
    parser.add_argument("--output-ext", type=str, default=".png")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Save masks instead of patches.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    images = list_images(input_dir, args.image_exts)
    if not images:
        raise SystemExit(f"No images found under {input_dir}")

    worker_count = args.workers if args.workers > 0 else min(8, os.cpu_count() or 1)
    tasks = []
    for i, img_path in enumerate(images):
        seed = args.seed + i
        tasks.append(
            (
                img_path,
                input_dir,
                out_dir,
                args.num_patches,
                args.patch_size,
                args.mask_mode,
                mask_dir,
                seed,
                args.edge_low,
                args.edge_high,
                args.sift_max_keypoints,
                args.sift_radius,
                args.dilate,
                args.remove_border,
                args.edge_morph,
                args.patch_min_distance,
                args.output_ext,
                args.dry_run,
            )
        )

    if worker_count <= 1:
        for task in tqdm(tasks, desc="prepare_data", unit="img"):
            _process_one(task)
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=worker_count) as pool:
            for _ in tqdm(
                pool.imap_unordered(_process_one, tasks, chunksize=4),
                total=len(tasks),
                desc="prepare_data",
                unit="img",
            ):
                pass


if __name__ == "__main__":
    main()
