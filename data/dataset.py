import csv
import os
import time
import warnings
import multiprocessing as mp
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Optional, Sequence

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _otsu_threshold(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    weight_bg = 0.0
    sum_bg = 0.0
    sum_total = (hist * np.arange(256)).sum()
    max_var = -1.0
    threshold = 0
    for i in range(256):
        weight_bg += hist[i]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break
        sum_bg += i * hist[i]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg
        between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if between > max_var:
            max_var = between
            threshold = i
    return threshold / 255.0


def _apply_otsu_binarize(patch: torch.Tensor) -> torch.Tensor:
    if patch.dim() != 3:
        raise ValueError("Expected patch tensor with shape (C, H, W)")
    gray = patch.mean(dim=0).cpu().numpy()
    thresh = _otsu_threshold(gray)
    binary = (gray >= thresh).astype(np.float32)
    return (
        torch.from_numpy(binary)
        .to(patch.device)
        .unsqueeze(0)
        .repeat(patch.size(0), 1, 1)
    )


def _apply_patch_transform(
    transform: Optional[Callable], patches_tensor: torch.Tensor
) -> torch.Tensor:
    if transform is None:
        return patches_tensor
    # Fast path: many torchvision tensor transforms support batched inputs.
    try:
        transformed = transform(patches_tensor)
        if isinstance(transformed, torch.Tensor) and transformed.shape == patches_tensor.shape:
            return transformed
    except Exception:
        pass
    return torch.stack([transform(patch) for patch in patches_tensor], dim=0)


def _crop_hwc_patch_replicate(
    image_hwc: torch.Tensor,
    y: int,
    x: int,
    patch_size: int,
) -> torch.Tensor:
    """Crop an HWC uint8 image to CHW float, replicating border pixels."""
    height, width = image_hwc.shape[:2]
    half = patch_size // 2
    extra = patch_size % 2
    ys = torch.arange(
        y - half,
        y + half + extra,
        device=image_hwc.device,
        dtype=torch.long,
    ).clamp_(0, height - 1)
    xs = torch.arange(
        x - half,
        x + half + extra,
        device=image_hwc.device,
        dtype=torch.long,
    ).clamp_(0, width - 1)
    patch = image_hwc.index_select(0, ys).index_select(1, xs)
    return patch.permute(2, 0, 1).float() / 255.0


def _valid_patch_center_bounds(
    height: int,
    width: int,
    patch_size: int,
) -> Tuple[int, int, int, int]:
    half = patch_size // 2
    extra = patch_size % 2
    y_min = min(half, max(0, height - 1))
    x_min = min(half, max(0, width - 1))
    y_max = max(y_min, height - half - extra)
    x_max = max(x_min, width - half - extra)
    return y_min, y_max, x_min, x_max


def _filter_numpy_patch_centers(
    coords: np.ndarray,
    height: int,
    width: int,
    patch_size: int,
) -> np.ndarray:
    if coords.size == 0:
        return coords
    y_min, y_max, x_min, x_max = _valid_patch_center_bounds(height, width, patch_size)
    keep = (
        (coords[:, 0] >= y_min)
        & (coords[:, 0] <= y_max)
        & (coords[:, 1] >= x_min)
        & (coords[:, 1] <= x_max)
    )
    return coords[keep]


def _filter_torch_patch_centers(
    coords: torch.Tensor,
    height: int,
    width: int,
    patch_size: int,
) -> torch.Tensor:
    if coords.numel() == 0:
        return coords
    y_min, y_max, x_min, x_max = _valid_patch_center_bounds(height, width, patch_size)
    keep = (
        (coords[:, 0] >= y_min)
        & (coords[:, 0] <= y_max)
        & (coords[:, 1] >= x_min)
        & (coords[:, 1] <= x_max)
    )
    return coords[keep]


def normalize_patch_tensor(
    patches: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Normalize patch tensors for 4D (N,C,H,W) or 5D (B,N,C,H,W) inputs."""
    mean_t = mean.to(device=patches.device, dtype=patches.dtype)
    std_t = std.to(device=patches.device, dtype=patches.dtype)
    while mean_t.dim() < patches.dim():
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)
    return (patches - mean_t) / std_t


def _pad_or_center_crop_to_size(
    image_chw: torch.Tensor,
    target_h: int,
    target_w: int,
) -> torch.Tensor:
    """Match target size without resizing: center-crop if too large, pad if too small."""
    if image_chw.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(image_chw.shape)}")
    _, h, w = image_chw.shape
    # Center-crop height/width when larger than target.
    if h > target_h:
        top = (h - target_h) // 2
        image_chw = image_chw[:, top : top + target_h, :]
        h = target_h
    if w > target_w:
        left = (w - target_w) // 2
        image_chw = image_chw[:, :, left : left + target_w]
        w = target_w
    # Symmetric pad when smaller than target.
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        image_chw = F.pad(
            image_chw,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )
    return image_chw


def _resize_longest_side(
    image_chw: torch.Tensor,
    max_side: int,
) -> torch.Tensor:
    if image_chw.dim() != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(image_chw.shape)}")
    if max_side <= 0:
        return image_chw
    _, h, w = image_chw.shape
    longest = max(h, w)
    if longest <= max_side:
        return image_chw
    scale = float(max_side) / float(longest)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    return F.interpolate(
        image_chw.unsqueeze(0),
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _sample_rrc_params(
    height: int,
    width: int,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
) -> Tuple[int, int, int, int]:
    area = float(height * width)
    log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
    for _ in range(10):
        target_area = area * float(torch.empty(1).uniform_(scale[0], scale[1]).item())
        aspect_ratio = float(np.exp(float(torch.empty(1).uniform_(log_ratio[0], log_ratio[1]).item())))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        if 0 < h <= height and 0 < w <= width:
            i = int(torch.randint(0, height - h + 1, (1,)).item())
            j = int(torch.randint(0, width - w + 1, (1,)).item())
            return i, j, h, w
    return 0, 0, height, width


def _apply_sequence_patch_augment(
    patches_tensor: torch.Tensor,
    split_per_sample: int,
    crop_size: int,
    strong: bool,
) -> torch.Tensor:
    if patches_tensor.dim() != 4:
        return patches_tensor
    num_patches, _, h, w = patches_tensor.shape
    split = max(1, int(split_per_sample))
    if split <= 1 or num_patches % split != 0:
        return patches_tensor
    chunk = num_patches // split
    out = patches_tensor.clone()
    target = int(crop_size) if crop_size > 0 else h
    target = h if target != h else target
    for s in range(split):
        start = s * chunk
        end = start + chunk
        seq = out[start:end]
        if strong:
            i, j, hh, ww = _sample_rrc_params(h, w, scale=(0.55, 1.0), ratio=(0.6, 1.6))
        else:
            i, j, hh, ww = _sample_rrc_params(h, w, scale=(0.8, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0))
        seq = TF.resized_crop(
            seq, i, j, hh, ww, size=[target, target], interpolation=TF.InterpolationMode.BILINEAR, antialias=True
        )
        flip_p = 0.2 if strong else 0.1
        if torch.rand(1).item() < flip_p:
            seq = torch.flip(seq, dims=[-1])
        if torch.rand(1).item() < flip_p:
            seq = torch.flip(seq, dims=[-2])
        if strong:
            if torch.rand(1).item() < 0.9:
                seq = TF.adjust_brightness(seq, 1.0 + float(torch.empty(1).uniform_(-0.5, 0.5).item()))
                seq = TF.adjust_contrast(seq, 1.0 + float(torch.empty(1).uniform_(-0.5, 0.5).item()))
                seq = TF.adjust_saturation(seq, 1.0 + float(torch.empty(1).uniform_(-0.5, 0.5).item()))
                seq = TF.adjust_hue(seq, float(torch.empty(1).uniform_(-0.15, 0.15).item()))
            if torch.rand(1).item() < 0.3:
                sigma = float(torch.empty(1).uniform_(0.1, 1.8).item())
                seq = TF.gaussian_blur(seq, kernel_size=[3, 3], sigma=[sigma, sigma])
        else:
            seq = TF.adjust_brightness(seq, 1.0 + float(torch.empty(1).uniform_(-0.3, 0.3).item()))
            seq = TF.adjust_contrast(seq, 1.0 + float(torch.empty(1).uniform_(-0.3, 0.3).item()))
            seq = TF.adjust_saturation(seq, 1.0 + float(torch.empty(1).uniform_(-0.3, 0.3).item()))
            seq = TF.adjust_hue(seq, float(torch.empty(1).uniform_(-0.1, 0.1).item()))
        if torch.rand(1).item() < (0.2 if strong else 0.05):
            gray = seq.mean(dim=1, keepdim=True)
            seq = gray.repeat(1, seq.size(1), 1, 1)
        out[start:end] = seq.clamp(0.0, 1.0)
    return out


def _resolve_path(root: str, path: str) -> str:
    if os.path.isabs(path) or not root:
        return path
    return os.path.join(root, path)


def _read_csv(csv_path: str) -> List[Tuple[str, str, int]]:
    samples: List[Tuple[str, str, int]] = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file must have a header with image, mask, and label columns")
        required = {"image", "mask", "label"}
        if not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV header must include {required}, got {reader.fieldnames}")
        for row in reader:
            samples.append((row["image"], row["mask"], int(row["label"])))
    return samples


def _is_image_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _parse_class_id(filename: str, split_char: str, index: int) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    if not split_char:
        raise ValueError("split_char must be non-empty")
    parts = base.split(split_char)
    if index < 0 or index >= len(parts):
        raise ValueError(
            f"class_id_index {index} out of range for filename: {filename}"
        )
    return parts[index]


def _index_folder(
    root: str,
    split_char: str,
    image_exts: Tuple[str, ...],
    class_id_index: int,
    class_id_use_folder: bool,
    debug: bool = False,
) -> Tuple[List[Tuple[str, str, int]], Dict[str, int]]:
    samples_raw: List[Tuple[str, str, str]] = []
    class_ids = set()
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fname in filenames:
            if not _is_image_file(fname):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in image_exts:
                continue
            image_path = os.path.join(dirpath, fname)
            class_id = _parse_class_id(fname, split_char, class_id_index)
            if class_id_use_folder:
                rel_dir = os.path.relpath(dirpath, root)
                if rel_dir != ".":
                    class_id = f"{rel_dir}/{class_id}"
            class_ids.add(class_id)
            rel_path = os.path.relpath(image_path, root)
            samples_raw.append((image_path, rel_path, class_id))

    if debug:
        print(
            f"[dataset] index_folder root={root} exts={image_exts} "
            f"classes={len(class_ids)} samples={len(samples_raw)}"
        )
        for sample in samples_raw[:3]:
            print(f"[dataset] sample: {sample[0]}")

    class_to_idx = {name: idx for idx, name in enumerate(sorted(class_ids))}
    samples: List[Tuple[str, str, int]] = [
        (image_path, rel_path, class_to_idx[class_id])
        for image_path, rel_path, class_id in samples_raw
    ]
    return samples, class_to_idx


def _replace_extension(path: str, new_ext: str) -> str:
    base, _ = os.path.splitext(path)
    return base + new_ext


def _resolve_mask_path(
    root: str, image_rel_path: str, mask_exts: Sequence[str]
) -> str:
    base, _ = os.path.splitext(image_rel_path)
    rel_dir = os.path.dirname(base)
    stem = os.path.basename(base)
    dir_parts = [p for p in rel_dir.split(os.sep) if p and p != "."]

    # Try exact relative path first, then a few fallbacks for datasets where
    # images have an extra nested writer folder that masks do not have.
    candidate_dirs: List[str] = [rel_dir]
    for drop in range(1, len(dir_parts) + 1):
        reduced = os.path.join(*dir_parts[:-drop]) if dir_parts[:-drop] else ""
        candidate_dirs.append(reduced)

    seen: set[str] = set()
    ordered_candidate_dirs: List[str] = []
    for rel in candidate_dirs:
        if rel not in seen:
            ordered_candidate_dirs.append(rel)
            seen.add(rel)

    for ext in mask_exts:
        for rel in ordered_candidate_dirs:
            if rel:
                candidate = os.path.join(root, rel, stem + ext)
                nested = os.path.join(root, rel, stem, stem + ext)
            else:
                candidate = os.path.join(root, stem + ext)
                nested = os.path.join(root, stem, stem + ext)
            if os.path.exists(candidate):
                return candidate
            if os.path.exists(nested):
                return nested

    fallback_ext = mask_exts[0] if mask_exts else ".png"
    if rel_dir:
        return os.path.join(root, rel_dir, stem + fallback_ext)
    return os.path.join(root, stem + fallback_ext)


def _load_binarized_sample(args: Tuple[str, int]) -> Tuple[np.ndarray, np.ndarray, int]:
    image_path, label = args
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    if image_np.ndim != 3:
        raise ValueError("Unexpected image format")
    mask_np = image_np.mean(axis=2).astype(np.uint8)
    return image_np, mask_np, int(label)


class PatchDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str],
        image_root: str,
        mask_root: str,
        num_patches: int,
        patch_size: int = 32,
        transform: Optional[Callable] = None,
        page_transform: Optional[Callable] = None,
        augment_patches: bool = False,
        augment_otsu: bool = False,
        val_gray: bool = False,
        precomputed_patches_root: str = "",
        class_split_char: str = "_",
        class_id_index: int = 0,
        class_id_use_folder: bool = False,
        debug: bool = False,
        image_ext: Optional[Sequence[str]] = None,
        mask_ext: Optional[Sequence[str]] = None,
        patch_ext: Optional[Sequence[str]] = None,
        cache_packed_patches_in_memory: bool = False,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        debug_first_batch: bool = False,
        full_image_input: bool = False,
        full_image_height: int = 0,
        full_image_width: int = 0,
        full_image_pad_to_size: bool = False,
        full_image_resize_longest_side_first: bool = False,
        sequence_patch_augment_in_dataset: bool = False,
        sequence_patch_split_per_sample: int = 1,
        sequence_patch_crop_size: int = 32,
        strong_patch_augment: bool = False,
    ) -> None:
        self.class_to_idx: Dict[str, int] = {}
        self.image_ext = [ext.lower() for ext in (image_ext or [".jpg"])]
        self.mask_ext = [ext.lower() for ext in (mask_ext or [".png"])]
        self.patch_ext = [ext.lower() for ext in (patch_ext or [".png"])]
        if csv_path:
            self.samples = _read_csv(csv_path)
        else:
            if any(not ext.startswith(".") for ext in self.image_ext):
                raise ValueError("image_ext entries must start with '.'")
            if any(not ext.startswith(".") for ext in self.mask_ext):
                raise ValueError("mask_ext entries must start with '.'")
            self.samples, self.class_to_idx = _index_folder(
                image_root,
                class_split_char,
                tuple(self.image_ext),
                class_id_index,
                class_id_use_folder,
                debug=debug,
            )
        self.image_root = image_root
        self.mask_root = mask_root
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.transform = transform
        self.page_transform = page_transform
        self.augment_patches = augment_patches
        self.augment_otsu = augment_otsu
        self.val_gray = val_gray
        self.precomputed_patches_root = precomputed_patches_root
        self.class_id_index = class_id_index
        self.class_id_use_folder = class_id_use_folder
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.full_image_input = bool(full_image_input)
        self.full_image_height = int(full_image_height)
        self.full_image_width = int(full_image_width)
        self.full_image_pad_to_size = bool(full_image_pad_to_size)
        self.full_image_resize_longest_side_first = bool(full_image_resize_longest_side_first)
        self.sequence_patch_augment_in_dataset = bool(sequence_patch_augment_in_dataset)
        self.sequence_patch_split_per_sample = max(1, int(sequence_patch_split_per_sample))
        self.sequence_patch_crop_size = int(sequence_patch_crop_size)
        self.strong_patch_augment = bool(strong_patch_augment)
        self.debug_first_batch = debug_first_batch
        self._debug_first_count = 0
        self._debug_first_limit = 8
        self.cache_packed_patches_in_memory = (
            bool(cache_packed_patches_in_memory) and bool(self.precomputed_patches_root)
        )
        self._sample_packed_files: Optional[List[str]] = None
        self._packed_patch_full_cache: Optional[Dict[str, torch.Tensor]] = None
        self._packed_patch_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._packed_patch_cache_size = 64
        if debug and csv_path:
            print(
                f"[dataset] csv={csv_path} samples={len(self.samples)} "
                f"classes={len(self.class_to_idx) if self.class_to_idx else 'csv'}"
            )
        if self.precomputed_patches_root:
            self._sample_packed_files = self._build_sample_packed_files()
            if self.cache_packed_patches_in_memory:
                self._preload_all_packed_patches()
        if self.full_image_input and self.precomputed_patches_root:
            raise ValueError("full_image_input is incompatible with precomputed packed patches.")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve_packed_patch_file(self, image_path: str, rel_path: str) -> str:
        if not self.precomputed_patches_root:
            return ""
        rel = rel_path
        if os.path.isabs(rel):
            try:
                rel = os.path.relpath(rel, self.image_root)
            except ValueError:
                rel = os.path.basename(rel)
        rel_dir = os.path.dirname(rel)
        stem = os.path.splitext(os.path.basename(rel))[0]
        primary = os.path.join(self.precomputed_patches_root, rel_dir, stem + ".pt")
        if not rel_dir or os.path.isfile(primary):
            return primary
        # Fallback: drop the last path component (e.g. writer folder) if present.
        rel_parent = os.path.dirname(rel_dir)
        if rel_parent and rel_parent != rel_dir:
            alt = os.path.join(self.precomputed_patches_root, rel_parent, stem + ".pt")
            if os.path.isfile(alt):
                return alt
        return primary

    def _build_sample_packed_files(self) -> List[str]:
        packed_files: List[str] = []
        for image_path, rel_path, _ in self.samples:
            source_rel_path = rel_path if self.class_to_idx else image_path
            packed_files.append(self._resolve_packed_patch_file(image_path, source_rel_path))
        return packed_files

    def _load_packed_patch_tensor(self, packed_file: str) -> torch.Tensor:
        if not os.path.isfile(packed_file):
            raise FileNotFoundError(f"Packed patch file not found: {packed_file}")

        payload = torch.load(packed_file, map_location="cpu")
        if isinstance(payload, dict):
            if "patches" not in payload:
                raise KeyError(f"Missing 'patches' key in packed file: {packed_file}")
            patches = payload["patches"]
        else:
            patches = payload

        if not isinstance(patches, torch.Tensor):
            raise TypeError(f"Packed patches must be a tensor: {packed_file}")
        if patches.dim() != 4:
            raise ValueError(
                f"Packed patches must be 4D tensor (N,C,H,W) or (N,H,W,C): {packed_file}"
            )
        if patches.shape[1] == 3:
            patches = patches.contiguous()
        elif patches.shape[-1] == 3:
            patches = patches.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError(f"Packed patches must have 3 channels: {packed_file}")
        if patches.size(0) <= 0:
            raise ValueError(f"Packed patch file has no patches: {packed_file}")
        return patches

    def _preload_all_packed_patches(self) -> None:
        if not self._sample_packed_files:
            return
        unique_files = list(dict.fromkeys(self._sample_packed_files))
        total = len(unique_files)
        self._packed_patch_full_cache = {}
        total_bytes = 0
        start = time.perf_counter()
        print(f"[dataset] preloading packed patches into memory cache: {total} files")
        for idx, packed_file in enumerate(unique_files, start=1):
            # Keep tensors in-process. With forked workers this is shared via copy-on-write
            # and avoids exhausting file descriptors from per-tensor shared-memory handles.
            patches = self._load_packed_patch_tensor(packed_file)
            self._packed_patch_full_cache[packed_file] = patches
            total_bytes += patches.numel() * patches.element_size()
            if idx == 1 or idx == total or idx % 5000 == 0:
                print(
                    f"[dataset] packed preload {idx}/{total} "
                    f"({total_bytes / (1024 ** 3):.2f} GiB)"
                )
        elapsed = time.perf_counter() - start
        print(
            f"[dataset] packed preload done in {elapsed:.1f}s "
            f"({total_bytes / (1024 ** 3):.2f} GiB)"
        )

    def _get_packed_patches(self, packed_file: str) -> torch.Tensor:
        if self._packed_patch_full_cache is not None:
            cached = self._packed_patch_full_cache.get(packed_file)
            if cached is None:
                raise FileNotFoundError(f"Packed patch file not found in preload cache: {packed_file}")
            return cached

        cached = self._packed_patch_cache.get(packed_file)
        if cached is not None:
            self._packed_patch_cache.move_to_end(packed_file)
            return cached

        patches = self._load_packed_patch_tensor(packed_file)

        self._packed_patch_cache[packed_file] = patches
        self._packed_patch_cache.move_to_end(packed_file)
        if len(self._packed_patch_cache) > self._packed_patch_cache_size:
            self._packed_patch_cache.popitem(last=False)
        return patches

    def _load_precomputed_patches(self, packed_file: str) -> torch.Tensor:
        patches = self._get_packed_patches(packed_file)
        count = int(patches.size(0))
        if self.num_patches > 0:
            if count >= self.num_patches:
                indices = np.random.choice(count, size=self.num_patches, replace=False)
            else:
                indices = np.random.choice(count, size=self.num_patches, replace=True)
            selected = patches.index_select(0, torch.from_numpy(indices).long())
        else:
            selected = patches

        if selected.dtype == torch.uint8:
            patches_tensor = selected.float() / 255.0
        else:
            patches_tensor = selected.float()

        if patches_tensor.size(-2) != self.patch_size or patches_tensor.size(-1) != self.patch_size:
            patches_tensor = F.interpolate(
                patches_tensor, size=(self.patch_size, self.patch_size), mode="bilinear", align_corners=False
            )
        if self.val_gray:
            gray = patches_tensor.mean(dim=1, keepdim=True)
            patches_tensor = gray.repeat(1, patches_tensor.size(1), 1, 1)
        if self.augment_patches:
            if self.sequence_patch_augment_in_dataset:
                patches_tensor = _apply_sequence_patch_augment(
                    patches_tensor,
                    split_per_sample=self.sequence_patch_split_per_sample,
                    crop_size=self.sequence_patch_crop_size,
                    strong=self.strong_patch_augment,
                )
            elif self.transform is not None:
                patches_tensor = _apply_patch_transform(self.transform, patches_tensor)
        if self.augment_otsu and self.augment_patches:
            patches_tensor = torch.stack(
                [_apply_otsu_binarize(patch) for patch in patches_tensor], dim=0
            )
        return patches_tensor

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        do_debug = self.debug_first_batch and self._debug_first_count < self._debug_first_limit
        if do_debug:
            self._debug_first_count += 1
            debug_start = time.perf_counter()
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker is not None else "main"
        image_path, mask_path, label = self.samples[index]
        if self.class_to_idx:
            image_path = image_path
            mask_path = _resolve_mask_path(self.mask_root, mask_path, self.mask_ext)
        else:
            image_path = _resolve_path(self.image_root, image_path)
            mask_path = _resolve_path(self.mask_root, mask_path)

        if self.precomputed_patches_root:
            if self._sample_packed_files is not None:
                packed_file = self._sample_packed_files[index]
            else:
                if self.class_to_idx:
                    rel_path = self.samples[index][1]
                else:
                    rel_path = image_path
                packed_file = self._resolve_packed_patch_file(image_path, rel_path)
            if do_debug:
                print(
                    f"[debug-first-batch] worker={worker_id} idx={index} "
                    f"precomputed packed_file={packed_file}"
                )
            patches_tensor = self._load_precomputed_patches(packed_file)
            if do_debug:
                elapsed = time.perf_counter() - debug_start
                print(f"[debug-first-batch] worker={worker_id} idx={index} done {elapsed:.3f}s")
            return patches_tensor, label

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if do_debug:
            print(
                f"[debug-first-batch] worker={worker_id} idx={index} "
                f"image={image_path} mask={mask_path}"
            )

        if mask.size != image.size:
            mask = mask.resize(image.size, resample=Image.NEAREST)

        if self.full_image_input:
            image_draw = image
            mask_draw = mask
            if self.page_transform is not None:
                image_draw, mask_draw = self.page_transform(image_draw, mask_draw)
            if self.transform is not None and not self.augment_patches:
                image_draw, mask_draw = self.transform(image_draw, mask_draw)
            image_tensor = TF.to_tensor(image_draw).float()
            target_h = self.full_image_height if self.full_image_height > 0 else self.patch_size
            target_w = self.full_image_width if self.full_image_width > 0 else self.patch_size
            if image_tensor.size(-2) != target_h or image_tensor.size(-1) != target_w:
                if self.full_image_pad_to_size:
                    if self.full_image_resize_longest_side_first:
                        image_tensor = _resize_longest_side(image_tensor, max(target_h, target_w))
                    image_tensor = _pad_or_center_crop_to_size(image_tensor, target_h, target_w)
                else:
                    image_tensor = F.interpolate(
                        image_tensor.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
            if self.val_gray:
                gray = image_tensor.mean(dim=0, keepdim=True)
                image_tensor = gray.repeat(image_tensor.size(0), 1, 1)
            if self.transform is not None and self.augment_patches:
                image_tensor = self.transform(image_tensor)
            if self.augment_otsu and self.augment_patches:
                image_tensor = _apply_otsu_binarize(image_tensor)
            if do_debug:
                elapsed = time.perf_counter() - debug_start
                print(f"[debug-first-batch] worker={worker_id} idx={index} done {elapsed:.3f}s")
            return image_tensor.unsqueeze(0), label

        half = self.patch_size // 2
        extra = self.patch_size % 2

        image_draw = image
        mask_draw = mask
        if self.page_transform is not None:
            image_draw, mask_draw = self.page_transform(image_draw, mask_draw)
        if self.transform is not None and not self.augment_patches:
            image_draw, mask_draw = self.transform(image_draw, mask_draw)

        image_np = np.array(image_draw)
        mask_np = np.array(mask_draw)
        if image_np.ndim != 3 or mask_np.ndim != 2:
            raise ValueError("Unexpected image or mask format")

        height, width = image_np.shape[:2]
        mask_height, mask_width = mask_np.shape
        if mask_height != height or mask_width != width:
            raise ValueError("Mask resize failed to match image size")

        coords = _filter_numpy_patch_centers(
            np.argwhere(mask_np >= 125),
            height,
            width,
            self.patch_size,
        )
        if coords.size == 0:
            y_min, y_max, x_min, x_max = _valid_patch_center_bounds(
                height,
                width,
                self.patch_size,
            )
            ys = np.random.randint(y_min, y_max + 1, size=self.num_patches)
            xs = np.random.randint(x_min, x_max + 1, size=self.num_patches)
            coords = np.stack([ys, xs], axis=1)
        else:
            idx = np.random.choice(coords.shape[0], size=self.num_patches, replace=True)
            coords = coords[idx]

        padded = np.pad(
            image_np,
            ((half, half + extra), (half, half + extra), (0, 0)),
            mode="edge",
        )
        patches = []
        for y, x in coords:
            y = y + half
            x = x + half
            patch = padded[y - half : y + half + extra, x - half : x + half + extra, :]
            patches.append(patch)
        patches_np = np.stack(patches, axis=0)
        patches_tensor = torch.from_numpy(patches_np).permute(0, 3, 1, 2).float() / 255.0
        if self.val_gray:
            gray = patches_tensor.mean(dim=1, keepdim=True)
            patches_tensor = gray.repeat(1, patches_tensor.size(1), 1, 1)
        if self.augment_patches:
            if self.sequence_patch_augment_in_dataset:
                patches_tensor = _apply_sequence_patch_augment(
                    patches_tensor,
                    split_per_sample=self.sequence_patch_split_per_sample,
                    crop_size=self.sequence_patch_crop_size,
                    strong=self.strong_patch_augment,
                )
            elif self.transform is not None:
                patches_tensor = _apply_patch_transform(self.transform, patches_tensor)
        if self.augment_otsu and self.augment_patches:
            patches_tensor = torch.stack(
                [_apply_otsu_binarize(patch) for patch in patches_tensor], dim=0
            )
        if do_debug:
            elapsed = time.perf_counter() - debug_start
            print(f"[debug-first-batch] worker={worker_id} idx={index} done {elapsed:.3f}s")
        return patches_tensor, label


class BinarizedMemoryDataset(Dataset):
    def __init__(
        self,
        csv_path: Optional[str],
        image_root: str,
        mask_root: str,
        num_patches: int,
        patch_size: int = 32,
        transform: Optional[Callable] = None,
        page_transform: Optional[Callable] = None,
        augment_patches: bool = False,
        augment_otsu: bool = False,
        val_gray: bool = False,
        class_split_char: str = "_",
        class_id_index: int = 0,
        class_id_use_folder: bool = False,
        image_ext: Optional[Sequence[str]] = None,
        mask_ext: Optional[Sequence[str]] = None,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        cache_on_gpu: bool = False,
        cache_workers: int = 0,
        debug: bool = False,
        full_image_input: bool = False,
        full_image_height: int = 0,
        full_image_width: int = 0,
        full_image_pad_to_size: bool = False,
        full_image_resize_longest_side_first: bool = False,
        sequence_patch_augment_in_dataset: bool = False,
        sequence_patch_split_per_sample: int = 1,
        sequence_patch_crop_size: int = 32,
        strong_patch_augment: bool = False,
    ) -> None:
        self.class_to_idx: Dict[str, int] = {}
        self.image_ext = [ext.lower() for ext in (image_ext or [".jpg"])]
        self.mask_ext = [ext.lower() for ext in (mask_ext or [".png"])]
        if csv_path:
            self.samples = _read_csv(csv_path)
        else:
            if any(not ext.startswith(".") for ext in self.image_ext):
                raise ValueError("image_ext entries must start with '.'")
            if any(not ext.startswith(".") for ext in self.mask_ext):
                raise ValueError("mask_ext entries must start with '.'")
            self.samples, self.class_to_idx = _index_folder(
                image_root,
                class_split_char,
                tuple(self.image_ext),
                class_id_index,
                class_id_use_folder,
                debug=debug,
            )
        self.image_root = image_root
        self.mask_root = mask_root
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.transform = transform
        self.page_transform = page_transform
        self.augment_patches = augment_patches
        self.augment_otsu = augment_otsu
        self.val_gray = val_gray
        self.class_id_index = class_id_index
        self.class_id_use_folder = class_id_use_folder
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.cache_on_gpu = cache_on_gpu and torch.cuda.is_available()
        if cache_on_gpu and not torch.cuda.is_available():
            warnings.warn("cache_on_gpu requested but CUDA not available; using CPU.")
        self.cache_workers = max(0, int(cache_workers))
        self.full_image_input = bool(full_image_input)
        self.full_image_height = int(full_image_height)
        self.full_image_width = int(full_image_width)
        self.full_image_pad_to_size = bool(full_image_pad_to_size)
        self.full_image_resize_longest_side_first = bool(full_image_resize_longest_side_first)
        self.sequence_patch_augment_in_dataset = bool(sequence_patch_augment_in_dataset)
        self.sequence_patch_split_per_sample = max(1, int(sequence_patch_split_per_sample))
        self.sequence_patch_crop_size = int(sequence_patch_crop_size)
        self.strong_patch_augment = bool(strong_patch_augment)
        self._cache_device = torch.device("cuda") if self.cache_on_gpu else torch.device("cpu")
        self._image_cache: list[torch.Tensor] = []
        self._mask_cache: list[torch.Tensor] = []
        self._coords_cache: list[torch.Tensor] = []
        self._label_cache: list[int] = []
        self._max_cached_coords = 8192
        if debug:
            print(
                f"[dataset] binarized cache building root={image_root} "
                f"samples={len(self.samples)} workers={self.cache_workers}"
            )
        self._build_cache()

    def _build_cache(self) -> None:
        entries: list[Tuple[str, int]] = []
        for image_path, _, label in self.samples:
            if self.class_to_idx:
                resolved_image = image_path
            else:
                resolved_image = _resolve_path(self.image_root, image_path)
            entries.append((resolved_image, int(label)))

        total = len(entries)
        report_step = max(1, int(total * 0.05))
        next_report = report_step
        if self.cache_workers > 1:
            ctx = mp.get_context("fork")
            chunksize = max(1, len(entries) // (self.cache_workers * 4) or 1)
            with ctx.Pool(processes=self.cache_workers) as pool:
                for idx, (image_np, mask_np, label) in enumerate(
                    pool.imap(_load_binarized_sample, entries, chunksize=chunksize),
                    start=1,
                ):
                    image_tensor = torch.from_numpy(image_np).to(self._cache_device)
                    mask_tensor = torch.from_numpy(mask_np).to(self._cache_device)
                    coords = _filter_torch_patch_centers(
                        (mask_tensor >= 125).nonzero(as_tuple=False).to(dtype=torch.int32),
                        image_tensor.shape[0],
                        image_tensor.shape[1],
                        self.patch_size,
                    )
                    if coords.size(0) > self._max_cached_coords:
                        sel = torch.randperm(coords.size(0), device=coords.device)[: self._max_cached_coords]
                        coords = coords[sel]
                    self._image_cache.append(image_tensor)
                    self._mask_cache.append(mask_tensor)
                    self._coords_cache.append(coords)
                    self._label_cache.append(int(label))
                    if idx >= next_report:
                        pct = int(round(100.0 * idx / max(1, total)))
                        print(f"[dataset] binarized cache {pct}% ({idx}/{total})")
                        next_report += report_step
        else:
            for idx, (image_path, label) in enumerate(entries, start=1):
                image_np, mask_np, label = _load_binarized_sample((image_path, label))
                image_tensor = torch.from_numpy(image_np).to(self._cache_device)
                mask_tensor = torch.from_numpy(mask_np).to(self._cache_device)
                coords = _filter_torch_patch_centers(
                    (mask_tensor >= 125).nonzero(as_tuple=False).to(dtype=torch.int32),
                    image_tensor.shape[0],
                    image_tensor.shape[1],
                    self.patch_size,
                )
                if coords.size(0) > self._max_cached_coords:
                    sel = torch.randperm(coords.size(0), device=coords.device)[: self._max_cached_coords]
                    coords = coords[sel]
                self._image_cache.append(image_tensor)
                self._mask_cache.append(mask_tensor)
                self._coords_cache.append(coords)
                self._label_cache.append(int(label))
                if idx >= next_report:
                    pct = int(round(100.0 * idx / max(1, total)))
                    print(f"[dataset] binarized cache {pct}% ({idx}/{total})")
                    next_report += report_step

    def __len__(self) -> int:
        return len(self._image_cache)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_tensor = self._image_cache[index]
        mask_tensor = self._mask_cache[index]
        coords_source = self._coords_cache[index]
        label = self._label_cache[index]

        if self.full_image_input:
            image_chw = image_tensor.permute(2, 0, 1).float() / 255.0
            target_h = self.full_image_height if self.full_image_height > 0 else self.patch_size
            target_w = self.full_image_width if self.full_image_width > 0 else self.patch_size
            if image_chw.size(-2) != target_h or image_chw.size(-1) != target_w:
                if self.full_image_pad_to_size:
                    if self.full_image_resize_longest_side_first:
                        image_chw = _resize_longest_side(image_chw, max(target_h, target_w))
                    image_chw = _pad_or_center_crop_to_size(image_chw, target_h, target_w)
                else:
                    image_chw = F.interpolate(
                        image_chw.unsqueeze(0),
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)
            if self.val_gray:
                gray = image_chw.mean(dim=0, keepdim=True)
                image_chw = gray.repeat(image_chw.size(0), 1, 1)
            if self.transform is not None and self.augment_patches:
                image_chw = self.transform(image_chw)
            if self.augment_otsu and self.augment_patches:
                image_chw = _apply_otsu_binarize(image_chw)
            return image_chw.unsqueeze(0), label

        coords = None
        image_draw = image_tensor
        mask_draw = mask_tensor

        # Fast path for the common training setup (augment_patches=True): avoid
        # costly PIL roundtrips by applying page crop directly on tensors.
        if self.page_transform is not None and self.augment_patches:
            image_chw = image_tensor.permute(2, 0, 1).float() / 255.0
            height, width = image_chw.shape[-2], image_chw.shape[-1]
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image_chw, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)
            )
            image_chw = TF.resized_crop(
                image_chw,
                i,
                j,
                h,
                w,
                size=[height, width],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            image_draw = (image_chw * 255.0).clamp(0, 255).byte().permute(1, 2, 0)

            if coords_source.numel() > 0:
                ys = coords_source[:, 0]
                xs = coords_source[:, 1]
                inside = (ys >= i) & (ys < i + h) & (xs >= j) & (xs < j + w)
                coords_in = coords_source[inside]
                if coords_in.numel() > 0:
                    idx = torch.randint(0, coords_in.size(0), (self.num_patches,), device=coords_in.device)
                    coords_pick = coords_in[idx]
                    ys = ((coords_pick[:, 0] - i).float() * (height / float(h))).long().clamp(0, height - 1)
                    xs = ((coords_pick[:, 1] - j).float() * (width / float(w))).long().clamp(0, width - 1)
                    coords = _filter_torch_patch_centers(
                        torch.stack([ys, xs], dim=1),
                        height,
                        width,
                        self.patch_size,
                    )
        elif self.page_transform is not None:
            image_pil = Image.fromarray(image_draw.cpu().numpy())
            mask_pil = Image.fromarray(mask_draw.cpu().numpy())
            image_pil, mask_pil = self.page_transform(image_pil, mask_pil)
            image_draw = torch.from_numpy(np.array(image_pil)).to(image_tensor.device)
            mask_draw = torch.from_numpy(np.array(mask_pil)).to(mask_tensor.device)

        if self.transform is not None and not self.augment_patches:
            image_draw = image_draw.permute(2, 0, 1).float() / 255.0
            mask_draw = mask_draw.unsqueeze(0).float() / 255.0
            image_draw, mask_draw = self.transform(image_draw, mask_draw)
            image_draw = (image_draw * 255.0).clamp(0, 255).byte()
            mask_draw = (mask_draw * 255.0).clamp(0, 255).byte().squeeze(0)
            image_draw = image_draw.permute(1, 2, 0)
        else:
            image_draw = image_draw.clone()
            mask_draw = mask_draw.clone()

        height, width = image_draw.shape[-3], image_draw.shape[-2]
        if coords is None:
            can_use_cached_coords = self.page_transform is None and (
                self.transform is None or self.augment_patches
            )
            if can_use_cached_coords:
                coords = coords_source
            else:
                coords = (mask_draw >= 125).nonzero(as_tuple=False).to(dtype=torch.int32)
            coords = _filter_torch_patch_centers(
                coords,
                height,
                width,
                self.patch_size,
            )

        if coords.numel() == 0:
            y_min, y_max, x_min, x_max = _valid_patch_center_bounds(
                height,
                width,
                self.patch_size,
            )
            ys = torch.randint(
                y_min,
                y_max + 1,
                (self.num_patches,),
                device=image_draw.device,
            )
            xs = torch.randint(
                x_min,
                x_max + 1,
                (self.num_patches,),
                device=image_draw.device,
            )
            coords = torch.stack([ys, xs], dim=1)
        else:
            idx = torch.randint(0, coords.size(0), (self.num_patches,), device=image_draw.device)
            coords = coords[idx]

        patches = []
        for yx in coords:
            patches.append(
                _crop_hwc_patch_replicate(
                    image_draw,
                    int(yx[0].item()),
                    int(yx[1].item()),
                    self.patch_size,
                )
            )
        patches_tensor = torch.stack(patches, dim=0)
        if self.val_gray:
            gray = patches_tensor.mean(dim=1, keepdim=True)
            patches_tensor = gray.repeat(1, patches_tensor.size(1), 1, 1)
        if self.augment_patches:
            if self.sequence_patch_augment_in_dataset:
                patches_tensor = _apply_sequence_patch_augment(
                    patches_tensor,
                    split_per_sample=self.sequence_patch_split_per_sample,
                    crop_size=self.sequence_patch_crop_size,
                    strong=self.strong_patch_augment,
                )
            elif self.transform is not None:
                patches_tensor = _apply_patch_transform(self.transform, patches_tensor)
        if self.augment_otsu and self.augment_patches:
            patches_tensor = torch.stack(
                [_apply_otsu_binarize(patch) for patch in patches_tensor], dim=0
            )
        return patches_tensor, label
