#!/usr/bin/env python3
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class PackTask:
    in_dir: str
    rel_dir: str
    files: Tuple[str, ...]
    out_file: str
    strict_size: bool


def _normalize_exts(exts: Sequence[str]) -> Tuple[str, ...]:
    out = []
    for ext in exts:
        ext = ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        out.append(ext)
    return tuple(sorted(set(out)))


def _iter_patch_dirs(root: str, exts: Tuple[str, ...]) -> Iterable[Tuple[str, str, Tuple[str, ...]]]:
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        patch_files = tuple(sorted(f for f in filenames if os.path.splitext(f)[1].lower() in exts))
        if not patch_files:
            continue
        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""
        yield dirpath, rel_dir, patch_files


def _pack_one(task: PackTask) -> Tuple[str, int, int, int, int]:
    arrays: List[np.ndarray] = []
    target_hw = None
    for fname in task.files:
        path = os.path.join(task.in_dir, fname)
        with Image.open(path) as image:
            arr = np.array(image.convert("RGB"), dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unexpected patch format: {path} -> {arr.shape}")
        hw = (arr.shape[0], arr.shape[1])
        if target_hw is None:
            target_hw = hw
        elif hw != target_hw:
            if task.strict_size:
                raise ValueError(
                    f"Mixed patch sizes in {task.in_dir}: first={target_hw}, got={hw} ({fname})"
                )
            resized = Image.fromarray(arr).resize((target_hw[1], target_hw[0]), resample=Image.BILINEAR)
            arr = np.array(resized, dtype=np.uint8)
        arrays.append(arr)

    if not arrays:
        raise RuntimeError(f"No patches found in {task.in_dir}")

    stacked = np.stack(arrays, axis=0)  # (N, H, W, C), uint8
    patches = torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()
    payload = {
        "patches": patches,  # uint8 tensor, shape (N, 3, H, W)
        "filenames": list(task.files),
        "source_rel_dir": task.rel_dir,
    }

    os.makedirs(os.path.dirname(task.out_file), exist_ok=True)
    tmp_file = task.out_file + ".tmp"
    torch.save(payload, tmp_file)
    os.replace(tmp_file, task.out_file)

    n, _, h, w = patches.shape
    size_bytes = os.path.getsize(task.out_file)
    return task.rel_dir, int(n), int(h), int(w), int(size_bytes)


def _build_tasks(
    input_root: str,
    output_root: str,
    exts: Tuple[str, ...],
    overwrite: bool,
    strict_size: bool,
) -> List[PackTask]:
    tasks: List[PackTask] = []
    for in_dir, rel_dir, files in _iter_patch_dirs(input_root, exts):
        out_file = os.path.join(output_root, rel_dir + ".pt") if rel_dir else os.path.join(output_root, "root.pt")
        if not overwrite and os.path.exists(out_file):
            continue
        tasks.append(
            PackTask(
                in_dir=in_dir,
                rel_dir=rel_dir,
                files=files,
                out_file=out_file,
                strict_size=strict_size,
            )
        )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack per-page patch folders into .pt files (one file per patch folder)."
    )
    parser.add_argument("--input-root", type=str, required=True, help="Root with patch folders.")
    parser.add_argument("--output-root", type=str, required=True, help="Output root for packed .pt files.")
    parser.add_argument(
        "--ext",
        type=str,
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Patch file extensions to include.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Parallel workers.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing packed files.")
    parser.add_argument(
        "--allow-resize",
        action="store_true",
        help="If patch sizes differ in one folder, resize to first patch size instead of failing.",
    )
    parser.add_argument("--log-every", type=int, default=500, help="Progress print frequency.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)
    exts = _normalize_exts(args.ext)
    strict_size = not args.allow_resize

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Input root not found: {input_root}")
    os.makedirs(output_root, exist_ok=True)

    tasks = _build_tasks(
        input_root=input_root,
        output_root=output_root,
        exts=exts,
        overwrite=args.overwrite,
        strict_size=strict_size,
    )
    total = len(tasks)
    if total == 0:
        print("No work to do (all packed files already exist).")
        return

    workers = max(1, int(args.workers))
    print(
        f"Packing {total} patch dirs from {input_root} -> {output_root} "
        f"(workers={workers}, exts={exts})"
    )

    done = 0
    total_patches = 0
    total_bytes = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_pack_one, task) for task in tasks]
        with tqdm(total=total, desc="Packing patch dirs", unit="dir") as pbar:
            for future in as_completed(futures):
                rel_dir, count, _, _, size_bytes = future.result()
                done += 1
                total_patches += count
                total_bytes += size_bytes
                pbar.update(1)
                if done == 1 or done % max(1, args.log_every) == 0 or done == total:
                    pbar.set_postfix(
                        last=rel_dir or "<root>",
                        patches=total_patches,
                        size_gib=f"{total_bytes / (1024 * 1024 * 1024):.2f}",
                        refresh=False,
                    )

    print(
        f"Done. Packed dirs={done}, patches={total_patches}, "
        f"size={total_bytes / (1024 * 1024 * 1024):.2f} GiB"
    )


if __name__ == "__main__":
    main()
