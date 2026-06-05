#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run patch extraction and packing in one go."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Input image root.")
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="",
        help="Mask root (required when --mask-mode=mask).",
    )
    parser.add_argument(
        "--staging-dir",
        type=str,
        required=True,
        help="Temporary folder for unpacked per-image patches.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Final output root containing packed .pt files.",
    )
    parser.add_argument("--num-patches", type=int, default=2048)
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--patch-min-distance", type=int, default=0)
    parser.add_argument("--mask-mode", choices=["edges", "sift", "mask"], default="mask")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="Workers for extraction.")
    parser.add_argument(
        "--pack-workers",
        type=int,
        default=0,
        help="Workers for packing (0 = auto).",
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=[".jpg", ".png", ".jpeg", ".tif", ".tiff"],
    )
    parser.add_argument("--patch-ext", nargs="+", default=[".png", ".jpg", ".jpeg"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Keep unpacked patches after successful packing.",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    prepare_script = repo_root / "prepare_data.py"
    pack_script = repo_root / "tools" / "pack_precomputed_patches.py"

    staging_dir = Path(args.staging_dir).resolve()
    output_root = Path(args.output_root).resolve()
    if args.overwrite and staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    prepare_cmd = [
        sys.executable,
        str(prepare_script),
        "--input-dir",
        args.input_dir,
        "--out-dir",
        str(staging_dir),
        "--num-patches",
        str(args.num_patches),
        "--patch-size",
        str(args.patch_size),
        "--mask-mode",
        args.mask_mode,
        "--patch-min-distance",
        str(args.patch_min_distance),
        "--seed",
        str(args.seed),
    ]
    if args.workers > 0:
        prepare_cmd.extend(["--workers", str(args.workers)])
    if args.mask_mode == "mask":
        if not args.mask_dir:
            raise ValueError("--mask-dir is required with --mask-mode mask")
        prepare_cmd.extend(["--mask-dir", args.mask_dir])
    if args.image_exts:
        prepare_cmd.extend(["--image-exts", *args.image_exts])
    _run(prepare_cmd)

    pack_cmd = [
        sys.executable,
        str(pack_script),
        "--input-root",
        str(staging_dir),
        "--output-root",
        str(output_root),
    ]
    if args.patch_ext:
        pack_cmd.extend(["--ext", *args.patch_ext])
    if args.pack_workers > 0:
        pack_cmd.extend(["--workers", str(args.pack_workers)])
    if args.overwrite:
        pack_cmd.append("--overwrite")
    _run(pack_cmd)

    if not args.keep_staging:
        shutil.rmtree(staging_dir, ignore_errors=True)
        print(f"Removed staging dir: {staging_dir}")
    print(f"Packed patches available at: {output_root}")


if __name__ == "__main__":
    main()
