#!/usr/bin/env python3
"""Create and download the FormWR/CM1 dataset from the Arolsen Archives.

The public Arolsen Archives CM/1 collection contains more files than the
FormWR subset used by this project. Use ``make-manifest`` against an existing
curated split, such as ``data/cm1_final``, to record the target files.
Then use ``download`` to fetch the corresponding hosted CM/1 originals and
recreate the split layout.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import gzip
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


API_BASE = "https://collections-server.arolsen-archives.org/ITS-WS.asmx/"
DEFAULT_SPLITS = ("train", "val", "test")
VALID_SIDES = {"left", "right"}
USER_AGENT = "patchformer-release-formwr-downloader/1.0"


@dataclass(frozen=True)
class FormWREntry:
    split: str
    filename: str
    obj_id: str
    scan_id: str
    page: str
    side: str
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    bytes: int | None = None

    @property
    def source_key(self) -> str:
        return f"{self.scan_id}/{self.page}"


def _request_json(url: str, payload: dict[str, Any], timeout: float, retries: int) -> Any:
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": USER_AGENT,
    }
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(min(8.0, 0.75 * (2**attempt)))
    raise RuntimeError(f"request failed for {url}: {last_exc}") from last_exc


def _download_bytes(url: str, timeout: float, retries: int) -> bytes:
    headers = {"User-Agent": USER_AGENT}
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(min(8.0, 0.75 * (2**attempt)))
    raise RuntimeError(f"download failed for {url}: {last_exc}") from last_exc


def _normalize_arolsen_url(url: str) -> str:
    # The API sometimes returns strings like ``https:\\collections-server...``.
    url = url.strip().replace("\\\\", "/").replace("\\", "/")
    if url.startswith("https:/") and not url.startswith("https://"):
        url = "https://" + url[len("https:/") :]
    if url.startswith("/remote/"):
        url = "https://" + url[len("/remote/") :]
    return url


def _parse_final_filename(filename: str, split: str) -> FormWREntry:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) not in (4, 5):
        raise ValueError(f"unexpected FormWR filename in {split}: {filename}")
    side = ""
    if len(parts) == 5:
        side = parts[4]
        if side not in VALID_SIDES:
            raise ValueError(f"unexpected side suffix in {split}: {filename}")
    return FormWREntry(
        split=split,
        filename=filename,
        obj_id=parts[0],
        scan_id=parts[2],
        page=parts[3],
        side=side,
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def _default_summary_path(manifest_path: Path) -> Path:
    if manifest_path.name.endswith(".jsonl.gz"):
        return manifest_path.with_name(manifest_path.name[: -len(".jsonl.gz")] + ".summary.json")
    if manifest_path.suffix == ".gz":
        return manifest_path.with_suffix("").with_suffix(".summary.json")
    return manifest_path.with_suffix(manifest_path.suffix + ".summary.json")


def _iter_final_files(root: Path, splits: Iterable[str]) -> Iterable[FormWREntry]:
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split directory: {split_dir}")
        for path in sorted(split_dir.glob("*.jpg")):
            yield _parse_final_filename(path.name, split)


def make_manifest(args: argparse.Namespace) -> int:
    source_root = Path(args.source_root)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"splits": {}, "total": 0, "with_side": 0, "full_page": 0}

    with _open_text(manifest_path, "wt") as out:
        for entry in _iter_final_files(source_root, args.splits):
            path = source_root / entry.split / entry.filename
            row = entry.__dict__.copy()
            row["bytes"] = path.stat().st_size
            if args.include_sha256:
                row["sha256"] = _sha256(path)
            if args.include_dimensions:
                with Image.open(path) as image:
                    row["width"], row["height"] = image.size
            out.write(json.dumps(row, sort_keys=True) + "\n")

            split_summary = summary["splits"].setdefault(
                entry.split, {"count": 0, "with_side": 0, "full_page": 0}
            )
            split_summary["count"] += 1
            summary["total"] += 1
            if entry.side:
                split_summary["with_side"] += 1
                summary["with_side"] += 1
            else:
                split_summary["full_page"] += 1
                summary["full_page"] += 1

    if args.summary:
        summary_path = Path(args.summary)
    else:
        summary_path = _default_summary_path(manifest_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _load_manifest(path: Path) -> list[FormWREntry]:
    entries = []
    with _open_text(path, "rt") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                entries.append(
                    FormWREntry(
                        split=row["split"],
                        filename=row["filename"],
                        obj_id=str(row["obj_id"]),
                        scan_id=str(row["scan_id"]),
                        page=str(row["page"]),
                        side=str(row.get("side") or ""),
                        width=row.get("width"),
                        height=row.get("height"),
                        sha256=row.get("sha256"),
                        bytes=row.get("bytes"),
                    )
                )
            except KeyError as exc:
                raise ValueError(f"manifest line {line_no} is missing {exc}") from exc
    return entries


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        tmp_name = handle.name
        handle.write(data)
    os.replace(tmp_name, path)


def _save_image_atomic(path: Path, image: Image.Image, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".jpg", dir=path.parent, delete=False) as handle:
        tmp_name = handle.name
    try:
        image.save(tmp_name, format="JPEG", quality=quality, optimize=True)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


class ArolsenResolver:
    def __init__(self, cache_path: Path, timeout: float, retries: int, id_window: int) -> None:
        self.cache_path = cache_path
        self.timeout = timeout
        self.retries = retries
        self.id_window = id_window
        self.cache: dict[str, list[dict[str, Any]]] = {}
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                self.cache = json.load(handle)

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8")

    def _fetch_obj(self, obj_id: str) -> list[dict[str, Any]]:
        if obj_id not in self.cache:
            payload = {"objId": obj_id, "lang": "en"}
            response = _request_json(API_BASE + "GetFileByObj", payload, self.timeout, self.retries)
            self.cache[obj_id] = response.get("d", [])
        return self.cache[obj_id]

    def resolve(self, entry: FormWREntry) -> str:
        candidate_ids = [entry.obj_id]
        if self.id_window > 0 and entry.obj_id.isdigit():
            base = int(entry.obj_id)
            for offset in range(1, self.id_window + 1):
                candidate_ids.append(str(base - offset))
                candidate_ids.append(str(base + offset))

        needle = f"/{entry.scan_id}/{entry.page}.jpg"
        for obj_id in candidate_ids:
            for item in self._fetch_obj(obj_id):
                image_url = _normalize_arolsen_url(str(item.get("image", "")))
                if needle in image_url:
                    return image_url
        raise LookupError(f"could not resolve {entry.filename} via obj_id={entry.obj_id}")


def _crop_side(image: Image.Image, side: str) -> Image.Image:
    if side not in VALID_SIDES:
        return image
    width, height = image.size
    midpoint = width // 2
    if side == "left":
        return image.crop((0, 0, midpoint, height))
    return image.crop((midpoint, 0, width, height))


def _download_one(
    entry: FormWREntry,
    image_url: str,
    output_root: Path,
    originals_root: Path,
    timeout: float,
    retries: int,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[str, str]:
    out_path = output_root / entry.split / entry.filename
    if out_path.exists() and not overwrite:
        return entry.filename, "exists"

    original_path = originals_root / entry.scan_id / f"{entry.page}.jpg"
    if not original_path.exists() or overwrite:
        data = _download_bytes(image_url, timeout, retries)
        _atomic_write(original_path, data)

    if entry.side:
        with Image.open(original_path) as image:
            cropped = _crop_side(image.convert("RGB"), entry.side)
            _save_image_atomic(out_path, cropped, jpeg_quality)
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not out_path.exists():
            shutil.copyfile(original_path, out_path)

    if entry.width is not None or entry.height is not None:
        with Image.open(out_path) as image:
            if entry.width is not None and image.width != entry.width:
                raise ValueError(f"{out_path} width {image.width} != manifest {entry.width}")
            if entry.height is not None and image.height != entry.height:
                raise ValueError(f"{out_path} height {image.height} != manifest {entry.height}")
    return entry.filename, "downloaded"


def download(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest)
    entries = _load_manifest(manifest_path)
    if args.limit:
        entries = entries[: args.limit]

    output_root = Path(args.output_root)
    originals_root = Path(args.originals_root) if args.originals_root else output_root / "_arolsen_originals"
    resolver = ArolsenResolver(
        cache_path=Path(args.lookup_cache),
        timeout=args.timeout,
        retries=args.retries,
        id_window=args.id_window,
    )

    print(f"Loaded {len(entries)} manifest entries")
    print(f"Output root: {output_root}")
    print(f"Originals cache: {originals_root}")

    resolved: list[tuple[FormWREntry, str]] = []
    failures: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        try:
            resolved.append((entry, resolver.resolve(entry)))
        except Exception as exc:
            failures.append(f"{entry.filename}: {exc}")
        if idx % args.cache_save_every == 0:
            resolver.save()
        if idx % args.log_every == 0:
            print(f"Resolved {idx}/{len(entries)} entries; failures={len(failures)}", flush=True)
    resolver.save()

    if failures:
        failure_path = output_root / "download_failures.txt"
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Resolution failures: {len(failures)} (wrote {failure_path})")
        if not args.keep_going:
            return 1

    counts = {"exists": 0, "downloaded": 0, "failed": 0}
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                _download_one,
                entry,
                image_url,
                output_root,
                originals_root,
                args.timeout,
                args.retries,
                args.jpeg_quality,
                args.overwrite,
            ): entry
            for entry, image_url in resolved
        }
        for idx, future in enumerate(futures.as_completed(future_map), start=1):
            entry = future_map[future]
            try:
                _, status = future.result()
                counts[status] += 1
            except Exception as exc:
                counts["failed"] += 1
                print(f"FAILED {entry.filename}: {exc}", file=sys.stderr, flush=True)
                if not args.keep_going:
                    raise
            if idx % args.log_every == 0:
                print(f"Downloaded {idx}/{len(resolved)} | {counts}", flush=True)

    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0 if counts["failed"] == 0 else 1


def verify(args: argparse.Namespace) -> int:
    entries = _load_manifest(Path(args.manifest))
    if args.limit:
        entries = entries[: args.limit]
    root = Path(args.root)
    counts = {"checked": 0, "missing": 0, "dimension_mismatch": 0, "sha256_mismatch": 0}
    problems: list[str] = []

    for entry in entries:
        path = root / entry.split / entry.filename
        counts["checked"] += 1
        if not path.exists():
            counts["missing"] += 1
            problems.append(f"missing {path}")
            continue
        if entry.width is not None or entry.height is not None:
            with Image.open(path) as image:
                if entry.width is not None and image.width != entry.width:
                    counts["dimension_mismatch"] += 1
                    problems.append(f"width mismatch {path}: {image.width} != {entry.width}")
                if entry.height is not None and image.height != entry.height:
                    counts["dimension_mismatch"] += 1
                    problems.append(f"height mismatch {path}: {image.height} != {entry.height}")
        if entry.sha256 and args.check_sha256:
            actual = _sha256(path)
            if actual != entry.sha256:
                counts["sha256_mismatch"] += 1
                problems.append(f"sha256 mismatch {path}: {actual} != {entry.sha256}")

    if problems:
        for problem in problems[: args.max_problems]:
            print(problem, file=sys.stderr)
        if len(problems) > args.max_problems:
            print(f"... {len(problems) - args.max_problems} more problems", file=sys.stderr)
    print(json.dumps(counts, indent=2, sort_keys=True))
    return 0 if not problems else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("make-manifest", help="Create a JSONL manifest from a curated FormWR split")
    manifest.add_argument("--source-root", default="data/cm1_final")
    manifest.add_argument("--manifest", required=True)
    manifest.add_argument("--summary", default="")
    manifest.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    manifest.add_argument("--include-sha256", action="store_true")
    manifest.add_argument("--include-dimensions", action=argparse.BooleanOptionalAction, default=True)
    manifest.set_defaults(func=make_manifest)

    dl = subparsers.add_parser("download", help="Download and recreate the manifest split from Arolsen CM/1")
    dl.add_argument("--manifest", required=True)
    dl.add_argument("--output-root", required=True)
    dl.add_argument("--originals-root", default="")
    dl.add_argument("--lookup-cache", default="outputs/formwr_arolsen_lookup_cache.json")
    dl.add_argument("--workers", type=int, default=4)
    dl.add_argument("--timeout", type=float, default=60.0)
    dl.add_argument("--retries", type=int, default=3)
    dl.add_argument("--id-window", type=int, default=2, help="Fallback search around obj_id when a file is not found")
    dl.add_argument("--jpeg-quality", type=int, default=95)
    dl.add_argument("--overwrite", action="store_true")
    dl.add_argument("--keep-going", action="store_true")
    dl.add_argument("--limit", type=int, default=0, help="Debug: process only the first N manifest rows")
    dl.add_argument("--log-every", type=int, default=1000)
    dl.add_argument("--cache-save-every", type=int, default=500)
    dl.set_defaults(func=download)

    vf = subparsers.add_parser("verify", help="Verify a downloaded split against a manifest")
    vf.add_argument("--manifest", required=True)
    vf.add_argument("--root", required=True)
    vf.add_argument("--check-sha256", action="store_true")
    vf.add_argument("--limit", type=int, default=0)
    vf.add_argument("--max-problems", type=int, default=50)
    vf.set_defaults(func=verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
