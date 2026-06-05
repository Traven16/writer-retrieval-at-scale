# Patchformer Research Release

This folder contains the research code needed to reproduce the CM1 Patchformer/X-VLAD training runs. The application under `app/`, local databases, cached bytecode, checkpoints, and prior run outputs are intentionally excluded.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if the default `pip` wheel is not appropriate.

## Data Layout

The CM1 configs expect the `train`/`val` splits under `PATCHFORMER_DATA_ROOT`:

```text
$PATCHFORMER_DATA_ROOT/
  cm1_final/train/          # training images (produced by the download below)
  cm1_final/val/            # validation images
  cm1_final_masks/train/    # segmentation masks per training image
  cm1_final_masks/val/      # segmentation masks per validation image
  cm1_final_patches/train/  # precomputed packed patch tensors for training
```

The `cm1_final/{train,val,test}` image splits come straight from the download
step below. The parallel `cm1_final_masks/` and `cm1_final_patches/` trees are
derived artifacts (masks and packed patches) keyed by the same filenames.

By default, configs look under `data/`. Set `PATCHFORMER_DATA_ROOT` if your
data lives elsewhere:

```bash
export PATCHFORMER_DATA_ROOT=/path/to/data
```

The `cm1_final/` images come from the download step below; `cm1_final_masks/`
and `cm1_final_patches/` are prepared in **Masks and Patches**.

## FormWR / CM1 Download

The public Arolsen Archives CM/1 collection contains more files than the
FormWR subset used here. This release includes a compressed manifest for the
curated final split:

- `manifests/formwr_cm1_final_manifest.jsonl.gz`
- `manifests/formwr_cm1_final_manifest.summary.json`

Download only the manifest-pinned files:

```bash
python tools/download_formwr.py download \
  --manifest manifests/formwr_cm1_final_manifest.jsonl.gz \
  --output-root data/cm1_final \
  --workers 4 \
  --keep-going

python tools/download_formwr.py verify \
  --manifest manifests/formwr_cm1_final_manifest.jsonl.gz \
  --root data/cm1_final
```

This writes the `train/`, `val/`, and `test/` image splits under
`data/cm1_final/` — the same `cm1_final/{train,val}` paths the CM1 configs read.

Each manifest row pins one exact hosted page via the final filename's scan id
and page number; other pages returned for the same CM/1 process are ignored.
The downloader queries the Arolsen Archives backend only to discover the CM/1
original image URL, caches originals under `<output-root>/_arolsen_originals/`,
and writes the curated `train/`, `val/`, and `test/` split. Filenames ending in
`_left` or `_right` are recreated by cropping the hosted full-page image.

If you need to regenerate the packaged manifest from the local curated split:

```bash
python tools/download_formwr.py make-manifest \
  --source-root /path/to/cm1_final \
  --manifest manifests/formwr_cm1_final_manifest.jsonl.gz
```

## Masks and Patches

The download step provides only the page images. Training also needs the
text-region **masks** and, for training, precomputed packed **patches**.

### Masks

The masks are distributed separately as compact 1-bit PNGs (binarized at 50%),
keyed by the same filenames as the image splits. Download and unpack them so the
layout matches the config paths:

```text
$PATCHFORMER_DATA_ROOT/cm1_final_masks/{train,val,test}/<image-stem>.png
```

```bash
# Download cm1_final_masks.tar from the dataset release, then:
tar -xf cm1_final_masks.tar -C data/        # -> data/cm1_final_masks/{train,val,test}
```

> Mask archive: `<ZENODO_URL>` (≈2 GB). They are plain `.png` files the loader
> reads via `Image.open(...).convert("L")`, so no special handling is required.

### Patches

The CM1 configs read precomputed packed patch tensors for the **train** split
(`train_use_precomputed_patches: true`); the **val** split samples patches live
from the images and masks, so it needs masks but not packed patches. Build the
train patches once from the images and masks:

```bash
python tools/extract_and_pack_patches.py \
  --input-dir data/cm1_final/train \
  --mask-dir data/cm1_final_masks/train \
  --staging-dir data/_patch_staging/train \
  --output-root data/cm1_final_patches/train \
  --mask-mode mask --patch-size 32 --num-patches 2048
```

This writes the packed `.pt` files the configs expect under
`cm1_final_patches/train/`. (`extract_and_pack_patches.py` chains
`prepare_data.py` for extraction and `tools/pack_precomputed_patches.py` for
packing; run them separately if you prefer.)

## Training

The training entry point is unchanged:

```bash
python experiments/run_experiments.py configs/main/cm1_xvlad.yaml
```

HisFrag20, based on the original `final_numbers_hisfrag20.yaml` settings:

```bash
python experiments/run_experiments.py configs/main/hisfrag20_xvlad.yaml
```

The active subset from the old monolithic CM1 ablation config is now:

```bash
python experiments/run_experiments.py configs/ablations/cm1_xvlad_ghost_centers.yaml
```

Run a command-only check across all release configs:

```bash
bash scripts/dry_run_configs.sh
```

Run the CM1 and HisFrag20 patch-level suites:

```bash
bash scripts/reproduce_main_results.sh
```

Outputs are written under `outputs/` by default. Override
`PATCHFORMER_OUTPUT_ROOT` and `PATCHFORMER_CHECKPOINT_ROOT` to place them
elsewhere.

- `outputs/results/.../<run_name>/config.yaml`
- `outputs/results/.../<run_name>/stdout.log`
- `outputs/results/.../<run_name>/summary.json`
- `outputs/checkpoints/.../<run_name>.pt`
- `outputs/runs/.../<run_name>/`

## Configs

Configs are split by responsibility:

- `configs/_base/`: reusable data, model, objective, and runtime defaults.
- `configs/main/`: main CM1 baseline runs.
- `configs/ablations/`: independent ablation groups.
- `configs/finetune/`: full-page finetuning initialized from patch-level checkpoints.

The code identifier is `xvlad`; paper-facing text uses `X-VLAD`.
