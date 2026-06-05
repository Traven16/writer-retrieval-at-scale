#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATCHFORMER_DATA_ROOT="${PATCHFORMER_DATA_ROOT:-data}"
MAX_PARALLEL="${PATCHFORMER_MAX_PARALLEL:-4}"
GPUS="${PATCHFORMER_GPUS:-0,1,2,3}"

python experiments/run_experiments.py \
  configs/semi_supervised/cm1_writer_fraction_xvlad.yaml \
  --max-parallel "$MAX_PARALLEL" \
  --gpus "$GPUS"
