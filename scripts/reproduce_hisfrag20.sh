#!/usr/bin/env bash
set -euo pipefail

python experiments/run_experiments.py \
  configs/main/hisfrag20_xvlad.yaml \
  --max-parallel "${PATCHFORMER_MAX_PARALLEL:-1}" \
  --gpus "${PATCHFORMER_GPUS:-}"

