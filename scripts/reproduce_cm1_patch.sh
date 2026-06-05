#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/main/cm1_xvlad.yaml
  configs/ablations/cm1_xvlad_heads.yaml
  configs/ablations/cm1_xvlad_ghost_centers.yaml
  configs/ablations/cm1_xvlad_losses.yaml
  configs/ablations/cm1_xvlad_aggregators.yaml
)

for cfg in "${configs[@]}"; do
  python experiments/run_experiments.py "$cfg" --max-parallel "${PATCHFORMER_MAX_PARALLEL:-1}" --gpus "${PATCHFORMER_GPUS:-}"
done
