#!/usr/bin/env bash
set -euo pipefail

configs=(
  configs/main/cm1_xvlad.yaml
  configs/main/hisfrag20_xvlad.yaml
  configs/ablations/cm1_xvlad_heads.yaml
  configs/ablations/cm1_xvlad_ghost_centers.yaml
  configs/ablations/cm1_xvlad_losses.yaml
  configs/ablations/cm1_xvlad_aggregators.yaml
  configs/finetune/cm1_fullpage_xvlad.yaml
)

for cfg in "${configs[@]}"; do
  python experiments/run_experiments.py "$cfg" --dry-run
done
