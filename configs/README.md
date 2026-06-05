# Configuration Layout

Run every file below with `python experiments/run_experiments.py <config>`.

- `main/cm1_xvlad.yaml`: CM1 patch-level X-VLAD baseline and compact variants.
- `main/hisfrag20_xvlad.yaml`: HisFrag20 patch-level X-VLAD runs based on the original `final_numbers_hisfrag20.yaml`.
- `ablations/cm1_xvlad_heads.yaml`: number-of-heads ablation.
- `ablations/cm1_xvlad_ghost_centers.yaml`: ghost-center ablation matching the active subset of the old monolithic CM1 ablation config.
- `ablations/cm1_xvlad_losses.yaml`: ArcFace/triplet objective ablations.
- `ablations/cm1_xvlad_aggregators.yaml`: X-VLAD, GhostVLAD, and mean-pooling aggregation comparisons.
- `finetune/cm1_fullpage_xvlad.yaml`: full-page finetuning runs initialized from patch-level checkpoints.

The top-level `cm1_xvlad.yaml`, `cm1_xvlad_ablations.yaml`, and `hisfrag20_xvlad.yaml` files are convenience shims for common runs.
