# Experiment Workflow

The release uses YAML configs under `configs/` and the runner `experiments/run_experiments.py`.

## Dry Run

```bash
python experiments/run_experiments.py configs/main/cm1_xvlad.yaml --dry-run
python experiments/run_experiments.py configs/ablations/cm1_xvlad_ghost_centers.yaml --dry-run
```

## Run

```bash
export PATCHFORMER_DATA_ROOT=/path/to/data
export PATCHFORMER_OUTPUT_ROOT=/path/to/outputs
python experiments/run_experiments.py configs/main/cm1_xvlad.yaml --max-parallel 1 --gpus 0
```

## Config Inheritance

Configs can use:

```yaml
extends:
  - ../_base/cm1_data.yaml
  - ../_base/xvlad_resnet18.yaml

defaults:
  run:
    results_dir: ${PATCHFORMER_OUTPUT_ROOT:-outputs}/results/my_suite
    log_dir: ${PATCHFORMER_OUTPUT_ROOT:-outputs}/runs/my_suite/{run_name}
    save_path: ${PATCHFORMER_CHECKPOINT_ROOT:-outputs/checkpoints}/my_suite/{run_name}.pt

experiments:
  - name: my_run
    overrides:
      model:
        d_model: 512
```

The runner merges `extends` first, then the current file. Section keys such as `model`, `data`, `train`, `eval`, and `run` are flattened into `main.py` CLI flags.

Each run writes:

- `outputs/results/.../<run_name>/config.yaml`
- `outputs/results/.../<run_name>/stdout.log`
- `outputs/results/.../<run_name>/summary.json`
- `outputs/results/.../registry.csv`
