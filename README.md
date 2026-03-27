# clusters_unmixing

A lean spectral unmixing experiment framework centered on one configurable experiment pipeline and one diagnostics notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnevo/clusters_unmixing/blob/main/notebooks/00_cluster_diagnostics.ipynb)

## Core modules
- `config/schema.py`: validated experiment config models
- `dataio.py`: cluster CSV loading
- `transforms/spectral_views.py`: wavelength selection and transforms
- `transforms/normalization.py`: normalization helpers
- `metrics.py`: cosine and SAM correlation metrics
- `pipelines/correlation_pipeline.py`: experiment runner
- `utils/notebook_diagnostics.py`: notebook diagnostics helpers, plotting, tables, and orchestration
- `models/sunsal.py`, `models/vpgdu.py`, `models/small_mlp.py`: model implementations

## Entry points
- `main.py`: run the configured experiment from the command line
- `notebooks/00_cluster_diagnostics.ipynb`: inspect results interactively, including per-model abundance tables and pixel preview plots
- `experiments/configs/configuration.yaml`: default experiment config

## Experiment config
- `experiment_name` controls the experiment output folder name under `experiments/outputs/`
- `cluster_sets` defines the available input cluster CSV files
- `metrics` selects which correlation metrics to compute
- `model_evaluation.models` configures model hyperparameters
- `model_evaluation.runs` defines the evaluation runs to execute

Relative paths in the config are resolved from the project root supplied by the caller when loading the config.

## Output layout
Results are written to:

- `experiments/outputs/{experiment_name}/correlation_summary.csv`
- `experiments/outputs/{experiment_name}/model_summary.csv`
- `experiments/outputs/{experiment_name}/abundance_preview.csv`

The output root is fixed in code to `experiments/outputs`; it is no longer configured in YAML.
