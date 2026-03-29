# clusters_unmixing

A compact spectral unmixing experiment framework for comparing preprocessing choices and unmixing models on configurable cluster sets.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnevo/clusters_unmixing/blob/main/notebooks/00_clusters_unmixing_experiments.ipynb)

## What the project does

The project runs configured experiment batches that:

- load one or more cluster CSV files
- generate synthetic abundance vectors and mixed pixels
- apply wavelength selection, normalization, and optional transform steps
- evaluate registered unmixing models
- write summary CSV outputs for metrics and abundance previews
- review those outputs in a notebook-friendly format

The default experiment configuration lives in `experiments/configs/configuration.yaml`.

## Main entry points

- `main.py`: runs the configured experiment batch from the command line
- `notebooks/00_clusters_unmixing_experiments.ipynb`: interactive notebook for reviewing experiment outputs, spectra, metrics, abundance tables, and synthetic pixel previews
- `notebooks/_notebook_smoke.py`: scriptable smoke test for the notebook flow

## Core modules

- `config/schema.py`: validated experiment configuration models and config loading
- `data/synthetic.py`: synthetic abundance generation
- `dataio.py`: cluster CSV loading
- `transforms/spectral_views.py`: wavelength selection and transform steps
- `transforms/normalization.py`: normalization helpers
- `metrics.py`: correlation metric computation and summary helpers
- `pipelines/experiment_pipeline.py`: experiment execution pipeline exposed as `run_experiments`
- `utils/notebook_diagnostics.py`: notebook orchestration, tables, and plotting helpers
- `models/runner_registry.py`: model registry and dispatch
- `models/sunsal.py`, `models/vpgdu.py`, `models/small_mlp.py`: unmixing model implementations

## Run the project

Install the package in editable mode:

```bash
python -m pip install -e .
```

Run the configured experiments:

```bash
python main.py
```

Run the notebook smoke path:

```bash
python notebooks/_notebook_smoke.py
```

Open `notebooks/00_clusters_unmixing_experiments.ipynb` for interactive review.

## Configuration model

`ExperimentConfig.from_config_file(project_root)` loads:

- `experiments/configs/configuration.yaml`

Key sections in the config:

- `experiment_name`: output folder name under `experiments/outputs/`
- `cluster_sets`: available input cluster CSV files
- `metrics`: required non-empty list of correlation metrics to compute for projected endmembers (`cosine`, `sam`)
- `model_evaluation.models`: model hyperparameters keyed by model name
- `model_evaluation.runs`: concrete experiment runs including bands, normalization, transforms, noise level, pixel count, and selected models

Relative paths in the config are resolved from the supplied project root.

## Outputs

Each run batch writes results under:

- `experiments/outputs/{experiment_name}/correlation_summary.csv`
- `experiments/outputs/{experiment_name}/model_summary.csv`
- `experiments/outputs/{experiment_name}/abundance_preview.csv`

At a high level:

- `correlation_summary.csv` stores per-run, per-metric, per-stage correlation statistics (`raw`, `normalized`, and any configured transform stages such as `pca`)
- `model_summary.csv` stores one row per run and metric, with one column per configured model
- `abundance_preview.csv` stores the notebook-ready abundance preview table with `pixel_index`, `source`, error columns, and `endmember_*` values

## Notes

- The experiment pipeline uses NumPy for synthetic sample generation and PyTorch for model execution.
- The notebook helpers additionally rely on notebook/visualization packages such as `IPython` and `plotly`.
- The output root is fixed in code to `experiments/outputs`.
