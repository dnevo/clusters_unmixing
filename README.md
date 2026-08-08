# clusters_unmixing

A compact spectral unmixing experiment framework for comparing preprocessing choices and unmixing models on configurable cluster sets.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnevo/clusters_unmixing/blob/main/notebooks/experiment_review.ipynb)

## What the project does

The project runs configured experiment batches that:

- load one or more cluster CSV files
- generate synthetic abundance vectors and mixed pixels
- apply wavelength selection and normalization
- evaluate registered unmixing models
- write summary CSV outputs for metrics and abundance previews
- review those outputs in a notebook-friendly format

The default experiment configuration lives in `experiments/configs/configuration.yaml`.

## Main entry points

- `main.py`: runs the configured experiment batch from the command line
- `notebooks/experiment_review.ipynb`: interactive notebook for reviewing experiment outputs, spectra, metrics, abundance tables, and synthetic pixel previews
- `notebooks/_notebook_smoke.py`: scriptable smoke test for the notebook flow

## Core modules

- `config/schema.py`: validated experiment configuration models and config loading
- `data/synthetic.py`: synthetic abundance generation
- `dataio.py`: cluster CSV loading
- `preprocessing/band_selection.py`: wavelength selection
- `preprocessing/normalization.py`: normalization helpers
- `core_math.py`: cosine similarity computation, SNR noise application, and shared numeric helpers
- `pipelines/experiment_pipeline.py`: experiment execution pipeline exposed as `run_experiments`
- `utils/notebook_diagnostics.py`: notebook orchestration, tables, and plotting helpers
- `models/runner_registry.py`: model registry and dispatch
- `models/sunsal.py`, `models/vpgdu.py`, `models/mlm.py`, `models/mlp.py`, `models/convnext1d.py`, `models/kan.py`, `models/mamba.py`: unmixing model implementations

## Run the project

Install the core package in editable mode:

```bash
python -m pip install -e .
```

Install notebook-only dependencies if you want to use the notebook or notebook smoke path:

```bash
python -m pip install ipython plotly
```

Run the configured experiments:

```bash
python main.py
```

Run the notebook smoke path:

```bash
python notebooks/_notebook_smoke.py
```

Open `notebooks/experiment_review.ipynb` for interactive review.

If your editor reports unresolved imports for `plotly` or `IPython` in the notebook helpers, that usually means the notebook-only dependencies have not been installed into the active interpreter yet.

## Configuration model

`ExperimentConfig.from_config_file(project_root)` loads:

- `experiments/configs/configuration.yaml`

Key sections in the config:

- `experiment_name`: output folder name under `experiments/outputs/`
- `extra_reproducibility`: opt-in bit-exact CUDA determinism (performance cost; auto-skipped for runs including `mamba`)
- `use_wandb`: opt-in Weights & Biases logging - one run per model call plus a batch summary run with the output CSVs as an artifact; requires `pip install -e .[wandb]` and a logged-in wandb account
- `cluster_sets`: available input cluster CSV files
- `models`: model hyperparameters keyed by model name
- `runs`: concrete experiment runs including bands, normalization, abundance distribution,
  noise level, pixel count, and selected models

Synthetic abundances are configured per run with:

```yaml
abundance_distribution: grid       # grid | dirichlet
dirichlet_alpha: 0.3               # used for dirichlet
```

`grid` is the legacy sparse generator: it selects active endmembers and assigns
abundances in discrete 0.04 increments, including pure endmembers. `dirichlet`
draws continuous abundances for all endmembers. Its `dirichlet_alpha` controls
the shape: values below 1 produce sparse-looking mixtures, 1 is uniform over
the continuous simplex, and values above 1 favor more balanced mixtures.
Dirichlet samples do not contain exact zeros or automatically inserted pure
pixels.

Relative paths in the config are resolved from the supplied project root.

## Outputs

Each run batch writes results under:

- `experiments/outputs/{experiment_name}/endmember_separability_summary.csv`
- `experiments/outputs/{experiment_name}/model_summary.csv`
- `experiments/outputs/{experiment_name}/abundance_preview.csv`

At a high level:

- `endmember_separability_summary.csv` stores per-run, per-stage cosine similarity statistics (`raw`, `normalized`)
- `model_summary.csv` stores one row per run and metric, with one column per configured model
- `abundance_preview.csv` stores the notebook-ready abundance preview table with `pixel_index`, `source`, error columns, and `endmember_*` values

## Notes

- The experiment pipeline uses NumPy for synthetic sample generation and PyTorch for model execution.
- The notebook helpers additionally rely on optional notebook/visualization packages such as `IPython` and `plotly`.
- The base project metadata keeps notebook dependencies separate from the core experiment pipeline install path.
- The output root is fixed in code to `experiments/outputs`.
