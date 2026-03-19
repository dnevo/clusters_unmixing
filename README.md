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
- `utils/diagnostics.py`: plotting and table helpers
- `utils/notebook_diagnostics.py`: notebook orchestration
- `models/sunsal.py`, `models/vpgdu.py`: model implementations

## Entry points
- `main.py`: run the configured experiment from the command line
- `notebooks/00_cluster_diagnostics.ipynb`: inspect results interactively
- `experiments/configs/correlation_options.yaml`: default experiment config
