# clusters_unmixing (lean refactor)

A slimmed-down spectral unmixing experiment framework.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dnevo/clusters_unmixing/blob/main/notebooks/00_cluster_diagnostics.ipynb)

Core modules:
- `config/schema.py`: validated config models
- `dataio/clusters.py`: CSV loading
- `transforms/spectral_views.py`: wavelength selection and transforms
- `metrics/correlation.py`: cosine and SAM
- `pipelines/correlation_pipeline.py`: one experiment runner
- `utils/diagnostics.py`: lightweight diagnostics helpers
- `models/sunsal.py`, `models/vpgdu.py`: unchanged from the original project

