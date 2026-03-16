# clusters_unmixing (lean refactor)

A slimmed-down spectral unmixing experiment framework.

Core modules:
- `config/schema.py`: validated config models
- `dataio/clusters.py`: CSV loading
- `transforms/spectral_views.py`: wavelength selection and transforms
- `metrics/correlation.py`: cosine and SAM
- `pipelines/correlation_pipeline.py`: one experiment runner
- `utils/diagnostics.py`: lightweight diagnostics helpers
- `models/sunsal.py`, `models/vpgdu.py`: unchanged from the original project

