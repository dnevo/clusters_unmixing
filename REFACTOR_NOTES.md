# Refactor notes

## What changed
- Replaced the large dataclass-based config layer with a much smaller validated schema layer.
- Collapsed preprocessing and experiment execution into one pipeline module.
- Kept only one set of spectral transform utilities.
- Replaced heavyweight notebook-oriented diagnostics with minimal helpers needed for experiment review.
- Removed coverage output, caches, editor settings, and other generated artifacts from the deliverable.

## Intentionally unchanged
- `src/clusters_unmixing/models/sunsal.py`
- `src/clusters_unmixing/models/vpgdu.py`

## Core structure
- `config/schema.py`
- `dataio/clusters.py`
- `transforms/spectral_views.py`
- `metrics/correlation.py`
- `pipelines/correlation_pipeline.py`
- `utils/diagnostics.py`

