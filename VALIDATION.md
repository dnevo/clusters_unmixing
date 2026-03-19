# Validation report

## Summary
- Package metadata is present in `pyproject.toml`.
- The command-line entrypoint in `main.py` now loads `experiments/configs/correlation_options.yaml`.
- The notebook smoke script and notebook helpers are aligned with the current flattened module layout.
- `src/clusters_unmixing/models/sunsal.py` and `src/clusters_unmixing/models/vpgdu.py` remain unchanged.

## Validation commands
```bash
python -m pip install -e .
python main.py
python notebooks/_notebook_smoke.py
```

## Notes
- The repository currently has no `tests/` directory, so `pytest` is not part of the current validation story.
- Recent refactors flattened `dataio/clusters.py` into `dataio.py` and `metrics/correlation.py` into `metrics.py`.
- Project documentation was updated to reflect the current structure and entry points.

## Source footprint
- Python files counted: 21
- Total Python lines: 1457
