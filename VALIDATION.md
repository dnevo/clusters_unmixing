# Validation report

## Summary
- Package metadata is present in `pyproject.toml`.
- The command-line entrypoint in `main.py` now loads `experiments/configs/configuration.yaml`.
- The notebook smoke script and notebook helpers are aligned with the current flattened module layout and render previews for all configured models present in the run outputs.
- `src/clusters_unmixing/models/sunsal.py`, `src/clusters_unmixing/models/vpgdu.py`, and `src/clusters_unmixing/models/small_mlp.py` remain unchanged.

## Validation commands
```bash
python -m pip install -e .
python main.py
python notebooks/_notebook_smoke.py
```

## Notes
- The repository currently has no `tests/` directory, so `pytest` is not part of the current validation story.
- Recent refactors flattened `dataio/clusters.py` into `dataio.py` and `metrics/correlation.py` into `core_math.py`.
- Project documentation was updated to reflect the current structure and entry points.

## Source footprint
- Python files counted: 21
- Total Python lines: 1894
