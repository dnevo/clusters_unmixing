# Validation report

## Result
- Editable install: passed
- Test suite: 11 passed
- Notebook added: `notebooks/00_cluster_diagnostics.ipynb`
- Notebook smoke validation: passed via extracted Python execution of notebook code cells

## Commands
```bash
python -m pip install -e .
pytest -q
python notebooks/_notebook_smoke.py
```

## Notes
- `src/clusters_unmixing/models/sunsal.py` unchanged
- `src/clusters_unmixing/models/vpgdu.py` unchanged
- The new notebook covers the same practical scope as the original one: run experiments, inspect raw/normalized/transformed spectra, compare correlation statistics, review model metrics, and preview abundances / synthetic pixels.
- Full Jupyter kernel execution was not persisted as an executed notebook artifact in this environment, but the notebook code itself was syntax-checked and smoke-tested successfully by running the extracted code cells as a script.

## Source footprint
- Python files counted (src + tests): 26
- Total lines (src + tests): 1880

