from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np


def load_wavelength_and_cluster_matrix(csv_path: str | PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    csv_path = Path(csv_path)
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8-sig")
    columns = list(data.dtype.names or [])
    if not columns:
        raise ValueError(f"Failed to load columns from {csv_path}")
    wavelength_col = columns[0]
    cluster_cols = [col for col in columns[1:] if str(col).lower().startswith("cluster")]
    if not cluster_cols:
        cluster_cols = columns[1:]
    wavelengths = np.asarray(data[wavelength_col], dtype=float)
    clusters = np.column_stack([np.asarray(data[col], dtype=float) for col in cluster_cols])
    return wavelengths, clusters
