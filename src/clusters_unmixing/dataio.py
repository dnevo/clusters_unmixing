from __future__ import annotations
from os import PathLike
import numpy as np

def load_wavelength_and_cluster_matrix(csv_path: PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    columns = list(data.dtype.names or [])
    wavelength_col = columns[0]
    cluster_cols = columns[1:]
    wavelength_axis = np.asarray(data[wavelength_col], dtype=float)
    endmembers = np.vstack([np.asarray(data[col], dtype=float) for col in cluster_cols])

    return wavelength_axis, endmembers
