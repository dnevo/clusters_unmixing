from __future__ import annotations

import numpy as np


def quadratic_normalize(signatures: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Subtract the fixed quadratic baseline used by the project notebook/pipeline."""
    signatures_arr = np.asarray(signatures, dtype=float)
    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    q_values = -0.20 * wavelengths_arr**2 + 0.68 * wavelengths_arr - 0.12
    return signatures_arr - q_values[:, None]


def apply_normalization(signatures: np.ndarray, wavelengths: np.ndarray, normalization: str) -> np.ndarray:
    mode = str(normalization).strip().lower()
    if mode == 'without':
        return np.asarray(signatures, dtype=float)
    if mode == 'with_quadratic':
        return quadratic_normalize(signatures=signatures, wavelengths=wavelengths)
    raise ValueError(f'Unsupported normalization mode: {normalization}')
