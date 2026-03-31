from __future__ import annotations

import numpy as np


def quadratic_normalize(spectra: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
    """Subtract the fixed quadratic baseline used by the project notebook/pipeline."""
    spectra_arr = np.asarray(spectra, dtype=float)
    wavelengths_arr = np.asarray(wavelengths, dtype=float)
    q_values = -0.20 * wavelengths_arr**2 + 0.68 * wavelengths_arr - 0.12
    return spectra_arr - q_values[None, :]


def apply_normalization(spectra: np.ndarray, wavelengths: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == 'without':
        return spectra
    if normalization == 'with_quadratic':
        return quadratic_normalize(spectra, wavelengths)
    raise ValueError(f'Unsupported normalization mode: {normalization}')