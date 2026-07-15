from __future__ import annotations

import numpy as np


def rmse(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values) - np.asarray(reference)))))


def mix_pixels(abundances: np.ndarray, endmembers: np.ndarray, gamma: float = 0.0) -> np.ndarray:
    """Combine abundances and endmembers into pixel spectra under the Generalized Bilinear Model.

    ``gamma=0.0`` is the standard linear mixing model (``abundances @ endmembers``).
    ``gamma=1.0`` is the Fan model: every pair of endmembers ``(i, j)`` contributes an
    extra ``a_i * a_j * (m_i ⊙ m_j)`` term (elementwise product of the two endmember
    spectra), approximating the multiple-scattering ("double bounce") interaction between
    two distinct materials touching at sub-pixel scale. Values in between interpolate
    linearly between the two regimes.

    Parameters
    ----------
    abundances : (n_pixels, n_endmembers)
    endmembers : (n_endmembers, n_bands)
    gamma : nonlinearity strength in [0, 1].
    """
    linear = abundances @ endmembers
    if gamma == 0.0:
        return linear

    n_endmembers = endmembers.shape[0]
    nonlinear = np.zeros_like(linear)
    for i in range(n_endmembers):
        for j in range(i + 1, n_endmembers):
            pair_spectrum = endmembers[i] * endmembers[j]
            pair_weight = abundances[:, i] * abundances[:, j]
            nonlinear += pair_weight[:, None] * pair_spectrum[None, :]

    return linear + gamma * nonlinear


def apply_snr_noise(
    clean_pixels: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Centralized SNR application logic for both pipelines and diagnostics."""
    if np.isinf(snr_db):
        return clean_pixels, np.zeros_like(clean_pixels)

    signal_power = float((clean_pixels ** 2).mean())
    signal_rms = float(np.sqrt(max(signal_power, 1e-12)))
    noise_std = signal_rms * float(10.0 ** (-snr_db / 20.0))
    noise = np.random.normal(loc=0.0, scale=noise_std, size=clean_pixels.shape)
    return clean_pixels + noise, noise


def _cosine_similarity_matrix(endmembers: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(endmembers, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = endmembers / norms
    return normalized @ normalized.T


def compute_correlation_matrix(endmembers: np.ndarray, metric: str = "cosine") -> np.ndarray:
    metric = metric.strip().lower()
    if metric == "cosine":
        return _cosine_similarity_matrix(endmembers)
    if metric == "sam":
        cosine = _cosine_similarity_matrix(endmembers)
        return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    raise ValueError(f"Unsupported correlation metric: {metric}")


def summarize_correlation_matrix(matrix: np.ndarray) -> dict[str, float]:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix[mask]
    abs_off_diag = np.abs(off_diag)
    return {
        "mean_abs_offdiag": float(abs_off_diag.mean()),
        "max_abs_offdiag": float(abs_off_diag.max()),
        "min_offdiag": float(off_diag.min()),
        "max_offdiag": float(off_diag.max()),
    }
