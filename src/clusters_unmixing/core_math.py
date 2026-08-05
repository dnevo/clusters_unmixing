from __future__ import annotations

import numpy as np


def rmse(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values) - np.asarray(reference)))))


def mix_pixels(abundances: np.ndarray, endmembers: np.ndarray, gamma: float = 0.0) -> np.ndarray:
    """Combine abundances and endmembers into pixel spectra under the Generalized Bilinear Model.

    ``gamma=0.0`` is the standard linear mixing model (``abundances @ endmembers``).
    For ``gamma > 0``, every pair of endmembers ``(i, j)`` contributes an extra
    ``gamma_ij * a_i * a_j * (m_i ⊙ m_j)`` term (elementwise product of the two
    endmember spectra), approximating the multiple-scattering ("double bounce")
    interaction between two distinct materials touching at sub-pixel scale. Each
    pixel draws its own ``gamma_ij`` independently from ``Uniform(0, gamma)`` for
    every pair, matching the per-pixel-per-pair granularity that this project's
    GBM inversion (``models/gbm.py``) fits. For chemically graded soil cluster
    sets (e.g. 6clusters_thomas), treat gamma as a numerical robustness knob
    rather than a physically motivated simulation, since the endmembers aren't
    optically distinct materials.

    Parameters
    ----------
    abundances : (n_pixels, n_endmembers)
    endmembers : (n_endmembers, n_bands)
    gamma : upper bound in [0, 1] for the per-pixel, per-pair ``Uniform(0, gamma)``
        coefficients; 0.0 disables the nonlinear term entirely.
    """
    linear = abundances @ endmembers
    if gamma == 0.0:
        return linear

    n_pixels = abundances.shape[0]
    n_endmembers = endmembers.shape[0]
    pair_indices = [(i, j) for i in range(n_endmembers) for j in range(i + 1, n_endmembers)]
    gammas = np.random.uniform(0.0, gamma, size=(n_pixels, len(pair_indices)))

    nonlinear = np.zeros_like(linear)
    for k, (i, j) in enumerate(pair_indices):
        pair_spectrum = endmembers[i] * endmembers[j]
        pair_weight = abundances[:, i] * abundances[:, j]
        nonlinear += (gammas[:, k] * pair_weight)[:, None] * pair_spectrum[None, :]

    return linear + nonlinear


def apply_snr_noise(
    clean_pixels: np.ndarray,
    snr_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Centralized SNR application logic for both pipelines and diagnostics.

    snr_db must be an actual float (use .inf, not YAML's bare inf, in configs -
    see configuration.yaml). A stray string here raises a confusing TypeError
    from np.isinf below rather than a clear validation error.
    """
    if np.isinf(snr_db):
        return clean_pixels, np.zeros_like(clean_pixels)

    signal_power = float((clean_pixels ** 2).mean())
    signal_rms = float(np.sqrt(max(signal_power, 1e-12)))
    noise_std = signal_rms * float(10.0 ** (-snr_db / 20.0))
    noise = np.random.normal(loc=0.0, scale=noise_std, size=clean_pixels.shape)
    return clean_pixels + noise, noise


def compute_cosine_similarity_matrix(endmembers: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(endmembers, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = endmembers / norms
    return normalized @ normalized.T


def summarize_cosine_similarity_matrix(matrix: np.ndarray) -> dict[str, float]:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    off_diag = matrix[mask]
    abs_off_diag = np.abs(off_diag)
    return {
        "mean_abs_offdiag": float(abs_off_diag.mean()),
        "max_abs_offdiag": float(abs_off_diag.max()),
        "min_offdiag": float(off_diag.min()),
        "max_offdiag": float(off_diag.max()),
    }


def compute_condition_number(endmembers: np.ndarray) -> float:
    return float(np.linalg.cond(endmembers))
