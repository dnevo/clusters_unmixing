from __future__ import annotations

import numpy as np


def _cosine_similarity_matrix(signatures: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(signatures, axis=0, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normalized = signatures / norms
    return normalized.T @ normalized


def compute_correlation_matrix(signatures: np.ndarray, metric: str = "cosine") -> np.ndarray:
    metric = metric.strip().lower()
    if metric == "cosine":
        return _cosine_similarity_matrix(signatures)
    if metric == "sam":
        cosine = _cosine_similarity_matrix(signatures)
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
