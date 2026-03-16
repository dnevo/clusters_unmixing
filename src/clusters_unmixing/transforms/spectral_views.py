from __future__ import annotations

from typing import Any

import numpy as np


def first_derivative(signatures: np.ndarray) -> np.ndarray:
    return np.gradient(signatures, axis=0)


def pca_reduce(signatures: np.ndarray, n_components: int) -> np.ndarray:
    if isinstance(n_components, bool) or not isinstance(n_components, int):
        raise ValueError("PCA 'n_components' must be an integer")
    if n_components <= 0:
        raise ValueError("PCA 'n_components' must be > 0")
    n_bands, _ = signatures.shape
    if n_components > n_bands:
        raise ValueError("PCA 'n_components' cannot exceed number of bands")
    centered = signatures - signatures.mean(axis=1, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    basis = u[:, :n_components]
    return basis.T @ centered


def apply_transform(signatures: np.ndarray, kind: str, params: dict[str, Any] | None = None) -> np.ndarray:
    params = {} if params is None else dict(params)
    kind = kind.strip().lower()
    if kind == "first_derivative":
        return first_derivative(signatures)
    if kind == "pca":
        return pca_reduce(signatures, n_components=params.get("n_components"))
    raise ValueError(f"Unsupported transform kind: {kind}")


def _normalize_bands_ranges(bands_ranges: list[Any]) -> list[tuple[float, float, str]]:
    normalized = []
    for item in bands_ranges:
        if isinstance(item, dict):
            range_value = item["range_µm"]
            reduce = str(item.get("reduce", "none")).strip().lower()
        else:
            range_value = item
            reduce = "none"
        if reduce not in {"none", "mean"}:
            raise ValueError("Bands range 'reduce' must be one of: none, mean")
        x_min, x_max = float(range_value[0]), float(range_value[1])
        if x_min > x_max:
            raise ValueError("Bands ranges require x_min <= x_max")
        normalized.append((x_min, x_max, reduce))
    return normalized


def select_wavelength_ranges(wavelengths: np.ndarray, signatures: np.ndarray, bands_ranges: list[Any]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    pieces_w = []
    pieces_s = []
    segment_lengths = []
    for x_min, x_max, reduce in _normalize_bands_ranges(bands_ranges):
        mask = (wavelengths >= x_min) & (wavelengths <= x_max)
        w = wavelengths[mask]
        s = signatures[mask]
        if w.size == 0:
            raise ValueError(f"No wavelengths found in range [{x_min}, {x_max}]")
        if reduce == "mean":
            pieces_w.append(np.asarray([w.mean()], dtype=float))
            pieces_s.append(np.asarray([s.mean(axis=0)], dtype=float))
            segment_lengths.append(1)
        else:
            pieces_w.append(w)
            pieces_s.append(s)
            segment_lengths.append(int(w.shape[0]))
    return np.concatenate(pieces_w), np.vstack(pieces_s), segment_lengths
