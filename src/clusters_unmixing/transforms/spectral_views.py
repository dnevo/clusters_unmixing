from __future__ import annotations

from typing import Any

import numpy as np


def first_derivative(endmembers: np.ndarray) -> np.ndarray:
    return np.gradient(endmembers, axis=1)


def pca_reduce(endmembers: np.ndarray, n_components: int) -> np.ndarray:
    n_members, n_bands = endmembers.shape
    if n_components > n_bands:
        raise ValueError("PCA 'n_components' cannot exceed number of endmembers")
    centered = endmembers - endmembers.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:n_components].T
    return centered @ basis


def apply_transform(endmembers: np.ndarray, pixels: np.ndarray, kind: str, params: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    params = {} if params is None else dict(params)
    kind = kind.strip().lower()
    if kind == "first_derivative":
        return first_derivative(endmembers), first_derivative(pixels)
    if kind == "pca":
        transformed = pca_reduce(np.concatenate((endmembers, pixels), axis=0), n_components=params.get("n_components"))
        return transformed[:endmembers.shape[0]], transformed[endmembers.shape[0]:]
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


def select_wavelength_ranges(wavelengths: np.ndarray, endmembers: np.ndarray, bands_ranges: list[Any]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    pieces_w = []
    pieces_s = []
    segment_lengths = []
    for x_min, x_max, reduce in _normalize_bands_ranges(bands_ranges):
        mask = (wavelengths >= x_min) & (wavelengths <= x_max)
        w = wavelengths[mask]
        s = endmembers[:, mask]
        if w.size == 0:
            raise ValueError(f"No wavelengths found in range [{x_min}, {x_max}]")
        if reduce == "mean":
            pieces_w.append(np.asarray([w.mean()], dtype=float))
            pieces_s.append(np.asarray(s.mean(axis=1, keepdims=True), dtype=float))
            segment_lengths.append(1)
        else:
            pieces_w.append(w)
            pieces_s.append(s)
            segment_lengths.append(int(w.shape[0]))
    return np.concatenate(pieces_w), np.concatenate(pieces_s, axis=1), segment_lengths


