from __future__ import annotations

from typing import Any

import numpy as np

from clusters_unmixing.config.schema import BandRangeSpec


def first_derivative(endmembers: np.ndarray) -> np.ndarray:
    return np.gradient(endmembers, axis=1)


def pca_reduce(
    endmembers: np.ndarray,
    pixels: np.ndarray,
    n_components: int,
    center: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce spectral dimensionality using SVD (PCA-equivalent).

    Args:
        endmembers:   (n_members, bands)  e.g. (6, 802)
        pixels:       (n_pixels, bands)   e.g. (256, 802)
        n_components: number of components to keep
        center:       mean-center before SVD (default False - preserves
                      simplex geometry for unmixing)
    Returns:
        endmembers_t: (n_members, n_components)
        pixels_t:     (n_pixels, n_components)
    """
    if n_components > pixels.shape[1]:
        raise ValueError("n_components cannot exceed number of bands")

    mean = pixels.mean(axis=0, keepdims=True) if center else 0
    x_c = pixels - mean

    _, s, vt = np.linalg.svd(x_c, full_matrices=False)
    basis = vt[:n_components].T

    pixels_t = x_c @ basis
    endmembers_t = (endmembers - mean) @ basis

    var = (s[:n_components] ** 2 / (s ** 2).sum()).sum()
    print(f"[SVD] {n_components} components -> {var:.1%} variance explained")

    return endmembers_t, pixels_t


def apply_transform(endmembers: np.ndarray, pixels: np.ndarray, kind: str, params: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    params = {} if params is None else dict(params)
    kind = kind.strip().lower()
    if kind == "first_derivative":
        return first_derivative(endmembers), first_derivative(pixels)
    if kind == "pca":
        return pca_reduce(endmembers, pixels, n_components=int(params["n_components"]))
    raise ValueError(f"Unsupported transform kind: {kind}")


def select_wavelength_ranges(wavelengths: np.ndarray, endmembers: np.ndarray, bands_ranges: list[BandRangeSpec]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    pieces_w = []
    pieces_s = []
    segment_lengths = []
    for x_min, x_max, reduce in bands_ranges:
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
