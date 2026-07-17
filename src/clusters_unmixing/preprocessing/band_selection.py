from __future__ import annotations

import numpy as np

from clusters_unmixing.config.schema import BandRangeSpec


def select_wavelength_ranges(wavelengths: np.ndarray, spectra: np.ndarray, band_ranges: list[BandRangeSpec]) -> tuple[np.ndarray, np.ndarray]:
    pieces_w = []
    pieces_s = []
    for x_min, x_max, reduce in band_ranges:
        mask = (wavelengths >= x_min) & (wavelengths <= x_max)
        w = wavelengths[mask]
        s = spectra[:, mask]
        if w.size == 0:
            raise ValueError(f"No wavelengths found in range [{x_min}, {x_max}]")
        if reduce == "mean":
            pieces_w.append(np.asarray([w.mean()], dtype=float))
            pieces_s.append(np.asarray(s.mean(axis=1, keepdims=True), dtype=float))
        else:
            pieces_w.append(w)
            pieces_s.append(s)
    return np.concatenate(pieces_w), np.concatenate(pieces_s, axis=1)