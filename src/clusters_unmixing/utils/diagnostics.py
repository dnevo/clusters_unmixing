from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display
from clusters_unmixing.config import ExperimentConfig


def _resolve_abundance_columns(abundance_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    true_cols = [c for c in abundance_df.columns if c.startswith('true_a') or c.startswith('true_abundance_')]
    pred_cols = [c for c in abundance_df.columns if c.startswith('est_a') or c.startswith('pred_abundance_')]
    true_cols = sorted(true_cols, key=lambda c: int(''.join(ch for ch in c if ch.isdigit()) or 0))
    pred_cols = sorted(pred_cols, key=lambda c: int(''.join(ch for ch in c if ch.isdigit()) or 0))
    return true_cols, pred_cols


def build_abundance_comparison_tables(abundance_df: pd.DataFrame, max_pixels: int = 5) -> list[dict[str, Any]]:
    true_cols, pred_cols = _resolve_abundance_columns(abundance_df)

    pixel_ids = sorted(int(v) for v in abundance_df['pixel_index'].dropna().unique())[:max_pixels]
    tables: list[dict[str, Any]] = []
    for pixel_index in pixel_ids:
        pixel_rows = abundance_df[abundance_df['pixel_index'].astype(int) == int(pixel_index)].copy()
        if pixel_rows.empty:
            continue
        true_vals = [float(pixel_rows.iloc[0][c]) for c in true_cols]
        rows: list[dict[str, Any]] = [{
            'source': 'true',
            'abundance_rmse_vs_true': 0.0,
            **{f'endmember_{j}': value for j, value in enumerate(true_vals, start=1)},
        }]
        for _, row in pixel_rows.sort_values('model').iterrows():
            pred_vals = [float(row[c]) for c in pred_cols]
            true_arr = pd.Series(true_vals, dtype=float).to_numpy()
            pred_arr = pd.Series(pred_vals, dtype=float).to_numpy()
            rmse = float(((pred_arr - true_arr) ** 2).mean() ** 0.5)
            rows.append({
                'source': str(row['model']),
                'abundance_rmse_vs_true': rmse,
                **{f'endmember_{j}': value for j, value in enumerate(pred_vals, start=1)},
            })
        tables.append({
            'pixel_index': pixel_index,
            'table': pd.DataFrame(rows).round(6),
        })
    return tables


def display_abundance_comparison_tables(abundance_df: pd.DataFrame, max_pixels: int = 5) -> None:
    tables = build_abundance_comparison_tables(abundance_df=abundance_df, max_pixels=max_pixels)
    for item in tables:
        display(Markdown(f"### Abundances  |  pixel_index={item['pixel_index']}"))
        display(item['table'])

        
def plot_cluster_overview(
    wavelengths: Any,
    signatures: Any,
    title: str,
    bands_ranges: list[Any] | None = None,
    y_title: str = "Reflectance",
) -> go.Figure:
    """Cluster-level spectra overview plot with styling by band-range type.

    Styles:
    - ranges in config with reduce="none" -> solid line
    - ranges in config with reduce="mean" -> dashed line + diamond marker at segment midpoint
    - wavelengths outside configured ranges -> dotted line
    """
    wavelengths_arr = pd.Series(wavelengths, dtype=float).to_numpy()
    signatures_arr = pd.DataFrame(signatures).to_numpy(dtype=float)
    if signatures_arr.ndim != 2:
        raise ValueError("signatures must be a 2D array of shape (bands, clusters)")
    if signatures_arr.shape[0] != len(wavelengths_arr):
        raise ValueError(
            f"wavelength/signature length mismatch: {len(wavelengths_arr)} vs {signatures_arr.shape[0]}"
        )

    def _normalize_ranges(raw_ranges: list[Any] | None) -> list[tuple[float, float, str]]:
        if not raw_ranges:
            return []
        normalized: list[tuple[float, float, str]] = []
        for item in raw_ranges:
            if isinstance(item, dict):
                range_vals = item.get("range_µm") or item.get("range_um")
                if range_vals is None or len(range_vals) != 2:
                    raise ValueError("bands_ranges object entries must contain range_µm=[x_min, x_max]")
                reduce = str(item.get("reduce", "none")).strip().lower()
                normalized.append((float(range_vals[0]), float(range_vals[1]), reduce))
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                normalized.append((float(item[0]), float(item[1]), str(item[2]).strip().lower()))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                normalized.append((float(item[0]), float(item[1]), "none"))
            else:
                raise ValueError("bands_ranges entries must be [x_min, x_max], [x_min, x_max, reduce], or object form")
        return normalized

    def _segment_slices(mask: np.ndarray) -> list[slice]:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        starts = [int(idx[0])]
        stops: list[int] = []
        for prev, cur in zip(idx[:-1], idx[1:]):
            if int(cur) != int(prev) + 1:
                stops.append(int(prev) + 1)
                starts.append(int(cur))
        stops.append(int(idx[-1]) + 1)
        return [slice(start, stop) for start, stop in zip(starts, stops)]

    normalized_ranges = _normalize_ranges(bands_ranges)
    point_kind = np.full(len(wavelengths_arr), "outside", dtype=object)
    for x_min, x_max, reduce in normalized_ranges:
        mask = (wavelengths_arr >= float(x_min)) & (wavelengths_arr <= float(x_max))
        point_kind[mask] = "mean" if reduce == "mean" else "none"

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]

    fig = go.Figure()
    for idx in range(signatures_arr.shape[1]):
        cluster_name = f"Cluster {idx + 1}"
        cluster_color = palette[idx % len(palette)]
        y = signatures_arr[:, idx]
        legend_shown = False

        for kind, dash_style in [("none", "solid"), ("mean", "dash"), ("outside", "dot")]:
            kind_mask = point_kind == kind
            for seg in _segment_slices(kind_mask):
                x_seg = wavelengths_arr[seg]
                y_seg = y[seg]
                if len(x_seg) == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode="lines",
                    name=cluster_name,
                    legendgroup=cluster_name,
                    showlegend=not legend_shown,
                    line={"color": cluster_color, "dash": dash_style},
                    hovertemplate="cluster=%{fullData.name}<br>wavelength=%{x:.3f} µm<br>value=%{y:.4f}<extra></extra>",
                ))
                legend_shown = True

                if kind == "mean":
                    mid_x = float((x_seg[0] + x_seg[-1]) / 2.0)
                    marker_idx = int(np.abs(x_seg - mid_x).argmin())
                    fig.add_trace(go.Scatter(
                        x=[float(x_seg[marker_idx])],
                        y=[float(y_seg[marker_idx])],
                        mode="markers",
                        marker={"symbol": "diamond", "size": 10, "color": cluster_color},
                        name=cluster_name,
                        legendgroup=cluster_name,
                        showlegend=False,
                        hovertemplate="cluster=%{fullData.name}<br>wavelength=%{x:.3f} µm<br>value=%{y:.4f}<extra></extra>",
                    ))

        if not legend_shown:
            fig.add_trace(go.Scatter(
                x=wavelengths_arr,
                y=y,
                mode="lines",
                name=cluster_name,
                legendgroup=cluster_name,
                showlegend=True,
                line={"color": cluster_color},
                hovertemplate="cluster=%{fullData.name}<br>wavelength=%{x:.3f} µm<br>value=%{y:.4f}<extra></extra>",
            ))

    fig.update_layout(
        title=title,
        xaxis_title="Wavelength (µm)",
        yaxis_title=y_title,
        height=480,
    )
    return fig
