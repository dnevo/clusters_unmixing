from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display

from clusters_unmixing.config import ExperimentConfig
from clusters_unmixing.config.schema import BandRangeSpec
from clusters_unmixing.metrics import compute_correlation_matrix, summarize_correlation_matrix
from clusters_unmixing.dataio import load_wavelength_and_cluster_matrix
from clusters_unmixing.pipelines import run_correlation_experiments
from clusters_unmixing.transforms import apply_normalization, apply_transform, select_wavelength_ranges
from clusters_unmixing.utils.run_helpers import resolve_cluster_path, bands_ranges_key


def _resolve_abundance_columns(abundance_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    true_cols = [c for c in abundance_df.columns if c.startswith('true_a')]
    pred_cols = [c for c in abundance_df.columns if c.startswith('est_a')]
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
    wavelength_axis: Any,
    endmembers: Any,
    title: str,
    bands_ranges: list[BandRangeSpec] | None = None,
    y_title: str = 'Reflectance',
) -> go.Figure:
    wavelength_axis_arr = np.asarray(wavelength_axis, dtype=float)
    endmembers_arr = np.asarray(endmembers, dtype=float)
    if endmembers_arr.ndim != 2:
        raise ValueError('endmembers must be a 2D array of shape (clusters, bands)')
    if endmembers_arr.shape[1] != len(wavelength_axis_arr):
        raise ValueError(
            f'wavelength/endmember length mismatch: {len(wavelength_axis_arr)} vs {endmembers_arr.shape[1]}'
        )

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

    point_kind = np.full(len(wavelength_axis_arr), 'outside', dtype=object)
    for x_min, x_max, reduce in bands_ranges or []:
        mask = (wavelength_axis_arr >= float(x_min)) & (wavelength_axis_arr <= float(x_max))
        point_kind[mask] = 'mean' if reduce == 'mean' else 'none'

    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c',
        '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    ]

    fig = go.Figure()
    for idx in range(endmembers_arr.shape[0]):
        cluster_name = f'Cluster {idx + 1}'
        cluster_color = palette[idx % len(palette)]
        y = endmembers_arr[idx]

        for kind, dash_style in [('none', 'solid'), ('mean', 'dash'), ('outside', 'dot')]:
            kind_mask = point_kind == kind
            for seg in _segment_slices(kind_mask):
                x_seg = wavelength_axis_arr[seg]
                y_seg = y[seg]
                if len(x_seg) == 0:
                    continue
                fig.add_trace(go.Scatter(
                    x=x_seg,
                    y=y_seg,
                    mode='lines',
                    name=cluster_name,
                    legendgroup=cluster_name,
                    showlegend=False,
                    line={'color': cluster_color, 'dash': dash_style},
                    hovertemplate='cluster=%{fullData.name}<br>wavelength=%{x:.3f} um<br>value=%{y:.4f}<extra></extra>',
                ))

                if kind == 'mean':
                    mid_x = float((x_seg[0] + x_seg[-1]) / 2.0)
                    marker_idx = int(np.abs(x_seg - mid_x).argmin())
                    fig.add_trace(go.Scatter(
                        x=[float(x_seg[marker_idx])],
                        y=[float(y_seg[marker_idx])],
                        mode='markers',
                        marker={'symbol': 'diamond', 'size': 10, 'color': cluster_color},
                        name=cluster_name,
                        legendgroup=cluster_name,
                        showlegend=False,
                        hovertemplate='cluster=%{fullData.name}<br>wavelength=%{x:.3f} um<br>value=%{y:.4f}<extra></extra>',
                    ))

    fig.update_layout(
        title=title,
        xaxis_title='Wavelength (µm)',
        yaxis_title=y_title,
        height=480,
    )
    return fig

def setup_notebook_imports(project_root: str | Path | None = None) -> Path:
    note_dir = Path.cwd()
    resolved_root = Path(project_root).resolve() if project_root is not None else (
        note_dir.parent if note_dir.name == 'notebooks' else note_dir
    ).resolve()
    src_dir = resolved_root / 'src'
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    pd.set_option('display.max_columns', 200)
    pd.set_option('display.width', 180)
    return resolved_root


def cosine_offdiag_stats(endmembers: np.ndarray) -> dict[str, float]:
    matrix = compute_correlation_matrix(endmembers, metric="cosine")
    return summarize_correlation_matrix(matrix)


def stats_table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(rows, orient='index').rename_axis('stage').round(6)


def format_model_metrics_table(model_df: pd.DataFrame) -> pd.DataFrame:
    pivot = model_df.pivot_table(index='metric', columns='model', values='mean', aggfunc='first')
    return pivot.sort_index().round(6)


def abundance_vector(row: pd.Series, prefix: str) -> np.ndarray:
    cols = sorted([c for c in row.index if str(c).startswith(prefix)])
    return row[cols].to_numpy(dtype=float)


def plot_pixel_preview(
    pixel_index: int,
    wavelength_axis_full: np.ndarray,
    endmembers_full: np.ndarray,
    abundance_rows: pd.DataFrame,
    snr_db: float,
) -> go.Figure:
    row_sunsal = abundance_rows[(abundance_rows['pixel_index'] == pixel_index) & (abundance_rows['model'] == 'sunsal')].iloc[0]
    row_vpgdu = abundance_rows[(abundance_rows['pixel_index'] == pixel_index) & (abundance_rows['model'] == 'vpgdu')].iloc[0]
    a_true = abundance_vector(row_sunsal, 'true_a')
    a_sunsal = abundance_vector(row_sunsal, 'est_a')
    a_vpgdu = abundance_vector(row_vpgdu, 'est_a')
    y_clean = np.asarray(a_true @ endmembers_full, dtype=float)
    if np.isinf(float(snr_db)):
        y_noisy = y_clean.copy()
    else:
        signal_power = float(np.mean(y_clean ** 2))
        signal_rms = float(np.sqrt(max(signal_power, 0.0)))
        noise_std = signal_rms * float(10.0 ** (-float(snr_db) / 20.0))
        rng = np.random.default_rng(seed=int(pixel_index))
        y_noisy = y_clean + rng.normal(loc=0.0, scale=noise_std, size=y_clean.shape)
    y_sunsal = np.asarray(a_sunsal @ endmembers_full, dtype=float)
    y_vpgdu = np.asarray(a_vpgdu @ endmembers_full, dtype=float)
    fig = go.Figure()
    fig.add_scatter(x=wavelength_axis_full, y=y_clean, mode='lines', name='without_noise')
    fig.add_scatter(x=wavelength_axis_full, y=y_noisy, mode='lines', name='with_noise', line=dict(dash='dash'))
    fig.add_scatter(x=wavelength_axis_full, y=y_sunsal, mode='lines', name='sunsal')
    fig.add_scatter(x=wavelength_axis_full, y=y_vpgdu, mode='lines', name='vpgdu')
    fig.update_layout(
        title=f'Reflectance by source | pixel_index={pixel_index}',
        xaxis_title='Wavelength (µm)',
        yaxis_title='Value',
        template='plotly_white',
        width=950,
        height=430,
    )
    return fig


def run_diagnostics_notebook(config_path: Path, project_root: Path) -> None:
    experiment_config = ExperimentConfig.from_file(config_path)
    result = run_correlation_experiments(experiment_config)
    cluster_paths = {
        item.name: resolve_cluster_path(experiment_config, item.path)
        for item in experiment_config.cluster_sets
    }

    model_summary_path = Path(result['model_evaluation']['summary_path'])
    abundance_preview_path = Path(result['model_evaluation']['abundance_preview_path'])

    model_df = pd.read_csv(model_summary_path)
    abundance_df = pd.read_csv(abundance_preview_path)

    display(Markdown(f"**Run name:** `{result['run_name']}`\n\n**Output dir:** `{result['output_dir']}`"))

    model_eval = experiment_config.model_evaluation
    for run_index, run_cfg in enumerate(model_eval.runs, start=1):
        display(Markdown(f'---\n## Run {run_index}/{len(model_eval.runs)}'))

        cluster_set = run_cfg.cluster_set
        bands_ranges = run_cfg.normalized_bands_ranges()
        normalization = run_cfg.normalization
        transform_steps = run_cfg.normalized_transform_steps()
        transform_label = run_cfg.normalized_transform()
        snr_db = run_cfg.snr_db
        bands_key = bands_ranges_key(bands_ranges)

        bands_label = ", ".join(
            f"{x_min:g}-{x_max:g} {reduce}"
            for x_min, x_max, reduce in bands_ranges
        )
        config_view = pd.DataFrame(
            {
                'value': [
                    cluster_set,
                    bands_label,
                    normalization,
                    transform_label,
                    run_cfg.num_pixels,
                    f'{snr_db:g} dB' if np.isfinite(snr_db) else 'inf',
                    ', '.join(run_cfg.normalized_models()),
                ]
            },
            index=[
                'cluster set',
                'bands',
                'normalization',
                'transform',
                'pixels',
                'snr',
                'models',
            ],
        )
        display(config_view)

        cluster_path = cluster_paths[cluster_set]
        wavelength_axis_full, endmembers_full = load_wavelength_and_cluster_matrix(cluster_path)

        display(plot_cluster_overview(
            wavelength_axis=wavelength_axis_full,
            endmembers=endmembers_full,
            title=f'Raw clusters | set={cluster_set}',
            bands_ranges=bands_ranges,
            y_title='Reflectance',
        ))

        _, raw_endmembers_selected = select_wavelength_ranges(
            wavelengths=wavelength_axis_full,
            endmembers=endmembers_full,
            bands_ranges=bands_ranges,
        )
        stats_payload = {'raw': cosine_offdiag_stats(raw_endmembers_selected)}

        normalized_endmembers_full, _ = apply_normalization(endmembers_full, endmembers_full, wavelength_axis_full, normalization)
        _, normalized_endmembers_selected = select_wavelength_ranges(
            wavelengths=wavelength_axis_full,
            endmembers=normalized_endmembers_full,
            bands_ranges=bands_ranges,
        )
        if normalization != 'without':
            stats_payload['normalized'] = cosine_offdiag_stats(normalized_endmembers_selected)
            display(plot_cluster_overview(
                wavelength_axis=wavelength_axis_full,
                endmembers=normalized_endmembers_full,
                title=f'Normalized spectra | {cluster_set} | {normalization}',
                y_title='Value',
                bands_ranges=bands_ranges,
            ))

        transformed_endmembers = normalized_endmembers_selected
        for step_name, step_params in transform_steps:
            transformed_endmembers, _ = apply_transform(transformed_endmembers, transformed_endmembers, kind=step_name, params=step_params)
            stats_payload[step_name] = cosine_offdiag_stats(transformed_endmembers)

        display(stats_table(stats_payload))

        model_rows = model_df[
            (model_df['cluster_set'] == cluster_set)
            & (model_df['bands_ranges'] == bands_key)
            & (model_df['normalization'] == normalization)
            & (model_df['transform'] == transform_label)
            & (model_df['snr_db'].astype(float) == float(snr_db))
        ].copy()
        display(Markdown('### Model metrics'))
        display(format_model_metrics_table(model_rows))

        abundance_rows = abundance_df[
            (abundance_df['cluster_set'] == cluster_set)
            & (abundance_df['bands_ranges'] == bands_key)
            & (abundance_df['normalization'] == normalization)
            & (abundance_df['transform'] == transform_label)
            & (abundance_df['snr_db'].astype(float) == float(snr_db))
        ].copy()
        display_abundance_comparison_tables(abundance_rows, max_pixels=5)

        display(Markdown('### Synthetic pixel spectra preview'))
        endmembers_full_for_plot, _ = apply_normalization(endmembers_full, endmembers_full, wavelength_axis_full, normalization)
        for pixel_index in sorted(abundance_rows['pixel_index'].astype(int).unique()):
            display(plot_pixel_preview(
                pixel_index=pixel_index,
                wavelength_axis_full=wavelength_axis_full,
                endmembers_full=endmembers_full_for_plot,
                abundance_rows=abundance_rows,
                snr_db=snr_db,
            ))
