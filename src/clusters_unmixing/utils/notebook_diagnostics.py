from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display

from clusters_unmixing.config import ExperimentConfig
from clusters_unmixing.dataio.clusters import load_wavelength_and_cluster_matrix
from clusters_unmixing.pipelines import run_correlation_experiments
from clusters_unmixing.transforms import apply_normalization, apply_transform, select_wavelength_ranges
from clusters_unmixing.utils.diagnostics import (
    build_model_run_diagnostics,
    display_abundance_comparison_tables,
    plot_cluster_overview,
)


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


def _bands_key(raw_bands_ranges: list[Any]) -> str:
    return json.dumps(raw_bands_ranges, separators=(",", ":"), ensure_ascii=False)


def _cluster_path_map(cfg: ExperimentConfig, project_root: str | Path | None = None) -> dict[str, Path]:
    root = Path(project_root).resolve() if project_root is not None else None
    config_dir = Path(cfg.config_dir or root or Path.cwd())
    return {item.name: (config_dir / item.path).resolve() for item in cfg.cluster_sets}


def cosine_matrix(signatures: np.ndarray) -> np.ndarray:
    matrix = np.asarray(signatures, dtype=float).T
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    return normalized @ normalized.T


def cosine_offdiag_stats(signatures: np.ndarray) -> dict[str, float]:
    cosine = cosine_matrix(signatures)
    mask = ~np.eye(cosine.shape[0], dtype=bool)
    offdiag = cosine[mask]
    return {
        'mean_abs_offdiag': float(np.mean(np.abs(offdiag))),
        'max_abs_offdiag': float(np.max(np.abs(offdiag))),
        'min_offdiag': float(np.min(offdiag)),
        'max_offdiag': float(np.max(offdiag)),
    }


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
    wavelengths_full: np.ndarray,
    endmembers_full: np.ndarray,
    abundance_rows: pd.DataFrame,
    snr_db: float,
) -> go.Figure:
    row_sunsal = abundance_rows[(abundance_rows['pixel_index'] == pixel_index) & (abundance_rows['model'] == 'sunsal')].iloc[0]
    row_vpgdu = abundance_rows[(abundance_rows['pixel_index'] == pixel_index) & (abundance_rows['model'] == 'vpgdu')].iloc[0]
    a_true = abundance_vector(row_sunsal, 'true_a')
    a_sunsal = abundance_vector(row_sunsal, 'est_a')
    a_vpgdu = abundance_vector(row_vpgdu, 'est_a')
    y_clean = np.asarray(endmembers_full @ a_true, dtype=float)
    if np.isinf(float(snr_db)):
        y_noisy = y_clean.copy()
    else:
        signal_power = float(np.mean(y_clean ** 2))
        signal_rms = float(np.sqrt(max(signal_power, 0.0)))
        noise_std = signal_rms * float(10.0 ** (-float(snr_db) / 20.0))
        rng = np.random.default_rng(seed=int(pixel_index))
        y_noisy = y_clean + rng.normal(loc=0.0, scale=noise_std, size=y_clean.shape)
    y_sunsal = np.asarray(endmembers_full @ a_sunsal, dtype=float)
    y_vpgdu = np.asarray(endmembers_full @ a_vpgdu, dtype=float)
    fig = go.Figure()
    fig.add_scatter(x=wavelengths_full, y=y_clean, mode='lines', name='without_noise')
    fig.add_scatter(x=wavelengths_full, y=y_noisy, mode='lines', name='with_noise', line=dict(dash='dash'))
    fig.add_scatter(x=wavelengths_full, y=y_sunsal, mode='lines', name='sunsal')
    fig.add_scatter(x=wavelengths_full, y=y_vpgdu, mode='lines', name='vpgdu')
    fig.update_layout(
        title=f'Reflectance by source | pixel_index={pixel_index}',
        xaxis_title='Wavelength (Âµm)',
        yaxis_title='Value',
        template='plotly_white',
        width=950,
        height=430,
    )
    return fig


def run_diagnostics_notebook(config_path: Path, project_root: Path) -> None:
    experiment_config = ExperimentConfig.from_file(config_path)
    result = run_correlation_experiments(experiment_config)
    cluster_paths = _cluster_path_map(experiment_config, project_root=project_root)

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
        normalization = run_cfg.normalized_normalization()
        transform_steps = run_cfg.normalized_transform_steps()
        transform_label = run_cfg.normalized_transform()
        snr_db = run_cfg.resolved_snr_db()
        bands_key = _bands_key(run_cfg.serialized_bands_ranges())

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
                    run_cfg.resolved_num_pixels(),
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
        wavelengths_full, signatures_full = load_wavelength_and_cluster_matrix(cluster_path)

        display(plot_cluster_overview(
            wavelengths=wavelengths_full,
            signatures=signatures_full,
            title=f'Raw clusters | set={cluster_set}',
            bands_ranges=bands_ranges,
            y_title='Reflectance',
        ))

        _, raw_selected, _ = select_wavelength_ranges(
            wavelengths=wavelengths_full,
            signatures=signatures_full,
            bands_ranges=run_cfg.serialized_bands_ranges(),
        )
        stats_payload = {'raw': cosine_offdiag_stats(raw_selected)}

        normalized_full = apply_normalization(signatures_full, wavelengths_full, normalization)
        _, normalized_selected, _ = select_wavelength_ranges(
            wavelengths=wavelengths_full,
            signatures=normalized_full,
            bands_ranges=run_cfg.serialized_bands_ranges(),
        )
        if normalization != 'without':
            stats_payload['normalized'] = cosine_offdiag_stats(normalized_selected)
            display(plot_cluster_overview(
                wavelengths=wavelengths_full,
                signatures=normalized_full,
                title=f'Normalized spectra | {cluster_set} | {normalization}',
                y_title='Value',
                bands_ranges=bands_ranges,
            ))

        transformed = normalized_selected
        for step_name, step_params in transform_steps:
            transformed = apply_transform(transformed, kind=step_name, params=step_params)
            stats_payload[step_name] = cosine_offdiag_stats(transformed)

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
        ].copy()
        display(Markdown('### Abundance comparison by sampled pixel'))
        display_abundance_comparison_tables(abundance_rows, max_pixels=5)

        display(Markdown('### Synthetic pixel spectra preview'))
        endmembers_full_for_plot = apply_normalization(signatures_full, wavelengths_full, normalization)
        for pixel_index in sorted(abundance_rows['pixel_index'].astype(int).unique()):
            display(plot_pixel_preview(
                pixel_index=pixel_index,
                wavelengths_full=wavelengths_full,
                endmembers_full=endmembers_full_for_plot,
                abundance_rows=abundance_rows,
                snr_db=snr_db,
            ))
