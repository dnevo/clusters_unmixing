from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from IPython.display import Markdown, display

from clusters_unmixing.config import ExperimentConfig
from clusters_unmixing.config.schema import BandRangeSpec
from clusters_unmixing.core_math import apply_snr_noise
from clusters_unmixing.dataio import load_wavelength_and_cluster_matrix
from clusters_unmixing.pipelines import run_experiments
from clusters_unmixing.preprocessing import apply_normalization
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 180)


def plot_cluster_overview(
    wavelength_axis: Any,
    endmembers: Any,
    bands_ranges: list[BandRangeSpec] | None = None,
    y_title: str = 'Reflectance',
    show_legend: bool = True,
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
        mask = np.logical_and(
            np.greater_equal(wavelength_axis_arr, float(x_min)),
            np.less_equal(wavelength_axis_arr, float(x_max)),
        )
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
        legend_shown = False

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
                    showlegend=not legend_shown,
                    line={'color': cluster_color, 'dash': dash_style, 'width': 3},
                    hovertemplate='cluster=%{fullData.name}<br>wavelength=%{x:.3f} um<br>value=%{y:.4f}<extra></extra>',
                ))
                legend_shown = True

                if kind == 'mean':
                    mid_x = float((x_seg[0] + x_seg[-1]) / 2.0)
                    marker_idx = int(np.abs(x_seg - mid_x).argmin())
                    fig.add_trace(go.Scatter(
                        x=[float(x_seg[marker_idx])],
                        y=[float(y_seg[marker_idx])],
                        mode='markers',
                        marker={'symbol': 'diamond', 'size': 12, 'color': cluster_color},
                        name=cluster_name,
                        legendgroup=cluster_name,
                        showlegend=False,
                        hovertemplate='cluster=%{fullData.name}<br>wavelength=%{x:.3f} um<br>value=%{y:.4f}<extra></extra>',
                    ))

    fig.update_layout(
        xaxis_title='Wavelength (µm)',
        yaxis_title=y_title,
        template='plotly_white',
        font={'family': 'Arial, sans-serif', 'size': 18, 'color': 'black'},
        showlegend=show_legend,
        legend={
            'title': {'text': 'Clusters', 'font': {'size': 18, 'color': 'black'}},
            'font': {'size': 16, 'color': 'black'},
            'orientation': 'h',
            'yanchor': 'bottom',
            'y': 1.02,
            'xanchor': 'left',
            'x': 0,
        },
        xaxis={'title': {'font': {'size': 20, 'color': 'black'}}, 'tickfont': {'size': 16, 'color': 'black'}},
        yaxis={'title': {'font': {'size': 20, 'color': 'black'}}, 'tickfont': {'size': 16, 'color': 'black'}},
        margin={'l': 80, 'r': 30, 't': 30, 'b': 70},
        width=1000,
        height=420,
    )
    return fig


def format_mean_std(mean: float, std: float, digits: int = 4) -> str:
    """Render a metric as ``mean±std`` to a fixed number of decimal places.

    ``std`` is NaN for a group with a single sample (e.g. ``num_seeds=1``,
    where ``groupby(...).std(ddof=1)`` is undefined) - shown as the mean alone
    rather than a misleading ``±0.0000``.
    """
    if pd.isna(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f}±{std:.{digits}f}"


def build_stability_table(model_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Aggregate a metric's per-seed values (see model_summary.csv's 'seed'
    column) into one ``mean±std`` string per resolved run and model, so
    stability across seeds can be read off directly instead of comparing raw
    per-seed rows by eye.
    """
    subset = model_df[model_df['metric'] == metric]
    id_cols = {'run_index', 'parent_run_index', 'run_overrides', 'seed', 'metric'}
    model_cols = [c for c in subset.columns if c not in id_cols]
    grouped = subset.groupby(['run_index', 'run_overrides'], sort=False)[model_cols]
    mean_df = grouped.mean()
    std_df = grouped.std(ddof=1)
    return pd.DataFrame(
        {col: [format_mean_std(m, s) for m, s in zip(mean_df[col], std_df[col])] for col in model_cols},
        index=mean_df.index,
    )


def abundance_vector(row: pd.Series) -> np.ndarray:
    cols = sorted([c for c in row.index if str(c).startswith('endmember_')])
    return row[cols].to_numpy(dtype=float)


def plot_pixel_preview(
    pixel_index: int,
    wavelength_axis_full: np.ndarray,
    endmembers_full: np.ndarray,
    abundance_rows: pd.DataFrame,
    snr_db: float,
) -> go.Figure:
    pixel_rows = abundance_rows[abundance_rows['pixel_index'] == pixel_index].sort_values('source').copy()
    if pixel_rows.empty:
        raise ValueError(f'No abundance rows found for pixel_index={pixel_index}')

    reference_row = pixel_rows[pixel_rows['source'] == 'true'].iloc[0]
    a_true = abundance_vector(reference_row)
    y_clean = np.asarray(a_true @ endmembers_full, dtype=float)
    y_noisy, _ = apply_snr_noise(y_clean, float(snr_db))

    fig = go.Figure()
    fig.add_scatter(x=wavelength_axis_full, y=y_clean, mode='lines', name='without_noise')
    fig.add_scatter(x=wavelength_axis_full, y=y_noisy, mode='lines', name='with_noise', line=dict(dash='dash'))
    for _, row in pixel_rows[pixel_rows['source'] != 'true'].iterrows():
        model_name = str(row['source'])
        a_est = abundance_vector(row)
        y_est = np.asarray(a_est @ endmembers_full, dtype=float)
        fig.add_scatter(x=wavelength_axis_full, y=y_est, mode='lines', name=model_name)

    fig.update_layout(
        title=f'Reflectance by source | pixel_index={pixel_index}',
        xaxis_title='Wavelength (µm)',
        yaxis_title='Value',
        template='plotly_white',
        width=950,
        height=430,
    )
    return fig


def run_experiments_notebook(project_root: Path) -> None:
    experiment_config = ExperimentConfig.from_config_file(project_root)
    result = run_experiments(experiment_config)
    cluster_paths = {
        item.name: experiment_config.resolve_path(item.path)
        for item in experiment_config.cluster_sets
    }

    model_summary_path = Path(result['model_evaluation']['model_summary_path'])
    abundance_preview_path = Path(result['model_evaluation']['abundance_preview_path'])
    endmember_separability_summary_path = Path(result['endmember_separability_summary_path'])

    # Not indexed by run_index here: a swept run expands into several resolved
    # run_index values sharing one parent_run_index, so filtering by
    # parent_run_index (below, per notebook section) is what actually maps back
    # to a single configured run.
    endmember_separability_df = pd.read_csv(endmember_separability_summary_path)
    model_df = pd.read_csv(model_summary_path)
    abundance_df = pd.read_csv(abundance_preview_path)

    display(Markdown(f"**Output dir:** `{result['output_dir']}`"))

    for run_index, run_cfg in enumerate(experiment_config.runs, start=1):
        display(Markdown(f'---\n## Run {run_index}/{len(experiment_config.runs)}'))

        cluster_set = run_cfg.cluster_set
        bands_ranges = run_cfg.normalized_bands_ranges()
        normalization = run_cfg.normalization
        abundance_distribution = run_cfg.abundance_distribution
        dirichlet_alpha = run_cfg.dirichlet_alpha
        snr_db = run_cfg.snr_db

        bands_label = ", ".join(
            f"{x_min:g}-{x_max:g} {reduce}"
            for x_min, x_max, reduce in bands_ranges
        )
        parts = [f"{k}={v}" for k, v in run_cfg.sweep_params.items()]
        parts += [
            f"{entry.name}.{pkey}={pvalues}"
            for entry in run_cfg.models
            for pkey, pvalues in entry.param_sweep.items()
        ]
        sweep_label = '; '.join(parts) if parts else '-'
        config_view = pd.DataFrame(
            {
                'value': [
                    cluster_set,
                    bands_label,
                    normalization,
                    abundance_distribution,
                    dirichlet_alpha,
                    run_cfg.num_pixels,
                    f'{snr_db:g} dB' if np.isfinite(snr_db) else 'inf',
                    ', '.join(run_cfg.effective_model_names()),
                    sweep_label,
                ]
            },
            index=[
                'cluster set',
                'bands',
                'normalization',
                'abundance distribution',
                'Dirichlet alpha',
                'pixels',
                'snr',
                'models',
                'sweep',
            ],
        )
        display(config_view)

        cluster_path = cluster_paths[cluster_set]
        wavelength_axis_full, endmembers_full = load_wavelength_and_cluster_matrix(cluster_path)

        display(Markdown(f'**Raw clusters | set={cluster_set}**'))
        display(plot_cluster_overview(
            wavelength_axis=wavelength_axis_full,
            endmembers=endmembers_full,
            bands_ranges=bands_ranges,
            y_title='Reflectance',
        ))

        normalized_endmembers_full = apply_normalization(
            endmembers_full,
            wavelength_axis_full,
            normalization
        )

        if normalization != 'none':
            display(Markdown(f'**Normalized spectra | {cluster_set} | {normalization}**'))
            display(plot_cluster_overview(
                wavelength_axis=wavelength_axis_full,
                endmembers=normalized_endmembers_full,
                y_title='Value',
                bands_ranges=bands_ranges,
                show_legend=False,
            ))

        display(Markdown('### Endmember separability metrics'))
        run_endmember_separability = endmember_separability_df[endmember_separability_df['parent_run_index'] == run_index].drop(
            columns=['run_index', 'parent_run_index', 'run_overrides']
        ).drop_duplicates().set_index('stage')
        display(run_endmember_separability.style.format({'condition_number': '{:.1f}'}))

        display(Markdown('### Model metrics'))
        run_model = model_df[model_df['parent_run_index'] == run_index].drop(columns=['parent_run_index'])
        run_model_indexed = run_model.set_index(['seed', 'run_index', 'run_overrides', 'metric'])

        display(Markdown('#### Abundance RMSE'))
        display(run_model_indexed.xs('abundance_rmse', level='metric', drop_level=False))

        display(Markdown('#### Reconstruction RMSE'))
        display(run_model_indexed.xs('reconstruction_rmse', level='metric', drop_level=False))

        abundance_rows = abundance_df[abundance_df['parent_run_index'] == run_index].drop(columns=['parent_run_index'])
        display(Markdown('### Abundances'))
        display(abundance_rows.reset_index(drop=True).style.hide(axis='index'))

        display(Markdown('### Synthetic pixel spectra preview'))
        for pixel_index in sorted(abundance_rows['pixel_index'].astype(int).unique()):
            display(plot_pixel_preview(
                pixel_index=pixel_index,
                wavelength_axis_full=wavelength_axis_full,
                endmembers_full=normalized_endmembers_full,
                abundance_rows=abundance_rows,
                snr_db=snr_db,
            ))

    display(Markdown('---\n## Stability across seeds'))
    display(Markdown(
        f'Each resolved run was repeated with `num_seeds={experiment_config.num_seeds}` seed(s) '
        f'(0..{experiment_config.num_seeds - 1}); cells are mean±std across seeds, 4 decimal places.'
    ))

    display(Markdown('### Abundance RMSE'))
    display(build_stability_table(model_df, 'abundance_rmse'))

    display(Markdown('### Reconstruction RMSE'))
    display(build_stability_table(model_df, 'reconstruction_rmse'))
