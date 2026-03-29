from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from clusters_unmixing.config.schema import ExperimentConfig
from clusters_unmixing.data import generate_samples
from clusters_unmixing.dataio import load_wavelength_and_cluster_matrix
from clusters_unmixing.metrics import compute_correlation_matrix, summarize_correlation_matrix
from clusters_unmixing.models.runner_registry import run_registered_model
from clusters_unmixing.transforms.normalization import apply_normalization
from clusters_unmixing.transforms.spectral_views import apply_transform, select_wavelength_ranges

def _planned_model_runs(exp: ExperimentConfig) -> list[dict[str, Any]]:
    model_eval = exp.model_evaluation
    model_params = {model.normalized_name(): dict(model.params) for model in model_eval.models}
    runs = []
    for item in model_eval.runs:
        bands_ranges = item.normalized_bands_ranges()
        runs.append({
            "cluster_set": item.cluster_set,
            "bands_ranges": bands_ranges,
            "normalization": item.normalization,
            "transform": item.normalized_transform(),
            "transform_steps": item.normalized_transform_steps(),
            "models": [{"name": name, "params": dict(model_params[name])} for name in item.normalized_models()],
            "num_pixels": item.num_pixels,
            "snr_db": item.snr_db,
        })
    return runs

def _build_stage_projections(
    run: dict[str, Any],
    wavelengths: np.ndarray,
    raw_endmembers: np.ndarray,
    raw_pixels: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    wavelengths_sel, selected_endmembers = select_wavelength_ranges(
        wavelengths,
        raw_endmembers,
        run["bands_ranges"],
    )
    _, selected_pixels = select_wavelength_ranges(
        wavelengths,
        raw_pixels,
        run["bands_ranges"],
    )
    projections: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("raw", selected_endmembers, selected_pixels)
    ]

    projected_endmembers, projected_pixels = apply_normalization(
        selected_endmembers,
        selected_pixels,
        wavelengths_sel,
        run["normalization"],
    )
    projections.append(("normalized", projected_endmembers, projected_pixels))

    for name, params in run["transform_steps"]:
        projected_endmembers, projected_pixels = apply_transform(projected_endmembers, projected_pixels, name, params)
        projections.append((name, projected_endmembers, projected_pixels))

    return projections

def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_synthetic_pixels(endmembers: np.ndarray, num_pixels: int, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    n_endmembers = endmembers.shape[0]
    abundances = generate_samples(
        num_samples=num_pixels,
        max_non_zero_endmembers=n_endmembers,
        num_endmembers=n_endmembers,
    )
    clean_pixels = abundances @ endmembers
    if np.isinf(snr_db):
        return clean_pixels, abundances
    signal_power = float((clean_pixels ** 2).mean())
    signal_rms = float(np.sqrt(max(signal_power, 0.0)))
    noise_std = signal_rms * float(10.0 ** (-snr_db / 20.0))
    noise = np.random.normal(loc=0.0, scale=1.0, size=clean_pixels.shape).astype(clean_pixels.dtype)
    noisy_pixels = clean_pixels + (noise_std * noise)
    return noisy_pixels, abundances


def _abundance_preview_row(
    run_index: int,
    pixel_index: int,
    source: str,
    abundances: np.ndarray,
    rmse_vs_true: float,
) -> dict[str, Any]:
    row = {
        'run_index': run_index,
        'pixel_index': pixel_index,
        'source': source,
        'abundance_rmse_vs_true': rmse_vs_true,
    }
    for j, value in enumerate(abundances, start=1):
        row[f'endmember_{j}'] = float(value)
    return row

def run_experiments(exp: ExperimentConfig) -> dict[str, Any]:
    output_dir = exp.experiment_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _planned_model_runs(exp)
    summary_metric = 'cosine' if 'cosine' in exp.metrics else (exp.metrics[0] if exp.metrics else 'cosine')

    correlation_summary_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    abundance_preview_rows: list[dict[str, Any]] = []

    if not runs:
        correlation_summary_path = output_dir / 'correlation_summary.csv'
        pd.DataFrame(correlation_summary_rows).to_csv(correlation_summary_path, index=False)
        return {
            'experiment_name': exp.experiment_name,
            'output_dir': str(output_dir),
            'correlation_summary_path': str(correlation_summary_path),
            'n_runs': 0,
            'model_evaluation': {'n_runs': 0},
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, run in enumerate(runs, start=1):
        cluster_cfg = next(item for item in exp.cluster_sets if item.name == run["cluster_set"])
        cluster_path = exp.resolve_path(cluster_cfg.path)
        wavelengths, raw_endmembers = load_wavelength_and_cluster_matrix(cluster_path)

        _set_global_seeds(0)
        raw_pixels, true_abundances = _make_synthetic_pixels(
            raw_endmembers,
            run["num_pixels"],
            run["snr_db"],
        )

        stage_projections = _build_stage_projections(
            run,
            wavelengths,
            raw_endmembers,
            raw_pixels,
        )
        _, projected_endmembers, projected_pixels = stage_projections[-1]
        for stage_name, stage_endmembers, _ in stage_projections:
            matrix = compute_correlation_matrix(stage_endmembers, summary_metric)
            correlation_summary_rows.append({
                'run_index': idx,
                'stage': stage_name,
                **summarize_correlation_matrix(matrix),
            })

        endmembers_t = torch.tensor(projected_endmembers, dtype=torch.float32, device=device)
        projected_pixels_t = torch.tensor(projected_pixels, dtype=torch.float32, device=device)

        n_preview_available = int(true_abundances.shape[0])
        preview_pixels: list[int] = []
        if n_preview_available > 0:
            preview_limit = min(5, n_preview_available)
            preview_pixels = np.random.choice(n_preview_available, size=preview_limit, replace=False).astype(int).tolist()
            preview_pixels.sort()
            abundance_preview_rows.extend(
                _abundance_preview_row(idx, sample_idx, 'true', true_abundances[sample_idx], 0.0)
                for sample_idx in preview_pixels
            )

        for model_spec in run['models']:
            _set_global_seeds(0)
            abundances_t, _ = run_registered_model(
                model_name=model_spec['name'],
                endmembers=endmembers_t,
                pixels=projected_pixels_t,
                true_abundances=torch.tensor(true_abundances, dtype=torch.float32, device=device),
                params=model_spec['params'],
            )
            abundances = abundances_t.detach().cpu().numpy()
            rmse = float(np.sqrt(np.mean(np.square(abundances - true_abundances))))
            mae = float(np.mean(np.abs(abundances - true_abundances)))
            for metric_name, value in [('abundance_rmse', rmse), ('abundance_mae', mae)]:
                model_rows.append({
                    'run_index': idx,
                    'model': model_spec['name'],
                    'metric': metric_name,
                    'mean': value,
                })
            for sample_idx in preview_pixels:
                pred_abundances = abundances[sample_idx]
                abundance_preview_rows.append(
                    _abundance_preview_row(
                        idx,
                        sample_idx,
                        model_spec['name'],
                        pred_abundances,
                        float(np.sqrt(np.mean(np.square(pred_abundances - true_abundances[sample_idx])))),
                    )
                )

    correlation_summary_path = output_dir / 'correlation_summary.csv'
    model_summary_path = output_dir / 'model_summary.csv'
    abundance_preview_path = output_dir / 'abundance_preview.csv'
    pd.DataFrame(correlation_summary_rows).to_csv(correlation_summary_path, index=False, float_format='%.6f')
    pd.DataFrame(model_rows).pivot(index=['run_index', 'metric'], columns='model', values='mean').reset_index().to_csv(
        model_summary_path, index=False, float_format='%.6f'
    )
    abundance_preview_df = pd.DataFrame(abundance_preview_rows).assign(
        source_order=lambda df: (df['source'] != 'true').astype(int)
    )
    abundance_preview_df.sort_values(['run_index', 'pixel_index', 'source_order', 'source']).drop(
        columns='source_order'
    ).to_csv(abundance_preview_path, index=False, float_format='%.6f')
    return {
        'experiment_name': exp.experiment_name,
        'output_dir': str(output_dir),
        'correlation_summary_path': str(correlation_summary_path),
        'n_runs': len(runs),
        'model_evaluation': {
            'n_runs': len(runs),
            'model_summary_path': str(model_summary_path),
            'abundance_preview_path': str(abundance_preview_path),
        },
    }
