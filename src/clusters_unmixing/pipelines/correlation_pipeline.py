from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from clusters_unmixing.config.schema import ExperimentConfig, serialize_bands_ranges_key
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
        ranges_key = serialize_bands_ranges_key(bands_ranges)
        runs.append({
            "cluster_set": item.cluster_set,
            "bands_ranges": bands_ranges,
            "bands_ranges_key": ranges_key,
            "normalization": item.normalization,
            "transform": item.normalized_transform(),
            "transform_steps": item.normalized_transform_steps(),
            "models": [{"name": name, "params": dict(model_params[name])} for name in item.normalized_models()],
            "num_pixels": item.num_pixels,
            "snr_db": item.snr_db,
        })
    return runs

def _build_projection(
    run: dict[str, Any],
    wavelengths: np.ndarray,
    raw_endmembers: np.ndarray,
    raw_pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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
    projected_endmembers, projected_pixels = apply_normalization(
        selected_endmembers,
        selected_pixels,
        wavelengths_sel,
        run["normalization"],
    )

    for name, params in run["transform_steps"]:
        projected_endmembers, projected_pixels = apply_transform(projected_endmembers, projected_pixels, name, params)

    return projected_endmembers, projected_pixels

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

def run_experiments(exp: ExperimentConfig) -> dict[str, Any]:
    output_dir = exp.experiment_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _planned_model_runs(exp)

    summary_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    abundance_preview_rows: list[dict[str, Any]] = []

    if not runs:
        summary_path = output_dir / 'correlation_summary.csv'
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        return {
            'experiment_name': exp.experiment_name,
            'output_dir': str(output_dir),
            'summary_path': str(summary_path),
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

        projected_endmembers, projected_pixels = _build_projection(
            run,
            wavelengths,
            raw_endmembers,
            raw_pixels,
        )
        for metric in exp.metrics:
            matrix = compute_correlation_matrix(projected_endmembers, metric)
            stats = summarize_correlation_matrix(matrix)
            row = {
                'run_index': idx,
                'cluster_set': run['cluster_set'],
                'bands_ranges': run['bands_ranges_key'],
                'normalization': run['normalization'],
                'transform': run['transform'],
                'metric': metric,
                **stats,
            }
            summary_rows.append(row)

        endmembers_t = torch.tensor(projected_endmembers, dtype=torch.float32, device=device)
        projected_pixels_t = torch.tensor(projected_pixels, dtype=torch.float32, device=device)

        n_preview_available = int(true_abundances.shape[0])
        preview_pixels: list[int] = []
        if n_preview_available > 0:
            preview_limit = min(5, n_preview_available)
            preview_pixels = np.random.choice(n_preview_available, size=preview_limit, replace=False).astype(int).tolist()
            preview_pixels.sort()

        for model_spec in run['models']:
            _set_global_seeds(0)
            abundances_t, metadata = run_registered_model(
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
                    'cluster_set': run['cluster_set'],
                    'bands_ranges': run['bands_ranges_key'],
                    'normalization': run['normalization'],
                    'transform': run['transform'],
                    'snr_db': run['snr_db'],
                    'model': model_spec['name'],
                    'metric': metric_name,
                    'mean': value,
                    'iterations_logged': metadata.get('iterations_logged', 0),
                    'last_active_pixels': metadata.get('last_active_pixels', projected_pixels.shape[0]),
                })
            for sample_idx in preview_pixels:
                row = {
                    'run_index': idx,
                    'cluster_set': run['cluster_set'],
                    'bands_ranges': run['bands_ranges_key'],
                    'normalization': run['normalization'],
                    'transform': run['transform'],
                    'snr_db': run['snr_db'],
                    'model': model_spec['name'],
                    'pixel_index': sample_idx,
                }
                for j, value in enumerate(true_abundances[sample_idx], start=1):
                    row[f'true_a{j}'] = float(value)
                for j, value in enumerate(abundances[sample_idx], start=1):
                    row[f'est_a{j}'] = float(value)
                abundance_preview_rows.append(row)

    summary_path = output_dir / 'correlation_summary.csv'
    model_summary_path = output_dir / 'model_summary.csv'
    abundance_preview_path = output_dir / 'abundance_preview.csv'
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(model_rows).to_csv(model_summary_path, index=False)
    pd.DataFrame(abundance_preview_rows).to_csv(abundance_preview_path, index=False)
    return {
        'experiment_name': exp.experiment_name,
        'output_dir': str(output_dir),
        'summary_path': str(summary_path),
        'n_runs': len(runs),
        'model_evaluation': {
            'n_runs': len(runs),
            'summary_path': str(model_summary_path),
            'abundance_preview_path': str(abundance_preview_path),
        },
    }
