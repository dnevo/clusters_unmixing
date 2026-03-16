from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from clusters_unmixing.config.schema import ExperimentConfig
from clusters_unmixing.data import generate_samples
from clusters_unmixing.dataio.clusters import load_wavelength_and_cluster_matrix
from clusters_unmixing.metrics.correlation import compute_correlation_matrix, summarize_correlation_matrix
from clusters_unmixing.models.runner_registry import run_registered_model
from clusters_unmixing.transforms.spectral_views import apply_transform, select_wavelength_ranges


def _resolve_cluster_path(exp: ExperimentConfig, cluster_path: str) -> Path:
    path = Path(cluster_path)
    if path.is_absolute():
        return path
    candidates: list[Path] = []
    if exp.config_dir:
        config_dir = Path(exp.config_dir)
        candidates.append(config_dir / path)
        candidates.extend(parent / path for parent in config_dir.parents)
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else path.resolve()


def _resolve_output_dir(exp: ExperimentConfig) -> Path:
    path = Path(exp.output_dir)
    if path.is_absolute():
        return path
    if exp.config_dir:
        return Path(exp.config_dir) / path
    return Path.cwd() / path


def _planned_model_runs(exp: ExperimentConfig) -> list[dict[str, Any]]:
    model_eval = exp.model_evaluation
    model_params = {model.normalized_name(): dict(model.params) for model in model_eval.models}
    runs = []
    for item in model_eval.runs:
        bands_ranges = item.serialized_bands_ranges()
        bands_ranges_key = json.dumps(bands_ranges, separators=(",", ":"), ensure_ascii=False)
        runs.append({
            "cluster_set": item.cluster_set,
            "bands_ranges": bands_ranges,
            "bands_ranges_key": bands_ranges_key,
            "normalization": item.normalized_normalization(),
            "transform": item.normalized_transform(),
            "transform_steps": item.normalized_transform_steps(),
            "projection": item.projection_name(),
            "models": [{"name": name, "params": dict(model_params[name])} for name in item.normalized_models()],
            "num_pixels": item.resolved_num_pixels(),
            "snr_db": item.resolved_snr_db(),
        })
    return runs


def _apply_normalization(signatures: np.ndarray, wavelengths: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "without":
        return signatures
    if normalization == "with_quadratic":
        q_values = -0.20 * wavelengths**2 + 0.68 * wavelengths - 0.12
        return signatures - q_values[:, None]
    raise ValueError(f"Unsupported normalization mode: {normalization}")


def _build_projection(exp: ExperimentConfig, run: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    cluster_cfg = next(item for item in exp.cluster_sets if item.name == run["cluster_set"])
    cluster_path = _resolve_cluster_path(exp, cluster_cfg.path)
    wavelengths, signatures = load_wavelength_and_cluster_matrix(cluster_path)
    wavelengths_sel, signatures_sel, _ = select_wavelength_ranges(wavelengths, signatures, run["bands_ranges"])
    processed = _apply_normalization(signatures_sel, wavelengths_sel, run["normalization"])
    for name, params in run["transform_steps"]:
        processed = apply_transform(processed, name, params)
    return wavelengths_sel, processed


def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_synthetic_pixels(endmembers: np.ndarray, num_pixels: int, snr_db: float) -> tuple[np.ndarray, np.ndarray]:
    n_endmembers = int(endmembers.shape[1])
    raw_endmembers = torch.as_tensor(endmembers, dtype=torch.float32)
    abundances = generate_samples(
        num_samples=num_pixels,
        max_non_zero_endmembers=n_endmembers,
        num_endmembers=n_endmembers,
    )
    clean_pixels = abundances @ raw_endmembers.T
    if np.isinf(snr_db):
        return clean_pixels.detach().cpu().numpy(), abundances.detach().cpu().numpy()
    signal_power = float(clean_pixels.pow(2).mean().item())
    signal_rms = float(np.sqrt(max(signal_power, 0.0)))
    noise_std = signal_rms * float(10.0 ** (-snr_db / 20.0))
    noisy_pixels = clean_pixels + (noise_std * torch.randn_like(clean_pixels))
    return noisy_pixels.detach().cpu().numpy(), abundances.detach().cpu().numpy()


def run_correlation_experiments(config_path: str | Path) -> dict[str, Any]:
    exp = ExperimentConfig.from_json_file(config_path)
    output_dir = _resolve_output_dir(exp) / exp.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _planned_model_runs(exp)
    summary_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    abundance_preview_rows: list[dict[str, Any]] = []
    spectra_preview_rows: list[dict[str, Any]] = []

    if not runs:
        summary_path = output_dir / 'correlation_summary.csv'
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        return {
            'run_name': exp.run_name,
            'output_dir': str(output_dir),
            'summary_path': str(summary_path),
            'n_runs': 0,
            'model_evaluation': {'n_runs': 0},
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for idx, run in enumerate(runs, start=1):
        wavelengths, endmembers = _build_projection(exp, run)
        for metric in exp.metrics:
            matrix = compute_correlation_matrix(endmembers, metric)
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

        _set_global_seeds(0)
        pixels, true_abundances = _make_synthetic_pixels(endmembers, run['num_pixels'], run['snr_db'])
        endmembers_t = torch.tensor(endmembers, dtype=torch.float32, device=device)
        pixels_t = torch.tensor(pixels, dtype=torch.float32, device=device)
        true_abundances_t = torch.tensor(true_abundances, dtype=torch.float32, device=device)

        n_preview_available = int(true_abundances.shape[0])
        preview_pixels: list[int] = []
        if n_preview_available > 0:
            preview_limit = min(5, n_preview_available)
            preview_pixels = np.random.choice(n_preview_available, size=preview_limit, replace=False).astype(int).tolist()
            preview_pixels.sort()

        for sample_idx in preview_pixels:
            row = {
                'run_index': idx,
                'cluster_set': run['cluster_set'],
                'bands_ranges': run['bands_ranges_key'],
                'normalization': run['normalization'],
                'transform': run['transform'],
                'snr_db': run['snr_db'],
                'pixel_index': int(sample_idx),
            }
            for band_idx, value in enumerate(pixels[sample_idx], start=1):
                row[f'band_{band_idx}'] = float(value)
            spectra_preview_rows.append(row)

        for model_spec in run['models']:
            _set_global_seeds(0)
            abundances_t, metadata = run_registered_model(
                model_name=model_spec['name'],
                endmembers=endmembers_t,
                pixels=pixels_t,
                transform=run['transform'],
                params=model_spec['params'],
            )
            abundances = abundances_t.detach().cpu().numpy()
            rmse = float(np.sqrt(np.mean((abundances - true_abundances) ** 2)))
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
                    'last_active_pixels': metadata.get('last_active_pixels', pixels.shape[0]),
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
    model_dir = output_dir / 'model_evaluation'
    model_dir.mkdir(parents=True, exist_ok=True)
    model_summary_path = model_dir / 'model_summary.csv'
    abundance_preview_path = model_dir / 'abundance_preview.csv'
    spectra_preview_path = model_dir / 'spectra_preview.csv'
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(model_rows).to_csv(model_summary_path, index=False)
    pd.DataFrame(abundance_preview_rows).to_csv(abundance_preview_path, index=False)
    pd.DataFrame(spectra_preview_rows).to_csv(spectra_preview_path, index=False)
    return {
        'run_name': exp.run_name,
        'output_dir': str(output_dir),
        'summary_path': str(summary_path),
        'n_runs': len(runs),
        'model_evaluation': {
            'n_runs': len(runs),
            'summary_path': str(model_summary_path),
            'abundance_preview_path': str(abundance_preview_path),
            'spectra_preview_path': str(spectra_preview_path),
        },
    }


