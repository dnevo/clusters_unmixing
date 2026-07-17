from __future__ import annotations

from asyncio import run
import itertools
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from clusters_unmixing.config.schema import ExperimentConfig
from clusters_unmixing.core_math import apply_snr_noise, compute_correlation_matrix, mix_pixels, rmse, summarize_correlation_matrix
from clusters_unmixing.data import generate_samples
from clusters_unmixing.dataio import load_wavelength_and_cluster_matrix
from clusters_unmixing.models.runner_registry import run_registered_model
from clusters_unmixing.preprocessing.normalization import apply_normalization
from clusters_unmixing.preprocessing.band_selection import select_wavelength_ranges

def _expand_param_grid(params: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a {param_name: [values]} mapping. No keys -> one empty combination."""
    if not params:
        return [{}]
    keys = list(params)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(params[k] for k in keys))]


def _model_label(name: str, overrides: dict[str, Any]) -> str:
    """Output-facing model identity: bare name when unswept, disambiguated with its
    overridden param values when swept, so repeated executions of the same model
    don't collide in a run's output rows."""
    if not overrides:
        return name
    return f"{name}[{','.join(f'{k}={v}' for k, v in sorted(overrides.items()))}]"


def _run_overrides_label(overrides: dict[str, Any]) -> str:
    if not overrides:
        return "-"
    return ",".join(f"{k}={v}" for k, v in sorted(overrides.items()))


def _planned_model_runs(exp: ExperimentConfig) -> list[dict[str, Any]]:
    model_params = {model.name: dict(model.params) for model in exp.model_evaluation.models}
    planned: list[dict[str, Any]] = []

    for parent_index, item in enumerate(exp.model_evaluation.runs, start=1):
        base = {
            "cluster_set": item.cluster_set,
            "bands_ranges": item.normalized_bands_ranges(),
            "normalization": item.normalization,
            "num_pixels": item.num_pixels,
            "snr_db": item.snr_db,
            "nonlinearity_gamma": item.nonlinearity_gamma,
        }

        for run_overrides in _expand_param_grid(item.sweep_params):
            models = [
                {
                    "name": entry.name,
                    "params": {**model_params[entry.name], **model_overrides},
                    "label": _model_label(entry.name, model_overrides),
                }
                for entry in item.models
                for model_overrides in _expand_param_grid(entry.param_sweep)
            ]
            planned.append({
                **base,
                **run_overrides,
                "parent_run_index": parent_index,
                "run_overrides": run_overrides,
                "models": models,
            })

    return planned

def _build_stage_projections(
    run: dict[str, Any],
    wavelengths: np.ndarray,
    raw_endmembers: np.ndarray,
    raw_pixels: np.ndarray,
) -> list[tuple[str, np.ndarray, np.ndarray]]:

    wavelengths_sel, endmembers_sel = select_wavelength_ranges(
        wavelengths=wavelengths,
        spectra=raw_endmembers,
        band_ranges=run["bands_ranges"],
    )
    wavelengths_sel, pixels_sel = select_wavelength_ranges(
        wavelengths=wavelengths,
        spectra=raw_pixels,
        band_ranges=run["bands_ranges"],
    )
    projections: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("raw", endmembers_sel, pixels_sel)
    ]

    projected_endmembers = apply_normalization(
        spectra=endmembers_sel, 
        wavelengths=wavelengths_sel, 
        normalization=run["normalization"]
    )
    
    projected_pixels = apply_normalization(
        spectra=pixels_sel, 
        wavelengths=wavelengths_sel, 
        normalization=run["normalization"]
    )

    projections.append(("normalized", projected_endmembers, projected_pixels))

    return projections

def _set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_held_out_test_indices(n_pixels: int, test_fraction: float = 0.1) -> np.ndarray:
    """Pick one held-out pixel subset per run, shared by every model.

    Iterative solvers (sunsal/vpgdu) never see labels, so scoring them on all
    pixels is fine; but small_mlp/convnext1d train on ~90% of the same pixels,
    so scoring everyone on all pixels would let those two "cheat" on the
    reported metric. Using the same held-out subset for every model keeps the
    comparison apples-to-apples.
    """
    test_count = int(n_pixels * test_fraction)
    return np.random.permutation(n_pixels)[:test_count]


def _make_synthetic_pixels(
    endmembers: np.ndarray,
    num_pixels: int,
    snr_db: float,
    nonlinearity_gamma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic abundances and their resulting noisy pixel spectra.

    Pixels are synthesized from abundances via the Generalized Bilinear Model
    (see ``core_math.mix_pixels``); ``nonlinearity_gamma=0.0`` keeps the standard
    linear mixing model. Additive noise is then applied at the requested SNR.
    """
    n_endmembers = endmembers.shape[0]
    abundances = generate_samples(
        num_samples=num_pixels,
        max_non_zero_endmembers=n_endmembers,
        num_endmembers=n_endmembers,
    )
    clean_pixels = mix_pixels(abundances, endmembers, nonlinearity_gamma)
    noisy_pixels, _ = apply_snr_noise(clean_pixels, snr_db)
    return noisy_pixels, abundances


def _abundance_preview_row(
    run_index: int,
    parent_run_index: int,
    run_overrides: str,
    pixel_index: int,
    source: str,
    abundances: np.ndarray,
    abundance_rmse: float,
    reconstruction_rmse: float,
) -> dict[str, Any]:
    return {
        'run_index': run_index,
        'parent_run_index': parent_run_index,
        'run_overrides': run_overrides,
        'pixel_index': pixel_index,
        'source': source,
        'abundance_rmse': abundance_rmse,
        'reconstruction_rmse': reconstruction_rmse,
        **{f'endmember_{j}': float(value) for j, value in enumerate(abundances, start=1)},
    }

def run_experiments(exp: ExperimentConfig) -> dict[str, Any]:
    output_dir = exp.experiment_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = _planned_model_runs(exp)

    correlation_summary_rows: list[dict[str, Any]] = []
    model_summary_rows: list[dict[str, Any]] = []
    abundance_preview_rows: list[dict[str, Any]] = []

    for idx, run in enumerate(runs, start=1):
        parent_run_index = run["parent_run_index"]
        run_overrides_label = _run_overrides_label(run["run_overrides"])
        run_overrides_suffix = f" ({run_overrides_label})" if run["run_overrides"] else ""

        cluster_cfg = next(item for item in exp.cluster_sets if item.name == run["cluster_set"])
        cluster_path = exp.resolve_path(cluster_cfg.path)
        wavelengths, raw_endmembers = load_wavelength_and_cluster_matrix(cluster_path)

        _set_global_seeds(0)
        raw_pixels, true_abundances = _make_synthetic_pixels(
            raw_endmembers,
            run["num_pixels"],
            run["snr_db"],
            run["nonlinearity_gamma"],
        )

        stage_projections = _build_stage_projections(
            run,
            wavelengths,
            raw_endmembers,
            raw_pixels,
        )
        _, projected_endmembers, projected_pixels = stage_projections[-1]
        test_indices = _make_held_out_test_indices(projected_pixels.shape[0])
        for metric_name in exp.metrics:
            for stage_name, stage_endmembers, _ in stage_projections:
                matrix = compute_correlation_matrix(stage_endmembers, metric_name)
                correlation_summary_rows.append({
                    'run_index': idx,
                    'parent_run_index': parent_run_index,
                    'run_overrides': run_overrides_label,
                    'metric': metric_name,
                    'stage': stage_name,
                    **summarize_correlation_matrix(matrix),
                })


        preview_pixels = np.random.choice(true_abundances.shape[0], size=5, replace=False).tolist()
        preview_pixels.sort()
        abundance_preview_rows.extend(
            _abundance_preview_row(
                idx,
                parent_run_index,
                run_overrides_label,
                sample_idx,
                'true',
                true_abundances[sample_idx],
                0.0,
                rmse(true_abundances[sample_idx] @ projected_endmembers, projected_pixels[sample_idx]),
            )
            for sample_idx in preview_pixels
        )

        for model_spec in run['models']:
            abundances, _ = run_registered_model(
                model_name=model_spec['name'],
                endmembers=projected_endmembers,
                pixels=projected_pixels,
                true_abundances=true_abundances,
                params=model_spec['params'],
                test_indices=test_indices,
                run_label=f"Run {idx}/{len(runs)}{run_overrides_suffix} | model={model_spec['label']}",
            )
            # Score every model on the same held-out pixels (see
            # _make_held_out_test_indices) rather than the full pixel set, so
            # models that trained on some of these pixels (small_mlp,
            # convnext1d) aren't compared unfairly against ones that didn't
            # (sunsal, vpgdu).
            test_abundances_pred = abundances[test_indices]
            test_true_abundances = true_abundances[test_indices]
            test_projected_pixels = projected_pixels[test_indices]
            reconstructed_pixels = test_abundances_pred @ projected_endmembers
            for metric_name, value in [
                ('abundance_rmse', rmse(test_abundances_pred, test_true_abundances)),
                ('reconstruction_rmse', rmse(reconstructed_pixels, test_projected_pixels)),
            ]:
                model_summary_rows.append({
                    'run_index': idx,
                    'parent_run_index': parent_run_index,
                    'run_overrides': run_overrides_label,
                    'model': model_spec['label'],
                    'metric': metric_name,
                    'mean': value,
                })
            for sample_idx in preview_pixels:
                pred_abundances = abundances[sample_idx]
                abundance_preview_rows.append(
                    _abundance_preview_row(
                        idx,
                        parent_run_index,
                        run_overrides_label,
                        sample_idx,
                        model_spec['label'],
                        pred_abundances,
                        rmse(pred_abundances, true_abundances[sample_idx]),
                        rmse(pred_abundances @ projected_endmembers, projected_pixels[sample_idx]),
                    )
                )

    correlation_summary_path = output_dir / 'correlation_summary.csv'
    model_summary_path = output_dir / 'model_summary.csv'
    abundance_preview_path = output_dir / 'abundance_preview.csv'
    pd.DataFrame(correlation_summary_rows).to_csv(correlation_summary_path, index=False, float_format='%.6f')

    model_summary_df = pd.DataFrame(model_summary_rows)
    run_meta = model_summary_df[['run_index', 'parent_run_index', 'run_overrides']].drop_duplicates('run_index')
    model_pivot = model_summary_df.pivot(index=['run_index', 'metric'], columns='model', values='mean').reset_index()
    model_cols = [c for c in model_pivot.columns if c not in ('run_index', 'metric')]
    model_pivot.merge(run_meta, on='run_index', how='left')[
        ['run_index', 'parent_run_index', 'run_overrides', 'metric'] + model_cols
    ].to_csv(model_summary_path, index=False, float_format='%.6f')
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
