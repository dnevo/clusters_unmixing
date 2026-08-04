from __future__ import annotations

from asyncio import run
from datetime import datetime
import itertools
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch

from clusters_unmixing.config.schema import ExperimentConfig
from clusters_unmixing.core_math import apply_snr_noise, compute_condition_number, compute_cosine_similarity_matrix, mix_pixels, rmse, summarize_cosine_similarity_matrix
from clusters_unmixing.data import generate_samples
from clusters_unmixing.dataio import load_wavelength_and_cluster_matrix
from clusters_unmixing.models.runner_registry import run_registered_model
from clusters_unmixing.preprocessing.normalization import apply_normalization
from clusters_unmixing.preprocessing.band_selection import select_wavelength_ranges
from clusters_unmixing.wandb_logging import log_batch_artifacts, log_model_run

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
    model_params = {model.name: dict(model.params) for model in exp.models}
    planned: list[dict[str, Any]] = []

    for parent_index, item in enumerate(exp.runs, start=1):
        base = {
            "cluster_set": item.cluster_set,
            "bands_ranges": item.normalized_bands_ranges(),
            "normalization": item.normalization,
            "abundance_distribution": item.abundance_distribution,
            "dirichlet_alpha": item.dirichlet_alpha,
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

    if run["normalization"] != "none":
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

def _set_global_seeds(seed: int, extra_reproducibility: bool = False, skip_strict_determinism: bool = False) -> None:
    """Reset every RNG this project touches, so a run's synthetic data and model
    training are reproducible given the same seed and config.

    Called once per experiment run (not once per process), so each run's
    synthetic-pixel generation is independently reproducible regardless of run
    order or how many runs came before it. ``torch.manual_seed`` also reseeds
    CUDA's generators, so no separate ``torch.cuda.manual_seed_all`` call is
    needed.

    ``extra_reproducibility=True`` additionally asks for bit-exact CUDA
    determinism, following PyTorch's own reproducibility recipe
    (https://pytorch.org/docs/stable/notes/randomness.html), at a performance
    cost:
    - ``CUBLAS_WORKSPACE_CONFIG`` and ``PYTHONHASHSEED`` are read once, before
      CUDA/interpreter start-up - setting them here only has any effect
      because this function's first call (run 1, before any model has
      touched CUDA) is also the first thing in the whole process that touches
      CUDA. Re-setting them on later runs is a harmless no-op, not a fresh
      application - don't move CUDA-touching code ahead of this call without
      re-checking that assumption.
    - ``torch.use_deterministic_algorithms(True)`` makes PyTorch raise instead
      of silently falling back to a non-deterministic algorithm. The mamba
      model's fused CUDA kernels (causal_conv1d/mamba_ssm) don't register a
      deterministic implementation, so strict mode would make any run
      containing mamba crash. Pass ``skip_strict_determinism=True`` for those
      runs - see the caller in ``run_experiments()``, which checks the run's
      model list.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if extra_reproducibility:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(not skip_strict_determinism)


def _make_held_out_test_indices(n_pixels: int, test_fraction: float = 0.1) -> np.ndarray:
    """Pick one held-out pixel subset per run, shared by every model.

    Iterative solvers (sunsal/vpgdu) never see labels, so scoring them on all
    pixels is fine; but mlp/convnext1d train on ~90% of the same pixels,
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
    abundance_distribution: str = "grid",
    dirichlet_alpha: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic abundances and their resulting noisy pixel spectra.

    Pixels are synthesized from the configured abundance distribution and then
    mixed via the Generalized Bilinear Model (see ``core_math.mix_pixels``);
    ``nonlinearity_gamma=0.0`` keeps the standard linear mixing model. Additive
    noise is then applied at the requested SNR.
    """
    n_endmembers = endmembers.shape[0]
    abundances = generate_samples(
        num_samples=num_pixels,
        max_non_zero_endmembers=n_endmembers,
        num_endmembers=n_endmembers,
        abundance_distribution=abundance_distribution,
        dirichlet_alpha=dirichlet_alpha,
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

    # Groups every W&B run this batch logs (one per model call, plus the
    # summary run at the end) so they stay browsable together in the W&B UI.
    wandb_group = f"{exp.experiment_name}_{datetime.now():%Y%m%d-%H%M%S}"

    cosine_similarity_summary_rows: list[dict[str, Any]] = []
    model_summary_rows: list[dict[str, Any]] = []
    abundance_preview_rows: list[dict[str, Any]] = []

    for idx, run in enumerate(runs, start=1):
        parent_run_index = run["parent_run_index"]
        run_overrides_label = _run_overrides_label(run["run_overrides"])
        run_overrides_suffix = f" ({run_overrides_label})" if run["run_overrides"] else ""

        cluster_cfg = next(item for item in exp.cluster_sets if item.name == run["cluster_set"])
        cluster_path = exp.resolve_path(cluster_cfg.path)
        wavelengths, raw_endmembers = load_wavelength_and_cluster_matrix(cluster_path)

        run_has_mamba = any(model_spec["name"] == "mamba" for model_spec in run["models"])
        _set_global_seeds(0, extra_reproducibility=exp.extra_reproducibility, skip_strict_determinism=run_has_mamba)
        raw_pixels, true_abundances = _make_synthetic_pixels(
            raw_endmembers,
            run["num_pixels"],
            run["snr_db"],
            run["nonlinearity_gamma"],
            run["abundance_distribution"],
            run["dirichlet_alpha"],
        )

        stage_projections = _build_stage_projections(
            run,
            wavelengths,
            raw_endmembers,
            raw_pixels,
        )
        _, projected_endmembers, projected_pixels = stage_projections[-1]
        test_indices = _make_held_out_test_indices(projected_pixels.shape[0])
        for stage_name, stage_endmembers, _ in stage_projections:
            matrix = compute_cosine_similarity_matrix(stage_endmembers)
            cosine_similarity_summary_rows.append({
                'run_index': idx,
                'parent_run_index': parent_run_index,
                'run_overrides': run_overrides_label,
                'stage': stage_name,
                'condition_number': compute_condition_number(stage_endmembers),
                **summarize_cosine_similarity_matrix(matrix),
            })


        preview_pixels = np.random.choice(true_abundances.shape[0], size=2, replace=False).tolist()
        preview_pixels.sort()
        true_preview_rows = [
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
        ]
        abundance_preview_rows.extend(true_preview_rows)

        for model_spec in run['models']:
            abundances, diagnostics = run_registered_model(
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
            # models that trained on some of these pixels (mlp,
            # convnext1d) aren't compared unfairly against ones that didn't
            # (sunsal, vpgdu).
            test_abundances_pred = abundances[test_indices]
            test_true_abundances = true_abundances[test_indices]
            test_projected_pixels = projected_pixels[test_indices]
            # Always a plain linear reconstruction from the returned abundances -
            # deliberately the same formula for every model, including nonlinear ones
            # (gbm, mlm), so reconstruction_rmse is comparable apples-to-apples. This
            # means it does NOT credit gbm/mlm for their own internally-fitted
            # nonlinear correction term (per-pair gamma_ij, or scalar P - see their
            # docstrings), so on nonlinearity_gamma>0 runs it shows a nonzero floor
            # even for a perfectly recovered abundance vector; it's testing "do these
            # abundances explain the pixel under plain LMM", not "how good was the
            # model's own nonlinear fit".
            reconstructed_pixels = test_abundances_pred @ projected_endmembers
            model_metrics = {
                'abundance_rmse': rmse(test_abundances_pred, test_true_abundances),
                'reconstruction_rmse': rmse(reconstructed_pixels, test_projected_pixels),
            }
            for metric_name, value in model_metrics.items():
                model_summary_rows.append({
                    'run_index': idx,
                    'parent_run_index': parent_run_index,
                    'run_overrides': run_overrides_label,
                    'model': model_spec['label'],
                    'metric': metric_name,
                    'mean': value,
                })
            model_preview_rows = []
            for sample_idx in preview_pixels:
                pred_abundances = abundances[sample_idx]
                row = _abundance_preview_row(
                    idx,
                    parent_run_index,
                    run_overrides_label,
                    sample_idx,
                    model_spec['label'],
                    pred_abundances,
                    rmse(pred_abundances, true_abundances[sample_idx]),
                    rmse(pred_abundances @ projected_endmembers, projected_pixels[sample_idx]),
                )
                abundance_preview_rows.append(row)
                model_preview_rows.append(row)

            if exp.use_wandb:
                log_model_run(
                    project=exp.experiment_name,
                    group=wandb_group,
                    run_name=f"run{idx}-{model_spec['label']}",
                    config={
                        'run_index': idx,
                        'parent_run_index': parent_run_index,
                        'run_overrides': run['run_overrides'],
                        'cluster_set': run['cluster_set'],
                        'normalization': run['normalization'],
                        'abundance_distribution': run['abundance_distribution'],
                        'dirichlet_alpha': run['dirichlet_alpha'],
                        'num_pixels': run['num_pixels'],
                        'snr_db': run['snr_db'],
                        'nonlinearity_gamma': run['nonlinearity_gamma'],
                        'model_name': model_spec['name'],
                        'model_label': model_spec['label'],
                        **model_spec['params'],
                    },
                    metrics={**model_metrics, **diagnostics},
                    abundance_preview_rows=true_preview_rows + model_preview_rows,
                )

    cosine_similarity_summary_path = output_dir / 'cosine_similarity_summary.csv'
    model_summary_path = output_dir / 'model_summary.csv'
    abundance_preview_path = output_dir / 'abundance_preview.csv'
    pd.DataFrame(cosine_similarity_summary_rows).to_csv(cosine_similarity_summary_path, index=False, float_format='%.6f')

    model_summary_df = pd.DataFrame(model_summary_rows)
    run_meta = model_summary_df[['run_index', 'parent_run_index', 'run_overrides']].drop_duplicates('run_index')
    model_pivot = model_summary_df.pivot(index=['run_index', 'metric'], columns='model', values='mean').reset_index()
    # pivot() sorts columns alphabetically; reorder to first-appearance order
    # in model_summary_rows, which follows each run's `models` list as given
    # in the experiment YAML, so the CSV's model columns read left-to-right
    # in the same order the config lists them.
    model_order = list(dict.fromkeys(model_summary_df['model']))
    model_cols = [c for c in model_order if c in model_pivot.columns]
    model_pivot.merge(run_meta, on='run_index', how='left')[
        ['run_index', 'parent_run_index', 'run_overrides', 'metric'] + model_cols
    ].to_csv(model_summary_path, index=False, float_format='%.6f')
    abundance_preview_df = pd.DataFrame(abundance_preview_rows).assign(
        source_order=lambda df: (df['source'] != 'true').astype(int)
    )
    abundance_preview_df.sort_values(['run_index', 'pixel_index', 'source_order', 'source']).drop(
        columns='source_order'
    ).to_csv(abundance_preview_path, index=False, float_format='%.6f')

    if exp.use_wandb:
        log_batch_artifacts(
            project=exp.experiment_name,
            group=wandb_group,
            run_name=f"{exp.experiment_name}-summary",
            artifact_name=f"{exp.experiment_name}-outputs",
            csv_paths={
                'cosine_similarity_summary': cosine_similarity_summary_path,
                'model_summary': model_summary_path,
                'abundance_preview': abundance_preview_path,
            },
        )

    return {
        'experiment_name': exp.experiment_name,
        'output_dir': str(output_dir),
        'cosine_similarity_summary_path': str(cosine_similarity_summary_path),
        'n_runs': len(runs),
        'model_evaluation': {
            'n_runs': len(runs),
            'model_summary_path': str(model_summary_path),
            'abundance_preview_path': str(abundance_preview_path),
        },
    }
