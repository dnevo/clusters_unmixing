from pathlib import Path
import sys

import pytest
import torch
from IPython.display import display

from clusters_unmixing.pipelines import run_correlation_experiments
from clusters_unmixing.pipelines import correlation_pipeline as pipeline
from clusters_unmixing.utils import (
    build_model_run_diagnostics,
    display_projection_reflectance,
    display_spectra_preview_plots,
)


def test_run_correlation_experiments_notebook_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mirror the diagnostics notebook model-results code path end-to-end."""

    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root / "notebooks")
    monkeypatch.setattr(sys.modules[__name__], "display_projection_reflectance", lambda _groups: None)

    def _fast_model_runner(
        *,
        model_name: str,
        endmembers: torch.Tensor,
        pixels: torch.Tensor,
        transform: str,
        params: dict[str, object],
    ) -> tuple[torch.Tensor, dict[str, int]]:
        del model_name, transform, params
        n_pixels = int(pixels.shape[0])
        n_endmembers = int(endmembers.shape[1])
        abundances = torch.full(
            (n_pixels, n_endmembers),
            1.0 / max(n_endmembers, 1),
            dtype=endmembers.dtype,
            device=endmembers.device,
        )
        return abundances, {
            "iterations_logged": 1,
            "last_active_pixels": n_pixels,
        }

    monkeypatch.setattr(pipeline, "run_registered_model", _fast_model_runner)

    config_path = Path("../experiments/configs/correlation_options.json")
    result = run_correlation_experiments(config_path)

    run_diagnostics = build_model_run_diagnostics(
        config_path=config_path,
        correlation_summary_path=result["summary_path"],
        model_summary_path=result["model_evaluation"]["summary_path"],
        abundance_preview_path=result["model_evaluation"].get("abundance_preview_path"),
        spectra_preview_path=result["model_evaluation"].get("spectra_preview_path"),
    )

    for run_payload in run_diagnostics:
        print(
            f"Run {run_payload['run_index']}/{run_payload['run_count']} | "
            f"set={run_payload['cluster_set']} | "
            f"normalization={run_payload['normalization']} | "
            f"transform={run_payload['transform']}"
        )

        group = run_payload.get("projection_group")
        if group is not None:
            display_projection_reflectance([group])

        comparison = run_payload["comparison"]
        display(comparison)

        abundance_table = run_payload.get("abundance_table")
        display(abundance_table)

        display_spectra_preview_plots(
            run_payload.get("spectra_preview"),
            models=run_payload.get("models"),
        )

        spectra_preview = run_payload.get("spectra_preview")
        if spectra_preview is not None and not spectra_preview.empty:
            band_cols = [
                col for col in spectra_preview.columns if str(col).startswith("band_")
            ]
            assert len(band_cols) > 5

        print()

    assert result["n_runs"] > 0
    assert result["model_evaluation"]["enabled"] is True
    assert len(run_diagnostics) > 0


def test_run_correlation_experiments_requires_model_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No model-evaluation runs should return a disabled stage result."""

    class _FakeExperiment:
        run_name = "unit_test_run"
        cluster_sets = [type("ClusterSet", (), {"name": "cluster_a"})()]
        metrics = ["cosine"]

    fake_experiment = _FakeExperiment()

    monkeypatch.setattr(
        pipeline.ExperimentConfig,
        "from_json_file",
        staticmethod(lambda _config_path: fake_experiment),
    )
    monkeypatch.setattr(pipeline, "_resolve_output_dir", lambda _exp: Path("."))
    monkeypatch.setattr(pipeline, "_planned_model_runs", lambda _exp: [])

    result = pipeline.run_correlation_experiments("dummy_config.json")
    assert result["model_evaluation"]["enabled"] is False
    assert result["model_evaluation"]["n_runs"] == 0
