from pathlib import Path

import pandas as pd

from clusters_unmixing.utils.diagnostics import build_model_run_comparisons


def test_build_model_run_comparisons_filters_by_snr(tmp_path: Path) -> None:
    """Runs that differ only by SNR must produce run-specific comparison values."""

    config_path = tmp_path / "config.json"
    summary_path = tmp_path / "model_summary.csv"

    config_path.write_text(
        """
{
  "cluster_sets": [{"name": "set_a", "path": "data/6clusters_thomas.csv"}],
  "model_evaluation": {
    "models": [{"name": "sunsal", "params": {}}],
    "runs": [
      {
        "cluster_set": "set_a",
        "bands_ranges": [[0.4, 2.4025]],
        "normalization": "without",
        "transform": {"steps": []},
        "num_pixels": 16,
        "snr_db": 10.0,
        "models": ["sunsal"]
      },
      {
        "cluster_set": "set_a",
        "bands_ranges": [[0.4, 2.4025]],
        "normalization": "without",
        "transform": {"steps": []},
        "num_pixels": 16,
        "snr_db": 30.0,
        "models": ["sunsal"]
      }
    ]
  }
}
""".strip(),
        encoding="utf-8",
    )

    summary_df = pd.DataFrame(
        [
            {
                "cluster_set": "set_a",
                "bands_ranges": "[[0.4,2.4025]]",
                "normalization": "without",
                "transform": {"steps": []},
                "snr_db": 10.0,
                "model": "sunsal",
                "metric": "abundance_rmse",
                "mean": 0.111,
            },
            {
                "cluster_set": "set_a",
                "bands_ranges": "[[0.4,2.4025]]",
                "normalization": "without",
                "transform": {"steps": []},
                "snr_db": 30.0,
                "model": "sunsal",
                "metric": "abundance_rmse",
                "mean": 0.333,
            },
        ]
    )
    summary_df.to_csv(summary_path, index=False)

    run_tables = build_model_run_comparisons(
        config_path=config_path,
        model_summary_path=summary_path,
    )

    assert len(run_tables) == 2
    first_value = float(run_tables[0]["comparison"].loc["abundance_rmse", "sunsal"])
    second_value = float(run_tables[1]["comparison"].loc["abundance_rmse", "sunsal"])

    assert first_value == 0.111
    assert second_value == 0.333


from pydantic import ValidationError
from clusters_unmixing.config.schema import ModelRunConfig


def test_model_run_config_requires_num_pixels_and_snr_db() -> None:
    try:
        ModelRunConfig.model_validate({
            "cluster_set": "set_a",
            "bands_ranges": [[0.4, 2.4025]],
            "normalization": "without",
            "transform": {"steps": []},
            "models": ["sunsal"],
        })
    except ValidationError as exc:
        message = str(exc)
        assert "num_pixels" in message
        assert "snr_db" in message
    else:
        raise AssertionError("Expected ValidationError for missing snr_db")


