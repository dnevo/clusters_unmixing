import numpy as np
import pytest

from clusters_unmixing.config.schema import ModelRunConfig
from clusters_unmixing.transforms.spectral_views import apply_transform


def _base_run_payload() -> dict[str, object]:
    return {
        "cluster_set": "set_a",
        "bands_ranges": [[0.4, 2.4]],
        "normalization": "without",
        "models": ["sunsal"],
        "num_pixels": 16,
        "snr_db": 30.0,
    }


def test_model_run_config_supports_transform_steps_with_pca() -> None:
    payload = {
        **_base_run_payload(),
        "transform": {
            "steps": [
                {"name": "first_derivative"},
                {"name": "pca", "params": {"n_components": 2}},
            ]
        },
    }

    run_cfg = ModelRunConfig.from_dict(payload)

    assert run_cfg.normalized_transform_steps() == [
        ("first_derivative", {}),
        ("pca", {"n_components": 2}),
    ]
    assert run_cfg.normalized_transform() == "first_derivative+pca(n_components=2)"


def test_model_run_config_rejects_non_integer_pca_n_components() -> None:
    payload = {
        **_base_run_payload(),
        "transform": {
            "steps": [
                {"name": "pca", "params": {"n_components": 0.95}},
            ]
        },
    }

    run_cfg = ModelRunConfig.from_dict(payload)
    with pytest.raises(ValueError, match="n_components must be an integer"):
        run_cfg.normalized_transform_steps()


def test_model_run_config_rejects_extra_pca_params() -> None:
    payload = {
        **_base_run_payload(),
        "transform": {
            "steps": [
                {
                    "name": "pca",
                    "params": {
                        "n_components": 2,
                        "whiten": True,
                    },
                },
            ]
        },
    }

    run_cfg = ModelRunConfig.from_dict(payload)
    with pytest.raises(ValueError, match="only supports params.n_components"):
        run_cfg.normalized_transform_steps()


def test_apply_transform_pca_reduces_band_axis() -> None:
    signatures = np.asarray(
        [
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.4, 0.5, 0.6],
            [0.6, 0.7, 0.8],
        ],
        dtype=np.float64,
    )

    transformed = apply_transform(
        signatures,
        kind="pca",
        params={"n_components": 2},
    )

    assert transformed.shape == (2, 3)


def test_apply_transform_pca_requires_integer_n_components() -> None:
    signatures = np.asarray(
        [
            [0.1, 0.2],
            [0.2, 0.3],
            [0.4, 0.5],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="n_components"):
        apply_transform(signatures, kind="pca", params={"n_components": 1.0})

