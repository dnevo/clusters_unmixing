import numpy as np

from clusters_unmixing.config.schema import ModelRunConfig
from clusters_unmixing.transforms.spectral_views import select_wavelength_ranges


def test_select_wavelength_ranges_supports_range_um_mean_reduction() -> None:
    """Object bands entries with reduce='mean' should collapse each segment."""

    wavelengths = np.asarray([0.4, 0.5, 0.6, 0.7], dtype=float)
    signatures = np.asarray(
        [
            [1.0, 10.0],
            [3.0, 30.0],
            [5.0, 50.0],
            [7.0, 70.0],
        ],
        dtype=float,
    )

    selected_wavelengths, selected_signatures, segment_lengths = select_wavelength_ranges(
        wavelengths=wavelengths,
        signatures=signatures,
        bands_ranges=[
            {"range_Âµm": [0.4, 0.6], "reduce": "mean"},
            {"range_Âµm": [0.7, 0.7], "reduce": "none"},
        ],
    )

    np.testing.assert_allclose(selected_wavelengths, np.asarray([0.5, 0.7], dtype=float))
    np.testing.assert_allclose(
        selected_signatures,
        np.asarray(
            [
                [3.0, 30.0],
                [7.0, 70.0],
            ],
            dtype=float,
        ),
    )
    assert segment_lengths == [1, 1]


def test_model_run_config_accepts_range_um_entries() -> None:
    """Model run config should parse object-form bands ranges."""

    run = ModelRunConfig.from_dict(
        {
            "cluster_set": "set_a",
            "bands_ranges": [
                {"range_Âµm": [0.43, 0.569], "reduce": "mean"},
                {"range_Âµm": [0.65, 0.809], "reduce": "none"},
            ],
            "normalization": "without",
            "transform": "raw",
            "num_pixels": 10,
            "snr_db": 30,
            "models": ["sunsal"],
        }
    )

    assert run.normalized_bands_ranges() == [
        (0.43, 0.569, "mean"),
        (0.65, 0.809, "none"),
    ]
    assert run.serialized_bands_ranges() == [
        {"range_Âµm": [0.43, 0.569], "reduce": "mean"},
        {"range_Âµm": [0.65, 0.809], "reduce": "none"},
    ]

