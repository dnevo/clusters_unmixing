from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from clusters_unmixing.config import ExperimentConfig
from clusters_unmixing.config.schema import BandRangeSpec


def resolve_cluster_path(exp: ExperimentConfig, cluster_path: str) -> Path:
    return Path(exp.config_dir) / cluster_path


def bands_ranges_key(bands_ranges: Sequence[BandRangeSpec]) -> str:
    payload: list[dict[str, Any]] = [
        {
            "range_µm": [x_min, x_max],
            "reduce": reduce,
        }
        for x_min, x_max, reduce in bands_ranges
    ]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
