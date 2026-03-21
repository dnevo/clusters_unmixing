from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from clusters_unmixing.config import ExperimentConfig
from clusters_unmixing.config.schema import BandRangeSpec


def resolve_cluster_path(exp: ExperimentConfig, cluster_path: str) -> Path:
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


def bands_ranges_key(bands_ranges: Sequence[BandRangeSpec]) -> str:
    payload: list[dict[str, Any]] = [
        {
            "range_µm": [x_min, x_max],
            "reduce": reduce,
        }
        for x_min, x_max, reduce in bands_ranges
    ]
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)