from __future__ import annotations

import hashlib
import json
import yaml

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BandRangeSpec = tuple[float, float, str]
TransformStepSpec = tuple[str, dict[str, Any]]


def _projection_name(bands_ranges: list[BandRangeSpec], transform: str, normalization: str) -> str:
    transform_suffix = "raw" if transform == "raw" else hashlib.sha1(transform.encode("utf-8")).hexdigest()[:8]
    normalization_suffix = "nq" if normalization == "with_quadratic" else "nw"
    ranges_key = json.dumps(_serialize_bands_ranges_for_config(bands_ranges), separators=(",", ":"), ensure_ascii=False)
    ranges_hash = hashlib.sha1(ranges_key.encode("utf-8")).hexdigest()[:8]
    return f"br_{ranges_hash}_{transform_suffix}_{normalization_suffix}"


def _serialize_bands_ranges_for_config(bands_ranges: list[BandRangeSpec]) -> list[Any]:
    if all(reduce == "none" for _, _, reduce in bands_ranges):
        return [[x_min, x_max] for x_min, x_max, _ in bands_ranges]
    return [{"range_µm": [x_min, x_max], "reduce": reduce} for x_min, x_max, reduce in bands_ranges]


class ClusterSetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: str


class ModelSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    def normalized_name(self) -> str:
        name = self.name.strip().lower()
        if not name:
            raise ValueError("Model entry requires non-empty 'name'")
        return name

    @classmethod
    def from_raw(cls, raw: Any) -> "ModelSpecConfig":
        if isinstance(raw, str):
            return cls(name=raw, params={})
        return cls.model_validate(raw)


class BandRangeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    range_um: tuple[float, float] = Field(alias="range_µm")
    reduce: Literal["none", "mean"] = "none"

    @model_validator(mode="after")
    def _check_order(self) -> "BandRangeModel":
        if self.range_um[0] > self.range_um[1]:
            raise ValueError("x_min must be <= x_max")
        return self

    def to_spec(self) -> BandRangeSpec:
        return (float(self.range_um[0]), float(self.range_um[1]), str(self.reduce))


class TransformStepModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["first_derivative", "pca"]
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_params(self) -> "TransformStepModel":
        if self.name == "first_derivative":
            if self.params:
                raise ValueError("first_derivative does not accept params")
        elif self.name == "pca":
            if set(self.params) != {"n_components"}:
                raise ValueError("pca only supports params.n_components")
            n_components = self.params.get("n_components")
            if isinstance(n_components, bool) or not isinstance(n_components, int):
                raise ValueError("n_components must be an integer")
            if n_components <= 0:
                raise ValueError("n_components must be > 0")
        return self

    def to_spec(self) -> TransformStepSpec:
        if self.name == "pca":
            return (self.name, {"n_components": int(self.params["n_components"])})
        return (self.name, {})


class ModelRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cluster_set: str
    bands_ranges: list[Any]
    normalization: str = "without"
    transform: Any = Field(default_factory=lambda: {"steps": []})
    models: list[str]
    num_pixels: int
    snr_db: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelRunConfig":
        return cls.model_validate(raw)

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Model run 'models' must be non-empty")
        normalized = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("Model run 'models' entries must be non-empty strings")
            normalized.append(item)
        return normalized


    @field_validator("snr_db", mode="before")
    @classmethod
    def _validate_snr_db(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"inf", "+inf", "infinity", "+infinity"}:
                return float("inf")
        return value

    def normalized_models(self) -> list[str]:
        return [m.strip().lower() for m in self.models]

    def normalized_bands_ranges(self) -> list[BandRangeSpec]:
        if not self.bands_ranges:
            raise ValueError("Model run 'bands_ranges' must be non-empty")
        result: list[BandRangeSpec] = []
        for item in self.bands_ranges:
            if isinstance(item, dict):
                result.append(BandRangeModel.model_validate(item).to_spec())
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                result.append(BandRangeModel(**{"range_µm": [item[0], item[1]], "reduce": "none"}).to_spec())
            else:
                raise ValueError("Model run 'bands_ranges' entries must be [x_min, x_max] or object form")
        return result

    def serialized_bands_ranges(self) -> list[Any]:
        return _serialize_bands_ranges_for_config(self.normalized_bands_ranges())

    def normalized_transform_steps(self) -> list[TransformStepSpec]:
        raw = self.transform
        if not isinstance(raw, dict) or "steps" not in raw or not isinstance(raw["steps"], list):
            raise ValueError("Model run 'transform' must be an object with a 'steps' list")
        steps = [TransformStepModel.model_validate(step).to_spec() for step in raw["steps"]]
        names = [name for name, _ in steps]
        if names.count("first_derivative") > 1:
            raise ValueError("first_derivative may appear at most once")
        if names.count("pca") > 1:
            raise ValueError("pca may appear at most once")
        if "pca" in names and "first_derivative" in names and names.index("pca") < names.index("first_derivative"):
            raise ValueError("first_derivative must appear before pca")
        return steps

    def normalized_transform(self) -> str:
        steps = self.normalized_transform_steps()
        if not steps:
            return "raw"
        labels = []
        for name, params in steps:
            if name == "first_derivative":
                labels.append(name)
            elif name == "pca":
                labels.append(f"pca(n_components={int(params['n_components'])})")
        return "+".join(labels)

    def normalized_normalization(self) -> str:
        value = self.normalization.strip().lower()
        if value not in {"without", "with_quadratic"}:
            raise ValueError("Model run 'normalization' must be one of: without, with_quadratic")
        return value

    def resolved_num_pixels(self) -> int:
        value = int(self.num_pixels)
        if value <= 0:
            raise ValueError("Model run 'num_pixels' must be > 0")
        return value

    def resolved_snr_db(self) -> float:
        return float(self.snr_db)

    def projection_name(self) -> str:
        return _projection_name(self.normalized_bands_ranges(), self.normalized_transform(), self.normalized_normalization())


class ModelEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    output_subdir: str = "model_evaluation"
    models: list[ModelSpecConfig] = Field(default_factory=list)
    runs: list[ModelRunConfig] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ModelEvaluationConfig":
        models = [ModelSpecConfig.from_raw(item) for item in raw.get("models", [])]
        runs = [ModelRunConfig.from_dict(item) for item in raw.get("runs", [])]
        payload = dict(raw)
        payload["models"] = models
        payload["runs"] = runs
        return cls.model_validate(payload)



class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str = "correlation_experiment"
    output_dir: str = "experiments/outputs"
    cluster_sets: list[ClusterSetConfig]
    metrics: list[str] = Field(default_factory=lambda: ["cosine", "sam"])
    model_evaluation: ModelEvaluationConfig
    config_dir: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], config_dir: str | None = None) -> "ExperimentConfig":
        payload = dict(raw)
        payload["cluster_sets"] = [ClusterSetConfig.model_validate(item) for item in raw["cluster_sets"]]
        payload["model_evaluation"] = ModelEvaluationConfig.from_dict(raw["model_evaluation"])
        payload["config_dir"] = config_dir
        return cls.model_validate(payload)

    @classmethod
    def from_file(cls, config_path: str | Path) -> "ExperimentConfig":
        path = Path(config_path)
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        config_dir = path.parent.resolve()
        for candidate in [config_dir, *config_dir.parents]:
            if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
                config_dir = candidate
                break
        return cls.from_dict(raw, config_dir=str(config_dir))

