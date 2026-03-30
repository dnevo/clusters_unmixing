from __future__ import annotations
import json
import math
import yaml
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

BandRangeSpec = tuple[float, float, str]
TransformStepSpec = tuple[str, dict[str, Any]]


ALLOWED_MODEL_NAMES = {"sunsal", "vpgdu", "small_mlp"}
ALLOWED_CORRELATION_METRICS = {"cosine", "sam"}


class SmallMLPParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hidden_dim_1: int = Field(64, ge=1)
    hidden_dim_2: int = Field(32, ge=1)
    epochs: int = Field(200, ge=1)
    batch_size: int = Field(128, ge=1)
    learning_rate: float = Field(1e-3, gt=0)
    weight_decay: float = Field(1e-5, ge=0)
    lambda_recon: float = Field(0.1, ge=0)
    clip_grad_norm: float = Field(1.0, ge=0)
    patience: int = Field(25, ge=1)
    verbose: bool = False


def _serialize_bands_ranges_for_config(bands_ranges: list[BandRangeSpec]) -> list[Any]:
    if all(reduce == "none" for _, _, reduce in bands_ranges):
        return [[x_min, x_max] for x_min, x_max, _ in bands_ranges]
    return [{"range_µm": [x_min, x_max], "reduce": reduce} for x_min, x_max, reduce in bands_ranges]

def serialize_bands_ranges_key(bands_ranges: list[BandRangeSpec]) -> str:
    payload = _serialize_bands_ranges_for_config(bands_ranges)
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


class ClusterSetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    path: str


class ModelSpecConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parse_str(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data, "params": {}}
        return data

    @model_validator(mode="after")
    def _validate_params(self) -> "ModelSpecConfig":
        name = self.normalized_name()
        if name not in ALLOWED_MODEL_NAMES:
            raise ValueError(f"Unsupported model '{self.name}'. Allowed models: {sorted(ALLOWED_MODEL_NAMES)}")
        if name == "small_mlp":
            self.params = SmallMLPParamsModel.model_validate(self.params).model_dump()
        return self

    def normalized_name(self) -> str:
        name = self.name.strip().lower()
        if not name:
            raise ValueError("Model entry requires non-empty 'name'")
        return name


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

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Model run 'models' must be non-empty")
        normalized = [item.strip().lower() for item in value if item.strip()]
        if len(normalized) != len(value):
            raise ValueError("Model run 'models' entries must be non-empty strings")
        invalid = [name for name in normalized if name not in ALLOWED_MODEL_NAMES]
        if invalid:
            raise ValueError(f"Unsupported model names in run: {sorted(set(invalid))}")
        return normalized

    @field_validator("num_pixels")
    @classmethod
    def _validate_num_pixels(cls, value: int) -> int:
        if value <= 10:
            raise ValueError("Model run 'num_pixels' must be > 10")
        return value

    @field_validator("normalization")
    @classmethod
    def _validate_normalization(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"without", "with_quadratic"}:
            raise ValueError("Model run 'normalization' must be one of: without, with_quadratic")
        return normalized

    @field_validator("snr_db")
    @classmethod
    def _validate_snr_db(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("Model run 'snr_db' must be numeric, not bool")
        if not isinstance(value, (int, float)):
            raise ValueError("Model run 'snr_db' must be numeric")
        if math.isnan(float(value)):
            raise ValueError("Model run 'snr_db' cannot be NaN")
        if float(value) < 0.0 and not math.isinf(float(value)):
            raise ValueError("Model run 'snr_db' must be >= 0 or inf")
        return float(value)

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


class ModelEvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    models: list[ModelSpecConfig] = Field(default_factory=list)
    runs: list[ModelRunConfig] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    experiment_name: str = "correlation_experiment"
    cluster_sets: list[ClusterSetConfig]
    metrics: list[str]
    model_evaluation: ModelEvaluationConfig
    project_root: Path

    @field_validator("metrics")
    @classmethod
    def _validate_metrics(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Experiment 'metrics' must be non-empty")
        normalized = [item.strip().lower() for item in value if item.strip()]
        if len(normalized) != len(value):
            raise ValueError("Experiment 'metrics' entries must be non-empty strings")
        invalid = [name for name in normalized if name not in ALLOWED_CORRELATION_METRICS]
        if invalid:
            raise ValueError(f"Unsupported correlation metrics: {sorted(set(invalid))}")
        return normalized

    @classmethod
    def from_dict(cls, raw: dict[str, Any], project_root: Path) -> "ExperimentConfig":
        payload = dict(raw)
        payload["cluster_sets"] = [ClusterSetConfig.model_validate(item) for item in raw["cluster_sets"]]
        payload["model_evaluation"] = ModelEvaluationConfig.model_validate(raw["model_evaluation"])
        payload["project_root"] = Path(project_root).resolve()
        return cls.model_validate(payload)

    def resolve_path(self, value: str) -> Path:
        return self.project_root / value

    @property
    def experiment_output_dir(self) -> Path:
        return self.project_root / "experiments" / "outputs" / self.experiment_name

    @classmethod
    def from_config_file(cls, project_root: Path) -> "ExperimentConfig":
        config_path = project_root / "experiments" / "configs" / "configuration.yaml"
        text = config_path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        return cls.from_dict(raw, project_root=project_root)
