from __future__ import annotations
import json
import math
import yaml
from pathlib import Path
from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from clusters_unmixing.models.runner_registry import known_model_names

BandRangeSpec = tuple[float, float, str]

# Run-level fields eligible for sweep_params. Deliberately excludes cluster_set and
# bands_ranges (both change array shapes downstream) and name/models/sweep_params
# (structural, not data).
SWEEPABLE_RUN_PARAMS = {
    "num_pixels",
    "snr_db",
    "nonlinearity_gamma",
    "normalization",
    "abundance_distribution",
    "dirichlet_alpha",
}


def _validate_sweep_value_list(owner: str, key: str, values: Any) -> list[Any]:
    # Note: values are Any-typed and not coerced/type-checked here (unlike the
    # matching top-level field, e.g. snr_db's float validator below). A YAML
    # config typo like bare `inf` instead of `.inf` for an snr_db sweep value
    # silently parses as the string "inf" and passes through unnoticed - it only
    # fails later, deep in core_math.py's np.isinf(), with a confusing TypeError.
    if not isinstance(values, list) or not values:
        raise ValueError(f"{owner}.{key} must be a non-empty list")
    if len(set(values)) != len(values):
        raise ValueError(f"{owner}.{key} must not contain duplicate values")
    return values


class MLPParamsModel(BaseModel):
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
        valid_models = known_model_names()
        if name not in valid_models:
            raise ValueError(f"Unsupported model '{self.name}'. Registered models: {sorted(valid_models)}")
        if name == "mlp":
            self.params = MLPParamsModel.model_validate(self.params).model_dump()
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


class ModelSelectionEntry(BaseModel):
    """One entry in a run's `models` list: either a bare model name (run once with
    the model's catalog defaults from the top-level `models` list) or a single-key
    mapping `name: {param: [values]}` sweeping that model over a param grid.

    Cross-checking the name/params against the top-level `models` catalog needs
    sibling data this model doesn't have, so that part happens in
    ExperimentConfig._validate_models_against_catalog instead.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    param_sweep: dict[str, list[Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parse(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data, "param_sweep": {}}
        if isinstance(data, dict) and "name" not in data:
            if len(data) != 1:
                raise ValueError(
                    "Model entry mapping must have exactly one key (the model name), "
                    "e.g. 'kan: {batch_size: [32, 128]}'"
                )
            (raw_name, param_sweep), = data.items()
            return {"name": raw_name, "param_sweep": param_sweep or {}}
        return data

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        name = value.strip().lower()
        if not name:
            raise ValueError("Model entry requires a non-empty name")
        valid_models = known_model_names()
        if name not in valid_models:
            raise ValueError(f"Unsupported model '{value}'. Registered models: {sorted(valid_models)}")
        return name

    @field_validator("param_sweep")
    @classmethod
    def _validate_param_sweep(cls, value: Any) -> dict[str, list[Any]]:
        if not isinstance(value, dict):
            raise ValueError("Model entry param overrides must be a mapping of parameter name -> list of values")
        for key, values in value.items():
            _validate_sweep_value_list("model entry", key, values)
        return value

    def is_swept(self) -> bool:
        return bool(self.param_sweep)


class ModelRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cluster_set: str
    bands_ranges: list[Any]
    normalization: str = "none"
    abundance_distribution: str = "grid"
    dirichlet_alpha: float = Field(
        0.3,
        gt=0,
        description=(
            "Symmetric Dirichlet concentration parameter used when abundance_distribution "
            "is 'dirichlet'. Values below 1 produce sparse-looking mixtures; 1 is uniform "
            "over the continuous simplex; values above 1 favor balanced mixtures."
        ),
    )
    models: list[ModelSelectionEntry]
    sweep_params: dict[str, list[Any]] = Field(default_factory=dict)
    num_pixels: int
    snr_db: float
    nonlinearity_gamma: float = Field(
        0.0,
        description=(
            "Generalized Bilinear Model (GBM) nonlinearity upper bound in [0, 1] used when "
            "synthesizing pixels from abundances and endmembers. 0.0 keeps the standard "
            "linear mixing model; for values > 0, each endmember pair's own nonlinear "
            "coefficient gamma_ij is drawn independently from Uniform(0, nonlinearity_gamma) "
            "(see core_math.mix_pixels). For chemically graded cluster sets (e.g. "
            "6clusters_thomas), this is a numerical robustness knob, not a physical model "
            "of distinct materials interacting."
        ),
    )

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: list[ModelSelectionEntry]) -> list[ModelSelectionEntry]:
        if not value:
            raise ValueError("Model run 'models' must be non-empty")
        return value

    @field_validator("sweep_params")
    @classmethod
    def _validate_sweep_params(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        unknown = sorted(set(value) - SWEEPABLE_RUN_PARAMS)
        if unknown:
            raise ValueError(
                f"sweep_params has unsupported key(s) {unknown}; allowed: {sorted(SWEEPABLE_RUN_PARAMS)}"
            )
        for key, values in value.items():
            _validate_sweep_value_list("sweep_params", key, values)
        return value

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
        if normalized not in {"none", "quadratic"}:
            raise ValueError("Model run 'normalization' must be one of: none, quadratic")
        return normalized

    @field_validator("abundance_distribution")
    @classmethod
    def _validate_abundance_distribution(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"grid", "dirichlet"}:
            raise ValueError("Model run 'abundance_distribution' must be one of: grid, dirichlet")
        return normalized

    @field_validator("dirichlet_alpha")
    @classmethod
    def _validate_dirichlet_alpha(cls, value: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Model run 'dirichlet_alpha' must be a positive number")
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError("Model run 'dirichlet_alpha' must be finite and > 0")
        return float(value)

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

    @field_validator("nonlinearity_gamma")
    @classmethod
    def _validate_nonlinearity_gamma(cls, value: float) -> float:
        if isinstance(value, bool):
            raise ValueError("Model run 'nonlinearity_gamma' must be numeric, not bool")
        if not isinstance(value, (int, float)):
            raise ValueError("Model run 'nonlinearity_gamma' must be numeric")
        if math.isnan(float(value)):
            raise ValueError("Model run 'nonlinearity_gamma' cannot be NaN")
        if not (0.0 <= float(value) <= 1.0):
            raise ValueError("Model run 'nonlinearity_gamma' must be within [0, 1]")
        return float(value)

    def effective_model_names(self) -> list[str]:
        """Model names this run will execute (order as configured; may repeat if a
        model appears in more than one `models` entry)."""
        return [entry.name for entry in self.models]

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


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    experiment_name: str = "cosine_similarity_experiment"
    cluster_sets: list[ClusterSetConfig]
    models: list[ModelSpecConfig] = Field(default_factory=list)
    runs: list[ModelRunConfig] = Field(default_factory=list, validate_default=True)
    project_root: Path
    extra_reproducibility: bool = Field(
        False,
        description=(
            "Ask PyTorch/cuDNN/cuBLAS for bit-exact CUDA determinism (see "
            "pipelines/experiment_pipeline.py's _set_global_seeds docstring), at a "
            "performance cost. Automatically skipped for runs that include the mamba "
            "model, whose fused CUDA kernels don't support it."
        ),
    )
    use_wandb: bool = Field(
        False,
        description=(
            "Log each model call to Weights & Biases as its own run (see "
            "wandb_logging.py), plus one extra summary run per batch that uploads the "
            "final output CSVs as an artifact. Requires the 'wandb' extra "
            "(`pip install -e .[wandb]`) and a logged-in wandb account."
        ),
    )
    num_seeds: int = Field(
        1,
        ge=1,
        description=(
            "Number of times every resolved run is repeated end-to-end (synthetic "
            "data generation through model fitting), once per seed in "
            "range(num_seeds) - i.e. seeds 0, 1, ..., num_seeds - 1. Each seed's "
            "per-model metrics are tagged with a 'seed' column in model_summary.csv, "
            "so downstream analysis (e.g. the review notebook's stability tables) can "
            "report mean +/- std across seeds instead of a single point estimate. "
            "1 keeps the historical single-seed (seed=0) behavior."
        ),
    )

    @field_validator("runs")
    @classmethod
    def _validate_runs(cls, value: list[ModelRunConfig]) -> list[ModelRunConfig]:
        if not value:
            raise ValueError("runs must contain at least one run")
        return value

    @model_validator(mode="after")
    def _validate_models_against_catalog(self) -> "ExperimentConfig":
        catalog = {model.normalized_name(): model.params for model in self.models}
        for run in self.runs:
            for entry in run.models:
                if entry.name not in catalog:
                    raise ValueError(
                        f"run references model '{entry.name}', which is not defined in 'models'"
                    )
                if entry.param_sweep:
                    default_params = catalog[entry.name]
                    unknown = sorted(set(entry.param_sweep) - set(default_params))
                    if unknown:
                        raise ValueError(
                            f"model '{entry.name}' has swept parameter(s) {unknown} not present in "
                            f"its default params {sorted(default_params)}"
                        )
        return self

    @classmethod
    def from_dict(cls, raw: dict[str, Any], project_root: Path) -> "ExperimentConfig":
        payload = dict(raw)
        payload["cluster_sets"] = [ClusterSetConfig.model_validate(item) for item in raw["cluster_sets"]]
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
