from __future__ import annotations

from typing import Any, Callable

import torch

from .sunsal import SunSAL, SunSALConfig
from .vpgdu import VPGDU, VPGDUConfig

ModelRunner = Callable[[torch.Tensor, torch.Tensor, dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]]


def _run_sunsal(endmembers: torch.Tensor, pixels: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    config_params = dict(params)
    if "μ" in config_params and "mu" not in config_params:
        config_params["mu"] = config_params.pop("μ")
    if "λ_reg" in config_params and "lambda_reg" not in config_params:
        config_params["lambda_reg"] = config_params.pop("λ_reg")
    solver = SunSAL(SunSALConfig(**config_params))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    return abundances, {"iterations_logged": int((history.get("iters") or [0])[-1] if history.get("iters") else 0), "last_active_pixels": int(pixels.shape[0])}


def _run_vpgdu(endmembers: torch.Tensor, pixels: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = VPGDU(VPGDUConfig(**params))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    active = history.get("active_pixels") or [pixels.shape[0]]
    iterations = history.get("iterations") or []
    return abundances, {"iterations_logged": int(len(iterations)), "last_active_pixels": int(active[-1])}


_MODEL_REGISTRY: dict[str, ModelRunner] = {"sunsal": _run_sunsal, "vpgdu": _run_vpgdu}


def register_model(name: str, runner: ModelRunner, *, overwrite: bool = False) -> None:
    key = name.strip().lower()
    if not overwrite and key in _MODEL_REGISTRY:
        raise ValueError(f"Model '{key}' is already registered")
    _MODEL_REGISTRY[key] = runner


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def run_registered_model(model_name: str, endmembers: torch.Tensor, pixels: torch.Tensor, params: dict[str, Any] | None = None) -> tuple[torch.Tensor, dict[str, Any]]:
    key = model_name.strip().lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available_models()}")
    return _MODEL_REGISTRY[key](endmembers, pixels, {} if params is None else dict(params))
