from __future__ import annotations

from typing import Any, Callable

import torch

from .small_mlp import SmallMLPConfig, SmallMLPUnmixing
from .sunsal import SunSAL, SunSALConfig
from .vpgdu import VPGDU, VPGDUConfig

ModelRunner = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]]


def _run_sunsal(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor | None, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    config_params = dict(params)
    if "μ" in config_params and "mu" not in config_params:
        config_params["mu"] = config_params.pop("μ")
    if "λ_reg" in config_params and "lambda_reg" not in config_params:
        config_params["lambda_reg"] = config_params.pop("λ_reg")
    solver = SunSAL(SunSALConfig(**config_params))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    return abundances, {"iterations_logged": int((history.get("iters") or [0])[-1] if history.get("iters") else 0), "last_active_pixels": int(pixels.shape[0])}


def _run_vpgdu(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor | None, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = VPGDU(VPGDUConfig(**params))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    active = history.get("active_pixels") or [pixels.shape[0]]
    iterations = history.get("iterations") or []
    return abundances, {"iterations_logged": int(len(iterations)), "last_active_pixels": int(active[-1])}


def _run_small_mlp(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor | None, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = SmallMLPUnmixing(SmallMLPConfig(**params))
    abundances = solver.solve(endmembers.T.contiguous(), pixels, true_abundances)
    history = getattr(solver, "history", {}) or {}
    epochs = history.get("epoch") or []
    return abundances, {
        "iterations_logged": int(epochs[-1] if epochs else 0),
        "last_active_pixels": int(pixels.shape[0]),
        "best_val_loss": float(solver.best_val_loss if solver.best_val_loss is not None else 0.0),
        "test_loss": float(solver.test_loss if solver.test_loss is not None else 0.0),
        "test_abund_loss": float(solver.test_abund_loss if solver.test_abund_loss is not None else 0.0),
        "test_recon_loss": float(solver.test_recon_loss if solver.test_recon_loss is not None else 0.0),
    }


_MODEL_REGISTRY: dict[str, ModelRunner] = {"sunsal": _run_sunsal, "vpgdu": _run_vpgdu, "small_mlp": _run_small_mlp}


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def run_registered_model(
    model_name: str,
    endmembers: torch.Tensor,
    pixels: torch.Tensor,
    true_abundances: torch.Tensor | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    key = model_name.strip().lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available_models()}")
    return _MODEL_REGISTRY[key](endmembers, pixels, true_abundances, {} if params is None else dict(params))
