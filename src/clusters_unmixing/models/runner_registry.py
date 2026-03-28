from __future__ import annotations

from typing import Any, Callable

import torch

from .small_mlp import SmallMLPConfig, SmallMLPUnmixing
from .sunsal import SunSAL, SunSALConfig
from .vpgdu import VPGDU, VPGDUConfig

ModelRunner = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]]


def _run_sunsal(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = SunSAL(SunSALConfig(**dict(params)))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    return abundances, {"iterations_logged": int((history.get("iters") or [0])[-1] if history.get("iters") else 0), "last_active_pixels": int(pixels.shape[0])}


def _run_vpgdu(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = VPGDU(VPGDUConfig(**params))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    active = history.get("active_pixels") or [pixels.shape[0]]
    iterations = history.get("iterations") or []
    return abundances, {"iterations_logged": int(len(iterations)), "last_active_pixels": int(active[-1])}


def _run_small_mlp(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    endmembers_bands_k = endmembers.T.contiguous()
    solver = SmallMLPUnmixing(
        SmallMLPConfig(**params),
        in_dim=int(pixels.shape[1]),
        out_dim=int(endmembers_bands_k.shape[1]),
    )
    abundances = solver.solve(endmembers_bands_k, pixels, true_abundances)
    history = getattr(solver, "history", {}) or {}
    epochs = history.get("epoch") or []
    return abundances, {
        "iterations_logged": int(epochs[-1] if epochs else 0),
        "last_active_pixels": int(pixels.shape[0]),
        "best_val_loss": float(solver.best_val_loss),
        "test_loss": float(solver.test_loss),
        "test_abund_loss": float(solver.test_abund_loss),
        "test_recon_loss": float(solver.test_recon_loss),
    }


_MODEL_REGISTRY: dict[str, ModelRunner] = {"sunsal": _run_sunsal, "vpgdu": _run_vpgdu, "small_mlp": _run_small_mlp}


def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY)


def run_registered_model(
    model_name: str,
    endmembers: torch.Tensor,
    pixels: torch.Tensor,
    true_abundances: torch.Tensor,
    params: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    key = model_name.strip().lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(f"Unsupported model '{model_name}'. Available models: {available_models()}")
    return _MODEL_REGISTRY[key](endmembers, pixels, true_abundances, {} if params is None else dict(params))
