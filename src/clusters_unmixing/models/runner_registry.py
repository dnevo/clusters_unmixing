from __future__ import annotations

from typing import Any, Callable

import numpy as np
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
    solver = SmallMLPUnmixing(
        SmallMLPConfig(**params),
        in_dim=int(pixels.shape[1]),
        out_dim=int(endmembers.shape[0]),
    )
    abundances = solver.solve(endmembers, pixels, true_abundances)
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
    endmembers: np.ndarray,
    pixels: np.ndarray,
    true_abundances: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    endmembers_t = torch.tensor(endmembers, dtype=torch.float32, device=device)
    pixels_t = torch.tensor(pixels, dtype=torch.float32, device=device)
    true_abundances_t = torch.tensor(true_abundances, dtype=torch.float32, device=device)
    predicted_abundances_t, diagnostics_dict = _MODEL_REGISTRY[model_name](endmembers_t, pixels_t, true_abundances_t, params)
    predicted_abundances = predicted_abundances_t.detach().cpu().numpy()
 
    return predicted_abundances, diagnostics_dict
