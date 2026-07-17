from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch

from .small_mlp import SmallMLPConfig, SmallMLPUnmixing
from .convnext1d import ConvNeXt1DConfig, ConvNeXt1DUnmixing
from .kan import KANConfig, KANUnmixing
from .mlm import MLM, MLMConfig
from .sunsal import SunSAL, SunSALConfig
from .vpgdu import VPGDU, VPGDUConfig

ModelRunner = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any], "torch.Tensor | None", "str | None"],
    tuple[torch.Tensor, dict[str, Any]],
]


def _run_sunsal(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any], test_indices: torch.Tensor | None, run_label: str | None) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = SunSAL(SunSALConfig(**dict(params)))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    return abundances, {"iterations_logged": int((history.get("iters") or [0])[-1] if history.get("iters") else 0), "last_active_pixels": int(pixels.shape[0])}


def _run_vpgdu(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any], test_indices: torch.Tensor | None, run_label: str | None) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = VPGDU(VPGDUConfig(**params))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    active = history.get("active_pixels") or [pixels.shape[0]]
    iterations = history.get("iterations") or []
    return abundances, {"iterations_logged": int(len(iterations)), "last_active_pixels": int(active[-1])}


def _run_mlm(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any], test_indices: torch.Tensor | None, run_label: str | None) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = MLM(MLMConfig(**dict(params)))
    abundances = solver.solve(endmembers, pixels)
    history = getattr(solver, "history", {}) or {}
    return abundances, {
        "iterations_logged": int(solver.cfg.refine_iters),
        "last_active_pixels": int(pixels.shape[0]),
        "p_mean": float((history.get("p_mean") or [0.0])[-1]),
        "p_std": float((history.get("p_std") or [0.0])[-1]),
    }


def _run_small_mlp(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any], test_indices: torch.Tensor | None, run_label: str | None) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = SmallMLPUnmixing(
        SmallMLPConfig(**params),
        in_dim=int(pixels.shape[1]),
        out_dim=int(endmembers.shape[0]),
    )
    abundances = solver.solve(endmembers, pixels, true_abundances, test_indices=test_indices)
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

def _run_convnext1d(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any], test_indices: torch.Tensor | None, run_label: str | None) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = ConvNeXt1DUnmixing(
        ConvNeXt1DConfig(**params),
        in_dim=int(pixels.shape[1]),
        out_dim=int(endmembers.shape[0]),
    )
    abundances = solver.solve(endmembers, pixels, true_abundances, test_indices=test_indices)
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


def _run_kan(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any], test_indices: torch.Tensor | None, run_label: str | None) -> tuple[torch.Tensor, dict[str, Any]]:
    solver = KANUnmixing(
        KANConfig(**params),
        in_dim=int(pixels.shape[1]),
        out_dim=int(endmembers.shape[0]),
    )
    abundances = solver.solve(endmembers, pixels, true_abundances, test_indices=test_indices)
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


_MODEL_REGISTRY: dict[str, ModelRunner] = {
    "sunsal": _run_sunsal, "vpgdu": _run_vpgdu, "mlm": _run_mlm, "small_mlp": _run_small_mlp, "convnext1d": _run_convnext1d, "kan": _run_kan}

# Mamba requires the GPU-only mamba-ssm package (see pyproject.toml's "mamba"
# extra); importing it is deferred and guarded so that its absence - the
# common case for CPU-only environments - doesn't break every other model.
try:
    from .mamba import MambaConfig, MambaUnmixing
    def _run_mamba(endmembers: torch.Tensor, pixels: torch.Tensor, true_abundances: torch.Tensor, params: dict[str, Any], test_indices: torch.Tensor | None, run_label: str | None) -> tuple[torch.Tensor, dict[str, Any]]:
        solver = MambaUnmixing(
            MambaConfig(**params),
            in_dim=int(pixels.shape[1]),
            out_dim=int(endmembers.shape[0]),
        )
        abundances = solver.solve(endmembers, pixels, true_abundances, test_indices=test_indices)
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

    _MODEL_REGISTRY["mamba"] = _run_mamba
except ImportError:
    pass


def available_models() -> list[str]:
    """Models runnable in this environment right now (e.g. excludes mamba without mamba-ssm)."""
    return sorted(_MODEL_REGISTRY)


# Every model name the codebase knows about, regardless of whether it can
# actually run in the current environment. Config validation (schema.py) must
# use this rather than available_models(): otherwise a perfectly valid config
# referencing "mamba" fails to even load on a machine without mamba-ssm,
# instead of only failing when that specific model is actually invoked.
KNOWN_MODEL_NAMES: frozenset[str] = frozenset(
    {"sunsal", "vpgdu", "mlm", "small_mlp", "convnext1d", "kan", "mamba"}
)


def known_model_names() -> list[str]:
    return sorted(KNOWN_MODEL_NAMES)


def run_registered_model(
    model_name: str,
    endmembers: np.ndarray,
    pixels: np.ndarray,
    true_abundances: np.ndarray,
    params: dict[str, Any],
    test_indices: np.ndarray | None = None,
    run_label: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:

    if model_name not in _MODEL_REGISTRY:
        if model_name in KNOWN_MODEL_NAMES:
            raise RuntimeError(
                f"Model '{model_name}' is a recognized model but isn't available in this "
                f"environment (e.g. mamba requires the GPU-only mamba-ssm package - see "
                f"pyproject.toml's 'mamba' extra). Available now: {available_models()}"
            )
        raise KeyError(f"Unknown model '{model_name}'. Known models: {known_model_names()}")

    print(f"--- {run_label} ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    endmembers_t = torch.tensor(endmembers, dtype=torch.float32, device=device)
    pixels_t = torch.tensor(pixels, dtype=torch.float32, device=device)
    true_abundances_t = torch.tensor(true_abundances, dtype=torch.float32, device=device)
    test_indices_t = (
        torch.tensor(test_indices, dtype=torch.long, device=device) if test_indices is not None else None
    )
    predicted_abundances_t, diagnostics_dict = _MODEL_REGISTRY[model_name](
        endmembers_t, pixels_t, true_abundances_t, params, test_indices_t, run_label
    )
    predicted_abundances = predicted_abundances_t.detach().cpu().numpy()

    # Models run back-to-back in the same process (both experiment runs, every
    # configured model). Without this, PyTorch's CUDA caching allocator keeps
    # reserving memory freed by earlier models instead of returning it, so later
    # models see less headroom than nvidia-smi/task memory would suggest.
    del endmembers_t, pixels_t, true_abundances_t, predicted_abundances_t
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return predicted_abundances, diagnostics_dict
