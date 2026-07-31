from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_nn import BaseNNUnmixing, TrainingConfig

"""
Kolmogorov-Arnold Network (KAN) implementation based on:

Liu, Z. et al. (2024)
KAN: Kolmogorov-Arnold Networks.

Each edge computes a learnable univariate function (a base activation plus a
B-spline correction) instead of a single scalar weight, per the efficient-kan
formulation.
"""


@dataclass(slots=True)
class KANConfig(TrainingConfig):
    """Configuration for the KAN unmixing model."""

    hidden_dim: int = 32
    grid_size: int = 5
    spline_order: int = 3
    grid_range: tuple[float, float] = (-1.0, 1.0)
    scale_noise: float = 0.1
    # Weight on the uniform grid vs. raw data quantiles in update_grid's blend
    # (see _KANLinear.update_grid). The efficient-kan reference default of 0.02
    # leans almost entirely on empirical quantiles, which chases noise outliers
    # under noisy + nonlinear/normalized pixel distributions; 0.3 trades some
    # calibration precision for robustness to that noise.
    grid_eps: float = 0.3
    # Epochs of ordinary supervised training run before grid calibration, so
    # layer2's grid is calibrated against layer1's actual (trained) activations
    # rather than layer1's near-arbitrary output at random initialization.
    grid_warmup_epochs: int = 5


class _KANLinear(nn.Module):
    """One KAN layer: per-edge base activation + B-spline function."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int,
        spline_order: int,
        grid_range: tuple[float, float],
        scale_noise: float,
        grid_eps: float = 0.02,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_eps = grid_eps

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.grid: torch.Tensor
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.base_activation = nn.SiLU()

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.spline_weight.normal_(mean=0.0, std=scale_noise / (grid_size + spline_order))

    def _b_splines(self, x: torch.Tensor, grid: torch.Tensor | None = None) -> torch.Tensor:
        # Cox-de Boor recursion. x: (batch, in_features) -> (batch, in_features, grid_size + spline_order)
        # Forced fp32: this chains several divisions/subtractions of nearby grid
        # values, which loses more precision under fp16 autocast than a plain
        # matmul would - so it opts out of the outer autocast context.
        with torch.autocast(device_type="cuda", enabled=False):
            x = x.float()
            grid = (self.grid if grid is None else grid).float()
            x = x.unsqueeze(-1)
            bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
            for k in range(1, self.spline_order + 1):
                left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
                right = (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:-k])
                bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_bases = self._b_splines(x).reshape(x.size(0), -1)
        spline_output = F.linear(spline_bases, self.spline_weight.reshape(self.out_features, -1))
        return base_output + spline_output

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        # Least-squares fit of new-grid B-spline coefficients that reproduce the
        # given (x, y) samples, used by update_grid to re-express the current
        # spline function on a re-fitted grid without changing what it computes.
        # x: (batch, in_features), y: (batch, in_features, out_features).
        a = self._b_splines(x, grid).transpose(0, 1)  # (in_features, batch, coeff)
        b = y.transpose(0, 1)  # (in_features, batch, out_features)
        solution = torch.linalg.lstsq(a, b).solution  # (in_features, coeff, out_features)
        return solution.permute(2, 0, 1).contiguous()  # (out_features, in_features, coeff)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin: float = 0.01) -> None:
        """Re-fit this layer's B-spline grid to the observed input distribution.

        The grid starts as a fixed, arbitrary range (see KANConfig.grid_range) with
        no knowledge of the actual input scale; most of its spline capacity can end
        up wasted on regions the data never visits. Blending a data-quantile grid
        with a uniform one (weighted by grid_eps) concentrates knots where the data
        actually is while keeping the grid well-formed for outliers. Coefficients
        are then re-solved (least squares) so the spline function's output is
        preserved across the grid change.
        """
        batch = x.size(0)
        splines = self._b_splines(x).permute(1, 0, 2)  # (in_features, batch, coeff)
        orig_coeff = self.spline_weight.permute(1, 2, 0)  # (in_features, coeff, out_features)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)  # (batch, in_features, out_features)

        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.cat(
            [
                grid[:1] - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:] + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        new_grid = grid.T.contiguous()
        self.spline_weight.data.copy_(self._curve2coeff(x, unreduced_spline_output, new_grid))
        self.grid.copy_(new_grid)


class _AbundanceKAN(nn.Module):
    """Two-layer KAN that maps a spectrum to softmax abundances."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        grid_size: int,
        spline_order: int,
        grid_range: tuple[float, float],
        scale_noise: float,
        grid_eps: float = 0.02,
    ) -> None:
        super().__init__()
        self.layer1 = _KANLinear(in_dim, hidden_dim, grid_size, spline_order, grid_range, scale_noise, grid_eps)
        self.layer2 = _KANLinear(hidden_dim, out_dim, grid_size, spline_order, grid_range, scale_noise, grid_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        logits = self.layer2(x)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor) -> None:
        """Calibrate each layer's B-spline grid to its actual input distribution,
        propagating the (updated) layer1 activations forward to calibrate layer2."""
        self.layer1.update_grid(x)
        self.layer2.update_grid(self.layer1(x))


class KANUnmixing(BaseNNUnmixing):
    """Kolmogorov-Arnold Network unmixing model with softmax abundance constraints."""

    cfg: KANConfig

    def _build_model(self, in_dim: int, out_dim: int) -> nn.Module:
        return _AbundanceKAN(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=self.cfg.hidden_dim,
            grid_size=self.cfg.grid_size,
            spline_order=self.cfg.spline_order,
            grid_range=self.cfg.grid_range,
            scale_noise=self.cfg.scale_noise,
            grid_eps=self.cfg.grid_eps,
        )

    def _prepare_model_for_training(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        endmembers_k_bands: torch.Tensor,
    ) -> None:
        # layer1's grid can be calibrated directly against x_train (the raw pixel
        # distribution doesn't change during training), but layer2's grid needs
        # layer1's *output* distribution - which at random init is close to
        # arbitrary and a poor proxy for what layer1 actually produces once
        # trained. So run a short supervised warm-up first (ordinary gradient
        # steps on the real objective) and only then calibrate both layers'
        # grids, using layer1's now-representative activations for layer2.
        if self.cfg.grid_warmup_epochs > 0:
            self._run_grid_warmup(x_train, y_train, endmembers_k_bands)
        cast(_AbundanceKAN, self.model).update_grid(x_train)

    def _run_grid_warmup(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        endmembers_k_bands: torch.Tensor,
    ) -> None:
        warmup_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        batch_size = max(1, min(int(self.cfg.batch_size), x_train.shape[0]))

        self.model.train()
        for _ in range(int(self.cfg.grid_warmup_epochs)):
            permutation = torch.randperm(x_train.shape[0], device=x_train.device)
            for start in range(0, x_train.shape[0], batch_size):
                batch_idx = permutation[start : start + batch_size]
                warmup_optimizer.zero_grad(set_to_none=True)
                total_loss, _, _, _ = self._compute_losses(
                    pixels=x_train[batch_idx],
                    true_abundances=y_train[batch_idx],
                    endmembers_k_bands=endmembers_k_bands,
                )
                total_loss.backward()
                if self.cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip_grad_norm)
                warmup_optimizer.step()
