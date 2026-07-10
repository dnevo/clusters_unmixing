from __future__ import annotations

import math
from dataclasses import dataclass

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
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        step = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step
            + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.base_activation = nn.SiLU()

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            self.spline_weight.normal_(mean=0.0, std=scale_noise / (grid_size + spline_order))

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # Cox-de Boor recursion. x: (batch, in_features) -> (batch, in_features, grid_size + spline_order)
        # Forced fp32: this chains several divisions/subtractions of nearby grid
        # values, which loses more precision under fp16 autocast than a plain
        # matmul would - so it opts out of the outer autocast context.
        with torch.autocast(device_type="cuda", enabled=False):
            x = x.float()
            grid = self.grid.float()
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
    ) -> None:
        super().__init__()
        self.layer1 = _KANLinear(in_dim, hidden_dim, grid_size, spline_order, grid_range, scale_noise)
        self.layer2 = _KANLinear(hidden_dim, out_dim, grid_size, spline_order, grid_range, scale_noise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        logits = self.layer2(x)
        return F.softmax(logits, dim=-1)


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
        )
