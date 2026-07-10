from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_nn import BaseNNUnmixing, TrainingConfig

"""
Minimal Mamba (selective state-space model) implementation based on:

Gu, A., & Dao, T. (2023)
Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

Uses a sequential (Python-loop) selective scan rather than a fused/parallel
CUDA kernel, so it needs no custom kernels and runs on CPU or GPU alike.
"""


@dataclass(slots=True)
class MambaConfig(TrainingConfig):
    """Configuration for the Mamba unmixing model."""

    d_model: int = 32
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    depth: int = 1


class _MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        d_inner = expand * d_model
        dt_rank = max(1, d_model // 16)

        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(dt_rank, d_inner)

        # A is time-invariant and negative (stable/decaying dynamics); the
        # per-timestep discretization by delta below is what makes the SSM
        # "selective" (input-dependent), per the Mamba paper.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.shape[1]
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :seq_len]
        x_conv = F.silu(x_conv.transpose(1, 2))  # (batch, seq_len, d_inner)

        dt_rank = self.dt_proj.in_features
        delta, B, C = torch.split(self.x_proj(x_conv), [dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # (batch, seq_len, d_inner)

        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        y = self._selective_scan(x_conv, delta, A, B, C)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)

    @staticmethod
    def _selective_scan(
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        # u, delta: (batch, seq_len, d_inner); A: (d_inner, d_state); B, C: (batch, seq_len, d_state)
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        delta_A = torch.exp(delta.unsqueeze(-1) * A)  # (batch, seq_len, d_inner, d_state)
        delta_B_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            outputs.append(torch.einsum("bdn,bn->bd", h, C[:, t]))
        return torch.stack(outputs, dim=1)  # (batch, seq_len, d_inner)


class _AbundanceMamba(nn.Module):
    """Stack of Mamba blocks mapping a spectrum to softmax abundances.

    Each band is treated as one sequence position (scalar reflectance value
    embedded to ``d_model``), mirroring how ConvNeXt1D treats the spectrum as
    a 1D sequence - but mixed across positions with a selective SSM instead
    of convolutions.
    """

    def __init__(
        self,
        out_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        depth: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([_MambaBlock(d_model, d_state, d_conv, expand) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x.unsqueeze(-1))  # (batch, bands, d_model)
        for block, norm in zip(self.blocks, self.norms):
            x = x + block(norm(x))
        pooled = x.mean(dim=1)
        logits = self.head(pooled)
        return F.softmax(logits, dim=-1)


class MambaUnmixing(BaseNNUnmixing):
    """Selective state-space (Mamba) unmixing model with softmax abundance constraints."""

    cfg: MambaConfig

    def _build_model(self, in_dim: int, out_dim: int) -> nn.Module:
        return _AbundanceMamba(
            out_dim=out_dim,
            d_model=self.cfg.d_model,
            d_state=self.cfg.d_state,
            d_conv=self.cfg.d_conv,
            expand=self.cfg.expand,
            depth=self.cfg.depth,
        )
