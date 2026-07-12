from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_nn import BaseNNUnmixing, TrainingConfig

"""
Minimal Mamba (selective state-space model) implementation based on:

Gu, A., & Dao, T. (2023)
Mamba: Linear-Time Sequence Modeling with Selective State Spaces.

GPU-only: the selective scan and the causal conv use mamba-ssm's/causal-conv1d's
fused CUDA kernels rather than pure-PyTorch equivalents, so both packages must
be installed and inputs must be on a CUDA device (see the notebook's install
cell).
"""

from causal_conv1d import causal_conv1d_fn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


@dataclass(slots=True)
class MambaConfig(TrainingConfig):
    """Configuration for the Mamba unmixing model."""

    d_model: int = 32
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    depth: int = 1
    patch_size: int = 8


class _MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        d_inner = expand * d_model
        dt_rank = max(1, d_model // 16)

        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, 2 * d_inner)

        # Raw (dim, width) weight/bias rather than nn.Conv1d, since
        # causal_conv1d_fn takes that shape directly (no singleton
        # in_channels/groups dim). Initialized to match nn.Conv1d's own
        # reset_parameters() for a depthwise conv of this shape.
        self.conv1d_weight = nn.Parameter(torch.empty(d_inner, d_conv))
        self.conv1d_bias = nn.Parameter(torch.empty(d_inner))
        nn.init.kaiming_uniform_(self.conv1d_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(d_conv)
        nn.init.uniform_(self.conv1d_bias, -bound, bound)

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
        x_in, z = self.in_proj(x).chunk(2, dim=-1)

        x_conv = causal_conv1d_fn(
            x_in.transpose(1, 2).contiguous(), self.conv1d_weight, self.conv1d_bias, activation="silu"
        ).transpose(1, 2)  # (batch, seq_len, d_inner)

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
        # mamba_ssm expects (batch, dim, seq_len) / (batch, dstate, seq_len), the
        # transpose of our (batch, seq_len, dim) convention. delta is already
        # softplus'd by the caller, so delta_softplus=False and D is applied by
        # the caller too, so D=None here - both avoided being applied twice.
        y = selective_scan_fn(
            u.transpose(1, 2).contiguous(),
            delta.transpose(1, 2).contiguous(),
            A,
            B.transpose(1, 2).contiguous(),
            C.transpose(1, 2).contiguous(),
            D=None,
            delta_softplus=False,
        )
        return y.transpose(1, 2)


class _AbundanceMamba(nn.Module):
    """Stack of Mamba blocks mapping a spectrum to softmax abundances.

    Adjacent bands are grouped into patches of ``patch_size`` and embedded to
    ``d_model`` (as ViT-style patch embedding does for images), rather than
    treating every single band as its own sequence position - this keeps the
    sequence length (and so the conv/SSM compute and memory) from scaling
    directly with the raw band count.
    """

    def __init__(
        self,
        out_dim: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        depth: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stem = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([_MambaBlock(d_model, d_state, d_conv, expand) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x.unsqueeze(1)).transpose(1, 2)  # (batch, bands // patch_size, d_model)
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
            patch_size=max(1, min(self.cfg.patch_size, in_dim)),
        )
