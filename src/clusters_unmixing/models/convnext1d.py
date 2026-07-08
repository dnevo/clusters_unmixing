from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_nn import BaseNNUnmixing, TrainingConfig


@dataclass(slots=True)
class ConvNeXt1DConfig(TrainingConfig):
    """Configuration for the ConvNeXt-1D supervised unmixing model."""

    channels: int = 32
    depth: int = 3
    kernel_size: int = 7
    mlp_ratio: int = 4
    dropout: float = 0.0
    pool_bins: int = 8


class _LayerNorm1D(nn.Module):
    """LayerNorm over channels for tensors with shape (batch, channels, length)."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)          # (batch, length, channels)
        x = self.norm(x)
        return x.transpose(1, 2)       # (batch, channels, length)


class _ConvNeXt1DBlock(nn.Module):
    """Small ConvNeXt-style block adapted to 1D spectra."""

    def __init__(self, channels: int, kernel_size: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        hidden_channels = channels * mlp_ratio
        padding = kernel_size // 2

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.norm = _LayerNorm1D(channels)
        self.pointwise_1 = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.pointwise_2 = nn.Conv1d(hidden_channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

        # Zero-init the residual branch's last layer so each block starts as an
        # identity map, matching the ConvNeXt/ResNet trick for stabler early training.
        nn.init.zeros_(self.pointwise_2.weight)
        nn.init.zeros_(self.pointwise_2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = F.gelu(self.pointwise_1(x))
        x = self.dropout(x)
        x = self.pointwise_2(x)
        return residual + x


class _AbundanceConvNeXt1D(nn.Module):
    """ConvNeXt-1D network that maps a spectrum to softmax abundances."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        channels: int,
        depth: int,
        kernel_size: int,
        mlp_ratio: int,
        dropout: float,
        pool_bins: int,
    ) -> None:
        super().__init__()
        self.stem = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            *[
                _ConvNeXt1DBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        # Pool to a small number of spectral bins instead of a single value so the
        # head still sees coarse band-position information (plain global average
        # pooling collapses the whole spectrum, which hurts abundance regression).
        self.pool_bins = max(1, min(int(pool_bins), int(in_dim)))
        self.pool = nn.AdaptiveAvgPool1d(self.pool_bins)
        self.head = nn.Linear(channels * self.pool_bins, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (batch, bands), e.g. (batch, 802)
        x = x.unsqueeze(1)             # (batch, 1, bands)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)               # (batch, channels, pool_bins)
        x = x.flatten(1)               # (batch, channels * pool_bins)
        logits = self.head(x)
        return F.softmax(logits, dim=-1)


class ConvNeXt1DUnmixing(BaseNNUnmixing):
    """ConvNeXt-1D neural unmixing model with softmax abundance constraints."""

    cfg: ConvNeXt1DConfig

    def _build_model(self, in_dim: int, out_dim: int) -> nn.Module:
        return _AbundanceConvNeXt1D(
            in_dim=in_dim,
            out_dim=out_dim,
            channels=self.cfg.channels,
            depth=self.cfg.depth,
            kernel_size=self.cfg.kernel_size,
            mlp_ratio=self.cfg.mlp_ratio,
            pool_bins=self.cfg.pool_bins,
            dropout=self.cfg.dropout,
        )
