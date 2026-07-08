from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_nn import BaseNNUnmixing, TrainingConfig


@dataclass(slots=True)
class SmallMLPConfig(TrainingConfig):
    """Configuration for the small supervised MLP unmixing model."""

    hidden_dim_1: int = 64
    hidden_dim_2: int = 32


class _AbundanceMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim_1: int, hidden_dim_2: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return F.softmax(logits, dim=-1)


class SmallMLPUnmixing(BaseNNUnmixing):
    """Small neural unmixing model (3-layer MLP with softmax abundances)."""

    cfg: SmallMLPConfig

    def _build_model(self, in_dim: int, out_dim: int) -> nn.Module:
        return _AbundanceMLP(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim_1=self.cfg.hidden_dim_1,
            hidden_dim_2=self.cfg.hidden_dim_2,
        )