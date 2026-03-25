from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class SmallMLPConfig:
    """Configuration for the small supervised MLP unmixing model."""

    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_recon: float = 0.1
    clip_grad_norm: float = 1.0
    patience: int = 25
    verbose: bool = False


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


class SmallMLPUnmixing:
    """Small neural unmixing model.

    Expected shapes
    ---------------
    endmembers : (bands, K)
    pixels : (N, bands)
    true_abundances : (N, K)
    """

    def __init__(self, config: SmallMLPConfig):
        self.cfg = config
        self.model: Optional[_AbundanceMLP] = None
        self.history: dict[str, list[float | int]] = {
            "epoch": [],
            "train_loss": [],
            "train_abund_loss": [],
            "train_recon_loss": [],
            "val_loss": [],
            "val_abund_loss": [],
            "val_recon_loss": [],
        }
        self.best_val_loss: Optional[float] = None
        self.test_loss: Optional[float] = None
        self.test_abund_loss: Optional[float] = None
        self.test_recon_loss: Optional[float] = None

    def _split_counts(self, n_samples: int) -> tuple[int, int, int]:
        train_count = int(n_samples * 0.70)
        val_count = int(n_samples * 0.20)
        test_count = n_samples - train_count - val_count
        return train_count, val_count, test_count

    def _compute_losses(
        self,
        pixels: torch.Tensor,
        true_abundances: torch.Tensor,
        endmembers_bands_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_abundances = self.model(pixels)
        abund_loss = F.mse_loss(pred_abundances, true_abundances)
        recon_pixels = pred_abundances @ endmembers_bands_k.T
        recon_loss = F.mse_loss(recon_pixels, pixels)
        total_loss = abund_loss + self.cfg.lambda_recon * recon_loss
        return total_loss, abund_loss, recon_loss, pred_abundances

    @torch.no_grad()
    def _evaluate_split(
        self,
        pixels: torch.Tensor,
        true_abundances: torch.Tensor,
        endmembers_bands_k: torch.Tensor,
    ) -> tuple[float, float, float]:
        total_loss, abund_loss, recon_loss, _ = self._compute_losses(
            pixels=pixels,
            true_abundances=true_abundances,
            endmembers_bands_k=endmembers_bands_k,
        )
        return float(total_loss.item()), float(abund_loss.item()), float(recon_loss.item())

    def solve(
        self,
        endmembers: torch.Tensor,
        pixels: torch.Tensor,
        true_abundances: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg

        device = pixels.device
        dtype = pixels.dtype

        endmembers_bands_k = endmembers.to(device=device, dtype=dtype).contiguous()
        pixels_n_bands = pixels.to(device=device, dtype=dtype).contiguous()
        true_abundances_n_k = true_abundances.to(device=device, dtype=dtype).contiguous()

        n_samples = int(pixels_n_bands.shape[0])
        bands = int(pixels_n_bands.shape[1])
        n_endmembers = int(endmembers_bands_k.shape[1])

        train_count, val_count, test_count = self._split_counts(n_samples)

        x_train = pixels_n_bands[:train_count]
        y_train = true_abundances_n_k[:train_count]

        x_val = pixels_n_bands[train_count : train_count + val_count]
        y_val = true_abundances_n_k[train_count : train_count + val_count]

        x_test = pixels_n_bands[train_count + val_count :]
        y_test = true_abundances_n_k[train_count + val_count :]

        self.model = _AbundanceMLP(
            in_dim=bands,
            out_dim=n_endmembers,
            hidden_dim_1=cfg.hidden_dim_1,
            hidden_dim_2=cfg.hidden_dim_2,
        ).to(device=device, dtype=dtype)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        batch_size = max(1, min(int(cfg.batch_size), train_count))
        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, int(cfg.epochs) + 1):
            self.model.train()
            permutation = torch.randperm(train_count, device=device)

            train_loss_sum = 0.0
            train_abund_sum = 0.0
            train_recon_sum = 0.0
            num_batches = 0

            for start in range(0, train_count, batch_size):
                batch_idx = permutation[start : start + batch_size]
                batch_pixels = x_train[batch_idx]
                batch_abundances = y_train[batch_idx]

                optimizer.zero_grad(set_to_none=True)
                total_loss, abund_loss, recon_loss, _ = self._compute_losses(
                    pixels=batch_pixels,
                    true_abundances=batch_abundances,
                    endmembers_bands_k=endmembers_bands_k,
                )
                total_loss.backward()
                if cfg.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)
                optimizer.step()

                train_loss_sum += float(total_loss.item())
                train_abund_sum += float(abund_loss.item())
                train_recon_sum += float(recon_loss.item())
                num_batches += 1

            mean_train_loss = train_loss_sum / max(1, num_batches)
            mean_train_abund = train_abund_sum / max(1, num_batches)
            mean_train_recon = train_recon_sum / max(1, num_batches)

            self.model.eval()
            val_loss, val_abund_loss, val_recon_loss = self._evaluate_split(
                pixels=x_val,
                true_abundances=y_val,
                endmembers_bands_k=endmembers_bands_k,
            )

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(mean_train_loss)
            self.history["train_abund_loss"].append(mean_train_abund)
            self.history["train_recon_loss"].append(mean_train_recon)
            self.history["val_loss"].append(val_loss)
            self.history["val_abund_loss"].append(val_abund_loss)
            self.history["val_recon_loss"].append(val_recon_loss)

            if cfg.verbose and (epoch == 1 or epoch % 20 == 0 or epoch == cfg.epochs):
                print(
                    f"Epoch {epoch:4d} | train={mean_train_loss:.6e} | "
                    f"val={val_loss:.6e} | val_abund={val_abund_loss:.6e} | "
                    f"val_recon={val_recon_loss:.6e}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(cfg.patience):
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.best_val_loss = best_val_loss

        self.model.eval()
        self.test_loss, self.test_abund_loss, self.test_recon_loss = self._evaluate_split(
            pixels=x_test,
            true_abundances=y_test,
            endmembers_bands_k=endmembers_bands_k,
        )

        with torch.no_grad():
            abundances_all = self.model(pixels_n_bands)

        return abundances_all
