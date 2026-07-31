from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(slots=True)
class TrainingConfig:
    """Model-agnostic training configuration shared by all NN unmixing models."""

    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_recon: float = 0.1
    clip_grad_norm: float = 1.0
    patience: int = 25
    verbose: bool = False


class BaseNNUnmixing(ABC):
    """Shared scaffolding for supervised NN unmixing models.

    Subclasses must implement :meth:`_build_model`, returning an ``nn.Module``
    that maps pixels ``(N, bands)`` to abundances ``(N, K)``. Subclasses may
    also override :meth:`_compute_losses` if they need a different objective.

    Expected shapes
    ---------------
    endmembers : (K, bands)
    pixels : (N, bands)
    true_abundances : (N, K)
    """

    def __init__(self, config: TrainingConfig, in_dim: int, out_dim: int):
        self.cfg = config
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.model = self._build_model(self.in_dim, self.out_dim)
        self.history: dict[str, list[float | int]] = {
            "epoch": [],
            "train_loss": [],
            "train_abund_loss": [],
            "train_recon_loss": [],
            "val_loss": [],
            "val_abund_loss": [],
            "val_recon_loss": [],
        }
        self.best_val_loss: float = float("inf")
        self.test_loss: float = float("nan")
        self.test_abund_loss: float = float("nan")
        self.test_recon_loss: float = float("nan")

    @abstractmethod
    def _build_model(self, in_dim: int, out_dim: int) -> nn.Module:
        """Build and return the network mapping pixels -> abundances."""

    def _prepare_model_for_training(
        self,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        endmembers_k_bands: torch.Tensor,
    ) -> None:
        """Hook for a one-time, data-dependent setup pass before training starts
        (e.g. KAN's B-spline grid calibration). No-op unless a subclass overrides it.
        y_train/endmembers_k_bands are provided so a subclass can run a small
        supervised warm-up (via self._compute_losses) if its setup needs
        representative *trained* activations rather than ones from a still-random
        network."""

    def _split_counts(self, n_samples: int) -> tuple[int, int, int]:
        train_count = int(n_samples * 0.70)
        val_count = int(n_samples * 0.20)
        test_count = n_samples - train_count - val_count
        return train_count, val_count, test_count

    def _compute_losses(
        self,
        pixels: torch.Tensor,
        true_abundances: torch.Tensor,
        endmembers_k_bands: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mixed precision only on CUDA (T4 tensor cores) - halves activation memory
        # and speeds up the conv/linear layers; CPU runs are unaffected.
        with torch.autocast(device_type="cuda", enabled=getattr(self, "_use_amp", False)):
            pred_abundances = self.model(pixels)
            abund_loss = F.mse_loss(pred_abundances, true_abundances)
            recon_pixels = pred_abundances @ endmembers_k_bands
            recon_loss = F.mse_loss(recon_pixels, pixels)
            total_loss = abund_loss + self.cfg.lambda_recon * recon_loss
        return total_loss, abund_loss, recon_loss, pred_abundances

    @torch.no_grad()
    def _evaluate_split(
        self,
        pixels: torch.Tensor,
        true_abundances: torch.Tensor,
        endmembers_k_bands: torch.Tensor,
        batch_size: int | None = None,
    ) -> tuple[float, float, float]:
        n_samples = pixels.shape[0]
        if batch_size is None or batch_size >= n_samples:
            total_loss, abund_loss, recon_loss, _ = self._compute_losses(
                pixels=pixels,
                true_abundances=true_abundances,
                endmembers_k_bands=endmembers_k_bands,
            )
            return float(total_loss.item()), float(abund_loss.item()), float(recon_loss.item())

        # Chunked: some models' per-sample intermediates (e.g. Mamba's selective
        # scan, which is 4D over batch/seq_len/d_inner/d_state) can OOM on a large
        # val/test split even though training batches of the same size are fine.
        # Each chunk's MSE is already a mean over a fixed per-sample element count,
        # so weighting by chunk size reproduces the true overall mean exactly.
        total_sum = abund_sum = recon_sum = 0.0
        for start in range(0, n_samples, batch_size):
            chunk_pixels = pixels[start : start + batch_size]
            chunk_abundances = true_abundances[start : start + batch_size]
            chunk_n = chunk_pixels.shape[0]
            total_loss, abund_loss, recon_loss, _ = self._compute_losses(
                pixels=chunk_pixels,
                true_abundances=chunk_abundances,
                endmembers_k_bands=endmembers_k_bands,
            )
            total_sum += float(total_loss.item()) * chunk_n
            abund_sum += float(abund_loss.item()) * chunk_n
            recon_sum += float(recon_loss.item()) * chunk_n

        return total_sum / n_samples, abund_sum / n_samples, recon_sum / n_samples

    def solve(
        self,
        endmembers: torch.Tensor,
        pixels: torch.Tensor,
        true_abundances: torch.Tensor,
        test_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cfg = self.cfg

        device = pixels.device
        dtype = pixels.dtype
        self._use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=self._use_amp)  # type: ignore[attr-defined]

        endmembers_k_bands = endmembers.to(device=device, dtype=dtype).contiguous()
        pixels_n_bands = pixels.to(device=device, dtype=dtype).contiguous()
        true_abundances_n_k = true_abundances.to(device=device, dtype=dtype).contiguous()

        n_samples = int(pixels_n_bands.shape[0])
        bands = int(pixels_n_bands.shape[1])
        n_endmembers = int(endmembers_k_bands.shape[0])

        if self.in_dim != bands or self.out_dim != n_endmembers:
            raise ValueError(
                f"{type(self).__name__} was initialized for in_dim={self.in_dim}, "
                f"out_dim={self.out_dim}, but solve received bands={bands}, "
                f"n_endmembers={n_endmembers}"
            )

        if test_indices is not None:
            # Test set is fixed by the caller so every model in the experiment is
            # scored on the exact same held-out pixels; only train/val (which this
            # model never gets scored on) are split randomly from what remains.
            test_idx = test_indices.to(device=device, dtype=torch.long)
            test_mask = torch.zeros(n_samples, dtype=torch.bool, device=device)
            test_mask[test_idx] = True
            remaining_idx = torch.nonzero(~test_mask, as_tuple=True)[0]
            remaining_idx = remaining_idx[torch.randperm(remaining_idx.numel(), device=device)]

            train_count = min(int(n_samples * 0.70), remaining_idx.numel())
            val_count = min(int(n_samples * 0.20), remaining_idx.numel() - train_count)

            train_idx = remaining_idx[:train_count]
            val_idx = remaining_idx[train_count : train_count + val_count]
        else:
            train_count, val_count, _ = self._split_counts(n_samples)
            split_indices = torch.randperm(n_samples, device=device)
            train_idx = split_indices[:train_count]
            val_idx = split_indices[train_count : train_count + val_count]
            test_idx = split_indices[train_count + val_count :]

        x_train = pixels_n_bands[train_idx]
        y_train = true_abundances_n_k[train_idx]

        x_val = pixels_n_bands[val_idx]
        y_val = true_abundances_n_k[val_idx]

        x_test = pixels_n_bands[test_idx]
        y_test = true_abundances_n_k[test_idx]

        model = self.model.to(device=device, dtype=dtype)
        self._prepare_model_for_training(x_train, y_train, endmembers_k_bands)

        # AdamW (decoupled weight decay) rather than Adam: with plain Adam, weight
        # decay is folded into the gradient before the adaptive step, which shrinks
        # differently-scaled parameter groups (e.g. KAN's per-edge spline
        # coefficients vs. a single base weight) inconsistently.
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )

        batch_size = max(1, min(int(cfg.batch_size), train_count))
        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, int(cfg.epochs) + 1):
            model.train()
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
                    endmembers_k_bands=endmembers_k_bands,
                )
                scaler.scale(total_loss).backward()
                if cfg.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += float(total_loss.item())
                train_abund_sum += float(abund_loss.item())
                train_recon_sum += float(recon_loss.item())
                num_batches += 1

            mean_train_loss = train_loss_sum / max(1, num_batches)
            mean_train_abund = train_abund_sum / max(1, num_batches)
            mean_train_recon = train_recon_sum / max(1, num_batches)

            model.eval()
            val_loss, val_abund_loss, val_recon_loss = self._evaluate_split(
                pixels=x_val,
                true_abundances=y_val,
                endmembers_k_bands=endmembers_k_bands,
                batch_size=batch_size,
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
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(cfg.patience):
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        self.best_val_loss = best_val_loss

        model.eval()
        self.test_loss, self.test_abund_loss, self.test_recon_loss = self._evaluate_split(
            pixels=x_test,
            true_abundances=y_test,
            endmembers_k_bands=endmembers_k_bands,
            batch_size=batch_size,
        )

        # Chunked rather than a single model(pixels_n_bands) call: with many samples
        # and no transform (raw high-band-count spectra), a one-shot forward over the
        # full dataset can allocate a multi-GB intermediate tensor and OOM the GPU.
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=self._use_amp):
            abundances_all = torch.cat(
                [
                    model(pixels_n_bands[start : start + batch_size])
                    for start in range(0, n_samples, batch_size)
                ],
                dim=0,
            )

        return abundances_all
