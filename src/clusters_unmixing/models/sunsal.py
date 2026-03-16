from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

"""
SUnSAL implementation based on:

Bioucas-Dias, J., & Figueiredo, M. (2010)
Alternating Direction Algorithms for Constrained Sparse Regression:
Application to Hyperspectral Unmixing.
IEEE JSTSP.
"""


def soft_threshold(x: torch.Tensor, tau: float) -> torch.Tensor:
    """Apply elementwise soft-thresholding.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    tau : float
        Non-negative shrinkage threshold.

    Returns
    -------
    torch.Tensor
        Thresholded tensor with the same shape as ``x``.
    """

    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 2D with shape ``(N, D)``.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor expected to be 1D or 2D.

    Returns
    -------
    torch.Tensor
        2D tensor, where 1D input is promoted with a leading dimension.
    """

    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x
    raise ValueError(f"Expected 1D or 2D tensor, got shape {tuple(x.shape)}")


@dataclass(slots=True)
class SunSALConfig:
    """Configuration for the SUnSAL ADMM solver."""

    mu: float = 0.05
    lambda_reg: float = 0.0
    max_iters: int = 500
    tol: float = 1e-6
    check_every: int = 10
    verbose: bool = False
    eps: float = 1e-9


class SunSAL:
    """SUnSAL solver for constrained sparse regression (ANC + ASC)."""

    def __init__(self, config: SunSALConfig):
        """Store configuration and initialize convergence history buffers.

        Parameters
        ----------
        config : SunSALConfig
            Solver configuration object.

        Returns
        -------
        None
            Initializes solver state in-place.
        """

        self.cfg = config
        self.history: dict[str, list[float | int]] = {
            "iters": [],
            "max_delta_x": [],
            "primal_res": [],
        }

    @torch.no_grad()
    def _solve_core(
        self,
        endmembers: torch.Tensor,
        pixels: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve abundance estimation on transformed inputs.

        Parameters
        ----------
        self : SunSAL
            Solver instance.
        endmembers : torch.Tensor
            Endmember matrix with shape ``(n_bands, n_endmembers)``.
        pixels : torch.Tensor
            Pixel matrix with shape ``(n_pixels, n_bands)``.
        x0 : Optional[torch.Tensor], optional
            Optional initial abundance estimate with shape
            ``(n_pixels, n_endmembers)``.

        Returns
        -------
        torch.Tensor
            Estimated abundances with shape ``(n_pixels, n_endmembers)``.
        """

        cfg = self.cfg
        device = endmembers.device
        dtype = endmembers.dtype

        measurements = ensure_2d(pixels).to(device=device, dtype=dtype)
        bands, n_endmembers = endmembers.shape
        n_pixels = measurements.shape[0]

        if measurements.shape[1] != bands:
            raise ValueError(
                f"pixels has features={measurements.shape[1]} but endmembers has features={bands}"
            )

        y = measurements.T

        if x0 is None:
            x = torch.full(
                (n_endmembers, n_pixels),
                1.0 / n_endmembers,
                device=device,
                dtype=dtype,
            )
        else:
            x0_2d = ensure_2d(x0)
            if x0_2d.shape != (n_pixels, n_endmembers):
                raise ValueError(
                    f"x0 must have shape ({n_pixels}, {n_endmembers}), got {tuple(x0_2d.shape)}"
                )
            x = x0_2d.T.to(device=device, dtype=dtype).contiguous()

        u = x.clone()
        d = torch.zeros_like(x)

        at = endmembers.T
        b = at @ endmembers + cfg.mu * torch.eye(n_endmembers, device=device, dtype=dtype)

        chol_ok = True
        try:
            chol = torch.linalg.cholesky(b)
        except Exception:
            chol_ok = False
            chol = None

        def solve_b(rhs: torch.Tensor) -> torch.Tensor:
            """Solve linear system against the ADMM matrix.

            Parameters
            ----------
            rhs : torch.Tensor
                Right-hand side matrix.

            Returns
            -------
            torch.Tensor
                Linear solve result with the same leading dimensions as ``rhs``.
            """

            if chol_ok:
                return torch.cholesky_solve(rhs, chol)
            return torch.linalg.solve(b, rhs)

        ones = torch.ones((n_endmembers, 1), device=device, dtype=dtype)
        b_inv_ones = solve_b(ones)
        denom = (ones.T @ b_inv_ones).squeeze().clamp_min(cfg.eps)
        c = b_inv_ones / denom

        x_prev = x.clone()
        check_every = max(1, int(cfg.check_every))

        for iteration in range(int(cfg.max_iters)):
            w = at @ y + cfg.mu * (u + d)
            b_inv_w = solve_b(w)

            s = ones.T @ b_inv_w
            x = b_inv_w - c @ (s - 1.0)

            v = x - d
            tau = cfg.lambda_reg / max(cfg.mu, cfg.eps)
            u = torch.clamp(soft_threshold(v, tau), min=0.0)

            residual = x - u
            d = d - residual

            if (iteration + 1) % check_every == 0:
                delta = torch.max(torch.abs(x - x_prev)).item()
                primal = torch.mean(residual * residual).sqrt().item()

                self.history["iters"].append(iteration + 1)
                self.history["max_delta_x"].append(delta)
                self.history["primal_res"].append(primal)

                if cfg.verbose:
                    print(f"Iter {iteration + 1:4d} | maxΔx={delta:.3e} | primal={primal:.3e}")

                if delta <= cfg.tol:
                    break

                x_prev = x.clone()

        return x.T.contiguous()

    def solve(
        self,
        endmembers: torch.Tensor,
        pixels: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve abundances for pixels.

        Parameters
        ----------
        self : SunSAL
            Solver instance.
        endmembers : torch.Tensor
            Endmember matrix with shape ``(n_bands, n_endmembers)``.
        pixels : torch.Tensor
            Pixel matrix with shape ``(n_pixels, n_bands)``.
        x0 : Optional[torch.Tensor], optional
            Optional initial abundance estimate.

        Returns
        -------
        torch.Tensor
            Estimated abundances with shape ``(n_pixels, n_endmembers)``.
        """

        return self._solve_core(endmembers, pixels, x0=x0)
