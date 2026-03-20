from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

'''
Implementation according to the paper:

A Stepwise Analytical Projected Gradient Descent Search for Hyperspectral Unmixing and Its Code Vectorization

by Fadi Kizel et all.
'''



@dataclass
class VPGDUConfig:
    """Configuration for the VPGDU solver."""

    initial_estimator_num_samples: int = 500
    threshold: float = 1e-6
    max_iters: int = 500
    t: int = 10
    verbose: bool = False


class VPGDU:
    """Vectorized Projected Gradient Descent Unmixing (Algorithm 2)."""

    def __init__(self, config: VPGDUConfig):
        """Store solver configuration and initialize iteration history.

        Parameters
        ----------
        config : VPGDUConfig
            Solver configuration object.

        Returns
        -------
        None
            Initializes solver state in-place.
        """

        self.cfg = config
        self.history = {"iterations": [], "active_pixels": []}

    def simplex_projection(self, values: torch.Tensor) -> torch.Tensor:
        """Project vectors onto probability simplex (Chen & Ye).

        Parameters
        ----------
        values : torch.Tensor
            Matrix of shape ``(n_endmembers, n_vectors)`` to project.

        Returns
        -------
        torch.Tensor
            Projected matrix with the same shape as ``values``.
        """

        n_dim, n_vec = values.shape
        device = values.device

        sorted_vals, _ = torch.sort(values, dim=0, descending=True)
        cumulative = torch.cumsum(sorted_vals, dim=0)
        index = torch.arange(1, n_dim + 1, device=device).view(n_dim, 1)

        cond = sorted_vals - (cumulative - 1) / index > 0
        rho = cond.sum(dim=0)
        rho_index = (rho - 1).view(1, n_vec)
        cumulative_rho = torch.gather(cumulative, 0, rho_index)
        theta = (cumulative_rho - 1) / rho.view(1, n_vec)
        return torch.clamp(values - theta, min=0)

    def _preprocess_initial_estimator(
        self,
        endmembers: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit linear functions between fractions and normalized SAM values.

        Parameters
        ----------
        self : VPGDU
            Solver instance.
        endmembers : torch.Tensor
            Endmember matrix with shape ``(n_endmembers, n_bands)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Pair ``(alpha, beta)`` used by the initial abundance estimator.
        """

        device = endmembers.device
        n_endmembers, _ = endmembers.shape

        alpha_list = []
        beta_list = []

        for endmember_index in range(n_endmembers):
            fractions = torch.linspace(
                0.0,
                1.0,
                self.cfg.initial_estimator_num_samples,
                device=device,
            )
            ns_values = torch.zeros(self.cfg.initial_estimator_num_samples, device=device)

            for sample_index, fraction_i in enumerate(fractions):
                remaining_fraction = 1.0 - fraction_i
                if remaining_fraction > 0 and n_endmembers > 1:
                    other = torch.rand(n_endmembers - 1, device=device)
                    other = other / other.sum() * remaining_fraction
                else:
                    other = torch.zeros(n_endmembers - 1, device=device)

                abundance = torch.zeros(n_endmembers, device=device)
                abundance[endmember_index] = fraction_i
                other_cursor = 0
                for j in range(n_endmembers):
                    if j != endmember_index:
                        abundance[j] = other[other_cursor]
                        other_cursor += 1

                mixed = abundance @ endmembers

                sam_values = torch.zeros(n_endmembers, device=device)
                for inner_index in range(n_endmembers):
                    em = endmembers[inner_index]
                    denominator = torch.norm(em) * torch.norm(mixed) + 1e-9
                    cosine = torch.dot(em, mixed) / denominator
                    cosine = torch.clamp(cosine, -1.0, 1.0)
                    sam_values[inner_index] = torch.acos(cosine)

                ns_values[sample_index] = sam_values[endmember_index] / (sam_values.sum() + 1e-9)

            design = torch.stack([ns_values, torch.ones_like(ns_values)], dim=1)
            target = fractions.unsqueeze(1)
            solution = torch.linalg.lstsq(design, target).solution
            alpha_list.append(solution[0, 0])
            beta_list.append(solution[1, 0])

        alpha = torch.stack(alpha_list)
        beta = torch.stack(beta_list)
        return alpha, beta

    def _get_initial_abundance_estimates(
        self,
        pixels: torch.Tensor,
        endmembers: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Compute initial abundance estimates via linear model on normalized SAM.

        Parameters
        ----------
        self : VPGDU
            Solver instance.
        pixels : torch.Tensor
            Pixel matrix with shape ``(n_pixels, n_bands)``.
        endmembers : torch.Tensor
            Endmember matrix with shape ``(n_endmembers, n_bands)``.
        alpha : torch.Tensor
            Per-endmember linear coefficient for normalized SAM values.
        beta : torch.Tensor
            Per-endmember linear intercept.

        Returns
        -------
        torch.Tensor
            Initial abundance matrix with shape ``(n_pixels, n_endmembers)``.
        """

        dot_products = pixels @ endmembers.T
        norm_pixels = torch.norm(pixels, dim=1, keepdim=True)
        norm_endmembers = torch.norm(endmembers, dim=1, keepdim=True).T

        denominators = norm_pixels @ norm_endmembers
        cosines = dot_products / (denominators + 1e-9)
        cosines = torch.clamp(cosines, -1.0, 1.0)

        sam_values = torch.acos(cosines)
        sam_sums = sam_values.sum(dim=1, keepdim=True) + 1e-9
        normalized_sam = sam_values / sam_sums

        abundance_init = alpha.unsqueeze(0) * normalized_sam + beta.unsqueeze(0)
        abundance_init = torch.clamp(abundance_init, min=0.0)

        abundance_sums = abundance_init.sum(dim=1, keepdim=True)
        needs_norm = (abundance_sums > 1.0).squeeze()
        if needs_norm.any():
            abundance_init[needs_norm] = abundance_init[needs_norm] / abundance_sums[needs_norm]

        return abundance_init

    def _solve_core(self, endmembers: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        """Solve the constrained unmixing problem.

        Parameters
        ----------
        self : VPGDU
            Solver instance.
        endmembers : torch.Tensor
            Endmember matrix with shape ``(n_endmembers, n_bands)``.
        pixels : torch.Tensor
            Pixel matrix with shape ``(n_pixels, n_bands)``.

        Returns
        -------
        torch.Tensor
            Estimated abundances with shape ``(n_pixels, n_endmembers)``.
        """

        self.endmembers = endmembers
        self.device = endmembers.device
        n_endmembers, _ = endmembers.shape

        alpha, beta = self._preprocess_initial_estimator(endmembers)
        abundance_init = self._get_initial_abundance_estimates(pixels, endmembers, alpha, beta)

        n_pixels, _ = pixels.shape

        measurements = pixels
        abundance_hat = abundance_init.T

        active_mask = torch.ones(n_pixels, dtype=torch.bool, device=self.device)
        pixel_indices = torch.arange(n_pixels, device=self.device)
        abundance_prev = abundance_hat.clone()
        endmembers_t_endmembers = self.endmembers @ self.endmembers.T

        for iteration in range(self.cfg.max_iters):
            measurements_active = measurements[active_mask]
            abundance_active = abundance_hat[:, active_mask]

            if measurements_active.shape[0] == 0:
                if self.cfg.verbose:
                    print(f"All pixels converged at iteration {iteration}")
                break

            grad_1 = self.endmembers @ measurements_active.T
            endmembers_times_abundance = abundance_active.T @ self.endmembers
            sum_sq = torch.sum(endmembers_times_abundance * endmembers_times_abundance, dim=1, keepdim=True).T
            grad_2 = sum_sq.expand(n_endmembers, -1)
            grad_3 = endmembers_t_endmembers @ abundance_active
            dot_m_ef = torch.sum(measurements_active * endmembers_times_abundance, dim=1, keepdim=True).T
            grad_4 = dot_m_ef.expand(n_endmembers, -1)
            norm_m = torch.sqrt(torch.sum(measurements_active * measurements_active, dim=1, keepdim=True)).T
            grad_5 = norm_m.expand(n_endmembers, -1)
            grad_n = grad_1 * grad_2 - grad_3 * grad_4
            grad_d = grad_5 * (grad_2.pow(1.5))
            grad_phi = grad_n / (grad_d + 1e-9)

            delta_1 = dot_m_ef
            delta_2 = torch.sum(grad_phi * grad_3, dim=0, keepdim=True)
            delta_3 = torch.sum(abundance_active * grad_3, dim=0, keepdim=True)
            endmembers_t_endmembers_grad = endmembers_t_endmembers @ grad_phi
            delta_4 = torch.sum(grad_phi * endmembers_t_endmembers_grad, dim=0, keepdim=True)
            delta_5 = torch.sum(grad_phi * grad_1, dim=0, keepdim=True)
            delta_n = delta_2 * delta_1 - delta_5 * delta_3
            delta_d = delta_5 * delta_2 - delta_4 * delta_1
            step_size = delta_n / (delta_d + 1e-9)
            delta_k = step_size.expand(n_endmembers, -1)

            unprojected = abundance_active + delta_k * grad_phi
            abundance_next_active = self.simplex_projection(unprojected)

            current_indices = pixel_indices[active_mask]
            abundance_hat[:, current_indices] = abundance_next_active

            if (iteration + 1) % self.cfg.t == 0:
                abundance_curr_active = abundance_hat[:, current_indices]
                abundance_prev_active = abundance_prev[:, current_indices]
                diff = torch.max(torch.abs(abundance_curr_active - abundance_prev_active), dim=0)[0]
                converged_local = diff <= self.cfg.threshold
                converged_global = current_indices[converged_local]
                active_mask[converged_global] = False

                self.history["iterations"].append(iteration + 1)
                self.history["active_pixels"].append(int(active_mask.sum().item()))

                abundance_prev = abundance_hat.clone()

        return abundance_hat.T

    def solve(self, endmembers: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        """Solve abundances for pixels given endmembers under selected transform.

        Parameters
        ----------
        self : VPGDU
            Solver instance.
        endmembers : torch.Tensor
            Endmember matrix with shape ``(n_endmembers, n_bands)``.
        pixels : torch.Tensor
            Pixel matrix with shape ``(n_pixels, n_bands)``.

        Returns
        -------
        torch.Tensor
            Estimated abundances with shape ``(n_pixels, n_endmembers)``.
        """

        return self._solve_core(endmembers, pixels)
