from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .sunsal import SunSAL, SunSALConfig

"""
Generalized Bilinear Model (GBM) implementation based on:

Halimi, A., Altmann, Y., Dobigeon, N., & Tourneret, J.-Y. (2011)
Nonlinear Unmixing of Hyperspectral Images Using a Generalized Bilinear Model.
IEEE TGRS, 49(11), 4153-4162.

Model (paper Eq. 3): each pixel is the usual linear mixture plus a pairwise
interaction term for every endmember pair (i, j), i < j, weighted by a
per-pair nonlinearity coefficient gamma_ij in [0, 1]:

    y = M @ alpha + sum_{i<j} gamma_ij * alpha_i * alpha_j * (m_i (.) m_j) + n

where (.) is the elementwise (Hadamard) product. gamma_ij = 0 for every pair
recovers the LMM; gamma_ij = 1 for every pair recovers the "Fan model" (see
this project's `core_math.mix_pixels`, which implements exactly this model
with a single *scalar* gamma shared by every pair - this module is the
per-pair-gamma inversion counterpart of that scalar forward generator).

The paper's contribution is not the model above but the *inversion*: a
hierarchical Bayesian estimator of theta = (alpha, gamma, sigma^2) per pixel,
approximated via a Metropolis-within-Gibbs sampler (paper Section IV,
Algorithm 1/2, Appendix A-C). This module follows that same three-block
structure - alpha, then every gamma_ij, then sigma^2, repeated for
n_burnin + n_samples iterations, MMSE-averaged over the post-burn-in samples
(Eq. 13) - vectorized across every pixel at once instead of the paper's
single-pixel-at-a-time MATLAB loop.

Two deliberate simplifications relative to the paper (documented here rather
than silently diverging, matching this project's convention in mlm.py):

1. gamma_ij conditional (Appendix A) is re-derived directly from the
   likelihood below rather than transcribed from the paper's printed
   summation limits verbatim, because those limits render ambiguously
   through OCR (`l=1,l=i ... p=l+1,p=j`) in a way that is not
   self-consistent with Eq. 3's dimensional structure (the printed
   posterior mean p_ij^T e_ij / ||p_ij||^2 lacks the alpha_i*alpha_j
   scaling that Eq. 3 requires - m_i (.) m_j is scaled by that product,
   not by gamma_ij alone). Re-deriving from the Gaussian likelihood in
   Eq. 5 directly gives an unambiguous, self-consistent conditional of
   the same family (truncated Gaussian on [0, 1]); see `_update_gamma`.
2. alpha_k conditional (Appendix B, Eq. 15) is an intractable density with
   a long list of helper terms (g, h, q, s, t, u, u', w, z, lambda, v) that
   is likewise very easy to mistranscribe from OCR and is not amenable to
   direct sampling in the paper either - the paper itself resorts to a
   Metropolis-Hastings step with a mode-centered truncated-Gaussian
   proposal (Eq. 16) whose mode requires that same helper-term algebra.
   This module keeps the paper's Metropolis-within-Gibbs *template*
   (Algorithm 2: propose, compute the exact posterior ratio, accept/reject)
   but substitutes a Uniform(0, alpha_k^+) independence-sampler proposal
   (alpha_k^+ defined exactly as the paper's Eq. 15 does: the simplex
   headroom left by every other coordinate). Because alpha_k^+ does not
   depend on alpha_k itself, the proposal density is identical whether
   evaluated at the current or candidate value, so the Hastings correction
   cancels and the acceptance ratio reduces to the raw likelihood ratio -
   still an exact Metropolis-Hastings step, just with a simpler proposal
   than the paper's. See `_update_alpha`.

sigma^2 (Appendix C) and the k* sum-to-one reparameterization (Section
III-B-1, Eq. 6-7) are simple and unambiguous, so they are implemented
exactly as written.
"""


def _std_normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF Phi(x), via the error function."""

    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _std_normal_icdf(u: torch.Tensor) -> torch.Tensor:
    """Standard normal inverse CDF, via the inverse error function."""

    u = u.clamp(1e-6, 1.0 - 1e-6)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


def _sample_truncated_normal_01(mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Sample N(mean, std^2) truncated to [0, 1], elementwise, via inverse-CDF sampling.

    This is the same basic principle the paper cites (Robert, 1995, [32]) for
    simulating truncated normal variables: map a uniform draw restricted to
    [Phi(a), Phi(b)] back through the inverse CDF, where a, b are the
    standardized truncation bounds.
    """

    std = std.clamp_min(eps)
    a = (0.0 - mean) / std
    b = (1.0 - mean) / std
    phi_a = _std_normal_cdf(a)
    phi_b = _std_normal_cdf(b)
    u = phi_a + (phi_b - phi_a) * torch.rand_like(mean)
    z = _std_normal_icdf(u)
    return (mean + std * z).clamp(0.0, 1.0)


@dataclass(slots=True)
class GBMConfig:
    """Configuration for the GBM Metropolis-within-Gibbs solver."""

    n_burnin: int = 300  # paper's N_bi (Section VI default)
    n_samples: int = 700  # paper's N_r (Section VI default)
    sunsal_mu: float = 0.05
    sunsal_max_iters: int = 300
    sunsal_tol: float = 1e-6
    check_every: int = 50
    eps: float = 1e-8
    verbose: bool = False


class GBM:
    """GBM Metropolis-within-Gibbs solver (Halimi et al., 2011)."""

    def __init__(self, config: GBMConfig):
        """Store configuration and initialize convergence-history buffers.

        Parameters
        ----------
        config : GBMConfig
            Solver configuration object.
        """

        self.cfg = config
        self.history: dict[str, list[float]] = {
            "sigma2_mean": [],
            "sigma2_std": [],
            "alpha_accept_rate": [],
        }

    def _forward(self, alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate mu_GBM = M @ alpha + sum_pairs gamma_ij * alpha_i * alpha_j * p_ij (Eq. 3, 5)."""

        linear = alpha @ self._endmembers
        alpha_products = alpha[:, self._pair_i] * alpha[:, self._pair_j]  # (n_pixels, n_pairs)
        nonlinear = (gamma * alpha_products) @ self._pair_products
        return linear + nonlinear

    def _update_alpha(self, y: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, sigma2: torch.Tensor) -> float:
        """Metropolis-within-Gibbs sweep over alpha_k, k != k* (Algorithm 1 lines 6-10, Algorithm 2).

        k* is re-randomized every call, exactly as Algorithm 1 line 6. For each
        k != k*, alpha_k^+ = 1 - sum_{j!=k} alpha_j (Eq. 15's simplex headroom)
        bounds a Uniform(0, alpha_k^+) independence-sampler proposal; see the
        module docstring for why this replaces the paper's Eq. 16 proposal.
        """

        n_pixels, n_endmembers = alpha.shape
        k_star = int(torch.randint(0, n_endmembers, (1,)).item())
        accepts = 0
        proposals = 0

        for k in range(n_endmembers):
            if k == k_star:
                continue

            others_sum = alpha.sum(dim=1) - alpha[:, k]
            alpha_k_plus = (1.0 - others_sum).clamp(min=0.0, max=1.0)

            candidate_value = alpha_k_plus * torch.rand(n_pixels, device=alpha.device, dtype=alpha.dtype)
            alpha_candidate = alpha.clone()
            alpha_candidate[:, k] = candidate_value

            mu_current = self._forward(alpha, gamma)
            mu_candidate = self._forward(alpha_candidate, gamma)
            ss_current = ((y - mu_current) ** 2).sum(dim=1)
            ss_candidate = ((y - mu_candidate) ** 2).sum(dim=1)

            # Posterior ratio under the likelihood (Eq. 5/12); uniform prior and
            # uniform independence-sampler proposal both cancel exactly (see
            # module docstring point 2), leaving just the likelihood ratio.
            log_ratio = (ss_current - ss_candidate) / (2.0 * sigma2)
            u = torch.rand(n_pixels, device=alpha.device, dtype=alpha.dtype)
            accept_mask = (log_ratio >= 0.0) | (u < torch.exp(log_ratio.clamp(max=0.0)))

            alpha[:, k] = torch.where(accept_mask, candidate_value, alpha[:, k])
            accepts += int(accept_mask.sum().item())
            proposals += n_pixels

        # Line 10: alpha_k* = 1 - sum_{i!=k*} alpha_i, enforcing the sum-to-one
        # constraint (Eq. 6-7) exactly after every other coordinate is fixed.
        others_sum = alpha.sum(dim=1) - alpha[:, k_star]
        alpha[:, k_star] = (1.0 - others_sum).clamp(min=0.0, max=1.0)

        return accepts / max(proposals, 1)

    def _update_gamma(self, y: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor, sigma2: torch.Tensor) -> None:
        """Direct Gibbs draw of every gamma_ij|... in turn (Appendix A conditional, re-derived).

        For fixed alpha and every other gamma, Eq. 3/5 is linear-Gaussian in
        gamma_ij with effective design vector q_ij = alpha_i * alpha_j * p_ij
        (p_ij = m_i (.) m_j), giving the truncated-Gaussian conditional

            gamma_ij | ... ~ N_[0,1]( p_ij^T e_ij / (alpha_i alpha_j ||p_ij||^2),
                                       sigma^2 / (alpha_i alpha_j)^2 ||p_ij||^2 )

        where e_ij is the residual with only this pair's own contribution
        removed (added back from the running residual below, then
        subtracted again after the new draw, so the residual stays exact
        for the next pair without a fresh forward pass each time).
        """

        eps = self.cfg.eps
        residual = y - self._forward(alpha, gamma)

        for m in range(self._pair_products.shape[0]):
            i = int(self._pair_i[m])
            j = int(self._pair_j[m])
            p = self._pair_products[m]  # (n_bands,)
            p_norm_sq = (p * p).sum()

            alpha_prod = alpha[:, i] * alpha[:, j]  # (n_pixels,)
            e_ij = residual + gamma[:, m : m + 1] * (alpha_prod.unsqueeze(1) * p.unsqueeze(0))

            denom = (alpha_prod * alpha_prod) * p_norm_sq
            denom_safe = denom.clamp_min(eps)
            mean = (e_ij @ p) / denom_safe
            std = torch.sqrt(sigma2 / denom_safe)

            new_gamma = _sample_truncated_normal_01(mean, std, eps=eps)
            residual = e_ij - new_gamma.unsqueeze(1) * (alpha_prod.unsqueeze(1) * p.unsqueeze(0))
            gamma[:, m] = new_gamma

    def _update_sigma2(self, residual: torch.Tensor, n_bands: int) -> torch.Tensor:
        """Exact Gibbs draw of sigma^2|y, alpha, gamma ~ IG(L/2, ||y - mu_GBM||^2 / 2) (Appendix C).

        Sampled via 1 / Gamma(shape=L/2, rate=||residual||^2/2), since
        X ~ Gamma(a, rate=b) implies 1/X ~ InverseGamma(a, b).
        """

        ss = (residual ** 2).sum(dim=1).clamp_min(self.cfg.eps)
        shape = torch.full_like(ss, n_bands / 2.0)
        rate = ss / 2.0
        gamma_draw = torch.distributions.Gamma(concentration=shape, rate=rate).sample()
        return (1.0 / gamma_draw).clamp_min(self.cfg.eps)

    def solve(self, endmembers: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        """Solve abundances for pixels under the GBM via Metropolis-within-Gibbs sampling.

        Parameters
        ----------
        endmembers : torch.Tensor
            Endmember matrix with shape ``(n_endmembers, n_bands)``.
        pixels : torch.Tensor
            Pixel matrix with shape ``(n_pixels, n_bands)``.

        Returns
        -------
        torch.Tensor
            MMSE-estimated abundances with shape ``(n_pixels, n_endmembers)``
            (Eq. 13, averaged over the post-burn-in samples and renormalized
            to sum to one).
        """

        device, dtype = pixels.device, pixels.dtype
        n_pixels, n_bands = pixels.shape
        n_endmembers = endmembers.shape[0]

        self._endmembers = endmembers
        pair_i, pair_j = [], []
        for i in range(n_endmembers):
            for j in range(i + 1, n_endmembers):
                pair_i.append(i)
                pair_j.append(j)
        self._pair_i = torch.tensor(pair_i, dtype=torch.long, device=device)
        self._pair_j = torch.tensor(pair_j, dtype=torch.long, device=device)
        self._pair_products = endmembers[self._pair_i] * endmembers[self._pair_j]  # (n_pairs, n_bands)
        n_pairs = self._pair_products.shape[0]

        # Warm start: alpha^(0) from the project's SunSAL solver (satisfies ANC/ASC
        # already, Algorithm 1's initialization just needs *a* feasible point) and
        # gamma^(0) = 0 (i.e. start the chain at the LMM, Eq. 3's gamma=0 case).
        sunsal = SunSAL(SunSALConfig(μ=self.cfg.sunsal_mu, max_iters=self.cfg.sunsal_max_iters, tol=self.cfg.sunsal_tol))
        alpha = sunsal.solve(endmembers, pixels).clone()
        gamma = torch.zeros(n_pixels, n_pairs, device=device, dtype=dtype)

        initial_residual = pixels - self._forward(alpha, gamma)
        sigma2 = (initial_residual ** 2).mean(dim=1).clamp_min(self.cfg.eps)

        alpha_sum = torch.zeros_like(alpha)
        gamma_sum = torch.zeros_like(gamma)
        sigma2_sum = torch.zeros_like(sigma2)

        n_total = self.cfg.n_burnin + self.cfg.n_samples
        for t in range(n_total):
            accept_rate = self._update_alpha(pixels, alpha, gamma, sigma2)
            self._update_gamma(pixels, alpha, gamma, sigma2)
            residual = pixels - self._forward(alpha, gamma)
            sigma2 = self._update_sigma2(residual, n_bands)

            if t >= self.cfg.n_burnin:
                alpha_sum += alpha
                gamma_sum += gamma
                sigma2_sum += sigma2

            if (t + 1) % self.cfg.check_every == 0:
                self.history["sigma2_mean"].append(float(sigma2.mean().item()))
                self.history["sigma2_std"].append(float(sigma2.std().item()))
                self.history["alpha_accept_rate"].append(float(accept_rate))
                if self.cfg.verbose:
                    print(
                        f"Iter {t + 1:4d} | sigma2 mean={sigma2.mean().item():.3e} | "
                        f"alpha accept={accept_rate:.3f}"
                    )

        n_r = max(self.cfg.n_samples, 1)
        abundances = alpha_sum / n_r  # Eq. 13 MMSE estimate
        abundances = abundances / abundances.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)

        self.gamma_ = gamma_sum / n_r
        self.sigma2_ = sigma2_sum / n_r
        return abundances
