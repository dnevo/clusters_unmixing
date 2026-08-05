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

with the *inversion* following the authors' companion gradient-descent
algorithm (GDA):

Halimi, A., Altmann, Y., Dobigeon, N., & Tourneret, J.-Y. (2011)
Unmixing Hyperspectral Images Using the Generalized Bilinear Model.
IEEE IGARSS 2011 (and the authors' reference MATLAB code `GBM_unmix_GDA`:
GBM_gradient.m / gene_Gamma.m / golden_section.m).

Model (TGRS paper Eq. 3): each pixel is the usual linear mixture plus a
pairwise interaction term for every endmember pair (i, j), i < j, weighted by
a per-pair nonlinearity coefficient gamma_ij in [0, 1]:

    y = M @ alpha + sum_{i<j} gamma_ij * alpha_i * alpha_j * (m_i (.) m_j) + n

where (.) is the elementwise (Hadamard) product. gamma_ij = 0 for every pair
recovers the LMM; gamma_ij = 1 for every pair recovers the "Fan model" (see
this project's `core_math.mix_pixels`, which implements exactly this model:
for `nonlinearity_gamma > 0` it draws its own gamma_ij independently from
Uniform(0, nonlinearity_gamma) per pixel per pair - the same granularity this
module's inversion fits, so the fitted gamma_ij is an attempt to recover the
generative ground truth, not an over-parameterization of it).

The inversion minimizes the squared reconstruction error

    J(theta) = ||y - mu_GBM(theta)||^2,   theta = (alpha_1..alpha_{R-1}, gamma)

by projected gradient descent with an exact constrained line search. The
sum-to-one constraint is handled by elimination: alpha_R = 1 - sum_{r<R}
alpha_r is a dependent coordinate, so the gradient is taken with respect to
the R-1 free abundances only (with d alpha_R / d alpha_r = -1 folded in) and
the constraint holds exactly at every iterate by construction. Each iteration
is: (1) analytical gradient of J w.r.t. the free abundances and every
gamma_ij; (2) boundary "gradient clipping" so the step cannot immediately
violate an active constraint; (3) per-pixel feasible step-size interval
[lb, ub] derived from the box constraints alpha_r in [0, 1], gamma_ij in
[0, 1] and sum_{r<R} alpha_r in [0, 1]; (4) golden-section line search for
the step size within that interval; (5) update and per-pixel convergence
test |J_t - J_{t-1}| < tol. This mirrors the reference MATLAB implementation
step for step, but is vectorized across every pixel at once (each pixel
carries its own gradient, bounds, line-search state, and convergence flag;
converged pixels are frozen and dropped from subsequent iterations) instead
of the MATLAB single-pixel-at-a-time loop.

Three deliberate deviations from the reference MATLAB code (documented here
rather than silently diverging, matching this project's convention in
mlm.py):

1. Zero-derivative handling in the step-size bounds. The MATLAB code forms
   each variable's contribution to [lb, ub] as
   `min(v/d*(d~=0), (v-1)/d)*(d~=0)` etc., which for d == 0 collapses the
   contribution to 0 (after MATLAB's Inf/NaN arithmetic washes through the
   masking products) - i.e. a zero-derivative variable can spuriously clamp
   ub to 0 and kill the step for the whole pixel. The clear *intent* is that
   a variable whose derivative is zero is unaffected by the step and should
   impose no constraint at all; this module implements that intent directly
   (zero-derivative variables contribute -inf/+inf to the running max/min).
2. Warm start. The MATLAB code initializes abundances with per-pixel FCLS
   (hyperFcls.m); this module uses the project's SunSAL solver instead,
   which solves the same fully-constrained problem for every pixel at once
   and is already this project's standard warm start. gamma^(0) = 0.01 for
   every pair, exactly as the MATLAB (`0.01*ones`), i.e. the chain starts
   essentially at the LMM.
3. Golden-section bookkeeping. The scalar MATLAB loop re-uses one interior
   evaluation per shrink and tracks which side moved with a flag; the
   vectorized version keeps the same one-new-evaluation-per-iteration
   economy, but selects the re-used point per pixel with masks, since
   different pixels shrink different sides in the same iteration. The
   iteration count is additionally capped (`ls_max_iters`) as a numerical
   safety net; at the golden ratio's 0.618 shrink factor the default cap is
   far beyond what the interval tolerance needs.
"""


_GOLDEN = 0.618  # golden-section shrink ratio used by the reference code


def _project_to_simplex(alpha: torch.Tensor, eps: float) -> torch.Tensor:
    """Clamp to non-negative and renormalize to sum one, rowwise.

    Used only on the warm start. The reference code initializes with FCLS,
    whose output is non-negative by construction; this project's SunSAL
    returns the ADMM split variable that satisfies the sum-to-one constraint
    exactly but may carry small negative entries (the non-negative copy is
    the solver's internal ``u``). A negative coordinate is infeasible for the
    step-size bounds below - the feasible interval for that variable would be
    empty, collapsing ``ub`` to 0 and freezing the pixel - so the warm start
    is projected onto the simplex first. Clamping then renormalizing keeps
    every entry in [0, 1] and the row sum at exactly 1.
    """

    alpha = alpha.clamp_min(0.0)
    total = alpha.sum(dim=1, keepdim=True)
    uniform = torch.full_like(alpha, 1.0 / alpha.shape[1])
    return torch.where(total > eps, alpha / total.clamp_min(eps), uniform)


@dataclass(slots=True)
class GBMConfig:
    """Configuration for the GBM projected-gradient solver."""

    max_iters: int = 900  # reference code's Niter
    tol: float = 1e-5  # reference code's |DER| stopping threshold
    gamma_init: float = 0.01  # reference code's gamma initialization
    boundary_eps: float = 0.01  # reference code's active-constraint band (0.99 / 0.01 tests)
    ls_tol: float = 1e-6  # golden_section.m's interval tolerance l
    ls_max_iters: int = 120  # vectorization safety cap (0.618^120 ~ 1e-25)
    sunsal_mu: float = 0.05
    sunsal_max_iters: int = 300
    sunsal_tol: float = 1e-6
    check_every: int = 10
    eps: float = 1e-12
    verbose: bool = False


class GBM:
    """GBM projected-gradient solver with golden-section line search (Halimi et al., 2011, GDA)."""

    def __init__(self, config: GBMConfig):
        """Store configuration and initialize convergence-history buffers.

        Parameters
        ----------
        config : GBMConfig
            Solver configuration object.
        """

        self.cfg = config
        self.history: dict[str, list[float]] = {
            "error_mean": [],
            "error_max_delta": [],
            "active_fraction": [],
        }

    def _forward(self, alpha: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """Evaluate mu_GBM = M @ alpha + sum_pairs gamma_ij * alpha_i * alpha_j * p_ij (Eq. 3).

        This is the vectorized counterpart of the reference gene_Gamma.m.
        """

        linear = alpha @ self._endmembers
        alpha_products = alpha[:, self._pair_i] * alpha[:, self._pair_j]  # (n_pixels, n_pairs)
        nonlinear = (gamma * alpha_products) @ self._pair_products
        return linear + nonlinear

    def _gradients(
        self, y: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Analytical gradient of J = ||y - mu_GBM||^2 / 2-free scaling, per pixel.

        Returns (df_alpha, df_gamma, err) where df_alpha has shape
        (n_pixels, R-1) - the derivative w.r.t. the free abundances with the
        dependent alpha_R eliminated - df_gamma has shape (n_pixels, n_pairs),
        and err is the current squared reconstruction error per pixel.

        Derivation (matches GBM_gradient.m's d_alpha/d_gamma loops): let
        g_r = d mu / d alpha_r treating *all* R abundances as free,

            g_r = m_r + sum_{i != r} Gamma_ri * alpha_i * (m_r (.) m_i),

        with Gamma the symmetric per-pixel gamma matrix (zero diagonal).
        Eliminating alpha_R = 1 - sum_{r<R} alpha_r contributes a factor
        d alpha_R / d alpha_r = -1, so the constrained derivative is simply
        g_r - g_R; expanding that difference reproduces, term by term, the
        reference code's explicit m_r - m_R, Gam(r,i), Gam(i,R), and
        Gam(r,R)*(alpha_R - alpha_r) pieces. The gamma derivative is direct:
        d mu / d gamma_ij = alpha_i * alpha_j * (m_i (.) m_j). The chain rule
        through J then contracts each of these with the residual (mu - y),
        exactly the reference code's (y0 - y)' * d_alpha and
        (y0 - y)' * d_gamma products.
        """

        n_pixels, n_endmembers = alpha.shape
        mu = self._forward(alpha, gamma)
        residual = mu - y  # (n_pixels, n_bands); reference code's (y0 - y)

        # Symmetric per-pixel gamma matrix (reference code's Gam_sq).
        gamma_mat = torch.zeros(
            n_pixels, n_endmembers, n_endmembers, device=alpha.device, dtype=alpha.dtype
        )
        gamma_mat[:, self._pair_i, self._pair_j] = gamma
        gamma_mat[:, self._pair_j, self._pair_i] = gamma

        # g_r for every r at once: (n_pixels, R, n_bands).
        weights = gamma_mat * alpha.unsqueeze(1)  # (n_pixels, R, R): Gamma_ri * alpha_i
        g_full = self._endmembers.unsqueeze(0) + torch.einsum(
            "nri,ril->nrl", weights, self._cross_products
        )
        d_alpha = g_full[:, :-1, :] - g_full[:, -1:, :]  # eliminate alpha_R: (n_pixels, R-1, n_bands)

        df_alpha = torch.einsum("nl,nrl->nr", residual, d_alpha)  # (n_pixels, R-1)

        alpha_products = alpha[:, self._pair_i] * alpha[:, self._pair_j]  # (n_pixels, n_pairs)
        # d mu / d gamma_ij contracted with the residual, all pairs at once.
        df_gamma = alpha_products * (residual @ self._pair_products.T)  # (n_pixels, n_pairs)

        err = (residual**2).sum(dim=1)
        return df_alpha, df_gamma, err

    def _clip_boundary_gradients(self, alpha: torch.Tensor, df_alpha: torch.Tensor) -> torch.Tensor:
        """Zero out gradient components that would immediately push an active constraint (GBM_gradient.m's "move optimization").

        Two rules, vectorized over pixels:

        1. If sum_{r<R} alpha_r > 1 - boundary_eps (i.e. alpha_R is nearly 0)
           and the step direction -df_alpha would *increase* that sum
           (sum(df_alpha) < 0), the last free component is overwritten so the
           gradient sums to zero and the step leaves alpha_R unchanged.
        2. Any free abundance already below boundary_eps whose gradient
           component is positive (step would push it further below 0) has
           that component zeroed.
        """

        df_alpha = df_alpha.clone()

        free_sum = alpha[:, :-1].sum(dim=1)
        grad_sum = df_alpha.sum(dim=1)
        rule1 = (free_sum > 1.0 - self.cfg.boundary_eps) & (grad_sum < 0.0)
        if rule1.any():
            df_alpha[rule1, -1] = -df_alpha[rule1, :-1].sum(dim=1)

        rule2 = (alpha[:, :-1] < self.cfg.boundary_eps) & (df_alpha > 0.0)
        df_alpha = torch.where(rule2, torch.zeros_like(df_alpha), df_alpha)
        return df_alpha

    def _step_bounds(
        self, alpha: torch.Tensor, gamma: torch.Tensor, df_alpha: torch.Tensor, df_gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pixel feasible step interval [lb, ub] for the update theta - lambda * df.

        Each box-constrained variable v in [0, 1] with derivative d != 0
        keeps v - lambda * d in [0, 1] for lambda between v/d and (v-1)/d
        (in whichever order); the feasible interval is the intersection over
        every free abundance, every gamma_ij, and the free-abundance *sum*
        (which is what keeps the dependent alpha_R in [0, 1]). Variables with
        d == 0 impose no constraint (see module docstring, deviation 1).
        Finally both bounds are clamped to lambda >= 0, as in the reference
        code, so the search never moves against the descent direction.
        """

        eps = self.cfg.eps

        def interval(v: torch.Tensor, d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            active = d.abs() > eps
            d_safe = torch.where(active, d, torch.ones_like(d))
            r0 = v / d_safe
            r1 = (v - 1.0) / d_safe
            lo = torch.minimum(r0, r1)
            hi = torch.maximum(r0, r1)
            lo = torch.where(active, lo, torch.full_like(lo, -torch.inf))
            hi = torch.where(active, hi, torch.full_like(hi, torch.inf))
            return lo, hi

        lo_a, hi_a = interval(alpha[:, :-1], df_alpha)
        lo_g, hi_g = interval(gamma, df_gamma)
        lo_s, hi_s = interval(
            alpha[:, :-1].sum(dim=1, keepdim=True), df_alpha.sum(dim=1, keepdim=True)
        )

        lb = torch.cat([lo_a, lo_g, lo_s], dim=1).amax(dim=1)
        ub = torch.cat([hi_a, hi_g, hi_s], dim=1).amin(dim=1)

        lb = lb.clamp_min(0.0)
        ub = torch.maximum(ub, torch.zeros_like(ub)).clamp_min(lb)  # degenerate -> [lb, lb]
        # A pixel whose gradient is exactly zero in every free coordinate (alpha,
        # gamma, and their sum) is already at a stationary point: every d == 0,
        # so every interval() call above falls into the "no constraint" branch
        # and ub stays +inf (amin of all +inf). Left as +inf, the line search
        # below would set lam = +inf and _apply_step would then compute
        # lam * df_alpha = inf * 0.0 = nan, permanently poisoning the pixel
        # (NaN never satisfies the delta < tol convergence test, so it's never
        # dropped from `active`). Since a zero gradient means no direction
        # improves the objective, collapsing to lam = 0 (lb == ub) is exact,
        # not an approximation - the step is genuinely a no-op either way.
        ub = torch.where(torch.isinf(ub), lb, ub)
        return lb, ub

    def _apply_step(
        self,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        df_alpha: torch.Tensor,
        df_gamma: torch.Tensor,
        lam: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Candidate (alpha, gamma) for per-pixel step sizes lam, restoring alpha_R = 1 - sum."""

        lam_col = lam.unsqueeze(1)
        alpha_free = alpha[:, :-1] - lam_col * df_alpha
        alpha_last = 1.0 - alpha_free.sum(dim=1, keepdim=True)
        alpha_new = torch.cat([alpha_free, alpha_last], dim=1)
        gamma_new = gamma - lam_col * df_gamma
        return alpha_new, gamma_new

    def _line_search(
        self,
        y: torch.Tensor,
        alpha: torch.Tensor,
        gamma: torch.Tensor,
        df_alpha: torch.Tensor,
        df_gamma: torch.Tensor,
        lb: torch.Tensor,
        ub: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized golden-section search for the per-pixel step size (golden_section.m).

        Minimizes J(lambda) = ||y - mu_GBM(theta - lambda * df)||^2 over
        [lb, ub] independently for every pixel, shrinking every interval by
        the golden ratio each iteration with one new function evaluation per
        iteration (the surviving interior point's value is re-used; which
        side survives is selected per pixel by mask). Returns the midpoint of
        the final interval, exactly as the reference code does.
        """

        def objective(lam: torch.Tensor) -> torch.Tensor:
            alpha_c, gamma_c = self._apply_step(alpha, gamma, df_alpha, df_gamma, lam)
            return ((y - self._forward(alpha_c, gamma_c)) ** 2).sum(dim=1)

        a = lb.clone()
        b = ub.clone()
        lam = a + (1.0 - _GOLDEN) * (b - a)
        mu = a + _GOLDEN * (b - a)
        f_lam = objective(lam)
        f_mu = objective(mu)

        for _ in range(self.cfg.ls_max_iters):
            if bool(((b - a) <= self.cfg.ls_tol).all()):
                break
            shrink_left = f_lam > f_mu  # minimum lies right of lam: discard [a, lam]
            a = torch.where(shrink_left, lam, a)
            b = torch.where(shrink_left, b, mu)
            lam_new = a + (1.0 - _GOLDEN) * (b - a)
            mu_new = a + _GOLDEN * (b - a)
            # Re-use the surviving interior evaluation; evaluate only the fresh
            # point. When the left side is discarded the old mu becomes the new
            # lam (value re-used) and the new mu is fresh; symmetrically when
            # the right side is discarded.
            fresh = torch.where(shrink_left, mu_new, lam_new)
            f_fresh = objective(fresh)
            f_lam, f_mu = (
                torch.where(shrink_left, f_mu, f_fresh),
                torch.where(shrink_left, f_fresh, f_lam),
            )
            lam, mu = lam_new, mu_new

        return a + (b - a) / 2.0

    def solve(self, endmembers: torch.Tensor, pixels: torch.Tensor) -> torch.Tensor:
        """Solve abundances for pixels under the GBM via projected gradient descent.

        Parameters
        ----------
        endmembers : torch.Tensor
            Endmember matrix with shape ``(n_endmembers, n_bands)``.
        pixels : torch.Tensor
            Pixel matrix with shape ``(n_pixels, n_bands)``.

        Returns
        -------
        torch.Tensor
            Estimated abundances with shape ``(n_pixels, n_endmembers)``,
            satisfying the non-negativity and sum-to-one constraints by
            construction. Per-pair nonlinearity coefficients are exposed as
            ``self.gamma_`` with shape ``(n_pixels, n_pairs)`` and the final
            per-pixel squared reconstruction error as ``self.error_``.
        """

        device, dtype = pixels.device, pixels.dtype
        n_pixels, _ = pixels.shape
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
        # Full (R, R, n_bands) cross-product table m_r (.) m_i for the gradient.
        self._cross_products = endmembers.unsqueeze(1) * endmembers.unsqueeze(0)
        n_pairs = self._pair_products.shape[0]

        # Warm start: alpha^(0) from the project's SunSAL solver, projected onto
        # the simplex so the first iterate is strictly feasible (the reference
        # code uses per-pixel FCLS for the same purpose - see module docstring,
        # deviation 2, and `_project_to_simplex` for why the projection is
        # needed). gamma^(0) = 0.01 for every pair, exactly as the reference code.
        sunsal = SunSAL(SunSALConfig(μ=self.cfg.sunsal_mu, max_iters=self.cfg.sunsal_max_iters, tol=self.cfg.sunsal_tol))
        alpha = _project_to_simplex(sunsal.solve(endmembers, pixels), self.cfg.eps)
        gamma = torch.full((n_pixels, n_pairs), self.cfg.gamma_init, device=device, dtype=dtype)

        prev_err = ((pixels - self._forward(alpha, gamma)) ** 2).sum(dim=1)
        active = torch.ones(n_pixels, dtype=torch.bool, device=device)

        for t in range(self.cfg.max_iters):
            idx = active.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                break

            y_a = pixels[idx]
            alpha_a = alpha[idx]
            gamma_a = gamma[idx]

            df_alpha, df_gamma, _ = self._gradients(y_a, alpha_a, gamma_a)
            df_alpha = self._clip_boundary_gradients(alpha_a, df_alpha)
            lb, ub = self._step_bounds(alpha_a, gamma_a, df_alpha, df_gamma)
            lam = self._line_search(y_a, alpha_a, gamma_a, df_alpha, df_gamma, lb, ub)

            alpha_new, gamma_new = self._apply_step(alpha_a, gamma_a, df_alpha, df_gamma, lam)
            # Numerical guard only: the step-size bounds keep the iterate
            # feasible up to floating-point noise.
            alpha_new = alpha_new.clamp(0.0, 1.0)
            gamma_new = gamma_new.clamp(0.0, 1.0)

            err = ((y_a - self._forward(alpha_new, gamma_new)) ** 2).sum(dim=1)
            delta = (err - prev_err[idx]).abs()

            alpha[idx] = alpha_new
            gamma[idx] = gamma_new
            prev_err[idx] = err

            # Per-pixel convergence: freeze pixels whose error stopped moving
            # (reference code's |DER| < 1e-5 test, applied independently).
            newly_converged = delta < self.cfg.tol
            active[idx[newly_converged]] = False

            if (t + 1) % self.cfg.check_every == 0:
                self.history["error_mean"].append(float(prev_err.mean().item()))
                self.history["error_max_delta"].append(float(delta.max().item()) if delta.numel() else 0.0)
                self.history["active_fraction"].append(float(active.float().mean().item()))
                if self.cfg.verbose:
                    print(
                        f"Iter {t + 1:4d} | err mean={prev_err.mean().item():.3e} | "
                        f"active={active.float().mean().item():.3f}"
                    )

        # Abundances already satisfy the constraints by construction; renormalize
        # only to absorb the clamp's floating-point crumbs.
        abundances = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(self.cfg.eps)

        self.gamma_ = gamma
        self.error_ = prev_err
        return abundances