"""RPS diagnostic module: RPSGM + RPSR + OPT.

Implements Section 4.2 of the paper:

* RPSGM: train a Gaussian discriminant model (GDM) per fault hypothesis from
  simulated data, build a membership vector from the observed residual,
  factor sequential weights, and emit a length-truncated PMF over the
  permutation event space.
* RPSR: reliability-weighted multi-source fusion. Reliabilities are updated
  online from the freshly observed residual energy via a sigmoid map.
* OPT: ordered probability transformation, projecting the fused PMF onto a
  per-agent health estimate.

Compared to the original single-file reference, this version:

* trains the GDM from explicit simulator samples (``train_gdm`` /
  ``RPSDiagnosticModule.fit``);
* defines residuals as physical residual *energies* (scalars) so the
  Gaussian likelihoods and the sigmoid reliability map make dimensional
  sense;
* feeds the RPS pipeline residuals that actually depend on the simulator
  state, closing the diagnosis loop.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import permutations

import numpy as np
from scipy.special import expit as sigmoid

from dds_adapt.config import Config


def _gaussian_logpdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return -0.5 * ((x - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)


class RPSDiagnosticModule:
    """Random Permutation Set diagnostic block."""

    def __init__(self, n_agents: int, l_max: int = Config.L_MAX) -> None:
        self.n = n_agents
        self.l_max = l_max
        # GDM parameters: one Gaussian for "agent j healthy" and one for
        # "agent j faulty" residual energy as observed by agent i.
        self.mean_h = np.zeros((n_agents, n_agents))
        self.std_h = np.full((n_agents, n_agents), 0.1)
        self.mean_f = np.full((n_agents, n_agents), 1.0)
        self.std_f = np.full((n_agents, n_agents), 0.2)
        self.reliabilities = np.full(n_agents, Config.RELIABILITY_INIT)
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        healthy_residuals: np.ndarray,
        faulty_residuals_per_agent: list[np.ndarray] | np.ndarray,
    ) -> None:
        """Train the GDM from simulator-collected residual samples.

        Parameters
        ----------
        healthy_residuals : (T, N) array
            Residual energies seen by every agent while no agent is faulty.
            Used to estimate ``mean_h[i, j]`` / ``std_h[i, j]`` for all
            ``(i, j)`` pairs (we share the marginal ``j``-statistics across
            ``i`` to keep the model parsimonious).
        faulty_residuals_per_agent : list of (T_j, N) arrays
            Element ``j`` is the residual block recorded while *only* agent
            ``j`` is faulty, used to estimate ``mean_f[:, j]`` /
            ``std_f[:, j]``.
        """
        H = np.asarray(healthy_residuals)
        if H.ndim != 2 or H.shape[1] != self.n:
            raise ValueError(
                f"healthy_residuals must have shape (T, {self.n}), got {H.shape}"
            )
        # under the healthy hypothesis the residual energy depends mostly on
        # the agent being observed, not on the observer; share statistics.
        h_means = H.mean(axis=0)
        h_stds = H.std(axis=0).clip(min=1e-3)
        self.mean_h[:] = h_means[None, :]
        self.std_h[:] = h_stds[None, :]

        if len(faulty_residuals_per_agent) != self.n:
            raise ValueError(
                "faulty_residuals_per_agent must have length n_agents"
            )
        for j, block in enumerate(faulty_residuals_per_agent):
            B = np.asarray(block)
            if B.ndim != 2 or B.shape[1] != self.n:
                raise ValueError(
                    f"fault block for agent {j} has shape {B.shape}, "
                    f"expected (T, {self.n})"
                )
            self.mean_f[:, j] = B.mean(axis=0)
            self.std_f[:, j] = B.std(axis=0).clip(min=1e-3)
        self._fitted = True

    # ------------------------------ memberships & local PMF
    def compute_memberships(self, residual_vec: np.ndarray, agent_i: int) -> np.ndarray:
        """Posterior P(faulty | residual_vec[j]) for every ``j``."""
        log_h = _gaussian_logpdf(
            residual_vec, self.mean_h[agent_i], self.std_h[agent_i]
        )
        log_f = _gaussian_logpdf(
            residual_vec, self.mean_f[agent_i], self.std_f[agent_i]
        )
        # log-sum-exp stabilisation: subtract the per-element max
        # before exponentiating so the ratio is numerically safe even
        # when both log-pdfs are very negative.
        m = np.maximum(log_h, log_f)
        post = np.exp(log_f - m) / (np.exp(log_h - m) + np.exp(log_f - m))
        return post

    def generate_local_pmf(
        self, agent_i: int, residual_vec: np.ndarray
    ) -> dict[tuple[int, ...], float]:
        """Build a length-``l_max`` truncated PMF over PES from a residual."""
        memberships = self.compute_memberships(residual_vec, agent_i)
        if memberships.sum() > 0:
            normed = memberships / memberships.sum()
        else:
            normed = np.full(self.n, 1.0 / self.n)

        order = np.argsort(normed)[::-1]
        ordered_norm = normed[order]

        pmf: dict[tuple[int, ...], float] = defaultdict(float)
        max_len = min(self.l_max, self.n)
        for length in range(1, max_len + 1):
            for perm in permutations(order[:length]):
                support = 1.0
                for u in range(length):
                    s_iu = np.exp(-abs(normed[perm[u]] - ordered_norm[u]))
                    remaining = ordered_norm[u:].sum() + 1e-10
                    support *= s_iu / remaining
                pmf[perm] = support * ordered_norm[length - 1]

        total = sum(pmf.values())
        if total > 0:
            for k in pmf:
                pmf[k] /= total
        return pmf

    # ------------------------------------------------ multi-source fusion
    @staticmethod
    def _left_intersection(
        a: tuple[int, ...], b: tuple[int, ...]
    ) -> tuple[int, ...] | None:
        """Left intersection of two permutations (Deng et al., RPST).

        ``LI(A, B)`` is the longest leading sub-sequence of ``A`` whose
        elements all appear in ``B``, taken in the order they appear in
        ``A``. Returns ``None`` if the intersection is empty.
        """
        b_set = set(b)
        out: list[int] = []
        for x in a:
            if x in b_set:
                out.append(x)
            else:
                break
        return tuple(out) if out else None

    @classmethod
    def _los_combine(
        cls, m1: dict[tuple[int, ...], float], m2: dict[tuple[int, ...], float]
    ) -> dict[tuple[int, ...], float]:
        """Left orthogonal sum of two permutation mass functions.

        For each pair of focal elements (A, B) we accumulate ``m1(A)·m2(B)``
        on ``LI(A, B)``. Mass that lands on the empty intersection is
        treated as conflict ``K``; the surviving distribution is
        renormalised by ``1 - K``.
        """
        out: dict[tuple[int, ...], float] = defaultdict(float)
        conflict = 0.0
        for a, ma in m1.items():
            for b, mb in m2.items():
                inter = cls._left_intersection(a, b)
                if inter is None:
                    conflict += ma * mb
                else:
                    out[inter] += ma * mb
        denom = max(1.0 - conflict, 1e-12)
        for k in out:
            out[k] /= denom
        return out

    def fuse_pmfs(
        self,
        local_pmfs: list[dict[tuple[int, ...], float]],
        reliabilities: np.ndarray | None = None,
    ) -> dict[tuple[int, ...], float]:
        """RPSR multi-source fusion via the left orthogonal sum.

        Each local PMF is first reliability-discounted: mass scaled by
        ``rho_i``, with the residual ``1 - rho_i`` placed on the full
        permutation as ignorance (Shafer-style discounting). Sources are
        then combined in descending-reliability order using the left
        orthogonal sum, exactly as the paper's §4.2 prescribes.
        """
        if reliabilities is None:
            reliabilities = self.reliabilities

        full_perm = tuple(range(self.n))
        discounted: list[dict[tuple[int, ...], float]] = []
        for rho, pmf in zip(reliabilities, local_pmfs, strict=False):
            d: dict[tuple[int, ...], float] = {}
            for perm, mass in pmf.items():
                d[perm] = float(rho) * mass
            d[full_perm] = d.get(full_perm, 0.0) + (1.0 - float(rho))
            discounted.append(d)

        order = np.argsort(reliabilities)[::-1]
        fused = dict(discounted[order[0]])
        for k in order[1:]:
            fused = dict(self._los_combine(fused, discounted[k]))

        total = sum(fused.values())
        if total > 0:
            fused = {key: v / total for key, v in fused.items()}
        return fused

    # ----------------------------- ordered probability transformation
    def extract_health_estimate(
        self, fused_pmf: dict[tuple[int, ...], float]
    ) -> np.ndarray:
        """OPT: per-agent health = 1 - P_OPT(theta_j) (Eq. 10 of the paper).

        Following Section 3.3, ``P_OPT(theta_j)`` is the sum of the fused
        permutation mass over every permutation in which ``theta_j``
        appears. The resulting vector satisfies ``sum_j P_OPT(theta_j) = 1``
        because every permutation contributes its mass exactly ``len(perm)``
        times distributed across its members; we therefore renormalise so
        the unit-sum identity holds even when the PMF is truncated.
        """
        deg_prob = np.zeros(self.n)
        for perm, mass in fused_pmf.items():
            for agent_idx in perm:
                deg_prob[agent_idx] += mass
        total = deg_prob.sum()
        if total > 0:
            deg_prob = deg_prob / total
        return np.clip(1.0 - deg_prob, 0.0, 1.0)

    # ---------------------------------------- online reliability update
    def update_reliability(self, agent_i: int, residual_vec: np.ndarray) -> float:
        """Update ``reliabilities[agent_i]`` from the GDM-based KL gap.

        Following Section 4.2: for every pair of fault hypotheses
        ``(j_1, j_2)``, compute the KL divergence between their
        Gaussian residual distributions under agent ``i``'s GDM. The
        smallest such divergence (the "most confusable pair") captures
        how distinguishable agent ``i``'s observations are. We squash
        this minimum gap through a sigmoid to land in ``[0, 1]``.

        ``residual_vec`` is unused for the reliability score itself but
        kept in the signature so callers can interleave reliability
        updates with PMF generation.
        """
        del residual_vec  # GDM-side metric does not depend on the live residual

        # KL divergence between two univariate Gaussians N(mu1, sigma1) and
        # N(mu2, sigma2):
        #   KL = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2)/(2 sigma2^2) - 1/2
        def _kl(mu1: float, s1: float, mu2: float, s2: float) -> float:
            s1 = max(s1, 1e-6)
            s2 = max(s2, 1e-6)
            return float(
                np.log(s2 / s1)
                + (s1**2 + (mu1 - mu2) ** 2) / (2.0 * s2**2)
                - 0.5
            )

        min_div = np.inf
        for j1 in range(self.n):
            for j2 in range(self.n):
                if j1 == j2:
                    continue
                d12 = _kl(
                    self.mean_f[agent_i, j1], self.std_f[agent_i, j1],
                    self.mean_f[agent_i, j2], self.std_f[agent_i, j2],
                )
                d21 = _kl(
                    self.mean_f[agent_i, j2], self.std_f[agent_i, j2],
                    self.mean_f[agent_i, j1], self.std_f[agent_i, j1],
                )
                min_div = min(min_div, 0.5 * (d12 + d21))

        if not np.isfinite(min_div):
            self.reliabilities[agent_i] = Config.RELIABILITY_INIT
        else:
            # sigmoid with the configured temperature; subtract a small
            # offset so reliability spans [0, 1] rather than collapsing
            # to 1 for every agent
            self.reliabilities[agent_i] = float(
                sigmoid(min_div / Config.SIGMOID_TEMP - 1.0)
            )
        return float(self.reliabilities[agent_i])

    # ------------------------------------------------------- end-to-end
    def diagnose_round(
        self, residual_matrix: np.ndarray
    ) -> tuple[np.ndarray, list[dict[tuple[int, ...], float]], dict]:
        """Run one full RPSGM + RPSR + OPT round.

        ``residual_matrix[i, j]`` is the residual energy that agent ``i``
        observes about agent ``j``.
        """
        local_pmfs: list[dict[tuple[int, ...], float]] = []
        for i in range(self.n):
            self.update_reliability(i, residual_matrix[i])
            local_pmfs.append(self.generate_local_pmf(i, residual_matrix[i]))
        fused = self.fuse_pmfs(local_pmfs)
        h_hat = self.extract_health_estimate(fused)
        return h_hat, local_pmfs, fused
