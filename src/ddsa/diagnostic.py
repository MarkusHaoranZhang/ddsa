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

Mapping to the paper's symbols
------------------------------
The paper expresses the per-configuration support as
``s_A = -log D(R_i, E[r|A])`` where ``D`` is the energy distance between
a residual window ``R_i`` and the expected residual under hypothesis
``A``. This module realises the same support function via a Gaussian
discriminant model: per-agent membership posteriors built from
healthy / faulty likelihoods (``compute_memberships``) play the role of
the residual-vs-expected agreement, and the per-position weighting
inside ``generate_local_pmf`` realises the *sequential conditional
weighting* the paper builds ``s_A`` from. The two formulations are
interchangeable as scoring functions over the truncated permutation
event space; the GDM form is what we ship because it factors the
training cost into a one-shot ``fit`` call (see ``train_gdm`` in
``residual.py``) instead of requiring an online ``E[r|A]`` predictor
at run time.

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

from ddsa.config import Config


def _gaussian_logpdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return -0.5 * ((x - mean) / std) ** 2 - np.log(std) - 0.5 * np.log(2 * np.pi)


class RPSDiagnosticModule:
    """Random Permutation Set diagnostic block."""

    def __init__(self, n_agents: int, l_max: int = Config.L_MAX) -> None:
        """Initialise an empty GDM (call ``fit`` before diagnosing)."""
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
            # Under the broadcast residual structure
            # ``R[i, j] = own_energy[j] + cross-agent noise``, every
            # observer sees the same expected residual at column j; so
            # ``mean_f[i, j]`` depends only on j, not on i. We therefore
            # broadcast the column-j *diagonal* of ``B_j.mean(axis=0)``
            # across all rows. Storing ``B.mean(axis=0)`` directly into
            # the column would mix into mean_f[i!=j, j] the *off-target*
            # statistics, which collapse back toward the healthy mean
            # and silently invert the diagnostic order at small N.
            faulty_mean_at_target = float(B.mean(axis=0)[j])
            faulty_std_at_target = max(float(B.std(axis=0)[j]), 1e-3)
            self.mean_f[:, j] = faulty_mean_at_target
            self.std_f[:, j] = faulty_std_at_target
        self._fitted = True

        # Reliability is a derived quantity of the fitted GDM (the
        # smallest pairwise KL gap between the per-fault Gaussians of
        # this observer); it does *not* depend on the live residual.
        # Compute it once here so every downstream ``diagnose_round``
        # call sees the same value, independent of how many times it
        # has been invoked.
        for i in range(self.n):
            self.update_reliability(i)

    # ------------------------------ memberships & local PMF
    def compute_memberships(self, residual_vec: np.ndarray, agent_i: int) -> np.ndarray:
        """Posterior P(faulty | residual_vec[j]) for every ``j``.

        Plays the role of the per-agent agreement score that the
        paper's ``s_A = -log D(R_i, E[r|A])`` is built from: a high
        membership for ``j`` means the residual that agent ``i``
        observes about ``j`` agrees with the GDM's *faulty*
        Gaussian, which is the GDM-side analogue of "the residual
        window matches the expected pattern under a hypothesis that
        includes ``j``".
        """
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
        """Build a length-``l_max`` truncated PMF over PES from a residual.

        This realises the sequential conditional weighting the paper
        uses to construct ``s_A``: each focal element ``A`` of length
        ``q`` accumulates a per-position support
        ``s_iu = exp(-|membership[A[u]] - ordered_membership[u]|)``,
        which scores the agreement between the membership of the
        agent placed at position ``u`` and the ``u``-th largest
        membership overall. The product across positions plays the
        same role as the cumulative log-support in the paper's
        formulation; the final ``support * ordered_norm[length - 1]``
        weighting, together with the normalisation step below, is the
        truncated softmax over PES that produces ``M_i^{(t)}(A)``.
        """
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

        Position-weighted form. Each permutation ``A`` distributes its
        mass to the agents it contains *in proportion to their rank*:
        an agent in position ``r`` of ``A`` (1-indexed, smallest = most
        suspect) receives ``m(A) · 1/r / H_{|A|}``, where ``H_k`` is
        the k-th harmonic number. This recovers the priority ordering
        the RPS pipeline carries through fusion: a long permutation
        ``(0, 1, 2)`` puts most of its mass on agent 0 (the leading
        suspect), not equally on all three.

        The plain "any agent in the permutation gets the full mass"
        rule, while structurally simpler, collapses to a uniform
        distribution as soon as length-N permutations dominate the
        fused PMF -- which is the regime our truncated PES sits in.
        """
        deg_prob = np.zeros(self.n)
        for perm, mass in fused_pmf.items():
            length = len(perm)
            if length == 0:
                continue
            # harmonic number normaliser: 1/1 + 1/2 + ... + 1/length
            harmonic = sum(1.0 / r for r in range(1, length + 1))
            for rank_minus_one, agent_idx in enumerate(perm):
                weight = (1.0 / (rank_minus_one + 1)) / harmonic
                deg_prob[agent_idx] += mass * weight
        total = deg_prob.sum()
        if total > 0:
            deg_prob = deg_prob / total
        return np.clip(1.0 - deg_prob, 0.0, 1.0)

    # ---------------------------------------- online reliability update
    def update_reliability(self, agent_i: int) -> float:
        """Update ``reliabilities[agent_i]`` from the GDM-based KL gap.

        Following Section 4.2: for every pair of fault hypotheses
        ``(j_1, j_2)``, compute the KL divergence between their
        Gaussian residual distributions under agent ``i``'s GDM. The
        smallest such divergence (the "most confusable pair") captures
        how distinguishable agent ``i``'s observations are. We squash
        this minimum gap through a sigmoid to land in ``[0, 1]``.

        The reliability is a *static* property of the fitted GDM: it
        depends on how well-separated the per-fault Gaussians are
        for this observer, not on the live residual. Recomputing it
        per tick (as ``diagnose_round`` does) is cheap and keeps the
        path open for a future GDM that updates online.
        """
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
    ) -> tuple[
        np.ndarray,
        list[dict[tuple[int, ...], float]],
        dict[tuple[int, ...], float],
    ]:
        """Run one full RPSGM + RPSR + OPT round.

        ``residual_matrix[i, j]`` is the residual energy that agent ``i``
        observes about agent ``j``.

        Reliabilities are static after :meth:`fit`; we do not refresh
        them here, so this loop is deterministic given the residual
        matrix alone.
        """
        local_pmfs: list[dict[tuple[int, ...], float]] = []
        for i in range(self.n):
            local_pmfs.append(self.generate_local_pmf(i, residual_matrix[i]))
        fused = self.fuse_pmfs(local_pmfs)
        h_hat = self.extract_health_estimate(fused)
        return h_hat, local_pmfs, fused
