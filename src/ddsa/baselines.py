"""Baseline methods used for the comparative study.

Each baseline mirrors a specific question raised in Section 5.1.2 of the
paper:

* Robust DO     - "why not just shrink confidence margins?"
* FDI-Reconf    - "why not detect-and-isolate above a threshold?"
* Byzantine     - "do existing adversarial-aggregation rules cope?"
* D-S Fusion    - "is RPS strictly better than Dempster-Shafer combination?"

The Oracle baseline (perfect health knowledge) is intentionally not a
class here: it is realised by ``ExperimentRunner.run_oracle`` calling
the engine with the ground-truth health profile as override, with no
state of its own.

The classes here are deliberately thin: they hold the *configuration*
of each baseline (margin, threshold, trim ratio) and the operators it
needs (e.g. trimmed mean, Dempster combination). Closed-loop wiring
lives in ``runner.py`` so that every method shares the same engine
plumbing and therefore the same logging surface.
"""

from __future__ import annotations

import numpy as np

from ddsa.config import Config
from ddsa.optimizer import DIGingOptimizer


# ---------------------------------------------------------------------- DO
class BaselineRobustDO:
    """Robust distributed optimisation: conservative cost-function weighting.

    The paper describes the baseline as treating degradation as bounded
    disturbance through a uniform conservative weight (``1 - margin``) on
    every agent's cost contribution, rather than estimating health.

    In this codebase ``ExperimentRunner.run_robust_do`` does not actually
    instantiate this class: it builds the conservative-health vector
    inline and feeds it as an override to the shared engine, so every
    method shares the same logging surface. ``BaselineRobustDO`` is
    kept as a configuration-only entry point for callers who want to
    drive a Robust-DO problem outside the comparative-table loop;
    ``self.optimizer`` is the ``DIGingOptimizer`` they would feed
    gradients to.
    """

    def __init__(
        self,
        n_agents: int,
        dim: int,
        W: np.ndarray,
        alpha: float = Config.ALPHA_NUM,
        robustness_margin: float = 0.3,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n = n_agents
        self.dim = dim
        self.W = W
        self.alpha = alpha
        self.margin = robustness_margin
        self.optimizer = DIGingOptimizer(n_agents, dim, W, alpha, rng=rng)


# ----------------------------------------------------------------- FDI
class BaselineFDIReconf:
    """Threshold-based fault detection-isolation-reconfiguration.

    Detection is triggered by the *residual* energy crossing a fixed
    threshold (the same residual the diagnostic module uses). Once an
    agent is flagged, the runner pins its iterate to the formation
    reference and zeroes its row of the gradient (see ``engine.py``'s
    ``isolation_mask`` argument), realising the paper's "recompute the
    formation using only healthy agents" semantics.
    """

    def __init__(
        self,
        n_agents: int,
        dim: int,
        W: np.ndarray,
        alpha: float = Config.ALPHA_NUM,
        residual_threshold: float = 0.3,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n = n_agents
        self.dim = dim
        self.W = W
        self.alpha = alpha
        self.residual_threshold = residual_threshold
        self.optimizer = DIGingOptimizer(n_agents, dim, W, alpha, rng=rng)
        self.isolated = np.zeros(n_agents, dtype=bool)

    def detect(self, residual_energy: np.ndarray) -> np.ndarray:
        """Mark agents whose residual energy exceeds the threshold.

        Isolation is *monotone*: once flagged, an agent stays flagged
        even if its residual energy later returns below threshold.
        This matches the paper's "FDI commits and reconfigures" semantics
        for the §5.4.1 single-fault scenario, but means a single
        ``BaselineFDIReconf`` instance carries state across runs.
        Build a fresh instance per trajectory if you want detection
        to start from a clean slate.
        """
        self.isolated |= residual_energy > self.residual_threshold
        return self.isolated.copy()

    def reset(self) -> None:
        """Clear the isolation latch (use between independent trajectories)."""
        self.isolated[:] = False


# -------------------------------------------------------------- Byzantine
class BaselineByzantineResilient:
    """Trimmed-mean aggregation with outlier quarantine.

    The class holds the trim ratio plus the residual-energy detector
    the runner uses for quarantine decisions (``detect_outliers``).
    ``self.optimizer`` is kept as the DIGing surface a caller would
    drive if it wants the trimmed-mean consensus directly. The runner
    quarantines flagged agents into a frozen safe hold while the
    healthy agents keep the nominal (no health estimation) objective.
    """

    def __init__(
        self,
        n_agents: int,
        dim: int,
        W: np.ndarray,
        alpha: float = Config.ALPHA_NUM,
        trim_ratio: float = 0.2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n = n_agents
        self.dim = dim
        self.W = W
        self.alpha = alpha
        self.trim_ratio = trim_ratio
        self.optimizer = DIGingOptimizer(n_agents, dim, W, alpha, rng=rng)

    def detect_outliers(
        self, residual_energy: np.ndarray, k_mad: float = Config.BYZ_OUTLIER_K
    ) -> np.ndarray:
        """Flag agents whose residual energy is a trimmed-core outlier.

        The detector trims ``trim_ratio`` of the sorted energies on both
        sides, computes the median and median-absolute-deviation of the
        surviving core, and flags any agent above ``median + k_mad *
        MAD``. With a single faulty agent the median tracks the healthy
        noise floor, so the faulty agent crosses the envelope as its
        degradation deepens while the healthy agents stay inside it.
        """
        energy = np.asarray(residual_energy, dtype=float)
        n = len(energy)
        n_trim = max(1, int(n * self.trim_ratio))
        sorted_e = np.sort(energy)
        core = (
            sorted_e[n_trim:-n_trim] if n_trim * 2 < n else sorted_e
        )
        median = float(np.median(core))
        mad = float(np.median(np.abs(core - median))) + 1e-9
        return energy > median + k_mad * mad


# -------------------------------------------------------- Dempster-Shafer
class BaselineDSFusion:
    """Conventional Dempster-Shafer evidence combination.

    Each agent ``i`` reports a binary BPA on the singletons
    :math:`\\{healthy, faulty\\}` for every other agent ``j``. The fused
    health estimate is built from the resulting belief in the
    ``healthy`` singleton. Unlike RPS, no priority ordering between
    different ``j`` is preserved.
    """

    def __init__(self, n_agents: int) -> None:
        self.n = n_agents

    @staticmethod
    def _combine_two(m1: dict[str, float], m2: dict[str, float]) -> dict[str, float]:
        """Pairwise Dempster combination on the frame ``{H, F}``."""
        K = m1["H"] * m2["F"] + m1["F"] * m2["H"]  # conflict
        denom = max(1.0 - K, 1e-12)
        out = {
            "H": (m1["H"] * m2["H"] + m1["H"] * m2["U"] + m1["U"] * m2["H"]) / denom,
            "F": (m1["F"] * m2["F"] + m1["F"] * m2["U"] + m1["U"] * m2["F"]) / denom,
            "U": (m1["U"] * m2["U"]) / denom,
        }
        # numerical clean-up
        s = sum(out.values())
        if s > 0:
            for k in out:
                out[k] /= s
        return out

    def fuse(
        self,
        local_health_estimates: np.ndarray,
        ignorance: float = 0.1,
    ) -> np.ndarray:
        """Combine local soft health estimates via Dempster's rule per agent.

        Parameters
        ----------
        local_health_estimates : (N, N) array
            ``local_health_estimates[i, j]`` is agent ``i``'s soft belief
            that agent ``j`` is healthy (a number in ``[0, 1]``).
        ignorance : float
            Mass assigned to the ignorance set ``U`` in each local BPA.
            The default 0.1 gives a decisive, near-binary fused belief
            (used where the baseline only needs a detection decision);
            a larger value keeps the fused belief close to the local
            evidence, which is what the closed-loop inline variant
            needs so the adaptation layer sees a continuous severity
            rather than a saturated label.
        """
        E = np.clip(np.asarray(local_health_estimates), 0.0, 1.0)
        u = float(np.clip(ignorance, 0.0, 1.0))
        out = np.zeros(self.n)
        for j in range(self.n):
            # convert each soft estimate into a BPA with the requested
            # mass on ignorance so Dempster combination stays well-behaved
            mass_list = []
            for i in range(self.n):
                p = float(E[i, j])
                h = (1 - u) * p
                f = (1 - u) * (1 - p)
                mass_list.append({"H": h, "F": f, "U": u})
            fused = mass_list[0]
            for k in range(1, self.n):
                fused = self._combine_two(fused, mass_list[k])
            # decision: belief in healthy + half of ignorance (pignistic)
            out[j] = fused["H"] + 0.5 * fused["U"]
        return np.clip(out, 0.0, 1.0)


# Note: there is intentionally no ``BaselineOracle`` class. Oracle in
# this codebase means "run the closed-loop engine with the true health
# vector as override", which is just a one-line call inside
# ``ExperimentRunner.run_oracle``; a class would only add ceremony.
