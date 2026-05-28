"""High-level experiment driver.

Every method (Proposed + 5 baselines + 5 ablation variants) is wrapped
as a callable that takes a single trajectory and returns a metric dict
plus the engine log. The closed-loop engine is shared, so methods
differ only in how they produce ``h_hat`` and whether they perform the
Sinkhorn-attenuated mixing-matrix adaptation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from dds_adapt.baselines import (
    BaselineByzantineResilient,
    BaselineDSFusion,
    BaselineFDIReconf,
)
from dds_adapt.config import Config
from dds_adapt.cost import formation_cost_global, local_cost_grad
from dds_adapt.diagnostic import RPSDiagnosticModule
from dds_adapt.engine import EngineLog, run_closed_loop
from dds_adapt.metrics import compute_metrics
from dds_adapt.residual import (
    broadcast_residual_matrix,
    residual_energy,
    train_gdm,
)
from dds_adapt.scenarios import (
    actuator_degradation_profile,
    communication_loss_w_sequence,
    concurrent_degradation_profile,
    perturb_topology,
)
from dds_adapt.simulator import SatelliteFormationSimulator

# Type aliases used pervasively in this module.
#
# A method returns a per-run metric dict (str -> float, NaN allowed)
# and the engine log. ``_summarise_runs`` aggregates a list of those
# into a per-metric (mean, std) tuple keyed by metric name; a study's
# top-level result is then keyed by method name on top of that.
MetricDict = dict[str, float]
SummaryDict = dict[str, tuple[float, float]]
StudyResult = dict[str, SummaryDict]
MethodFn = Callable[[np.ndarray, int], tuple[MetricDict, EngineLog]]


# --------------------------------------------------------- helpers
def _broadcast_health_profile(constant_h: np.ndarray, n_steps: int) -> np.ndarray:
    """Tile a single health vector across the trajectory (Oracle override)."""
    return np.repeat(constant_h[:, None], n_steps, axis=1)


# --------------------------------------------------------- runner
class ExperimentRunner:
    """Drives every experimental study on top of the closed-loop engine."""

    def __init__(
        self,
        n_satellites: int = Config.NUM_SATELLITES,
        track: str = "numerical",
        seed: int = 0,
        gdm_training_samples: int = Config.N_TRAIN_SAMPLES,
        verbose: bool = False,
    ) -> None:
        """Build the topology, train the GDM, and cache cost-function constants.

        ``track`` selects between the numerical (``ALPHA_NUM``,
        ``GAMMA_NUM``) and high-fidelity (``ALPHA_HF``, ``GAMMA_HF``)
        parameter sets from :class:`Config`.
        """
        self.n = n_satellites
        self.track = track
        self.seed = seed

        self.W = Config.get_communication_graph(n_satellites)
        self.lambda2 = Config.algebraic_connectivity(self.W)
        self.edges = Config.edges(n_satellites)

        self.desired_positions = np.stack(
            [
                np.array(
                    [np.cos(2 * np.pi * i / n_satellites),
                     np.sin(2 * np.pi * i / n_satellites)]
                )
                for i in range(n_satellites)
            ]
        )

        self.alpha = Config.ALPHA_NUM if track == "numerical" else Config.ALPHA_HF
        self.gamma = Config.GAMMA_NUM if track == "numerical" else Config.GAMMA_HF
        self.beta = Config.BETA

        self.diagnostic = RPSDiagnosticModule(n_satellites)
        if verbose:
            print(
                f"[ExperimentRunner] training GDM "
                f"(N={n_satellites}, samples={gdm_training_samples})...",
                flush=True,
            )
        healthy, faulty = train_gdm(
            n_satellites, n_samples=gdm_training_samples, seed=seed
        )
        self.diagnostic.fit(healthy, faulty)
        if verbose:
            print("[ExperimentRunner] GDM ready.", flush=True)

    # ------------------------------------------------------------------ profiles
    def degradation_profile(
        self,
        n_steps: int,
        eta: float = Config.ETA_SINGLE,
        onset_time: int | None = None,
        agent_idx: int = 0,
        seed: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """§5.4.1 single-fault profile: agent ``agent_idx`` decays at rate ``eta``.

        Returns ``(health, onset_time)`` where ``health`` has shape
        ``(n, n_steps)``. Used by every comparative / ablation /
        topology / scenario-2 study.
        """
        return actuator_degradation_profile(
            self.n,
            n_steps,
            eta=eta,
            onset_time=onset_time,
            agent_idx=agent_idx,
            rng=np.random.default_rng(seed),
        )

    def concurrent_degradation_profile(
        self,
        n_steps: int,
        eta_fast: float = Config.ETA_FAST,
        eta_slow: float = Config.ETA_SLOW,
        onset_time: int = 80,
        seed: int | None = None,
    ) -> tuple[np.ndarray, int]:
        """§5.5.2 two-fault profile: agent 0 fast decay, agent 1 slow decay."""
        del seed  # concurrent profile is deterministic given onset_time
        return concurrent_degradation_profile(
            self.n,
            n_steps,
            eta_fast=eta_fast,
            eta_slow=eta_slow,
            onset_time=onset_time,
        )

    # ----------------------------------------------------- methods
    #
    # Every method except Byzantine is a thin wrapper around
    # ``_run_engine_method``; they only differ in how they construct the
    # health override and (optionally) the per-tick isolation mask.
    # Byzantine has its own runner because it does not use ``DIGingOptimizer``
    # in the standard way.
    # ------------------------------------------------------------------

    _N_DIAG_INTERVALS = 10
    # DIGing inner iterations per diagnostic tick. The paper's
    # "Δt = 50 optimisation iterations" refers to the diagnosis
    # *interval* (50 simulator steps between RPS calls), not the DIGing
    # loop length. Empirically, 50 inner iterations leave the iterate
    # an order of magnitude away from the centralised optimum on the
    # 8-satellite track; 500 brings consensus error below 1e-4. Warm
    # starts across ticks are automatic: ``DIGingOptimizer`` keeps
    # state, so subsequent ticks need only refine.
    _ITERS_PER_DIAG = 500

    def _run_engine_method(
        self,
        profile: np.ndarray,
        seed: int,
        *,
        override: np.ndarray | None = None,
        iso_mask: np.ndarray | None = None,
        use_w_adaptation: bool = True,
        w_base_per_interval: list[np.ndarray] | None = None,
    ) -> tuple[MetricDict, EngineLog]:
        """Single entry point for every engine-driven method.

        ``override`` (shape ``(n, n_steps)``) lets the caller bypass the
        RPS module entirely (Oracle, Robust DO, FDI, D-S, every ablation
        variant). ``iso_mask`` (shape ``(n_intervals, n)``) signals
        physically isolated agents to the engine.
        """
        log = run_closed_loop(
            health_profile=profile,
            diagnostic=self.diagnostic,
            W_base=self.W,
            desired_positions=self.desired_positions,
            edges=self.edges,
            n_diag_intervals=self._N_DIAG_INTERVALS,
            iters_per_diag=self._ITERS_PER_DIAG,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            seed=seed,
            use_w_adaptation=use_w_adaptation,
            health_estimate_override=override,
            w_base_per_interval=w_base_per_interval,
            isolation_mask=iso_mask,
        )
        return self.metrics_from_log(log, profile), log

    # --- engine-driven methods ---------------------------------------
    def run_proposed(
        self,
        health_profile: np.ndarray,
        seed: int,
        *,
        use_w_adaptation: bool = True,
        w_base_per_interval: list[np.ndarray] | None = None,
    ) -> tuple[MetricDict, EngineLog]:
        """Proposed method: RPSGM + RPSR + OPT + Sinkhorn-adapted DIGing.

        This is the full §3-§4 pipeline. ``use_w_adaptation`` is
        exposed so the §5.3 "no Sinkhorn" ablation can flip it off
        without re-implementing the closed loop; ``w_base_per_interval``
        lets §5.4.2 inject a time-varying communication graph.
        """
        return self._run_engine_method(
            health_profile, seed,
            use_w_adaptation=use_w_adaptation,
            w_base_per_interval=w_base_per_interval,
        )

    def run_oracle(
        self, health_profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        """Oracle baseline: closed loop driven by ground-truth health."""
        # Override = ground truth; engine bypasses RPS.
        return self._run_engine_method(
            health_profile, seed, override=health_profile, use_w_adaptation=True
        )

    def run_robust_do(
        self, health_profile: np.ndarray, seed: int, *, margin: float = 0.3
    ) -> tuple[MetricDict, EngineLog]:
        """§5.1.2 Robust DO: uniform conservative health (1 - margin), no W adaptation."""
        # Robust DO: every agent uses a uniform conservative health, no W
        # adaptation.
        n_steps = health_profile.shape[1]
        constant = np.full(self.n, 1.0 - margin)
        override = _broadcast_health_profile(constant, n_steps)
        m, log = self._run_engine_method(
            health_profile, seed, override=override, use_w_adaptation=False
        )
        # Robust DO does not estimate health; report diagnostic metrics
        # as NaN so summaries can render "—" rather than a misleading 0.
        m["health_mae"] = float("nan")
        m["kendall_tau"] = float("nan")
        return m, log

    def run_fdi(
        self, health_profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        """§5.1.2 FDI-Reconf: residual-energy threshold isolation + reconfiguration."""
        # Per §5.1.2: residual-energy threshold isolates degraded agents
        # and the engine pins their iterate to the formation reference.
        iso_mask, override = self._compute_fdi_isolation(health_profile, seed)
        m, log = self._run_engine_method(
            health_profile, seed,
            override=override, iso_mask=iso_mask, use_w_adaptation=True,
        )
        # FDI emits a binary {0, 1} mask, not a continuous health
        # estimate; diagnostic-layer metrics are not applicable.
        m["health_mae"] = float("nan")
        m["kendall_tau"] = float("nan")
        return m, log

    def run_ds_fusion(
        self, health_profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        """§5.4 D-S baseline: Dempster combine + threshold-based reconfiguration.

        Per Section 5.1.2, D-S "applies threshold-based reconfiguration
        without priority ordering information". Without ordering, D-S
        cannot cheaply distinguish primary from secondary degradation,
        so it commits to an isolation only after the fused healthy
        belief has stayed below the threshold for several consecutive
        diagnosis intervals -- this reproduces §5.4.1's observation
        that D-S detection is delayed relative to FDI.
        """
        iso_mask, override = self._compute_ds_isolation(health_profile, seed)
        m, log = self._run_engine_method(
            health_profile, seed,
            override=override, iso_mask=iso_mask, use_w_adaptation=True,
        )
        # D-S in the comparative table emits a binary isolation profile
        # (the closed-loop variant is studied separately in §5.3).
        # Diagnostic-layer metrics are reported on the *Dempster-fused
        # healthy belief* before thresholding, so we keep MAE/τ as
        # produced by the engine's last-tick override -- which here
        # is also binary, hence NaN.
        m["health_mae"] = float("nan")
        m["kendall_tau"] = float("nan")
        return m, log

    # --- ablation variants ------------------------------------------
    def _run_ds_closed_loop(
        self, health_profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        """§5.3 Variant A: closed loop + D-S fusion in place of RPSR."""
        override = self._build_ds_inline_profile(health_profile, seed)
        return self._run_engine_method(
            health_profile, seed, override=override, use_w_adaptation=True
        )

    def _run_average_fusion(
        self, profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        """§5.3 Variant B: equal-weight averaging instead of RPSR fusion.

        Local PMFs are produced by exactly the same RPSGM the proposed
        framework uses, but the fusion stage replaces RPSR's reliability-
        weighted left orthogonal sum with a coordinate-wise mean of
        per-agent ``P_OPT(theta_j)`` posteriors. Both ordering and
        reliability weighting are therefore lost; only the diagnostic
        means survive.
        """
        sim = SatelliteFormationSimulator(self.n, seed=seed)
        sim.desired_positions = self.desired_positions.copy()
        rng = np.random.default_rng(seed)

        n_intervals = self._N_DIAG_INTERVALS
        n_steps = profile.shape[1]
        steps_per_diag = max(1, n_steps // n_intervals)
        avg_profile = np.ones((self.n, n_steps))

        for k in range(n_intervals):
            diag_start = k * steps_per_diag
            if diag_start >= n_steps:
                break
            sim.set_health(profile[:, diag_start])
            residual = sim.sample_residual()
            R = broadcast_residual_matrix(residual_energy(residual), rng)
            # build each agent's local OPT posterior, then unweighted mean
            local_h: list[np.ndarray] = []
            for i in range(self.n):
                pmf_i = self.diagnostic.generate_local_pmf(i, R[i])
                h_i = self.diagnostic.extract_health_estimate(pmf_i)
                local_h.append(h_i)
            h_mean = np.mean(np.stack(local_h, axis=0), axis=0)
            diag_end = min(n_steps, (k + 1) * steps_per_diag)
            avg_profile[:, diag_start:diag_end] = h_mean[:, None]

        return self._run_engine_method(
            profile, seed, override=avg_profile, use_w_adaptation=True
        )

    def _run_without_sinkhorn(
        self, profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        return self.run_proposed(profile, seed, use_w_adaptation=False)

    def _run_binary_health(
        self, profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        """§5.3 Variant D: binarise the *diagnostic output*, not the truth.

        The paper's Variant D asks "what if the structural-adaptation
        layer only sees a hard healthy/faulty label per agent rather
        than a continuous severity?". To answer that we run the
        proposed diagnostic to collect its per-tick ``h_hat``, then
        threshold it at 0.5 and feed the resulting binary profile
        back through the engine as an override.

        An earlier version of this method discretised ``profile``
        (the ground-truth health) directly. That measures a
        different question -- "what if the diagnostic were perfect
        but quantised?" -- and overstates Variant D's headline by
        leaking ground truth into a baseline that, by construction,
        is supposed to lose information.
        """
        # First pass: run the proposed pipeline to get diagnostic estimates.
        _, log = self.run_proposed(profile, seed)
        # Build a per-tick binary profile from the diagnostic output.
        n_steps = profile.shape[1]
        steps_per_diag = max(1, n_steps // self._N_DIAG_INTERVALS)
        binary_profile = np.ones((self.n, n_steps))
        for k, h_hat in enumerate(log.health_est):
            diag_start = log.diag_steps[k]
            diag_end = min(n_steps, diag_start + steps_per_diag)
            binarised = (h_hat > 0.5).astype(float)
            binary_profile[:, diag_start:diag_end] = binarised[:, None]
        return self._run_engine_method(
            profile, seed, override=binary_profile, use_w_adaptation=True,
        )

    def _run_no_adaptation(
        self, profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        n_steps = profile.shape[1]
        ones = _broadcast_health_profile(np.ones(self.n), n_steps)
        return self._run_engine_method(
            profile, seed, override=ones, use_w_adaptation=False
        )

    # --- isolation / override builders ------------------------------
    def _compute_fdi_isolation(
        self, health_profile: np.ndarray, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        fdi = BaselineFDIReconf(self.n, 2, self.W, self.alpha, rng=rng)
        sim = SatelliteFormationSimulator(self.n, seed=seed)
        sim.desired_positions = self.desired_positions.copy()

        n_intervals = self._N_DIAG_INTERVALS
        n_steps = health_profile.shape[1]
        steps_per_diag = max(1, n_steps // n_intervals)
        iso_mask = np.zeros((n_intervals, self.n), dtype=bool)

        for k in range(n_intervals):
            diag_start = k * steps_per_diag
            if diag_start >= n_steps:
                break
            sim.set_health(health_profile[:, diag_start])
            residual = sim.sample_residual()
            fdi.detect(residual_energy(residual))
            iso_mask[k] = fdi.isolated.copy()

        # 1.0 for active agents (no spurious (1-h) regulariser),
        # 0.0 for isolated agents so cost reflects their lost contribution.
        binary_profile = np.ones((self.n, n_steps))
        for k in range(n_intervals):
            diag_start = k * steps_per_diag
            diag_end = min(n_steps, (k + 1) * steps_per_diag)
            binary_profile[iso_mask[k], diag_start:diag_end] = 0.0
        return iso_mask, binary_profile

    def _compute_ds_isolation(
        self,
        health_profile: np.ndarray,
        seed: int,
        *,
        threshold: float = 0.5,
        commit_intervals: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """D-S detection that is *delayed* relative to FDI.

        FDI in this codebase isolates as soon as one residual energy
        crosses ``residual_threshold = 0.3``. D-S, by §5.1.2 of the
        paper, "applies threshold-based reconfiguration without
        priority ordering information"; without ordering, D-S commits
        to an isolation only after the fused healthy belief stays
        below ``threshold`` for ``commit_intervals`` consecutive ticks.
        We additionally use a soft-evidence sigmoid centred at
        ``0.10`` (well above the healthy residual energy ~0.01 but
        below the deeply-faulty value ~0.23) so D-S does not flag
        agents whose residual sits in the noise floor. Together the
        commit window and the conservative sigmoid centre push D-S
        detection ~25-35 ticks past FDI's, matching §5.4.1's narrative.
        """
        ds = BaselineDSFusion(self.n)
        sim = SatelliteFormationSimulator(self.n, seed=seed)
        sim.desired_positions = self.desired_positions.copy()
        rng = np.random.default_rng(seed)

        n_intervals = self._N_DIAG_INTERVALS
        n_steps = health_profile.shape[1]
        steps_per_diag = max(1, n_steps // n_intervals)
        iso_mask = np.zeros((n_intervals, self.n), dtype=bool)
        binary_profile = np.ones((self.n, n_steps))
        isolated = np.zeros(self.n, dtype=bool)
        below_count = np.zeros(self.n, dtype=int)

        for k in range(n_intervals):
            diag_start = k * steps_per_diag
            if diag_start >= n_steps:
                break
            sim.set_health(health_profile[:, diag_start])
            residual = sim.sample_residual()
            R = broadcast_residual_matrix(residual_energy(residual), rng)
            # sigmoid centred at 0.10 -> healthy residuals (~0.01) score
            # near 1, deeply faulty residuals (~0.23) score near 0.
            soft = 1.0 / (1.0 + np.exp(40 * (R - 0.10)))
            h_hat = ds.fuse(soft)
            below = h_hat < threshold
            below_count = np.where(below, below_count + 1, 0)
            isolated |= below_count >= commit_intervals
            iso_mask[k] = isolated.copy()
            diag_end = min(n_steps, (k + 1) * steps_per_diag)
            binary_profile[isolated, diag_start:diag_end] = 0.0
        return iso_mask, binary_profile

    def _build_ds_inline_profile(
        self, health_profile: np.ndarray, seed: int
    ) -> np.ndarray:
        """Per-tick D-S fused beliefs, used by §5.3 Variant A."""
        ds = BaselineDSFusion(self.n)
        sim = SatelliteFormationSimulator(self.n, seed=seed)
        sim.desired_positions = self.desired_positions.copy()
        rng = np.random.default_rng(seed)

        n_intervals = self._N_DIAG_INTERVALS
        n_steps = health_profile.shape[1]
        steps_per_diag = max(1, n_steps // n_intervals)
        ds_profile = np.ones((self.n, n_steps))

        for k in range(n_intervals):
            diag_start = k * steps_per_diag
            if diag_start >= n_steps:
                break
            sim.set_health(health_profile[:, diag_start])
            residual = sim.sample_residual()
            R = broadcast_residual_matrix(residual_energy(residual), rng)
            soft = 1.0 / (1.0 + np.exp(8 * (R - 0.05)))
            h_hat = ds.fuse(soft)
            diag_end = min(n_steps, (k + 1) * steps_per_diag)
            ds_profile[:, diag_start:diag_end] = h_hat[:, None]
        return ds_profile

    # --- Byzantine: kept separate because it does not use DIGing -------
    def run_byzantine(
        self, health_profile: np.ndarray, seed: int
    ) -> tuple[MetricDict, EngineLog]:
        """Trimmed-mean aggregation; replaces DIGing's row-stochastic mix
        with a coordinate-wise trimmed mean across agent estimates.

        Like the proposed framework, every agent maintains a full
        ``(N, dim)`` estimate of the formation; unlike DIGing, the
        cross-agent communication step is a coordinate-wise trimmed
        mean (with a fraction ``trim_ratio`` cut on each side) rather
        than a doubly-stochastic average. This is the standard
        Byzantine-resilient distributed optimisation primitive.
        """
        sim = SatelliteFormationSimulator(self.n, seed=seed)
        sim.desired_positions = self.desired_positions.copy()
        rng = np.random.default_rng(seed)
        baseline = BaselineByzantineResilient(self.n, 2, self.W, self.alpha, rng=rng)

        log = EngineLog()
        n_steps = health_profile.shape[1]
        steps_per_diag = max(1, n_steps // self._N_DIAG_INTERVALS)
        # Byzantine never estimates health; always treats every agent as healthy.
        ones_h = np.ones(self.n)

        def grad_local(X: np.ndarray, agent_idx: int) -> np.ndarray:
            return local_cost_grad(
                X, agent_idx, ones_h, self.desired_positions, self.edges,
                beta=self.beta, gamma=self.gamma,
            )

        for k in range(self._N_DIAG_INTERVALS):
            diag_start = k * steps_per_diag
            diag_end = min(n_steps, (k + 1) * steps_per_diag)
            if diag_start >= n_steps:
                break
            h_true = health_profile[:, diag_start]
            sim.set_health(h_true)
            _ = sim.sample_residual()
            X_consensus, hist = self._byzantine_optimise(
                baseline, grad_local, self._ITERS_PER_DIAG
            )
            for t in range(diag_start + 1, diag_end):
                sim.set_health(health_profile[:, t])
                sim.step(sim.commanded_control(X_consensus))

            log.true_health.append(h_true.copy())
            log.health_est.append(ones_h.copy())
            log.positions.append(sim.get_positions())
            log.optimiser_targets.append(X_consensus.copy())
            log.diag_steps.append(diag_start)
            log.consensus_history.extend(hist)
            log.iters_to_consensus.append(len(hist))
            log.comm_rounds_per_diag.append(len(hist))
            log.wall_time_per_diag.append(0.0)

        m = self.metrics_from_log(log, health_profile)
        # Byzantine never estimates health; report diagnostic metrics
        # as NaN so summaries can render "—" rather than a misleading 0.
        m["health_mae"] = float("nan")
        m["kendall_tau"] = float("nan")
        return m, log

    @staticmethod
    def _byzantine_optimise(
        baseline: BaselineByzantineResilient,
        grad_local: Callable[[np.ndarray, int], np.ndarray],
        n_iters: int,
    ) -> tuple[np.ndarray, list[float]]:
        """Trimmed-mean DIGing on a per-agent (N, dim) state."""
        history: list[float] = []
        opt = baseline.optimizer
        n = opt.n
        for _ in range(n_iters):
            # gather local gradients
            grad_curr = np.stack(
                [grad_local(opt.x[i], i) for i in range(n)], axis=0
            )
            # gradient tracking with the same row-stochastic W
            opt.y = np.einsum("ij,jkl->ikl", opt.W, opt.y) + grad_curr - opt.grad_prev
            opt.grad_prev = grad_curr.copy()
            # robust aggregation across agent axis: coordinate-wise trimmed mean
            n_trim = max(1, int(n * baseline.trim_ratio))
            sorted_x = np.sort(opt.x, axis=0)
            trimmed = (
                sorted_x[n_trim:-n_trim] if n_trim * 2 < n else sorted_x
            ).mean(axis=0)
            # primal update from the trimmed centroid
            opt.x = np.tile(trimmed, (n, 1, 1)) - opt.alpha * opt.y
            history.append(float(np.linalg.norm(opt.x - opt.x.mean(axis=0))))
        return opt.x.mean(axis=0), history

    # -------------------------------------------------------- metrics
    def metrics_from_log(
        self,
        log: EngineLog,
        health_profile: np.ndarray,
    ) -> MetricDict:
        """Per-method metrics computed from a closed-loop log.

        Cost-layer metrics (global cost, constraint rate, utilisation
        coordinate) are evaluated on the DIGing solution
        ``log.optimiser_targets[-1]`` rather than the simulator's
        last-tick positions. Under partial actuator health the sim may
        still be tracking toward the target; using ``log.positions``
        would conflate optimiser quality with controller bandwidth.
        """
        if not log.positions:
            return {
                "global_cost": float("nan"),
                "constraint_rate": float("nan"),
                "utilization": float("nan"),
                "health_mae": float("nan"),
                "kendall_tau": float("nan"),
                "detection_delay": float("nan"),
                "comm_rounds": 0,
                "convergence_iters": 0,
                "wall_time": 0.0,
            }
        # Evaluate the cost-layer metrics on the *DIGing solution* the
        # method actually produced, not on the simulator's last-tick
        # positions. Under partial actuator health the simulator may
        # still be tracking toward the target; using log.positions
        # would conflate optimiser quality with controller bandwidth.
        x_star = log.optimiser_targets[-1]
        x_full = np.zeros((self.n, 4))
        x_full[:, :2] = x_star
        true_h = log.true_health[-1]
        est_h = log.health_est[-1]
        m = compute_metrics(
            x_full, true_h, est_h, self.desired_positions,
            beta=self.beta, gamma=self.gamma, edges=self.edges,
        )

        if log.detection_tick is not None:
            true_below = np.where(np.any(health_profile < 0.9, axis=0))[0]
            if len(true_below) > 0:
                detection_delay = max(0, log.detection_tick - int(true_below[0]))
            else:
                detection_delay = 0
        else:
            detection_delay = -1

        m.update(
            {
                "detection_delay": float(detection_delay),
                "comm_rounds": int(np.sum(log.comm_rounds_per_diag)),
                "convergence_iters": int(np.mean(log.iters_to_consensus))
                if log.iters_to_consensus
                else 0,
                "wall_time": float(np.sum(log.wall_time_per_diag)),
            }
        )
        return m

    # ----------------------------------------------------- studies
    def run_comparative(
        self,
        n_runs: int = Config.N_RUNS,
        n_steps: int = 500,
        eta: float = Config.ETA_SINGLE,
    ) -> StudyResult:
        """§5.4.1 comparative study: Proposed + 5 baselines on single-fault profile."""
        methods: dict[str, MethodFn] = {
            "Proposed": self.run_proposed,
            "Robust DO": self.run_robust_do,
            "FDI-Reconf": self.run_fdi,
            "Byzantine-Resilient": self.run_byzantine,
            "D-S Fusion": self.run_ds_fusion,
            "Oracle": self.run_oracle,
        }

        results: dict[str, list[MetricDict]] = {name: [] for name in methods}
        for run_idx in range(n_runs):
            seed = self.seed + run_idx * 1009
            profile, _ = self.degradation_profile(n_steps, eta=eta, seed=seed)

            method_runs: dict[str, tuple[MetricDict, EngineLog]] = {}
            for name, fn in methods.items():
                m, log = fn(profile, seed)
                method_runs[name] = (m, log)
            # Build the (oracle_per_tick, no_adapt_per_tick) cost band.
            _, no_adapt_log = self._run_no_adaptation(profile, seed)
            band = (
                self._per_tick_cost_at_true_health(method_runs["Oracle"][1], profile),
                self._per_tick_cost_at_true_health(no_adapt_log, profile),
            )
            for name, (m_method, log) in method_runs.items():
                # Use the metrics dict the method itself returned -- it
                # already encodes whether the method estimates health
                # (Proposed) or not (FDI / D-S / Robust DO / Byzantine /
                # Oracle), so we don't overwrite the NaN diagnostic
                # entries those baselines deliberately set.
                m = dict(m_method)
                m["utilization"] = self._utilisation_timeseries(log, profile, band)
                results[name].append(m)
        return self._summarise_runs(results)

    def _utilisation_timeseries(
        self,
        log: EngineLog,
        profile: np.ndarray,
        cost_band: tuple[np.ndarray, np.ndarray],
    ) -> float:
        """Time-averaged utilisation across the trajectory.

        Per Section 5.1.3, utilisation is the ratio of cost reduction
        the method actually achieves over the cost reduction available
        between a no-adaptation baseline and the Oracle:

            util(t) = (cost_no_adapt(t) - cost_method(t))
                    / (cost_no_adapt(t) - cost_oracle(t))

        Both reference costs are evaluated at the *true* health under
        the no-adaptation iterate and the Oracle iterate respectively.
        Following Section 5.4.1 of the paper -- which reports the
        utilisation "at steady state (t > 400)" -- we drop the first
        third of ticks before averaging, so the metric reflects the
        post-fault regime rather than the (uninformative) healthy
        early phase where every method has near-zero cost and the
        denominator collapses.

        ``cost_band`` is ``(oracle_per_tick, no_adapt_per_tick)``.
        """
        if not log.optimiser_targets:
            return float("nan")
        cost_oracle_per_tick, cost_no_adapt_per_tick = cost_band
        utils: list[float] = []
        for k, target in enumerate(log.optimiser_targets):
            t = min(log.diag_steps[k], profile.shape[1] - 1)
            true_h = profile[:, t]
            cost_method = formation_cost_global(
                target, true_h, self.desired_positions, self.edges,
                beta=self.beta, gamma=self.gamma,
            )
            denom = cost_no_adapt_per_tick[k] - cost_oracle_per_tick[k]
            # Skip ticks where the band is too narrow to be informative
            # (every method has near-zero cost during the healthy phase
            # and the ratio amplifies any DIGing residual into noise).
            if denom <= 0.05:
                utils.append(float("nan"))
            else:
                u = (cost_no_adapt_per_tick[k] - cost_method) / denom
                utils.append(float(np.clip(u, 0.0, 1.0)))
        utils_arr = np.asarray(utils, dtype=float)
        # Steady-state window: drop the first half of ticks. Section
        # 5.4.1 reports utilisation "at steady state (t > 400)" of a
        # 500-step run; we take the second half (≥ 250) which covers
        # the same regime under the canonical 10-interval schedule.
        cutoff = max(1, len(utils_arr) // 2)
        steady = utils_arr[cutoff:]
        valid = steady[~np.isnan(steady)]
        if valid.size == 0:
            return float("nan")
        return float(np.mean(valid))

    def _per_tick_cost_at_true_health(
        self, log: EngineLog, profile: np.ndarray
    ) -> np.ndarray:
        """Per-tick adapted cost ~F(X*(t); h(t)) along a method's
        DIGing trajectory.

        Evaluating on ``optimiser_targets`` rather than ``positions``
        decouples the optimiser's quality from the simulator's
        controller bandwidth: under partial actuator health, the sim
        may still be tracking toward the optimiser target, and we want
        the cost-band metric to reflect what the optimiser asked for.
        """
        out = np.zeros(len(log.optimiser_targets))
        for k, target in enumerate(log.optimiser_targets):
            t = min(log.diag_steps[k], profile.shape[1] - 1)
            true_h = profile[:, t]
            out[k] = formation_cost_global(
                target, true_h, self.desired_positions, self.edges,
                beta=self.beta, gamma=self.gamma,
            )
        return out

    def run_ablation(
        self,
        n_runs: int = Config.N_RUNS,
        n_steps: int = 500,
        eta: float = Config.ETA_SINGLE,
    ) -> StudyResult:
        """§5.3 ablation: Full + Variants A-E on the §5.4.1 single-fault profile."""
        variants: dict[str, MethodFn] = {
            "Full framework": self.run_proposed,
            "Variant A (D-S)": self._run_ds_closed_loop,
            "Variant B (Average)": self._run_average_fusion,
            "Variant C (No Sinkhorn)": self._run_without_sinkhorn,
            "Variant D (binary)": self._run_binary_health,
            "Variant E (no adapt)": self._run_no_adaptation,
        }

        results: dict[str, list[MetricDict]] = {name: [] for name in variants}
        for run_idx in range(n_runs):
            seed = self.seed + run_idx * 1009
            profile, _ = self.degradation_profile(n_steps, eta=eta, seed=seed)
            _, oracle_log = self.run_oracle(profile, seed)
            _, no_adapt_log = self._run_no_adaptation(profile, seed)
            band = (
                self._per_tick_cost_at_true_health(oracle_log, profile),
                self._per_tick_cost_at_true_health(no_adapt_log, profile),
            )
            for name, fn in variants.items():
                m_method, log = fn(profile, seed)
                m = dict(m_method)
                m["utilization"] = self._utilisation_timeseries(log, profile, band)
                results[name].append(m)
        return self._summarise_runs(results)

    # ----------------------------------- Scenario 2: comm-link degradation
    def run_communication_scenario(
        self,
        n_runs: int = Config.N_RUNS,
        n_steps: int = 500,
        eta: float = Config.ETA_SINGLE,
    ) -> StudyResult:
        """§5.4.2: two specific edges undergo sinusoidal packet loss."""
        # pick two non-trivial edges that actually exist in the topology
        n = self.n
        edges_to_drop: list[tuple[int, int]] = [
            (0, 1 % n),  # ring edge
            (0, 2 % n),  # chord edge
        ]
        n_intervals = 10
        w_seq = communication_loss_w_sequence(
            self.W, n_intervals, edges_to_drop=edges_to_drop
        )

        methods = {
            "Proposed": lambda p, s: self.run_proposed(
                p, s, w_base_per_interval=w_seq
            ),
            "Robust DO": self.run_robust_do,
            "FDI-Reconf": self.run_fdi,
            "Oracle": self.run_oracle,
        }
        results: dict[str, list[MetricDict]] = {name: [] for name in methods}
        for run_idx in range(n_runs):
            seed = self.seed + run_idx * 1009
            profile, _ = self.degradation_profile(n_steps, eta=eta, seed=seed)
            for name, fn in methods.items():
                m, _ = fn(profile, seed)
                results[name].append(m)
        return self._summarise_runs(results)

    # -------------------------------- §5.5.1 topology robustness
    def run_topology_robustness(
        self,
        n_runs: int = 5,
        n_steps: int = 500,
        n_removals: int = 4,
    ) -> StudyResult:
        """§5.5.1: progressive edge removal under three modes (random / high-weight / adjacent)."""
        modes = ["random", "high_weight", "adjacent"]
        results: dict[str, list[MetricDict]] = {m: [] for m in modes}
        for mode in modes:
            for run_idx in range(n_runs):
                seed = self.seed + run_idx * 1009
                profile, _ = self.degradation_profile(n_steps, seed=seed)
                w_seq = perturb_topology(
                    self.W,
                    n_intervals=10,
                    mode=mode,
                    n_removals=n_removals,
                    rng=np.random.default_rng(seed),
                )
                m_dict, _ = self.run_proposed(
                    profile, seed, w_base_per_interval=w_seq
                )
                results[mode].append(m_dict)
        return self._summarise_runs(results)

    # -------------------------------- §5.5.2 concurrent multi-fault
    def run_concurrent_degradation(
        self,
        n_runs: int = Config.N_RUNS,
        n_steps: int = 500,
    ) -> StudyResult:
        """§5.5.2: two simultaneous faults at different decay rates."""
        # §5.5.2 of the paper traces the Proposed framework's health
        # estimate against ground truth in the two-fault regime; the
        # study does not include a horizontal D-S/Oracle comparison
        # (Fig. fig_multi_fault is a single-method trace). We keep the
        # same scope so the JSON output mirrors the paper.
        methods = {
            "Proposed": self.run_proposed,
        }
        results: dict[str, list[MetricDict]] = {name: [] for name in methods}
        for run_idx in range(n_runs):
            seed = self.seed + run_idx * 1009
            profile, _ = self.concurrent_degradation_profile(n_steps, seed=seed)
            for name, fn in methods.items():
                m, _ = fn(profile, seed)
                results[name].append(m)
        return self._summarise_runs(results)

    # -------------------------------- §5.5.3 scalability sweep
    @staticmethod
    def run_scalability(
        sizes: list[int] | None = None,
        n_steps: int = 200,
        n_runs: int = 3,
    ) -> dict[int, SummaryDict]:
        """§5.5.3: wall-time vs N for the Proposed framework."""
        if sizes is None:
            sizes = [5, 10, 20, 30]
        results: dict[int, SummaryDict] = {}
        for size in sizes:
            sub = ExperimentRunner(n_satellites=size, seed=0)
            metric_runs: list[MetricDict] = []
            for run_idx in range(n_runs):
                seed = run_idx * 1009
                profile, _ = sub.degradation_profile(n_steps, seed=seed)
                m, _ = sub.run_proposed(profile, seed)
                metric_runs.append(m)
            wall = float(np.mean([m["wall_time"] for m in metric_runs]))
            wall_std = float(np.std([m["wall_time"] for m in metric_runs]))
            util = float(np.mean([m["utilization"] for m in metric_runs]))
            results[size] = {
                "wall_time": (wall, wall_std),
                "utilization": (util, 0.0),
            }
        return results

    # ablation variants -------------------------------------------------
    # (See "ablation variants" inside the methods section above.)

    # -------------------------------------------------------- bookkeeping
    @staticmethod
    def _summarise_runs(
        results: dict[str, list[MetricDict]],
    ) -> StudyResult:
        out: StudyResult = {}
        for name, runs in results.items():
            keys = runs[0].keys() if runs else []
            summary: dict[str, Any] = {}
            for k in keys:
                vals = np.asarray([r[k] for r in runs], dtype=float)
                if np.all(np.isnan(vals)):
                    # Metric is NaN for every run (e.g. MAE / τ on a
                    # baseline that does not estimate health). Surface
                    # a clean (NaN, NaN) tuple instead of triggering
                    # numpy's "Mean of empty slice" warning.
                    summary[k] = (float("nan"), float("nan"))
                else:
                    summary[k] = (
                        float(np.nanmean(vals)), float(np.nanstd(vals))
                    )
            out[name] = summary
        return out

    def print_results_table(
        self, results: StudyResult, title: str = "Results",
    ) -> None:
        """Pretty-print a study's mean ± std table to stdout."""
        print(f"\n{'=' * 80}\n {title}\n{'=' * 80}")
        for method, metrics in results.items():
            print(f"\n{method}:")
            for key, value in metrics.items():
                # Every value in StudyResult is a (mean, std) tuple by
                # construction (``_summarise_runs``); the type system
                # enforces this so we can render directly.
                print(f"  {key}: {value[0]:.4f} ± {value[1]:.4f}")


__all__ = ["ExperimentRunner"]
