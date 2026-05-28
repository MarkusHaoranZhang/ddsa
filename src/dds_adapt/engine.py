"""Closed-loop diagnosis-driven structural adaptation engine.

This is the run-time realisation of the architecture diagrammed in
Figure 1 of the paper. Each diagnosis interval:

1. The simulator is set to the *current* health and the diagnostic
   probe samples a residual that depends on the actuator effectiveness.
2. The diagnostic module fuses per-agent residuals into a fused PMF and
   emits a continuous health estimate ``h_hat``.
3. ``h_hat`` parametrises the optimisation problem (Section 4.2): cost
   weights are re-scaled, the mixing matrix is attenuated by
   ``min(h_i, h_j)`` and projected back to doubly-stochastic by
   Sinkhorn-Knopp.
4. DIGing runs on the adapted problem with the global gradient that
   includes both the per-agent tracking term and the formation-keeping
   coupling (Section 5.1.1).
5. The DIGing iterate :math:`x^{*}` is fed back as the simulator's
   target, closing the diagnosis-optimisation-control loop.

Every quantity downstream experiments need is recorded in ``EngineLog``
so plots and tables never reach into private state.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from dds_adapt.config import Config
from dds_adapt.cost import local_cost_grad
from dds_adapt.diagnostic import RPSDiagnosticModule
from dds_adapt.optimizer import DIGingOptimizer
from dds_adapt.residual import broadcast_residual_matrix, residual_energy
from dds_adapt.simulator import SatelliteFormationSimulator
from dds_adapt.utils import adapt_mixing_matrix


class _SimulatorProtocol(Protocol):
    """Subset of ``SatelliteFormationSimulator`` the engine actually uses."""

    desired_positions: np.ndarray

    def set_health(self, health: np.ndarray) -> None: ...
    def sample_residual(self) -> np.ndarray: ...
    def commanded_control(self, target: np.ndarray) -> np.ndarray: ...
    def step(self, commanded: np.ndarray): ...
    def get_positions(self) -> np.ndarray: ...


@dataclass
class EngineLog:
    """Per-tick recordings used by every experiment in Section 5."""

    true_health: list[np.ndarray] = field(default_factory=list)
    health_est: list[np.ndarray] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)
    optimiser_targets: list[np.ndarray] = field(default_factory=list)
    diag_steps: list[int] = field(default_factory=list)
    consensus_history: list[float] = field(default_factory=list)
    iters_to_consensus: list[int] = field(default_factory=list)
    comm_rounds_per_diag: list[int] = field(default_factory=list)
    wall_time_per_diag: list[float] = field(default_factory=list)
    detection_tick: int | None = None  # first tick where any h_hat < 0.9


def run_closed_loop(
    health_profile: np.ndarray,
    diagnostic: RPSDiagnosticModule,
    W_base: np.ndarray,
    desired_positions: np.ndarray,
    *,
    edges: list[tuple[int, int]] | None = None,
    n_diag_intervals: int = 10,
    iters_per_diag: int = 50,
    sim_dt: float = 0.1,
    alpha: float = Config.ALPHA_NUM,
    gamma: float = Config.GAMMA_NUM,
    beta: float = Config.BETA,
    cross_agent_noise_std: float = 0.005,
    seed: int = 0,
    use_w_adaptation: bool = True,
    detection_threshold: float = 0.9,
    health_estimate_override: np.ndarray | None = None,
    w_base_per_interval: list[np.ndarray] | None = None,
    simulator_factory: Callable[[], _SimulatorProtocol] | None = None,
    isolation_mask: np.ndarray | None = None,
) -> EngineLog:
    """Run the full closed loop and return the recorded telemetry.

    ``health_estimate_override`` lets the caller bypass the RPS module
    (Oracle baseline / no-adaptation ablation): when provided it is
    used directly instead of the diagnostic module's output. Shape
    must match ``health_profile`` row-wise.

    ``w_base_per_interval`` lets the caller swap the *base* communication
    graph at every diagnostic interval; this is what scenario 2 needs to
    inject communication-link degradation. Length must be at least
    ``n_diag_intervals``.

    ``isolation_mask`` is a per-tick boolean ``(n_diag_intervals, n)``
    matrix; an agent flagged ``True`` at a given tick is treated as
    physically removed from the formation: its position is held at the
    formation reference and its cost contribution is taken from the
    DIGing iterate but its row of the gradient is zeroed so subsequent
    DIGing updates do not move it. FDI and threshold-based D-S use this
    to model the paper's "recompute formation using only healthy agents"
    semantics.
    """
    n = W_base.shape[0]
    if health_profile.shape[0] != n:
        raise ValueError(
            f"health_profile has {health_profile.shape[0]} rows, "
            f"expected {n} agents."
        )
    if desired_positions.shape != (n, 2):
        raise ValueError(
            f"desired_positions must have shape ({n}, 2), got "
            f"{desired_positions.shape}."
        )
    if edges is None:
        edges = Config.edges(n)

    rng = np.random.default_rng(seed)
    sim: _SimulatorProtocol
    if simulator_factory is None:
        concrete_sim = SatelliteFormationSimulator(n_satellites=n, dt=sim_dt, seed=seed)
        concrete_sim.desired_positions = desired_positions.copy()
        sim = concrete_sim
    else:
        sim = simulator_factory()
    optimiser = DIGingOptimizer(
        n_agents=n, dim=2, W=W_base.copy(), alpha=alpha, rng=rng
    )

    log = EngineLog()
    total_steps = health_profile.shape[1]
    steps_per_diag = max(1, total_steps // n_diag_intervals)

    for k in range(n_diag_intervals):
        diag_start = k * steps_per_diag
        diag_end = min(total_steps, (k + 1) * steps_per_diag)
        if diag_start >= total_steps:
            break

        # base graph for this interval (scenario 2: time-varying)
        if w_base_per_interval is not None and k < len(w_base_per_interval):
            W_base_now = w_base_per_interval[k]
        else:
            W_base_now = W_base

        # ------------------ 1. apply true health, sample residual -------
        h_true = health_profile[:, diag_start]
        sim.set_health(h_true)
        residual = sim.sample_residual()
        own_energy = residual_energy(residual)
        residual_matrix = broadcast_residual_matrix(
            own_energy, rng, cross_agent_noise_std=cross_agent_noise_std
        )

        # ------------------ 2. diagnose ---------------------------------
        t0 = time.perf_counter()
        if health_estimate_override is not None:
            h_hat = np.clip(health_estimate_override[:, diag_start], 0.0, 1.0)
        else:
            h_hat, _, _ = diagnostic.diagnose_round(residual_matrix)

        # ------------------ 3. adapt the mixing matrix ------------------
        W_tilde = adapt_mixing_matrix(W_base_now, h_hat) if use_w_adaptation else W_base_now
        optimiser.set_mixing_matrix(W_tilde)

        # ------------------ 4. distributed optimisation -----------------
        # If an isolation mask is supplied, agents flagged True at this
        # tick are physically removed from the formation: their iterate
        # is pinned to the reference, their gradient is zeroed, AND
        # every formation-keeping edge incident to them is dropped.
        # The dropped edges matter: without filtering, surviving
        # neighbours still feel the coupling pull toward the isolated
        # agent's reference position, which makes the FDI-Reconf
        # solution numerically indistinguishable from "no adaptation"
        # despite the very different physical interpretation.
        iso_mask = (
            isolation_mask[k] if isolation_mask is not None and k < len(isolation_mask)
            else np.zeros(n, dtype=bool)
        )
        active_edges = (
            [(a, b) for (a, b) in edges if not iso_mask[a] and not iso_mask[b]]
            if iso_mask.any() else edges
        )

        def grad(
            X: np.ndarray,
            agent_idx: int,
            _h: np.ndarray = h_hat,
            _iso: np.ndarray = iso_mask,
            _edges: list[tuple[int, int]] = active_edges,
        ) -> np.ndarray:
            # agent_idx-owned local cost gradient: own tracking + own
            # safe-anchor + edges (agent_idx, j) for j > agent_idx,
            # restricted to edges between two active agents.
            g = local_cost_grad(
                X, agent_idx, _h, desired_positions, _edges,
                beta=beta, gamma=gamma,
            )
            if _iso[agent_idx]:
                # an isolated agent contributes no force
                g[:] = 0.0
            return g

        if iso_mask.any():
            # pin every agent's *estimate* of the isolated rows to the reference
            for iso_idx in np.where(iso_mask)[0]:
                optimiser.x[:, iso_idx, :] = desired_positions[iso_idx]
        x_per_agent, hist = optimiser.optimize(
            grad, n_iters=iters_per_diag
        )
        # consensus estimate of the whole formation
        X_consensus = optimiser.consensus_estimate()
        if iso_mask.any():
            X_consensus = X_consensus.copy()
            X_consensus[iso_mask] = desired_positions[iso_mask]
        wall = time.perf_counter() - t0
        # Each agent steers toward its own row of the consensus formation;
        # after DIGing converges the per-agent estimates agree closely so
        # the steering target is essentially the global optimum X*.
        per_agent_target = X_consensus

        # ------------------ 5. apply optimiser output to physics --------
        for inner in range(diag_start + 1, diag_end):
            sim.set_health(health_profile[:, inner])
            sim.step(sim.commanded_control(per_agent_target))

        # ------------------ 6. log --------------------------------------
        log.true_health.append(h_true.copy())
        log.health_est.append(h_hat.copy())
        log.positions.append(sim.get_positions())
        log.optimiser_targets.append(per_agent_target.copy())
        log.diag_steps.append(diag_start)
        log.consensus_history.extend(hist)
        log.iters_to_consensus.append(len(hist))
        log.comm_rounds_per_diag.append(len(hist))
        log.wall_time_per_diag.append(wall)
        if log.detection_tick is None and (h_hat < detection_threshold).any():
            log.detection_tick = diag_start

    return log
