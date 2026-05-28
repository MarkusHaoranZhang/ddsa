"""Three-layer metric system used by the experiments.

This module is responsible for the *cost-layer* and *diagnostic-layer*
slice of the paper's nine-metric protocol: ``global_cost``,
``constraint_rate``, ``utilization`` (proximity-based fallback),
``health_mae``, ``kendall_tau``. The system-efficiency layer
(``detection_delay``, ``comm_rounds``, ``convergence_iters``,
``wall_time``) is added by ``runner.metrics_from_log`` from the
engine telemetry, so the two together produce the full 9-metric dict.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ddsa.config import Config
from ddsa.cost import formation_cost_global


def compute_metrics(
    x_final: np.ndarray,
    true_health: np.ndarray,
    health_estimate: np.ndarray,
    desired_positions: np.ndarray,
    *,
    beta: float = Config.BETA,
    gamma: float = Config.GAMMA_NUM,
    edges: list[tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Return the dictionary of task / diagnostic metrics.

    Utilisation is computed as a proximity-based proxy here; the runner
    overrides it with the time-averaged cost-band version for the
    comparative and ablation tables.
    """
    n_agents = len(true_health)
    if edges is None:
        edges = Config.edges(n_agents)

    # Task layer ----------------------------------------------------------
    global_cost = formation_cost_global(
        x_final[:, :2],
        true_health,
        desired_positions,
        edges,
        beta=beta,
        gamma=gamma,
    )

    constraints_satisfied = 0
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            error = np.linalg.norm(
                x_final[i, :2]
                - x_final[j, :2]
                - (desired_positions[i] - desired_positions[j])
            )
            # The 2.0 m tolerance is twice the unit formation radius
            # (Config.SAFE_OFFSET-scale): a pair is "in formation" if
            # its inter-agent error is below 2x the nominal spacing.
            if error <= 2.0:
                constraints_satisfied += 1
    total_pairs = n_agents * (n_agents - 1) / 2
    constraint_rate = constraints_satisfied / total_pairs if total_pairs > 0 else 1.0

    # Proximity-based utilisation (fallback only; the comparative /
    # ablation tables in runner.run_comparative override this with a
    # cost-band time-average per Section 5.4.1 of the paper).
    actual = 0.0
    denom = 0.0
    for i in range(n_agents):
        if true_health[i] < 1.0:
            err = float(np.linalg.norm(x_final[i, :2] - desired_positions[i]))
            # Same 2.0 m scale as the constraint tolerance above: an
            # agent at the reference scores 1.0, an agent 2 m away
            # scores 0.0, linearly in between.
            proximity = max(0.0, 1.0 - err / 2.0)
            actual += true_health[i] * proximity
            denom += 1.0
    utilization = actual / denom if denom > 0 else 0.0

    # Diagnostic layer ----------------------------------------------------
    health_mae = float(np.mean(np.abs(health_estimate - true_health)))

    true_severity = 1.0 - np.asarray(true_health, dtype=float)
    est_severity = 1.0 - np.asarray(health_estimate, dtype=float)
    # Methods that do not estimate continuous health (FDI / Byzantine /
    # binary D-S) emit a constant vector. Kendall τ on a constant
    # ranking is undefined; report NaN so downstream summaries can
    # render "—" rather than a misleading 0.
    if np.unique(est_severity).size < 2 or np.unique(true_severity).size < 2:
        kendall_tau = float("nan")
    else:
        tau, _ = stats.kendalltau(true_severity, est_severity)
        kendall_tau = float("nan") if (tau is None or np.isnan(tau)) else float(tau)

    return {
        "global_cost": float(global_cost),
        "constraint_rate": float(constraint_rate),
        "utilization": float(utilization),
        "health_mae": health_mae,
        "kendall_tau": float(kendall_tau),
    }
