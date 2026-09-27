"""High-fidelity track driver (Section 5.1.1, eight-satellite GTO).

The high-fidelity track exists to *validate* that the diagnostic
module's behaviour survives realistic disturbances (J2, solar pressure,
gravity gradient, residual drag) rather than to re-run the full
comparative table at low fidelity. The paper uses it for the same
purpose: preliminary studies with a smaller formation under more
representative dynamics.

This driver:

1. Trains a GDM on residuals collected from the high-fidelity
   simulator (`hf_simulator.NASA42StandInSimulator`).
2. Runs a degradation profile through the same simulator and feeds the
   per-tick residual to the RPS diagnostic.
3. Reports the diagnostic-layer metrics (health MAE, Kendall τ,
   detection delay) – the same three numbers that justify the value
   of RPS in the paper.

The DIGing optimisation loop is intentionally *not* part of this
driver: optimising over a 3-D inertial state requires the Hill-Clohessy-
Wiltshire frame (or similar) which is out of scope for the companion
code. What we do test here is whether the diagnostic component
continues to function under realistic dynamics, which is the open
question §5.1.1 actually asks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ddsa.config import Config
from ddsa.diagnostic import RPSDiagnosticModule
from ddsa.hf_simulator import NASA42StandInSimulator
from ddsa.residual import broadcast_residual_matrix


def _energy_3d(residual: np.ndarray) -> np.ndarray:
    """Reduce a 3-D acceleration residual to a non-negative scalar energy."""
    return np.linalg.norm(residual, axis=1)


def _collect_hf_samples(
    sim: NASA42StandInSimulator,
    n_samples: int,
    fault_index: int | None,
    fault_health: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gather residual rows under a fixed fault hypothesis (no propagation)."""
    n = sim.n
    out = np.empty((n_samples, n))
    for t in range(n_samples):
        if fault_index is None:
            health = np.ones(n)
        else:
            health = np.ones(n)
            health[fault_index] = float(fault_health)
        sim.set_health(health)
        residual = sim.sample_residual()
        own_energy = _energy_3d(residual)
        R = broadcast_residual_matrix(own_energy, rng, cross_agent_noise_std=1e-5)
        out[t] = R.mean(axis=0)
    return out


def train_gdm_hf(
    n_agents: int = Config.NUM_SATELLITES_HF,
    n_samples: int = Config.N_TRAIN_SAMPLES,
    fault_healths: tuple[float, ...] = (0.7, 0.5, 0.3, 0.1),
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build a healthy / per-fault training set using the HF stand-in."""
    sim = NASA42StandInSimulator(n_satellites=n_agents, seed=seed)
    rng = np.random.default_rng(seed + 17)

    healthy = _collect_hf_samples(sim, n_samples, None, 1.0, rng)

    per_health = max(1, n_samples // len(fault_healths))
    faulty: list[np.ndarray] = []
    for j in range(n_agents):
        chunks: list[np.ndarray] = []
        for h_idx, fh in enumerate(fault_healths):
            sim_j = NASA42StandInSimulator(n_satellites=n_agents, seed=seed + j + h_idx)
            chunks.append(
                _collect_hf_samples(
                    sim_j, per_health, j, fh,
                    np.random.default_rng(seed + 100 + j + h_idx)
                )
            )
        faulty.append(np.concatenate(chunks, axis=0))
    return healthy, faulty


@dataclass
class HFResult:
    health_mae: float
    kendall_tau: float
    detection_delay: float
    mean_estimate_per_agent: np.ndarray
    true_per_agent: np.ndarray


def run_hf_diagnostic_experiment(
    n_agents: int = Config.NUM_SATELLITES_HF,
    n_steps: int = 600,
    eta: float = Config.ETA_SINGLE,
    onset_step: int = 50,
    seed: int = 0,
) -> HFResult:
    """Trace the diagnostic estimate through one high-fidelity degradation run.

    The relative detection rule below assumes ``onset_step`` is large
    enough that the first diagnosis tick falls inside the healthy phase
    (specifically: ``onset_step >= Config.DELTA_T_HF`` so the t=0 tick
    is unambiguously fault-free). Callers that want an onset before
    the first diagnosis tick should pass an explicit healthy baseline
    instead of relying on the t=0 estimate.
    """
    healthy, faulty = train_gdm_hf(n_agents=n_agents, n_samples=200, seed=seed)
    diag = RPSDiagnosticModule(n_agents=n_agents)
    diag.fit(healthy, faulty)

    sim = NASA42StandInSimulator(n_satellites=n_agents, seed=seed)
    rng = np.random.default_rng(seed + 41)

    health_profile = np.ones((n_agents, n_steps))
    t = np.arange(n_steps)
    health_profile[0, onset_step:] = np.exp(
        -eta * (t[onset_step:] - onset_step)
    )

    # diagnose every Config.DELTA_T_HF steps
    diag_step = Config.DELTA_T_HF
    h_hat_log: list[np.ndarray] = []
    h_true_log: list[np.ndarray] = []
    detection_tick: int | None = None
    # Use a *relative* detection rule: trigger when any agent's h_hat
    # drops below ``baseline_h_hat - 0.2``, where the baseline is the
    # estimate at t=0 (no fault yet). The OPT distribution is not
    # uniform-1 even on a fully healthy formation, so an absolute
    # threshold like ``h_hat < 0.9`` would fire on every run from
    # tick 0 -- giving a meaningless negative detection delay.
    baseline_h_hat: np.ndarray | None = None
    for tick in range(0, n_steps, diag_step):
        sim.set_health(health_profile[:, tick])
        residual = sim.sample_residual()
        own_energy = _energy_3d(residual)
        R = broadcast_residual_matrix(own_energy, rng, cross_agent_noise_std=1e-5)
        h_hat, _, _ = diag.diagnose_round(R)
        h_hat_log.append(h_hat)
        h_true_log.append(health_profile[:, tick].copy())
        if baseline_h_hat is None:
            baseline_h_hat = h_hat.copy()
        if (
            detection_tick is None
            and ((h_hat - baseline_h_hat) < -0.2).any()
        ):
            detection_tick = tick

    h_hat_arr = np.stack(h_hat_log)
    h_true_arr = np.stack(h_true_log)

    mae = float(np.mean(np.abs(h_hat_arr - h_true_arr)))

    # Kendall tau between true and estimated severity vectors at the
    # last diagnosis tick. Mirrors the metric used by the numerical
    # track in metrics.compute_metrics.
    true_sev = 1.0 - h_true_arr[-1]
    est_sev = 1.0 - h_hat_arr[-1]
    if np.unique(true_sev).size < 2 and np.unique(est_sev).size < 2:
        tau = 1.0 if np.allclose(true_sev, est_sev) else 0.0
    else:
        tau_val, _ = stats.kendalltau(true_sev, est_sev)
        tau = 0.0 if (tau_val is None or np.isnan(tau_val)) else float(tau_val)

    if detection_tick is not None:
        true_below = np.where(np.any(health_profile < 0.9, axis=0))[0]
        det_delay = (
            float(detection_tick - true_below[0]) if true_below.size else 0.0
        )
    else:
        det_delay = -1.0

    return HFResult(
        health_mae=mae,
        kendall_tau=float(tau),
        detection_delay=det_delay,
        mean_estimate_per_agent=h_hat_arr.mean(axis=0),
        true_per_agent=h_true_arr.mean(axis=0),
    )
