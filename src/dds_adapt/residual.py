"""Residual generation and GDM training utilities.

The diagnostic module operates on a residual *matrix* ``r[i, j]`` whose
entry is the residual energy that agent ``i`` attributes to agent ``j``.
Building such a matrix from the simulator requires two pieces:

1. a per-agent residual signal produced by the physical simulator
   (commanded vs. realised acceleration, plus sensor noise);
2. a way for agent ``i`` to estimate every other agent ``j``'s residual
   from its local view. We assume agents broadcast their own residual
   energy to neighbours and corrupt it by an observer-specific noise
   term, which is the simplest model that still yields a non-trivial
   inter-agent diagnostic structure.

The same generator is used both at training time (collecting healthy /
faulty traces for the GDM) and at run time (feeding the live RPS
pipeline).
"""

from __future__ import annotations

import numpy as np

from dds_adapt.config import Config
from dds_adapt.simulator import SatelliteFormationSimulator


def residual_energy(residual_2d: np.ndarray) -> np.ndarray:
    """Reduce a 2-d acceleration residual to a non-negative scalar energy."""
    return np.linalg.norm(residual_2d, axis=1)


def broadcast_residual_matrix(
    own_energy: np.ndarray,
    rng: np.random.Generator,
    cross_agent_noise_std: float = 0.005,
) -> np.ndarray:
    """Form ``r[i, j]`` from each agent's broadcast of its own energy.

    Self-observations (``i == j``) carry only the local sensor noise that
    the simulator already injected. Cross observations add a small
    transmission/observer-specific noise to model the fact that agent
    ``i`` sees agent ``j`` through a noisy channel. The noise is small
    enough (default 0.005) that it does not swamp the GDM's between-class
    separation, which is on the order of 0.05 between healthy and faulty
    residual energies.
    """
    n = len(own_energy)
    R = np.broadcast_to(own_energy, (n, n)).copy().astype(float)
    cross = rng.normal(0, cross_agent_noise_std, (n, n))
    np.fill_diagonal(cross, 0.0)
    return np.maximum(R + cross, 0.0)


def collect_residual_samples(
    n_samples: int,
    n_agents: int,
    fault_index: int | None = None,
    fault_health: float = 0.3,
    cross_agent_noise_std: float = 0.005,
    seed: int | None = None,
) -> np.ndarray:
    """Sample residuals at the diagnostic probe state.

    For training the GDM we want residuals that reflect health alone.
    The simulator exposes ``sample_residual``, which evaluates the
    controller at a fixed perturbation from the reference and returns
    the resulting commanded-vs-realised residual without propagating
    state. We re-use that here so training and runtime see the same
    probe geometry.
    """
    sim = SatelliteFormationSimulator(n_satellites=n_agents, seed=seed)
    rng = np.random.default_rng(None if seed is None else seed + 17)
    out = np.empty((n_samples, n_agents, n_agents))

    for t in range(n_samples):
        if fault_index is None:
            health = np.ones(n_agents)
        else:
            health = np.ones(n_agents)
            health[fault_index] = float(fault_health)
        sim.set_health(health)
        residual = sim.sample_residual()
        own_energy = residual_energy(residual)
        R = broadcast_residual_matrix(
            own_energy, rng, cross_agent_noise_std=cross_agent_noise_std
        )
        out[t] = R
    return out


def train_gdm(
    n_agents: int,
    n_samples: int = Config.N_TRAIN_SAMPLES,
    fault_healths: tuple[float, ...] = (0.5, 0.3, 0.1),
    cross_agent_noise_std: float = 0.005,
    seed: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build training residuals for ``RPSDiagnosticModule.fit``.

    The faulty block for agent ``j`` is collected over a *range* of
    fault-health values rather than a single hard-fault corner, so the
    GDM learns a faulty distribution that covers partial degradation.
    Section 4.2 of the paper means by "GDM parameters are estimated
    from historical data" precisely this: the historical envelope, not
    a single operating point.

    The default range starts at ``0.5`` (a 50% capability loss) rather
    than ``0.7``: the latter sits close enough to the healthy
    distribution that the resulting GDM produces healthy-time
    posterior P(faulty|residual) of ~0.15 even on uncorrupted residuals,
    which feeds spurious "agent slightly degraded" signals into the
    closed loop and biases X* away from the true optimum during the
    fault-free phase.
    """
    if not fault_healths:
        raise ValueError("fault_healths must contain at least one value")
    per_health = max(1, n_samples // len(fault_healths))

    healthy = collect_residual_samples(
        n_samples,
        n_agents,
        fault_index=None,
        cross_agent_noise_std=cross_agent_noise_std,
        seed=seed,
    ).mean(axis=1)
    faulty: list[np.ndarray] = []
    for j in range(n_agents):
        chunks: list[np.ndarray] = []
        for h_idx, fh in enumerate(fault_healths):
            block = collect_residual_samples(
                per_health,
                n_agents,
                fault_index=j,
                fault_health=fh,
                cross_agent_noise_std=cross_agent_noise_std,
                seed=seed + 1 + j + h_idx * 31,
            ).mean(axis=1)
            chunks.append(block)
        faulty.append(np.concatenate(chunks, axis=0))
    return healthy, faulty
