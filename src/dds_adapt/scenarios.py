"""Scenario builders for Section 5.4 and Section 5.5 of the paper.

Each builder returns whichever combination of (health_profile,
W_base_per_interval, base_graph_extra) the engine needs to reproduce
the corresponding scenario. Keeping the construction here lets
experiments and unit tests share a single source of truth.
"""

from __future__ import annotations

import numpy as np

from dds_adapt.config import Config


# --------------------------------------------------------- Scenario 1
def actuator_degradation_profile(
    n_agents: int,
    n_steps: int,
    *,
    eta: float = Config.ETA_SINGLE,
    onset_time: int | None = None,
    agent_idx: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Single agent decays exponentially from ``onset_time``.

    This is the Scenario 1 profile of §5.4.1 of the paper.
    """
    if rng is None:
        rng = np.random.default_rng()
    if onset_time is None:
        onset_time = int(rng.integers(*Config.ONSET_WINDOW))

    t = np.arange(n_steps)
    h = np.ones((n_agents, n_steps))
    h[agent_idx, onset_time:] = np.exp(-eta * (t[onset_time:] - onset_time))
    return h, onset_time


# --------------------------------------------------------- Scenario 2
def communication_loss_w_sequence(
    W_base: np.ndarray,
    n_intervals: int,
    *,
    edges_to_drop: list[tuple[int, int]],
    period: int = 4,
    loss_min: float = 0.1,
    loss_max: float = 0.6,
) -> list[np.ndarray]:
    """Build a list of mixing matrices with sinusoidal packet loss.

    For each interval ``k`` we attenuate the rows/cols of
    ``edges_to_drop`` by ``1 - loss(k)`` where ``loss(k)`` is a sinusoid
    bouncing between ``loss_min`` and ``loss_max`` with period
    ``period`` intervals. After the attenuation we row-renormalise so
    the matrix stays row-stochastic; Sinkhorn-Knopp inside the engine
    completes the column normalisation.

    This is the §5.4.2 communication-link degradation scenario.
    """
    seq: list[np.ndarray] = []
    for k in range(n_intervals):
        phase = 2 * np.pi * k / period
        loss = (loss_max + loss_min) / 2 + (loss_max - loss_min) / 2 * np.sin(phase)
        scale = 1.0 - loss
        W = W_base.copy()
        for i, j in edges_to_drop:
            W[i, j] *= scale
            W[j, i] *= scale
        # row-renormalise
        rs = W.sum(axis=1, keepdims=True)
        rs = np.where(rs > 0, rs, 1.0)
        W = W / rs
        seq.append(W)
    # ensure type stability if the caller iterates
    return seq


# --------------------------------------------------------- §5.5.1
def perturb_topology(
    W_base: np.ndarray,
    n_intervals: int,
    *,
    mode: str = "random",
    n_removals: int,
    degraded_agent: int = 0,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Progressive edge removal across diagnostic intervals.

    ``mode`` is one of:

    * ``"random"`` – uniformly random edge removal (consults ``rng``).
    * ``"high_weight"`` – remove the highest-weight edges first.
      Deterministic by construction; ``rng`` is unused for this mode.
    * ``"adjacent"`` – remove edges incident to ``degraded_agent``.
      Consults ``rng`` to break ties when multiple incident edges have
      equal weight.

    ``n_removals`` edges are removed by the *last* interval, and the
    sequence linearly interpolates the cumulative count between
    intervals.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = W_base.shape[0]
    # candidate ordered list
    candidates: list[tuple[int, int]] = []
    if mode == "random":
        for i in range(n):
            for j in range(i + 1, n):
                if W_base[i, j] > 0:
                    candidates.append((i, j))
        rng.shuffle(candidates)
    elif mode == "high_weight":
        rows = []
        for i in range(n):
            for j in range(i + 1, n):
                if W_base[i, j] > 0:
                    rows.append((W_base[i, j], i, j))
        rows.sort(reverse=True)
        candidates = [(i, j) for _, i, j in rows]
    elif mode == "adjacent":
        for j in range(n):
            if j != degraded_agent and W_base[degraded_agent, j] > 0:
                candidates.append(
                    (min(degraded_agent, j), max(degraded_agent, j))
                )
        rng.shuffle(candidates)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # build cumulative removal schedule
    removals = candidates[: n_removals]
    seq: list[np.ndarray] = []
    for k in range(n_intervals):
        # progressive: remove an extra edge every n_intervals/n_removals steps
        active_count = int(round((k + 1) / n_intervals * len(removals)))
        active_count = min(active_count, len(removals))
        W = W_base.copy()
        for idx in range(active_count):
            i, j = removals[idx]
            W[i, j] = 0.0
            W[j, i] = 0.0
        rs = W.sum(axis=1, keepdims=True)
        rs = np.where(rs > 0, rs, 1.0)
        W = W / rs
        seq.append(W)
    return seq


# --------------------------------------------------------- §5.5.2
def concurrent_degradation_profile(
    n_agents: int,
    n_steps: int,
    *,
    eta_fast: float = Config.ETA_FAST,
    eta_slow: float = Config.ETA_SLOW,
    onset_time: int = 80,
    fast_agent: int = 0,
    slow_agent: int = 1,
) -> tuple[np.ndarray, int]:
    """Two agents degrade at different rates from the same onset."""
    t = np.arange(n_steps)
    h = np.ones((n_agents, n_steps))
    if n_agents >= 2:
        h[fast_agent, onset_time:] = np.exp(-eta_fast * (t[onset_time:] - onset_time))
        h[slow_agent, onset_time:] = np.exp(-eta_slow * (t[onset_time:] - onset_time))
    return h, onset_time
