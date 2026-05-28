"""Boundary tests for the scenario builders in ``dds_adapt.scenarios``.

These functions are driven by the study runners (so the closed-loop
tests cover them indirectly), but the boundary behaviour at parameter
extremes is what fixes get rolled back over time. Pinning that here
catches silent regressions when someone adjusts a default.
"""

from __future__ import annotations

import numpy as np
import pytest

from dds_adapt.config import Config
from dds_adapt.scenarios import (
    actuator_degradation_profile,
    communication_loss_w_sequence,
    concurrent_degradation_profile,
    perturb_topology,
)


# ===================================================================
# actuator_degradation_profile (§5.4.1)
# ===================================================================
def test_actuator_profile_shape_and_dtype() -> None:
    h, onset = actuator_degradation_profile(
        n_agents=8, n_steps=200, eta=0.002, onset_time=50, agent_idx=0
    )
    assert h.shape == (8, 200)
    assert h.dtype == np.float64
    assert onset == 50


def test_actuator_profile_pre_onset_is_unity() -> None:
    """Health must be exactly 1.0 for every tick before onset."""
    h, onset = actuator_degradation_profile(
        n_agents=4, n_steps=300, eta=0.005, onset_time=120, agent_idx=2
    )
    assert np.all(h[:, :onset] == 1.0)


def test_actuator_profile_post_onset_decays_for_target_only() -> None:
    """Only the target agent's row decays after onset."""
    h, onset = actuator_degradation_profile(
        n_agents=5, n_steps=200, eta=0.01, onset_time=50, agent_idx=3
    )
    # target row: monotonically non-increasing after onset
    target = h[3, onset:]
    assert np.all(np.diff(target) <= 1e-12)
    assert target[-1] < 1.0
    # every other row stays at 1.0
    for i in range(5):
        if i == 3:
            continue
        assert np.all(h[i] == 1.0)


def test_actuator_profile_eta_zero_keeps_unity() -> None:
    """eta = 0 means no decay at all — the target row stays at 1.0."""
    h, _ = actuator_degradation_profile(
        n_agents=4, n_steps=100, eta=0.0, onset_time=10, agent_idx=0
    )
    np.testing.assert_allclose(h[0], 1.0)


def test_actuator_profile_random_onset_lies_in_window() -> None:
    """When onset_time is None, the RNG must draw from Config.ONSET_WINDOW."""
    rng = np.random.default_rng(42)
    lo, hi = Config.ONSET_WINDOW
    for _ in range(8):
        _, onset = actuator_degradation_profile(
            n_agents=4, n_steps=300, rng=rng
        )
        assert lo <= onset < hi


def test_actuator_profile_onset_at_or_past_n_steps_keeps_unity() -> None:
    """onset_time >= n_steps means no degradation gets written."""
    h, _ = actuator_degradation_profile(
        n_agents=4, n_steps=100, eta=0.01, onset_time=100, agent_idx=0
    )
    np.testing.assert_allclose(h, 1.0)


# ===================================================================
# concurrent_degradation_profile (§5.5.2)
# ===================================================================
def test_concurrent_profile_two_agents_decay_at_different_rates() -> None:
    """Fast agent must drop faster than the slow agent."""
    h, onset = concurrent_degradation_profile(
        n_agents=4, n_steps=300, eta_fast=0.01, eta_slow=0.001, onset_time=50
    )
    assert h.shape == (4, 300)
    assert onset == 50
    # at the final tick, fast row is below slow row
    assert h[0, -1] < h[1, -1]
    # untouched agents remain at 1.0
    assert np.all(h[2:] == 1.0)


def test_concurrent_profile_n_agents_below_two_is_a_noop() -> None:
    """With only 1 agent there is no fast/slow pair — function silently returns ones."""
    h, _ = concurrent_degradation_profile(n_agents=1, n_steps=50)
    np.testing.assert_allclose(h, 1.0)


# ===================================================================
# communication_loss_w_sequence (§5.4.2)
# ===================================================================
def test_communication_loss_returns_n_intervals_matrices() -> None:
    W = Config.get_communication_graph(8)
    seq = communication_loss_w_sequence(
        W, n_intervals=10, edges_to_drop=[(0, 1), (0, 2)]
    )
    assert len(seq) == 10
    for W_k in seq:
        assert W_k.shape == (8, 8)


def test_communication_loss_keeps_row_stochastic() -> None:
    """Every matrix in the sequence must row-sum to 1 (engine relies on it)."""
    W = Config.get_communication_graph(6)
    seq = communication_loss_w_sequence(
        W, n_intervals=5, edges_to_drop=[(0, 1), (1, 2)]
    )
    for W_k in seq:
        np.testing.assert_allclose(W_k.sum(axis=1), 1.0, atol=1e-10)


def test_communication_loss_attenuates_named_edges() -> None:
    """The named edges should drop below their base weight in at least one interval."""
    W = Config.get_communication_graph(6)
    e = (0, 1)
    seq = communication_loss_w_sequence(
        W, n_intervals=8, edges_to_drop=[e],
        loss_min=0.1, loss_max=0.9, period=4,
    )
    # at the sinusoidal trough the edge weight (post row-renorm) must be
    # strictly smaller than the base weight
    edge_weights = np.array([W_k[e] for W_k in seq])
    assert edge_weights.min() < W[e]


# ===================================================================
# perturb_topology (§5.5.1)
# ===================================================================
def test_perturb_topology_random_mode_doubly_stochastic_after_renorm() -> None:
    W = Config.get_communication_graph(8)
    seq = perturb_topology(
        W, n_intervals=10, mode="random", n_removals=4,
        rng=np.random.default_rng(0),
    )
    assert len(seq) == 10
    for W_k in seq:
        np.testing.assert_allclose(W_k.sum(axis=1), 1.0, atol=1e-10)


def test_perturb_topology_high_weight_is_deterministic() -> None:
    """high_weight mode does not consult the RNG, so two calls match exactly."""
    W = Config.get_communication_graph(8)
    a = perturb_topology(W, n_intervals=5, mode="high_weight", n_removals=3)
    b = perturb_topology(W, n_intervals=5, mode="high_weight", n_removals=3)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)


def test_perturb_topology_adjacent_targets_only_degraded_neighbours() -> None:
    """The 'adjacent' mode must only remove edges incident to degraded_agent."""
    W = Config.get_communication_graph(8)
    seq = perturb_topology(
        W, n_intervals=4, mode="adjacent", n_removals=4,
        degraded_agent=3, rng=np.random.default_rng(0),
    )
    final = seq[-1]
    # Every edge that was removed (W_base > 0 and final == 0) must have
    # 3 as one endpoint.
    for i in range(8):
        for j in range(i + 1, 8):
            if W[i, j] > 0 and final[i, j] == 0.0 and i != j:
                assert 3 in (i, j)


def test_perturb_topology_n_removals_capped_by_candidate_count() -> None:
    """Asking for more removals than candidates exist should not raise."""
    W = Config.get_communication_graph(8)
    # adjacent mode on agent 3 has at most ~6 candidates; ask for 999
    seq = perturb_topology(
        W, n_intervals=3, mode="adjacent", n_removals=999,
        degraded_agent=3, rng=np.random.default_rng(0),
    )
    # function must complete and produce row-stochastic matrices
    for W_k in seq:
        np.testing.assert_allclose(W_k.sum(axis=1), 1.0, atol=1e-10)


def test_perturb_topology_n_removals_zero_is_a_noop() -> None:
    """n_removals = 0 means every interval matrix equals the base graph."""
    W = Config.get_communication_graph(6)
    seq = perturb_topology(
        W, n_intervals=4, mode="random", n_removals=0,
        rng=np.random.default_rng(0),
    )
    for W_k in seq:
        np.testing.assert_allclose(W_k, W, atol=1e-10)


def test_perturb_topology_unknown_mode_raises() -> None:
    """Unknown mode strings must raise ValueError, not produce silent output."""
    W = Config.get_communication_graph(4)
    with pytest.raises(ValueError, match="unknown mode"):
        perturb_topology(W, n_intervals=2, mode="bogus", n_removals=1)
