"""Smoke + property tests for the closed-loop pipeline."""

from __future__ import annotations

import numpy as np

from dds_adapt.baselines import (
    BaselineByzantineResilient,
    BaselineDSFusion,
    BaselineFDIReconf,
    BaselineOracle,
    BaselineRobustDO,
)
from dds_adapt.config import Config
from dds_adapt.cost import formation_cost, formation_gradient
from dds_adapt.diagnostic import RPSDiagnosticModule
from dds_adapt.engine import run_closed_loop
from dds_adapt.metrics import compute_metrics
from dds_adapt.optimizer import DIGingOptimizer
from dds_adapt.residual import (
    broadcast_residual_matrix,
    collect_residual_samples,
    residual_energy,
    train_gdm,
)
from dds_adapt.simulator import SatelliteFormationSimulator
from dds_adapt.utils import adapt_mixing_matrix, sinkhorn_double_stochastic


# ----------------------------------------------------- topology + utils
def test_communication_graph_doubly_stochastic():
    W = Config.get_communication_graph(8)
    np.testing.assert_allclose(W.sum(axis=0), 1.0, atol=1e-10)
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-10)


def test_algebraic_connectivity_positive():
    W = Config.get_communication_graph(8)
    assert Config.algebraic_connectivity(W) > 0


def test_sinkhorn_yields_doubly_stochastic():
    rng = np.random.default_rng(0)
    M = rng.uniform(0, 1, (6, 6))
    np.fill_diagonal(M, M.diagonal() + 0.5)  # make diagonal heavy
    M = sinkhorn_double_stochastic(M)
    np.testing.assert_allclose(M.sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(M.sum(axis=1), 1.0, atol=1e-6)


def test_adapt_mixing_matrix_attenuates_degraded_rows():
    n = 5
    W = Config.get_communication_graph(n)
    h = np.array([1.0, 1.0, 0.2, 1.0, 1.0])
    W_tilde = adapt_mixing_matrix(W, h)
    np.testing.assert_allclose(W_tilde.sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(W_tilde.sum(axis=1), 1.0, atol=1e-6)
    # the degraded row should be dominated by its self loop
    assert W_tilde[2, 2] > 0.5


# ----------------------------------------------------- simulator
def test_simulator_residual_grows_with_degradation():
    sim = SatelliteFormationSimulator(n_satellites=4, seed=0)
    sim.set_health(np.array([1.0, 1.0, 1.0, 1.0]))
    e_h = residual_energy(sim.sample_residual()).mean()

    sim = SatelliteFormationSimulator(n_satellites=4, seed=0)
    sim.set_health(np.array([0.2, 1.0, 1.0, 1.0]))
    e_f = residual_energy(sim.sample_residual())
    # the degraded agent must show a bigger residual energy than the others
    assert e_f[0] > e_f[1:].mean()
    # and bigger than its own healthy value
    assert e_f[0] > e_h


# ----------------------------------------------------- diagnostic
def test_gdm_training_and_health_estimation():
    n = 4
    healthy, faulty = train_gdm(n_agents=n, n_samples=200, seed=0)
    diag = RPSDiagnosticModule(n_agents=n)
    diag.fit(healthy, faulty)

    rng = np.random.default_rng(0)
    own_energy = np.array([0.012, 0.012, 0.012, 0.012])  # everybody healthy
    R = broadcast_residual_matrix(own_energy, rng)
    h_hat, _, _ = diag.diagnose_round(R)
    assert (h_hat > 0.5).all()

    # now agent 0 is degraded -> its residual energy is much higher
    own_energy = np.array([0.07, 0.012, 0.012, 0.012])
    R = broadcast_residual_matrix(own_energy, rng)
    h_hat_f, _, _ = diag.diagnose_round(R)
    assert h_hat_f[0] < h_hat_f[1:].mean()


def test_diagnostic_pmf_normalised():
    n = 4
    healthy, faulty = train_gdm(n_agents=n, n_samples=100, seed=0)
    diag = RPSDiagnosticModule(n_agents=n)
    diag.fit(healthy, faulty)
    rng = np.random.default_rng(0)
    R = broadcast_residual_matrix(np.array([0.012, 0.012, 0.012, 0.012]), rng)
    pmf = diag.generate_local_pmf(0, R[0])
    s = sum(pmf.values())
    assert s > 0
    assert abs(s - 1.0) < 1e-6


# ----------------------------------------------------- engine + closed loop
def test_engine_closed_loop_runs_and_logs():
    n = 4
    healthy, faulty = train_gdm(n_agents=n, n_samples=200, seed=0)
    diag = RPSDiagnosticModule(n_agents=n)
    diag.fit(healthy, faulty)

    W = Config.get_communication_graph(n)
    desired = np.stack(
        [
            np.array([np.cos(2 * np.pi * i / n), np.sin(2 * np.pi * i / n)])
            for i in range(n)
        ]
    )
    profile = np.ones((n, 200))
    profile[0, :] = np.linspace(1.0, 0.3, 200)  # smooth degradation

    log = run_closed_loop(
        health_profile=profile,
        diagnostic=diag,
        W_base=W,
        desired_positions=desired,
        n_diag_intervals=8,
        iters_per_diag=10,
        seed=0,
    )
    assert len(log.health_est) == 8
    assert len(log.positions) == 8
    # Average over the second half of the run (post-degradation): the
    # degraded agent's estimated health should be lower than the others.
    second_half = np.stack(log.health_est[-4:]).mean(axis=0)
    assert second_half[0] < second_half[1:].mean()


def test_engine_w_adaptation_keeps_w_doubly_stochastic():
    """Sanity: adapt_mixing_matrix should always be DS, even mid-engine."""
    n = 4
    W = Config.get_communication_graph(n)
    h = np.array([0.3, 1.0, 1.0, 0.5])
    W_tilde = adapt_mixing_matrix(W, h)
    np.testing.assert_allclose(W_tilde.sum(axis=0), 1.0, atol=1e-6)
    np.testing.assert_allclose(W_tilde.sum(axis=1), 1.0, atol=1e-6)


# ----------------------------------------------------- baselines
def test_ds_fusion_dempster_combination():
    n = 4
    ds = BaselineDSFusion(n)
    # all observers agree agent 0 is faulty (low healthy belief)
    soft = np.tile(np.array([0.05, 0.95, 0.95, 0.95]), (n, 1))
    h = ds.fuse(soft)
    assert h[0] < 0.2
    assert h[1] > 0.8


def test_baselines_instantiate():
    n = 4
    W = Config.get_communication_graph(n)
    BaselineRobustDO(n, 2, W)
    BaselineFDIReconf(n, 2, W)
    BaselineByzantineResilient(n, 2, W)
    BaselineDSFusion(n)
    BaselineOracle(n, 2, W)


# ----------------------------------------------------- metrics
def test_utilization_higher_is_better():
    n = 4
    desired = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    def make_x(err: float) -> np.ndarray:
        x_full = np.zeros((n, 4))
        x_full[:, :2] = desired + err
        return x_full

    health = np.array([0.5, 0.5, 1.0, 1.0])
    near = compute_metrics(make_x(0.1), health, health, desired)
    far = compute_metrics(make_x(0.5), health, health, desired)
    assert near["utilization"] > far["utilization"]


def test_metrics_keys_present():
    np.random.seed(0)
    n = 4
    desired = np.random.randn(n, 2)
    x = desired + np.random.randn(n, 2) * 0.05
    x_full = np.zeros((n, 4))
    x_full[:, :2] = x
    metrics = compute_metrics(x_full, np.ones(n), np.ones(n), desired)
    assert {
        "global_cost",
        "constraint_rate",
        "utilization",
        "health_mae",
        "kendall_tau",
    } <= set(metrics.keys())


def test_cost_and_gradient_finite():
    np.random.seed(0)
    n = 4
    desired = np.random.randn(n, 2)
    x = np.random.randn(4)
    c = formation_cost(x, 0, 0.7, desired)
    g = formation_gradient(x, 0, 0.7, desired)
    assert np.isfinite(c)
    assert np.all(np.isfinite(g))


# ----------------------------------------------------- optimizer
def test_optimizer_runs_one_step():
    np.random.seed(0)
    n = 4
    W = Config.get_communication_graph(n)
    opt = DIGingOptimizer(n, dim=2, W=W, alpha=0.01)
    desired = np.random.randn(n, 2)

    def grad(x_i, i, h_i):
        return formation_gradient(x_i, i, h_i, desired)

    x = opt.step(grad, np.ones(n))
    assert x.shape == (n, 2)


def test_optimizer_set_mixing_matrix_updates_W():
    n = 4
    W1 = Config.get_communication_graph(n)
    opt = DIGingOptimizer(n, dim=2, W=W1, alpha=0.01)
    h = np.array([0.5, 1.0, 1.0, 1.0])
    W2 = adapt_mixing_matrix(W1, h)
    opt.set_mixing_matrix(W2)
    np.testing.assert_allclose(opt.W, W2)


# ----------------------------------------------------- residual collection
def test_collect_residual_samples_shape():
    R = collect_residual_samples(n_samples=20, n_agents=3, fault_index=None, seed=0)
    assert R.shape == (20, 3, 3)
    assert (R >= 0).all()



# ----------------------------------------------------- HF stand-in
def test_hf_simulator_step_finite():
    from dds_adapt.hf_simulator import NASA42StandInSimulator

    sim = NASA42StandInSimulator(n_satellites=3, dt=1.0, seed=0)
    cmd = sim.commanded_control(sim.desired_positions)
    state, realised, residual = sim.step(cmd)
    assert state.shape == (3, 6)
    assert np.all(np.isfinite(state))
    assert np.all(np.isfinite(residual))


def test_hf_runner_returns_finite_metrics():
    from dds_adapt.hf_runner import run_hf_diagnostic_experiment

    out = run_hf_diagnostic_experiment(n_steps=80, seed=0)
    assert np.isfinite(out.health_mae)
    assert -1.0 <= out.kendall_tau <= 1.0
    assert out.mean_estimate_per_agent.shape == (3,)


def test_rho_max_calibration_returns_numbers():
    from dds_adapt.rho_max_calibration import calibrate_rho_max
    from dds_adapt.runner import ExperimentRunner as _R

    runner = _R(n_satellites=4, seed=0)
    cal = calibrate_rho_max(
        runner,
        rho_factors=np.array([0.5, 1.0, 3.0]),
        seed=0,
        n_steps=80,
        n_repeats=1,
    )
    assert cal.rho_star_empirical > 0
    assert cal.rho_max_theoretical > 0
    assert np.isfinite(cal.constant_C)
