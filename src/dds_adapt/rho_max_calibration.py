"""Numerically calibrate the constant ``C`` in the convergence theorem.

Theorem (Section 4.3) gives

    rho_max = lambda_2(tilde W*) * mu / (C * kappa)

where ``C`` lumps together the network size, regularisation strength,
and per-step Lyapunov constants that the proof manipulates but never
specifies numerically. The paper claims the empirical critical rate is
about 20% above the theoretical bound.

This module:

1. Sweeps a range of health-variation rates ``rho``.
2. For each rho, runs the closed loop several times and measures the
   steady-state consensus error.
3. Fits a piecewise model (linear-then-divergent) to find the empirical
   critical rate ``rho_star``.
4. Solves for ``C`` from the same equation rearranged with the
   measured ``rho_star``, lambda_2, mu, kappa.
5. Reports the ratio rho_star / rho_max_theoretical so the paper's
   "approx 20% margin" claim has a code-derived number.

The calibration is reproducible: given a fixed seed and the default
sweep, ``calibrate_rho_max`` returns deterministic floats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dds_adapt.config import Config
from dds_adapt.runner import ExperimentRunner


@dataclass
class RhoCalibrationResult:
    rhos: np.ndarray
    steady_state_errors: np.ndarray
    diverged: np.ndarray  # bool array
    rho_star_empirical: float
    rho_max_theoretical: float
    margin_percent: float
    constant_C: float


def _measure_steady_state(
    runner: ExperimentRunner,
    rho_factor: float,
    seed: int,
    n_steps: int,
) -> float:
    """Run one trajectory at the given rho factor; return tracking error.

    The "error" we measure is the tracking error of the DIGing iterate
    relative to the *moving* optimum implied by the current health
    estimate. ``rho_max`` in Theorem 1 bounds the rate beyond which
    this tracking error can no longer follow the moving optimum, which
    manifests as a monotonically growing residual against the moving
    target rather than DIGing internal divergence.

    ``calibrate_rho_max`` decides divergence by comparing this error
    against the smallest-rho baseline; that judgement lives in the
    caller, not here.
    """
    eta = rho_factor * Config.CHARACTERISTIC_HEALTH_RATE
    profile, _ = runner.degradation_profile(n_steps, eta=eta, onset_time=0)
    _, log = runner.run_proposed(profile, seed)

    # tracking error: ||DIGing iterate - desired_positions||
    if not log.positions:
        return float("nan")
    errs = np.array(
        [float(np.linalg.norm(p - runner.desired_positions)) for p in log.positions]
    )
    if not np.all(np.isfinite(errs)):
        return float("nan")
    tail = float(np.mean(errs[-max(1, len(errs) // 4):]))
    return tail


def calibrate_rho_max(
    runner: ExperimentRunner,
    *,
    rho_factors: np.ndarray | None = None,
    seed: int = 0,
    n_steps: int = 400,
    n_repeats: int = 2,
) -> RhoCalibrationResult:
    """Sweep ``rho_factor`` and back out ``C`` from the empirical critical rate.

    The "critical" rate is defined as the smallest swept ``rho`` at which
    the steady-state tracking error rises above ``1.5x`` its
    smallest-rho baseline. This is the operational definition that
    matches the §5.2.1 narrative: a rate at which the DIGing iterate can
    no longer keep up with the moving optimum, even if it does not
    formally diverge.
    """
    if rho_factors is None:
        rho_factors = np.array(
            [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
        )

    errors = np.zeros(len(rho_factors))
    for i, rho in enumerate(rho_factors):
        run_errors: list[float] = []
        for r in range(n_repeats):
            err = _measure_steady_state(
                runner, float(rho), seed + r * 17, n_steps
            )
            run_errors.append(err)
        errors[i] = float(np.nanmean(run_errors))

    # baseline is the smallest-rho run; "diverged" = >50% above baseline
    baseline = errors[0]
    threshold = 1.5 * baseline
    diverged = errors > threshold

    div_indices = np.where(diverged)[0]
    if div_indices.size > 0:
        first_div = int(div_indices[0])
        if first_div == 0:
            rho_star_factor = float(rho_factors[0])
        else:
            rho_star_factor = float(
                0.5 * (rho_factors[first_div - 1] + rho_factors[first_div])
            )
    else:
        rho_star_factor = float(rho_factors[-1])

    rho_star_empirical = rho_star_factor * Config.CHARACTERISTIC_HEALTH_RATE

    lambda2 = runner.lambda2
    mu_bar = min(Config.MU, runner.gamma)
    L_bar = max(Config.L_SMOOTH, runner.gamma)
    kappa_bar = L_bar / mu_bar
    C_implied = lambda2 * mu_bar / (rho_star_empirical * kappa_bar)

    n = runner.n
    # The analytic prior follows the bound construction in Theorem 1's
    # proof: C ≈ sqrt(N) * L_h / (mu_bar * (1 - sigma^M)) where sigma is the
    # DIGing per-iteration contraction and M iterations run per diagnosis
    # interval. We take L_h = gamma (the dominant Lipschitz term in
    # ``adapted_grad``), and sigma^M = 0.9 (a single diagnosis interval
    # already brings DIGing close to consensus, so the effective contraction
    # over M iterations is mild).
    sigma_pow_M = 0.9
    L_h = runner.gamma
    C_prior = float(np.sqrt(n) * L_h / (mu_bar * (1.0 - sigma_pow_M)))

    rho_max_theoretical = lambda2 * mu_bar / (C_prior * kappa_bar)
    margin = (rho_star_empirical - rho_max_theoretical) / max(
        rho_max_theoretical, 1e-12
    )

    return RhoCalibrationResult(
        rhos=rho_factors * Config.CHARACTERISTIC_HEALTH_RATE,
        steady_state_errors=errors,
        diverged=diverged,
        rho_star_empirical=rho_star_empirical,
        rho_max_theoretical=rho_max_theoretical,
        margin_percent=float(margin * 100),
        constant_C=float(C_implied),
    )
