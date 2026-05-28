"""Numerical regression tests.

After the §5.4.1 alignment pass that taught DIGing the proper
``(N, N, dim)`` formation state, every method-ordering claim the
paper actually argues for can be tested directly against the run
output. Absolute utilisation numbers are documented in
``KNOWN_DISCREPANCIES.md``; the tests below pin the *qualitative*
claims that the paper's narrative depends on.

The fixture is a paper-spec ``n_runs=2, n_steps=500`` run sized for
~50 seconds on a laptop. Anything stronger than ordering inequalities
here would have to be relaxed to absorb the small-fixture noise; we
avoid that trap by keeping the assertions ordinal.
"""

from __future__ import annotations

import numpy as np
import pytest

from dds_adapt.config import Config
from dds_adapt.runner import ExperimentRunner


@pytest.fixture(scope="module")
def comparative_results() -> dict:
    """Tiny fixed-seed comparative run shared by every test.

    n_steps=500 matches the paper's Section 5.1.4 protocol; running
    fewer would compress the cost band too much for the utilisation
    metric to be informative. n_runs is kept at 2 so the whole
    fixture finishes in well under a minute.
    """
    np.random.seed(0)  # belt-and-braces RNG anchor
    runner = ExperimentRunner(n_satellites=Config.NUM_SATELLITES, seed=0)
    return runner.run_comparative(n_runs=2, n_steps=500)


@pytest.fixture(scope="module")
def ablation_results() -> dict:
    np.random.seed(0)
    runner = ExperimentRunner(n_satellites=Config.NUM_SATELLITES, seed=0)
    return runner.run_ablation(n_runs=2, n_steps=500)


# --------------------------------------------------- §5.4.1 ordering claims
def test_oracle_dominates_all(comparative_results):
    """Oracle is the upper bound by construction."""
    oracle = comparative_results["Oracle"]["utilization"][0]
    for name in ("Proposed", "Robust DO", "FDI-Reconf",
                 "Byzantine-Resilient", "D-S Fusion"):
        u = comparative_results[name]["utilization"][0]
        assert oracle >= u - 1e-9, f"Oracle {oracle:.3f} < {name} {u:.3f}"


def test_proposed_dominates_non_oracle(comparative_results):
    """§5.4.1 central claim: continuous structural adaptation beats
    every binary / fixed-margin baseline."""
    p = comparative_results["Proposed"]["utilization"][0]
    for name in ("Robust DO", "FDI-Reconf",
                 "Byzantine-Resilient", "D-S Fusion"):
        u = comparative_results[name]["utilization"][0]
        assert p > u, (
            f"Proposed {p:.3f} should outperform {name} {u:.3f}"
        )


def test_d_s_detection_lags_fdi(comparative_results):
    """§5.4.1: D-S commits to isolation only after several intervals
    below threshold, so its detection delay is strictly larger than
    FDI's whenever both methods detect at all."""
    fdi = comparative_results["FDI-Reconf"]["detection_delay"][0]
    ds = comparative_results["D-S Fusion"]["detection_delay"][0]
    if fdi >= 0 and ds >= 0:
        assert ds > fdi, f"D-S delay {ds:.0f} should exceed FDI {fdi:.0f}"


def test_baselines_report_nan_for_health_metrics(comparative_results):
    """FDI / Robust DO / Byzantine / D-S don't estimate continuous
    health; their MAE / τ should render as NaN, not a misleading 0."""
    for name in ("Robust DO", "FDI-Reconf",
                 "Byzantine-Resilient", "D-S Fusion"):
        mae = comparative_results[name]["health_mae"][0]
        tau = comparative_results[name]["kendall_tau"][0]
        assert np.isnan(mae), f"{name} MAE should be NaN, got {mae}"
        assert np.isnan(tau), f"{name} τ should be NaN, got {tau}"


def test_proposed_health_estimation_is_finite(comparative_results):
    """Proposed *does* estimate health, so MAE and τ must be finite."""
    p = comparative_results["Proposed"]
    assert np.isfinite(p["health_mae"][0])
    assert np.isfinite(p["kendall_tau"][0])
    assert p["kendall_tau"][0] > 0.0  # severity ordering signal exists


# --------------------------------------------------- §5.3 ablation claims
def test_full_dominates_ds_variant(ablation_results):
    """§5.3.1: removing priority ordering (Variant A, D-S in loop)
    reduces utilisation while leaving health-MAE statistically similar.
    The utilisation drop is the central claim of §5.3.1."""
    full = ablation_results["Full framework"]["utilization"][0]
    ds = ablation_results["Variant A (D-S)"]["utilization"][0]
    assert full > ds, f"Full {full:.3f} should beat Variant A {ds:.3f}"


def test_no_adaptation_collapses(ablation_results):
    """§5.3.2: removing structural adaptation altogether (Variant E)
    drops utilisation to ~zero, validating that adaptation is
    necessary, not decorative."""
    full = ablation_results["Full framework"]["utilization"][0]
    no_adapt = ablation_results["Variant E (no adapt)"]["utilization"][0]
    # Variant E should be essentially zero utilisation; require a clear
    # gap to Full so the test catches regressions where adaptation
    # silently turns off.
    assert no_adapt < 0.1, f"Variant E util {no_adapt:.3f} should be ~0"
    assert full > no_adapt + 0.1, (
        f"Full {full:.3f} should clearly beat Variant E {no_adapt:.3f}"
    )


def test_full_matches_no_sinkhorn_under_ideal_diagnosis(ablation_results):
    """§5.3.1: Variant C (no Sinkhorn double-stochastic projection)
    should be statistically indistinguishable from Full when the
    health estimate is well-conditioned, because Sinkhorn is a
    stability safeguard rather than a performance lever."""
    full = ablation_results["Full framework"]["utilization"][0]
    no_sk = ablation_results["Variant C (No Sinkhorn)"]["utilization"][0]
    assert abs(full - no_sk) < 0.05
