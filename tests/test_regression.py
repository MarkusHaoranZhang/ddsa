"""Numerical regression tests.

After the RNG-isolation refactor (commit ``e3889e6`` -> next), every
metric except wall-clock time is bit-identical across runs and across
fixture-call ordering. The tests below take advantage of that:

* **Qualitative claims** the paper actually argues for (method ordering
  and ablation gaps). These are the assertions a future contributor
  most needs protected: if a refactor silently breaks them, the paper's
  narrative is no longer supported by the code.
* **Pinned numerical signatures** for every method on a small fixed
  configuration. Because the underlying simulation is now fully
  deterministic, these are exact equalities (rtol=1e-9) for everything
  except ``wall_time``, which is excluded by name.

The reference numbers are *not* the headline numbers in STATUS.md;
those are produced from n_runs=30 / n_steps=500 and are too expensive
for CI. The fixture here is sized for ~10 seconds on a laptop.
"""

from __future__ import annotations

import pytest

from dds_adapt.config import Config
from dds_adapt.runner import ExperimentRunner


@pytest.fixture(scope="module")
def comparative_results() -> dict:
    """Tiny fixed-seed comparative run shared by every regression test."""
    runner = ExperimentRunner(n_satellites=Config.NUM_SATELLITES, seed=0)
    return runner.run_comparative(n_runs=3, n_steps=200)


@pytest.fixture(scope="module")
def ablation_results() -> dict:
    runner = ExperimentRunner(n_satellites=Config.NUM_SATELLITES, seed=0)
    return runner.run_ablation(n_runs=3, n_steps=200)


# --------------------------------------------------- qualitative claims
def test_oracle_dominates_proposed(comparative_results):
    """Oracle is the upper bound on utilisation by construction."""
    oracle = comparative_results["Oracle"]["utilization"][0]
    proposed = comparative_results["Proposed"]["utilization"][0]
    assert oracle >= proposed - 1e-6


def test_oracle_health_mae_zero(comparative_results):
    """Oracle's health estimate is the ground truth."""
    assert comparative_results["Oracle"]["health_mae"][0] == pytest.approx(0.0, abs=1e-9)


def test_oracle_kendall_tau_one(comparative_results):
    """Oracle preserves the true severity ordering perfectly."""
    assert comparative_results["Oracle"]["kendall_tau"][0] == pytest.approx(1.0, abs=1e-6)


def test_proposed_outperforms_robust_do(comparative_results):
    """§5.4.1 central claim: adaptive estimation beats a fixed margin."""
    proposed_mae = comparative_results["Proposed"]["health_mae"][0]
    robust_mae = comparative_results["Robust DO"]["health_mae"][0]
    assert proposed_mae < robust_mae


def test_rps_preserves_severity_ordering(ablation_results):
    """§5.3.1 central claim: the RPS pipeline keeps Kendall τ above
    the D-S variant's τ on every fixture configuration."""
    full_tau = ablation_results["Full framework"]["kendall_tau"][0]
    ds_tau = ablation_results["Variant A (D-S)"]["kendall_tau"][0]
    assert full_tau >= ds_tau


def test_adaptation_provides_ordering_signal(ablation_results):
    """§5.3.2 in the diagnostic-layer form: removing adaptation kills
    the severity ordering signal. This is the small-fixture-stable
    way to express "adaptation is necessary"; the utilisation form
    of the same claim only holds at n_runs=10+ / n_steps=500+."""
    full_tau = ablation_results["Full framework"]["kendall_tau"][0]
    no_adapt_tau = ablation_results["Variant E (no adapt)"]["kendall_tau"][0]
    assert full_tau > no_adapt_tau + 0.3


# ------------------------------------------------- pinned numerical signature
# Mean of every metric across n_runs=3, n_steps=200, seed=0, recorded
# after the RNG-isolation refactor. ``wall_time`` is excluded because
# it is wall-clock and intentionally non-deterministic.
PINNED_COMPARATIVE: dict[str, dict[str, float]] = {
    "Proposed": {
        "global_cost":       17.203917971097642,
        "constraint_rate":    0.8809523809523809,
        "utilization":        0.008658184275408596,
        "health_mae":         0.10929738173527843,
        "kendall_tau":        0.5839205713659519,
        "detection_delay":    0.0,
        "comm_rounds":      500.0,
        "convergence_iters": 50.0,
    },
    "Robust DO": {
        "global_cost":       17.135785975842516,
        "constraint_rate":    0.8571428571428571,
        "utilization":        0.012313705715845722,
        "health_mae":         0.28429738173527846,
        "kendall_tau":        0.0,
        "detection_delay":    0.0,
        "comm_rounds":      148.33333333333334,
        "convergence_iters": 14.0,
    },
    "FDI-Reconf": {
        "global_cost":       17.051456915003925,
        "constraint_rate":    0.8571428571428571,
        "utilization":        0.6333333479525476,
        "health_mae":         0.015702618264721555,
        "kendall_tau":        0.0,
        "detection_delay":   -1.0,
        "comm_rounds":      144.0,
        "convergence_iters": 14.0,
    },
    "Byzantine-Resilient": {
        "global_cost":       17.05685772445903,
        "constraint_rate":    0.8571428571428571,
        "utilization":        0.7006799439206914,
        "health_mae":         0.015702618264721555,
        "kendall_tau":        0.0,
        "detection_delay":   -1.0,
        "comm_rounds":      500.0,
        "convergence_iters": 50.0,
    },
    "D-S Fusion": {
        "global_cost":       16.32485818460687,
        "constraint_rate":    0.8690476190476191,
        "utilization":        0.6666666812840173,
        "health_mae":         0.04226351435455337,
        "kendall_tau":        0.33333333333333326,
        "detection_delay":   15.0,
        "comm_rounds":      156.66666666666666,
        "convergence_iters": 15.333333333333334,
    },
    "Oracle": {
        "global_cost":       16.941993668991675,
        "constraint_rate":    0.8571428571428571,
        "utilization":        1.0,
        "health_mae":         0.0,
        "kendall_tau":        0.9999999999999997,
        "detection_delay":    3.6666666666666665,
        "comm_rounds":      273.3333333333333,
        "convergence_iters": 26.666666666666668,
    },
}


@pytest.mark.parametrize("method", list(PINNED_COMPARATIVE))
def test_comparative_signature(comparative_results, method):
    """Bit-identical signature: any drift in the simulator, optimiser,
    or diagnostic block lights this up immediately."""
    pinned = PINNED_COMPARATIVE[method]
    actual = comparative_results[method]
    for metric_key, expected in pinned.items():
        observed = actual[metric_key][0]
        assert observed == pytest.approx(expected, rel=1e-9, abs=1e-12), (
            f"{method}.{metric_key}: expected {expected!r}, got {observed!r}"
        )
