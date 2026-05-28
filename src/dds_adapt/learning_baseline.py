"""§5.5.4: learning-based fault-tolerant control baseline.

A small MLP is trained to map an observed residual vector (length N) to
a continuous health vector (length N). Training is done on a slow
degradation regime; the test sweep deliberately includes a faster
regime to expose distribution-shift sensitivity.

We use scikit-learn's ``MLPRegressor`` to keep the dependency footprint
small. PyTorch is intentionally avoided so the public companion code
runs without GPU drivers.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from dds_adapt.config import Config
from dds_adapt.residual import broadcast_residual_matrix, residual_energy
from dds_adapt.simulator import SatelliteFormationSimulator


def _collect_supervised(
    n_agents: int,
    n_samples: int,
    eta: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect ``(residual_vector, health_vector)`` pairs at one decay rate.

    We string together short episodes, each one a single-fault trace
    starting at ``onset`` and decaying for the rest of the episode.
    The ``eta`` value controls how fast the decay progresses, so a
    different ``eta`` at evaluation time genuinely represents a
    distribution shift rather than a different fault distribution.
    """
    sim = SatelliteFormationSimulator(n_satellites=n_agents, seed=seed)
    rng = np.random.default_rng(seed + 31)
    X = np.empty((n_samples, n_agents))
    Y = np.empty((n_samples, n_agents))

    episode_len = 80
    onset = 20  # within each episode

    t_idx = 0
    while t_idx < n_samples:
        faulty = int(rng.integers(0, n_agents))
        for inner in range(episode_len):
            if t_idx >= n_samples:
                break
            h = np.ones(n_agents)
            time_into_decay = max(0, inner - onset)
            h[faulty] = float(np.exp(-eta * time_into_decay))
            sim.set_health(h)
            sim.state[:, :2] = sim.desired_positions
            sim.state[:, 2:] = 0.0
            residual = sim.sample_residual()
            own = residual_energy(residual)
            R = broadcast_residual_matrix(own, rng)
            X[t_idx] = R.mean(axis=0)
            Y[t_idx] = h
            t_idx += 1
    return X, Y


def train_learning_baseline(
    n_agents: int = Config.NUM_SATELLITES,
    n_train: int = 2000,
    eta_train: float = Config.ETA_SINGLE,
    seed: int = 0,
) -> Any:
    """Train and return a fitted ``sklearn`` MLPRegressor.

    Returns ``Any`` because ``sklearn`` is an optional dependency and we
    do not want to force its import at type-check time.
    """
    try:
        from sklearn.neural_network import MLPRegressor
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "The learning baseline needs scikit-learn. Install it with "
            "``pip install scikit-learn``."
        ) from exc

    X, Y = _collect_supervised(n_agents, n_train, eta_train, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MLPRegressor(
            hidden_layer_sizes=(128, 128, 128),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            max_iter=200,
            random_state=seed,
        )
        model.fit(X, Y)
    return model


def evaluate_learning_baseline(
    n_agents: int = Config.NUM_SATELLITES,
    n_test: int = 500,
    seed: int = 0,
    eta_train: float = Config.ETA_SINGLE,
    eta_in_dist: float | None = None,
    eta_out_dist: float | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Train once and report MAE on in-distribution vs out-of-distribution traces."""
    if eta_in_dist is None:
        eta_in_dist = eta_train
    if eta_out_dist is None:
        eta_out_dist = 2.5 * eta_train

    model = train_learning_baseline(
        n_agents=n_agents, n_train=2000, eta_train=eta_train, seed=seed
    )

    X_in, Y_in = _collect_supervised(n_agents, n_test, eta_in_dist, seed + 7)
    X_out, Y_out = _collect_supervised(n_agents, n_test, eta_out_dist, seed + 11)
    pred_in = np.clip(model.predict(X_in), 0.0, 1.0)
    pred_out = np.clip(model.predict(X_out), 0.0, 1.0)

    mae_in = float(np.mean(np.abs(pred_in - Y_in)))
    mae_out = float(np.mean(np.abs(pred_out - Y_out)))
    drop_pct = (mae_out - mae_in) / max(mae_in, 1e-6)

    return {
        "in_distribution": {"health_mae": (mae_in, 0.0)},
        "out_of_distribution": {"health_mae": (mae_out, 0.0)},
        "summary": {
            "drop_pct": (drop_pct, 0.0),
        },
    }
