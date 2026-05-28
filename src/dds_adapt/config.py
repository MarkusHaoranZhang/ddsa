"""Global experiment configuration.

Mirrors the "Implementation details" section of the paper. Default values are
the ones reported in the manuscript; nothing here should be tweaked silently
during a run.
"""

from __future__ import annotations

import numpy as np


class Config:
    """Experiment configuration parameters."""

    # ---- Tracks ---------------------------------------------------------
    NUM_SATELLITES: int = 8           # numerical track
    NUM_SATELLITES_HF: int = 3        # high-fidelity (NASA 42) track

    # ---- Cost function --------------------------------------------------
    # MU and L_SMOOTH are the strong-convexity / smoothness constants of
    # the per-agent quadratic tracking term in §5.1.1; they parametrise
    # the rho_max bound in Theorem 1 (see rho_max_calibration.py).
    MU: float = 1.0
    L_SMOOTH: float = 1.0
    BETA: float = 0.5
    SAFE_OFFSET: float = 0.3  # x_i^0 = x_i^des + SAFE_OFFSET (Section 3.2)

    # ---- Optimisation step sizes ---------------------------------------
    ALPHA_NUM: float = 0.01
    ALPHA_HF: float = 0.005

    # ---- Diagnostic module ---------------------------------------------
    GAMMA_NUM: float = 10.0
    GAMMA_HF: float = 1.0
    L_MAX: int = 3
    N_TRAIN_SAMPLES: int = 500
    RELIABILITY_INIT: float = 0.8
    SIGMOID_TEMP: float = 0.5
    SINKHORN_ITERS: int = 100
    SINKHORN_TOL: float = 1e-6

    # ---- Degradation model ---------------------------------------------
    ETA_SINGLE: float = 0.002
    ETA_FAST: float = 0.003
    ETA_SLOW: float = 0.001
    # Reference per-step decay rate used when reporting health-variation
    # rates relative to rho_max (see rho_max_calibration). The numerical
    # value is the upper end of the eta range used in §5.2.1; "rho_factor"
    # in the calibration script multiplies this constant.
    CHARACTERISTIC_HEALTH_RATE: float = 0.005

    # ---- Diagnosis cadence ---------------------------------------------
    DELTA_T_NUM: int = 50
    DELTA_T_HF: int = 20

    # ---- Statistics -----------------------------------------------------
    N_RUNS: int = 30
    ONSET_WINDOW: tuple[int, int] = (50, 150)

    # ---- Sensors --------------------------------------------------------
    MEASUREMENT_NOISE_STD: float = 0.01

    # ---- Topology -------------------------------------------------------
    @staticmethod
    def get_communication_graph(n_satellites: int = NUM_SATELLITES) -> np.ndarray:
        """Build the doubly stochastic mixing matrix.

        Topology is a ring augmented by four short-cut chords on
        every node (offsets ±2 and ±3). Self / ring / chord weights
        are placed symmetrically around every node, so the resulting
        matrix is symmetric. Single row-sum normalisation then yields
        a doubly stochastic mixing matrix (DIGing's standing
        assumption).
        """
        W = np.zeros((n_satellites, n_satellites))
        for i in range(n_satellites):
            W[i, i] = 0.4
            W[i, (i + 1) % n_satellites] = 0.3
            W[i, (i - 1) % n_satellites] = 0.3
        # symmetric chord pattern that wraps around the ring
        for i in range(n_satellites):
            W[i, (i + 2) % n_satellites] += 0.1
            W[i, (i - 2) % n_satellites] += 0.1
            W[i, (i + 3) % n_satellites] += 0.05
            W[i, (i - 3) % n_satellites] += 0.05
        # the pattern is symmetric so row-sum normalisation makes the matrix
        # doubly stochastic.
        for i in range(n_satellites):
            W[i] /= W[i].sum()
        return W

    @staticmethod
    def edges(n_satellites: int = NUM_SATELLITES) -> list[tuple[int, int]]:
        """Edge list ``(i, j)`` with ``i < j`` for the formation-keeping penalty."""
        edges: list[tuple[int, int]] = []
        for i in range(n_satellites):
            for offset in (1, 2, 3):
                j = (i + offset) % n_satellites
                pair = (min(i, j), max(i, j))
                if pair[0] != pair[1] and pair not in edges:
                    edges.append(pair)
        return edges

    @staticmethod
    def algebraic_connectivity(W: np.ndarray) -> float:
        """Second smallest eigenvalue of the graph Laplacian.

        ``np.linalg.eigvalsh`` already returns a sorted ascending
        array, so the second-smallest eigenvalue is just index 1.
        """
        L = np.diag(W.sum(axis=1)) - W
        eigenvalues = np.linalg.eigvalsh(L)
        return float(eigenvalues[1])
