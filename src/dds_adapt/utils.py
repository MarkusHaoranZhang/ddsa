"""Small numerical helpers used across the project."""

from __future__ import annotations

import numpy as np

from dds_adapt.config import Config


def sinkhorn_double_stochastic(
    M: np.ndarray,
    n_iters: int = Config.SINKHORN_ITERS,
    tol: float = Config.SINKHORN_TOL,
) -> np.ndarray:
    """Project a non-negative matrix to a doubly stochastic one.

    Implements the Sinkhorn-Knopp iteration: alternately rescale rows and
    columns to sum to one. The sparsity pattern of ``M`` is preserved
    (zeros stay zero). This is the operation invoked by Section 4.2 of the
    paper after the health-driven attenuation
    :math:`\\tilde W_{ij} = w_{ij} \\min(h_i, h_j)`.
    """
    M = np.asarray(M, dtype=float).copy()
    if (M < 0).any():
        raise ValueError("Sinkhorn-Knopp requires a non-negative matrix.")
    for _ in range(n_iters):
        row_sum = M.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        M = M / row_sum
        col_sum = M.sum(axis=0, keepdims=True)
        col_sum = np.where(col_sum > 0, col_sum, 1.0)
        M = M / col_sum
        if (
            np.max(np.abs(M.sum(axis=1) - 1.0)) < tol
            and np.max(np.abs(M.sum(axis=0) - 1.0)) < tol
        ):
            break
    return M


def adapt_mixing_matrix(
    W_base: np.ndarray,
    h_est: np.ndarray,
    self_loop_eps: float = 1e-3,
) -> np.ndarray:
    """Adapt the mixing matrix to a health vector.

    Implements Equation (5) of the paper followed by Sinkhorn-Knopp:
    :math:`\\tilde W_{ij} = w_{ij} \\min(h_i, h_j)`, then double-stochastic
    projection. A tiny self-loop is added so that an agent with health
    near zero does not produce a zero row that breaks the projection.
    """
    M = np.minimum.outer(h_est, h_est)
    tilde = W_base * M
    n = len(h_est)
    tilde = tilde + np.eye(n) * self_loop_eps
    return sinkhorn_double_stochastic(tilde)
