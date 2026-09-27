"""Cost functions and gradients for the formation-keeping objective.

Implements the cost described in Section 5.1.1 of the paper:

* per-agent tracking term :math:`f_i(x_i) = \\tfrac{1}{2}\\|x_i - x_i^{des}\\|^2`
* formation-keeping coupling
  :math:`\\tfrac{\\beta}{2} \\sum_{(i,j) \\in \\mathcal{E}}
  \\|x_i - x_j - d_{ij}\\|^2`
* health-driven safe-anchor regulariser
  :math:`\\tfrac{1-h_i}{2} \\gamma \\|x_i - x_i^{ref}\\|^2`

The first two come from the paper's experimental setting; the third is
the structural-adaptation term from Section 3.2 (Eq. 3 + 4).

Because the coupling term needs neighbour positions, every cost / gradient
function takes a global state ``X`` of shape ``(N, 2)`` instead of a
single agent's ``x_i``. The DIGing optimiser supplies ``X`` from its
own iterate matrix, which is what the gradient-tracking algorithm
already maintains.
"""

from __future__ import annotations

import numpy as np

from ddsa.config import Config


def _safe_anchor(desired_positions: np.ndarray, safe_offset: float) -> np.ndarray:
    """Per-agent safe state.

    Section 5.1.1 of the paper does not specify a separate ``x_i^0`` that
    differs from the formation reference ``x_i^des``; we therefore
    default to ``Config.SAFE_OFFSET`` and let callers override it for
    a distinct safe state.
    """
    return desired_positions + safe_offset


def formation_cost_global(
    X: np.ndarray,
    health: np.ndarray,
    desired_positions: np.ndarray,
    edges: list[tuple[int, int]],
    beta: float = Config.BETA,
    gamma: float = Config.GAMMA_NUM,
    safe_offset: float = Config.SAFE_OFFSET,
) -> float:
    """Sum of all three cost components for a global state ``X``."""
    n = len(desired_positions)
    safe = _safe_anchor(desired_positions, safe_offset)

    tracking = 0.0
    reg = 0.0
    for i in range(n):
        tracking += 0.5 * health[i] * np.linalg.norm(X[i] - desired_positions[i]) ** 2
        reg += 0.5 * (1 - health[i]) * gamma * np.linalg.norm(X[i] - safe[i]) ** 2

    coupling = 0.0
    for i, j in edges:
        d_ij = desired_positions[i] - desired_positions[j]
        coupling += float(np.linalg.norm(X[i] - X[j] - d_ij) ** 2)
    coupling *= 0.5 * beta

    return float(tracking + coupling + reg)


def local_cost_grad(
    X: np.ndarray,
    agent_idx: int,
    health: np.ndarray,
    desired_positions: np.ndarray,
    edges: list[tuple[int, int]],
    beta: float = Config.BETA,
    gamma: float = Config.GAMMA_NUM,
    safe_offset: float = Config.SAFE_OFFSET,
) -> np.ndarray:
    """Gradient ∇_X f_i(X) of agent ``i``'s local cost contribution.

    The decomposition splits the global cost into agent-owned pieces:
    agent ``i`` owns its own tracking + safe-anchor terms, and every
    edge ``(a, b)`` with ``a < b`` is owned by its smaller endpoint.
    Summing ``local_cost_grad`` over all agents reproduces the
    centralised gradient of :func:`formation_cost_global` exactly,
    which is the standing requirement of distributed optimisation.

    The returned array has the same shape ``(N, 2)`` as the global
    state ``X``, with non-zero entries only on the rows of agents
    whose decision variables actually appear in ``f_i``.
    """
    safe = _safe_anchor(desired_positions, safe_offset)
    g = np.zeros_like(X)

    # tracking + safe-anchor: agent i owns its own row only
    g[agent_idx] += health[agent_idx] * (X[agent_idx] - desired_positions[agent_idx])
    g[agent_idx] += (
        (1.0 - health[agent_idx]) * gamma * (X[agent_idx] - safe[agent_idx])
    )

    # coupling: agent i owns every edge (i, j) with j > i
    for a, b in edges:
        if a == agent_idx and b > a:
            d_ab = desired_positions[a] - desired_positions[b]
            diff = X[a] - X[b] - d_ab
            g[a] += beta * diff
            g[b] -= beta * diff
    return g
