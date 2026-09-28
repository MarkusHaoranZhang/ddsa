"""DIGing distributed optimiser.

Each agent maintains a full ``(N, dim)`` estimate of the global decision
variable -- the "formation state" in our application -- and consensus
on the *agent axis* (axis 0) drives all agents toward the same
estimate. This is the standard DIGing setup of Nedic, Olshevsky, Shi
(2017), where the local cost ``f_i: R^{Nd} -> R`` is a function of the
*entire* decision vector, not just agent ``i``'s row.

A common pitfall is to give each agent only its own row in ``R^d`` and
mix those rows by ``W``. That degenerates into "average the rows"
rather than "agree on the joint optimiser", and for any cost with
inter-agent coupling (such as a formation-keeping penalty) it
collapses to a meaningless point. We avoid that pitfall here: ``self.x``
has shape ``(n_agents, n_agents, dim)`` and ``W @ x`` only acts on
axis 0.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ddsa.config import Config
from ddsa.utils import sinkhorn_double_stochastic


class DIGingOptimizer:
    """Distributed Inexact Gradient method with gradient tracking."""

    def __init__(
        self,
        n_agents: int,
        dim: int,
        W: np.ndarray,
        alpha: float = Config.ALPHA_NUM,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialise per-agent state matrices and cache the mixing matrix ``W``."""
        self.n = n_agents
        self.dim = dim
        self.W = W
        self.alpha = alpha
        self._rng = rng if rng is not None else np.random.default_rng()

        # x[i] is agent i's estimate of the full (n_agents, dim) state.
        # 0.05 = ~5% of the unit formation radius: small enough for
        # the optimiser to converge inside _ITERS_PER_DIAG iterations
        # without the initial spread dominating the early-step error
        # (which biases the convergence-rate sweep in §5.2.1).
        self.x = self._rng.standard_normal((n_agents, n_agents, dim)) * 0.05
        self.y = np.zeros((n_agents, n_agents, dim))
        self.grad_prev = np.zeros((n_agents, n_agents, dim))

        # Per-iteration message loss (scenario 2): a dropped transmission
        # makes the receiver fall back to its own current estimate for
        # that link. ``_loss_rebalance`` re-projects every realised
        # mixing operator to double stochasticity (the framework's
        # re-balancing); without it the row-stochastic-only operator's
        # average drifts with the loss.
        self._loss_edges: list[tuple[int, int]] = []
        self._loss_rate: float = 0.0
        self._loss_rebalance: bool = False

    def set_message_loss(
        self,
        edges: list[tuple[int, int]],
        rate: float,
        rebalance: bool,
    ) -> None:
        """Configure per-iteration message loss on ``edges``."""
        self._loss_edges = list(edges)
        self._loss_rate = float(rate)
        self._loss_rebalance = bool(rebalance)

    def clear_message_loss(self) -> None:
        """Disable per-iteration message loss."""
        self._loss_edges = []
        self._loss_rate = 0.0

    def _effective_mixing(self) -> np.ndarray:
        """Realised mixing operator for one iteration under message loss."""
        if not self._loss_edges or self._loss_rate <= 0.0:
            return self.W
        W = self.W.copy()
        for a, b in self._loss_edges:
            for i, j in ((a, b), (b, a)):
                if self._rng.random() < self._loss_rate:
                    W[i, i] += W[i, j]
                    W[i, j] = 0.0
        if self._loss_rebalance:
            return sinkhorn_double_stochastic(W)
        rs = W.sum(axis=1, keepdims=True)
        rs = np.where(rs > 0, rs, 1.0)
        return W / rs

    def step(
        self,
        local_grad_func: Callable[[np.ndarray, int], np.ndarray],
    ) -> np.ndarray:
        """One DIGing update.

        ``local_grad_func(X, i)`` returns ``∇_X f_i(X)`` where ``X`` is
        the *full* ``(n_agents, dim)`` formation state seen from agent
        ``i``. Each agent applies this gradient against its own copy
        ``self.x[i]``.
        """
        # gather gradients: g[i] = ∇ f_i(self.x[i])
        grad_curr = np.stack(
            [local_grad_func(self.x[i], i) for i in range(self.n)], axis=0
        )
        W_eff = self._effective_mixing()

        # gradient tracking: y_{i}^{k+1} = sum_j W_{ij} y_j + g_i^{k+1} - g_i^{k}
        self.y = np.einsum("ij,jkl->ikl", W_eff, self.y) + grad_curr - self.grad_prev
        self.grad_prev = grad_curr.copy()
        # primal: x_{i}^{k+1} = sum_j W_{ij} x_j - alpha * y_i
        self.x = np.einsum("ij,jkl->ikl", W_eff, self.x) - self.alpha * self.y
        return self.x.copy()

    def optimize(
        self,
        local_grad_func: Callable[[np.ndarray, int], np.ndarray],
        n_iters: int = 50,
        tol: float = 1e-6,
    ) -> tuple[np.ndarray, list[float]]:
        """Run up to ``n_iters`` DIGing steps; stop on consensus."""
        history: list[float] = []
        for k in range(n_iters):
            x = self.step(local_grad_func)
            # consensus error: spread of agent estimates around their mean
            mean = x.mean(axis=0)
            err = float(np.linalg.norm(x - mean))
            history.append(err)
            # Skip the early-iteration warm-up before honouring the
            # tolerance: gradient tracking needs ~10 steps to bring
            # ``y`` close to the average gradient, and consensus error
            # can be transiently small in that window for the wrong
            # reason (initial randomness still dominates over the
            # control direction).
            if k > 10 and err < tol:
                break
        return self.x, history

    def consensus_estimate(self) -> np.ndarray:
        """Average the per-agent estimates into a single ``(n_agents, dim)`` state.

        After DIGing has converged the spread across agents is
        negligible; we return the mean as a robust scalar summary.
        """
        return self.x.mean(axis=0)

    def reset(self) -> None:
        """Re-randomise per-agent state and zero the gradient-tracking buffers."""
        self.x = self._rng.standard_normal((self.n, self.n, self.dim)) * 0.05
        self.y = np.zeros((self.n, self.n, self.dim))
        self.grad_prev = np.zeros((self.n, self.n, self.dim))

    def set_mixing_matrix(self, W: np.ndarray) -> None:
        """Swap in a new mixing matrix mid-run (Section 4.2 adaptation)."""
        if W.shape != self.W.shape:
            raise ValueError(
                f"Mixing matrix shape mismatch: expected {self.W.shape}, got {W.shape}"
            )
        self.W = np.asarray(W, dtype=float)
