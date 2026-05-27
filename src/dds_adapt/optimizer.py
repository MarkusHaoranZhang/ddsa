"""DIGing distributed optimiser."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dds_adapt.config import Config


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
        self.n = n_agents
        self.dim = dim
        self.W = W
        self.alpha = alpha
        self._rng = rng if rng is not None else np.random.default_rng()

        self.x = self._rng.standard_normal((n_agents, dim)) * 0.1
        self.y = np.zeros((n_agents, dim))
        self.grad_prev = np.zeros((n_agents, dim))

    def step(
        self,
        grad_func: Callable[[np.ndarray, int, float], np.ndarray] | None = None,
        health: np.ndarray | None = None,
        *,
        global_grad_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        """One DIGing update.

        Either ``grad_func`` (per-agent, signature ``(x_i, i, h_i)``) or
        ``global_grad_func`` (signature ``(X, health) -> (N, dim)``) must
        be supplied. The global form lets the cost include inter-agent
        coupling such as the formation-keeping penalty in Section 5.1.1
        of the paper, which the per-agent form cannot express.
        """
        if health is None:
            health = np.ones(self.n)
        if global_grad_func is None and grad_func is None:
            raise ValueError("supply either grad_func or global_grad_func")

        if global_grad_func is not None:
            grad_curr = global_grad_func(self.x, health)
        else:
            assert grad_func is not None  # narrowed by the raise above
            grad_curr = np.zeros_like(self.x)
            for i in range(self.n):
                grad_curr[i] = grad_func(self.x[i], i, health[i])

        self.y = self.W @ self.y + grad_curr - self.grad_prev
        self.grad_prev = grad_curr.copy()
        self.x = self.W @ self.x - self.alpha * self.y
        return self.x.copy()

    def optimize(
        self,
        grad_func: Callable[[np.ndarray, int, float], np.ndarray] | None = None,
        health: np.ndarray | None = None,
        n_iters: int = 50,
        tol: float = 1e-6,
        *,
        global_grad_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> tuple[np.ndarray, list[float]]:
        """Run up to ``n_iters`` DIGing steps; stop on consensus."""
        if health is None:
            health = np.ones(self.n)
        history: list[float] = []
        for k in range(n_iters):
            x = self.step(
                grad_func, health, global_grad_func=global_grad_func
            )
            consensus_error = float(np.linalg.norm(x - x.mean(axis=0)))
            history.append(consensus_error)
            if k > 10 and consensus_error < tol:
                break
        return self.x, history

    def reset(self) -> None:
        self.x = self._rng.standard_normal((self.n, self.dim)) * 0.1
        self.y = np.zeros((self.n, self.dim))
        self.grad_prev = np.zeros((self.n, self.dim))

    def set_mixing_matrix(self, W: np.ndarray) -> None:
        """Swap in a new mixing matrix mid-run (Section 4.2 adaptation)."""
        if W.shape != self.W.shape:
            raise ValueError(
                f"Mixing matrix shape mismatch: expected {self.W.shape}, got {W.shape}"
            )
        self.W = np.asarray(W, dtype=float)
