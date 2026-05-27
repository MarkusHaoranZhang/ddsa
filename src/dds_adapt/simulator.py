"""Numerical-track satellite formation simulator.

Each satellite is modelled as a planar double integrator with a thrust
saturation that scales with its actuator health degree. The simulator
exposes both the full state and the *residual* that the diagnostic module
needs (commanded versus realised acceleration), so the diagnosis loop is
closed against the physical layer rather than fed synthetic noise.
"""

from __future__ import annotations

import numpy as np

from dds_adapt.config import Config


class SatelliteFormationSimulator:
    """Planar double-integrator formation with health-modulated actuation."""

    def __init__(
        self,
        n_satellites: int = Config.NUM_SATELLITES,
        dt: float = 0.1,
        formation_radius: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.n = n_satellites
        self.dt = dt
        self.formation_radius = formation_radius

        rng = np.random.default_rng(seed)

        # state per satellite: [x, y, vx, vy]
        self.state = np.zeros((n_satellites, 4))

        # circular reference formation
        self.desired_positions = np.zeros((n_satellites, 2))
        for i in range(n_satellites):
            angle = 2 * np.pi * i / n_satellites
            self.desired_positions[i] = [
                formation_radius * np.cos(angle),
                formation_radius * np.sin(angle),
            ]

        # initial positions perturbed around the reference
        for i in range(n_satellites):
            self.state[i, 0:2] = self.desired_positions[i] + rng.normal(0, 0.05, 2)

        # double-integrator pseudo-LQR gains
        self.K_pos = 0.5
        self.K_vel = 1.0

        self.health = np.ones(n_satellites)
        self._rng = rng

    # ---- Health -------------------------------------------------------
    def set_health(self, health: np.ndarray) -> None:
        """Set the per-satellite actuator health (clipped to [0, 1])."""
        self.health = np.clip(health, 0, 1)

    # ---- Diagnostic probe -------------------------------------------
    def sample_residual(
        self,
        sensor_noise_std: float = Config.MEASUREMENT_NOISE_STD,
    ) -> np.ndarray:
        """Return ``commanded - realised + noise`` at a probe state.

        The diagnostic loop wants a residual that reflects the current
        health, not the agent's accumulated tracking error. We
        therefore evaluate the controller at a small fixed perturbation
        from the reference (a uniform per-agent offset of 0.5 m and a
        velocity of 0.5 m/s along +x), so the commanded acceleration
        is non-zero and the residual signal scales with health. The
        simulator's running state is left untouched.
        """
        state_backup = self.state.copy()
        # probe perturbation: offset = 0.5 m along +x, velocity = 0.5 m/s along +x
        self.state[:, :2] = self.desired_positions + np.array([0.5, 0.0])
        self.state[:, 2:] = np.array([0.5, 0.0])
        cmd = self.commanded_control(self.desired_positions)
        realised = cmd * self.health[:, None]
        sensor_noise = self._rng.normal(0, sensor_noise_std, realised.shape)
        residual = cmd - realised + sensor_noise
        self.state = state_backup
        return residual

    # ---- Dynamics -----------------------------------------------------
    def commanded_control(self, target_positions: np.ndarray) -> np.ndarray:
        """Nominal LQR command for the planar double integrator.

        Pure feedback against position and velocity error. At the
        formation reference with zero velocity the command is also
        zero, so the simulator does not drift in steady state. The
        diagnostic probe (``sample_residual``) deliberately evaluates
        the controller at a small *off-reference* perturbation so the
        commanded acceleration is non-zero and the residual is
        informative about actuator health.
        """
        ctrl = np.zeros((self.n, 2))
        for i in range(self.n):
            pos_error = target_positions[i] - self.state[i, 0:2]
            vel_error = -self.state[i, 2:4]
            ctrl[i] = self.K_pos * pos_error + self.K_vel * vel_error
        return ctrl

    def step(
        self,
        commanded: np.ndarray,
        sensor_noise_std: float = Config.MEASUREMENT_NOISE_STD,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply the health-attenuated command and return the residual.

        Returns
        -------
        state : np.ndarray
            Updated [x, y, vx, vy] for every agent.
        realised : np.ndarray
            The acceleration actually applied (commanded * health).
        residual : np.ndarray
            ``commanded - realised + sensor_noise``. This is what each
            agent's diagnostic block will see; the magnitude is
            informative about the agent's own health.
        """
        realised = commanded * self.health[:, None]
        # noisy proprioceptive readout used by the diagnostic module
        sensor_noise = self._rng.normal(0, sensor_noise_std, realised.shape)
        residual = commanded - realised + sensor_noise

        # propagate state under the realised acceleration
        self.state[:, 2:4] += realised * self.dt
        self.state[:, 0:2] += self.state[:, 2:4] * self.dt

        return self.state.copy(), realised, residual

    # ---- Helpers ------------------------------------------------------
    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def get_positions(self) -> np.ndarray:
        return self.state[:, :2].copy()
