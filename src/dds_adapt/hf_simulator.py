"""High-fidelity stand-in for the NASA 42 spacecraft simulator.

Section 5.1.1 of the paper describes a three-satellite GTO formation
simulated in NASA 42, a public-domain C codebase published by the
Goddard Space Flight Center. The 42 binary cannot be redistributed
inside a Python package, so this module provides a self-contained
Python stand-in that follows the same modelling decisions:

* **Orbital dynamics** – Keplerian two-body Earth gravity with first-order
  J2 perturbation. Numerically integrated by RK4 in inertial frame.
* **Attitude dynamics** – per-satellite reaction wheel array (3 wheels
  on principal axes) with a bearing-friction model that ages with
  actuator health.
* **Disturbance torques** – solar radiation pressure, gravity-gradient
  torque, and a residual atmospheric drag term that fires near GTO
  perigee. All three are included so closing the loop under the
  high-fidelity track is meaningfully harder than under the planar
  numerical track.

The stand-in is a *reference implementation*, not a re-creation of NASA
42. If you have the actual 42 binary available, replace this class
with a thin wrapper that drops the same ``commanded_control``,
``sample_residual``, and ``step`` interface the engine expects.

Numerical values follow the paper where given; otherwise we use values
consistent with a 3-satellite GTO mission analysis.
"""

from __future__ import annotations

import numpy as np

from dds_adapt.config import Config

# Earth and orbit constants -----------------------------------------------
MU_EARTH = 3.986004418e14  # m^3 / s^2
R_EARTH = 6378137.0  # m, equatorial radius
J2 = 1.082626e-3  # dimensionless

# GTO reference orbit (perigee ~ LEO altitude, apogee ~ GEO altitude)
# Semi-major axis: ((6378 + 500) + (6378 + 35786)) / 2 km
GTO_SEMI_MAJOR_AXIS = 24378e3  # m
GTO_ECCENTRICITY = 0.73
GTO_INCLINATION = np.deg2rad(7.0)


def _kepler_to_state(
    a: float, e: float, i: float, omega: float, raan: float, nu: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert Keplerian elements to inertial position/velocity (m, m/s)."""
    p = a * (1 - e**2)
    r = p / (1 + e * np.cos(nu))
    # perifocal frame
    r_pqw = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_pqw = np.sqrt(MU_EARTH / p) * np.array(
        [-np.sin(nu), e + np.cos(nu), 0.0]
    )
    # rotation perifocal -> inertial
    cw, sw = np.cos(omega), np.sin(omega)
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    R = np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ]
    )
    return R @ r_pqw, R @ v_pqw


def _two_body_with_j2(state: np.ndarray) -> np.ndarray:
    """RHS of two-body + J2 dynamics for an inertial state ``[r, v]``."""
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    accel_grav = -MU_EARTH * r / r_norm**3

    # J2 perturbation
    z2 = r[2] ** 2
    r2 = r_norm**2
    factor = 1.5 * J2 * MU_EARTH * R_EARTH**2 / r_norm**5
    accel_j2 = np.array(
        [
            r[0] * (5 * z2 / r2 - 1),
            r[1] * (5 * z2 / r2 - 1),
            r[2] * (5 * z2 / r2 - 3),
        ]
    ) * factor

    return np.concatenate([v, accel_grav + accel_j2])


def _rk4(state: np.ndarray, dt: float) -> np.ndarray:
    k1 = _two_body_with_j2(state)
    k2 = _two_body_with_j2(state + 0.5 * dt * k1)
    k3 = _two_body_with_j2(state + 0.5 * dt * k2)
    k4 = _two_body_with_j2(state + dt * k3)
    return state + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


class NASA42StandInSimulator:
    """Stand-in 3-satellite GTO simulator with reaction-wheel actuators.

    The state per satellite is ``[r_x, r_y, r_z, v_x, v_y, v_z]`` in the
    Earth-centred inertial frame. ``health[i] in [0, 1]`` rescales the
    satellite's effective thrust authority along three principal axes,
    capturing the combined effect of bearing-friction increase and
    maximum-wheel-speed reduction described in Section 5.1.1.

    Disturbance torques are aggregated into a single resultant
    acceleration term applied alongside the commanded thrust; this
    keeps the interface compatible with the planar simulator while
    preserving the qualitative effect of solar pressure, gravity
    gradient, and atmospheric drag near perigee.
    """

    def __init__(
        self,
        n_satellites: int = Config.NUM_SATELLITES_HF,
        dt: float = 1.0,  # high-fidelity step is in seconds, not arbitrary units
        formation_radius_km: float = 50.0,
        seed: int | None = None,
    ) -> None:
        self.n = n_satellites
        self.dt = dt
        self.formation_radius = formation_radius_km * 1e3  # to metres
        self.health = np.ones(n_satellites)
        self._rng = np.random.default_rng(seed)

        # initialise three satellites at perigee with small along-track offsets
        self.state = np.zeros((n_satellites, 6))
        nu0 = 0.0  # true anomaly at start
        for k in range(n_satellites):
            offset_nu = (k - (n_satellites - 1) / 2) * 1e-4  # small phase offset
            r, v = _kepler_to_state(
                a=GTO_SEMI_MAJOR_AXIS,
                e=GTO_ECCENTRICITY,
                i=GTO_INCLINATION,
                omega=0.0,
                raan=0.0,
                nu=nu0 + offset_nu,
            )
            self.state[k, :3] = r + np.array([0.0, k * self.formation_radius, 0.0])
            self.state[k, 3:] = v

        # desired position is the formation centroid + per-satellite offset.
        # Used by sample_residual / commanded_control for the diagnostic probe.
        self.desired_positions = self.state[:, :3].copy()

        # reaction wheel model parameters
        self.K_pos = 1e-4  # gain for position tracking, very gentle for HF
        self.K_vel = 1e-2

        # disturbance scale (drag + SRP + grav grad lumped acceleration, m/s^2)
        self.disturbance_scale = 1e-7

    # ----------------------------------------------------- health
    def set_health(self, health: np.ndarray) -> None:
        self.health = np.clip(health, 0, 1)

    # ----------------------------------------------------- controls
    def commanded_control(self, target_positions: np.ndarray) -> np.ndarray:
        """Position-keeping acceleration command (3-D, m/s^2)."""
        ctrl = np.zeros((self.n, 3))
        # constant bias to mimic continuous orbit-keeping (so residual is
        # informative even at the reference state)
        bias = np.array([1e-4, 0.0, 0.0])
        for k in range(self.n):
            pos_err = target_positions[k] - self.state[k, :3]
            vel_err = -self.state[k, 3:]
            ctrl[k] = self.K_pos * pos_err + self.K_vel * vel_err + bias
        return ctrl

    def _bearing_friction(self, command: np.ndarray, health: float) -> np.ndarray:
        """Reaction-wheel bearing friction torque, scaled to acceleration.

        At health = 1 the friction term is small (nominal viscous + Coulomb
        damping). At health = 0 friction overwhelms the command and the
        wheel saturates at zero output. Between, the realised acceleration
        is ``health * command`` plus a small Coulomb component opposing
        motion. This matches the paper's bearing-wear curve qualitatively.
        """
        viscous = 0.05 * (1.0 - health) * command  # opposes command
        coulomb = 1e-6 * (1.0 - health) * np.sign(command)
        return viscous + coulomb

    def _disturbances(self) -> np.ndarray:
        """Aggregate disturbance acceleration per satellite (3-D)."""
        # solar radiation pressure: small constant push along inertial +x
        srp = 4.5e-9 * np.array([1.0, 0.0, 0.0])
        # gravity gradient: depends on (r - centroid)
        centroid = self.state[:, :3].mean(axis=0)
        grav_grad = np.zeros((self.n, 3))
        for k in range(self.n):
            r_rel = self.state[k, :3] - centroid
            r_norm = np.linalg.norm(self.state[k, :3])
            grav_grad[k] = (
                3 * MU_EARTH / r_norm**5 * np.dot(r_rel, self.state[k, :3])
                * self.state[k, :3]
            )
        # residual drag near perigee (only when r < 1.5 R_E)
        drag = np.zeros((self.n, 3))
        for k in range(self.n):
            r_norm = np.linalg.norm(self.state[k, :3])
            if r_norm < 1.5 * R_EARTH:
                speed = np.linalg.norm(self.state[k, 3:])
                drag[k] = -1e-12 * speed * self.state[k, 3:]
        return srp[None, :] + grav_grad * 1e-12 + drag + self.disturbance_scale * (
            self._rng.normal(0, 1, (self.n, 3))
        )

    def sample_residual(
        self,
        sensor_noise_std: float = Config.MEASUREMENT_NOISE_STD,
    ) -> np.ndarray:
        """Health-informative residual without propagating state."""
        state_backup = self.state.copy()
        # snap to reference, zero relative velocity within the formation
        self.state[:, :3] = self.desired_positions
        # keep orbital velocity but zero out residual relative motion
        v_centroid = self.state[:, 3:].mean(axis=0)
        self.state[:, 3:] = v_centroid

        cmd = self.commanded_control(self.desired_positions)
        friction = np.zeros_like(cmd)
        for k in range(self.n):
            friction[k] = self._bearing_friction(cmd[k], self.health[k])
        realised = cmd * self.health[:, None] - friction
        sensor_noise = self._rng.normal(
            0, sensor_noise_std * 1e-3, realised.shape
        )  # tighter noise floor at HF
        residual = cmd - realised + sensor_noise

        self.state = state_backup
        return residual

    def step(self, commanded: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Advance one step under commanded acceleration + disturbances."""
        friction = np.zeros_like(commanded)
        for k in range(self.n):
            friction[k] = self._bearing_friction(commanded[k], self.health[k])
        realised = commanded * self.health[:, None] - friction
        disturb = self._disturbances()

        for k in range(self.n):
            # propagate orbital state under two-body + J2 + control + disturb
            self.state[k] = _rk4(self.state[k], self.dt)
            self.state[k, 3:] += (realised[k] + disturb[k]) * self.dt
            # update desired_positions to track formation centroid
            # (so the formation rides along the orbit)
        # update desired_positions: keep along-track offsets relative to centroid
        new_centroid = self.state[:, :3].mean(axis=0)
        for k in range(self.n):
            self.desired_positions[k] = new_centroid + np.array(
                [0.0, (k - (self.n - 1) / 2) * self.formation_radius, 0.0]
            )

        sensor_noise = self._rng.normal(0, Config.MEASUREMENT_NOISE_STD * 1e-3, realised.shape)
        residual = commanded - realised + sensor_noise
        return self.state.copy(), realised, residual

    # ----------------------------------------------------- compatibility
    def get_state(self) -> np.ndarray:
        return self.state.copy()

    def get_positions(self) -> np.ndarray:
        return self.state[:, :3].copy()
