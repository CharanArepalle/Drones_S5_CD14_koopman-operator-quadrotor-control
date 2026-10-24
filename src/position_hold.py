"""Shared conservative world-position hold outer loop for quadrotor adapters.

The research attitude controllers remain the inner loop. This module maps a
world-frame position/velocity error to a desired body attitude and collective
thrust, allowing PyBullet, MuJoCo and Gazebo adapters to share the same logic.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

Array = np.ndarray


def _finite_vector(value: Array, name: str) -> Array:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 3-vector.")
    return array.copy()


@dataclass(frozen=True)
class PositionHoldConfig:
    position_kp: tuple[float, float, float] = (1.4, 1.4, 2.0)
    velocity_kd: tuple[float, float, float] = (1.8, 1.8, 1.7)
    max_lateral_acceleration_m_s2: float = 1.6
    max_vertical_acceleration_m_s2: float = 1.5
    max_tilt_deg: float = 12.0

    def validate(self) -> None:
        for name in ("position_kp", "velocity_kd"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain three finite, non-negative values.")
        if not np.isfinite(self.max_lateral_acceleration_m_s2) or self.max_lateral_acceleration_m_s2 <= 0.0:
            raise ValueError("max_lateral_acceleration_m_s2 must be positive and finite.")
        if not np.isfinite(self.max_vertical_acceleration_m_s2) or self.max_vertical_acceleration_m_s2 <= 0.0:
            raise ValueError("max_vertical_acceleration_m_s2 must be positive and finite.")
        if not np.isfinite(self.max_tilt_deg) or not (0.0 < self.max_tilt_deg <= 30.0):
            raise ValueError("max_tilt_deg must be in (0, 30].")


@dataclass(frozen=True)
class PositionHoldCommand:
    desired_rotation_body_to_world: Array
    collective_thrust_n: float
    acceleration_command_world_m_s2: Array
    position_error_world_m: Array


def desired_attitude_from_force(force_world_n: Array, yaw_des_rad: float = 0.0) -> Array:
    """Return a body-to-world rotation with body +Z aligned to force direction."""
    force = _finite_vector(force_world_n, "force_world_n")
    if np.linalg.norm(force) <= 1e-8:
        raise ValueError("force_world_n must be non-zero.")
    if not np.isfinite(yaw_des_rad):
        raise ValueError("yaw_des_rad must be finite.")

    b3 = force / np.linalg.norm(force)
    b1_heading = np.array([np.cos(yaw_des_rad), np.sin(yaw_des_rad), 0.0])
    b2 = np.cross(b3, b1_heading)
    if np.linalg.norm(b2) <= 1e-8:
        b1_heading = np.array([0.0, 1.0, 0.0])
        b2 = np.cross(b3, b1_heading)
    b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    rotation = np.column_stack([b1, b2, b3])

    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise ArithmeticError("Desired attitude construction lost orthogonality.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
        raise ArithmeticError("Desired attitude must be right-handed.")
    return rotation


def compute_position_hold_command(
    position_world_m: Array,
    velocity_world_m_s: Array,
    target_position_world_m: Array,
    mass_kg: float,
    gravity_m_s2: float,
    config: PositionHoldConfig | None = None,
    *,
    target_yaw_rad: float = 0.0,
) -> PositionHoldCommand:
    """PD outer loop with bounded acceleration, tilt and collective thrust.

    Coordinate convention: world +Z points upward, and collective thrust is
    along the vehicle's body +Z axis. All distances use metres, acceleration
    m/s^2, mass kg, thrust N and angles radians internally.
    """
    cfg = config or PositionHoldConfig()
    cfg.validate()
    position = _finite_vector(position_world_m, "position_world_m")
    velocity = _finite_vector(velocity_world_m_s, "velocity_world_m_s")
    target = _finite_vector(target_position_world_m, "target_position_world_m")
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive and finite.")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be positive and finite.")

    position_error = target - position
    kp = np.asarray(cfg.position_kp, dtype=float)
    kd = np.asarray(cfg.velocity_kd, dtype=float)
    acceleration = kp * position_error - kd * velocity
    acceleration[2] = float(np.clip(
        acceleration[2], -cfg.max_vertical_acceleration_m_s2,
        cfg.max_vertical_acceleration_m_s2,
    ))

    lateral_norm = float(np.linalg.norm(acceleration[:2]))
    lateral_cap_from_tilt = (gravity_m_s2 + acceleration[2]) * np.tan(np.deg2rad(cfg.max_tilt_deg))
    lateral_limit = max(0.0, min(cfg.max_lateral_acceleration_m_s2, lateral_cap_from_tilt))
    if lateral_norm > lateral_limit and lateral_norm > 0.0:
        acceleration[:2] *= lateral_limit / lateral_norm

    force_world = mass_kg * (acceleration + np.array([0.0, 0.0, gravity_m_s2]))
    thrust = float(np.linalg.norm(force_world))
    desired_rotation = desired_attitude_from_force(force_world, target_yaw_rad)
    return PositionHoldCommand(
        desired_rotation_body_to_world=desired_rotation,
        collective_thrust_n=thrust,
        acceleration_command_world_m_s2=acceleration.copy(),
        position_error_world_m=position_error.copy(),
    )
