"""Shared, simulator-independent types and validation for quadrotor experiments.

All angles use radians. Rotations map body-frame vectors to world-frame vectors.
Body angular velocity and body torque are expressed in the body frame.
Position and linear velocity are expressed in the world frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

Array = np.ndarray


def _vector(value: Array, size: int, name: str) -> Array:
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.shape != (size,):
        raise ValueError(f"{name} must contain exactly {size} values; got {out.shape}.")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values.")
    return out.copy()


def validate_rotation(rotation_body_to_world: Array, *, atol: float = 1e-6) -> Array:
    """Return a validated 3x3 body-to-world rotation matrix."""
    rotation = np.asarray(rotation_body_to_world, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation_body_to_world must have shape (3, 3); got {rotation.shape}.")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_body_to_world must contain only finite values.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=atol, rtol=0.0):
        raise ValueError("rotation_body_to_world must be orthonormal.")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol, rtol=0.0):
        raise ValueError(f"rotation_body_to_world must have determinant +1; got {determinant:.8f}.")
    return rotation.copy()


@dataclass(frozen=True)
class SimulationObservation:
    """One normalized simulator observation at a single sample time."""

    time_s: float
    rotation_body_to_world: Array
    body_rates_rad_s: Array
    position_world_m: Array
    velocity_world_m_s: Array

    def __post_init__(self) -> None:
        time_s = float(self.time_s)
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative.")
        object.__setattr__(self, "time_s", time_s)
        object.__setattr__(self, "rotation_body_to_world", validate_rotation(self.rotation_body_to_world))
        object.__setattr__(self, "body_rates_rad_s", _vector(self.body_rates_rad_s, 3, "body_rates_rad_s"))
        object.__setattr__(self, "position_world_m", _vector(self.position_world_m, 3, "position_world_m"))
        object.__setattr__(self, "velocity_world_m_s", _vector(self.velocity_world_m_s, 3, "velocity_world_m_s"))


@dataclass(frozen=True)
class TorqueCommand:
    """Controller request in SI units; collective thrust is optional."""

    body_torque_nm: Array
    collective_thrust_n: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "body_torque_nm", _vector(self.body_torque_nm, 3, "body_torque_nm"))
        if self.collective_thrust_n is not None:
            thrust = float(self.collective_thrust_n)
            if not np.isfinite(thrust) or thrust < 0.0:
                raise ValueError("collective_thrust_n must be finite and non-negative when supplied.")
            object.__setattr__(self, "collective_thrust_n", thrust)


@dataclass(frozen=True)
class MixerOutput:
    """Motor allocation and its *algebraically implied* wrench.

    ``allocator_implied_body_torque_nm`` is reconstructed from the allocation
    matrix and rotor thrust model. It is not a direct measurement of the actual
    torque applied by a native physics engine. A live simulator response check
    must validate motor-axis signs independently.
    """

    motor_rpm: Array
    requested_rotor_thrust_n: Array
    applied_rotor_thrust_n: Array
    requested_body_torque_nm: Array
    allocator_implied_body_torque_nm: Array
    requested_collective_thrust_n: float
    achieved_collective_thrust_n: float
    saturation_scale: float
    saturated: bool

    @property
    def achieved_body_torque_nm(self) -> Array:
        """Backward-compatible alias for the mixer-implied torque (not telemetry)."""
        return self.allocator_implied_body_torque_nm


@dataclass(frozen=True)
class SimulationTrace:
    """Common episode trace used for cross-backend metric calculations.

    State arrays have N samples; command arrays have N-1 intervals. This explicit
    convention avoids silently shifting torque and disturbance timestamps.
    """

    backend: str
    controller: str
    time_s: Array
    rotations_body_to_world: Array
    body_rates_rad_s: Array
    position_world_m: Array
    velocity_world_m_s: Array
    requested_body_torque_nm: Array
    applied_body_torque_nm: Array
    motor_rpm: Array
    saturated: Array

    def validate(self, *, rotation_atol: float = 2e-5) -> None:
        if not self.backend.strip() or not self.controller.strip():
            raise ValueError("backend and controller labels must be non-empty.")
        time = np.asarray(self.time_s, dtype=float).reshape(-1)
        n = time.size
        if n < 2:
            raise ValueError("A trace must contain at least two state samples.")
        if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
            raise ValueError("time_s must be finite and strictly increasing.")
        expected = {
            "rotations_body_to_world": (n, 3, 3),
            "body_rates_rad_s": (n, 3),
            "position_world_m": (n, 3),
            "velocity_world_m_s": (n, 3),
            "requested_body_torque_nm": (n - 1, 3),
            "applied_body_torque_nm": (n - 1, 3),
            "motor_rpm": (n - 1, 4),
            "saturated": (n - 1,),
        }
        for name, shape in expected.items():
            values = np.asarray(getattr(self, name))
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}; got {values.shape}.")
            if name == "saturated":
                if values.dtype.kind != "b":
                    raise ValueError("saturated must be a boolean array.")
            elif not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values.")
        if np.any(np.asarray(self.motor_rpm) < 0.0):
            raise ValueError("motor_rpm cannot contain negative values.")
        rotations = np.asarray(self.rotations_body_to_world, dtype=float)
        for index, rotation in enumerate(rotations):
            try:
                validate_rotation(rotation, atol=rotation_atol)
            except ValueError as exc:
                raise ValueError(f"Invalid rotation at sample {index}: {exc}") from exc


@runtime_checkable
class QuadrotorBackend(Protocol):
    """Minimal API required of a simulator adapter.

    Implementations must report the true simulated body rates (rad/s), not Euler
    angle derivatives, and must accept non-negative per-motor RPM commands.
    """

    name: str

    def reset(self, *, initial_position_world_m: Array, initial_euler_zyx_rad: Array) -> SimulationObservation:
        """Reset the episode and return the initial observation."""
        ...

    def observe(self) -> SimulationObservation:
        """Return a fresh observation."""
        ...

    def step(self, motor_rpm: Array, *, duration_s: float) -> SimulationObservation:
        """Apply motor RPMs for duration_s and return the next observation."""
        ...

    def close(self) -> None:
        """Release simulator resources."""
        ...
