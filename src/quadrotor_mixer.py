"""Transparent motor allocation for the CF2X X-configuration quadrotor.

The allocation follows the supported CF2X motor ordering and body-axis sign
conventions. The roll sign was checked against the installed native
``Physics.PYB`` dynamics with measured body-rate responses; an earlier presumed
rotor-link sign was contradicted by that test and has been corrected. The
simplified ``Physics.DYN`` branch in gym-pybullet-drones uses the same CF2X roll
sign in its explicit dynamics equation. The run-time smoke test rechecks signs
against observed angular-rate increments, so a changed simulator/URDF mapping
fails loudly. Inputs are body torques in N*m and collective thrust in N. KF is
in N/RPM^2 and KM is in N*m/RPM^2.
"""
from __future__ import annotations

import numpy as np

from src.simulation_contract import MixerOutput

Array = np.ndarray


class CF2XMixer:
    """Convert a body-torque request to four feasible motor RPM commands.

    The allocation keeps total collective thrust fixed where feasible and scales
    the requested torque vector uniformly if an individual rotor would otherwise
    require negative thrust or exceed maximum thrust. The achieved torque and
    scale are returned so saturation is observable rather than hidden.
    """

    def __init__(
        self,
        *,
        mass_kg: float,
        arm_length_m: float,
        thrust_coefficient_n_per_rpm2: float,
        yaw_moment_coefficient_nm_per_rpm2: float,
        max_rpm: float,
        gravity_m_s2: float = 9.81,
        physics_mode: str = "DYN",
    ) -> None:
        values = {
            "mass_kg": mass_kg,
            "arm_length_m": arm_length_m,
            "thrust_coefficient_n_per_rpm2": thrust_coefficient_n_per_rpm2,
            "yaw_moment_coefficient_nm_per_rpm2": yaw_moment_coefficient_nm_per_rpm2,
            "max_rpm": max_rpm,
            "gravity_m_s2": gravity_m_s2,
        }
        for name, value in values.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        self.mass_kg = float(mass_kg)
        self.arm_length_m = float(arm_length_m)
        self.kf = float(thrust_coefficient_n_per_rpm2)
        self.km = float(yaw_moment_coefficient_nm_per_rpm2)
        self.max_rpm = float(max_rpm)
        self.gravity_m_s2 = float(gravity_m_s2)
        mode = str(physics_mode).strip().upper()
        if mode not in {"PYB", "DYN"}:
            raise ValueError("physics_mode must be either 'PYB' or 'DYN'.")
        self.physics_mode = mode
        self.max_rotor_thrust_n = self.kf * self.max_rpm**2
        self.hover_thrust_n = self.mass_kg * self.gravity_m_s2

        arm = self.arm_length_m / np.sqrt(2.0)
        yaw_ratio = self.km / self.kf
        # Positive CF2X body-roll torque requires motors 0/1 to produce less
        # thrust and motors 2/3 to produce more thrust. The previous PYB branch
        # used the opposite sign; the uploaded native-PYB trace falsified that
        # assumption: +tau_x yielded a negative measured p-dot. Keep the sign
        # explicit here and retain the observed-response acceptance test in the
        # smoke runner so simulator/URDF changes cannot silently regress it.
        # Pitch: motors 1/2 increase for +tau_y. Yaw: motors 1/3 increase for
        # +tau_z under this library's CF2X motor ordering.
        roll_signs = [-arm, -arm, arm, arm]
        self.allocation_matrix = np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                roll_signs,
                [-arm, arm, arm, -arm],
                [-yaw_ratio, yaw_ratio, -yaw_ratio, yaw_ratio],
            ],
            dtype=float,
        )
        if np.linalg.matrix_rank(self.allocation_matrix) != 4:
            raise ValueError("CF2X allocation matrix is singular for the supplied parameters.")
        self._inverse = np.linalg.inv(self.allocation_matrix)

    def allocate(self, body_torque_nm: Array, collective_thrust_n: float | None = None) -> MixerOutput:
        torque = np.asarray(body_torque_nm, dtype=float).reshape(-1)
        if torque.shape != (3,) or not np.all(np.isfinite(torque)):
            raise ValueError("body_torque_nm must contain three finite values.")
        collective = self.hover_thrust_n if collective_thrust_n is None else float(collective_thrust_n)
        if not np.isfinite(collective) or collective < 0.0:
            raise ValueError("collective_thrust_n must be finite and non-negative.")

        max_collective = 4.0 * self.max_rotor_thrust_n
        feasible_collective = float(np.clip(collective, 0.0, max_collective))
        delta = self._inverse @ np.array([0.0, torque[0], torque[1], torque[2]], dtype=float)
        nominal_each = feasible_collective / 4.0
        requested_thrusts = nominal_each + delta
        upper = 1.0
        for change in delta:
            if change > 0.0:
                upper = min(upper, (self.max_rotor_thrust_n - nominal_each) / change)
            elif change < 0.0:
                upper = min(upper, nominal_each / (-change))
        scale = float(np.clip(upper, 0.0, 1.0))
        applied_thrusts = nominal_each + scale * delta
        applied_thrusts = np.clip(applied_thrusts, 0.0, self.max_rotor_thrust_n)
        rpm = np.sqrt(applied_thrusts / self.kf)
        achieved_wrench = self.allocation_matrix @ applied_thrusts
        achieved_torque = achieved_wrench[1:]
        saturated = bool(
            scale < 1.0 - 1e-10
            or abs(feasible_collective - collective) > 1e-10
            or np.any(requested_thrusts < -1e-10)
            or np.any(requested_thrusts > self.max_rotor_thrust_n + 1e-10)
        )
        return MixerOutput(
            motor_rpm=rpm,
            requested_rotor_thrust_n=requested_thrusts,
            applied_rotor_thrust_n=applied_thrusts,
            requested_body_torque_nm=torque.copy(),
            allocator_implied_body_torque_nm=achieved_torque,
            requested_collective_thrust_n=collective,
            achieved_collective_thrust_n=float(np.sum(applied_thrusts)),
            saturation_scale=scale,
            saturated=saturated,
        )

    def torque_from_motor_rpm(self, motor_rpm: Array) -> Array:
        """Return the mixer-model-implied wrench for a four-motor RPM command.

        Returns ``[collective thrust N, tau_x N*m, tau_y N*m, tau_z N*m]``.
        This is an algebraic allocation result, not direct torque telemetry from
        native PyBullet. The smoke test separately checks measured body-rate
        increments for the expected control direction.
        """
        rpm = np.asarray(motor_rpm, dtype=float).reshape(-1)
        if rpm.shape != (4,) or not np.all(np.isfinite(rpm)) or np.any(rpm < 0.0):
            raise ValueError("motor_rpm must contain four finite, non-negative values.")
        thrusts = self.kf * rpm**2
        return self.allocation_matrix @ thrusts
