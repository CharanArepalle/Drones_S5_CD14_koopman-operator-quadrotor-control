"""Adapter for the CtrlAviary environment in gym-pybullet-drones.

The project Koopman model was identified using an inertia vector different from
that of the stock CF2X vehicle. To keep the learned input in its original
(model-equivalent) coordinates while commanding physical CF2X motor RPM, this
adapter performs an explicit diagonal-inertia input mapping:

    tau_physical = diag(J_simulator / J_identified) @ tau_model_equivalent

For a diagonal rigid-body model, this makes the commanded torque contribution
J_simulator^{-1} tau_physical equal to J_identified^{-1} tau_model_equivalent.
It does NOT make the two physical plants identical: gyroscopic terms,
translational dynamics, motor/propeller effects, and other simulator physics
still differ. All model-equivalent and physical torque channels are kept
separate in the returned command record.

Supported vehicle in this first adapter: CF2X. Its physical mass, inertia,
arm length, thrust/moment coefficients, and RPM limit are read from the
installed gym-pybullet-drones model; fixture values are never used at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np

from src.quadrotor_mixer import CF2XMixer
from src.simulation_contract import SimulationObservation

Array = np.ndarray


def _finite_vector(value: Any, size: int, name: str, *, positive: bool = False) -> Array:
    result = np.asarray(value, dtype=float).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} must contain {size} values; got shape {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    if positive and np.any(result <= 0.0):
        raise ValueError(f"{name} values must be strictly positive.")
    return result.copy()


def quaternion_xyzw_to_rotation(quaternion_xyzw: Array) -> Array:
    """Convert a PyBullet xyzw quaternion to a body-to-world rotation matrix."""
    q = _finite_vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion_xyzw must have a non-zero norm.")
    x, y, z, w = q / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )
    # Numerical guard: malformed/incorrectly decoded quaternions fail loudly.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0):
        raise ValueError("Quaternion conversion did not produce an orthonormal rotation.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("Quaternion conversion did not produce a proper rotation.")
    return rotation


def rotation_to_euler_zyx(rotation_body_to_world: Array) -> Array:
    """Return [roll, pitch, yaw] for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    rotation = np.asarray(rotation_body_to_world, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_body_to_world must be a finite 3x3 matrix.")
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


def map_model_equivalent_torque_to_physical(
    model_equivalent_torque_nm: Array,
    simulator_inertia_kg_m2: Array,
    identified_model_inertia_kg_m2: Array,
) -> Array:
    """Map a learned-model torque coordinate into physical simulator torque.

    For diagonal inertias, the input-dependent part of angular acceleration is
    matched because J_sim^-1 @ diag(J_sim/J_model) @ tau_model = J_model^-1 @ tau_model.
    This is an explicit interface mapping, not a claim that other dynamics match.
    """
    tau_model = _finite_vector(model_equivalent_torque_nm, 3, "model_equivalent_torque_nm")
    j_sim = _finite_vector(simulator_inertia_kg_m2, 3, "simulator_inertia_kg_m2", positive=True)
    j_model = _finite_vector(identified_model_inertia_kg_m2, 3, "identified_model_inertia_kg_m2", positive=True)
    return (j_sim / j_model) * tau_model


@dataclass(frozen=True)
class PyBulletAdapterConfig:
    control_frequency_hz: int = 20
    physics_frequency_hz: int = 240
    gui: bool = False
    record: bool = False
    user_debug_gui: bool = False
    gravity_fallback_m_s2: float = 9.81
    identified_model_inertia_kg_m2: tuple[float, float, float] = (0.021, 0.024, 0.039)
    identified_model_torque_limits_nm: tuple[float, float, float] = (0.12, 0.12, 0.06)
    hover_compensation: bool = True
    minimum_vertical_projection_for_hover_compensation: float = 0.35
    output_folder: str = "results/simulation_runs"

    def validate(self) -> None:
        if self.control_frequency_hz <= 0 or self.physics_frequency_hz <= 0:
            raise ValueError("Simulation frequencies must be positive integers.")
        if self.physics_frequency_hz % self.control_frequency_hz != 0:
            raise ValueError("physics_frequency_hz must be divisible by control_frequency_hz.")
        _finite_vector(self.identified_model_inertia_kg_m2, 3, "identified_model_inertia_kg_m2", positive=True)
        _finite_vector(self.identified_model_torque_limits_nm, 3, "identified_model_torque_limits_nm", positive=True)
        if not np.isfinite(self.gravity_fallback_m_s2) or self.gravity_fallback_m_s2 <= 0:
            raise ValueError("gravity_fallback_m_s2 must be positive and finite.")
        if not (0.0 < self.minimum_vertical_projection_for_hover_compensation <= 1.0):
            raise ValueError("minimum_vertical_projection_for_hover_compensation must lie in (0, 1].")

    @classmethod
    def from_json(cls, path: str | Path) -> "PyBulletAdapterConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        cfg = cls(
            control_frequency_hz=int(raw.get("control_frequency_hz", 20)),
            physics_frequency_hz=int(raw.get("physics_frequency_hz", 240)),
            gui=bool(raw.get("gui", False)),
            record=bool(raw.get("record", False)),
            user_debug_gui=bool(raw.get("user_debug_gui", False)),
            gravity_fallback_m_s2=float(raw.get("gravity_fallback_m_s2", 9.81)),
            identified_model_inertia_kg_m2=tuple(raw.get("identified_model_inertia_kg_m2", (0.021, 0.024, 0.039))),
            identified_model_torque_limits_nm=tuple(raw.get("identified_model_torque_limits_nm", (0.12, 0.12, 0.06))),
            hover_compensation=bool(raw.get("hover_compensation", True)),
            minimum_vertical_projection_for_hover_compensation=float(
                raw.get("minimum_vertical_projection_for_hover_compensation", 0.35)
            ),
            output_folder=str(raw.get("output_folder", "results/simulation_runs")),
        )
        cfg.validate()
        return cfg


@dataclass(frozen=True)
class MappedTorqueCommand:
    """Auditable command separating learned-model and physical actuator units."""

    model_equivalent_torque_nm: Array
    requested_physical_torque_nm: Array
    mixer_output: Any
    collective_thrust_requested_n: float


class PyBulletDronesAdapter:
    """Single-CF2X adapter implementing the project's common backend contract."""

    name = "gym-pybullet-drones/CF2X"

    def __init__(self, config: PyBulletAdapterConfig | None = None) -> None:
        self.config = config or PyBulletAdapterConfig()
        self.config.validate()
        self._env: Any | None = None
        self._mixer: CF2XMixer | None = None
        self._simulator_inertia_kg_m2: Array | None = None
        self._mass_kg: float | None = None
        self._gravity_m_s2: float = self.config.gravity_fallback_m_s2
        self._time_s = 0.0
        self._closed = False
        self._last_observation: Any = None

    @property
    def control_timestep_s(self) -> float:
        return 1.0 / self.config.control_frequency_hz

    @property
    def mixer(self) -> CF2XMixer:
        if self._mixer is None:
            raise RuntimeError("Adapter is not initialized; call reset() first.")
        return self._mixer

    @property
    def simulator_inertia_kg_m2(self) -> Array:
        if self._simulator_inertia_kg_m2 is None:
            raise RuntimeError("Adapter is not initialized; call reset() first.")
        return self._simulator_inertia_kg_m2.copy()

    @property
    def mass_kg(self) -> float:
        if self._mass_kg is None:
            raise RuntimeError("Adapter is not initialized; call reset() first.")
        return self._mass_kg

    @property
    def gravity_m_s2(self) -> float:
        return self._gravity_m_s2

    def _create_environment(self, initial_position_world_m: Array, initial_euler_zyx_rad: Array) -> None:
        try:
            from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
            from gym_pybullet_drones.utils.enums import DroneModel, Physics
        except Exception as exc:  # pragma: no cover - depends on user's environment
            raise RuntimeError(
                "gym-pybullet-drones could not be imported in this Python environment. "
                "Install your already-selected gym-pybullet-drones checkout into this project's .venv, "
                "then rerun this command. No fallback simulator is substituted."
            ) from exc

        if not hasattr(DroneModel, "CF2X") or not hasattr(Physics, "PYB"):
            raise RuntimeError(
                "This adapter requires the gym-pybullet-drones CtrlAviary API with DroneModel.CF2X and Physics.PYB. "
                "The installed fork/version exposes a different API; do not silently substitute another vehicle."
            )

        initial_xyzs = np.asarray(initial_position_world_m, dtype=float).reshape(1, 3)
        initial_rpys = np.asarray(initial_euler_zyx_rad, dtype=float).reshape(1, 3)
        try:
            self._env = CtrlAviary(
                drone_model=DroneModel.CF2X,
                num_drones=1,
                initial_xyzs=initial_xyzs,
                initial_rpys=initial_rpys,
                physics=Physics.PYB,
                pyb_freq=self.config.physics_frequency_hz,
                ctrl_freq=self.config.control_frequency_hz,
                gui=self.config.gui,
                record=self.config.record,
                obstacles=False,
                user_debug_gui=self.config.user_debug_gui,
                output_folder=self.config.output_folder,
            )
        except TypeError as exc:  # pragma: no cover - depends on user's package version
            raise RuntimeError(
                "Installed CtrlAviary constructor does not match this adapter's documented API. "
                "Please provide the exact installed gym-pybullet-drones version before changing any project files."
            ) from exc

        # Explicit reset accommodates old Gym and newer Gymnasium-style wrappers.
        reset_result = self._env.reset()
        self._last_observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        self._time_s = 0.0
        self._closed = False
        self._initialize_vehicle_parameters()

    def _initialize_vehicle_parameters(self) -> None:
        required = ("M", "L", "KF", "KM", "MAX_RPM", "J")
        missing = [name for name in required if not hasattr(self._env, name)]
        if missing:
            raise RuntimeError(
                "Installed CtrlAviary is missing required physical-model attributes: " + ", ".join(missing)
            )

        inertia = np.asarray(self._env.J, dtype=float)
        if inertia.shape == (3, 3):
            j_diag = np.diag(inertia)
        elif inertia.shape == (3,):
            j_diag = inertia
        else:
            raise RuntimeError(f"Unrecognized simulator inertia shape {inertia.shape}; expected (3,3) or (3,).")
        self._simulator_inertia_kg_m2 = _finite_vector(j_diag, 3, "simulator inertia", positive=True)
        self._mass_kg = float(self._env.M)
        if not np.isfinite(self._mass_kg) or self._mass_kg <= 0.0:
            raise RuntimeError("Simulator reports a non-positive or invalid mass.")
        env_gravity = getattr(self._env, "G", self.config.gravity_fallback_m_s2)
        self._gravity_m_s2 = float(env_gravity)
        if not np.isfinite(self._gravity_m_s2) or self._gravity_m_s2 <= 0.0:
            raise RuntimeError("Simulator reports an invalid gravitational acceleration.")

        self._mixer = CF2XMixer(
            mass_kg=self._mass_kg,
            arm_length_m=float(self._env.L),
            thrust_coefficient_n_per_rpm2=float(self._env.KF),
            yaw_moment_coefficient_nm_per_rpm2=float(self._env.KM),
            max_rpm=float(self._env.MAX_RPM),
            gravity_m_s2=self._gravity_m_s2,
            physics_mode="PYB",
        )

    def reset(
        self,
        *,
        initial_position_world_m: Array,
        initial_euler_zyx_rad: Array,
    ) -> SimulationObservation:
        position = _finite_vector(initial_position_world_m, 3, "initial_position_world_m")
        euler = _finite_vector(initial_euler_zyx_rad, 3, "initial_euler_zyx_rad")
        if position[2] <= 0.0:
            raise ValueError("initial position Z must be above the ground (positive metres).")
        if self._env is not None:
            self.close()
        self._create_environment(position, euler)
        return self.observe()

    def _read_raw_state(self) -> tuple[Array, Array, Array, Array]:
        """Return position, quaternion xyzw, world velocity, and world angular velocity."""
        env = self._env
        if env is None:
            raise RuntimeError("Adapter is not initialized; call reset() first.")

        # These BaseAviary arrays are the most direct, version-stable source of
        # the physical state. Angular velocity is in world coordinates in the
        # source implementation; convert it to body coordinates in observe().
        names = ("pos", "quat", "vel", "ang_v")
        if all(hasattr(env, name) for name in names):
            pos = np.asarray(env.pos[0], dtype=float).reshape(3)
            quat = np.asarray(env.quat[0], dtype=float).reshape(4)
            vel = np.asarray(env.vel[0], dtype=float).reshape(3)
            omega_world = np.asarray(env.ang_v[0], dtype=float).reshape(3)
            return pos.copy(), quat.copy(), vel.copy(), omega_world.copy()

        # Compatibility fallback for forks exposing the canonical 20-vector.
        if hasattr(env, "_getDroneStateVector"):
            state = np.asarray(env._getDroneStateVector(0), dtype=float).reshape(-1)
            if state.shape == (20,):
                return state[0:3].copy(), state[3:7].copy(), state[10:13].copy(), state[13:16].copy()
        raise RuntimeError(
            "Could not extract position, quaternion, world velocity, and world angular velocity from CtrlAviary."
        )

    def observe(self) -> SimulationObservation:
        if self._env is None or self._closed:
            raise RuntimeError("Adapter is not running; call reset() first.")
        position, quaternion, velocity_world, omega_world = self._read_raw_state()
        rotation = quaternion_xyzw_to_rotation(quaternion)
        omega_body = rotation.T @ omega_world
        return SimulationObservation(
            time_s=self._time_s,
            rotation_body_to_world=rotation,
            body_rates_rad_s=omega_body,
            position_world_m=position,
            velocity_world_m_s=velocity_world,
        )

    def step(self, motor_rpm: Array, *, duration_s: float) -> SimulationObservation:
        if self._env is None or self._closed:
            raise RuntimeError("Adapter is not running; call reset() first.")
        rpm = _finite_vector(motor_rpm, 4, "motor_rpm")
        if np.any(rpm < 0.0):
            raise ValueError("motor_rpm cannot contain negative values.")
        if np.any(rpm > self.mixer.max_rpm + 1e-8):
            raise ValueError("motor_rpm exceeds the simulator's declared maximum; clipping must be logged upstream.")
        duration = float(duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be positive and finite.")
        n_steps_float = duration / self.control_timestep_s
        n_steps = int(round(n_steps_float))
        if n_steps < 1 or not np.isclose(n_steps_float, n_steps, atol=1e-8, rtol=0.0):
            raise ValueError(
                f"duration_s must be an integer multiple of control timestep {self.control_timestep_s:.9g} s."
            )
        action = rpm.reshape(1, 4).astype(np.float32)
        for _ in range(n_steps):
            step_result = self._env.step(action)
            if isinstance(step_result, tuple) and len(step_result) >= 1:
                self._last_observation = step_result[0]
            self._time_s += self.control_timestep_s
        return self.observe()

    def allocate_model_equivalent_torque(
        self,
        model_equivalent_torque_nm: Array,
        *,
        collective_thrust_n: float | None = None,
    ) -> MappedTorqueCommand:
        """Map learned-model torque to actual CF2X torque and allocate four motors."""
        if self._env is None or self._closed or self._mixer is None:
            raise RuntimeError("Adapter is not initialized; call reset() first.")
        tau_model = _finite_vector(model_equivalent_torque_nm, 3, "model_equivalent_torque_nm")
        tau_physical = map_model_equivalent_torque_to_physical(
            tau_model,
            self.simulator_inertia_kg_m2,
            self.config.identified_model_inertia_kg_m2,
        )
        if collective_thrust_n is None:
            if self.config.hover_compensation:
                observation = self.observe()
                vertical_projection = float(observation.rotation_body_to_world[2, 2])
                denominator = max(
                    vertical_projection,
                    self.config.minimum_vertical_projection_for_hover_compensation,
                )
                collective = self.mass_kg * self.gravity_m_s2 / denominator
            else:
                collective = self.mass_kg * self.gravity_m_s2
        else:
            collective = float(collective_thrust_n)
        if not np.isfinite(collective) or collective < 0.0:
            raise ValueError("collective_thrust_n must be finite and non-negative.")
        mixer_output = self.mixer.allocate(tau_physical, collective)
        return MappedTorqueCommand(
            model_equivalent_torque_nm=tau_model.copy(),
            requested_physical_torque_nm=tau_physical.copy(),
            mixer_output=mixer_output,
            collective_thrust_requested_n=float(collective),
        )

    def step_model_equivalent_torque(
        self,
        model_equivalent_torque_nm: Array,
        *,
        duration_s: float,
        collective_thrust_n: float | None = None,
    ) -> tuple[SimulationObservation, MappedTorqueCommand]:
        command = self.allocate_model_equivalent_torque(
            model_equivalent_torque_nm,
            collective_thrust_n=collective_thrust_n,
        )
        observation = self.step(command.mixer_output.motor_rpm, duration_s=duration_s)
        return observation, command

    def compatibility_report(self) -> dict[str, Any]:
        """Return the runtime vehicle/input-map facts to write into run manifests."""
        if self._env is None or self._mixer is None:
            raise RuntimeError("Adapter is not initialized; call reset() first.")
        identified = np.asarray(self.config.identified_model_inertia_kg_m2, dtype=float)
        model_torque_limits = np.asarray(self.config.identified_model_torque_limits_nm, dtype=float)
        ratios = self.simulator_inertia_kg_m2 / identified
        bound_checks = []
        # Check both signs of each declared single-axis controller bound at
        # hover collective thrust before the controller is allowed to drive.
        hover = self.mass_kg * self.gravity_m_s2
        for axis, axis_name in enumerate(("roll", "pitch", "yaw")):
            for sign in (-1.0, 1.0):
                model_torque = np.zeros(3, dtype=float)
                model_torque[axis] = sign * model_torque_limits[axis]
                physical_torque = map_model_equivalent_torque_to_physical(
                    model_torque, self.simulator_inertia_kg_m2, identified
                )
                allocation = self._mixer.allocate(physical_torque, hover)
                bound_checks.append({
                    "axis": axis_name,
                    "sign": int(sign),
                    "model_equivalent_bound_nm": float(model_torque[axis]),
                    "requested_physical_torque_nm": physical_torque.tolist(),
                    "allocator_implied_torque_nm": allocation.allocator_implied_body_torque_nm.tolist(),
                    "allocation_residual_nm": (allocation.allocator_implied_body_torque_nm - physical_torque).tolist(),
                    "saturated": bool(allocation.saturated),
                    "saturation_scale": float(allocation.saturation_scale),
                })
        return {
            "backend": self.name,
            "simulator_vehicle": "CF2X stock vehicle; physical parameters read from CtrlAviary at runtime",
            "identified_model_inertia_kg_m2": identified.tolist(),
            "simulator_inertia_kg_m2": self.simulator_inertia_kg_m2.tolist(),
            "inertia_equivalent_input_scale": ratios.tolist(),
            "identified_model_torque_limits_nm": model_torque_limits.tolist(),
            "single_axis_bound_actuator_checks_at_hover": bound_checks,
            "all_single_axis_bounds_feasible": all(not item["saturated"] for item in bound_checks),
            "simulator_mass_kg": self.mass_kg,
            "simulator_arm_length_m": float(self._env.L),
            "simulator_thrust_coefficient_n_per_rpm2": float(self._env.KF),
            "simulator_yaw_moment_coefficient_nm_per_rpm2": float(self._env.KM),
            "simulator_max_rpm": float(self._env.MAX_RPM),
            "mixer_physics_mode": self._mixer.physics_mode,
            "gravity_m_s2": self.gravity_m_s2,
            "input_mapping": "tau_physical = diag(J_simulator / J_identified) @ tau_model_equivalent",
            "mixer_sign_convention": (
                "CF2X roll/pitch/yaw motor signs are checked in the live smoke test using measured body-rate increments. "
                "The allocator_implied_torque_nm fields are predictions from the mixer matrix, not direct torque telemetry."
            ),
            "warning": (
                "The map matches the input-induced angular-acceleration term for diagonal inertia models. "
                "It does not make the physical plants identical; report remaining model/plant differences. "
                "The bounds check covers each single axis independently, not every simultaneous control corner. "
                "Runtime measured-response sign checks are required before this adapter can pass acceptance."
            ),
        }

    def close(self) -> None:
        env, self._env = self._env, None
        self._mixer = None
        if env is not None:
            try:
                env.close()
            finally:
                self._closed = True

