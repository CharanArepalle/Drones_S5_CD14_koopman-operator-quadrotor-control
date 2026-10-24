"""MuJoCo free-flight quadrotor adapter with CF2X-consistent parameters.

The adapter implements the project-wide QuadrotorBackend contract.  Motor RPM
commands are converted to thrust using the CF2X KF/KM constants; the resultant
world-frame force and body-generated moment are applied to MuJoCo's free body.
The model uses explicit mass and diagonal inertia and is intentionally
transparent so its differences from gym-pybullet-drones remain reportable.

MuJoCo is imported lazily so unit tests can validate model construction and
coordinate helpers even when the optional simulator package is not installed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from src.quadrotor_mixer import CF2XMixer
from src.simulation_contract import SimulationObservation

Array = np.ndarray


@dataclass(frozen=True)
class MuJoCoAdapterConfig:
    physics_frequency_hz: int = 240
    control_frequency_hz: int = 20
    gravity_m_s2: float = 9.81
    mass_kg: float = 0.027
    inertia_kg_m2: tuple[float, float, float] = (1.4e-5, 1.4e-5, 2.17e-5)
    arm_length_m: float = 0.0397
    thrust_coefficient_n_per_rpm2: float = 3.16e-10
    yaw_moment_coefficient_nm_per_rpm2: float = 7.94e-12
    thrust_to_weight_ratio: float = 2.25
    target_position_world_m: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def validate(self) -> None:
        if self.physics_frequency_hz <= 0 or self.control_frequency_hz <= 0:
            raise ValueError("Physics and control frequencies must be positive integers.")
        if self.physics_frequency_hz % self.control_frequency_hz:
            raise ValueError("Physics frequency must be divisible by control frequency.")
        for name in (
            "gravity_m_s2", "mass_kg", "arm_length_m",
            "thrust_coefficient_n_per_rpm2", "yaw_moment_coefficient_nm_per_rpm2",
            "thrust_to_weight_ratio",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        inertia = np.asarray(self.inertia_kg_m2, dtype=float)
        target = np.asarray(self.target_position_world_m, dtype=float)
        if inertia.shape != (3,) or not np.all(np.isfinite(inertia)) or np.any(inertia <= 0):
            raise ValueError("inertia_kg_m2 must contain three positive finite values.")
        if target.shape != (3,) or not np.all(np.isfinite(target)) or target[2] <= 0:
            raise ValueError("target_position_world_m must be a finite 3-vector above ground.")

    @property
    def physics_timestep_s(self) -> float:
        return 1.0 / float(self.physics_frequency_hz)

    @property
    def control_period_s(self) -> float:
        return 1.0 / float(self.control_frequency_hz)

    @property
    def substeps_per_control(self) -> int:
        return self.physics_frequency_hz // self.control_frequency_hz

    @property
    def max_rpm(self) -> float:
        # Same declared thrust-to-weight ratio used by the CF2X model.
        maximum_per_motor_thrust = (
            self.thrust_to_weight_ratio * self.mass_kg * self.gravity_m_s2 / 4.0
        )
        return math.sqrt(maximum_per_motor_thrust / self.thrust_coefficient_n_per_rpm2)


def _fmt(value: float) -> str:
    return f"{float(value):.8f}"


def _ring_geoms(
    *, name_prefix: str, x: float, y: float, z: float, radius: float,
    segments: int, rgba: tuple[float, float, float, float], thickness: float,
) -> str:
    parts: list[str] = []
    for index in range(segments):
        a0 = 2.0 * math.pi * index / segments
        a1 = 2.0 * math.pi * (index + 1) / segments
        p0 = (x + radius * math.cos(a0), y + radius * math.sin(a0), z)
        p1 = (x + radius * math.cos(a1), y + radius * math.sin(a1), z)
        coords = " ".join(_fmt(v) for v in (*p0, *p1))
        color = " ".join(_fmt(v) for v in rgba)
        parts.append(
            f'<geom name="{name_prefix}_{index:02d}" type="capsule" fromto="{coords}" '
            f'size="{_fmt(thickness)}" rgba="{color}" contype="0" conaffinity="0" '
            'group="2"/>'
        )
    return "\n      ".join(parts)


def build_cf2x_mjcf(config: MuJoCoAdapterConfig | None = None) -> str:
    """Build an MJCF model string without importing the MuJoCo runtime.

    The upper gold ring lies exactly at the requested world-frame hover altitude;
    the lower, dim ring is a floor projection only. Four motor visual markers
    and red/green/blue body axes make the vehicle orientation inspectable.
    """
    cfg = config or MuJoCoAdapterConfig()
    cfg.validate()
    tx, ty, tz = cfg.target_position_world_m
    upper_ring = _ring_geoms(
        name_prefix="hover_target_ring", x=tx, y=ty, z=tz, radius=0.135,
        segments=32, rgba=(1.0, 0.68, 0.06, 1.0), thickness=0.0045,
    )
    floor_ring = _ring_geoms(
        name_prefix="floor_projection_ring", x=tx, y=ty, z=0.018, radius=0.135,
        segments=32, rgba=(0.68, 0.48, 0.12, 0.56), thickness=0.003,
    )
    guide = []
    guide_x = tx + 0.18
    for index in range(1, 13):
        z = 0.04 + (tz - 0.08) * index / 13.0
        guide.append(
            f'<geom name="target_altitude_guide_{index:02d}" type="sphere" '
            f'pos="{_fmt(guide_x)} {_fmt(ty)} {_fmt(z)}" size="0.004" '
            'rgba="1.0 0.68 0.06 0.72" contype="0" conaffinity="0" group="2"/>'
        )
    guide_xml = "\n      ".join(guide)

    arm = cfg.arm_length_m / math.sqrt(2.0)
    rotor_xy = ((arm, -arm), (-arm, -arm), (-arm, arm), (arm, arm))
    motor_rgba = (
        (0.19, 0.53, 0.98, 1.0), (0.96, 0.43, 0.15, 1.0),
        (0.18, 0.78, 0.48, 1.0), (0.76, 0.34, 0.88, 1.0),
    )
    arms_xml = []
    motors_xml = []
    for index, ((x, y), rgba) in enumerate(zip(rotor_xy, motor_rgba, strict=True)):
        color = " ".join(_fmt(v) for v in rgba)
        arms_xml.append(
            f'<geom name="arm_{index}" type="capsule" fromto="0 0 0 {_fmt(x)} {_fmt(y)} 0" '
            'size="0.0042" rgba="0.30 0.34 0.40 1" contype="0" conaffinity="0" group="1"/>'
        )
        motors_xml.append(
            f'<geom name="motor_pod_{index}" type="cylinder" pos="{_fmt(x)} {_fmt(y)} 0.002" '
            'size="0.018 0.005" rgba="0.12 0.15 0.19 1" contype="0" conaffinity="0" group="1"/>\n'
            f'<geom name="prop_disk_{index}" type="cylinder" pos="{_fmt(x)} {_fmt(y)} 0.008" '
            'size="0.022 0.0016" rgba="0.78 0.82 0.87 0.72" contype="0" conaffinity="0" group="1"/>\n'
            f'<geom name="motor_hub_{index}" type="sphere" pos="{_fmt(x)} {_fmt(y)} 0.010" '
            'size="0.006" '
            f'rgba="{color}" contype="0" conaffinity="0" group="1"/>'
        )

    ix, iy, iz = cfg.inertia_kg_m2
    inertia = " ".join(_fmt(v) for v in (ix, iy, iz))
    target_label_y = ty - 0.20
    return f'''<mujoco model="CF2X Koopman controller comparison">
  <compiler angle="radian" coordinate="local" inertiafromgeom="false" autolimits="true"/>
  <option timestep="{_fmt(cfg.physics_timestep_s)}" gravity="0 0 -{_fmt(cfg.gravity_m_s2)}"
          integrator="RK4" solver="Newton" iterations="60" tolerance="1e-10"/>
  <visual>
    <global offwidth="1600" offheight="1000" azimuth="135" elevation="-23"/>
    <quality shadowsize="2048" offsamples="4"/>
    <map znear="0.01" zfar="20" fogstart="5" fogend="12"/>
    <rgba haze="0.42 0.50 0.60 1"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.10 0.18 0.30" rgb2="0.72 0.82 0.92" width="512" height="512"/>
    <texture name="floor_checker" type="2d" builtin="checker" rgb1="0.18 0.22 0.27" rgb2="0.24 0.28 0.33" width="512" height="512"/>
    <material name="floor_material" texture="floor_checker" texrepeat="7 7" reflectance="0.07"/>
  </asset>
  <default>
    <geom friction="0.8 0.02 0.001" solref="0.008 1" solimp="0.95 0.99 0.001"/>
  </default>
  <worldbody>
    <light name="key_light" pos="-2 -3 5" dir="0.25 0.35 -1" directional="true" diffuse="0.88 0.88 0.88" specular="0.20 0.20 0.20"/>
    <light name="fill_light" pos="3 2 4" dir="-0.25 -0.15 -1" directional="true" diffuse="0.48 0.52 0.60" specular="0.05 0.05 0.05"/>
    <geom name="ground" type="plane" size="6 6 0.1" material="floor_material" contype="1" conaffinity="1" group="0"/>
    {upper_ring}
    {floor_ring}
    {guide_xml}
    <geom name="hover_target_center" type="sphere" pos="{_fmt(tx)} {_fmt(ty)} {_fmt(tz)}" size="0.010"
          rgba="1.0 0.79 0.20 1" contype="0" conaffinity="0" group="2"/>
    <geom name="target_height_label_anchor" type="sphere" pos="{_fmt(tx + 0.18)} {_fmt(target_label_y)} {_fmt(tz + 0.08)}" size="0.001"
          rgba="1 1 1 0" contype="0" conaffinity="0" group="2"/>
    <body name="quadrotor" pos="0 0 1">
      <freejoint name="quadrotor_free"/>
      <inertial pos="0 0 0" mass="{_fmt(cfg.mass_kg)}" diaginertia="{inertia}"/>
      <geom name="central_body_collision" type="box" size="0.026 0.026 0.009" rgba="0.10 0.16 0.23 1"
            contype="1" conaffinity="1" group="1"/>
      <geom name="top_shell" type="box" pos="0 0 0.009" size="0.020 0.020 0.006" rgba="0.20 0.29 0.39 1"
            contype="0" conaffinity="0" group="1"/>
      {''.join(arms_xml)}
      {''.join(motors_xml)}
      <geom name="axis_x" type="capsule" fromto="0 0 0 0.145 0 0" size="0.0035" rgba="0.97 0.18 0.16 1"
            contype="0" conaffinity="0" group="1"/>
      <geom name="axis_y" type="capsule" fromto="0 0 0 0 0.145 0" size="0.0035" rgba="0.20 0.84 0.35 1"
            contype="0" conaffinity="0" group="1"/>
      <geom name="axis_z" type="capsule" fromto="0 0 0 0 0 0.145" size="0.0035" rgba="0.18 0.49 1.00 1"
            contype="0" conaffinity="0" group="1"/>
      <geom name="axis_x_tip" type="sphere" pos="0.145 0 0" size="0.007" rgba="0.97 0.18 0.16 1" contype="0" conaffinity="0" group="1"/>
      <geom name="axis_y_tip" type="sphere" pos="0 0.145 0" size="0.007" rgba="0.20 0.84 0.35 1" contype="0" conaffinity="0" group="1"/>
      <geom name="axis_z_tip" type="sphere" pos="0 0 0.145" size="0.007" rgba="0.18 0.49 1.00 1" contype="0" conaffinity="0" group="1"/>
    </body>
  </worldbody>
</mujoco>'''


def compute_body_wrench_from_rpm(
    motor_rpm: Array, config: MuJoCoAdapterConfig | None = None,
) -> tuple[Array, Array]:
    """Return resultant thrust force and rotor torque in body coordinates.

    The torque signs are deliberately aligned with the shared CF2XMixer motor
    ordering: X-configuration motor positions produce roll/pitch moments by
    ``r cross thrust`` and alternating motor drag produces yaw moment.
    """
    cfg = config or MuJoCoAdapterConfig()
    cfg.validate()
    rpm = np.asarray(motor_rpm, dtype=float).reshape(-1)
    if rpm.shape != (4,) or not np.all(np.isfinite(rpm)) or np.any(rpm < 0.0):
        raise ValueError("motor_rpm must contain four finite non-negative values.")
    if np.any(rpm > cfg.max_rpm + 1e-6):
        raise ValueError("A motor command exceeds the configured CF2X RPM limit.")
    thrusts = cfg.thrust_coefficient_n_per_rpm2 * rpm**2
    arm = cfg.arm_length_m / math.sqrt(2.0)
    rotor_positions = np.array([
        [arm, -arm, 0.0], [-arm, -arm, 0.0],
        [-arm, arm, 0.0], [arm, arm, 0.0],
    ], dtype=float)
    force_body = np.array([0.0, 0.0, float(np.sum(thrusts))], dtype=float)
    rotor_forces = np.column_stack([np.zeros(4), np.zeros(4), thrusts])
    torque_body = np.sum(np.cross(rotor_positions, rotor_forces), axis=0)
    yaw_direction = np.array([-1.0, 1.0, -1.0, 1.0], dtype=float)
    torque_body[2] += float(np.sum(
        yaw_direction * cfg.yaw_moment_coefficient_nm_per_rpm2 * rpm**2
    ))
    if not np.all(np.isfinite(force_body)) or not np.all(np.isfinite(torque_body)):
        raise FloatingPointError("Motor commands generated a non-finite wrench.")
    return force_body, torque_body


def rotation_to_euler_zyx(rotation_body_to_world: Array) -> Array:
    """Return roll, pitch, yaw using R = Rz(yaw) Ry(pitch) Rx(roll)."""
    rotation = np.asarray(rotation_body_to_world, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_body_to_world must be a finite 3x3 matrix.")
    pitch = float(np.arctan2(-rotation[2, 0], np.hypot(rotation[0, 0], rotation[1, 0])))
    roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return np.array([roll, pitch, yaw], dtype=float)


def euler_zyx_to_quaternion_wxyz(euler_zyx_rad: Array) -> Array:
    """Convert roll/pitch/yaw to MuJoCo's scalar-first quaternion convention."""
    roll, pitch, yaw = np.asarray(euler_zyx_rad, dtype=float).reshape(3)
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    quaternion = np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


class MuJoCoQuadrotorAdapter:
    """CF2X-like free-flight vehicle driven by four motor RPM commands."""

    name = "MuJoCo/CF2X-parameterized"

    def __init__(self, config: MuJoCoAdapterConfig | None = None) -> None:
        self.config = config or MuJoCoAdapterConfig()
        self.config.validate()
        try:
            import mujoco  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "MuJoCo is not installed in the active Python environment. Activate the Drones_Project .venv, "
                "then run `python -m pip install mujoco` and verify with `python -c \"import mujoco; print(mujoco.__version__)\"`."
            ) from exc
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(build_cf2x_mjcf(self.config))
        self.data = mujoco.MjData(self.model)
        self.body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quadrotor"))
        if self.body_id <= 0:
            raise RuntimeError("The MJCF model did not produce the expected quadrotor body.")
        self.mixer = CF2XMixer(
            mass_kg=self.config.mass_kg,
            arm_length_m=self.config.arm_length_m,
            thrust_coefficient_n_per_rpm2=self.config.thrust_coefficient_n_per_rpm2,
            yaw_moment_coefficient_nm_per_rpm2=self.config.yaw_moment_coefficient_nm_per_rpm2,
            max_rpm=self.config.max_rpm,
            gravity_m_s2=self.config.gravity_m_s2,
            physics_mode="DYN",
        )
        arm = self.config.arm_length_m / math.sqrt(2.0)
        self.rotor_positions_body_m = np.array([
            [arm, -arm, 0.0], [-arm, -arm, 0.0],
            [-arm, arm, 0.0], [arm, arm, 0.0],
        ], dtype=float)
        self.yaw_direction = np.array([-1.0, 1.0, -1.0, 1.0], dtype=float)
        self._viewer: Any | None = None
        self._label_warning_printed = False
        self._last_applied_wrench_world = np.zeros(6, dtype=float)

    @property
    def mass_kg(self) -> float:
        return self.config.mass_kg

    @property
    def gravity_m_s2(self) -> float:
        return self.config.gravity_m_s2

    @property
    def simulator_inertia_kg_m2(self) -> Array:
        return np.asarray(self.config.inertia_kg_m2, dtype=float).copy()

    def reset(
        self,
        *,
        initial_position_world_m: Array = np.array([0.0, 0.0, 1.0]),
        initial_euler_zyx_rad: Array = np.zeros(3),
    ) -> SimulationObservation:
        position = np.asarray(initial_position_world_m, dtype=float).reshape(-1)
        euler = np.asarray(initial_euler_zyx_rad, dtype=float).reshape(-1)
        if position.shape != (3,) or euler.shape != (3,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(euler)):
            raise ValueError("Initial position and attitude must be finite 3-vectors.")
        if position[2] <= 0.0:
            raise ValueError("Initial altitude must be above ground.")
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = position
        self.data.qpos[3:7] = euler_zyx_to_quaternion_wxyz(euler)
        self.data.qvel[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self) -> SimulationObservation:
        mujoco = self.mujoco
        rotation = np.asarray(self.data.xmat[self.body_id], dtype=float).reshape(3, 3).copy()
        angular_local = np.zeros(6, dtype=float)
        velocity_world = np.zeros(6, dtype=float)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.body_id, angular_local, 1,
        )
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY,
            self.body_id, velocity_world, 0,
        )
        return SimulationObservation(
            time_s=float(self.data.time),
            rotation_body_to_world=rotation,
            body_rates_rad_s=np.asarray(angular_local[:3], dtype=float).copy(),
            position_world_m=np.asarray(self.data.xpos[self.body_id], dtype=float).copy(),
            velocity_world_m_s=np.asarray(velocity_world[3:], dtype=float).copy(),
        )

    def _apply_motor_wrench(self, motor_rpm: Array) -> None:
        rpm = np.asarray(motor_rpm, dtype=float).reshape(-1)
        if rpm.shape != (4,) or not np.all(np.isfinite(rpm)) or np.any(rpm < 0.0):
            raise ValueError("motor_rpm must contain four finite non-negative values.")
        if np.any(rpm > self.mixer.max_rpm + 1e-6):
            raise ValueError("A motor command exceeds the CF2X RPM limit.")
        force_body, torque_body = compute_body_wrench_from_rpm(rpm, self.config)
        rotation = np.asarray(self.data.xmat[self.body_id], dtype=float).reshape(3, 3)
        force_world = rotation @ force_body
        torque_world = rotation @ torque_body
        # MuJoCo's xfrc_applied is a world-frame wrench at the body's COM.
        self.data.xfrc_applied[:] = 0.0
        self.data.xfrc_applied[self.body_id, 0:3] = force_world
        self.data.xfrc_applied[self.body_id, 3:6] = torque_world
        self._last_applied_wrench_world[:] = np.concatenate([force_world, torque_world])

    def step(self, motor_rpm: Array, *, duration_s: float) -> SimulationObservation:
        duration = float(duration_s)
        if not np.isfinite(duration) or duration <= 0.0:
            raise ValueError("duration_s must be positive and finite.")
        steps_float = duration / self.config.physics_timestep_s
        if not np.isclose(steps_float, round(steps_float), atol=1e-8, rtol=0.0):
            raise ValueError("duration_s must be a multiple of the configured MuJoCo timestep.")
        for _ in range(int(round(steps_float))):
            self._apply_motor_wrench(motor_rpm)
            self.mujoco.mj_step(self.model, self.data)
        self.data.xfrc_applied[:] = 0.0
        return self.observe()

    def open_viewer(self) -> Any:
        if self._viewer is not None:
            return self._viewer
        try:
            import mujoco.viewer  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("The MuJoCo Python viewer could not be imported. Reinstall the `mujoco` package in this venv.") from exc
        self._viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False,
        )
        with self._viewer.lock():
            camera = self._viewer.cam
            camera.type = self.mujoco.mjtCamera.mjCAMERA_FREE
            camera.lookat[:] = np.array([0.0, 0.0, 0.60])
            camera.distance = 2.05
            camera.azimuth = 135.0
            camera.elevation = -23.0
            # Keep built-in diagnostic clutter off; our labels explain the task.
            try:
                for flag_name in (
                    "mjVIS_CONTACTPOINT",
                    "mjVIS_CONTACTFORCE",
                    "mjVIS_JOINT",
                    # The hovering vehicle's ground shadow looked like a second
                    # drone and obscured the distinction between the target's
                    # floor projection and the real vehicle position.
                    "mjVIS_SHADOW",
                ):
                    flag = getattr(self.mujoco.mjtVisFlag, flag_name, None)
                    if flag is not None:
                        self._viewer.opt.flags[flag] = 0
            except Exception:
                pass
        self._viewer.sync()
        return self._viewer

    def update_presentation(
        self,
        controller: str,
        observation: SimulationObservation,
        *,
        desired_rotation_body_to_world: Array,
        target_position_world_m: Array,
        attitude_error_deg: float,
        position_error_m: float,
        elapsed_s: float,
        duration_s: float,
        final_metrics: dict[str, Any] | None = None,
    ) -> None:
        """Present a screen-fixed HUD plus the world-anchored hover-target label.

        Controller/status text uses the viewer's screen overlay, not 3-D world
        coordinates. This keeps the active controller and mission explanation
        visible regardless of the camera projection or target altitude.
        """
        if self._viewer is None or not self._viewer.is_running():
            return

        viewer = self._viewer
        mujoco = self.mujoco
        target = np.asarray(target_position_world_m, dtype=float).reshape(3)
        position = np.asarray(observation.position_world_m, dtype=float).reshape(3)
        status = (
            "HOLDING TARGET"
            if position_error_m < 0.05 and attitude_error_deg < 2.0
            else "RECOVERING / CORRECTING"
        )

        if final_metrics is None:
            left_lines = [
                f"MUJOCO | ACTIVE CONTROLLER: {controller}",
                "MISSION: recover attitude + hold 3-D hover position",
                f"STATUS: {status}",
                f"ATTITUDE TRACKING ERROR: {attitude_error_deg:5.2f} deg",
                f"DISTANCE TO TARGET: {position_error_m:5.3f} m",
                f"SIMULATION TIME: {elapsed_s:4.1f} / {duration_s:.0f} s",
            ]
            right_lines = [
                "TARGET: (0, 0, 1.0) m",
                "BRIGHT GOLD RING = exact 3-D hover target",
                "DIM LOWER RING = ground projection only",
                "BODY AXES: red X | green Y | blue Z",
            ]
        else:
            settling = final_metrics.get("settling_time_s")
            settle_text = f"{float(settling):.2f} s" if settling is not None else "not reached"
            left_lines = [
                f"RUN COMPLETE | {controller}",
                f"ATTITUDE RMSE: {float(final_metrics['attitude_tracking_rmse_deg']):.3f} deg",
                f"SETTLING WITHIN 2 DEG: {settle_text}",
                f"3-D POSITION RMSE: {float(final_metrics['position_rmse_3d_m']):.3f} m",
                f"FINAL DISTANCE TO TARGET: {position_error_m:.4f} m",
            ]
            right_lines = [
                "TARGET: (0, 0, 1.0) m",
                "GOLD RING = 3-D SETPOINT",
                "LOWER RING = FLOOR PROJECTION",
                "Press Enter in the terminal for the next controller",
            ]

        # MuJoCo's passive viewer supports screen-space text overlays. Use them
        # for critical explanatory text so camera changes cannot clip it.
        try:
            set_texts = getattr(viewer, "set_texts", None)
            if callable(set_texts):
                font = mujoco.mjtFontScale.mjFONTSCALE_150
                grid = mujoco.mjtGridPos.mjGRID_TOPLEFT
                set_texts([(font, grid, "\n".join(left_lines), "\n".join(right_lines))])
            else:
                # Keep a compatible fallback for older passive-viewer APIs.
                print_once = not self._label_warning_printed
                if print_once:
                    print("\n[MuJoCo HUD warning] Screen text overlay API is unavailable; using 3-D labels.")
                    self._label_warning_printed = True

            # Keep just one 3-D text anchored beside the upper gold ring. The
            # rest of the UI is screen-fixed and always on screen.
            with viewer.lock():
                scene = viewer.user_scn
                if scene is None:
                    return
                scene.ngeom = 0
                identity = np.eye(3, dtype=float).reshape(-1)
                geom = scene.geoms[0]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LABEL,
                    np.array([0.0, 0.0, 0.0], dtype=float),
                    np.array([target[0] + 0.16, target[1] - 0.20, target[2] + 0.09], dtype=float),
                    identity,
                    np.array([1.0, 0.78, 0.20, 1.0], dtype=float),
                )
                geom.label = "HOVER TARGET"
                geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
                scene.ngeom = 1
            viewer.sync()
        except Exception as exc:
            if not self._label_warning_printed:
                print(f"\n[MuJoCo HUD warning] Presentation overlay could not be refreshed: {exc}")
                print("The simulation metrics remain in the terminal and controller_validation.png.")
                self._label_warning_printed = True

    def sync_viewer(self) -> None:
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            finally:
                self._viewer = None
