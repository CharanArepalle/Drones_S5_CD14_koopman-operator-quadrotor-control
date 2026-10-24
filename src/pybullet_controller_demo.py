"""Live CF2X controller demonstration with runtime path tracking and noisy feedback.

The drone is simulated by gym-pybullet-drones. PID, Koopman-LQR and Koopman-MPC
use the existing saved order-5 attitude model where applicable and command the
same CF2X adapter/motor allocator. Optional analytic 3-D references are generated
at runtime, optional seeded sensor-measurement noise affects controller feedback
only, and three live evaluation plots update during each episode. The final
metrics are computed from true live simulator observations, not model predictions.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import hashlib
from datetime import datetime
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from src.attitude_core import attitude_lift
from src.simulation_contract import SimulationObservation
from src.koopman_mpc import KoopmanMPC
from src.position_hold import (
    PositionHoldConfig,
    desired_attitude_from_force as _desired_attitude_from_force,
)
from src.simulators.pybullet_adapter import (
    PyBulletAdapterConfig,
    PyBulletDronesAdapter,
    rotation_to_euler_zyx,
)

Array = np.ndarray
CONTROLLERS = ("PID", "Koopman-LQR", "Koopman-MPC")
MODEL_TORQUE_LIMITS_NM = np.array([0.12, 0.12, 0.06], dtype=float)


@dataclass(frozen=True)
class ControllerDemoConfig:
    """Explicit, reportable settings for the live attitude demonstration."""
    duration_s: float = 20.0
    initial_position_world_m: tuple[float, float, float] = (0.0, 0.0, 1.0)
    initial_attitude_deg: tuple[float, float, float] = (10.0, 8.0, 6.0)
    selected_order: int = 5
    mpc_horizon: int = 20
    mpc_q_rotation: float = 5.0
    mpc_q_rate: float = 2.0
    mpc_q_higher: float = 0.2
    mpc_r_weights: tuple[float, float, float] = (0.01, 0.01, 0.008)
    settling_tolerance_deg: float = 2.0
    safety_geodesic_limit_deg: float = 100.0
    safety_min_altitude_m: float = 0.15
    safety_max_altitude_m: float = 5.0
    safety_max_body_rate_rad_s: float = 20.0
    # Higher derivative damping and smaller integral gains address the underdamped PID trace.
    pid_kp: tuple[float, float, float] = (0.35, 0.35, 0.18)
    pid_ki: tuple[float, float, float] = (0.002, 0.002, 0.001)
    pid_kd: tuple[float, float, float] = (0.16, 0.16, 0.12)
    pid_integral_limit: tuple[float, float, float] = (2.0, 2.0, 2.0)
    position_hold_enabled: bool = True
    position_kp: tuple[float, float, float] = (1.4, 1.4, 2.0)
    position_kd: tuple[float, float, float] = (1.8, 1.8, 1.7)
    max_lateral_acceleration_m_s2: float = 1.6
    max_vertical_acceleration_m_s2: float = 1.5
    max_tilt_deg: float = 12.0
    camera_follow: bool = True
    camera_distance_m: float = 0.55
    camera_yaw_deg: float = 45.0
    camera_pitch_deg: float = -24.0
    # Runtime reference trajectory. The trained Koopman model remains an attitude model.
    trajectory: str = "hover"  # hover | figure8
    trajectory_radius_m: float = 0.25
    trajectory_vertical_amplitude_m: float = 0.08
    trajectory_period_s: float = 12.0
    # Reproducible measurement noise is applied only to controller feedback.
    sensor_noise_profile: str = "none"  # none | low | moderate
    sensor_noise_seed: int = 20261024

    def validate(self, sample_period_s: float) -> None:
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive and finite.")
        steps_float = self.duration_s / sample_period_s
        if not np.isclose(steps_float, round(steps_float), atol=1e-8, rtol=0.0):
            raise ValueError(
                f"duration_s ({self.duration_s:g}) must be a multiple of control period "
                f"({sample_period_s:g} s)."
            )
        if self.selected_order < 2 or self.mpc_horizon < 1:
            raise ValueError("selected_order must be >= 2 and mpc_horizon must be >= 1.")
        if len(self.initial_position_world_m) != 3 or len(self.initial_attitude_deg) != 3:
            raise ValueError("Initial position and attitude must each have three components.")
        if self.initial_position_world_m[2] <= 0.0:
            raise ValueError("The initial altitude must be above ground.")
        if self.settling_tolerance_deg <= 0.0 or self.safety_geodesic_limit_deg <= 0.0:
            raise ValueError("Attitude thresholds must be positive.")
        for name in ("pid_kp", "pid_ki", "pid_kd", "pid_integral_limit"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values.")
        if np.any(np.asarray(self.pid_kp) < 0) or np.any(np.asarray(self.pid_ki) < 0) or np.any(np.asarray(self.pid_kd) < 0):
            raise ValueError("PID gains must be non-negative.")
        if np.any(np.asarray(self.pid_integral_limit) <= 0):
            raise ValueError("PID integral limits must be positive.")
        for name in ("position_kp", "position_kd"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"{name} must contain three finite, non-negative values.")
        if self.max_lateral_acceleration_m_s2 <= 0 or self.max_vertical_acceleration_m_s2 <= 0:
            raise ValueError("Position-controller acceleration limits must be positive.")
        if not (0.0 < self.max_tilt_deg <= 30.0):
            raise ValueError("max_tilt_deg must lie in (0, 30] for the demo.")
        if len(self.mpc_r_weights) != 3 or not np.all(np.isfinite(self.mpc_r_weights)) or np.any(np.asarray(self.mpc_r_weights) <= 0):
            raise ValueError("mpc_r_weights must contain three positive finite values.")
        if any(not np.isfinite(v) or v < 0 for v in (self.mpc_q_rotation, self.mpc_q_rate, self.mpc_q_higher)):
            raise ValueError("MPC state weights must be finite and non-negative.")
        if self.camera_distance_m <= 0.0:
            raise ValueError("camera_distance_m must be positive.")
        if self.trajectory not in {"hover", "figure8"}:
            raise ValueError("trajectory must be 'hover' or 'figure8'.")
        if not np.isfinite(self.trajectory_radius_m) or not (0.05 <= self.trajectory_radius_m <= 0.50):
            raise ValueError("trajectory_radius_m must be between 0.05 and 0.50 m.")
        if not np.isfinite(self.trajectory_vertical_amplitude_m) or not (0.0 <= self.trajectory_vertical_amplitude_m <= 0.15):
            raise ValueError("trajectory_vertical_amplitude_m must be between 0 and 0.15 m.")
        if not np.isfinite(self.trajectory_period_s) or self.trajectory_period_s < 6.0:
            raise ValueError("trajectory_period_s must be at least 6 seconds for safe reference motion.")
        if self.sensor_noise_profile not in {"none", "low", "moderate"}:
            raise ValueError("sensor_noise_profile must be none, low, or moderate.")
        if not isinstance(self.sensor_noise_seed, (int, np.integer)):
            raise ValueError("sensor_noise_seed must be an integer.")


def wrap_angle(angle: Array | float) -> Array | float:
    """Wrap radians to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def geodesic_attitude_error_deg(rotation_body_to_world: Array) -> float:
    """Geodesic distance from level attitude, in degrees."""
    rotation = np.asarray(rotation_body_to_world, dtype=float)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix.")
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cosine)))


def angular_velocity_between_rotations(
    previous_rotation_body_to_world: Array,
    current_rotation_body_to_world: Array,
    dt: float,
) -> Array:
    """Estimate desired body angular velocity from consecutive desired rotations.

    For a body-to-world rotation R, the small-time relation is
    R(t + dt) ~= R(t) exp([omega_body]x dt).  The log map of
    R(t).T @ R(t + dt) therefore estimates angular velocity in the body frame.
    The estimate is re-expressed in the current desired body frame.
    """
    previous = np.asarray(previous_rotation_body_to_world, dtype=float)
    current = np.asarray(current_rotation_body_to_world, dtype=float)
    if previous.shape != (3, 3) or current.shape != (3, 3):
        raise ValueError("Desired rotations must both be 3x3 matrices.")
    if not np.all(np.isfinite(previous)) or not np.all(np.isfinite(current)):
        raise ValueError("Desired rotations must be finite.")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite.")

    delta = previous.T @ current
    cosine = float(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    skew_vector = np.array([
        delta[2, 1] - delta[1, 2],
        delta[0, 2] - delta[2, 0],
        delta[1, 0] - delta[0, 1],
    ], dtype=float)
    if theta < 1e-7:
        omega_previous = 0.5 * skew_vector / dt
    else:
        sine = float(np.sin(theta))
        if abs(sine) < 1e-8:
            # The position outer loop limits commanded tilt to <= 30 degrees,
            # so this branch is only a numerical guard for malformed inputs.
            omega_previous = 0.5 * skew_vector / dt
        else:
            omega_previous = (theta / (2.0 * sine * dt)) * skew_vector
    return delta.T @ omega_previous


def _rotation_from_euler_zyx(euler_zyx_rad: Array) -> Array:
    """Build a body-to-world rotation using roll, pitch, yaw (ZYX convention)."""
    roll, pitch, yaw = np.asarray(euler_zyx_rad, dtype=float).reshape(3)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def trajectory_reference(
    time_s: float,
    initial_position_world_m: Array,
    cfg: ControllerDemoConfig,
) -> tuple[Array, Array, Array]:
    """Generate a smooth reference online; it is not used to fit the Koopman model.

    ``hover`` retains the established static setpoint. ``figure8`` generates a
    bounded three-dimensional lemniscate-like path whose position, velocity and
    acceleration are evaluated analytically at the current simulation time.
    All values are world-frame SI units.
    """
    if not np.isfinite(time_s) or time_s < 0.0:
        raise ValueError("time_s must be finite and non-negative.")
    origin = np.asarray(initial_position_world_m, dtype=float).reshape(-1)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("initial_position_world_m must be a finite 3-vector.")
    if cfg.trajectory == "hover":
        return origin.copy(), np.zeros(3, dtype=float), np.zeros(3, dtype=float)

    omega = 2.0 * np.pi / float(cfg.trajectory_period_s)
    phase = omega * float(time_s)
    radius = float(cfg.trajectory_radius_m)
    vertical = float(cfg.trajectory_vertical_amplitude_m)
    # Starts at the same initial position, uses an eight-shaped XY path, and
    # changes altitude smoothly between z0 and z0 + 2*vertical.
    position = origin + np.array([
        radius * np.sin(phase),
        0.5 * radius * np.sin(2.0 * phase),
        vertical * (1.0 - np.cos(phase)),
    ])
    velocity = np.array([
        radius * omega * np.cos(phase),
        radius * omega * np.cos(2.0 * phase),
        vertical * omega * np.sin(phase),
    ])
    acceleration = np.array([
        -radius * omega**2 * np.sin(phase),
        -2.0 * radius * omega**2 * np.sin(2.0 * phase),
        vertical * omega**2 * np.cos(phase),
    ])
    return position, velocity, acceleration


def trajectory_samples(cfg: ControllerDemoConfig, sample_period_s: float) -> Array:
    """Sample the runtime reference for display; the controller evaluates it online."""
    count = int(round(cfg.duration_s / sample_period_s)) + 1
    times = np.linspace(0.0, cfg.duration_s, count)
    origin = np.asarray(cfg.initial_position_world_m, dtype=float)
    return np.stack([trajectory_reference(float(t), origin, cfg)[0] for t in times])


def sensor_noise_sigmas(profile: str) -> dict[str, float]:
    """Return reproducible synthetic sensor-noise standard deviations (SI units)."""
    profiles = {
        "none": {"position_m": 0.0, "velocity_m_s": 0.0, "attitude_rad": 0.0, "body_rate_rad_s": 0.0},
        "low": {"position_m": 0.005, "velocity_m_s": 0.02,
                "attitude_rad": float(np.deg2rad(0.5)), "body_rate_rad_s": float(np.deg2rad(0.5))},
        "moderate": {"position_m": 0.015, "velocity_m_s": 0.05,
                     "attitude_rad": float(np.deg2rad(1.0)), "body_rate_rad_s": float(np.deg2rad(1.5))},
    }
    if profile not in profiles:
        raise ValueError("sensor-noise profile must be none, low, or moderate.")
    return dict(profiles[profile])


def noisy_controller_observation(
    truth: SimulationObservation,
    profile: str,
    rng: np.random.Generator,
) -> SimulationObservation:
    """Create noisy feedback for the controller without changing ground-truth data.

    The returned observation is used only to compute controls. Metrics and saved
    actual-state traces continue to use the simulator's original observation.
    """
    sigma = sensor_noise_sigmas(profile)
    if profile == "none":
        return truth
    attitude_perturbation = _rotation_from_euler_zyx(
        rng.normal(0.0, sigma["attitude_rad"], size=3)
    )
    rotation_noisy = np.asarray(truth.rotation_body_to_world, dtype=float) @ attitude_perturbation
    return SimulationObservation(
        time_s=truth.time_s,
        rotation_body_to_world=rotation_noisy,
        body_rates_rad_s=np.asarray(truth.body_rates_rad_s, dtype=float)
            + rng.normal(0.0, sigma["body_rate_rad_s"], size=3),
        position_world_m=np.asarray(truth.position_world_m, dtype=float)
            + rng.normal(0.0, sigma["position_m"], size=3),
        velocity_world_m_s=np.asarray(truth.velocity_world_m_s, dtype=float)
            + rng.normal(0.0, sigma["velocity_m_s"], size=3),
    )


class LiveEvaluationDashboard:
    """Three live plots: XY path, attitude error, and 3-D position-error norm."""

    COLORS = {"PID": "tab:blue", "Koopman-LQR": "tab:orange", "Koopman-MPC": "tab:green"}

    def __init__(self, duration_s: float) -> None:
        backend = str(matplotlib.get_backend()).lower()
        non_interactive = {"agg", "pdf", "ps", "svg", "template", "cairo"}
        if backend in non_interactive or "inline" in backend:
            raise RuntimeError(
                f"Matplotlib backend {matplotlib.get_backend()!r} is non-interactive. "
                "Install/enable a desktop GUI backend, or run with --no-live-plots."
            )
        self.duration_s = float(duration_s)
        self.figure, self.axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
        try:
            self.figure.canvas.manager.set_window_title("Live quadrotor tracking dashboard")
        except Exception:
            pass
        self.ax_xy, self.ax_att, self.ax_pos = self.axes
        self.ref_xy, = self.ax_xy.plot([], [], color="goldenrod", linestyle="--", linewidth=2.0, label="requested path")
        self.actual_xy, = self.ax_xy.plot([], [], color="tab:blue", linewidth=1.8, label="measured drone path")
        self.target_xy, = self.ax_xy.plot([], [], marker="*", color="goldenrod", markersize=10, linestyle="None", label="current target")
        self.attitude_line, = self.ax_att.plot([], [], color="tab:red", linewidth=1.8, label="attitude error")
        self.position_line, = self.ax_pos.plot([], [], color="tab:purple", linewidth=1.8, label="3-D position error")
        self.ax_att.axhline(2.0, color="black", linestyle="--", linewidth=1.0, label="2° settling tolerance")
        self.ax_pos.axhline(0.05, color="black", linestyle=":", linewidth=1.0, label="5 cm reference")
        self.ax_xy.set_title("XY trajectory")
        self.ax_xy.set_xlabel("World X (m)"); self.ax_xy.set_ylabel("World Y (m)")
        self.ax_xy.grid(True, alpha=0.28); self.ax_xy.legend(loc="best", fontsize=8)
        self.ax_att.set_title("Attitude tracking")
        self.ax_att.set_xlabel("Time (s)"); self.ax_att.set_ylabel("Error (deg)")
        self.ax_att.set_xlim(0.0, self.duration_s); self.ax_att.grid(True, alpha=0.28)
        self.ax_att.legend(loc="best", fontsize=8)
        self.ax_pos.set_title("Position tracking")
        self.ax_pos.set_xlabel("Time (s)"); self.ax_pos.set_ylabel("3-D error (m)")
        self.ax_pos.set_xlim(0.0, self.duration_s); self.ax_pos.grid(True, alpha=0.28)
        self.ax_pos.legend(loc="best", fontsize=8)
        self._times: list[float] = []
        self._actual_xy_values: list[Array] = []
        self._reference_xy_values: list[Array] = []
        self._attitude_values: list[float] = []
        self._position_values: list[float] = []
        self._controller = ""
        self._reference_path = np.empty((0, 3), dtype=float)
        plt.ion()
        self.figure.show()
        plt.pause(0.001)

    def start_trial(self, controller: str, reference_path_world_m: Array, trajectory: str, noise_profile: str) -> None:
        self._controller = controller
        self._reference_path = np.asarray(reference_path_world_m, dtype=float).reshape(-1, 3).copy()
        self._times.clear(); self._actual_xy_values.clear(); self._reference_xy_values.clear()
        self._attitude_values.clear(); self._position_values.clear()
        color = self.COLORS.get(controller, "tab:blue")
        self.actual_xy.set_color(color)
        self.attitude_line.set_color(color); self.position_line.set_color(color)
        self.ref_xy.set_data(self._reference_path[:, 0], self._reference_path[:, 1])
        self.actual_xy.set_data([], []); self.target_xy.set_data([], [])
        self.attitude_line.set_data([], []); self.position_line.set_data([], [])
        self.figure.suptitle(
            f"LIVE EVALUATION — {controller} | reference: {trajectory} | sensor noise: {noise_profile}",
            fontsize=13, fontweight="bold",
        )
        span_x = float(np.ptp(self._reference_path[:, 0])) if len(self._reference_path) else 0.0
        span_y = float(np.ptp(self._reference_path[:, 1])) if len(self._reference_path) else 0.0
        margin_x = max(0.18, 0.18 * span_x)
        margin_y = max(0.18, 0.18 * span_y)
        x_min = float(np.min(self._reference_path[:, 0])) if len(self._reference_path) else -0.2
        x_max = float(np.max(self._reference_path[:, 0])) if len(self._reference_path) else 0.2
        y_min = float(np.min(self._reference_path[:, 1])) if len(self._reference_path) else -0.2
        y_max = float(np.max(self._reference_path[:, 1])) if len(self._reference_path) else 0.2
        self.ax_xy.set_xlim(x_min - margin_x, x_max + margin_x)
        self.ax_xy.set_ylim(y_min - margin_y, y_max + margin_y)
        self.ax_xy.set_aspect("equal", adjustable="box")
        self.ax_xy.set_title(f"XY path — {controller}")
        try:
            self.figure.canvas.draw_idle(); plt.pause(0.001)
        except Exception:
            pass

    def update(
        self, time_s: float, actual_position_world_m: Array,
        target_position_world_m: Array, attitude_error_deg: float,
        position_error_m: float,
    ) -> None:
        self._times.append(float(time_s))
        self._actual_xy_values.append(np.asarray(actual_position_world_m, dtype=float).reshape(3).copy())
        self._reference_xy_values.append(np.asarray(target_position_world_m, dtype=float).reshape(3).copy())
        self._attitude_values.append(float(attitude_error_deg))
        self._position_values.append(float(position_error_m))
        actual = np.asarray(self._actual_xy_values, dtype=float)
        target = np.asarray(self._reference_xy_values, dtype=float)
        self.actual_xy.set_data(actual[:, 0], actual[:, 1])
        self.target_xy.set_data([target[-1, 0]], [target[-1, 1]])
        self.attitude_line.set_data(self._times, self._attitude_values)
        self.position_line.set_data(self._times, self._position_values)
        for ax, line in ((self.ax_att, self.attitude_line), (self.ax_pos, self.position_line)):
            ax.relim(); ax.autoscale_view(scalex=False, scaley=True)
            ax.set_xlim(0.0, self.duration_s)
        try:
            self.figure.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass

    def close(self) -> None:
        try:
            plt.ioff()
            plt.close(self.figure)
        except Exception:
            pass


def load_controller_assets(project_root: Path, order: int) -> tuple[Any, Array, dict[str, Any]]:
    """Load the actual saved EDMD matrices and LQR gain; never refit silently."""
    model_path = project_root / "results" / "models" / "attitude_models.npz"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Required trained-model artifact was not found: {model_path}. "
            "Run the existing attitude study first; this demo will not regenerate the dataset or refit a model silently."
        )
    with np.load(model_path, allow_pickle=False) as artifact:
        required = (f"A_N{order}", f"B_N{order}", "lqr_gain", "selected_control_order")
        missing = [key for key in required if key not in artifact.files]
        if missing:
            raise ValueError(f"Model artifact is missing required fields: {missing}.")
        A = np.asarray(artifact[f"A_N{order}"], dtype=float)
        B = np.asarray(artifact[f"B_N{order}"], dtype=float)
        gain = np.asarray(artifact["lqr_gain"], dtype=float)
        selected = int(np.asarray(artifact["selected_control_order"]).reshape(()))
    if selected != order:
        raise ValueError(
            f"Saved LQR gain was generated for selected order N={selected}, but the demo requested N={order}."
        )
    if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape != (9 * order, 9 * order):
        raise ValueError(f"A_N{order} has invalid shape {A.shape}.")
    if B.shape != (9 * order, 3):
        raise ValueError(f"B_N{order} has invalid shape {B.shape}; expected {(9 * order, 3)}.")
    if gain.shape != (3, 9 * order):
        raise ValueError(f"Saved LQR gain has invalid shape {gain.shape}; expected {(3, 9 * order)}.")
    if not all(np.all(np.isfinite(x)) for x in (A, B, gain)):
        raise ValueError("Saved model matrices or gain contain non-finite values.")
    model = SimpleNamespace(A=A, B=B, order=order)
    metadata = {
        "model_artifact": model_path.relative_to(project_root).as_posix(),
        "selected_order": selected,
        "A_shape": list(A.shape),
        "B_shape": list(B.shape),
        "lqr_gain_shape": list(gain.shape),
        "closed_loop_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A - B @ gain)))),
        "model_torque_limits_nm": MODEL_TORQUE_LIMITS_NM.tolist(),
    }
    return model, gain, metadata


def desired_attitude_from_force(force_world_n: Array, yaw_des_rad: float = 0.0) -> Array:
    """Backward-compatible wrapper for the shared position-hold utility."""
    return _desired_attitude_from_force(force_world_n, yaw_des_rad)


def position_hold_command(
    position_world_m: Array,
    velocity_world_m_s: Array,
    target_position_world_m: Array,
    mass_kg: float,
    gravity_m_s2: float,
    config: ControllerDemoConfig,
    *,
    target_yaw_rad: float = 0.0,
    target_velocity_world_m_s: Array | None = None,
    target_acceleration_world_m_s2: Array | None = None,
) -> tuple[Array, float, Array]:
    """Position PD/feed-forward outer loop returning desired attitude and thrust.

    The default zero target velocity/acceleration exactly preserves hover-mode
    behavior. For a generated trajectory, velocity and acceleration feed-forward
    reduce lag while the same bounded PD feedback stabilizes tracking error.
    """
    outer = PositionHoldConfig(
        position_kp=config.position_kp,
        velocity_kd=config.position_kd,
        max_lateral_acceleration_m_s2=config.max_lateral_acceleration_m_s2,
        max_vertical_acceleration_m_s2=config.max_vertical_acceleration_m_s2,
        max_tilt_deg=config.max_tilt_deg,
    )
    outer.validate()
    position = np.asarray(position_world_m, dtype=float).reshape(3)
    velocity = np.asarray(velocity_world_m_s, dtype=float).reshape(3)
    target = np.asarray(target_position_world_m, dtype=float).reshape(3)
    target_velocity = (np.zeros(3, dtype=float) if target_velocity_world_m_s is None
                       else np.asarray(target_velocity_world_m_s, dtype=float).reshape(3))
    target_acceleration = (np.zeros(3, dtype=float) if target_acceleration_world_m_s2 is None
                           else np.asarray(target_acceleration_world_m_s2, dtype=float).reshape(3))
    all_vectors = (position, velocity, target, target_velocity, target_acceleration)
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in all_vectors):
        raise ValueError("Position/velocity references and observations must be finite 3-vectors.")
    if not np.isfinite(mass_kg) or mass_kg <= 0.0:
        raise ValueError("mass_kg must be positive and finite.")
    if not np.isfinite(gravity_m_s2) or gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be positive and finite.")

    position_error = target - position
    kp = np.asarray(outer.position_kp, dtype=float)
    kd = np.asarray(outer.velocity_kd, dtype=float)
    acceleration = (
        kp * position_error
        + kd * (target_velocity - velocity)
        + target_acceleration
    )
    acceleration[2] = float(np.clip(
        acceleration[2], -outer.max_vertical_acceleration_m_s2,
        outer.max_vertical_acceleration_m_s2,
    ))
    lateral_norm = float(np.linalg.norm(acceleration[:2]))
    lateral_cap_from_tilt = (gravity_m_s2 + acceleration[2]) * np.tan(np.deg2rad(outer.max_tilt_deg))
    lateral_limit = max(0.0, min(outer.max_lateral_acceleration_m_s2, lateral_cap_from_tilt))
    if lateral_norm > lateral_limit and lateral_norm > 0.0:
        acceleration[:2] *= lateral_limit / lateral_norm

    force_world = mass_kg * (acceleration + np.array([0.0, 0.0, gravity_m_s2]))
    thrust = float(np.linalg.norm(force_world))
    desired_rotation = _desired_attitude_from_force(force_world, target_yaw_rad)
    return desired_rotation, thrust, acceleration.copy()


def _pybullet_gui_client(adapter: Any) -> tuple[Any, int] | None:
    """Return PyBullet and the active GUI client; no-op for mock/headless adapters."""
    env = getattr(adapter, "_env", None)
    if env is None or not bool(getattr(getattr(adapter, "config", None), "gui", False)):
        return None
    try:
        import pybullet as p
        return p, int(getattr(env, "CLIENT"))
    except Exception:
        return None


def _replace_scene_text(
    adapter: Any,
    key: str,
    text: str,
    position_world_m: Array,
    color: list[float],
    text_size: float = 0.70,
) -> None:
    """Create/update one camera-facing PyBullet text label without leaving trails."""
    resolved = _pybullet_gui_client(adapter)
    if resolved is None:
        return
    p, client_id = resolved
    ids = getattr(adapter, "_presentation_text_ids", None)
    if not isinstance(ids, dict):
        ids = {}
        setattr(adapter, "_presentation_text_ids", ids)
    kwargs: dict[str, Any] = {"lifeTime": 0.0, "physicsClientId": client_id}
    previous_id = ids.get(key)
    if previous_id is not None:
        kwargs["replaceItemUniqueId"] = int(previous_id)
    try:
        item_id = p.addUserDebugText(
            text,
            np.asarray(position_world_m, dtype=float).reshape(3).tolist(),
            textColorRGB=color,
            textSize=float(text_size),
            **kwargs,
        )
        ids[key] = int(item_id)
    except Exception:
        # A failed debug label must never interrupt the physics/controller loop.
        return


def _draw_position_target_marker(adapter: Any, target_position_world_m: Array) -> None:
    """Update the exact 3-D reference marker and its floor projection without leaks."""
    resolved = _pybullet_gui_client(adapter)
    if resolved is None:
        return
    p, client_id = resolved
    target = np.asarray(target_position_world_m, dtype=float).reshape(3)
    if not np.all(np.isfinite(target)):
        return
    gold = [1.0, 0.68, 0.08]
    ground_gold = [0.78, 0.52, 0.10]
    radius = 0.13
    target_z = float(target[2])
    ground_z = 0.025
    count = 32
    ids = getattr(adapter, "_presentation_target_line_ids", None)
    if not isinstance(ids, list):
        ids = []
        setattr(adapter, "_presentation_target_line_ids", ids)
    # Reuse the first 32 IDs for the moving upper ring, next 32 for its floor
    # projection, then two crosshairs and one altitude guide.
    required = 32 + 32 + 2 + 1
    while len(ids) < required:
        ids.append(-1)
    cursor = 0
    for z, color, width in ((target_z, gold, 2.6), (ground_z, ground_gold, 1.4)):
        for i in range(count):
            a0 = 2.0 * np.pi * i / count
            a1 = 2.0 * np.pi * (i + 1) / count
            start_pt = [target[0] + radius * np.cos(a0), target[1] + radius * np.sin(a0), z]
            end_pt = [target[0] + radius * np.cos(a1), target[1] + radius * np.sin(a1), z]
            kwargs: dict[str, Any] = {"lineWidth": width, "lifeTime": 0.0, "physicsClientId": client_id}
            if ids[cursor] >= 0:
                kwargs["replaceItemUniqueId"] = int(ids[cursor])
            try:
                ids[cursor] = int(p.addUserDebugLine(start_pt, end_pt, color, **kwargs))
            except Exception:
                pass
            cursor += 1

    cross = 0.045
    for start_pt, end_pt in (
        ([target[0] - cross, target[1], target_z], [target[0] + cross, target[1], target_z]),
        ([target[0], target[1] - cross, target_z], [target[0], target[1] + cross, target_z]),
    ):
        kwargs = {"lineWidth": 2.2, "lifeTime": 0.0, "physicsClientId": client_id}
        if ids[cursor] >= 0:
            kwargs["replaceItemUniqueId"] = int(ids[cursor])
        try:
            ids[cursor] = int(p.addUserDebugLine(start_pt, end_pt, gold, **kwargs))
        except Exception:
            pass
        cursor += 1
    if target_z > ground_z:
        x_guide = float(target[0] + radius)
        start_pt, end_pt = [x_guide, target[1], ground_z], [x_guide, target[1], target_z]
        kwargs = {"lineWidth": 1.6, "lifeTime": 0.0, "physicsClientId": client_id}
        if ids[cursor] >= 0:
            kwargs["replaceItemUniqueId"] = int(ids[cursor])
        try:
            ids[cursor] = int(p.addUserDebugLine(start_pt, end_pt, ground_gold, **kwargs))
        except Exception:
            pass

    label_position = target + np.array([radius + 0.045, 0.0, 0.018], dtype=float)
    label = "3-D HOVER TARGET" if getattr(adapter, "_presentation_trajectory", "hover") == "hover" else "LIVE 3-D TRAJECTORY REFERENCE"
    _replace_scene_text(
        adapter, "hover_target",
        f"{label}  |  x={target[0]:.2f} m  y={target[1]:.2f} m  z={target_z:.2f} m",
        label_position, gold, text_size=0.80,
    )
    _replace_scene_text(
        adapter, "hover_target_projection",
        "FLOOR PROJECTION  |  vertical guide shows target altitude",
        np.array([target[0] + radius + 0.045, target[1], max(ground_z + 0.04, target_z * 0.45)], dtype=float),
        ground_gold, text_size=0.66,
    )


def _draw_runtime_reference_path(adapter: Any, reference_path_world_m: Array, controller: str) -> None:
    """Draw the planned runtime reference once per episode in gold."""
    resolved = _pybullet_gui_client(adapter)
    if resolved is None:
        return
    p, client_id = resolved
    path = np.asarray(reference_path_world_m, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.all(np.isfinite(path)):
        raise ValueError("reference_path_world_m must be a finite (N,3) array with N>=2.")
    ids: list[int] = []
    if float(np.max(np.ptp(path, axis=0))) <= 1e-8:
        setattr(adapter, "_presentation_reference_path_ids", ids)
        return  # A hover setpoint is a point, not a zero-length line cloud.
    stride = max(1, int(np.ceil((len(path) - 1) / 100.0)))
    sampled_path = path[::stride]
    if not np.allclose(sampled_path[-1], path[-1]):
        sampled_path = np.vstack([sampled_path, path[-1]])
    for start_point, end_point in zip(sampled_path[:-1], sampled_path[1:], strict=True):
        try:
            item_id = p.addUserDebugLine(
                start_point.tolist(), end_point.tolist(), [1.0, 0.68, 0.08],
                lineWidth=1.8, lifeTime=0.0, physicsClientId=client_id,
            )
            ids.append(int(item_id))
        except Exception:
            continue
    setattr(adapter, "_presentation_reference_path_ids", ids)
    setattr(adapter, "_presentation_trajectory", getattr(adapter, "_presentation_trajectory", "hover"))


def _draw_actual_path_segment(adapter: Any, start: Array, end: Array, controller: str) -> None:
    """Accumulate a thin colour-coded actual path behind the vehicle."""
    resolved = _pybullet_gui_client(adapter)
    if resolved is None:
        return
    p, client_id = resolved
    color = _controller_explanation(controller)[1]
    try:
        p.addUserDebugLine(
            np.asarray(start, dtype=float).reshape(3).tolist(),
            np.asarray(end, dtype=float).reshape(3).tolist(), color,
            lineWidth=1.7, lifeTime=0.0, physicsClientId=client_id,
        )
    except Exception:
        return


def _controller_explanation(controller: str) -> tuple[str, list[float]]:
    """Plain-language subtitle and consistent colour for a controller."""
    details = {
        "PID": ("CLASSICAL PID BASELINE", [0.12, 0.50, 1.0]),
        "Koopman-LQR": ("LEARNED KOOPMAN MODEL + LQR FEEDBACK", [1.0, 0.46, 0.08]),
        "Koopman-MPC": ("KOOPMAN PREDICTIVE CONTROL (LQR SAFETY BACKUP)", [0.05, 0.78, 0.40]),
    }
    return details.get(controller, (controller.upper(), [0.92, 0.92, 0.92]))


def _update_body_axes(adapter: Any, observation: Any, length_m: float = 0.17) -> None:
    """Draw a short body-frame triad on the real simulated drone: X red, Y green, Z blue."""
    resolved = _pybullet_gui_client(adapter)
    if resolved is None:
        return
    p, client_id = resolved
    position = np.asarray(observation.position_world_m, dtype=float).reshape(3)
    rotation = np.asarray(observation.rotation_body_to_world, dtype=float).reshape(3, 3)
    axis_specs = (
        ("x", np.array([1.0, 0.20, 0.20]), rotation[:, 0]),
        ("y", np.array([0.20, 1.0, 0.28]), rotation[:, 1]),
        ("z", np.array([0.25, 0.58, 1.0]), rotation[:, 2]),
    )
    ids = getattr(adapter, "_presentation_axis_ids", None)
    if not isinstance(ids, dict):
        ids = {}
        setattr(adapter, "_presentation_axis_ids", ids)
    for key, color, axis in axis_specs:
        kwargs: dict[str, Any] = {"lineWidth": 2.4, "lifeTime": 0.0, "physicsClientId": client_id}
        if key in ids:
            kwargs["replaceItemUniqueId"] = int(ids[key])
        try:
            item_id = p.addUserDebugLine(
                position.tolist(), (position + length_m * axis).tolist(), color.tolist(), **kwargs
            )
            ids[key] = int(item_id)
        except Exception:
            continue


def _update_live_presentation_overlay(
    adapter: Any,
    controller: str,
    observation: Any,
    desired_rotation_body_to_world: Array,
    target_position_world_m: Array,
    cfg: ControllerDemoConfig,
) -> None:
    """Explain the live demonstration directly in the 3-D view using plain language."""
    if _pybullet_gui_client(adapter) is None:
        return
    controller_description, controller_color = _controller_explanation(controller)
    position = np.asarray(observation.position_world_m, dtype=float).reshape(3)
    target = np.asarray(target_position_world_m, dtype=float).reshape(3)
    actual_rotation = np.asarray(observation.rotation_body_to_world, dtype=float).reshape(3, 3)
    attitude_error = geodesic_attitude_error_deg(
        np.asarray(desired_rotation_body_to_world, dtype=float).T @ actual_rotation
    )
    position_error = float(np.linalg.norm(target - position))

    # The compact 4-line label is deliberately anchored just above the drone. It
    # gives a non-specialist the controller name, what it is doing, both live
    # tracking errors, and elapsed time. Updating every ~0.2 s avoids debug spam.
    task_text = (
        "TASK: STRAIGHTEN AND HOLD AT GOLD TARGET"
        if cfg.trajectory == "hover"
        else "TASK: FOLLOW THE NEW 3-D FIGURE-EIGHT REFERENCE"
    )
    label_lines = (
        ("controller", f"ACTIVE CONTROLLER: {controller}  |  {controller_description}", controller_color, 0.17, 1.15),
        ("mission", task_text, [0.95, 0.95, 0.95], 0.25, 0.88),
        ("errors", f"TRUE TRACKING ERROR: attitude {attitude_error:.2f} deg  |  position {position_error:.3f} m", [0.95, 0.95, 0.95], 0.33, 0.78),
        ("time", f"TIME: {float(observation.time_s):.1f} / {cfg.duration_s:.1f} s  |  FEEDBACK NOISE: {cfg.sensor_noise_profile.upper()}", [0.88, 0.90, 0.94], 0.41, 0.70),
    )
    for key, text, color, z_offset, size in label_lines:
        _replace_scene_text(
            adapter, f"live_{key}", text,
            position + np.array([0.0, 0.0, z_offset]), color, text_size=size,
        )

    # The short colour-coded triad makes body orientation visible even though CF2X
    # is physically only a few centimetres wide. The gold segment links the current
    # position to the commanded hover point when the two differ.
    _update_body_axes(adapter, observation)
    resolved = _pybullet_gui_client(adapter)
    if resolved is not None:
        p, client_id = resolved
        error_id = getattr(adapter, "_presentation_position_error_id", None)
        kwargs: dict[str, Any] = {"lineWidth": 1.8, "lifeTime": 0.0, "physicsClientId": client_id}
        if error_id is not None:
            kwargs["replaceItemUniqueId"] = int(error_id)
        try:
            new_id = p.addUserDebugLine(
                position.tolist(), target.tolist(), [1.0, 0.76, 0.16], **kwargs
            )
            setattr(adapter, "_presentation_position_error_id", int(new_id))
        except Exception:
            pass


def _update_final_presentation_overlay(
    adapter: Any,
    controller: str,
    trace: dict[str, Any],
    metrics: dict[str, Any],
    cfg: ControllerDemoConfig,
) -> None:
    """Replace the live HUD with a simple, final result card during the review pause."""
    if _pybullet_gui_client(adapter) is None:
        return
    position = np.asarray(trace["position_world_m"][-1], dtype=float).reshape(3)
    _controller_description, controller_color = _controller_explanation(controller)
    settling = metrics.get("settling_time_to_tolerance_s")
    settling_text = f"{float(settling):.2f} s" if settling is not None else "NOT SETTLED"
    lines = (
        ("controller", f"RUN COMPLETE: {controller}", controller_color, 0.17, 1.15),
        ("mission", f"ATTITUDE RMSE: {float(metrics['geodesic_attitude_rmse_deg']):.3f} deg  |  SETTLING: {settling_text}", [0.95, 0.95, 0.95], 0.25, 0.88),
        ("errors", f"POSITION RMSE: {float(metrics['position_rmse_3d_m']):.3f} m", [0.95, 0.95, 0.95], 0.33, 0.88),
        ("time", "GOLD = CURRENT REFERENCE  |  COLOURED TRAIL = ACTUAL PATH  |  LOWER RING = FLOOR PROJECTION", [0.88, 0.90, 0.94], 0.41, 0.68),
    )
    for key, text, color, z_offset, size in lines:
        _replace_scene_text(
            adapter, f"live_{key}", text,
            position + np.array([0.0, 0.0, z_offset]), color, text_size=size,
        )


def follow_pybullet_camera(
    adapter: Any,
    position_world_m: Array,
    cfg: ControllerDemoConfig,
    target_position_world_m: Array | None = None,
) -> None:
    """Frame both the quadrotor and hover ring with a close, adaptive follow camera."""
    if not cfg.camera_follow:
        return
    resolved = _pybullet_gui_client(adapter)
    if resolved is None:
        return
    p, client_id = resolved
    try:
        if not bool(getattr(adapter, "_presentation_gui_configured", False)):
            for flag_name in (
                "COV_ENABLE_GUI", "COV_ENABLE_RGB_BUFFER_PREVIEW",
                "COV_ENABLE_DEPTH_BUFFER_PREVIEW", "COV_ENABLE_SEGMENTATION_MARK_PREVIEW",
            ):
                flag = getattr(p, flag_name, None)
                if flag is not None:
                    try:
                        p.configureDebugVisualizer(flag, 0, physicsClientId=client_id)
                    except Exception:
                        pass
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client_id)
            except Exception:
                pass
            setattr(adapter, "_presentation_gui_configured", True)

        position = np.asarray(position_world_m, dtype=float).reshape(3)
        target = np.asarray(
            cfg.initial_position_world_m if target_position_world_m is None else target_position_world_m,
            dtype=float,
        ).reshape(3)
        separation = float(np.linalg.norm(position - target))
        # Center midway between aircraft and goal; zoom out gently only if they
        # separate, keeping both in view without returning to the distant horizon shot.
        camera_target = 0.5 * (position + target)
        camera_target[2] += 0.035
        camera_distance = float(cfg.camera_distance_m + min(0.28, 0.28 * separation))
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=float(cfg.camera_yaw_deg),
            cameraPitch=float(cfg.camera_pitch_deg),
            cameraTargetPosition=camera_target.tolist(),
            physicsClientId=client_id,
        )
    except Exception:
        return

def controller_model_torque(
    controller: str,
    rotation_body_to_world: Array,
    body_rates_rad_s: Array,
    model: Any,
    lqr_gain: Array,
    mpc: Any | None,
    pid_integral: Array,
    dt: float,
    config: ControllerDemoConfig,
    desired_rotation_body_to_world: Array | None = None,
    desired_body_rates_rad_s: Array | None = None,
) -> tuple[Array, Array, dict[str, Any]]:
    """Return requested and limited model-equivalent torque plus diagnostics."""
    rotation = np.asarray(rotation_body_to_world, dtype=float)
    omega = np.asarray(body_rates_rad_s, dtype=float).reshape(3)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(omega)):
        raise ValueError("Controller received an invalid state.")
    limits = MODEL_TORQUE_LIMITS_NM
    desired_rotation = np.eye(3) if desired_rotation_body_to_world is None else np.asarray(desired_rotation_body_to_world, dtype=float)
    if desired_rotation.shape != (3, 3) or not np.all(np.isfinite(desired_rotation)):
        raise ValueError("desired_rotation_body_to_world must be a finite 3x3 matrix.")
    # Express the current attitude as an error rotation around the controller's
    # identity equilibrium. This is a local tracking approximation; the outer
    # loop limits desired tilt and desired-attitude rates remain slow.
    attitude_error_rotation = desired_rotation.T @ rotation
    desired_omega = (
        np.zeros(3, dtype=float) if desired_body_rates_rad_s is None
        else np.asarray(desired_body_rates_rad_s, dtype=float).reshape(3)
    )
    if not np.all(np.isfinite(desired_omega)):
        raise ValueError("desired_body_rates_rad_s must contain three finite values.")
    # Convert the desired angular velocity into the measured body's coordinates,
    # then form rate error. This is important when the position outer loop asks
    # for a changing (tilted) attitude instead of a fixed level attitude.
    desired_omega_in_actual_body = attitude_error_rotation.T @ desired_omega
    omega_error = omega - desired_omega_in_actual_body
    next_integral = np.asarray(pid_integral, dtype=float).reshape(3).copy()
    diagnostics: dict[str, Any] = {
        "mpc_solve_time_ms": 0.0,
        "mpc_success": True,
        "mpc_fallback": False,
        "mpc_iterations": 0,
    }

    if controller == "PID":
        angles = rotation_to_euler_zyx(attitude_error_rotation)
        error = np.asarray(wrap_angle(-angles), dtype=float)
        kp, ki, kd = map(lambda x: np.asarray(x, dtype=float), (config.pid_kp, config.pid_ki, config.pid_kd))
        integral_limit = np.asarray(config.pid_integral_limit, dtype=float)
        candidate = np.clip(next_integral + error * dt, -integral_limit, integral_limit)
        trial = kp * error + ki * candidate - kd * omega_error
        pushing_positive = (trial > limits) & (error > 0.0)
        pushing_negative = (trial < -limits) & (error < 0.0)
        accept = ~(pushing_positive | pushing_negative)
        next_integral[accept] = candidate[accept]
        requested = kp * error + ki * next_integral - kd * omega_error
    elif controller == "Koopman-LQR":
        z = attitude_lift(attitude_error_rotation, omega_error, model.order)
        z_equilibrium = attitude_lift(np.eye(3), np.zeros(3), model.order)
        requested = -np.asarray(lqr_gain, dtype=float) @ (z - z_equilibrium)
    elif controller == "Koopman-MPC":
        if mpc is None:
            raise ValueError("Koopman-MPC was selected but no MPC controller was initialized.")
        result = mpc.command(attitude_error_rotation, omega_error)
        requested = np.asarray(result.torque, dtype=float)
        diagnostics = {
            "mpc_solve_time_ms": float(result.solve_time_ms),
            "mpc_success": bool(result.success),
            "mpc_fallback": bool(result.used_fallback),
            "mpc_iterations": int(result.iterations),
        }
    else:
        raise ValueError(f"Unknown controller {controller!r}; expected one of {CONTROLLERS}.")

    requested = np.asarray(requested, dtype=float).reshape(-1)
    if requested.shape != (3,) or not np.all(np.isfinite(requested)):
        raise FloatingPointError(f"{controller} produced an invalid torque request: {requested!r}.")
    limited = np.clip(requested, -limits, limits)
    diagnostics["controller_torque_limited"] = bool(np.any(np.abs(limited - requested) > 1e-12))
    return requested, limited, {**diagnostics, "pid_integral_next": next_integral}


def _obs_euler_deg(observation: Any) -> Array:
    return np.rad2deg(rotation_to_euler_zyx(observation.rotation_body_to_world))


def _safety_stop_reason(observation: Any, cfg: ControllerDemoConfig) -> str | None:
    altitude = float(observation.position_world_m[2])
    rate_norm = float(np.linalg.norm(observation.body_rates_rad_s))
    attitude_error = geodesic_attitude_error_deg(observation.rotation_body_to_world)
    if not np.all(np.isfinite(observation.position_world_m)) or not np.all(np.isfinite(observation.body_rates_rad_s)):
        return "non_finite_state"
    if altitude <= cfg.safety_min_altitude_m:
        return "minimum_altitude_limit"
    if altitude >= cfg.safety_max_altitude_m:
        return "maximum_altitude_limit"
    if attitude_error >= cfg.safety_geodesic_limit_deg:
        return "attitude_excursion_limit"
    if rate_norm >= cfg.safety_max_body_rate_rad_s:
        return "body_rate_limit"
    return None


def _new_trace_lists(
    initial_observation: Any,
    initial_controller_observation: Any | None = None,
    initial_target_position_world_m: Array | None = None,
    initial_target_velocity_world_m_s: Array | None = None,
    initial_target_acceleration_world_m_s2: Array | None = None,
    initial_desired_rotation_body_to_world: Array | None = None,
) -> dict[str, Any]:
    """Create aligned state (N) and command (N-1) trace buffers."""
    feedback = initial_observation if initial_controller_observation is None else initial_controller_observation
    target = np.asarray(
        initial_observation.position_world_m if initial_target_position_world_m is None
        else initial_target_position_world_m, dtype=float,
    ).reshape(3)
    target_velocity = np.zeros(3) if initial_target_velocity_world_m_s is None else np.asarray(
        initial_target_velocity_world_m_s, dtype=float,
    ).reshape(3)
    target_acceleration = np.zeros(3) if initial_target_acceleration_world_m_s2 is None else np.asarray(
        initial_target_acceleration_world_m_s2, dtype=float,
    ).reshape(3)
    desired_rotation = np.eye(3) if initial_desired_rotation_body_to_world is None else np.asarray(
        initial_desired_rotation_body_to_world, dtype=float,
    ).reshape(3, 3)
    return {
        "time_s": [float(initial_observation.time_s)],
        "rotation_body_to_world": [np.asarray(initial_observation.rotation_body_to_world, dtype=float).copy()],
        "angles_rad": [rotation_to_euler_zyx(initial_observation.rotation_body_to_world)],
        "body_rates_rad_s": [np.asarray(initial_observation.body_rates_rad_s, dtype=float).copy()],
        "position_world_m": [np.asarray(initial_observation.position_world_m, dtype=float).copy()],
        "velocity_world_m_s": [np.asarray(initial_observation.velocity_world_m_s, dtype=float).copy()],
        "controller_observed_rotation_body_to_world": [np.asarray(feedback.rotation_body_to_world, dtype=float).copy()],
        "controller_observed_body_rates_rad_s": [np.asarray(feedback.body_rates_rad_s, dtype=float).copy()],
        "controller_observed_position_world_m": [np.asarray(feedback.position_world_m, dtype=float).copy()],
        "controller_observed_velocity_world_m_s": [np.asarray(feedback.velocity_world_m_s, dtype=float).copy()],
        "requested_model_torque_nm": [],
        "limited_model_torque_nm": [],
        "requested_physical_torque_nm": [],
        "allocator_implied_physical_torque_nm": [],
        "motor_rpm": [],
        "requested_collective_thrust_n": [],
        "achieved_collective_thrust_n": [],
        "saturation_scale": [],
        "saturated": [],
        "mpc_solve_time_ms": [],
        "mpc_success": [],
        "mpc_fallback": [],
        "mpc_iterations": [],
        "position_target_world_m": [target.copy()],
        "position_target_velocity_world_m_s": [target_velocity.copy()],
        "position_target_acceleration_world_m_s2": [target_acceleration.copy()],
        "desired_attitude_deg": [],
        "position_hold_acceleration_command_m_s2": [],
        # These histories are state-aligned (N), including the initial state.
        "desired_rotation_history": [desired_rotation.copy()],
        "desired_body_rates_history": [np.zeros(3, dtype=float)],
    }


def _outer_loop_command(
    controller_observation: Any,
    target_position: Array,
    target_velocity: Array,
    target_acceleration: Array,
    adapter: Any,
    cfg: ControllerDemoConfig,
) -> tuple[Array, float, Array]:
    if cfg.position_hold_enabled:
        return position_hold_command(
            controller_observation.position_world_m,
            controller_observation.velocity_world_m_s,
            target_position,
            float(getattr(adapter, "mass_kg", 0.027)),
            float(getattr(adapter, "gravity_m_s2", 9.81)),
            cfg,
            target_yaw_rad=0.0,
            target_velocity_world_m_s=target_velocity,
            target_acceleration_world_m_s2=target_acceleration,
        )
    return (
        np.eye(3),
        float(adapter.mass_kg * adapter.gravity_m_s2),
        np.zeros(3, dtype=float),
    )


def run_live_trial(
    controller: str,
    adapter: Any,
    model: Any,
    lqr_gain: Array,
    cfg: ControllerDemoConfig,
    *,
    realtime: bool,
    overlay: bool,
    live_dashboard: LiveEvaluationDashboard | None = None,
) -> dict[str, Any]:
    """Run the selected controller on live PyBullet feedback.

    The analytic reference is generated at runtime. Optional sensor noise affects
    only the feedback supplied to the controller; metrics use true simulator state.
    """
    dt = adapter.control_timestep_s
    cfg.validate(dt)
    observation = adapter.reset(
        initial_position_world_m=np.asarray(cfg.initial_position_world_m, dtype=float),
        initial_euler_zyx_rad=np.deg2rad(np.asarray(cfg.initial_attitude_deg, dtype=float)),
    )
    rng = np.random.default_rng(int(cfg.sensor_noise_seed))
    controller_observation = noisy_controller_observation(observation, cfg.sensor_noise_profile, rng)
    reference_origin = np.asarray(cfg.initial_position_world_m, dtype=float)
    target_position, target_velocity, target_acceleration = trajectory_reference(
        observation.time_s, reference_origin, cfg,
    )
    desired_rotation, collective_thrust_n, accel_command = _outer_loop_command(
        controller_observation, target_position, target_velocity, target_acceleration, adapter, cfg,
    )
    if live_dashboard is not None:
        reference_path = trajectory_samples(cfg, dt)
        live_dashboard.start_trial(controller, reference_path, cfg.trajectory, cfg.sensor_noise_profile)
        initial_attitude_error = geodesic_attitude_error_deg(
            desired_rotation.T @ observation.rotation_body_to_world
        )
        initial_position_error = float(np.linalg.norm(target_position - observation.position_world_m))
        live_dashboard.update(
            observation.time_s, observation.position_world_m, target_position,
            initial_attitude_error, initial_position_error,
        )
    else:
        reference_path = trajectory_samples(cfg, dt)

    if cfg.camera_follow:
        follow_pybullet_camera(adapter, observation.position_world_m, cfg, target_position)
    setattr(adapter, "_presentation_trajectory", cfg.trajectory)
    _draw_runtime_reference_path(adapter, reference_path, controller)
    _draw_position_target_marker(adapter, target_position)

    trace = _new_trace_lists(
        observation, controller_observation, target_position, target_velocity,
        target_acceleration, desired_rotation,
    )
    mpc = None
    if controller == "Koopman-MPC":
        mpc = KoopmanMPC(
            model,
            torque_limits=MODEL_TORQUE_LIMITS_NM,
            fallback_gain=lqr_gain,
            horizon=cfg.mpc_horizon,
            q_rotation=cfg.mpc_q_rotation,
            q_rate=cfg.mpc_q_rate,
            q_higher=cfg.mpc_q_higher,
            r_weights=cfg.mpc_r_weights,
        )

    steps = int(round(cfg.duration_s / dt))
    pid_integral = np.zeros(3, dtype=float)
    previous_desired_rotation: Array | None = None
    started = time.perf_counter()
    next_wall_deadline = started
    stop_reason = "duration_complete"
    presentation_stride = max(1, int(round(0.10 / dt)))
    dashboard_stride = max(1, int(round(0.10 / dt)))
    previous_trail_position = np.asarray(observation.position_world_m, dtype=float).copy()

    for k in range(steps):
        # Reference is time-varying and computed online, independently of training data.
        target_position, target_velocity, target_acceleration = trajectory_reference(
            observation.time_s, reference_origin, cfg,
        )
        desired_rotation, collective_thrust_n, accel_command = _outer_loop_command(
            controller_observation, target_position, target_velocity, target_acceleration, adapter, cfg,
        )
        if previous_desired_rotation is None:
            desired_body_rates = np.zeros(3, dtype=float)
        else:
            desired_body_rates = angular_velocity_between_rotations(
                previous_desired_rotation, desired_rotation, dt
            )
        previous_desired_rotation = desired_rotation.copy()
        desired_angles_deg = np.rad2deg(rotation_to_euler_zyx(desired_rotation))

        requested, limited, diagnostics = controller_model_torque(
            controller, controller_observation.rotation_body_to_world,
            controller_observation.body_rates_rad_s,
            model, lqr_gain, mpc, pid_integral, dt, cfg,
            desired_rotation_body_to_world=desired_rotation,
            desired_body_rates_rad_s=desired_body_rates,
        )
        pid_integral = np.asarray(diagnostics.pop("pid_integral_next"), dtype=float)
        next_observation, mapped = adapter.step_model_equivalent_torque(
            limited, duration_s=dt, collective_thrust_n=collective_thrust_n,
        )
        mix = mapped.mixer_output
        trace["requested_model_torque_nm"].append(requested.copy())
        trace["limited_model_torque_nm"].append(limited.copy())
        trace["requested_physical_torque_nm"].append(np.asarray(mapped.requested_physical_torque_nm, dtype=float).copy())
        trace["allocator_implied_physical_torque_nm"].append(np.asarray(mix.allocator_implied_body_torque_nm, dtype=float).copy())
        trace["motor_rpm"].append(np.asarray(mix.motor_rpm, dtype=float).copy())
        trace["requested_collective_thrust_n"].append(float(mapped.collective_thrust_requested_n))
        trace["achieved_collective_thrust_n"].append(float(mix.achieved_collective_thrust_n))
        trace["saturation_scale"].append(float(mix.saturation_scale))
        trace["saturated"].append(bool(mix.saturated or diagnostics["controller_torque_limited"]))
        trace["mpc_solve_time_ms"].append(float(diagnostics["mpc_solve_time_ms"]))
        trace["mpc_success"].append(bool(diagnostics["mpc_success"]))
        trace["mpc_fallback"].append(bool(diagnostics["mpc_fallback"]))
        trace["mpc_iterations"].append(int(diagnostics["mpc_iterations"]))
        trace["desired_attitude_deg"].append(desired_angles_deg.copy())
        trace["position_hold_acceleration_command_m_s2"].append(accel_command.copy())

        # Ground truth is the simulator observation. Noise is applied only to the next
        # controller feedback sample, never to the metrics or actual-path display.
        observation = next_observation
        controller_observation = noisy_controller_observation(
            observation, cfg.sensor_noise_profile, rng,
        )
        next_target_position, next_target_velocity, next_target_acceleration = trajectory_reference(
            observation.time_s, reference_origin, cfg,
        )
        next_desired_rotation, _next_thrust, _next_acceleration = _outer_loop_command(
            controller_observation, next_target_position, next_target_velocity,
            next_target_acceleration, adapter, cfg,
        )
        desired_rate_next = angular_velocity_between_rotations(desired_rotation, next_desired_rotation, dt)

        trace["time_s"].append(float(observation.time_s))
        trace["rotation_body_to_world"].append(np.asarray(observation.rotation_body_to_world, dtype=float).copy())
        trace["angles_rad"].append(rotation_to_euler_zyx(observation.rotation_body_to_world))
        trace["body_rates_rad_s"].append(np.asarray(observation.body_rates_rad_s, dtype=float).copy())
        trace["position_world_m"].append(np.asarray(observation.position_world_m, dtype=float).copy())
        trace["velocity_world_m_s"].append(np.asarray(observation.velocity_world_m_s, dtype=float).copy())
        trace["controller_observed_rotation_body_to_world"].append(np.asarray(controller_observation.rotation_body_to_world, dtype=float).copy())
        trace["controller_observed_body_rates_rad_s"].append(np.asarray(controller_observation.body_rates_rad_s, dtype=float).copy())
        trace["controller_observed_position_world_m"].append(np.asarray(controller_observation.position_world_m, dtype=float).copy())
        trace["controller_observed_velocity_world_m_s"].append(np.asarray(controller_observation.velocity_world_m_s, dtype=float).copy())
        trace["position_target_world_m"].append(next_target_position.copy())
        trace["position_target_velocity_world_m_s"].append(next_target_velocity.copy())
        trace["position_target_acceleration_world_m_s2"].append(next_target_acceleration.copy())
        trace["desired_rotation_history"].append(next_desired_rotation.copy())
        trace["desired_body_rates_history"].append(desired_rate_next.copy())

        if cfg.camera_follow:
            follow_pybullet_camera(adapter, observation.position_world_m, cfg, next_target_position)
        if (k + 1) % presentation_stride == 0:
            _draw_position_target_marker(adapter, next_target_position)
            _draw_actual_path_segment(adapter, previous_trail_position, observation.position_world_m, controller)
            if overlay:
                _update_live_presentation_overlay(
                    adapter, controller, observation, next_desired_rotation,
                    next_target_position, cfg,
                )
            previous_trail_position = np.asarray(observation.position_world_m, dtype=float).copy()
        attitude_error = geodesic_attitude_error_deg(
            next_desired_rotation.T @ observation.rotation_body_to_world
        )
        position_error = float(np.linalg.norm(next_target_position - observation.position_world_m))
        if live_dashboard is not None and (k + 1) % dashboard_stride == 0:
            live_dashboard.update(
                observation.time_s, observation.position_world_m, next_target_position,
                attitude_error, position_error,
            )
        if (k + 1) % max(1, int(round(1.0 / dt))) == 0:
            print(
                f"\r{controller:12s} t={observation.time_s:5.2f}/{cfg.duration_s:5.2f}s  "
                f"attitude error={attitude_error:5.2f} deg  "
                f"position error={position_error:5.3f} m",
                end="", flush=True,
            )
        stop_reason = _safety_stop_reason(observation, cfg) or "duration_complete"
        if stop_reason != "duration_complete":
            print(f"\nSafety stop for {controller}: {stop_reason}.")
            break

        target_position = next_target_position
        target_velocity = next_target_velocity
        target_acceleration = next_target_acceleration
        desired_rotation = next_desired_rotation
        next_wall_deadline += dt
        if realtime:
            time.sleep(max(0.0, next_wall_deadline - time.perf_counter()))

    print()
    arrays = {name: np.asarray(value) for name, value in trace.items()}
    arrays["angles_deg"] = np.rad2deg(arrays["angles_rad"])
    arrays["position_error_world_m"] = arrays["position_target_world_m"] - arrays["position_world_m"]
    arrays["position_error_norm_m"] = np.linalg.norm(arrays["position_error_world_m"], axis=1)
    arrays["desired_rotation_body_to_world"] = np.asarray(trace["desired_rotation_history"], dtype=float)
    arrays["desired_body_rates_rad_s"] = np.asarray(trace["desired_body_rates_history"], dtype=float)
    arrays["desired_attitude_deg_state"] = np.rad2deg(np.stack([
        rotation_to_euler_zyx(rotation) for rotation in arrays["desired_rotation_body_to_world"]
    ]))
    arrays["desired_attitude_deg"] = np.asarray(trace["desired_attitude_deg"], dtype=float)
    arrays["position_hold_acceleration_command_m_s2"] = np.asarray(
        trace["position_hold_acceleration_command_m_s2"], dtype=float,
    )
    arrays["level_attitude_error_deg"] = np.asarray([
        geodesic_attitude_error_deg(rotation) for rotation in arrays["rotation_body_to_world"]
    ], dtype=float)
    arrays["attitude_tracking_error_deg"] = np.asarray([
        geodesic_attitude_error_deg(desired.T @ actual)
        for desired, actual in zip(
            arrays["desired_rotation_body_to_world"], arrays["rotation_body_to_world"], strict=True
        )
    ], dtype=float)
    arrays["geodesic_attitude_error_deg"] = arrays["attitude_tracking_error_deg"].copy()
    arrays["position_hold_enabled"] = bool(cfg.position_hold_enabled)
    arrays["camera_follow_enabled"] = bool(cfg.camera_follow)
    arrays["controller"] = str(controller)
    arrays["backend"] = str(adapter.name)
    arrays["trajectory"] = str(cfg.trajectory)
    arrays["sensor_noise_profile"] = str(cfg.sensor_noise_profile)
    arrays["sensor_noise_seed"] = int(cfg.sensor_noise_seed)
    arrays["stop_reason"] = stop_reason
    arrays["control_period_s"] = dt
    arrays["wall_duration_s"] = float(time.perf_counter() - started)
    validate_live_trace(arrays)
    return arrays


def validate_live_trace(trace: dict[str, Any]) -> None:
    """Enforce state/command alignment and finite values before saving results."""
    n = len(np.asarray(trace["time_s"]).reshape(-1))
    if n < 2:
        raise ValueError("Live trace must contain at least two state samples.")
    time_axis = np.asarray(trace["time_s"], dtype=float)
    if not np.all(np.isfinite(time_axis)) or np.any(np.diff(time_axis) <= 0.0):
        raise ValueError("Live simulation timestamps must be finite and strictly increasing.")
    state_shapes = {
        "rotation_body_to_world": (n, 3, 3), "angles_rad": (n, 3), "angles_deg": (n, 3),
        "body_rates_rad_s": (n, 3), "position_world_m": (n, 3), "velocity_world_m_s": (n, 3),
        "geodesic_attitude_error_deg": (n,),
        "position_target_world_m": (n, 3),
        "position_error_world_m": (n, 3),
        "position_error_norm_m": (n,),
    }
    # Reference-tracking fields are mandatory in current live traces but optional
    # for backwards compatibility with older hand-built unit-test fixtures.
    optional_state_shapes = {
        "desired_rotation_body_to_world": (n, 3, 3),
        "desired_attitude_deg_state": (n, 3),
        "desired_body_rates_rad_s": (n, 3),
        "attitude_tracking_error_deg": (n,),
        "level_attitude_error_deg": (n,),
        "position_target_velocity_world_m_s": (n, 3),
        "position_target_acceleration_world_m_s2": (n, 3),
        "controller_observed_rotation_body_to_world": (n, 3, 3),
        "controller_observed_body_rates_rad_s": (n, 3),
        "controller_observed_position_world_m": (n, 3),
        "controller_observed_velocity_world_m_s": (n, 3),
    }
    interval_shapes = {
        "requested_model_torque_nm": (n - 1, 3), "limited_model_torque_nm": (n - 1, 3),
        "requested_physical_torque_nm": (n - 1, 3), "allocator_implied_physical_torque_nm": (n - 1, 3),
        "motor_rpm": (n - 1, 4), "requested_collective_thrust_n": (n - 1,),
        "achieved_collective_thrust_n": (n - 1,), "saturation_scale": (n - 1,), "saturated": (n - 1,),
        "mpc_solve_time_ms": (n - 1,), "mpc_success": (n - 1,), "mpc_fallback": (n - 1,),
        "mpc_iterations": (n - 1,),
        "desired_attitude_deg": (n - 1, 3),
        "position_hold_acceleration_command_m_s2": (n - 1, 3),
    }
    shapes_to_check = {**state_shapes, **interval_shapes}
    shapes_to_check.update({k: v for k, v in optional_state_shapes.items() if k in trace})
    for name, shape in shapes_to_check.items():
        values = np.asarray(trace[name])
        if values.shape != shape:
            raise ValueError(f"{name} has shape {values.shape}; expected {shape}.")
        if values.dtype.kind not in "bui" and not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
    if "position_target_world_m" in trace and np.asarray(trace["position_target_world_m"]).shape != (n, 3):
        raise ValueError("position_target_world_m must have shape (N,3).")
    if "position_error_world_m" in trace and np.asarray(trace["position_error_world_m"]).shape != (n, 3):
        raise ValueError("position_error_world_m must have shape (N,3).")
    if "desired_attitude_deg" in trace and np.asarray(trace["desired_attitude_deg"]).shape != (n - 1, 3):
        raise ValueError("desired_attitude_deg must have shape (N-1,3).")
    if "position_hold_acceleration_command_m_s2" in trace and np.asarray(trace["position_hold_acceleration_command_m_s2"]).shape != (n - 1, 3):
        raise ValueError("position_hold_acceleration_command_m_s2 must have shape (N-1,3).")
    if np.any(np.asarray(trace["motor_rpm"]) < 0.0):
        raise ValueError("Motor RPM trace cannot be negative.")
    if np.any(np.diff(time_axis) < 0.99 * float(trace["control_period_s"])):
        raise ValueError("The state timestamps are not aligned to the configured control period.")


def calculate_live_metrics(trace: dict[str, Any], cfg: ControllerDemoConfig) -> dict[str, Any]:
    """Compute metrics from measured simulator outputs, not model predictions."""
    time_axis = np.asarray(trace["time_s"], dtype=float)
    angles_deg = np.asarray(trace["angles_deg"], dtype=float)
    rates = np.asarray(trace["body_rates_rad_s"], dtype=float)
    geo = np.asarray(
        trace.get("attitude_tracking_error_deg", trace["geodesic_attitude_error_deg"]), dtype=float
    )
    level_geo = np.asarray(
        trace.get("level_attitude_error_deg", trace["geodesic_attitude_error_deg"]), dtype=float
    )
    pos = np.asarray(trace["position_world_m"], dtype=float)
    requested_model = np.asarray(trace["requested_model_torque_nm"], dtype=float)
    requested_physical = np.asarray(trace["requested_physical_torque_nm"], dtype=float)
    allocated_physical = np.asarray(trace["allocator_implied_physical_torque_nm"], dtype=float)
    motors = np.asarray(trace["motor_rpm"], dtype=float)
    saturated = np.asarray(trace["saturated"], dtype=bool)
    if len(time_axis) != len(geo) or len(requested_model) != len(time_axis) - 1:
        raise ValueError("State metrics and control-interval logs are misaligned.")

    dt = float(trace["control_period_s"])
    final_geo = float(geo[-1])
    settled_index: int | None = None
    below = geo <= cfg.settling_tolerance_deg
    if np.any(below):
        suffix_pass = np.logical_and.accumulate(below[::-1])[::-1]
        candidates = np.flatnonzero(suffix_pass)
        if candidates.size:
            settled_index = int(candidates[0])
    mpc = str(trace["controller"]) == "Koopman-MPC"
    mpc_times = np.asarray(trace["mpc_solve_time_ms"], dtype=float) if mpc else np.array([], dtype=float)
    iae = float(np.trapezoid(geo, time_axis) if hasattr(np, "trapezoid") else np.trapz(geo, time_axis))
    position_target = np.asarray(trace.get("position_target_world_m", np.zeros_like(pos)), dtype=float)
    position_error = position_target - pos
    velocity_rmse = float(np.sqrt(np.mean(np.sum(np.asarray(trace["velocity_world_m_s"], dtype=float) ** 2, axis=1))))
    position_rmse = float(np.sqrt(np.mean(np.sum(position_error ** 2, axis=1))))
    desired_rotation = np.asarray(trace.get(
        "desired_rotation_body_to_world", np.repeat(np.eye(3)[None, :, :], len(time_axis), axis=0)
    ), dtype=float)
    desired_rates = np.asarray(trace.get("desired_body_rates_rad_s", np.zeros_like(rates)), dtype=float)
    actual_rotations = np.asarray(trace["rotation_body_to_world"], dtype=float)
    desired_rates_in_actual_body = np.asarray([
        actual.T @ desired @ rate for actual, desired, rate in zip(
            actual_rotations, desired_rotation, desired_rates, strict=True
        )
    ], dtype=float)
    rate_error = rates - desired_rates_in_actual_body
    final_rotation_error = desired_rotation[-1].T @ actual_rotations[-1]
    final_angle_error_deg = np.rad2deg(rotation_to_euler_zyx(final_rotation_error))
    result: dict[str, Any] = {
        "controller": str(trace["controller"]),
        "backend": str(trace["backend"]),
        "status": str(trace["stop_reason"]),
        "state_samples": int(len(time_axis)),
        "control_intervals": int(len(time_axis) - 1),
        "duration_s": float(time_axis[-1] - time_axis[0]),
        "geodesic_attitude_rmse_deg": float(np.sqrt(np.mean(geo**2))),
        "geodesic_attitude_iae_deg_s": iae,
        "peak_geodesic_attitude_error_deg": float(np.max(geo)),
        "final_geodesic_attitude_error_deg": final_geo,
        "level_attitude_rmse_deg": float(np.sqrt(np.mean(level_geo**2))),
        "final_level_attitude_error_deg": float(level_geo[-1]),
        "final_roll_error_deg": float(abs(final_angle_error_deg[0])),
        "final_pitch_error_deg": float(abs(final_angle_error_deg[1])),
        "final_yaw_error_deg": float(abs(float(wrap_angle(np.deg2rad(final_angle_error_deg[2])))) * 180.0 / np.pi),
        "settling_time_to_tolerance_s": float(time_axis[settled_index]) if settled_index is not None else None,
        "settling_tolerance_deg": float(cfg.settling_tolerance_deg),
        "body_rate_vector_rmse_rad_s": float(np.sqrt(np.mean(np.sum(rates**2, axis=1)))),
        "body_rate_tracking_rmse_rad_s": float(np.sqrt(np.mean(np.sum(rate_error**2, axis=1)))),
        "peak_body_rate_norm_rad_s": float(np.max(np.linalg.norm(rates, axis=1))),
        "min_altitude_m": float(np.min(pos[:, 2])),
        "max_altitude_m": float(np.max(pos[:, 2])),
        "position_hold_enabled": bool(trace.get("position_hold_enabled", False)),
        "position_rmse_3d_m": position_rmse,
        "final_position_error_3d_m": float(np.linalg.norm(position_error[-1])),
        "peak_position_error_3d_m": float(np.max(np.linalg.norm(position_error, axis=1))),
        "velocity_rmse_3d_m_s": velocity_rmse,
        "rms_model_equivalent_torque_nm": float(np.sqrt(np.mean(np.sum(requested_model**2, axis=1)))),
        "rms_requested_physical_torque_nm": float(np.sqrt(np.mean(np.sum(requested_physical**2, axis=1)))),
        "max_allocator_implied_torque_residual_nm": float(np.max(np.linalg.norm(allocated_physical - requested_physical, axis=1))),
        "motor_rpm_max": float(np.max(motors)),
        "motor_rpm_mean": float(np.mean(motors)),
        "saturated_interval_percent": float(100.0 * np.mean(saturated)),
        "mpc_solve_time_median_ms": float(np.median(mpc_times)) if mpc else None,
        "mpc_solve_time_p95_ms": float(np.percentile(mpc_times, 95)) if mpc else None,
        "mpc_solver_success_percent": float(100.0 * np.mean(trace["mpc_success"])) if mpc else None,
        "mpc_fallback_count": int(np.sum(trace["mpc_fallback"])) if mpc else None,
        "model_artifact": "results/models/attitude_models.npz",
        "trace_source": "live gym-pybullet-drones/CF2X observations",
    }
    return result


def _safe_name(controller: str) -> str:
    return controller.lower().replace("-", "_")


def save_live_trial(trace: dict[str, Any], cfg: ControllerDemoConfig, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    trace_dir = output_root / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(str(trace["controller"]))
    npz_path = trace_dir / f"{safe}_trace.npz"
    csv_path = trace_dir / f"{safe}_trace.csv"
    # Backward-compatible defaults for unit tests/older traces; live S3 runs
    # populate these with actual outer-loop commands before saving.
    n_states = len(np.asarray(trace["time_s"]).reshape(-1))
    positions_for_defaults = np.asarray(trace["position_world_m"], dtype=float)
    default_target = np.repeat(positions_for_defaults[0:1], n_states, axis=0)
    trace.setdefault("position_target_world_m", default_target)
    trace.setdefault("position_error_world_m", np.asarray(trace["position_target_world_m"]) - positions_for_defaults)
    trace.setdefault("position_error_norm_m", np.linalg.norm(np.asarray(trace["position_error_world_m"]), axis=1))
    trace.setdefault("desired_attitude_deg", np.zeros((n_states - 1, 3), dtype=float))
    trace.setdefault("position_hold_acceleration_command_m_s2", np.zeros((n_states - 1, 3), dtype=float))
    desired_interval_angles = np.asarray(trace["desired_attitude_deg"], dtype=float)
    if desired_interval_angles.shape != (n_states - 1, 3):
        raise ValueError("desired_attitude_deg must have shape (state_samples - 1, 3).")
    trace.setdefault("desired_attitude_deg_state", np.concatenate([desired_interval_angles, desired_interval_angles[-1:]], axis=0))
    if "desired_rotation_body_to_world" not in trace:
        trace["desired_rotation_body_to_world"] = np.stack([
            _rotation_from_euler_zyx(np.deg2rad(value)) for value in trace["desired_attitude_deg_state"]
        ])
    trace.setdefault("desired_body_rates_rad_s", np.zeros((n_states, 3), dtype=float))
    if "level_attitude_error_deg" not in trace:
        trace["level_attitude_error_deg"] = np.asarray([
            geodesic_attitude_error_deg(rotation) for rotation in np.asarray(trace["rotation_body_to_world"])
        ], dtype=float)
    if "attitude_tracking_error_deg" not in trace:
        trace["attitude_tracking_error_deg"] = np.asarray([
            geodesic_attitude_error_deg(desired.T @ actual)
            for desired, actual in zip(
                np.asarray(trace["desired_rotation_body_to_world"]),
                np.asarray(trace["rotation_body_to_world"]), strict=True,
            )
        ], dtype=float)
    # Keep the legacy field name as an alias for the corrected tracking metric.
    trace["geodesic_attitude_error_deg"] = np.asarray(trace["attitude_tracking_error_deg"], dtype=float)
    numeric_keys = [
        "time_s", "rotation_body_to_world", "angles_rad", "angles_deg", "body_rates_rad_s",
        "position_world_m", "velocity_world_m_s", "geodesic_attitude_error_deg",
        "requested_model_torque_nm", "limited_model_torque_nm", "requested_physical_torque_nm",
        "allocator_implied_physical_torque_nm", "motor_rpm", "requested_collective_thrust_n",
        "achieved_collective_thrust_n", "saturation_scale", "saturated", "mpc_solve_time_ms",
        "mpc_success", "mpc_fallback", "mpc_iterations",
        "position_target_world_m", "position_error_world_m", "position_error_norm_m",
        "desired_attitude_deg", "desired_attitude_deg_state", "desired_rotation_body_to_world",
        "desired_body_rates_rad_s", "attitude_tracking_error_deg", "level_attitude_error_deg",
        "position_hold_acceleration_command_m_s2",
    ]
    np.savez_compressed(npz_path, **{key: np.asarray(trace[key]) for key in numeric_keys})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_index", "time_s", "roll_deg", "pitch_deg", "yaw_deg", "p_rad_s", "q_rad_s", "r_rad_s",
            "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s", "attitude_tracking_error_deg",
            "desired_roll_deg", "desired_pitch_deg", "desired_yaw_deg", "level_attitude_error_deg",
            "desired_p_rad_s", "desired_q_rad_s", "desired_r_rad_s",
            "target_x_m", "target_y_m", "target_z_m", "position_error_x_m", "position_error_y_m", "position_error_z_m",
            "interval_index", "requested_model_tau_x_nm", "requested_model_tau_y_nm", "requested_model_tau_z_nm",
            "limited_model_tau_x_nm", "limited_model_tau_y_nm", "limited_model_tau_z_nm",
            "requested_physical_tau_x_nm", "requested_physical_tau_y_nm", "requested_physical_tau_z_nm",
            "allocator_implied_tau_x_nm", "allocator_implied_tau_y_nm", "allocator_implied_tau_z_nm",
            "motor_0_rpm", "motor_1_rpm", "motor_2_rpm", "motor_3_rpm", "collective_requested_n",
            "collective_allocator_implied_n", "saturation_scale", "saturated", "mpc_solve_time_ms", "mpc_success", "mpc_fallback", "mpc_iterations",
        ])
        n = len(trace["time_s"])
        for i in range(n):
            angles = np.asarray(trace["angles_deg"])[i]
            rates = np.asarray(trace["body_rates_rad_s"])[i]
            pos = np.asarray(trace["position_world_m"])[i]
            vel = np.asarray(trace["velocity_world_m_s"])[i]
            desired_angles = np.asarray(trace["desired_attitude_deg_state"])[i]
            desired_rates = np.asarray(trace["desired_body_rates_rad_s"])[i]
            target = np.asarray(trace["position_target_world_m"])[i]
            pos_error = np.asarray(trace["position_error_world_m"])[i]
            row: list[Any] = [
                i, trace["time_s"][i], *angles.tolist(), *rates.tolist(), *pos.tolist(), *vel.tolist(),
                trace["attitude_tracking_error_deg"][i], *desired_angles.tolist(),
                trace["level_attitude_error_deg"][i], *desired_rates.tolist(), *target.tolist(), *pos_error.tolist(),
            ]
            j = i if i < n - 1 else n - 2
            if i == n - 1:
                # Final state has no outgoing command; leave command columns blank.
                row.extend([""] * 25)
            else:
                row.extend([
                    i, *np.asarray(trace["requested_model_torque_nm"])[j].tolist(),
                    *np.asarray(trace["limited_model_torque_nm"])[j].tolist(),
                    *np.asarray(trace["requested_physical_torque_nm"])[j].tolist(),
                    *np.asarray(trace["allocator_implied_physical_torque_nm"])[j].tolist(),
                    *np.asarray(trace["motor_rpm"])[j].tolist(), trace["requested_collective_thrust_n"][j],
                    trace["achieved_collective_thrust_n"][j], trace["saturation_scale"][j], bool(trace["saturated"][j]),
                    trace["mpc_solve_time_ms"][j], bool(trace["mpc_success"][j]), bool(trace["mpc_fallback"][j]), int(trace["mpc_iterations"][j]),
                ])
            writer.writerow(row)
    metrics = calculate_live_metrics(trace, cfg)
    metrics["trace_csv"] = csv_path.relative_to(output_root).as_posix()
    metrics["trace_npz"] = npz_path.relative_to(output_root).as_posix()
    with (output_root / f"{safe}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(metrics), handle, indent=2, allow_nan=False)
    return metrics


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.ndarray,)):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _plot_comparison(traces: dict[str, dict[str, Any]], figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = {"PID": "tab:blue", "Koopman-LQR": "tab:orange", "Koopman-MPC": "tab:green"}
    data = list(traces.items())
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for name, trace in data:
        actual_angles = np.asarray(trace["angles_deg"], dtype=float)
        desired_angles = np.asarray(trace.get("desired_attitude_deg_state", np.zeros_like(actual_angles)), dtype=float)
        for axis_index in range(3):
            axes[axis_index].plot(
                trace["time_s"], actual_angles[:, axis_index], label=f"{name} actual",
                color=colors.get(name), linewidth=1.7,
            )
            axes[axis_index].plot(
                trace["time_s"], desired_angles[:, axis_index], label=f"{name} target",
                color=colors.get(name), linewidth=1.2, linestyle="--", alpha=0.85,
            )
    for i, label in enumerate(("Roll (deg)", "Pitch (deg)", "Yaw (deg)")):
        axes[i].axhline(0.0, linestyle="--", linewidth=0.8, color="black", alpha=0.5)
        axes[i].set_ylabel(label); axes[i].grid(True, alpha=0.3)
        if i == 0:
            axes[i].legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("Live CF2X attitude tracking — solid: measured, dashed: commanded")
    attitude_file = figure_dir / "attitude_response.png"
    fig.savefig(attitude_file, dpi=180); plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.4), constrained_layout=True)
    for name, trace in data:
        geo = np.asarray(trace.get("attitude_tracking_error_deg", trace["geodesic_attitude_error_deg"]), dtype=float)
        rmse = float(np.sqrt(np.mean(geo ** 2)))
        final = float(geo[-1])
        ax.plot(trace["time_s"], geo, label=f"{name} (RMSE {rmse:.2f} deg; final {final:.2f} deg)", color=colors.get(name), linewidth=1.8)
    ax.axhline(2.0, linestyle="--", linewidth=1.1, color="black", label="2 deg settling tolerance")
    ax.set_xlabel("Simulation time (s)"); ax.set_ylabel("Geodesic attitude error (deg)")
    ax.grid(True, alpha=0.28); ax.legend(loc="best", ncol=1)
    ax.set_title("Actual-to-commanded geodesic attitude tracking error")
    fig.savefig(figure_dir / "attitude_error.png", dpi=180); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for name, trace in data:
        rates_deg = np.rad2deg(trace["body_rates_rad_s"])
        for i, ax in enumerate(axes):
            ax.plot(trace["time_s"], rates_deg[:, i], label=name, color=colors.get(name))
            ax.grid(True, alpha=0.3); ax.set_ylabel(("p", "q", "r")[i] + " (deg/s)")
    axes[0].legend(loc="best"); axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("Measured body rates")
    fig.savefig(figure_dir / "body_rate_response.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for name, trace in data:
        for i, ax in enumerate(axes):
            ax.plot(trace["time_s"][:-1], trace["limited_model_torque_nm"][:, i], label=name, color=colors.get(name))
            ax.grid(True, alpha=0.3); ax.set_ylabel(("tau_x", "tau_y", "tau_z")[i] + " (model-equivalent N m)")
    axes[0].legend(loc="best"); axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("Controller torque commands before physical-inertia mapping")
    fig.savefig(figure_dir / "model_torque_commands.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    for name, trace in data:
        for i, ax in enumerate(axes):
            ax.plot(trace["time_s"][:-1], trace["motor_rpm"][:, i], label=name, color=colors.get(name))
            ax.grid(True, alpha=0.3); ax.set_ylabel(f"Motor {i} RPM")
    axes[0].legend(loc="best"); axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("CF2X motor commands allocated by the shared adapter")
    fig.savefig(figure_dir / "motor_rpm.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axis_names = ("x", "y", "z")
    for name, trace in data:
        target = np.asarray(trace["position_target_world_m"])
        actual = np.asarray(trace["position_world_m"])
        for i, ax in enumerate(axes):
            ax.plot(trace["time_s"], actual[:, i], label=f"{name} actual", color=colors.get(name))
            if name == data[0][0]:
                ax.plot(trace["time_s"], target[:, i], linestyle=":", color="black", label="position target")
            ax.grid(True, alpha=0.3); ax.set_ylabel(f"{axis_names[i]} (m)")
    axes[0].legend(loc="best", ncol=2); axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("Live PyBullet position hold — measured state versus target")
    fig.savefig(figure_dir / "position_hold.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    for name, trace in data:
        err = np.asarray(trace["position_error_world_m"])
        for i, ax in enumerate(axes):
            ax.plot(trace["time_s"], err[:, i], label=name, color=colors.get(name))
            ax.grid(True, alpha=0.3); ax.set_ylabel(f"e{axis_names[i]} (m)")
    axes[0].legend(loc="best"); axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("World-frame position-hold error")
    fig.savefig(figure_dir / "position_hold_error.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5.4), constrained_layout=True)
    for name, trace in data:
        err = np.asarray(trace["position_error_world_m"], dtype=float)
        norm = np.linalg.norm(err, axis=1)
        ax.plot(trace["time_s"], norm, label=f"{name} (final {norm[-1]:.3f} m; peak {np.max(norm):.3f} m)", color=colors.get(name), linewidth=1.8)
    ax.axhline(0.05, linestyle=":", linewidth=1.1, color="black", label="5 cm reference")
    ax.set_xlabel("Simulation time (s)"); ax.set_ylabel("3D position error norm (m)")
    ax.grid(True, alpha=0.28); ax.legend(loc="best")
    ax.set_title("World-frame position-hold accuracy")
    fig.savefig(figure_dir / "position_error_norm.png", dpi=180); plt.close(fig)

    png_paths = sorted(figure_dir.glob("*.png"))
    if len(png_paths) < 8:
        raise RuntimeError(f"Presentation plot generation produced only {len(png_paths)} PNG files in {figure_dir}.")
    return png_paths


def _write_figure_manifest(figure_dir: Path, png_paths: list[Path], source_trace_dir: Path) -> Path:
    """Record exact file timestamps, byte sizes and hashes to verify generated plots."""
    files = []
    for path in png_paths:
        stat = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({
            "file": path.name,
            "size_bytes": int(stat.st_size),
            "modified_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            "sha256": digest,
        })
    payload = {
        "generated_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": "saved live PyBullet trace NPZ/CSV data; no simulator replay used for plotting",
        "source_trace_dir": str(source_trace_dir.resolve()),
        "figure_count": len(files),
        "figures": files,
    }
    path = figure_dir / "figure_generation_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_summary_print(row: dict[str, Any]) -> None:
    settling = row["settling_time_to_tolerance_s"]
    settling_text = f"{settling:.2f}s" if settling is not None else "not settled"
    print(
        f"  attitude geodesic RMSE={row['geodesic_attitude_rmse_deg']:.3f} deg; "
        f"final error={row['final_geodesic_attitude_error_deg']:.3f} deg; "
        f"settling={settling_text}; "
        f"saturation={row['saturated_interval_percent']:.1f}% | "
        f"position RMSE={row.get('position_rmse_3d_m', float('nan')):.3f}m | "
        f"final position error={row.get('final_position_error_3d_m', float('nan')):.3f}m"
    )



def _plot_minimal_validation(
    traces: dict[str, dict[str, Any]], metrics_rows: list[dict[str, Any]], path: Path
) -> Path:
    """Write one compact figure showing only the two traces behind the headline metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"PID": "tab:blue", "Koopman-LQR": "tab:orange", "Koopman-MPC": "tab:green"}
    metrics_by_name = {str(row["controller"]): row for row in metrics_rows}
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True, constrained_layout=True)
    for name, trace in traces.items():
        color = colors.get(name, "tab:blue")
        time_axis = np.asarray(trace["time_s"], dtype=float)
        attitude_error = np.asarray(
            trace.get("attitude_tracking_error_deg", trace["geodesic_attitude_error_deg"]), dtype=float
        )
        row = metrics_by_name[name]
        settle = row.get("settling_time_to_tolerance_s")
        settle_text = f"{float(settle):.2f} s" if settle is not None else "not settled"
        axes[0].plot(
            time_axis, attitude_error, color=color, linewidth=1.8,
            label=f"{name} | RMSE {float(row['geodesic_attitude_rmse_deg']):.2f}° | settle {settle_text}",
        )
        axes[1].plot(
            time_axis, np.asarray(trace["position_error_norm_m"], dtype=float),
            color=color, linewidth=1.8,
            label=f"{name} | RMSE {float(row['position_rmse_3d_m']):.3f} m",
        )
    axes[0].axhline(2.0, color="black", linestyle="--", linewidth=1.0, label="2° settling threshold")
    axes[0].set_ylabel("Attitude tracking error (deg)")
    trajectory_names = {str(trace.get("trajectory", "hover")) for trace in traces.values()}
    noise_names = {str(trace.get("sensor_noise_profile", "none")) for trace in traces.values()}
    axes[0].set_title(
        "Live CF2X tracking — " + ", ".join(sorted(trajectory_names))
        + " reference | feedback noise: " + ", ".join(sorted(noise_names))
    )
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.28)
    axes[1].set_ylabel("3D position error (m)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(True, alpha=0.28)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_minimal_trace_bundle(traces: dict[str, dict[str, Any]], path: Path) -> Path:
    """Store numeric per-sample and per-control-step arrays in one audit-friendly NPZ."""
    payload: dict[str, np.ndarray] = {}
    for name, trace in traces.items():
        prefix = _safe_name(name)
        for key, value in trace.items():
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
                payload[f"{prefix}__{key}"] = value
    if not payload:
        raise ValueError("No numeric live traces were available to save.")
    np.savez_compressed(path, **payload)
    return path


def _headline_row(row: dict[str, Any]) -> dict[str, Any]:
    """Expose exactly three primary performance metrics for panel/report use."""
    return {
        "controller": str(row["controller"]),
        "attitude_tracking_rmse_deg": float(row["geodesic_attitude_rmse_deg"]),
        "settling_time_s": (
            None if row.get("settling_time_to_tolerance_s") is None
            else float(row["settling_time_to_tolerance_s"])
        ),
        "position_rmse_3d_m": float(row["position_rmse_3d_m"]),
        "trial_status": str(row.get("status", "unknown")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal live PID/Koopman-LQR/Koopman-MPC CF2X tracking demonstration."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--controller", choices=("all", "PID", "Koopman-LQR", "Koopman-MPC"), default="all")
    parser.add_argument("--duration", type=float, default=20.0, help="Duration per controller in seconds (default: 20).")
    parser.add_argument("--initial-attitude-deg", type=float, nargs=3, default=(10.0, 8.0, 6.0), metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--settling-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--mpc-horizon", type=int, default=20)
    parser.add_argument("--trajectory", choices=("hover", "figure8"), default="hover",
                        help="Runtime-generated reference; hover preserves the existing static-target demonstration.")
    parser.add_argument("--trajectory-radius", type=float, default=0.25, help="Figure-eight horizontal scale in metres (0.05-0.50).")
    parser.add_argument("--trajectory-vertical-amplitude", type=float, default=0.08, help="Figure-eight vertical amplitude in metres (0-0.15).")
    parser.add_argument("--trajectory-period", type=float, default=12.0, help="Figure-eight period in seconds (minimum 6).")
    parser.add_argument("--sensor-noise", choices=("none", "low", "moderate"), default="none",
                        help="Reproducible noise added to controller feedback only; metrics use simulator ground truth.")
    parser.add_argument("--noise-seed", type=int, default=20261024,
                        help="Fixed seed; the same noise sequence is reused for each controller comparison.")
    parser.add_argument("--no-live-plots", action="store_true",
                        help="Disable the three live evaluation plots if the interactive backend is unavailable.")
    parser.add_argument("--attitude-only", action="store_true", help="Disable position hold; trajectory tracking will not be controlled.")
    parser.add_argument("--fixed-camera", action="store_true", help="Disable the close-up follow camera.")
    parser.add_argument("--headless", action="store_true", help="Run without a GUI, live plots, or real-time pacing.")
    parser.add_argument("--fast", action="store_true", help="Run as fast as possible even with GUI enabled.")
    parser.add_argument("--no-pause", action="store_true", help="Do not pause between controller episodes.")
    args = parser.parse_args(argv)

    root = args.project_root.resolve()
    cfg_path = root / "configs" / "simulation.json"
    if cfg_path.is_file():
        adapter_base_cfg = PyBulletAdapterConfig.from_json(cfg_path)
    else:
        adapter_base_cfg = PyBulletAdapterConfig()
        print(f"Warning: {cfg_path} was not found; using explicit adapter defaults.")

    gui = not args.headless
    adapter_cfg = replace(adapter_base_cfg, gui=gui, user_debug_gui=False)
    demo_cfg = ControllerDemoConfig(
        duration_s=float(args.duration),
        initial_attitude_deg=tuple(float(x) for x in args.initial_attitude_deg),
        settling_tolerance_deg=float(args.settling_tolerance_deg),
        mpc_horizon=int(args.mpc_horizon),
        position_hold_enabled=not bool(args.attitude_only),
        camera_follow=not bool(args.fixed_camera),
        trajectory=str(args.trajectory),
        trajectory_radius_m=float(args.trajectory_radius),
        trajectory_vertical_amplitude_m=float(args.trajectory_vertical_amplitude),
        trajectory_period_s=float(args.trajectory_period),
        sensor_noise_profile=str(args.sensor_noise),
        sensor_noise_seed=int(args.noise_seed),
    )
    demo_cfg.validate(1.0 / adapter_cfg.control_frequency_hz)
    model, gain, model_metadata = load_controller_assets(root, demo_cfg.selected_order)
    live_dashboard: LiveEvaluationDashboard | None = None
    if gui and not args.no_live_plots:
        try:
            live_dashboard = LiveEvaluationDashboard(demo_cfg.duration_s)
        except Exception as exc:
            raise RuntimeError(
                "The live evaluation dashboard could not be opened, so the simulation was not started. "
                "Use an interactive Matplotlib backend (TkAgg/QtAgg) and retry. "
                "--no-live-plots is available only when you intentionally want the previous 3-D-only view."
            ) from exc

    output_base = root / "results" / "simulation_runs" / "pybullet_controller_demo"
    output_base.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output_root = output_base / run_id
    output_root.mkdir(parents=True, exist_ok=False)

    controller_names = list(CONTROLLERS) if args.controller == "all" else [args.controller]
    traces: dict[str, dict[str, Any]] = {}
    raw_metrics: list[dict[str, Any]] = []

    print("Live CF2X controller demonstration")
    print(f"Backend: gym-pybullet-drones/CF2X | GUI={'on' if gui else 'off'} | duration={demo_cfg.duration_s:.1f}s per controller")
    print("Primary evaluation metrics: attitude tracking RMSE, 2° settling time, 3D position RMSE")
    print("Live feedback comes from the simulator; the learned model is not refit.")
    print(f"Runtime reference: {demo_cfg.trajectory}; sensor-noise profile: {demo_cfg.sensor_noise_profile}; noise seed: {demo_cfg.sensor_noise_seed}")
    print("Noise, when enabled, affects controller feedback only. Headline metrics use true simulator state.")
    if live_dashboard is not None:
        print("Live dashboard: XY reference-vs-actual path, attitude-error trace, and 3-D position-error trace.")
    if gui:
        print("How to read the scene: coloured ACTIVE CONTROLLER label identifies the method;")
        print("the BRIGHT GOLD RING is the exact 3-D target; the lower ring is only its floor projection;")
        print("red/green/blue axes show body X/Y/Z; the vertical gold guide shows target altitude;")
        print("live text shows attitude error, distance to target, and elapsed time.")

    for controller in controller_names:
        adapter = PyBulletDronesAdapter(adapter_cfg)
        try:
            trace = run_live_trial(
                controller, adapter, model, gain, demo_cfg,
                realtime=(gui and not args.fast),
                overlay=gui,
                live_dashboard=live_dashboard,
            )
            metrics = calculate_live_metrics(trace, demo_cfg)
            traces[controller] = trace
            raw_metrics.append(metrics)

            settling = metrics["settling_time_to_tolerance_s"]
            settling_text = f"{float(settling):.2f} s" if settling is not None else "not settled"
            print(
                f"{controller:12s} | attitude RMSE {metrics['geodesic_attitude_rmse_deg']:.3f}°"
                f" | settling {settling_text}"
                f" | position RMSE {metrics['position_rmse_3d_m']:.3f} m"
            )
            if controller == "Koopman-MPC":
                fallback_count = int(metrics.get("mpc_fallback_count") or 0)
                solver_calls = int(len(np.asarray(trace.get("mpc_fallback", []))))
                if fallback_count:
                    print(
                        f"  MPC safety diagnostic: LQR fallback used on {fallback_count}/{solver_calls} "
                        "control intervals; this is disclosed separately from the three headline metrics."
                    )
            if str(trace.get("stop_reason", "")) != "duration_complete":
                print(f"  WARNING: trial stopped early: {trace['stop_reason']}")
            if gui:
                _update_final_presentation_overlay(adapter, controller, trace, metrics, demo_cfg)
            if gui and not args.no_pause:
                try:
                    input("Review the labelled live view and result card, then press Enter for the next controller... ")
                except EOFError:
                    pass
        finally:
            adapter.close()

    headline_metrics = [_headline_row(row) for row in raw_metrics]
    plot_path = _plot_minimal_validation(traces, raw_metrics, output_root / "controller_validation.png")
    traces_path = _save_minimal_trace_bundle(traces, output_root / "live_traces.npz")

    mpc_diagnostic = None
    for row in raw_metrics:
        if row["controller"] == "Koopman-MPC":
            mpc_diagnostic = {
                "fallback_intervals": int(row.get("mpc_fallback_count") or 0),
                "solver_success_percent": row.get("mpc_solver_success_percent"),
                "note": "When the optimizer reports failure, the controller uses the saved LQR gain as a safety fallback.",
            }
            break

    summary = {
        "experiment": "Live CF2X attitude tracking with world-position hold",
        "run_id": run_id,
        "generated_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "backend": "gym-pybullet-drones/CF2X",
        "duration_s_per_controller": float(demo_cfg.duration_s),
        "control_period_s": float(1.0 / adapter_cfg.control_frequency_hz),
        "runtime_reference": {
            "trajectory": demo_cfg.trajectory,
            "generated_at_runtime": demo_cfg.trajectory == "figure8",
            "radius_m": float(demo_cfg.trajectory_radius_m),
            "vertical_amplitude_m": float(demo_cfg.trajectory_vertical_amplitude_m),
            "period_s": float(demo_cfg.trajectory_period_s),
            "description": ("static world-frame hover setpoint" if demo_cfg.trajectory == "hover"
                            else "analytic 3-D figure-eight reference generated online; not used to train the attitude model"),
        },
        "sensor_noise": {
            "profile": demo_cfg.sensor_noise_profile,
            "seed": int(demo_cfg.sensor_noise_seed),
            "standard_deviations": sensor_noise_sigmas(demo_cfg.sensor_noise_profile),
            "application": "controller feedback only; reported metrics use uncorrupted simulator ground truth",
        },
        "live_evaluation_plots_enabled": bool(live_dashboard is not None),
        "initial_position_world_m": list(demo_cfg.initial_position_world_m),
        "initial_attitude_deg": list(demo_cfg.initial_attitude_deg),
        "settling_tolerance_deg": float(demo_cfg.settling_tolerance_deg),
        "model_artifact": model_metadata["model_artifact"],
        "koopman_order": int(model_metadata["selected_order"]),
        "primary_metrics": [
            "attitude_tracking_rmse_deg",
            "settling_time_s",
            "position_rmse_3d_m",
        ],
        "controllers": headline_metrics,
        "execution_diagnostics": {"mpc": mpc_diagnostic} if mpc_diagnostic is not None else {},
        "interpretation": [
            "All three primary metrics are calculated from measured live simulator traces.",
            "Settling time is the first sample after which actual-to-commanded geodesic attitude error stays within the configured tolerance for the remainder of the run.",
            "Position RMSE is the root mean squared Euclidean error between measured position and the world-frame reference trajectory.",
            "For figure8 mode, the reference is generated analytically online and is not used to train or refit the attitude model.",
            "Optional measurement noise is applied only to the controller feedback; performance metrics use uncorrupted simulator observations.",
            "The MPC result includes the disclosed LQR safety fallback at any optimizer-failure intervals.",
            "The learned-model torque is inertia-mapped to CF2X and is not direct physical-torque telemetry; this validates the simulation adapter, not hardware equivalence.",
        ],
        "artifacts": [plot_path.name, traces_path.name, "simulation_summary.json"],
    }
    summary_path = output_root / "simulation_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False), encoding="utf-8")

    print("\nSimulation finished.")
    print(f"Run output folder: {output_root}")
    print(f"Three-metric summary: {summary_path}")
    print(f"Single validation figure: {plot_path}")
    print(f"Raw live traces: {traces_path}")
    if not gui:
        print("Headless mode selected; no live window was opened.")
    if live_dashboard is not None:
        live_dashboard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
