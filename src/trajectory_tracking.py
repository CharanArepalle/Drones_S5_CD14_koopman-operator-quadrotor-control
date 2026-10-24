"""Shared six-degree-of-freedom quadrotor trajectory-tracking experiment.

The outer loop maps position/velocity tracking errors to collective thrust and
an attitude reference. Three inner-loop options are compared: PID, the project's
learned Koopman-LQR controller, and the project's bounded Koopman-MPC controller.
All cases use the same reference trajectory, nonlinear 6-DoF rigid-body plant,
sampling rate, integration settings, initial state, mass/inertia and actuators.

The plant equations follow the translational and rotational quadrotor model in
Bayrak & Efe's disturbance-observer tutorial, while parameters and the trajectory
are declared experiment choices rather than claimed hardware-identical values.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import attitude_study as baseline
from src.attitude_core import (
    attitude_lift,
    euler_zyx_to_rotation,
    project_rotation,
    rotation_matrix_to_vector,
    rotation_to_euler_zyx,
    skew,
)
from src.identification import KoopmanModel, fit_koopman_edmd
from src.koopman_mpc import KoopmanMPC

Array = np.ndarray


@dataclass(frozen=True)
class TrackingConfig:
    duration_s: float = 15.0
    sample_period_s: float = 0.05
    integration_substeps: int = 5
    mass_kg: float = 1.2
    gravity_m_s2: float = 9.81
    max_thrust_multiple_of_weight: float = 2.3
    order: int = 5
    mpc_horizon: int = 10
    position_kp: tuple[float, float, float] = (2.4, 2.4, 3.0)
    position_kd: tuple[float, float, float] = (2.0, 2.0, 2.4)
    pid_kp_nm_rad: tuple[float, float, float] = (0.22, 0.22, 0.12)
    pid_ki_nm_rad_s: tuple[float, float, float] = (0.004, 0.004, 0.002)
    pid_kd_nm_rad_s: tuple[float, float, float] = (0.06, 0.06, 0.036)
    pid_integral_limit: tuple[float, float, float] = (0.7, 0.7, 0.7)
    trajectory_frequency_rad_s: float = 0.4
    x_amplitude_m: float = 1.0
    y_amplitude_m: float = 0.65
    z_base_m: float = 1.0
    z_amplitude_m: float = 0.25
    yaw_amplitude_rad: float = 0.25
    yaw_frequency_rad_s: float = 0.2
    initial_position_offset_m: tuple[float, float, float] = (0.15, -0.10, 0.08)
    initial_velocity_offset_m_s: tuple[float, float, float] = (-0.10, 0.07, -0.04)
    initial_attitude_offset_deg: tuple[float, float, float] = (4.0, -3.0, 5.0)
    initial_rate_offset_rad_s: tuple[float, float, float] = (0.04, -0.03, 0.025)
    translational_disturbance_force_n: tuple[float, float, float] = (0.12, 0.08, 0.05)
    linear_drag_n_per_m_s: float = 0.10
    seed: int = 20261018

    def validate(self) -> None:
        if self.duration_s <= 0 or self.sample_period_s <= 0:
            raise ValueError("duration_s and sample_period_s must be positive.")
        if not np.isclose(self.duration_s / self.sample_period_s,
                          round(self.duration_s / self.sample_period_s), atol=1e-10):
            raise ValueError("duration_s must be divisible by sample_period_s.")
        if self.integration_substeps < 1 or self.mpc_horizon < 1:
            raise ValueError("integration_substeps and mpc_horizon must be >= 1.")
        if self.mass_kg <= 0 or self.gravity_m_s2 <= 0:
            raise ValueError("mass and gravity must be positive.")
        if self.max_thrust_multiple_of_weight <= 1:
            raise ValueError("max_thrust_multiple_of_weight must exceed 1.")
        if self.order < 2:
            raise ValueError("order must be >= 2.")

    @property
    def steps(self) -> int:
        return int(round(self.duration_s / self.sample_period_s))


def _wrap(a: Array | float) -> Array | float:
    return np.arctan2(np.sin(a), np.cos(a))


def trajectory_reference(time_s: Array, cfg: TrackingConfig) -> dict[str, Array]:
    """Smooth 3-D Lissajous/figure-eight reference with analytic derivatives."""
    t = np.asarray(time_s, dtype=float).reshape(-1)
    w = cfg.trajectory_frequency_rad_s
    wy = cfg.yaw_frequency_rad_s
    x, y, z = cfg.x_amplitude_m * np.sin(w*t), cfg.y_amplitude_m * np.sin(2*w*t), cfg.z_base_m + cfg.z_amplitude_m*(1-np.cos(w*t))
    vx, vy, vz = cfg.x_amplitude_m*w*np.cos(w*t), 2*cfg.y_amplitude_m*w*np.cos(2*w*t), cfg.z_amplitude_m*w*np.sin(w*t)
    ax, ay, az = -cfg.x_amplitude_m*w*w*np.sin(w*t), -4*cfg.y_amplitude_m*w*w*np.sin(2*w*t), cfg.z_amplitude_m*w*w*np.cos(w*t)
    yaw = cfg.yaw_amplitude_rad*np.sin(wy*t)
    yaw_rate = cfg.yaw_amplitude_rad*wy*np.cos(wy*t)
    yaw_acc = -cfg.yaw_amplitude_rad*wy*wy*np.sin(wy*t)
    return {
        "time": t,
        "position": np.column_stack([x, y, z]),
        "velocity": np.column_stack([vx, vy, vz]),
        "acceleration": np.column_stack([ax, ay, az]),
        "yaw": yaw,
        "yaw_rate": yaw_rate,
        "yaw_acceleration": yaw_acc,
    }


def acceleration_to_attitude(
    acceleration_world: Array,
    yaw: float,
    mass_kg: float,
    gravity_m_s2: float = 9.81,
) -> tuple[float, Array]:
    """Map desired world acceleration and yaw to collective thrust and R_des.

    ENU/Z-up convention: world acceleration = T/m * R[:,2] - g*e_z.
    """
    acc = np.asarray(acceleration_world, dtype=float).reshape(3)
    if not np.all(np.isfinite(acc)) or not np.isfinite(yaw):
        raise ValueError("acceleration and yaw must be finite.")
    if mass_kg <= 0 or gravity_m_s2 <= 0:
        raise ValueError("mass and gravity must be positive.")
    force = mass_kg * (acc + np.array([0.0, 0.0, gravity_m_s2]))
    thrust = float(np.linalg.norm(force))
    if thrust < 1e-8:
        raise ValueError("Desired thrust vector is too small to define attitude.")
    b3 = force / thrust
    heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    b2 = np.cross(b3, heading)
    if np.linalg.norm(b2) < 1e-8:
        heading = np.array([np.cos(yaw + 1e-3), np.sin(yaw + 1e-3), 0.0])
        b2 = np.cross(b3, heading)
    b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    rotation = project_rotation(np.column_stack([b1, b2, b3]))
    return thrust, rotation


def _pack_state(position: Array, velocity: Array, rotation: Array, omega: Array) -> Array:
    return np.concatenate([
        np.asarray(position, float).reshape(3),
        np.asarray(velocity, float).reshape(3),
        np.asarray(rotation, float).reshape(9),
        np.asarray(omega, float).reshape(3),
    ])


def _unpack_state(state: Array) -> tuple[Array, Array, Array, Array]:
    x = np.asarray(state, dtype=float).reshape(18)
    return x[:3].copy(), x[3:6].copy(), project_rotation(x[6:15].reshape(3, 3)), x[15:18].copy()


def quadrotor_derivative(
    state: Array,
    thrust_n: float,
    torque_nm: Array,
    mass_kg: float,
    inertia: Array,
    gravity_m_s2: float = 9.81,
    force_disturbance_n: Array | None = None,
    linear_drag_n_per_m_s: float = 0.0,
) -> Array:
    """Continuous nonlinear 6-DoF ENU rigid-body dynamics derivative."""
    position, velocity, rotation, omega = _unpack_state(state)
    del position
    torque = np.asarray(torque_nm, dtype=float).reshape(3)
    inertia = np.asarray(inertia, dtype=float).reshape(3)
    if mass_kg <= 0 or np.any(inertia <= 0) or not np.isfinite(thrust_n):
        raise ValueError("Plant parameters must be finite and positive.")
    if force_disturbance_n is None:
        force_disturbance = np.zeros(3, dtype=float)
    else:
        force_disturbance = np.asarray(force_disturbance_n, dtype=float).reshape(3)
    if not np.all(np.isfinite(force_disturbance)):
        raise ValueError("force_disturbance_n must be finite.")
    if not np.isfinite(linear_drag_n_per_m_s) or linear_drag_n_per_m_s < 0:
        raise ValueError("linear_drag_n_per_m_s must be finite and non-negative.")
    position_dot = velocity
    velocity_dot = (float(thrust_n) / mass_kg) * rotation[:, 2] - np.array([0.0, 0.0, gravity_m_s2])
    velocity_dot += (force_disturbance - linear_drag_n_per_m_s * velocity) / mass_kg
    rotation_dot = rotation @ skew(omega)
    omega_dot = (torque - np.cross(omega, inertia * omega)) / inertia
    return np.concatenate([position_dot, velocity_dot, rotation_dot.reshape(9), omega_dot])


def integrate_quadrotor_rk4(
    position: Array,
    velocity: Array,
    rotation: Array,
    omega: Array,
    thrust_n: float,
    torque_nm: Array,
    mass_kg: float,
    inertia: Array,
    dt: float,
    gravity_m_s2: float = 9.81,
    force_disturbance_n: Array | None = None,
    linear_drag_n_per_m_s: float = 0.0,
) -> tuple[Array, Array, Array, Array]:
    """Integrate translation and rotation together with a projected RK4 step."""
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError("dt must be positive and finite.")
    y0 = _pack_state(position, velocity, rotation, omega)
    f = lambda y: quadrotor_derivative(
        y, thrust_n, torque_nm, mass_kg, inertia, gravity_m_s2,
        force_disturbance_n=force_disturbance_n,
        linear_drag_n_per_m_s=linear_drag_n_per_m_s,
    )
    k1 = f(y0)
    k2 = f(y0 + 0.5*dt*k1)
    k3 = f(y0 + 0.5*dt*k2)
    k4 = f(y0 + dt*k3)
    y1 = y0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    p1, v1, r1, w1 = _unpack_state(y1)
    return p1, v1, r1, w1


def _precompute_nominal_attitude_reference(ref: dict[str, Array], cfg: TrackingConfig) -> tuple[Array, Array, Array]:
    n = len(ref["time"])
    rotations = np.empty((n, 3, 3))
    for k in range(n):
        _, rotations[k] = acceleration_to_attitude(
            ref["acceleration"][k], float(ref["yaw"][k]), cfg.mass_kg, cfg.gravity_m_s2
        )
    omega_ref = np.zeros((n, 3))
    for k in range(n - 1):
        delta = rotations[k].T @ rotations[k+1]
        omega_previous_frame = rotation_matrix_to_vector(delta) / cfg.sample_period_s
        omega_ref[k+1] = delta.T @ omega_previous_frame
    if n > 1:
        omega_ref[0] = omega_ref[1]
    omega_dot = np.gradient(omega_ref, cfg.sample_period_s, axis=0, edge_order=1)
    return rotations, omega_ref, omega_dot


def _load_model_and_controllers(project_root: Path, data: dict[str, Array], cfg: TrackingConfig) -> tuple[KoopmanModel, Array, KoopmanMPC, dict[str, Any]]:
    training = baseline.transitions_for_split(data, "training")
    fit_start = time.perf_counter()
    model = fit_koopman_edmd(training, order=cfg.order)
    fit_seconds = time.perf_counter() - fit_start
    _, _, _, gain, lqr_diagnostics = baseline._lqr_design(model)
    mpc = KoopmanMPC(
        model,
        torque_limits=baseline.TORQUE_LIMITS,
        fallback_gain=gain,
        horizon=cfg.mpc_horizon,
        q_rotation=5.0,
        q_rate=2.0,
        q_higher=0.2,
        r_weights=(0.08, 0.08, 0.05),
    )
    return model, gain, mpc, {"training_transition_count": len(training), "model_fit_time_seconds": fit_seconds, "lqr_diagnostics": lqr_diagnostics}


def simulate_trajectory_controller(
    controller: str,
    model: KoopmanModel,
    lqr_gain: Array,
    mpc: KoopmanMPC,
    cfg: TrackingConfig,
) -> dict[str, Array | str | float]:
    cfg.validate()
    dt = cfg.sample_period_s
    sub_dt = dt / cfg.integration_substeps
    time_s = np.arange(cfg.steps + 1) * dt
    ref = trajectory_reference(time_s, cfg)
    nominal_rotations, nominal_omega, nominal_omega_dot = _precompute_nominal_attitude_reference(ref, cfg)

    # Start on the reference state at t=0; subsequent errors are due to dynamics
    # and feedback rather than an arbitrary launch transient.
    position = ref["position"][0] + np.asarray(cfg.initial_position_offset_m, dtype=float)
    velocity = ref["velocity"][0] + np.asarray(cfg.initial_velocity_offset_m_s, dtype=float)
    _thrust_reference_initial, rotation_ref0 = acceleration_to_attitude(
        ref["acceleration"][0], float(ref["yaw"][0]), cfg.mass_kg, cfg.gravity_m_s2
    )
    rotation = rotation_ref0 @ euler_zyx_to_rotation(
        *np.deg2rad(cfg.initial_attitude_offset_deg)
    )
    omega = nominal_omega[0] + np.asarray(cfg.initial_rate_offset_rad_s, dtype=float)
    inertia = baseline.INERTIA.copy()
    max_thrust = cfg.max_thrust_multiple_of_weight * cfg.mass_kg * cfg.gravity_m_s2
    torque_limits = baseline.TORQUE_LIMITS.copy()
    z_equilibrium = attitude_lift(np.eye(3), np.zeros(3), model.order)

    n = cfg.steps + 1
    positions = np.zeros((n, 3)); velocities = np.zeros((n, 3)); rotations = np.zeros((n, 3, 3)); omegas = np.zeros((n, 3))
    desired_rotations = np.zeros((n, 3, 3)); desired_thrusts = np.zeros(n)
    applied_torques = np.zeros((cfg.steps, 3)); requested_torques = np.zeros((cfg.steps, 3))
    thrust_commands = np.zeros(cfg.steps, dtype=float)
    thrust_saturated = np.zeros(cfg.steps, dtype=bool); torque_saturated = np.zeros(cfg.steps, dtype=bool)
    mpc_solve_ms = np.zeros(cfg.steps); mpc_success = np.ones(cfg.steps, dtype=bool); mpc_fallback = np.zeros(cfg.steps, dtype=bool)
    att_integral = np.zeros(3)
    previous_desired_rotation = nominal_rotations[0].copy()
    previous_desired_omega = nominal_omega[0].copy()

    kp_pos = np.asarray(cfg.position_kp); kd_pos = np.asarray(cfg.position_kd)
    kp_pid = np.asarray(cfg.pid_kp_nm_rad); ki_pid = np.asarray(cfg.pid_ki_nm_rad_s); kd_pid = np.asarray(cfg.pid_kd_nm_rad_s)
    int_limit = np.asarray(cfg.pid_integral_limit)

    for k, t in enumerate(time_s):
        positions[k], velocities[k], rotations[k], omegas[k] = position, velocity, rotation, omega
        pos_error = ref["position"][k] - position
        vel_error = ref["velocity"][k] - velocity
        acceleration_command = ref["acceleration"][k] + kp_pos*pos_error + kd_pos*vel_error
        force_world = cfg.mass_kg * (acceleration_command + np.array([0.0, 0.0, cfg.gravity_m_s2]))
        thrust_desired = float(np.linalg.norm(force_world))
        if thrust_desired < 1e-8:
            raise RuntimeError("Outer-loop thrust command became too small.")
        b3 = force_world / thrust_desired
        yaw = float(ref["yaw"][k])
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        b2 = np.cross(b3, heading)
        if np.linalg.norm(b2) < 1e-8:
            heading = np.array([np.cos(yaw+1e-3), np.sin(yaw+1e-3), 0.0])
            b2 = np.cross(b3, heading)
        b2 /= np.linalg.norm(b2)
        b1 = np.cross(b2, b3)
        desired_rotation = project_rotation(np.column_stack([b1, b2, b3]))
        desired_rotations[k] = desired_rotation
        thrust_applied = float(np.clip(thrust_desired, 0.0, max_thrust))

        if k == 0:
            desired_omega = nominal_omega[k].copy()
            desired_omega_dot = nominal_omega_dot[k].copy()
        else:
            delta = previous_desired_rotation.T @ desired_rotation
            omega_prev_frame = rotation_matrix_to_vector(delta) / dt
            desired_omega = delta.T @ omega_prev_frame
            desired_omega_dot = (desired_omega - previous_desired_omega) / dt
        feedforward_desired = inertia*desired_omega_dot + np.cross(desired_omega, inertia*desired_omega)
        feedforward = rotation.T @ desired_rotation @ feedforward_desired

        rotation_error_matrix = project_rotation(desired_rotation.T @ rotation)
        omega_reference_in_actual = rotation.T @ desired_rotation @ desired_omega
        omega_error = omega - omega_reference_in_actual

        if k == cfg.steps:
            break

        if controller == "PID":
            rotation_error = rotation_matrix_to_vector(rotation_error_matrix)
            trial_integral = np.clip(att_integral + rotation_error*dt, -int_limit, int_limit)
            trial_torque = feedforward - kp_pid*rotation_error - kd_pid*omega_error - ki_pid*trial_integral
            pushing_high = (trial_torque > torque_limits) & ((-ki_pid*rotation_error) > 0)
            pushing_low = (trial_torque < -torque_limits) & ((-ki_pid*rotation_error) < 0)
            accept = ~(pushing_high | pushing_low)
            att_integral[accept] = trial_integral[accept]
            torque_requested = feedforward - kp_pid*rotation_error - kd_pid*omega_error - ki_pid*att_integral
        elif controller == "Koopman-LQR":
            error_lift = attitude_lift(rotation_error_matrix, omega_error, model.order)
            torque_requested = feedforward - lqr_gain @ (error_lift - z_equilibrium)
        elif controller == "Koopman-MPC":
            mpc_result = mpc.command(rotation_error_matrix, omega_error)
            torque_requested = feedforward + mpc_result.torque
            mpc_solve_ms[k] = mpc_result.solve_time_ms
            mpc_success[k] = mpc_result.success
            mpc_fallback[k] = mpc_result.used_fallback
        else:
            raise ValueError(f"Unknown controller {controller!r}.")

        torque_applied = np.clip(torque_requested, -torque_limits, torque_limits)
        requested_torques[k] = torque_requested
        applied_torques[k] = torque_applied
        thrust_saturated[k] = abs(thrust_desired - thrust_applied) > 1e-8
        torque_saturated[k] = bool(np.any(np.abs(torque_requested) >= torque_limits - 1e-7))

        thrust_commands[k] = thrust_applied
        force_disturbance = np.asarray(cfg.translational_disturbance_force_n, dtype=float) * np.array([
            np.sin(0.7*t), np.cos(0.5*t), np.sin(0.9*t)
        ])
        for _ in range(cfg.integration_substeps):
            position, velocity, rotation, omega = integrate_quadrotor_rk4(
                position, velocity, rotation, omega,
                thrust_applied, torque_applied, cfg.mass_kg, inertia, sub_dt, cfg.gravity_m_s2,
                force_disturbance_n=force_disturbance,
                linear_drag_n_per_m_s=cfg.linear_drag_n_per_m_s,
            )
        previous_desired_rotation = desired_rotation.copy()
        previous_desired_omega = desired_omega.copy()

    return {
        "controller": controller,
        "time": time_s,
        "position": positions,
        "velocity": velocities,
        "rotation": rotations,
        "angles": np.stack([rotation_to_euler_zyx(r) for r in rotations]),
        "omega": omegas,
        "reference_position": ref["position"],
        "reference_velocity": ref["velocity"],
        "reference_acceleration": ref["acceleration"],
        "reference_yaw": ref["yaw"],
        "desired_rotation": desired_rotations,
        "desired_thrust": np.asarray([np.linalg.norm(cfg.mass_kg*(a + [0, 0, cfg.gravity_m_s2])) for a in ref["acceleration"]]),
        "thrust_applied": thrust_commands,
        "torque": applied_torques,
        "requested_torque": requested_torques,
        "thrust_saturated": thrust_saturated,
        "torque_saturated": torque_saturated,
        "mpc_solve_ms": mpc_solve_ms,
        "mpc_success": mpc_success,
        "mpc_fallback": mpc_fallback,
    }


def calculate_metrics(run: dict[str, Any], cfg: TrackingConfig) -> dict[str, Any]:
    t = np.asarray(run["time"]); dt = cfg.sample_period_s
    p = np.asarray(run["position"]); v = np.asarray(run["velocity"]); ang = np.asarray(run["angles"])
    pref = np.asarray(run["reference_position"]); vref = np.asarray(run["reference_velocity"]); yawref = np.asarray(run["reference_yaw"])
    perr = p-pref; verr=v-vref
    distance_error = np.linalg.norm(perr, axis=1)
    velocity_error = np.linalg.norm(verr, axis=1)
    yaw_error = _wrap(ang[:,2]-yawref)
    position_rmse_axis = np.sqrt(np.mean(perr**2, axis=0))
    velocity_rmse_axis = np.sqrt(np.mean(verr**2, axis=0))
    # Rotation error uses the desired rotation constructed by outer-loop state feedback.
    desired_rotations = np.asarray(run["desired_rotation"])
    rot_error_deg = np.rad2deg(np.stack([
        rotation_matrix_to_vector(desired_rotations[k].T @ np.asarray(run["rotation"])[k])
        for k in range(len(t))
    ]))
    torque = np.asarray(run["torque"])
    mpc_ms = np.asarray(run["mpc_solve_ms"])
    mpc_ok = np.asarray(run["mpc_success"])
    mpc_fallback = np.asarray(run["mpc_fallback"])
    # Use trapezoidal integration on the recorded state timeline.
    position_iae = float(np.sum(0.5 * (distance_error[1:] + distance_error[:-1]) * np.diff(t)))
    control_effort = float(np.sum(torque**2)*dt)
    finite = all(np.all(np.isfinite(np.asarray(run[key]))) for key in ("position", "velocity", "angles", "omega", "torque"))
    return {
        "controller": run["controller"],
        "position_rmse_3d_m": float(np.sqrt(np.mean(distance_error**2))),
        "position_rmse_x_m": float(position_rmse_axis[0]),
        "position_rmse_y_m": float(position_rmse_axis[1]),
        "position_rmse_z_m": float(position_rmse_axis[2]),
        "position_mean_euclidean_error_m": float(np.mean(distance_error)),
        "position_iae_m_s": position_iae,
        "position_peak_euclidean_error_m": float(np.max(distance_error)),
        "velocity_rmse_3d_m_s": float(np.sqrt(np.mean(velocity_error**2))),
        "velocity_rmse_x_m_s": float(velocity_rmse_axis[0]),
        "velocity_rmse_y_m_s": float(velocity_rmse_axis[1]),
        "velocity_rmse_z_m_s": float(velocity_rmse_axis[2]),
        "altitude_rmse_m": float(position_rmse_axis[2]),
        "yaw_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(yaw_error**2)))),
        "attitude_setpoint_error_rmse_deg": float(np.sqrt(np.mean(rot_error_deg**2))),
        "peak_tilt_deg": float(np.rad2deg(np.max(np.arccos(np.clip(np.asarray(run["rotation"])[:,2,2],-1,1))))),
        "control_effort_integral_nm2_s": control_effort,
        "torque_saturated_sample_percent": float(100*np.mean(run["torque_saturated"])),
        "thrust_saturated_sample_percent": float(100*np.mean(run["thrust_saturated"])),
        "finite_trajectory": bool(finite),
        "mpc_solve_time_mean_ms": float(np.mean(mpc_ms)) if run["controller"]=="Koopman-MPC" else None,
        "mpc_solver_success_percent": float(100*np.mean(mpc_ok)) if run["controller"]=="Koopman-MPC" else None,
        "mpc_fallback_count": int(np.sum(mpc_fallback)) if run["controller"]=="Koopman-MPC" else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty results.")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols); writer.writeheader(); writer.writerows(rows)


def _json_safe(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): _json_safe(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [_json_safe(v) for v in x]
    if isinstance(x, np.ndarray): return _json_safe(x.tolist())
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (np.floating,float)): return float(x) if np.isfinite(x) else None
    return x


def _save_fig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(tmp, dpi=180, bbox_inches="tight")
        tmp.replace(path)
    finally:
        plt.close(fig)
        if tmp.exists(): tmp.unlink()


def _plot_position(run_by_name: dict[str, dict[str, Any]], path: Path) -> None:
    names = ("X", "Y", "Z")
    fig, axes = plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
    first = next(iter(run_by_name.values()))
    t = np.asarray(first["time"]); pref=np.asarray(first["reference_position"])
    for i, ax in enumerate(axes):
        ax.plot(t,pref[:,i],"k--",label="Reference",linewidth=1.5)
        for name,run in run_by_name.items(): ax.plot(t,np.asarray(run["position"])[:,i],label=name)
        ax.set_ylabel(f"{names[i]} (m)"); ax.grid(True,alpha=.3)
        if i==0: ax.legend(ncol=2)
    axes[-1].set_xlabel("Time (s)"); fig.suptitle("Six-DoF trajectory tracking: position response")
    _save_fig(fig,path)


def _plot_3d(run_by_name: dict[str, dict[str, Any]], path: Path) -> None:
    fig=plt.figure(figsize=(9,7),constrained_layout=True); ax=fig.add_subplot(111,projection="3d")
    first=next(iter(run_by_name.values())); pref=np.asarray(first["reference_position"])
    ax.plot(pref[:,0],pref[:,1],pref[:,2],"k--",label="Reference",linewidth=2)
    for name,run in run_by_name.items():
        p=np.asarray(run["position"]); ax.plot(p[:,0],p[:,1],p[:,2],label=name,linewidth=1.3)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("3-D trajectory tracking"); ax.legend()
    _save_fig(fig,path)


def _plot_position_error(run_by_name: dict[str, dict[str, Any]], path: Path) -> None:
    fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for name,run in run_by_name.items():
        err=np.asarray(run["position"])-np.asarray(run["reference_position"])
        ax.plot(run["time"],np.linalg.norm(err,axis=1),label=name)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("3-D position error (m)"); ax.set_title("Trajectory tracking error")
    ax.grid(True,alpha=.3); ax.legend()
    _save_fig(fig,path)


def _plot_attitude_and_inputs(run_by_name: dict[str, dict[str, Any]], figure_dir: Path) -> None:
    first=next(iter(run_by_name.values())); t=np.asarray(first["time"]); names=("Roll","Pitch","Yaw")
    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
    for i,ax in enumerate(axes):
        for name,run in run_by_name.items(): ax.plot(t,np.rad2deg(np.asarray(run["angles"])[:,i]),label=name)
        ax.set_ylabel(f"{names[i]} (deg)"); ax.grid(True,alpha=.3)
        if i==0: ax.legend(ncol=2)
    axes[-1].set_xlabel("Time (s)"); fig.suptitle("Attitude response during position tracking")
    _save_fig(fig,figure_dir/"attitude_response.png")

    fig,axes=plt.subplots(3,1,figsize=(11,8),sharex=True,constrained_layout=True)
    for i,ax in enumerate(axes):
        for name,run in run_by_name.items(): ax.plot(np.asarray(run["time"][:-1]),np.asarray(run["torque"])[:,i],label=name)
        ax.set_ylabel(("τx","τy","τz")[i]+" (N m)"); ax.grid(True,alpha=.3)
        if i==0: ax.legend(ncol=2)
    axes[-1].set_xlabel("Time (s)"); fig.suptitle("Body torque commands during trajectory tracking")
    _save_fig(fig,figure_dir/"control_torque.png")


def run_tracking_study(project_root: Path, cfg: TrackingConfig) -> dict[str, Any]:
    cfg.validate()
    result_root=project_root/"results"/"trajectory_tracking"
    tables=result_root/"tables"; figures=result_root/"figures"; models=result_root/"models"
    for folder in (tables,figures,models): folder.mkdir(parents=True,exist_ok=True)
    dataset_path=project_root/"results"/"data"/"attitude_dataset.npz"
    data=baseline.load_dataset(dataset_path)
    model,gain,mpc,training_info=_load_model_and_controllers(project_root,data,cfg)
    run_by_name={}
    for controller in ("PID","Koopman-LQR","Koopman-MPC"):
        run_by_name[controller]=simulate_trajectory_controller(controller,model,gain,mpc,cfg)
    metrics=[calculate_metrics(run,cfg) for run in run_by_name.values()]
    _write_csv(tables/"trajectory_tracking_metrics.csv",metrics)
    _plot_position(run_by_name,figures/"position_tracking.png")
    _plot_3d(run_by_name,figures/"trajectory_3d.png")
    _plot_position_error(run_by_name,figures/"position_error.png")
    _plot_attitude_and_inputs(run_by_name,figures)

    np.savez_compressed(models/"trajectory_tracking_logs.npz", **{
        f"{name.lower().replace('-', '_')}_{key}": np.asarray(value)
        for name,run in run_by_name.items() for key,value in run.items()
        if isinstance(value,(np.ndarray,list,tuple)) and np.asarray(value).dtype.kind in "biufc"
    })
    summary={
        "method": {
            "model_reference": "Bayrak and Efe, Disturbance observer-based control schemes for quadrotors: a tutorial, equations (1)-(4), translational and rotational dynamics and Euler/body-rate relationships.",
            "architecture": "One shared nonlinear 6-DoF rigid-body plant and one shared position outer loop; PID, Koopman-LQR and Koopman-MPC form the inner attitude loop.",
            "ENU_dynamics": "p_dot=v; v_dot=(T/m)*R[:,2]-g*e_z; R_dot=R*skew(omega); J*omega_dot=tau-cross(omega,J*omega).",
            "outer_loop": "a_cmd=a_ref+Kp_position*(p_ref-p)+Kd_velocity*(v_ref-v); thrust and desired attitude are calculated from m*(a_cmd+g*e_z) and reference yaw.",
            "inner_loop_comparison": "PID uses rotation-log attitude error and body-rate error. Koopman-LQR and Koopman-MPC use the same lifted attitude error coordinates and a common rigid-body reference-torque feed-forward term.",
            "trajectory": "Analytic 3-D Lissajous/figure-eight reference with variable altitude and yaw; the same time series is used for all three controllers.",
            "limitations": "This is an independent model-in-the-loop six-DoF simulation, not one of the four platform adapters. Aerodynamics, rotor-by-rotor allocation, motor lag, ground effect and sensor noise are not modeled in this extension.",
        },
        "configuration": asdict(cfg),
        "plant": {"mass_kg":cfg.mass_kg,"inertia_kg_m2":baseline.INERTIA.tolist(),"gravity_m_s2":cfg.gravity_m_s2,"max_thrust_n":cfg.max_thrust_multiple_of_weight*cfg.mass_kg*cfg.gravity_m_s2,"torque_limits_nm":baseline.TORQUE_LIMITS.tolist(),"integration_substeps":cfg.integration_substeps,"linear_drag_n_per_m_s":cfg.linear_drag_n_per_m_s,"translational_disturbance_force_amplitudes_n":list(cfg.translational_disturbance_force_n)},
        "training_model": training_info,
        "reference": {"trajectory":"3-D Lissajous/figure-eight; 15 seconds; 20 Hz controller update by default","initial_state":"The reference is smooth at t=0; the plant starts with the same declared position, velocity, attitude, and body-rate offsets for all controllers, so the run measures recovery and trajectory tracking rather than perfect-state initialization."},
        "metric_definitions": {
            "position_rmse_3d_m":"sqrt(mean(||position-reference||_2^2)).",
            "position_rmse_axis_m":"Per-axis position RMSE in meters.",
            "position_iae_m_s":"Trapezoidal integral of Euclidean position error magnitude over the run.",
            "velocity_rmse_3d_m_s":"sqrt(mean(||velocity-reference_velocity||_2^2)).",
            "yaw_rmse_deg":"RMSE of wrapped yaw error, converted to degrees.",
            "attitude_setpoint_error_rmse_deg":"RMSE of the rotation-log error vector in degrees; measures inner-loop tracking of the outer-loop attitude setpoint.",
            "torque_saturated_sample_percent":"Percent of torque-control samples with any channel at its saturation limit.",
            "thrust_saturated_sample_percent":"Percent of samples where desired collective thrust was clipped to its configured range.",
            "control_effort_integral_nm2_s":"sum of squared body torque commands times controller sample period.",
        },
        "metrics": metrics,
        "files":["tables/trajectory_tracking_metrics.csv","figures/position_tracking.png","figures/trajectory_3d.png","figures/position_error.png","figures/attitude_response.png","figures/control_torque.png","models/trajectory_tracking_logs.npz"],
    }
    with (result_root/"trajectory_tracking_summary.json").open("w",encoding="utf-8") as f:
        json.dump(_json_safe(summary),f,indent=2,allow_nan=False)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(description="Run the shared six-DoF trajectory-tracking extension.")
    parser.add_argument("--project-root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    cfg=TrackingConfig()
    summary=run_tracking_study(args.project_root.resolve(),cfg)
    print("Six-DoF trajectory-tracking study completed.")
    print("Shared dataset reused; no data regeneration performed.")
    print(f"Controllers: PID, Koopman-LQR, Koopman-MPC")
    for row in summary["metrics"]:
        print(f"{row['controller']}: 3D position RMSE={row['position_rmse_3d_m']:.4f} m; velocity RMSE={row['velocity_rmse_3d_m_s']:.4f} m/s; yaw RMSE={row['yaw_rmse_deg']:.3f} deg")
    print(f"Results: {(args.project_root.resolve()/'results'/'trajectory_tracking')}")

if __name__=="__main__":
    main()
