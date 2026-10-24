"""Live MuJoCo controller demonstration for the project quadrotor.

Runs the same saved order-5 Koopman model and the same three controllers used
in the accepted PyBullet benchmark: PID, Koopman-LQR, and Koopman-MPC. Each trial
starts from the same initial position and attitude and uses the same bounded
world-position hold outer loop. The MuJoCo model is a transparent CF2X-
parameterized rigid-body model, not a claim of physics-engine equivalence.

Primary metrics (and only primary metrics):
  1. attitude-tracking RMSE (degrees)
  2. settling time within 2 degrees (seconds)
  3. 3D position RMSE (metres)

Example:
  python -m src.mujoco_controller_demo --self-check
  python -m src.mujoco_controller_demo --controller all --duration 20
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.position_hold import PositionHoldConfig, compute_position_hold_command
from src.quadrotor_mixer import CF2XMixer
from src.simulation_contract import SimulationObservation
from src.simulators.mujoco_adapter import (
    MuJoCoAdapterConfig,
    MuJoCoQuadrotorAdapter,
    rotation_to_euler_zyx,
)

Array = np.ndarray
CONTROLLERS = ("PID", "Koopman-LQR", "Koopman-MPC")
MODEL_TORQUE_LIMITS_NM = np.array([0.12, 0.12, 0.06], dtype=float)
MODEL_INERTIA_KG_M2 = np.array([0.021, 0.024, 0.039], dtype=float)


@dataclass(frozen=True)
class MuJoCoDemoConfig:
    duration_s: float = 20.0
    initial_position_world_m: tuple[float, float, float] = (0.0, 0.0, 1.0)
    initial_attitude_deg: tuple[float, float, float] = (10.0, 8.0, 6.0)
    settling_tolerance_deg: float = 2.0
    selected_order: int = 5
    mpc_horizon: int = 20
    mpc_q_rotation: float = 5.0
    mpc_q_rate: float = 2.0
    mpc_q_higher: float = 0.2
    mpc_r_weights: tuple[float, float, float] = (0.01, 0.01, 0.008)
    pid_kp: tuple[float, float, float] = (0.35, 0.35, 0.18)
    pid_ki: tuple[float, float, float] = (0.002, 0.002, 0.001)
    pid_kd: tuple[float, float, float] = (0.16, 0.16, 0.12)
    pid_integral_limit: tuple[float, float, float] = (2.0, 2.0, 2.0)
    position_kp: tuple[float, float, float] = (1.4, 1.4, 2.0)
    position_kd: tuple[float, float, float] = (1.8, 1.8, 1.7)
    max_lateral_acceleration_m_s2: float = 1.6
    max_vertical_acceleration_m_s2: float = 1.5
    max_tilt_deg: float = 12.0
    safety_min_altitude_m: float = 0.15
    safety_max_altitude_m: float = 5.0
    safety_max_attitude_error_deg: float = 100.0
    safety_max_body_rate_rad_s: float = 20.0

    def validate(self, dt: float) -> None:
        if not np.isfinite(self.duration_s) or self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive and finite.")
        if not np.isclose(self.duration_s / dt, round(self.duration_s / dt), atol=1e-8, rtol=0.0):
            raise ValueError(f"duration_s must be a multiple of the {dt:g}s control period.")
        if self.selected_order < 2 or self.mpc_horizon < 1:
            raise ValueError("selected_order must be >=2 and mpc_horizon must be >=1.")
        if np.asarray(self.initial_position_world_m).shape != (3,) or np.asarray(self.initial_attitude_deg).shape != (3,):
            raise ValueError("Initial position and attitude must each contain three values.")
        for name in ("pid_kp", "pid_ki", "pid_kd", "pid_integral_limit", "position_kp", "position_kd"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain three finite values.")
        for name in ("pid_kp", "pid_ki", "pid_kd"):
            if np.any(np.asarray(getattr(self, name), dtype=float) < 0.0):
                raise ValueError(f"{name} must be non-negative.")
        if np.any(np.asarray(self.pid_integral_limit, dtype=float) <= 0.0):
            raise ValueError("PID integral limits must be positive.")
        if any(np.any(np.asarray(getattr(self, name), dtype=float) < 0.0) for name in ("position_kp", "position_kd")):
            raise ValueError("Position gains must be non-negative.")
        if not (0 < self.max_tilt_deg <= 30):
            raise ValueError("max_tilt_deg must be in (0, 30].")
        if self.settling_tolerance_deg <= 0:
            raise ValueError("settling_tolerance_deg must be positive.")


def wrap_angle(angle: Array | float) -> Array | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def geodesic_rotation_error_deg(rotation: Array) -> float:
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Rotation error requires a finite 3x3 matrix.")
    cos_theta = float(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_theta)))


def angular_velocity_between_rotations(previous: Array, current: Array, dt: float) -> Array:
    """Finite-difference SO(3) log-map estimate of desired body angular rates."""
    previous = np.asarray(previous, dtype=float)
    current = np.asarray(current, dtype=float)
    if previous.shape != (3, 3) or current.shape != (3, 3) or dt <= 0.0:
        raise ValueError("Rotations must be 3x3 and dt must be positive.")
    delta = previous.T @ current
    theta = float(np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))
    skew = np.array([
        delta[2, 1] - delta[1, 2],
        delta[0, 2] - delta[2, 0],
        delta[1, 0] - delta[0, 1],
    ], dtype=float)
    if theta < 1e-7 or abs(np.sin(theta)) < 1e-8:
        rate_previous = 0.5 * skew / dt
    else:
        rate_previous = theta * skew / (2.0 * np.sin(theta) * dt)
    return delta.T @ rate_previous


def load_controller_assets(project_root: Path, order: int) -> tuple[Any, Array, dict[str, Any]]:
    path = project_root / "results" / "models" / "attitude_models.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Required saved model not found: {path}. This MuJoCo demo never silently refits the model."
        )
    with np.load(path, allow_pickle=False) as saved:
        required = (f"A_N{order}", f"B_N{order}", "lqr_gain", "selected_control_order")
        missing = [key for key in required if key not in saved.files]
        if missing:
            raise ValueError(f"Saved model artifact is missing {missing}.")
        A = np.asarray(saved[f"A_N{order}"], dtype=float)
        B = np.asarray(saved[f"B_N{order}"], dtype=float)
        gain = np.asarray(saved["lqr_gain"], dtype=float)
        selected = int(np.asarray(saved["selected_control_order"]).reshape(()))
    if selected != order:
        raise ValueError(f"Saved LQR gain uses order {selected}; the requested controller order is {order}.")
    expected_n = 9 * order
    if A.shape != (expected_n, expected_n) or B.shape != (expected_n, 3) or gain.shape != (3, expected_n):
        raise ValueError(f"Saved A/B/K dimensions are inconsistent for order N={order}.")
    if not all(np.all(np.isfinite(x)) for x in (A, B, gain)):
        raise ValueError("Saved A/B/K contains non-finite values.")
    model = SimpleNamespace(A=A, B=B, order=order)
    metadata = {
        "model_artifact": path.relative_to(project_root).as_posix(),
        "selected_order": order,
        "A_shape": list(A.shape),
        "B_shape": list(B.shape),
        "lqr_gain_shape": list(gain.shape),
        "identified_model_inertia_kg_m2": MODEL_INERTIA_KG_M2.tolist(),
    }
    return model, gain, metadata


def _position_reference(obs: SimulationObservation, cfg: MuJoCoDemoConfig, adapter: MuJoCoQuadrotorAdapter) -> tuple[Array, float, Array]:
    hold_cfg = PositionHoldConfig(
        position_kp=cfg.position_kp,
        velocity_kd=cfg.position_kd,
        max_lateral_acceleration_m_s2=cfg.max_lateral_acceleration_m_s2,
        max_vertical_acceleration_m_s2=cfg.max_vertical_acceleration_m_s2,
        max_tilt_deg=cfg.max_tilt_deg,
    )
    command = compute_position_hold_command(
        obs.position_world_m, obs.velocity_world_m_s,
        np.asarray(adapter.config.target_position_world_m, dtype=float),
        adapter.mass_kg, adapter.gravity_m_s2, hold_cfg,
        target_yaw_rad=0.0,
    )
    return command.desired_rotation_body_to_world, command.collective_thrust_n, command.acceleration_command_world_m_s2


def controller_model_torque(
    controller: str,
    rotation_body_to_world: Array,
    body_rates_rad_s: Array,
    model: Any,
    lqr_gain: Array,
    mpc: Any | None,
    pid_integral: Array,
    dt: float,
    cfg: MuJoCoDemoConfig,
    desired_rotation: Array,
    desired_body_rates: Array,
) -> tuple[Array, Array, Array, dict[str, Any]]:
    """Shared PID / Koopman-LQR / Koopman-MPC control laws in model coordinates."""
    actual_rotation = np.asarray(rotation_body_to_world, dtype=float)
    actual_rates = np.asarray(body_rates_rad_s, dtype=float).reshape(3)
    desired_rotation = np.asarray(desired_rotation, dtype=float)
    desired_body_rates = np.asarray(desired_body_rates, dtype=float).reshape(3)
    if actual_rotation.shape != (3, 3) or desired_rotation.shape != (3, 3):
        raise ValueError("Actual and desired rotations must be 3x3 matrices.")
    if not all(np.all(np.isfinite(v)) for v in (actual_rotation, actual_rates, desired_rotation, desired_body_rates)):
        raise FloatingPointError("Non-finite controller input detected.")
    error_rotation = desired_rotation.T @ actual_rotation
    desired_rate_in_actual_frame = error_rotation.T @ desired_body_rates
    rate_error = actual_rates - desired_rate_in_actual_frame
    next_integral = np.asarray(pid_integral, dtype=float).reshape(3).copy()
    diagnostics: dict[str, Any] = {"mpc_solve_time_ms": 0.0, "mpc_success": True, "mpc_fallback": False}

    if controller == "PID":
        angle_error = np.asarray(wrap_angle(-rotation_to_euler_zyx(error_rotation)), dtype=float)
        kp, ki, kd = (np.asarray(cfg.pid_kp), np.asarray(cfg.pid_ki), np.asarray(cfg.pid_kd))
        limits = MODEL_TORQUE_LIMITS_NM
        candidate = np.clip(next_integral + angle_error * dt, -np.asarray(cfg.pid_integral_limit), np.asarray(cfg.pid_integral_limit))
        trial = kp * angle_error + ki * candidate - kd * rate_error
        accept = ~(((trial > limits) & (angle_error > 0.0)) | ((trial < -limits) & (angle_error < 0.0)))
        next_integral[accept] = candidate[accept]
        requested = kp * angle_error + ki * next_integral - kd * rate_error
    elif controller == "Koopman-LQR":
        from src.attitude_core import attitude_lift
        lifted = attitude_lift(error_rotation, rate_error, model.order)
        equilibrium = attitude_lift(np.eye(3), np.zeros(3), model.order)
        requested = -np.asarray(lqr_gain) @ (lifted - equilibrium)
    elif controller == "Koopman-MPC":
        if mpc is None:
            raise ValueError("Koopman-MPC was selected but an MPC controller was not created.")
        result = mpc.command(error_rotation, rate_error)
        requested = np.asarray(result.torque, dtype=float)
        diagnostics.update({
            "mpc_solve_time_ms": float(result.solve_time_ms),
            "mpc_success": bool(result.success),
            "mpc_fallback": bool(result.used_fallback),
        })
    else:
        raise ValueError(f"Unknown controller {controller!r}.")

    requested = np.asarray(requested, dtype=float).reshape(3)
    if not np.all(np.isfinite(requested)):
        raise FloatingPointError(f"{controller} generated a non-finite torque request.")
    limited = np.clip(requested, -MODEL_TORQUE_LIMITS_NM, MODEL_TORQUE_LIMITS_NM)
    diagnostics["controller_torque_limited"] = bool(np.any(np.abs(limited - requested) > 1e-12))
    diagnostics["pid_integral_next"] = next_integral
    return requested, limited, rate_error, diagnostics


def _append_observation(trace: dict[str, list[Any]], obs: SimulationObservation) -> None:
    trace["time_s"].append(float(obs.time_s))
    trace["rotation_body_to_world"].append(np.asarray(obs.rotation_body_to_world, dtype=float).copy())
    trace["body_rates_rad_s"].append(np.asarray(obs.body_rates_rad_s, dtype=float).copy())
    trace["position_world_m"].append(np.asarray(obs.position_world_m, dtype=float).copy())
    trace["velocity_world_m_s"].append(np.asarray(obs.velocity_world_m_s, dtype=float).copy())


def _safety_stop(obs: SimulationObservation, attitude_error_deg: float, cfg: MuJoCoDemoConfig) -> str | None:
    if not all(np.all(np.isfinite(v)) for v in (obs.position_world_m, obs.velocity_world_m_s, obs.body_rates_rad_s)):
        return "non_finite_state"
    if obs.position_world_m[2] <= cfg.safety_min_altitude_m:
        return "minimum_altitude_limit"
    if obs.position_world_m[2] >= cfg.safety_max_altitude_m:
        return "maximum_altitude_limit"
    if attitude_error_deg >= cfg.safety_max_attitude_error_deg:
        return "attitude_excursion_limit"
    if np.linalg.norm(obs.body_rates_rad_s) >= cfg.safety_max_body_rate_rad_s:
        return "body_rate_limit"
    return None


def calculate_primary_metrics(trace: dict[str, Array], cfg: MuJoCoDemoConfig) -> dict[str, Any]:
    """Calculate exactly the three headline measures from measured MuJoCo states."""
    time_s = np.asarray(trace["time_s"], dtype=float)
    actual = np.asarray(trace["rotation_body_to_world"], dtype=float)
    desired = np.asarray(trace["desired_rotation_body_to_world"], dtype=float)
    position = np.asarray(trace["position_world_m"], dtype=float)
    target = np.asarray(trace["position_target_world_m"], dtype=float)
    errors = np.asarray([geodesic_rotation_error_deg(d.T @ a) for d, a in zip(desired, actual, strict=True)])
    position_distances = np.linalg.norm(target - position, axis=1)
    below = errors <= cfg.settling_tolerance_deg
    settling_index = None
    suffix_pass = np.logical_and.accumulate(below[::-1])[::-1]
    candidates = np.flatnonzero(suffix_pass)
    if candidates.size:
        settling_index = int(candidates[0])
    return {
        "controller": str(trace["controller"]),
        "attitude_tracking_rmse_deg": float(np.sqrt(np.mean(errors**2))),
        "settling_time_s": None if settling_index is None else float(time_s[settling_index]),
        "position_rmse_3d_m": float(np.sqrt(np.mean(position_distances**2))),
        "trial_status": str(trace["trial_status"]),
    }


def run_live_trial(
    controller: str,
    adapter: MuJoCoQuadrotorAdapter,
    model: Any,
    lqr_gain: Array,
    cfg: MuJoCoDemoConfig,
    *,
    gui: bool,
    realtime: bool,
) -> tuple[dict[str, Array], dict[str, Any]]:
    dt = adapter.config.control_period_s
    cfg.validate(dt)
    initial_pos = np.asarray(cfg.initial_position_world_m, dtype=float)
    initial_rpy = np.deg2rad(np.asarray(cfg.initial_attitude_deg, dtype=float))
    obs = adapter.reset(initial_position_world_m=initial_pos, initial_euler_zyx_rad=initial_rpy)
    if gui:
        adapter.open_viewer()

    mpc = None
    if controller == "Koopman-MPC":
        from src.koopman_mpc import KoopmanMPC
        mpc = KoopmanMPC(
            model,
            torque_limits=MODEL_TORQUE_LIMITS_NM,
            fallback_gain=lqr_gain,
            horizon=cfg.mpc_horizon,
            q_rotation=cfg.mpc_q_rotation,
            q_rate=cfg.mpc_q_rate,
            q_higher=cfg.mpc_q_higher,
            r_weights=cfg.mpc_r_weights,
            max_iterations=80,
        )

    target = np.asarray(adapter.config.target_position_world_m, dtype=float)
    trace_lists: dict[str, list[Any]] = {
        "time_s": [], "rotation_body_to_world": [], "body_rates_rad_s": [],
        "position_world_m": [], "velocity_world_m_s": [], "desired_rotation_body_to_world": [],
        "desired_body_rates_rad_s": [], "position_target_world_m": [],
        "motor_rpm": [], "model_torque_nm": [], "physical_torque_nm": [],
        "saturated": [], "mpc_fallback": [], "mpc_success": [],
    }
    _append_observation(trace_lists, obs)
    trace_lists["position_target_world_m"].append(target.copy())
    pid_integral = np.zeros(3, dtype=float)
    previous_desired: Array | None = None
    wall_start = time.perf_counter()
    next_deadline = wall_start
    trial_status = "duration_complete"
    steps = int(round(cfg.duration_s / dt))

    for step_index in range(steps):
        desired_rotation, collective_thrust, _ = _position_reference(obs, cfg, adapter)
        desired_rates = np.zeros(3, dtype=float) if previous_desired is None else angular_velocity_between_rotations(previous_desired, desired_rotation, dt)
        requested_model, limited_model, _, diagnostics = controller_model_torque(
            controller, obs.rotation_body_to_world, obs.body_rates_rad_s,
            model, lqr_gain, mpc, pid_integral, dt, cfg, desired_rotation, desired_rates,
        )
        pid_integral = np.asarray(diagnostics["pid_integral_next"], dtype=float)
        physical_torque = (
            np.asarray(adapter.simulator_inertia_kg_m2, dtype=float) / MODEL_INERTIA_KG_M2
        ) * limited_model
        mixer_out = adapter.mixer.allocate(physical_torque, collective_thrust)

        trace_lists["desired_rotation_body_to_world"].append(desired_rotation.copy())
        trace_lists["desired_body_rates_rad_s"].append(desired_rates.copy())
        trace_lists["motor_rpm"].append(np.asarray(mixer_out.motor_rpm, dtype=float).copy())
        trace_lists["model_torque_nm"].append(limited_model.copy())
        trace_lists["physical_torque_nm"].append(np.asarray(physical_torque, dtype=float).copy())
        trace_lists["saturated"].append(bool(mixer_out.saturated or diagnostics["controller_torque_limited"]))
        trace_lists["mpc_fallback"].append(bool(diagnostics["mpc_fallback"]))
        trace_lists["mpc_success"].append(bool(diagnostics["mpc_success"]))
        previous_desired = desired_rotation.copy()

        next_obs = adapter.step(mixer_out.motor_rpm, duration_s=dt)
        _append_observation(trace_lists, next_obs)
        trace_lists["position_target_world_m"].append(target.copy())
        obs = next_obs

        current_att_error = geodesic_rotation_error_deg(desired_rotation.T @ obs.rotation_body_to_world)
        current_pos_error = float(np.linalg.norm(target - obs.position_world_m))
        if gui:
            adapter.update_presentation(
                controller, obs,
                desired_rotation_body_to_world=desired_rotation,
                target_position_world_m=target,
                attitude_error_deg=current_att_error,
                position_error_m=current_pos_error,
                elapsed_s=float(obs.time_s), duration_s=cfg.duration_s,
            )
        if (step_index + 1) % int(round(1.0 / dt)) == 0:
            print(
                f"\r{controller:12s} t={obs.time_s:5.1f}/{cfg.duration_s:4.0f}s | "
                f"attitude error={current_att_error:5.2f} deg | target distance={current_pos_error:5.3f} m",
                end="", flush=True,
            )
        trial_status = _safety_stop(obs, current_att_error, cfg) or "duration_complete"
        if trial_status != "duration_complete":
            print(f"\nSafety stop for {controller}: {trial_status}.")
            break
        next_deadline += dt
        if realtime:
            time.sleep(max(0.0, next_deadline - time.perf_counter()))

    print()
    # Append the final state's exact outer-loop reference so actual and desired
    # arrays have matching N-state lengths. The final sample carries a zero
    # desired-rate value, which is appropriate after the recorded control stops.
    final_desired, _, _ = _position_reference(obs, cfg, adapter)
    trace_lists["desired_rotation_body_to_world"].append(final_desired.copy())
    trace_lists["desired_body_rates_rad_s"].append(np.zeros(3, dtype=float))

    trace = {key: np.asarray(value) for key, value in trace_lists.items()}
    trace["controller"] = controller
    trace["backend"] = adapter.name
    trace["trial_status"] = trial_status
    trace["control_period_s"] = np.array(dt)
    trace["position_error_norm_m"] = np.linalg.norm(target[None, :] - trace["position_world_m"], axis=1)
    trace["attitude_tracking_error_deg"] = np.asarray([
        geodesic_rotation_error_deg(d.T @ a) for d, a in zip(
            trace["desired_rotation_body_to_world"], trace["rotation_body_to_world"], strict=True
        )
    ], dtype=float)
    trace["wall_duration_s"] = np.array(time.perf_counter() - wall_start)
    return trace, {"mpc_fallback_count": int(np.sum(trace["mpc_fallback"])),
                   "mpc_solver_success_percent": float(100.0 * np.mean(trace["mpc_success"])) if controller == "Koopman-MPC" else None}


def _write_validation_figure(traces: dict[str, dict[str, Array]], metrics: list[dict[str, Any]], path: Path) -> None:
    colors = {"PID": "#3388dd", "Koopman-LQR": "#ef8b2c", "Koopman-MPC": "#20a56a"}
    metric_by_name = {row["controller"]: row for row in metrics}
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True, constrained_layout=True)
    for controller, trace in traces.items():
        t = np.asarray(trace["time_s"], dtype=float)
        attitude = np.asarray(trace["attitude_tracking_error_deg"], dtype=float)
        position = np.asarray(trace["position_error_norm_m"], dtype=float)
        row = metric_by_name[controller]
        settle = row["settling_time_s"]
        settle_text = "not settled" if settle is None else f"{float(settle):.2f} s"
        axes[0].plot(t, attitude, color=colors[controller], linewidth=1.9,
                     label=f"{controller} | RMSE {row['attitude_tracking_rmse_deg']:.2f}° | settle {settle_text}")
        axes[1].plot(t, position, color=colors[controller], linewidth=1.9,
                     label=f"{controller} | RMSE {row['position_rmse_3d_m']:.3f} m")
    axes[0].axhline(2.0, color="#333333", linestyle="--", linewidth=1.0, label="2° settling tolerance")
    axes[0].set_ylabel("Attitude tracking error (deg)")
    axes[0].set_title("MuJoCo controller validation — measured free-body observations")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[1].set_ylabel("3D distance to target (m)")
    axes[1].set_xlabel("Simulation time (s)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _save_trace_bundle(traces: dict[str, dict[str, Array]], path: Path) -> None:
    numeric: dict[str, Array] = {}
    for controller, trace in traces.items():
        prefix = controller.lower().replace("-", "_")
        for key, value in trace.items():
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
                numeric[f"{prefix}__{key}"] = value
    np.savez_compressed(path, **numeric)


def _self_check(project_root: Path, order: int) -> int:
    print("MuJoCo preflight: loading the native physics runtime and compiling the CF2X MJCF model...")
    adapter = MuJoCoQuadrotorAdapter()
    try:
        # Ensure that the exact trained artifact needed by all controllers is present and shaped correctly.
        load_controller_assets(project_root, order)
        print(f"  MJCF compiled: bodies={adapter.model.nbody}, geoms={adapter.model.ngeom}, timestep={adapter.model.opt.timestep:.7f}s")
        hover_rpm = math.sqrt(adapter.mass_kg * adapter.gravity_m_s2 / (4.0 * adapter.config.thrust_coefficient_n_per_rpm2))
        adapter.reset(initial_position_world_m=np.array([0.0, 0.0, 1.0]), initial_euler_zyx_rad=np.zeros(3))
        initial = adapter.observe()
        adapter.step(np.full(4, hover_rpm), duration_s=0.5)
        hover = adapter.observe()
        altitude_delta = abs(float(hover.position_world_m[2] - initial.position_world_m[2]))
        level_error = geodesic_rotation_error_deg(hover.rotation_body_to_world)
        print(f"  Hover balance: altitude drift={altitude_delta:.4f}m; attitude change={level_error:.3f}deg")
        if altitude_delta > 0.02 or level_error > 1.0:
            raise RuntimeError("Balanced-hover preflight failed. Do not start the full controller comparison.")

        inertia_ratio = adapter.simulator_inertia_kg_m2 / MODEL_INERTIA_KG_M2
        for axis, axis_name in enumerate(("roll X", "pitch Y", "yaw Z")):
            adapter.reset(initial_position_world_m=np.array([0.0, 0.0, 1.0]), initial_euler_zyx_rad=np.zeros(3))
            torque_model = np.zeros(3, dtype=float)
            torque_model[axis] = 0.01
            torque_physical = inertia_ratio * torque_model
            mix = adapter.mixer.allocate(torque_physical, adapter.mass_kg * adapter.gravity_m_s2)
            before = adapter.observe()
            # Use an integer number of native physics steps. 0.08 s is not
            # divisible by the 1/240 s timestep and caused the preflight to
            # abort before it could test the three motor-axis signs.
            axis_probe_duration_s = 20 * adapter.config.physics_timestep_s
            after = adapter.step(mix.motor_rpm, duration_s=axis_probe_duration_s)
            rate_delta = float(after.body_rates_rad_s[axis] - before.body_rates_rad_s[axis])
            print(f"  {axis_name} sign: measured rate change={rate_delta:+.4f}rad/s")
            if not np.isfinite(rate_delta) or rate_delta <= 0.003:
                raise RuntimeError(f"Motor-allocation sign test failed for {axis_name}.")
        print("MuJoCo preflight PASSED. No GUI or full controller trial was run.")
        return 0
    finally:
        adapter.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Live MuJoCo comparison: PID, Koopman-LQR, Koopman-MPC.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--controller", choices=("all", *CONTROLLERS), default="all")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulated seconds per controller (default 20).")
    parser.add_argument("--headless", action="store_true", help="Disable the GUI and run without realtime pacing.")
    parser.add_argument("--fast", action="store_true", help="Keep GUI on, but do not pace to wall-clock realtime.")
    parser.add_argument("--no-pause", action="store_true", help="Do not pause for review between controller runs.")
    parser.add_argument("--self-check", action="store_true", help="Compile the MJCF, check balanced hover and verify X/Y/Z torque signs; do not run full trials.")
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    if args.self_check:
        return _self_check(project_root, order=5)

    adapter_cfg = MuJoCoAdapterConfig()
    cfg = MuJoCoDemoConfig(duration_s=float(args.duration))
    cfg.validate(adapter_cfg.control_period_s)
    model, gain, model_metadata = load_controller_assets(project_root, cfg.selected_order)
    output_base = project_root / "results" / "simulation_runs" / "mujoco_controller_demo"
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output_root = output_base / run_id
    output_root.mkdir(parents=True, exist_ok=False)

    controller_names = list(CONTROLLERS) if args.controller == "all" else [args.controller]
    traces: dict[str, dict[str, Array]] = {}
    metric_rows: list[dict[str, Any]] = []
    execution: dict[str, Any] = {}
    gui = not args.headless

    print("MuJoCo live quadrotor controller demonstration")
    print(f"Backend: {MuJoCoQuadrotorAdapter.name} | GUI={'on' if gui else 'off'} | duration={cfg.duration_s:.1f}s per controller")
    print("Initial state: position (0, 0, 1.0)m; attitude (10, 8, 6)deg; target (0, 0, 1.0)m")
    print("Primary metrics: attitude tracking RMSE, 2-degree settling time, 3D position RMSE")
    print("Live status labels explain active controller, task, attitude error, target distance and time.")
    print("Physics note: MuJoCo uses the same declared CF2X mass/inertia/KF/KM, but a transparent free-body model.")

    try:
        for controller in controller_names:
            adapter = MuJoCoQuadrotorAdapter(adapter_cfg)
            try:
                trace, diagnostics = run_live_trial(
                    controller, adapter, model, gain, cfg,
                    gui=gui, realtime=(gui and not args.fast),
                )
                metrics = calculate_primary_metrics(trace, cfg)
                traces[controller] = trace
                metric_rows.append(metrics)
                execution[controller] = diagnostics
                if gui:
                    obs_final = adapter.observe()
                    adapter.update_presentation(
                        controller, obs_final,
                        desired_rotation_body_to_world=np.asarray(trace["desired_rotation_body_to_world"][-1]),
                        target_position_world_m=np.asarray(adapter.config.target_position_world_m),
                        attitude_error_deg=float(trace["attitude_tracking_error_deg"][-1]),
                        position_error_m=float(trace["position_error_norm_m"][-1]),
                        elapsed_s=float(obs_final.time_s), duration_s=cfg.duration_s,
                        final_metrics=metrics,
                    )
                settle = metrics["settling_time_s"]
                settle_text = "not settled" if settle is None else f"{float(settle):.2f}s"
                print(
                    f"{controller:12s} | attitude RMSE={metrics['attitude_tracking_rmse_deg']:.3f}deg "
                    f"| settle={settle_text} | position RMSE={metrics['position_rmse_3d_m']:.3f}m "
                    f"| final distance={float(trace['position_error_norm_m'][-1]):.4f}m"
                )
                if controller == "Koopman-MPC" and diagnostics["mpc_fallback_count"]:
                    n_intervals = int(np.asarray(trace["mpc_fallback"]).size)
                    print(
                        f"  MPC diagnostic: LQR safety fallback on {diagnostics['mpc_fallback_count']}/{n_intervals} "
                        "control intervals; reported separately from headline metrics."
                    )
                if gui and not args.no_pause:
                    try:
                        input("Review the labelled MuJoCo scene and result card, then press Enter for the next controller... ")
                    except EOFError:
                        pass
            finally:
                adapter.close()

        figure_path = output_root / "controller_validation.png"
        trace_path = output_root / "live_traces.npz"
        _write_validation_figure(traces, metric_rows, figure_path)
        _save_trace_bundle(traces, trace_path)
        summary = {
            "experiment": "Live MuJoCo CF2X-parameterized attitude tracking with world-position hold",
            "run_id": run_id,
            "generated_local": datetime.now().astimezone().isoformat(timespec="seconds"),
            "backend": MuJoCoQuadrotorAdapter.name,
            "duration_s_per_controller": cfg.duration_s,
            "control_period_s": adapter_cfg.control_period_s,
            "physics_timestep_s": adapter_cfg.physics_timestep_s,
            "controller_configuration": asdict(cfg),
            "adapter_configuration": {**asdict(adapter_cfg), "derived_max_rpm": adapter_cfg.max_rpm},
            "initial_position_world_m": list(cfg.initial_position_world_m),
            "initial_attitude_deg": list(cfg.initial_attitude_deg),
            "target_position_world_m": list(adapter_cfg.target_position_world_m),
            "settling_tolerance_deg": cfg.settling_tolerance_deg,
            "model": model_metadata,
            "primary_metrics": ["attitude_tracking_rmse_deg", "settling_time_s", "position_rmse_3d_m"],
            "controllers": metric_rows,
            "execution_diagnostics": execution,
            "limitations": [
                "MuJoCo is a separate rigid-body model using declared CF2X mass, diagonal inertia, thrust coefficient and yaw-moment coefficient.",
                "The model uses resultant motor thrust and moment; it does not model native PyBullet propeller downwash, detailed rotor aerodynamics, or hardware equivalence.",
                "Learned-model torque is mapped by the simulator-to-identified diagonal-inertia ratio; model mismatch remains a limitation.",
                "Koopman-MPC uses its saved LQR gain as a safety fallback if optimization fails; fallback intervals are recorded.",
            ],
            "artifacts": [figure_path.name, trace_path.name],
        }
        summary_path = output_root / "simulation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nMuJoCo comparison complete.")
        print(f"Output folder: {output_root}")
        print(f"Three-metric summary: {summary_path}")
        print(f"Single validation figure: {figure_path}")
        print(f"Raw live traces: {trace_path}")
        print("Run ID and summary are stored together in the same unique directory.")
        return 0
    except Exception as exc:
        print(f"\nMuJoCo demonstration failed: {type(exc).__name__}: {exc}")
        print("No claim of a completed cross-platform run should be made until this error is resolved and the live trial completes.")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
