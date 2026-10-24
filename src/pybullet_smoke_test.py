"""Run a measured-response actuator/axis smoke test in gym-pybullet-drones.

Run from the project root:
    python -m src.pybullet_smoke_test
    python -m src.pybullet_smoke_test --gui

This is a preflight diagnostic, not the closed-loop controller demonstration.
Unlike a log-only check, it FAILS when measured body-rate increments have the
wrong sign for a positive/negative torque pulse.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.simulators.pybullet_adapter import PyBulletAdapterConfig, PyBulletDronesAdapter

Array = np.ndarray
_AXIS_NAMES = ("roll", "pitch", "yaw")
_RATE_NAMES = ("p", "q", "r")
_SIGN_TOLERANCE_RAD_S2 = 1.0e-4


def _pulse_torque(time_s: float) -> Array:
    """Paired body-axis torque pulses in model-equivalent N*m."""
    if 0.50 <= time_s < 0.75:
        return np.array([0.020, 0.0, 0.0])
    if 0.75 <= time_s < 1.00:
        return np.array([-0.020, 0.0, 0.0])
    if 1.00 <= time_s < 1.25:
        return np.array([0.0, 0.020, 0.0])
    if 1.25 <= time_s < 1.50:
        return np.array([0.0, -0.020, 0.0])
    if 1.50 <= time_s < 1.75:
        return np.array([0.0, 0.0, 0.010])
    if 1.75 <= time_s < 2.00:
        return np.array([0.0, 0.0, -0.010])
    return np.zeros(3, dtype=float)


def _state_row(observation: Any, command: Any | None) -> dict[str, Any]:
    """Normalize one state sample and, where applicable, its starting-time command."""
    from src.simulators.pybullet_adapter import rotation_to_euler_zyx

    euler_deg = np.rad2deg(rotation_to_euler_zyx(observation.rotation_body_to_world))
    row: dict[str, Any] = {
        "time_s": float(observation.time_s),
        "x_m": float(observation.position_world_m[0]),
        "y_m": float(observation.position_world_m[1]),
        "z_m": float(observation.position_world_m[2]),
        "roll_deg": float(euler_deg[0]),
        "pitch_deg": float(euler_deg[1]),
        "yaw_deg": float(euler_deg[2]),
        "body_rate_p_rad_s": float(observation.body_rates_rad_s[0]),
        "body_rate_q_rad_s": float(observation.body_rates_rad_s[1]),
        "body_rate_r_rad_s": float(observation.body_rates_rad_s[2]),
        "vx_world_m_s": float(observation.velocity_world_m_s[0]),
        "vy_world_m_s": float(observation.velocity_world_m_s[1]),
        "vz_world_m_s": float(observation.velocity_world_m_s[2]),
        "observed_alpha_p_rad_s2": 0.0,
        "observed_alpha_q_rad_s2": 0.0,
        "observed_alpha_r_rad_s2": 0.0,
    }
    if command is None:
        row.update({
            "command_valid": False,
            "tau_model_equiv_x_nm": 0.0, "tau_model_equiv_y_nm": 0.0, "tau_model_equiv_z_nm": 0.0,
            "tau_physical_requested_x_nm": 0.0, "tau_physical_requested_y_nm": 0.0, "tau_physical_requested_z_nm": 0.0,
            "tau_allocator_implied_x_nm": 0.0, "tau_allocator_implied_y_nm": 0.0, "tau_allocator_implied_z_nm": 0.0,
            "allocator_residual_x_nm": 0.0, "allocator_residual_y_nm": 0.0, "allocator_residual_z_nm": 0.0,
            "collective_requested_n": 0.0, "collective_allocator_implied_n": 0.0,
            "saturation_scale": 1.0, "saturated": False,
            "motor0_rpm": 0.0, "motor1_rpm": 0.0, "motor2_rpm": 0.0, "motor3_rpm": 0.0,
        })
        return row

    mixer = command.mixer_output
    requested = np.asarray(command.requested_physical_torque_nm, dtype=float)
    implied = np.asarray(mixer.allocator_implied_body_torque_nm, dtype=float)
    residual = implied - requested
    row.update({
        "command_valid": True,
        "tau_model_equiv_x_nm": float(command.model_equivalent_torque_nm[0]),
        "tau_model_equiv_y_nm": float(command.model_equivalent_torque_nm[1]),
        "tau_model_equiv_z_nm": float(command.model_equivalent_torque_nm[2]),
        "tau_physical_requested_x_nm": float(requested[0]),
        "tau_physical_requested_y_nm": float(requested[1]),
        "tau_physical_requested_z_nm": float(requested[2]),
        # These are predicted by the mixer allocation matrix, not measured
        # torque telemetry from the native PyBullet plant.
        "tau_allocator_implied_x_nm": float(implied[0]),
        "tau_allocator_implied_y_nm": float(implied[1]),
        "tau_allocator_implied_z_nm": float(implied[2]),
        "allocator_residual_x_nm": float(residual[0]),
        "allocator_residual_y_nm": float(residual[1]),
        "allocator_residual_z_nm": float(residual[2]),
        "collective_requested_n": float(command.collective_thrust_requested_n),
        "collective_allocator_implied_n": float(mixer.achieved_collective_thrust_n),
        "saturation_scale": float(mixer.saturation_scale),
        "saturated": bool(mixer.saturated),
        "motor0_rpm": float(mixer.motor_rpm[0]),
        "motor1_rpm": float(mixer.motor_rpm[1]),
        "motor2_rpm": float(mixer.motor_rpm[2]),
        "motor3_rpm": float(mixer.motor_rpm[3]),
    })
    return row


def _axis_sign_diagnostics(
    model_torque_nm: Array,
    observed_angular_acceleration_rad_s2: Array,
    *,
    min_acceleration_rad_s2: float = _SIGN_TOLERANCE_RAD_S2,
) -> dict[str, Any]:
    """Compare the sign of measured body-rate increments with paired torque pulses.

    The command for interval i is paired with (omega[i+1] - omega[i]) / dt[i].
    Both positive and negative pulse responses are required for every axis.
    """
    torque = np.asarray(model_torque_nm, dtype=float)
    alpha = np.asarray(observed_angular_acceleration_rad_s2, dtype=float)
    if torque.ndim != 2 or torque.shape[1] != 3:
        raise ValueError("model_torque_nm must have shape (N, 3).")
    if alpha.shape != torque.shape:
        raise ValueError("observed_angular_acceleration_rad_s2 must have the same shape as torque.")
    if not np.all(np.isfinite(torque)) or not np.all(np.isfinite(alpha)):
        raise ValueError("Torque and measured angular acceleration must be finite.")
    if not np.isfinite(min_acceleration_rad_s2) or min_acceleration_rad_s2 < 0.0:
        raise ValueError("min_acceleration_rad_s2 must be finite and non-negative.")

    result: dict[str, Any] = {}
    for axis, name in enumerate(_AXIS_NAMES):
        pos_mask = torque[:, axis] > 1.0e-8
        neg_mask = torque[:, axis] < -1.0e-8
        pos_mean = float(np.mean(alpha[pos_mask, axis])) if np.any(pos_mask) else None
        neg_mean = float(np.mean(alpha[neg_mask, axis])) if np.any(neg_mask) else None
        passed = (
            pos_mean is not None and neg_mean is not None
            and pos_mean > min_acceleration_rad_s2
            and neg_mean < -min_acceleration_rad_s2
        )
        result[name] = {
            "positive_pulse_interval_count": int(np.sum(pos_mask)),
            "negative_pulse_interval_count": int(np.sum(neg_mask)),
            "mean_observed_alpha_for_positive_torque_rad_s2": pos_mean,
            "mean_observed_alpha_for_negative_torque_rad_s2": neg_mean,
            "positive_negative_signs_match": bool(passed),
            "criterion": f"positive mean > {min_acceleration_rad_s2:g}, negative mean < -{min_acceleration_rad_s2:g} rad/s^2",
        }
    result["all_axes_pass"] = all(result[n]["positive_negative_signs_match"] for n in _AXIS_NAMES)
    return result


def _write_results(rows: list[dict[str, Any]], compatibility: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    if len(rows) < 2:
        raise RuntimeError("Smoke test returned fewer than two state samples.")
    out_dir.mkdir(parents=True, exist_ok=True)

    state_time_s = np.asarray([r["time_s"] for r in rows], dtype=float)
    if not np.all(np.isfinite(state_time_s)) or np.any(np.diff(state_time_s) <= 0.0):
        raise RuntimeError("State timestamps must be finite and strictly increasing.")
    angles = np.column_stack([[r[f"{axis}_deg"] for r in rows] for axis in _AXIS_NAMES])
    rates = np.column_stack([[r[f"body_rate_{axis}_rad_s"] for r in rows] for axis in _RATE_NAMES])
    command_rows = [r for r in rows if bool(r["command_valid"])]
    if len(command_rows) != len(rows) - 1:
        raise RuntimeError("Expected N command intervals for N+1 state samples.")

    command_time_s = np.asarray([r["time_s"] for r in command_rows], dtype=float)
    dt = np.diff(state_time_s)
    if not np.allclose(command_time_s, state_time_s[:-1], rtol=0.0, atol=1e-9):
        raise RuntimeError("Each command must be timestamped at the start of its corresponding state interval.")
    if not np.allclose(dt, np.median(dt), rtol=0.0, atol=1e-8):
        raise RuntimeError("Smoke-test state intervals are not uniformly spaced.")

    model_torque = np.column_stack([[r[f"tau_model_equiv_{axis}_nm"] for r in command_rows] for axis in ("x", "y", "z")])
    requested_physical = np.column_stack([[r[f"tau_physical_requested_{axis}_nm"] for r in command_rows] for axis in ("x", "y", "z")])
    allocator_implied = np.column_stack([[r[f"tau_allocator_implied_{axis}_nm"] for r in command_rows] for axis in ("x", "y", "z")])
    allocator_residual = allocator_implied - requested_physical
    rpm = np.column_stack([[r[f"motor{i}_rpm"] for r in command_rows] for i in range(4)])
    saturation = np.asarray([bool(r["saturated"]) for r in command_rows], dtype=bool)

    observed_alpha = np.diff(rates, axis=0) / dt[:, None]
    sign_report = _axis_sign_diagnostics(model_torque, observed_alpha)
    for i in range(len(command_rows)):
        for j, rate_name in enumerate(_RATE_NAMES):
            rows[i][f"observed_alpha_{rate_name}_rad_s2"] = float(observed_alpha[i, j])
    for j, rate_name in enumerate(_RATE_NAMES):
        rows[-1][f"observed_alpha_{rate_name}_rad_s2"] = 0.0

    numeric_columns = [c for c in rows[0] if c not in {"command_valid", "saturated"}]
    numeric = np.asarray([[float(r[c]) for c in numeric_columns] for r in rows], dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError("Non-finite numeric value detected in the PyBullet smoke-test trace.")

    columns = list(rows[0].keys())
    with (out_dir / "pybullet_smoke_trace.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out_dir / "pybullet_smoke_trace.npz",
        time_s=state_time_s,
        command_time_s=command_time_s,
        attitude_euler_zyx_deg=angles,
        body_rates_rad_s=rates,
        motor_rpm=rpm,
        model_equivalent_torque_nm=model_torque,
        requested_physical_torque_nm=requested_physical,
        allocator_implied_torque_nm=allocator_implied,
        allocator_residual_nm=allocator_residual,
        measured_angular_acceleration_rad_s2=observed_alpha,
        saturated=saturation,
    )

    # Angle plot and input plot are deliberately separate: this makes the
    # command timing and the measured response inspectable, not implicit.
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for i, label in enumerate(("Roll", "Pitch", "Yaw")):
        axes[i].plot(state_time_s, angles[:, i], label=f"{label} angle")
        axes[i].set_ylabel(f"{label} (deg)")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("gym-pybullet-drones CF2X: measured actuator-axis smoke test")
    fig.tight_layout()
    fig.savefig(out_dir / "pybullet_smoke_attitude.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex="col")
    torque_labels = (r"$\tau_x$ model-equivalent (N m)", r"$\tau_y$ model-equivalent (N m)", r"$\tau_z$ model-equivalent (N m)")
    for i, (axis_name, rate_name) in enumerate(zip(_AXIS_NAMES, _RATE_NAMES)):
        axes[i, 0].plot(state_time_s, rates[:, i], label=f"Measured body rate {rate_name}")
        axes[i, 0].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axes[i, 0].set_ylabel(f"{rate_name} (rad/s)")
        axes[i, 0].set_title(f"{axis_name.title()} angular-rate response")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].legend(loc="best")
        axes[i, 1].step(command_time_s, model_torque[:, i], where="post", label=torque_labels[i])
        axes[i, 1].axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axes[i, 1].set_ylabel("N m")
        axes[i, 1].set_title(f"{axis_name.title()} command")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].legend(loc="best")
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Command interval start time (s)")
    fig.suptitle("Axis-sign acceptance: commands paired with measured body-rate increments")
    fig.tight_layout()
    fig.savefig(out_dir / "pybullet_smoke_axis_diagnostics.png", dpi=160)
    plt.close(fig)

    allocator_residual_max = np.max(np.abs(allocator_residual), axis=0)
    numeric_pass = bool(
        np.all(np.isfinite(numeric))
        and bool(compatibility.get("all_single_axis_bounds_feasible", False))
        and np.all(allocator_residual_max < 1e-9)
    )
    axis_pass = bool(sign_report["all_axes_pass"])
    passed = numeric_pass and axis_pass
    summary = {
        "status": "PASS_ACTUATOR_AXIS_AND_NUMERICAL_CHECKS" if passed else "FAIL_ACTUATOR_AXIS_OR_NUMERICAL_CHECKS",
        "passed": passed,
        "note": (
            "Measured axis signs are checked using body-rate increments across each control interval. "
            "Allocator-implied torque is a mixer-matrix calculation, not direct physical-torque telemetry. "
            "This remains a preflight test, not a closed-loop PID/Koopman-LQR/Koopman-MPC demonstration."
        ),
        "state_sample_count": int(len(rows)),
        "control_interval_count": int(len(command_rows)),
        "duration_s": float(state_time_s[-1] - state_time_s[0]),
        "sample_period_s": float(np.median(dt)),
        "all_numeric_logs_finite": bool(np.all(np.isfinite(numeric))),
        "saturation_interval_count": int(np.sum(saturation)),
        "max_motor_rpm": float(np.max(rpm)),
        "min_altitude_m": float(np.min([r["z_m"] for r in rows])),
        "max_abs_attitude_deg": np.max(np.abs(angles), axis=0).tolist(),
        "max_abs_body_rate_rad_s": np.max(np.abs(rates), axis=0).tolist(),
        "max_abs_measured_angular_acceleration_rad_s2": np.max(np.abs(observed_alpha), axis=0).tolist(),
        "max_allocator_residual_nm": allocator_residual_max.tolist(),
        "axis_sign_diagnostics": sign_report,
        "compatibility": compatibility,
        "artifacts": [
            "pybullet_smoke_summary.json",
            "pybullet_smoke_trace.csv",
            "pybullet_smoke_trace.npz",
            "pybullet_smoke_attitude.png",
            "pybullet_smoke_axis_diagnostics.png",
        ],
    }
    with (out_dir / "pybullet_smoke_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a short measured-response CF2X actuator/axis smoke test.")
    parser.add_argument("--gui", action="store_true", help="Show the PyBullet GUI. Omit for headless execution.")
    parser.add_argument("--duration", type=float, default=2.5, help="Duration in seconds; must align to the control period.")
    parser.add_argument("--output", default="results/simulation_runs/pybullet_smoke", help="Output directory relative to project root.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "simulation.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Required configuration not found: {config_path}")
    config = PyBulletAdapterConfig.from_json(config_path)
    if args.gui:
        from dataclasses import replace
        config = replace(config, gui=True, user_debug_gui=True)
    config.validate()
    with config_path.open("r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    if not np.isfinite(args.duration) or args.duration <= 0:
        raise ValueError("--duration must be positive and finite.")
    sample_count_float = args.duration / (1.0 / config.control_frequency_hz)
    sample_count = int(round(sample_count_float))
    if sample_count < 1 or not np.isclose(sample_count_float, sample_count, atol=1e-8):
        raise ValueError("--duration must be a positive integer multiple of the control timestep.")
    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    adapter = PyBulletDronesAdapter(config)
    rows: list[dict[str, Any]] = []
    try:
        initial_position = np.asarray(raw_config.get("initial_position_world_m", [0.0, 0.0, 1.0]), dtype=float)
        initial_euler = np.asarray(raw_config.get("initial_euler_zyx_rad", [0.0, 0.0, 0.0]), dtype=float)
        obs = adapter.reset(initial_position_world_m=initial_position, initial_euler_zyx_rad=initial_euler)
        compatibility = adapter.compatibility_report()
        print("Runtime vehicle/input mapping (allocator-implied torque is not torque telemetry):")
        print(json.dumps(compatibility, indent=2))
        dt = adapter.control_timestep_s
        for step_index in range(sample_count):
            t0 = step_index * dt
            tau_model = _pulse_torque(t0)
            command = adapter.allocate_model_equivalent_torque(tau_model)
            rows.append(_state_row(obs, command))
            obs = adapter.step(command.mixer_output.motor_rpm, duration_s=dt)
        rows.append(_state_row(obs, None))
        summary = _write_results(rows, compatibility, out_dir)
        print("PyBullet smoke-test artifacts written.")
        print(json.dumps(summary, indent=2))
        print(f"Results directory: {out_dir}")
        if not summary["passed"]:
            print("ACCEPTANCE FAILED: do not continue to closed-loop controller demonstrations until the axis/sign issue is corrected.")
            return 2
        print("Measured actuator-axis smoke check passed.")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
