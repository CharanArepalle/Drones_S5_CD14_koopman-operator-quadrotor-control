"""Disturbance-rejection experiment using the shared attitude model and controllers.

Runs PID, Koopman-LQR, and Koopman-MPC with and without an observer-based
matched-torque compensation term under identical external disturbances.
It reuses the existing dataset and does not modify baseline result files.
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

from src.attitude_core import (
    attitude_lift,
    euler_zyx_to_rotation,
    integrate_attitude_rk4,
    rotation_to_euler_zyx,
)
from src.identification import fit_koopman_edmd
from src.koopman_mpc import KoopmanMPC
from src.disturbance_observer import LowPassTorqueDOB
from src import attitude_study as baseline

Array = np.ndarray


def disturbance_torque(time_s: float) -> Array:
    """Known injected benchmark profile, in body-torque Nm.

    A low-amplitude multi-frequency component is present throughout, with
    three unmatched step changes that test transient disturbance rejection.
    This signal is generated only for evaluation; it is not given to controllers.
    """
    t = float(time_s)
    # Allow the controllers to begin regulation without injected disturbance.
    # The sinusoidal and step disturbances start together at t=1.5 s.
    if t < 1.5:
        return np.zeros(3, dtype=float)

    disturbance = np.array([
        0.0025 * np.sin(2.0 * np.pi * 0.45 * (t - 1.5)),
        0.0020 * np.sin(2.0 * np.pi * 0.35 * (t - 1.5) + 0.7),
        0.0015 * np.sin(2.0 * np.pi * 0.55 * (t - 1.5) + 1.1),
    ], dtype=float)

    if 1.5 <= t < 4.0:
        disturbance += np.array([0.014, -0.010, 0.006])
    elif 4.0 <= t < 6.5:
        disturbance += np.array([-0.014, 0.010, -0.007])
    elif 6.5 <= t < 8.5:
        disturbance += np.array([0.008, 0.012, 0.005])

    return disturbance


def _wrap_angles(values: Array) -> Array:
    return np.arctan2(np.sin(values), np.cos(values))


def _trapezoid_time_integral(values: Array, time: Array) -> Array:
    """Integrate a time-first array using the composite trapezoid rule."""
    values = np.asarray(values, dtype=float)
    time = np.asarray(time, dtype=float).reshape(-1)
    if values.shape[0] != time.size or time.size < 2:
        raise ValueError("values and time must have matching first dimensions.")
    widths = np.diff(time).reshape((-1,) + (1,) * (values.ndim - 1))
    return np.sum(0.5 * (values[1:] + values[:-1]) * widths, axis=0)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def simulate_disturbed_controller(
    controller_name: str,
    model,
    lqr_gain: Array,
    config,
    *,
    use_dob: bool,
    dob_cutoff_hz: float = 1.5,
) -> dict[str, Array | str | float]:
    """Run one controller with the same plant, disturbance profile and limits."""
    dt = float(config.sample_period_s)
    sub_dt = dt / int(config.integration_substeps)
    steps = int(round(config.control_duration_s / dt))
    time = np.arange(steps + 1, dtype=float) * dt

    rotation = euler_zyx_to_rotation(
        *np.deg2rad(config.control_initial_angles_deg)
    )
    omega = np.zeros(3, dtype=float)
    n = model.A.shape[0]
    z_equilibrium = attitude_lift(np.eye(3), np.zeros(3), model.order)

    states = np.empty((steps + 1, 6), dtype=float)
    applied_torques = np.empty((steps, 3), dtype=float)
    raw_torques = np.empty((steps, 3), dtype=float)
    disturbance_log = np.stack([disturbance_torque(t) for t in time])
    estimate_log = np.zeros((steps + 1, 3), dtype=float)
    saturation_log = np.zeros((steps, 3), dtype=bool)
    mpc_success = np.ones(steps, dtype=bool)
    mpc_fallback = np.zeros(steps, dtype=bool)
    mpc_solve_ms = np.zeros(steps, dtype=float)

    observer = LowPassTorqueDOB(
        inertia=baseline.INERTIA,
        sample_period_s=dt,
        cutoff_hz=dob_cutoff_hz,
    )

    mpc_controller = None
    if controller_name == "Koopman-MPC":
        mpc_controller = KoopmanMPC(
            model=model,
            torque_limits=baseline.TORQUE_LIMITS,
            fallback_gain=lqr_gain,
            horizon=int(config.mpc_horizon),
        )

    pid_kp = np.array([0.20, 0.20, 0.12])
    pid_ki = np.array([0.010, 0.010, 0.005])
    pid_kd = np.array([0.05, 0.05, 0.036])
    pid_integral = np.zeros(3, dtype=float)
    pid_integral_limit = np.array([2.0, 2.0, 2.0])

    for k in range(steps + 1):
        angles = rotation_to_euler_zyx(rotation)
        states[k] = np.concatenate([angles, omega])

        estimate = observer.update(omega)
        estimate_log[k] = estimate
        control_disturbance_estimate = estimate if use_dob else np.zeros(3)

        if k == steps:
            break

        if controller_name == "PID":
            error = _wrap_angles(-angles)
            candidate_integral = np.clip(
                pid_integral + error * dt,
                -pid_integral_limit, pid_integral_limit,
            )
            candidate_nominal = (
                pid_kp * error
                + pid_ki * candidate_integral
                - pid_kd * omega
            )
            candidate_request = candidate_nominal - control_disturbance_estimate
            pushing_positive = (
                (candidate_request > baseline.TORQUE_LIMITS) & (error > 0.0)
            )
            pushing_negative = (
                (candidate_request < -baseline.TORQUE_LIMITS) & (error < 0.0)
            )
            accept_integral = ~(pushing_positive | pushing_negative)
            pid_integral[accept_integral] = candidate_integral[accept_integral]
            nominal_torque = (
                pid_kp * error + pid_ki * pid_integral - pid_kd * omega
            )
        elif controller_name == "Koopman-LQR":
            z = attitude_lift(rotation, omega, model.order)
            nominal_torque = -lqr_gain @ (z - z_equilibrium)
        elif controller_name == "Koopman-MPC":
            if mpc_controller is None:
                raise RuntimeError("MPC controller initialization failed.")
            result = mpc_controller.command(rotation, omega)
            nominal_torque = result.torque
            mpc_success[k] = bool(result.success)
            mpc_fallback[k] = bool(result.used_fallback)
            mpc_solve_ms[k] = float(result.solve_time_ms)
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        requested_torque = nominal_torque - control_disturbance_estimate
        actual_torque = np.clip(
            requested_torque,
            -baseline.TORQUE_LIMITS,
            baseline.TORQUE_LIMITS,
        )

        raw_torques[k] = requested_torque
        applied_torques[k] = actual_torque
        saturation_log[k] = (
            np.abs(requested_torque) >= baseline.TORQUE_LIMITS - 1e-7
        )
        observer.record_applied_torque(actual_torque)

        injected_disturbance = disturbance_log[k]
        for _ in range(int(config.integration_substeps)):
            rotation, omega = integrate_attitude_rk4(
                rotation,
                omega,
                actual_torque,
                baseline.INERTIA,
                sub_dt,
                disturbance=injected_disturbance,
            )

    return {
        "time": time,
        "state": states,
        "torque": applied_torques,
        "requested_torque": raw_torques,
        "disturbance": disturbance_log,
        "disturbance_estimate": estimate_log,
        "saturated": saturation_log,
        "mpc_success": mpc_success,
        "mpc_fallback": mpc_fallback,
        "mpc_solve_ms": mpc_solve_ms,
        "controller": controller_name,
        "dob_enabled": str(bool(use_dob)),
        "dob_cutoff_hz": float(dob_cutoff_hz),
    }


def calculate_metrics(
    run: dict[str, Any],
    config,
) -> dict[str, Any]:
    dt = float(config.sample_period_s)
    states = np.asarray(run["state"], dtype=float)
    torque = np.asarray(run["torque"], dtype=float)
    disturbance = np.asarray(run["disturbance"], dtype=float)
    estimate = np.asarray(run["disturbance_estimate"], dtype=float)
    time = np.asarray(run["time"], dtype=float)

    angle_error = _wrap_angles(states[:, :3])
    rate = states[:, 3:]
    angle_deg = np.rad2deg(angle_error)

    iae_by_axis = _trapezoid_time_integral(np.abs(angle_deg), time)
    angle_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(angle_error ** 2))))
    rate_rmse_rad_s = float(np.sqrt(np.mean(rate ** 2)))
    torque_effort = float(np.sum(torque ** 2) * dt)
    saturation_percent = float(
        100.0 * np.mean(np.any(np.asarray(run["saturated"]), axis=1))
    )
    dob_enabled = str(run["dob_enabled"]) == "True"

    # Evaluate disturbance rejection separately from initial regulation.
    # Disturbances begin at t=1.5 s; include the later sinusoid and step changes.
    disturbance_start_s = 1.5
    window = time >= disturbance_start_s
    if np.count_nonzero(window) < 2:
        raise ValueError("Control duration is too short for disturbance-window metrics.")
    window_time = time[window]
    window_angle_deg = angle_deg[window]
    window_rate = rate[window]
    window_torque = torque[window[:-1]] if len(window) == len(torque) + 1 else torque[window]
    window_iae_by_axis = _trapezoid_time_integral(np.abs(window_angle_deg), window_time)
    window_angle_rmse_deg = float(
        np.rad2deg(np.sqrt(np.mean(angle_error[window] ** 2)))
    )
    window_rate_rmse_rad_s = float(
        np.sqrt(np.mean(window_rate ** 2))
    )
    window_control_effort = float(np.sum(window_torque ** 2) * dt)
    window_peak_angle_deg = float(np.max(np.abs(window_angle_deg)))

    if dob_enabled:
        disturbance_rmse = float(
            np.sqrt(np.mean((estimate[1:] - disturbance[:-1]) ** 2))
        )
    else:
        disturbance_rmse = None

    solver_ms = np.asarray(run["mpc_solve_ms"], dtype=float)
    solver_ok = np.asarray(run["mpc_success"], dtype=bool)
    fallback = np.asarray(run["mpc_fallback"], dtype=bool)
    controller = str(run["controller"])

    return {
        "controller": controller,
        "dob_enabled": dob_enabled,
        "angle_rmse_deg": angle_rmse_deg,
        "angle_iae_total_deg_s": float(np.sum(iae_by_axis)),
        "disturbance_window_start_s": disturbance_start_s,
        "disturbance_window_angle_rmse_deg": window_angle_rmse_deg,
        "disturbance_window_iae_total_deg_s": float(np.sum(window_iae_by_axis)),
        "disturbance_window_peak_angle_deg": window_peak_angle_deg,
        "disturbance_window_body_rate_rmse_rad_s": window_rate_rmse_rad_s,
        "disturbance_window_control_effort_integral_nm2_s": window_control_effort,
        "roll_iae_deg_s": float(iae_by_axis[0]),
        "pitch_iae_deg_s": float(iae_by_axis[1]),
        "yaw_iae_deg_s": float(iae_by_axis[2]),
        "body_rate_rmse_rad_s": rate_rmse_rad_s,
        "peak_absolute_angle_deg": float(np.max(np.abs(angle_deg))),
        "peak_absolute_body_rate_rad_s": float(np.max(np.abs(rate))),
        "control_effort_integral_nm2_s": torque_effort,
        "saturated_sample_percent": saturation_percent,
        "disturbance_estimation_rmse_nm": disturbance_rmse,
        "mpc_solve_time_mean_ms": (
            float(np.mean(solver_ms)) if controller == "Koopman-MPC" else None
        ),
        "mpc_solver_success_percent": (
            float(100.0 * np.mean(solver_ok)) if controller == "Koopman-MPC" else None
        ),
        "mpc_fallback_count": (
            int(np.sum(fallback)) if controller == "Koopman-MPC" else None
        ),
    }


def _plot_disturbance_profile(
    time: Array, actual: Array, estimated: Array, path: Path
) -> None:
    names = ("Roll disturbance", "Pitch disturbance", "Yaw disturbance")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.6), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.plot(time, actual[:, i], label="Injected disturbance", linewidth=1.6)
        ax.plot(time, estimated[:, i], "--", label="DOB estimate", linewidth=1.3)
        ax.set_ylabel(f"{names[i]} (N m)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Matched torque disturbance and observer estimate (Koopman-LQR + DOB)")
    _safe_savefig(fig, path)


def _plot_attitude_response(
    runs: dict[str, dict[str, Any]], path: Path
) -> None:
    axes_names = ("Roll", "Pitch", "Yaw")
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes):
        for key, run in runs.items():
            ax.plot(
                run["time"],
                np.rad2deg(run["state"][:, i]),
                label=key,
                linewidth=1.3,
            )
        ax.axhline(0.0, linestyle=":", linewidth=1.0)
        ax.set_ylabel(f"{axes_names[i]} (deg)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Attitude disturbance rejection: baseline versus DOB compensation")
    _safe_savefig(fig, path)


def _plot_rate_response(
    runs: dict[str, dict[str, Any]], path: Path
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.0), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes):
        for key, run in runs.items():
            ax.plot(
                run["time"],
                run["state"][:, i + 3],
                label=key,
                linewidth=1.25,
            )
        ax.set_ylabel(f"Body rate {i + 1} (rad/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Body-rate response under matched torque disturbances")
    _safe_savefig(fig, path)


def _plot_iae_comparison(rows: list[dict[str, Any]], path: Path) -> None:
    controllers = ("PID", "Koopman-LQR", "Koopman-MPC")
    labels = ["PID", "Koopman-LQR", "Koopman-MPC"]
    base_values = []
    dob_values = []
    for name in controllers:
        base_row = next(
            r for r in rows if r["controller"] == name and not r["dob_enabled"]
        )
        dob_row = next(
            r for r in rows if r["controller"] == name and r["dob_enabled"]
        )
        base_values.append(float(base_row["disturbance_window_iae_total_deg_s"]))
        dob_values.append(float(dob_row["disturbance_window_iae_total_deg_s"]))

    x = np.arange(len(controllers))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.7, 5.0), constrained_layout=True)
    ax.bar(x - width / 2, base_values, width, label="Without DOB")
    ax.bar(x + width / 2, dob_values, width, label="With DOB")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Disturbance-window attitude IAE (deg·s)")
    ax.set_title("Disturbance-window tracking error: baseline versus DOB")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    _safe_savefig(fig, path)


def _safe_savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{__import__('uuid').uuid4().hex}.tmp{path.suffix}")
    try:
        fig.savefig(temp, dpi=180, bbox_inches="tight")
        try:
            temp.replace(path)
        except OSError:
            temp.replace(path.with_name(f"{path.stem}_new{path.suffix}"))
    finally:
        plt.close(fig)
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def run_disturbance_study(
    dataset: dict[str, np.ndarray],
    results_root: str | Path,
    *,
    order: int = 5,
    dob_cutoff_hz: float = 1.5,
    control_duration_s: float = 10.0,
    initial_angles_deg: tuple[float, float, float] = (10.0, 8.0, 6.0),
) -> dict[str, Any]:
    """Run all controller/DOB combinations on the same disturbance profile."""
    training = baseline.transitions_for_split(dataset, "training")
    model = fit_koopman_edmd(training, order=order)
    lqr_gain, lqr_diagnostics = baseline._lqr_gain(model)

    config = baseline.StudyConfig(
        control_duration_s=control_duration_s,
        control_initial_angles_deg=initial_angles_deg,
    )

    controller_names = ("PID", "Koopman-LQR", "Koopman-MPC")
    runs: dict[str, dict[str, Any]] = {}
    metrics_rows: list[dict[str, Any]] = []

    for controller_name in controller_names:
        for use_dob in (False, True):
            label = f"{controller_name}{'+DOB' if use_dob else ''}"
            run = simulate_disturbed_controller(
                controller_name,
                model,
                lqr_gain,
                config,
                use_dob=use_dob,
                dob_cutoff_hz=dob_cutoff_hz,
            )
            runs[label] = run
            metrics_rows.append(calculate_metrics(run, config))

    for controller_name in controller_names:
        baseline_row = next(
            r for r in metrics_rows
            if r["controller"] == controller_name and not r["dob_enabled"]
        )
        dob_row = next(
            r for r in metrics_rows
            if r["controller"] == controller_name and r["dob_enabled"]
        )
        base_iae = float(baseline_row["angle_iae_total_deg_s"])
        compensated_iae = float(dob_row["angle_iae_total_deg_s"])
        improvement = (
            100.0 * (base_iae - compensated_iae) / max(base_iae, 1e-12)
        )
        baseline_row["iae_change_with_dob_percent"] = 0.0
        dob_row["iae_change_with_dob_percent"] = improvement
        base_window_iae = float(baseline_row["disturbance_window_iae_total_deg_s"])
        dob_window_iae = float(dob_row["disturbance_window_iae_total_deg_s"])
        baseline_row["disturbance_window_iae_improvement_percent"] = 0.0
        dob_row["disturbance_window_iae_improvement_percent"] = (
            100.0 * (base_window_iae - dob_window_iae) / max(base_window_iae, 1e-12)
        )

    results_root = Path(results_root)
    output_dir = results_root / "disturbance_study"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"

    for folder in (tables_dir, figures_dir, models_dir):
        folder.mkdir(parents=True, exist_ok=True)

    _write_csv(tables_dir / "disturbance_rejection_metrics.csv", metrics_rows)

    estimation_rows: list[dict[str, Any]] = []
    for key, run in runs.items():
        if not key.endswith("+DOB"):
            continue
        estimate = np.asarray(run["disturbance_estimate"])
        actual = np.asarray(run["disturbance"])
        err = estimate[1:] - actual[:-1]
        for axis, name in enumerate(("roll", "pitch", "yaw")):
            estimation_rows.append({
                "controller": str(run["controller"]),
                "axis": name,
                "cutoff_hz": dob_cutoff_hz,
                "disturbance_rmse_nm": float(np.sqrt(np.mean(err[:, axis] ** 2))),
                "disturbance_bias_nm": float(np.mean(err[:, axis])),
            })
    _write_csv(tables_dir / "disturbance_estimation_metrics.csv", estimation_rows)

    # Use the Koopman-LQR + DOB case for the representative observer plot.
    _plot_disturbance_profile(
        np.asarray(runs["Koopman-LQR+DOB"]["time"]),
        np.asarray(runs["Koopman-LQR+DOB"]["disturbance"]),
        np.asarray(runs["Koopman-LQR+DOB"]["disturbance_estimate"]),
        figures_dir / "disturbance_profile_and_estimate.png",
    )
    _plot_attitude_response(
        runs, figures_dir / "disturbance_attitude_response.png"
    )
    _plot_rate_response(
        runs, figures_dir / "disturbance_body_rate_response.png"
    )
    _plot_iae_comparison(
        metrics_rows, figures_dir / "disturbance_rejection_comparison.png"
    )

    log_arrays: dict[str, np.ndarray] = {}
    for label, run in runs.items():
        safe_label = label.lower().replace("-", "_").replace("+", "_")
        for field in (
            "time", "state", "torque", "requested_torque", "disturbance",
            "disturbance_estimate", "saturated",
        ):
            log_arrays[f"{safe_label}_{field}"] = np.asarray(run[field])
    np.savez_compressed(models_dir / "disturbance_study_logs.npz", **log_arrays)

    summary = {
        "method": {
            "reference": (
                "Bayrak and Efe, 'Disturbance observer-based control schemes for "
                "quadrotors: a tutorial', rotational matched disturbance in Eq. (2), "
                "conventional inverse-nominal-plant plus low-pass Q-filter structure."
            ),
            "implemented_observer": (
                "Discrete filtered inverse-dynamics residual observer. It is a "
                "first-order Q-filter DOB adaptation, not a claim to reproduce "
                "the tutorial's complete CDOBC or TDDOBC architecture."
            ),
            "plant_equation": (
                "J*omega_dot = applied_torque - cross(omega, J*omega) + disturbance_torque."
            ),
            "raw_residual": (
                "d_raw[k] = J*(omega[k]-omega[k-1])/dt "
                "+ cross(omega[k-1], J*omega[k-1]) - applied_torque[k-1]."
            ),
            "filter": (
                "d_hat[k] = alpha*d_hat[k-1] + (1-alpha)*d_raw[k], "
                "alpha=exp(-2*pi*cutoff_hz*dt)."
            ),
            "compensation": "requested_torque = base_controller_torque - estimated_disturbance.",
            "fair_comparison": (
                "All six cases use identical initial conditions, disturbance signal, "
                "integration settings and saturation limits. DOB cases differ only by "
                "the estimated torque compensation term."
            ),
            "disturbance_window_metrics": (
                "Disturbance-window metrics are evaluated from t=1.5 s onward, after "
                "the common initial-regulation period, to avoid letting initial error "
                "dominate the disturbance-rejection score."
            ),
            "disturbance_estimation_metric_alignment": (
                "Estimate[k] uses the state increment over [t[k-1], t[k]] and the torque "
                "applied on that interval, so it is compared with disturbance[k-1] using "
                "estimate[1:] against disturbance[:-1]."
            ),
        },
        "configuration": {
            "order": order,
            "cutoff_hz": dob_cutoff_hz,
            "control_duration_s": control_duration_s,
            "sample_period_s": config.sample_period_s,
            "initial_angles_deg": list(initial_angles_deg),
            "inertia_kg_m2": baseline.INERTIA.tolist(),
            "torque_limits_nm": baseline.TORQUE_LIMITS.tolist(),
            "lqr_diagnostics": lqr_diagnostics,
            "mpc_horizon": config.mpc_horizon,
        },
        "metrics": metrics_rows,
        "outputs": [
            "disturbance_study/tables/disturbance_rejection_metrics.csv",
            "disturbance_study/tables/disturbance_estimation_metrics.csv",
            "disturbance_study/figures/disturbance_profile_and_estimate.png",
            "disturbance_study/figures/disturbance_attitude_response.png",
            "disturbance_study/figures/disturbance_body_rate_response.png",
            "disturbance_study/figures/disturbance_rejection_comparison.png",
            "disturbance_study/models/disturbance_study_logs.npz",
        ],
    }

    with (output_dir / "disturbance_study_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate matched-torque disturbance observer compensation."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/data/attitude_dataset.npz"),
        help="Path to the existing shared dataset; it is not regenerated.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory for extension outputs.",
    )
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--cutoff-hz", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    data = baseline.load_dataset(args.dataset)
    summary = run_disturbance_study(
        data,
        args.results_root,
        order=args.order,
        dob_cutoff_hz=args.cutoff_hz,
        control_duration_s=args.duration,
    )

    print("Disturbance-observer study completed.")
    print(f"Dataset reused: {args.dataset.resolve()}")
    print("Controllers: PID, Koopman-LQR, Koopman-MPC (each with/without DOB)")
    print(f"Results: {(args.results_root / 'disturbance_study').resolve()}")
    for item in summary["outputs"]:
        print(f"  {args.results_root / item}")


if __name__ == "__main__":
    main()
