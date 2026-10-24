"""Attitude-model evaluation, controller comparison, and generated reports."""
from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solve_discrete_are
from matplotlib.figure import Figure

from src.attitude_core import (
    attitude_lift,
    decode_attitude_lift,
    euler_zyx_to_rotation,
    integrate_attitude_rk4,
    rotation_to_euler_zyx,
)
from src.identification import (
    AttitudeTransitions,
    KoopmanModel,
    fit_koopman_edmd,
)
from src.koopman_mpc import KoopmanMPC

Array = np.ndarray
LIFT_ORDERS = (3, 4, 5, 6, 7, 8)
INERTIA = np.array([0.021, 0.024, 0.039], dtype=float)
TORQUE_LIMITS = np.array([0.12, 0.12, 0.06], dtype=float)
OUTPUT_REDIRECTS: dict[str, str] = {}


@dataclass
class StudyConfig:
    trajectory_count: int = 40
    training_trajectory_count: int = 30
    duration_s: float = 15.0
    sample_period_s: float = 0.05
    integration_substeps: int = 5
    seed: int = 20261018
    control_duration_s: float = 10.0
    # Initial values estimated visually from the controller-response chart; exact values are not tabulated.
    # They intentionally test control beyond the small-angle excitation used for identification.
    control_initial_angles_deg: tuple[float, float, float] = (40.0, 35.0, 20.0)
    mpc_horizon: int = 10

    def validate(self) -> None:
        if self.trajectory_count < 2:
            raise ValueError("trajectory_count must be >= 2.")
        if not 1 <= self.training_trajectory_count < self.trajectory_count:
            raise ValueError("training_trajectory_count is invalid.")
        if self.duration_s <= 0 or self.sample_period_s <= 0:
            raise ValueError("duration and sample period must be positive.")
        if self.integration_substeps < 1 or self.mpc_horizon < 1:
            raise ValueError("integration_substeps and mpc_horizon must be >= 1.")
        if not np.isclose(self.duration_s / self.sample_period_s,
                          round(self.duration_s / self.sample_period_s), atol=1e-10):
            raise ValueError("duration_s must be divisible by sample_period_s.")
        if self.control_duration_s <= 0:
            raise ValueError("control_duration_s must be positive.")
        if len(self.control_initial_angles_deg) != 3:
            raise ValueError("control_initial_angles_deg must contain 3 values.")

    @property
    def intervals_per_trajectory(self) -> int:
        return int(round(self.duration_s / self.sample_period_s))


def _wrap_angle(angle: Array | float) -> Array | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _safe_savefig(fig, path: str | Path, **kwargs) -> Path:
    """Write to a unique temporary PNG before replacing the published file.

    If Windows/OneDrive or an image viewer temporarily locks the requested
    destination, save to a clearly named alternate file rather than failing
    the entire experiment run.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}_{uuid.uuid4().hex}.tmp{path.suffix}")
    try:
        Figure.savefig(fig, temp, **kwargs)
        try:
            os.replace(temp, path)
            actual = path
        except OSError as exc:
            from datetime import datetime
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            actual = path.with_name(f"{path.stem}_updated_{suffix}{path.suffix}")
            os.replace(temp, actual)
            print(
                f"Warning: could not replace '{path.name}' ({exc}); "
                f"saved updated figure as '{actual.name}'."
            )
        OUTPUT_REDIRECTS[str(path.resolve())] = str(actual.resolve())
        return actual
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def _reported_path(root: Path, relative_name: str) -> str:
    requested = (root / relative_name).resolve()
    actual = Path(OUTPUT_REDIRECTS.get(str(requested), str(requested)))
    try:
        return actual.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(actual)


def generate_dataset(config: StudyConfig) -> dict[str, np.ndarray]:
    """Generate 40 randomized sinusoidal attitude trajectories from rigid-body dynamics."""
    config.validate()
    rng = np.random.default_rng(config.seed)
    ntraj = config.trajectory_count
    n_intervals = config.intervals_per_trajectory
    nstates = n_intervals + 1
    dt = config.sample_period_s
    sub_dt = dt / config.integration_substeps
    time = np.arange(nstates) * dt

    amplitudes = rng.uniform(np.deg2rad(1.0), np.deg2rad(10.0), (ntraj, 3))
    frequencies = rng.uniform(0.05, 0.59, (ntraj, 3))
    phases = rng.uniform(-np.pi, np.pi, (ntraj, 3))
    phase = 2.0 * np.pi * frequencies[:, None, :] * time[None, :, None] + phases[:, None, :]
    refs = amplitudes[:, None, :] * np.sin(phase)
    ref_rates = amplitudes[:, None, :] * (2.0 * np.pi * frequencies[:, None, :]) * np.cos(phase)

    rotations = np.empty((ntraj, nstates, 3, 3), dtype=float)
    rates = np.empty((ntraj, nstates, 3), dtype=float)
    torques = np.empty((ntraj, n_intervals, 3), dtype=float)
    kp = np.array([8.0, 8.0, 4.5])
    kd = np.array([2.6, 2.6, 1.8])

    for i in range(ntraj):
        angles0 = refs[i, 0] + rng.normal(0.0, np.deg2rad(0.5), 3)
        rotation = euler_zyx_to_rotation(*angles0)
        omega = rng.normal(0.0, 0.02, 3)
        rotations[i, 0] = rotation
        rates[i, 0] = omega
        for k in range(n_intervals):
            angles = rotation_to_euler_zyx(rotation)
            error = _wrap_angle(refs[i, k] - angles)
            # Direct feedback torque: do not cancel the plant's gyroscopic
            # term or inject exact angular-acceleration feed-forward. This keeps
            # the generated data informative for nonlinear identification.
            torque = INERTIA * (kp * error + kd * (ref_rates[i, k] - omega))
            torque = np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)
            torques[i, k] = torque
            for _ in range(config.integration_substeps):
                rotation, omega = integrate_attitude_rk4(rotation, omega, torque, INERTIA, sub_dt)
            rotations[i, k + 1] = rotation
            rates[i, k + 1] = omega

    split_rng = np.random.default_rng(config.seed + 1009)
    permutation = split_rng.permutation(ntraj)
    training_ids = np.sort(permutation[:config.training_trajectory_count])
    validation_ids = np.sort(permutation[config.training_trajectory_count:])
    data = {
        "rotations": rotations,
        "angular_rates": rates,
        "torques": torques,
        "reference_angles": refs,
        "reference_rates": ref_rates,
        "amplitudes_rad": amplitudes,
        "frequencies_hz": frequencies,
        "phases_rad": phases,
        "training_indices": training_ids.astype(int),
        "validation_indices": validation_ids.astype(int),
        "sample_period_s": np.array(dt),
        "duration_s": np.array(config.duration_s),
        "seed": np.array(config.seed, dtype=int),
        "training_trajectory_count": np.array(config.training_trajectory_count, dtype=int),
    }
    validate_dataset(data)
    return data


def validate_dataset(data: dict[str, np.ndarray]) -> None:
    required = {
        "rotations", "angular_rates", "torques", "reference_angles", "reference_rates",
        "amplitudes_rad", "frequencies_hz", "phases_rad", "training_indices",
        "validation_indices", "sample_period_s", "duration_s", "seed",
        "training_trajectory_count",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Dataset is missing keys: {sorted(missing)}")
    rotations = np.asarray(data["rotations"])
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (trajectory, state, 3, 3).")
    ntraj, nstates = rotations.shape[:2]
    if data["angular_rates"].shape != (ntraj, nstates, 3):
        raise ValueError("angular_rates has inconsistent shape.")
    if data["torques"].shape != (ntraj, nstates - 1, 3):
        raise ValueError("torques has inconsistent shape.")
    if data["reference_angles"].shape != (ntraj, nstates, 3):
        raise ValueError("reference_angles has inconsistent shape.")
    if data["reference_rates"].shape != (ntraj, nstates, 3):
        raise ValueError("reference_rates has inconsistent shape.")
    tr = np.asarray(data["training_indices"], dtype=int)
    va = np.asarray(data["validation_indices"], dtype=int)
    if np.intersect1d(tr, va).size:
        raise ValueError("Training and validation IDs overlap.")
    all_ids = np.concatenate([tr, va])
    if len(all_ids) != ntraj or not np.array_equal(np.sort(all_ids), np.arange(ntraj)):
        raise ValueError("Training and validation IDs must partition trajectories.")
    for key, value in data.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Dataset field '{key}' has non-finite values.")


def save_dataset(data: dict[str, np.ndarray], path: Path) -> None:
    validate_dataset(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}_{uuid.uuid4().hex}.tmp.npz")
    try:
        np.savez_compressed(temp, **data)
        os.replace(temp, path)
    finally:
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    validate_dataset(data)
    return data


def transitions_for_split(data: dict[str, np.ndarray], split: str) -> AttitudeTransitions:
    validate_dataset(data)
    if split == "training":
        ids = data["training_indices"]
    elif split == "validation":
        ids = data["validation_indices"]
    else:
        raise ValueError("split must be 'training' or 'validation'.")
    return AttitudeTransitions.from_trajectories(
        data["rotations"][ids], data["angular_rates"][ids], data["torques"][ids]
    )


def evaluate_one_step_by_order(model: KoopmanModel, validation: AttitudeTransitions) -> dict[str, float]:
    """Evaluate one-step prediction errors, including the H-projected coordinates."""
    angle_pred, rate_pred, angle_true, rate_true = [], [], [], []
    lifted_sq_errors, h_projected_sq_errors = [], []
    for r, w, u, rn, wn in zip(
        validation.rotation_current, validation.omega_current, validation.torque,
        validation.rotation_next, validation.omega_next, strict=True,
    ):
        z = attitude_lift(r, w, model.order)
        z_pred = model.A @ z + model.B @ u
        z_true = attitude_lift(rn, wn, model.order)
        rp, wp = decode_attitude_lift(z_pred)
        angle_pred.append(rotation_to_euler_zyx(rp))
        rate_pred.append(wp)
        angle_true.append(rotation_to_euler_zyx(rn))
        rate_true.append(wn)
        lifted_sq_errors.append(float(np.mean((z_pred - z_true) ** 2)))
        # H=[I_18, 0]: the first 18 observable coordinates are compared directly.
        h_projected_sq_errors.append(float(np.mean((z_pred[:18] - z_true[:18]) ** 2)))

    ap, rp, at, rt = map(np.asarray, (angle_pred, rate_pred, angle_true, rate_true))
    ae = _wrap_angle(ap - at)
    re = rp - rt
    physical = np.column_stack([ae, re])
    scales = np.concatenate([np.std(at, axis=0), np.std(rt, axis=0)])
    fallback = np.maximum(np.sqrt(np.mean(np.column_stack([at, rt]) ** 2, axis=0)), 1e-6)
    scales = np.maximum(np.where(scales > 1e-8, scales, fallback), 1e-6)
    channel_rmse = np.sqrt(np.mean(physical ** 2, axis=0))
    return {
        "angle_rmse_deg": float(np.rad2deg(np.sqrt(np.mean(ae ** 2)))),
        "angular_velocity_rmse_rad_s": float(np.sqrt(np.mean(re ** 2))),
        "normalized_state_rmse": float(np.sqrt(np.mean((channel_rmse / scales) ** 2))),
        "lifted_rmse": float(np.sqrt(np.mean(lifted_sq_errors))),
        "h_projected_lifted_rmse": float(np.sqrt(np.mean(h_projected_sq_errors))),
    }


def _lqr_design(model: KoopmanModel) -> tuple[Array, Array, Array, Array, dict[str, float]]:
    """Return the chosen LQR weights, Riccati solution, gain, and diagnostics."""
    n = model.A.shape[0]
    qdiag = np.concatenate([np.full(9, 5.0), np.full(9, 2.0), np.full(n - 18, 0.2)])
    Q = np.diag(qdiag)
    R = np.diag([0.08, 0.08, 0.05])
    P = solve_discrete_are(model.A, model.B, Q, R)
    gain = np.linalg.solve(R + model.B.T @ P @ model.B, model.B.T @ P @ model.A)
    rho = float(np.max(np.abs(np.linalg.eigvals(model.A - model.B @ gain))))
    dare_residual = P - (
        model.A.T @ P @ model.A
        - model.A.T @ P @ model.B
        @ np.linalg.solve(R + model.B.T @ P @ model.B, model.B.T @ P @ model.A)
        + Q
    )
    if not np.all(np.isfinite(gain)):
        raise FloatingPointError("LQR gain contains non-finite values.")
    if rho >= 1.0:
        raise RuntimeError(f"Learned closed-loop model is not stable under LQR (spectral radius={rho:.6f}).")
    return Q, R, P, gain, {
        "closed_loop_spectral_radius": rho,
        "dare_residual_frobenius_norm": float(np.linalg.norm(dare_residual, ord="fro")),
        "Q_diagonal": qdiag.tolist(),
        "R_diagonal": np.diag(R).tolist(),
    }


def _lqr_gain(model: KoopmanModel) -> tuple[Array, dict[str, float]]:
    """Compatibility wrapper returning only the LQR gain and basic diagnostics."""
    _, _, _, gain, diagnostics = _lqr_design(model)
    return gain, diagnostics


def _save_matrix_artifacts(
    results_dir: Path, training: AttitudeTransitions, model: KoopmanModel,
    Q_lqr: Array, R_lqr: Array, P_lqr: Array, omega_lqr: Array,
) -> dict[str, Any]:
    """Expose the moment matrices and controller matrices used in the study."""
    matrix_dir = results_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    order = model.order
    z_x = np.stack([
        attitude_lift(r, w, order)
        for r, w in zip(training.rotation_current, training.omega_current, strict=True)
    ])
    z_y = np.stack([
        attitude_lift(r, w, order)
        for r, w in zip(training.rotation_next, training.omega_next, strict=True)
    ])
    # Row-sample convention: X has shape M x (9N+3), Y has shape M x 9N.
    X = np.column_stack([z_x, training.torque])
    Y = z_y
    sample_count = X.shape[0]
    G1 = (Y.T @ X) / sample_count
    G2 = (X.T @ X) / sample_count
    K_moments = G1 @ np.linalg.pinv(G2, rcond=1e-12)
    K_lstsq = np.column_stack([model.A, model.B])
    A_moments = K_moments[:, :9 * order]
    B_moments = K_moments[:, 9 * order:]
    equilibrium = attitude_lift(np.eye(3), np.zeros(3), order)
    equilibrium_defect = model.A @ equilibrium - equilibrium
    matrices = {
        "G1": G1,
        "G2": G2,
        "K_operator_moments": K_moments,
        "K_operator_lstsq": K_lstsq,
        "A_lstsq": model.A,
        "B_lstsq": model.B,
        "A_moments": A_moments,
        "B_moments": B_moments,
        "Q_lqr": Q_lqr,
        "R_lqr": R_lqr,
        "P_riccati": P_lqr,
        "Omega_lqr": omega_lqr,
        "equilibrium_lift": equilibrium.reshape(-1, 1),
        "equilibrium_defect": equilibrium_defect.reshape(-1, 1),
    }
    np.savez_compressed(matrix_dir / "attitude_matrices_N5.npz", **matrices)
    csv_names = {
        "G1": G1, "G2": G2,
        "K_operator_moments": K_moments, "K_operator_lstsq": K_lstsq,
        "A_lstsq": model.A, "B_lstsq": model.B,
        "Q_lqr": Q_lqr, "R_lqr": R_lqr, "P_riccati": P_lqr,
        "Omega_lqr": omega_lqr,
    }
    for name, matrix in csv_names.items():
        np.savetxt(matrix_dir / f"{name}.csv", matrix, delimiter=",", fmt="%.12e")

    x_singular = np.linalg.svd(X, compute_uv=False)
    g2_singular = np.linalg.svd(G2, compute_uv=False)
    relative_fit_difference = float(
        np.linalg.norm(K_moments - K_lstsq, ord="fro")
        / max(np.linalg.norm(K_lstsq, ord="fro"), 1e-12)
    )
    diagnostics = {
        "selected_order": order,
        "training_transition_count": sample_count,
        "matrix_shapes": {name: list(value.shape) for name, value in matrices.items()},
        "definitions": {
            "G1": "(1/M) * Y.T @ X, with X_k = [z_k; u_k] and Y_k = z_(k+1)",
            "G2": "(1/M) * X.T @ X",
            "K_operator_moments": "G1 @ pinv(G2); this is the moment-matrix form of the controlled least-squares estimate",
            "K_operator_lstsq": "[A B] obtained using numerically stable least squares",
            "Omega_lqr": "Discrete LQR feedback gain; u = -Omega_lqr @ (z - z_equilibrium)",
        },
        "X_design_rank": int(np.linalg.matrix_rank(X)),
        "X_design_condition_number": float(np.linalg.cond(X)),
        "G2_rank": int(np.linalg.matrix_rank(G2)),
        "G2_condition_number": float(np.linalg.cond(G2)),
        "K_moment_vs_lstsq_relative_frobenius_difference": relative_fit_difference,
        "K_moment_vs_lstsq_max_absolute_difference": float(np.max(np.abs(K_moments - K_lstsq))),
        "equilibrium_defect_norm": float(np.linalg.norm(equilibrium_defect)),
        "dare_residual_frobenius_norm": float(np.linalg.norm(
            model.A.T @ P_lqr @ model.A - P_lqr
            - model.A.T @ P_lqr @ model.B
            @ np.linalg.solve(R_lqr + model.B.T @ P_lqr @ model.B, model.B.T @ P_lqr @ model.A)
            + Q_lqr, ord="fro")),
        "lqr_closed_loop_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(model.A - model.B @ omega_lqr)))),
        "artifacts_directory": "matrices",
    }
    with (matrix_dir / "matrix_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(diagnostics), handle, indent=2, allow_nan=False)

    # Each panel uses a signed-log transform so large and small coefficients are visible.
    import matplotlib.colors as mcolors
    panel_items = [
        ("A (least squares)", model.A), ("B (least squares)", model.B),
        ("G1", G1), ("G2", G2), ("K = G1 pinv(G2)", K_moments),
        ("Omega_LQR", omega_lqr),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (title, matrix) in zip(axes.flat, panel_items, strict=True):
        scale = max(float(np.percentile(np.abs(matrix), 95)), 1e-12)
        transformed = np.sign(matrix) * np.log10(1.0 + np.abs(matrix) / scale)
        image = ax.imshow(transformed, aspect="auto", cmap="coolwarm")
        ax.set_title(f"{title}  {matrix.shape}")
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Identified system and controller matrices (signed-log visualization)")
    _safe_savefig(fig, results_dir / "figures" / "figure_7_matrix_transparency.png", dpi=180)
    plt.close(fig)
    return diagnostics


def _lqr_gain(model: KoopmanModel) -> tuple[Array, dict[str, float]]:
    n = model.A.shape[0]
    qdiag = np.concatenate([np.full(9, 5.0), np.full(9, 2.0), np.full(n - 18, 0.2)])
    Q = np.diag(qdiag)
    R = np.diag([0.08, 0.08, 0.05])
    P = solve_discrete_are(model.A, model.B, Q, R)
    gain = np.linalg.solve(R + model.B.T @ P @ model.B, model.B.T @ P @ model.A)
    rho = float(np.max(np.abs(np.linalg.eigvals(model.A - model.B @ gain))))
    if not np.all(np.isfinite(gain)):
        raise FloatingPointError("LQR gain contains non-finite values.")
    if rho >= 1.0:
        raise RuntimeError(f"Learned closed-loop model is not stable under LQR (spectral radius={rho:.6f}).")
    return gain, {"closed_loop_spectral_radius": rho}


def _simulate_regulation(
    controller_name: str, model: KoopmanModel, gain: Array, config: StudyConfig,
    mpc_controller: KoopmanMPC | None = None,
) -> dict[str, np.ndarray]:
    """Run one controller against identical nonlinear plant conditions and torque limits."""
    dt = config.sample_period_s
    sub_dt = dt / config.integration_substeps
    steps = int(round(config.control_duration_s / dt))
    time = np.arange(steps + 1) * dt
    rotation = euler_zyx_to_rotation(*np.deg2rad(config.control_initial_angles_deg))
    omega = np.zeros(3, dtype=float)
    z_eq = attitude_lift(np.eye(3), np.zeros(3), model.order)
    states = np.empty((steps + 1, 6), dtype=float)
    torques = np.empty((steps, 3), dtype=float)
    saturated = np.zeros((steps, 3), dtype=bool)
    solve_ms = np.zeros(steps, dtype=float)
    solve_success = np.ones(steps, dtype=bool)
    solve_fallback = np.zeros(steps, dtype=bool)
    iterations = np.zeros(steps, dtype=int)

    pid_kp = np.array([0.20, 0.20, 0.12])
    pid_ki = np.array([0.010, 0.010, 0.005])
    pid_kd = np.array([0.05, 0.05, 0.036])
    pid_integral = np.zeros(3, dtype=float)
    pid_integral_limit = np.array([2.0, 2.0, 2.0])

    for k in range(steps + 1):
        angles = rotation_to_euler_zyx(rotation)
        states[k] = np.concatenate([angles, omega])
        if k == steps:
            break

        if controller_name == "PID":
            error = _wrap_angle(-angles)
            candidate_integral = np.clip(
                pid_integral + error * dt,
                -pid_integral_limit, pid_integral_limit,
            )
            trial_torque = (
                pid_kp * error + pid_ki * candidate_integral - pid_kd * omega
            )
            # Conditional integration prevents the integral term from winding
            # up further when the requested torque is already saturated.
            pushing_positive = (trial_torque > TORQUE_LIMITS) & (error > 0.0)
            pushing_negative = (trial_torque < -TORQUE_LIMITS) & (error < 0.0)
            accept_integral = ~(pushing_positive | pushing_negative)
            pid_integral[accept_integral] = candidate_integral[accept_integral]
            raw_torque = pid_kp * error + pid_ki * pid_integral - pid_kd * omega
        elif controller_name == "Koopman-LQR":
            z = attitude_lift(rotation, omega, model.order)
            raw_torque = -gain @ (z - z_eq)
        elif controller_name == "Koopman-MPC":
            if mpc_controller is None:
                raise ValueError("mpc_controller is required for Koopman-MPC.")
            result = mpc_controller.command(rotation, omega)
            raw_torque = result.torque
            solve_ms[k] = result.solve_time_ms
            solve_success[k] = result.success
            solve_fallback[k] = result.used_fallback
            iterations[k] = result.iterations
        else:
            raise ValueError(f"Unknown controller: {controller_name}")

        saturated[k] = np.abs(raw_torque) >= TORQUE_LIMITS - 1e-7
        torque = np.clip(raw_torque, -TORQUE_LIMITS, TORQUE_LIMITS)
        torques[k] = torque
        for _ in range(config.integration_substeps):
            rotation, omega = integrate_attitude_rk4(rotation, omega, torque, INERTIA, sub_dt)

    return {
        "time": time,
        "state": states,
        "torque": torques,
        "controller": np.array(controller_name),
        "saturated": saturated,
        "solve_time_ms": solve_ms,
        "solve_success": solve_success,
        "solve_fallback": solve_fallback,
        "solver_iterations": iterations,
        "mpc_horizon": np.array(mpc_controller.horizon if mpc_controller is not None else 0),
    }


def _control_metrics(run: dict[str, np.ndarray], config: StudyConfig) -> list[dict[str, Any]]:
    time, states, torques = run["time"], run["state"], run["torque"]
    controller = str(run["controller"])
    names = ("Roll", "Pitch", "Yaw")
    initial = np.deg2rad(config.control_initial_angles_deg)
    last_window = max(1, int(round(1.0 / config.sample_period_s)))
    rows: list[dict[str, Any]] = []

    for axis, name in enumerate(names):
        response = states[:, axis]
        band = max(0.02 * abs(float(initial[axis])), np.deg2rad(0.1))
        within = np.abs(_wrap_angle(response)) <= band
        settle_index = next((i for i in range(len(time)) if np.all(within[i:])), None)
        settling_time = float(time[settle_index]) if settle_index is not None else None
        if initial[axis] >= 0:
            overshoot = max(0.0, -float(np.min(response))) / abs(initial[axis]) * 100.0
        else:
            overshoot = max(0.0, float(np.max(response))) / abs(initial[axis]) * 100.0
        rows.append({
            "Channel": name,
            "Algorithm": controller,
            "Settling time (s)": settling_time,
            "Overshoot (%)": float(overshoot),
            "Steady-state absolute error (deg)": float(np.rad2deg(np.mean(np.abs(response[-last_window:])))),
            "Settling band (deg)": float(np.rad2deg(band)),
            "Peak absolute rate (deg/s)": float(np.rad2deg(np.max(np.abs(states[:, axis + 3])))),
            "Peak absolute torque (N m)": float(np.max(np.abs(torques[:, axis]))),
            "Control effort integral (N m)^2 s": float(np.sum(torques[:, axis] ** 2) * config.sample_period_s),
            "Torque saturation fraction (%)": float(np.mean(run["saturated"][:, axis]) * 100.0),
        })
    return rows


def _controller_computation_metrics(runs: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for name, run in runs.items():
        is_mpc = name == "Koopman-MPC"
        solve_ms = run["solve_time_ms"]
        active = solve_ms > 0.0
        rows.append({
            "Algorithm": name,
            "Mean controller solve time (ms)": float(np.mean(solve_ms[active])) if np.any(active) else None,
            "95th percentile solve time (ms)": float(np.percentile(solve_ms[active], 95)) if np.any(active) else None,
            "Optimizer success rate (%)": float(np.mean(run["solve_success"][active]) * 100.0) if is_mpc and np.any(active) else None,
            "Fallback steps": int(np.sum(run["solve_fallback"])) if is_mpc else None,
            "Actuator saturation fraction (%)": float(np.mean(np.any(run["saturated"], axis=1)) * 100.0),
            "Total control effort integral (N m)^2 s": float(np.sum(run["torque"] ** 2) * (float(np.mean(np.diff(run["time"]))))),
            "MPC horizon (steps)": int(run.get("mpc_horizon", 0)) if is_mpc else None,
        })
    return rows


def _plot_three_channels(time: Array, actual: Array, predicted: Array, names: tuple[str, str, str], units: str, title: str, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.8), sharex=True, constrained_layout=True)
    for i, ax in enumerate(axes):
        ax.plot(time, actual[:, i], label="Simulated plant", linewidth=1.5)
        ax.plot(time, predicted[:, i], "--", label="Koopman prediction", linewidth=1.3)
        ax.set_ylabel(f"{names[i]} ({units})")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    _safe_savefig(fig, path, dpi=200)
    plt.close(fig)


def _plot_controller_responses(runs: dict[str, dict[str, np.ndarray]], figure_dir: Path, initial_angles_deg: tuple[float, float, float]) -> None:
    axes_names = ("Roll", "Pitch", "Yaw")
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.8), sharex=True, constrained_layout=True)
    for axis, ax in enumerate(axes):
        ax.axhline(0.0, linestyle=":", label="Reference")
        for name, run in runs.items():
            ax.plot(run["time"], np.rad2deg(run["state"][:, axis]), label=name)
        ax.set_ylabel(f"{axes_names[axis]} (deg)")
        ax.grid(True, alpha=0.3)
        if axis == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Attitude response: PID, Koopman-LQR and Koopman-MPC (initial {initial_angles_deg} deg)")
    _safe_savefig(fig, figure_dir / "figure_4_attitude_response.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.8), sharex=True, constrained_layout=True)
    for axis, ax in enumerate(axes):
        for name, run in runs.items():
            ax.plot(run["time"], np.rad2deg(run["state"][:, axis + 3]), label=name)
        ax.set_ylabel(f"Body rate {axis + 1} (deg/s)")
        ax.grid(True, alpha=0.3)
        if axis == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Angular-velocity response")
    _safe_savefig(fig, figure_dir / "figure_5_angular_velocity_response.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.8), sharex=True, constrained_layout=True)
    for axis, ax in enumerate(axes):
        for name, run in runs.items():
            ax.plot(run["time"][:-1], run["torque"][:, axis], label=name)
        ax.set_ylabel(("τx", "τy", "τz")[axis] + " (N m)")
        ax.grid(True, alpha=0.3)
        if axis == 0:
            ax.legend(loc="best")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Attitude control torque")
    _safe_savefig(fig, figure_dir / "figure_6_control_torque.png", dpi=200)
    plt.close(fig)


def _plot_model_order(rows: list[dict[str, Any]], path: Path) -> None:
    xs = [int(row["N"]) for row in rows]
    ys = [float(row["Relative prediction error (N=8 ratio)"]) for row in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(xs, ys, marker="o")
    ax.set_xticks(xs)
    ax.set_xlabel("Lifting order N")
    ax.set_ylabel("H-projected relative error (N=8 = 1.00)")
    ax.set_title("Model-order selection using the first 18 lifted coordinates")
    ax.grid(True, alpha=0.3)
    _safe_savefig(fig, path, dpi=200)
    plt.close(fig)


def _plot_table(rows: list[dict[str, Any]], columns: list[str | tuple[str, str]], path: Path, title: str, font_size: int = 8) -> None:
    keys = [c[0] if isinstance(c, tuple) else c for c in columns]
    labels = [c[1] if isinstance(c, tuple) else c for c in columns]
    cells = []
    for row in rows:
        line = []
        for key in keys:
            value = row.get(key)
            if value is None:
                line.append("—")
            elif isinstance(value, (float, np.floating)):
                line.append(f"{value:.4f}")
            else:
                line.append(str(value))
        cells.append(line)
    fig, ax = plt.subplots(figsize=(max(12, 1.65 * len(keys)), max(2.8, 0.48 * (len(rows) + 2))))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.45)
    for (row_index, _), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold", wrap=True)
        else:
            cell.set_text_props(wrap=True)
    table.auto_set_column_width(col=list(range(len(keys))))
    ax.set_title(title, pad=14)
    fig.tight_layout()
    _safe_savefig(fig, path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
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


def run_study(project_root: Path, config: StudyConfig, regenerate_data: bool = False) -> dict[str, Any]:
    config.validate()
    results = project_root / "results"
    data_dir, tables_dir, figures_dir, models_dir, matrices_dir = (results / x for x in ("data", "tables", "figures", "models", "matrices"))
    for directory in (data_dir, tables_dir, figures_dir, models_dir, matrices_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_path = data_dir / "attitude_dataset.npz"
    if dataset_path.exists() and not regenerate_data:
        data = load_dataset(dataset_path)
    else:
        data = generate_dataset(config)
        save_dataset(data, dataset_path)

    train = transitions_for_split(data, "training")
    validation = transitions_for_split(data, "validation")
    models: dict[int, KoopmanModel] = {}
    table_one: list[dict[str, Any]] = []

    for order in LIFT_ORDERS:
        model = fit_koopman_edmd(train, order=order)
        models[order] = model
        metrics = evaluate_one_step_by_order(model, validation)
        table_one.append({
            "N": order,
            "Lifted dimension": 9 * order,
            "Design rank": model.design_rank,
            "Design condition number": model.design_condition,
            "H-projected lifted RMSE": metrics["h_projected_lifted_rmse"],
            "Angle RMSE (deg)": metrics["angle_rmse_deg"],
            "Body-rate RMSE (rad/s)": metrics["angular_velocity_rmse_rad_s"],
            "Normalized state RMSE": metrics["normalized_state_rmse"],
            "Full lifted-state RMSE": metrics["lifted_rmse"],
        })

    ref_h_rmse = next(float(row["H-projected lifted RMSE"]) for row in table_one if int(row["N"]) == 8)
    for row in table_one:
        row["Relative prediction error (N=8 ratio)"] = float(row["H-projected lifted RMSE"]) / max(ref_h_rmse, 1e-12)

    selected_model = models[5]
    Q_lqr, R_lqr, P_lqr, gain, lqr_diagnostics = _lqr_design(selected_model)
    matrix_diagnostics = _save_matrix_artifacts(
        results, train, selected_model, Q_lqr, R_lqr, P_lqr, gain
    )
    mpc = KoopmanMPC(
        selected_model,
        torque_limits=TORQUE_LIMITS,
        fallback_gain=gain,
        horizon=config.mpc_horizon,
        q_rotation=5.0,
        q_rate=2.0,
        q_higher=0.2,
        r_weights=(0.08, 0.08, 0.05),
    )

    val_id = int(data["validation_indices"][0])
    rtraj, wtraj, utraj = data["rotations"][val_id], data["angular_rates"][val_id], data["torques"][val_id]
    true_angles, pred_angles, true_rates, pred_rates = [], [], [], []
    for k in range(len(utraj)):
        r_pred, w_pred = selected_model.predict_state(rtraj[k], wtraj[k], utraj[k])
        true_angles.append(rotation_to_euler_zyx(rtraj[k + 1]))
        pred_angles.append(rotation_to_euler_zyx(r_pred))
        true_rates.append(wtraj[k + 1])
        pred_rates.append(w_pred)
    time_pred = (np.arange(len(utraj)) + 1) * config.sample_period_s
    _plot_three_channels(time_pred, np.rad2deg(np.asarray(true_angles)), np.rad2deg(np.asarray(pred_angles)),
                         ("Roll", "Pitch", "Yaw"), "deg", "One-step-ahead attitude prediction on a held-out trajectory",
                         figures_dir / "figure_2_attitude_prediction.png")
    _plot_three_channels(time_pred, np.asarray(true_rates), np.asarray(pred_rates),
                         ("Rate 1", "Rate 2", "Rate 3"), "rad/s", "One-step-ahead body angular-velocity prediction on a held-out trajectory",
                         figures_dir / "figure_3_angular_velocity_prediction.png")

    runs = {
        "PID": _simulate_regulation("PID", selected_model, gain, config),
        "Koopman-LQR": _simulate_regulation("Koopman-LQR", selected_model, gain, config),
        "Koopman-MPC": _simulate_regulation("Koopman-MPC", selected_model, gain, config, mpc_controller=mpc),
    }
    _plot_controller_responses(runs, figures_dir, config.control_initial_angles_deg)

    table_two: list[dict[str, Any]] = []
    for run in runs.values():
        table_two.extend(_control_metrics(run, config))
    controller_metrics = _controller_computation_metrics(runs)

    _write_csv(tables_dir / "table_I_model_order.csv", table_one)
    _write_csv(tables_dir / "table_II_control_performance.csv", table_two)
    _write_csv(tables_dir / "controller_computational_metrics.csv", controller_metrics)
    _plot_model_order(table_one, figures_dir / "model_order_error_plot.png")
    _plot_table(
        table_one,
        ["N", "Lifted dimension", ("Relative prediction error (N=8 ratio)", "Relative error\n(N=8 = 1.00)"),
         ("H-projected lifted RMSE", "H-projected\nRMSE"), ("Angle RMSE (deg)", "Angle RMSE\n(deg)"),
         ("Body-rate RMSE (rad/s)", "Rate RMSE\n(rad/s)")],
        figures_dir / "table_I_model_order.png",
        "Table I equivalent: first-18-coordinate relative prediction errors", 8,
    )
    _plot_table(
        table_two,
        ["Channel", "Algorithm", "Settling time (s)", "Overshoot (%)",
         ("Steady-state absolute error (deg)", "Steady-state error\n(deg)")],
        figures_dir / "table_II_control_performance.png",
        "Controller performance: PID, Koopman-LQR and Koopman-MPC", 8,
    )

    np.savez_compressed(
        models_dir / "attitude_models.npz",
        **{f"A_N{n}": m.A for n, m in models.items()},
        **{f"B_N{n}": m.B for n, m in models.items()},
        lqr_gain=gain, Q_lqr=Q_lqr, R_lqr=R_lqr, P_riccati=P_lqr,
        selected_control_order=np.array(5),
    )
    for name, run in runs.items():
        safe_name = name.lower().replace("-", "_")
        np.savez_compressed(
            results / f"{safe_name}_response.npz",
            time=run["time"], state=run["state"], torque=run["torque"],
            saturated=run["saturated"], solve_time_ms=run["solve_time_ms"],
            solve_success=run["solve_success"], solve_fallback=run["solve_fallback"],
            solver_iterations=run["solver_iterations"],
        )

    file_list = [
        "tables/table_I_model_order.csv", "tables/table_II_control_performance.csv",
        "tables/controller_computational_metrics.csv", "figures/table_I_model_order.png",
        "figures/model_order_error_plot.png", "figures/table_II_control_performance.png",
        "figures/figure_2_attitude_prediction.png", "figures/figure_3_angular_velocity_prediction.png",
        "figures/figure_4_attitude_response.png", "figures/figure_5_angular_velocity_response.png",
        "figures/figure_6_control_torque.png", "figures/figure_7_matrix_transparency.png",
        "models/attitude_models.npz", "matrices/attitude_matrices_N5.npz",
        "matrices/matrix_diagnostics.json", "matrices/A_lstsq.csv", "matrices/B_lstsq.csv",
        "matrices/G1.csv", "matrices/G2.csv", "matrices/K_operator_moments.csv",
        "matrices/K_operator_lstsq.csv", "matrices/Q_lqr.csv", "matrices/R_lqr.csv",
        "matrices/P_riccati.csv", "matrices/Omega_lqr.csv",
    ]
    summary = {
        "configuration": asdict(config),
        "dataset": {
            "path": str(dataset_path.relative_to(project_root)),
            "trajectory_count": int(data["rotations"].shape[0]),
            "states_per_trajectory": int(data["rotations"].shape[1]),
            "transitions_per_trajectory": int(data["torques"].shape[1]),
            "training_trajectory_count": int(len(data["training_indices"])),
            "validation_trajectory_count": int(len(data["validation_indices"])),
            "training_transition_count": len(train),
            "validation_transition_count": len(validation),
            "training_indices": data["training_indices"].tolist(),
            "validation_indices": data["validation_indices"].tolist(),
        },
        "table_I_metric_definition": {
            "H_projection": "H = [I_18, 0]; compare the first 18 entries of predicted and actual lifted states.",
            "raw_error": "RMSE over the first 18 lifted coordinates on held-out one-step transitions.",
            "relative_error": "Raw H-projected RMSE for each order divided by the H-projected RMSE for N=8.",
            "additional_metrics": "Physical-angle and angular-rate RMSE are included as supplementary metrics.",
            "note": "Original raw measurement data and the complete error calculation are unavailable; the generated values are simulation-based and must be reported as computed.",
        },
        "table_II_metric_definition": {
            "reference": "Zero roll, pitch and yaw.",
            "settling_time": "First sample after which absolute wrapped angle error remains inside the band for the rest of the run.",
            "settling_band": "2% of the absolute initial angle, with a 0.1-degree minimum; the source does not state its tolerance.",
            "overshoot": "Maximum excursion beyond zero target divided by absolute initial error, expressed as percentage.",
            "steady_state_error": "Mean absolute angle error over the final second.",
            "torque_limits_nm": TORQUE_LIMITS.tolist(),
            "initial_angles_deg": list(config.control_initial_angles_deg),
            "lqr_diagnostics": lqr_diagnostics,
            "pid_design": {
            "kp_nm_per_rad": [0.20, 0.20, 0.12],
            "ki_nm_per_rad_s": [0.010, 0.010, 0.005],
            "kd_nm_per_rad_s": [0.05, 0.05, 0.036],
            "integral_limit_rad_s": [2.0, 2.0, 2.0],
            "anti_windup": "conditional integration under torque saturation",
        },
        "mpc": {"horizon_steps": config.mpc_horizon, "cost_weights_state": {"rotation": 5.0, "rate_features": 2.0, "higher_features": 0.2}, "cost_weights_torque": [0.08, 0.08, 0.05], "solver": "SciPy L-BFGS-B with analytic gradient and bound constraints", "failure_behavior": "Fallback to clipped Koopman-LQR command"},
            "initial_condition_note": "Initial angles were estimated visually from the published response plot; the exact values are not stated in the text.",
            "note": "Controller gains, inertias and physical-platform conditions are not fully specified in the article; numbers describe this simulation setup, not a guaranteed numerical match.",
        },
        "model_order_results": table_one,
        "controller_results": table_two,
        "controller_computational_metrics": controller_metrics,
        "matrix_diagnostics": matrix_diagnostics,
        "data_generation_note": (
            "Synthetic trajectories are generated from the nonlinear rigid-body equations. "
            "The reference is randomized sinusoidal attitude, and a direct inertia-scaled PD "
            "tracking torque is used without exact angular-acceleration feed-forward or exact "
            "gyroscopic cancellation. This is a simulation substitute for unavailable physical logs."
        ),
        "representative_validation_trajectory_id": val_id,
        "files": [_reported_path(project_root, x) for x in file_list],
    }
    with (results / "attitude_study_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, allow_nan=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate attitude figures and controller metrics.")
    parser.add_argument("--regenerate-data", action="store_true", help="Generate a new data set.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config = StudyConfig()
    summary = run_study(root, config, regenerate_data=args.regenerate_data)
    print("Attitude study completed.")
    print(f"Dataset trajectories: {summary['dataset']['trajectory_count']}")
    print("Training/validation trajectories: "
          f"{summary['dataset']['training_trajectory_count']}/"
          f"{summary['dataset']['validation_trajectory_count']}")
    print("Controllers evaluated: PID, Koopman-LQR, Koopman-MPC")
    diag = summary["matrix_diagnostics"]
    print("Identified-matrix transparency artifacts:")
    print("  G1 shape:", diag["matrix_shapes"]["G1"])
    print("  G2 shape:", diag["matrix_shapes"]["G2"])
    print("  K_operator shape:", diag["matrix_shapes"]["K_operator_moments"])
    print("  Relative difference between moment K and least-squares [A B]:",
          f"{diag['K_moment_vs_lstsq_relative_frobenius_difference']:.3e}")
    print(f"Results directory: {root / 'results'}")
    for name in summary["files"]:
        print(f"  {root / 'results' / name}")


if __name__ == "__main__":
    main()
