"""Trajectory-level measurement-noise sensitivity study for Koopman EDMD.

This is a NEW, separate study.  It does not replace src/noise_study.py, which
compares LS-EDMD with covariance-weighted TLS-EDMD under noise.

Protocol
--------
1. Load the project's existing simulated attitude dataset.
2. Fit a clean-data order-N Koopman EDMD reference using training trajectories.
3. For each noise condition and independent seed, perturb the *training
   trajectory measurements once per timestamp* (Euler attitude, body angular
   rates, and recorded torque/input telemetry) and refit EDMD.
4. Keep all held-out validation trajectories clean.
5. Perform open-loop multi-step forecasts at the requested horizons, using the
   clean held-out torque sequence to isolate sensitivity of the learned model.
6. Report normalized state-prediction RMSE mean, sample variance, and standard
   deviation across independent noise realizations.

Important scope: the shared dataset contains attitude rotations, body rates,
and recorded torques. It contains no position/GPS or raw IMU channels; this
study therefore does not inject GPS/position noise or claim to model sensor
bias, drift, correlated vibration, or hardware-calibrated noise statistics.
The Gaussian levels are explicit sensitivity scenarios, not a substitute for
noise characterization from a named physical sensor.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.attitude_core import (
    attitude_lift,
    decode_attitude_lift,
    euler_zyx_to_rotation,
    rotation_to_euler_zyx,
)
from src.identification import (
    AttitudeTransitions,
    fit_koopman_edmd,
    physical_state,
)
from src.model_benchmark import load_common_dataset, validate_dataset

Array = np.ndarray


@dataclass(frozen=True)
class NoiseLevel:
    """Gaussian standard deviations for recorded measurement channels."""

    name: str
    attitude_std_deg: float
    angular_rate_std_rad_s: float
    torque_reading_std_nm: float

    def validate(self) -> None:
        if not self.name or self.name.strip() == "":
            raise ValueError("Noise level name must be a non-empty string.")
        for field_name, value in (
            ("attitude_std_deg", self.attitude_std_deg),
            ("angular_rate_std_rad_s", self.angular_rate_std_rad_s),
            ("torque_reading_std_nm", self.torque_reading_std_nm),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative.")


DEFAULT_NOISE_LEVELS: tuple[NoiseLevel, ...] = (
    NoiseLevel("low", 0.05, 0.003, 0.0001),
    NoiseLevel("medium", 0.15, 0.008, 0.0002),
    NoiseLevel("high", 0.30, 0.015, 0.0005),
)
DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 50)
DEFAULT_REPETITIONS = 5
DEFAULT_SEED = 20261023


def _wrapped_angle_error(predicted: Array, truth: Array) -> Array:
    """Smallest signed angular difference in radians."""
    difference = np.asarray(predicted, dtype=float) - np.asarray(truth, dtype=float)
    return np.arctan2(np.sin(difference), np.cos(difference))


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(sorted(set(int(part.strip()) for part in value.split(","))))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Horizons must be comma-separated positive integers, e.g. 1,5,10,20,50."
        ) from exc
    if not horizons or horizons[0] < 1:
        raise argparse.ArgumentTypeError("Every horizon must be a positive integer.")
    return horizons


def _training_state_scales(data: dict[str, Array]) -> Array:
    """Fixed per-channel scales from clean training trajectories only.

    State channels are [roll, pitch, yaw, p, q, r]; angles are radians and
    angular rates are rad/s.  The same scales are used for all conditions and
    horizons to make NRMSE comparisons fair and avoid validation leakage.
    """
    train_ids = np.asarray(data["training_indices"], dtype=int)
    states = []
    for trajectory_id in train_ids:
        for rotation, omega in zip(
            data["rotations"][trajectory_id],
            data["angular_rates"][trajectory_id],
            strict=True,
        ):
            states.append(physical_state(rotation, omega))
    physical = np.asarray(states, dtype=float)
    scales = np.std(physical, axis=0)
    fallback = np.maximum(np.sqrt(np.mean(physical**2, axis=0)), 1e-6)
    scales = np.maximum(np.where(scales > 1e-8, scales, fallback), 1e-6)
    if scales.shape != (6,) or not np.all(np.isfinite(scales)):
        raise ValueError("Could not derive valid normalization scales from clean training data.")
    return scales


def inject_training_measurement_noise(
    data: dict[str, Array],
    noise_level: NoiseLevel,
    rng: np.random.Generator,
) -> AttitudeTransitions:
    """Create noisy training transitions without modifying clean source data.

    Noise is applied once to each timestamped training observation, so the same
    noisy state is shared by the preceding transition's target and following
    transition's regressor.  Rotation matrices remain valid SO(3) matrices
    because Gaussian error is added to ZYX Euler attitude readings before they
    are converted back to rotation matrices.
    """
    validate_dataset(data)
    noise_level.validate()
    train_ids = np.asarray(data["training_indices"], dtype=int)

    clean_rotations = np.asarray(data["rotations"])[train_ids]
    clean_rates = np.asarray(data["angular_rates"])[train_ids]
    clean_torques = np.asarray(data["torques"])[train_ids]

    noisy_rotations = np.empty_like(clean_rotations, dtype=float)
    angle_std_rad = np.deg2rad(noise_level.attitude_std_deg)

    # Apply one angle perturbation per trajectory timestamp, then reconstruct
    # a proper rotation matrix rather than perturbing matrix entries directly.
    for trajectory_index in range(clean_rotations.shape[0]):
        for sample_index in range(clean_rotations.shape[1]):
            angles = rotation_to_euler_zyx(
                clean_rotations[trajectory_index, sample_index]
            )
            angles = angles + rng.normal(0.0, angle_std_rad, size=3)
            angles = np.arctan2(np.sin(angles), np.cos(angles))
            noisy_rotations[trajectory_index, sample_index] = euler_zyx_to_rotation(
                float(angles[0]), float(angles[1]), float(angles[2])
            )

    noisy_rates = clean_rates + rng.normal(
        0.0, noise_level.angular_rate_std_rad_s, size=clean_rates.shape
    )
    noisy_torque_readings = clean_torques + rng.normal(
        0.0, noise_level.torque_reading_std_nm, size=clean_torques.shape
    )

    # The applied physical trajectory and validation data are never changed;
    # only the training readings supplied to the estimator are perturbed.
    return AttitudeTransitions.from_trajectories(
        noisy_rotations, noisy_rates, noisy_torque_readings
    )


def _rollout_origins(number_of_intervals: int, maximum_horizon: int) -> list[int]:
    if maximum_horizon < 1 or number_of_intervals < maximum_horizon:
        raise ValueError(
            f"Each validation trajectory needs at least {maximum_horizon} intervals; "
            f"it has {number_of_intervals}."
        )
    stride = max(1, maximum_horizon // 2)
    return list(range(0, number_of_intervals - maximum_horizon + 1, stride))


def evaluate_multistep_rollouts(
    model: Any,
    data: dict[str, Array],
    horizons: Sequence[int],
    state_scales: Array,
) -> list[dict[str, Any]]:
    """Evaluate clean held-out open-loop predictions at multiple horizons.

    For each validation trajectory, forecasts start from clean measured states
    at spaced origins. The model is recursively advanced with the *clean
    recorded input sequence* for that held-out interval. It is not re-anchored
    to measured states during the forecast. This isolates errors arising from
    identification on noisy training data from added test-time input noise.
    """
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must contain positive integers.")
    maximum_horizon = max(horizons)
    scales = np.asarray(state_scales, dtype=float).reshape(6)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
        raise ValueError("state_scales must be six finite positive values.")

    rotations_all = np.asarray(data["rotations"])
    rates_all = np.asarray(data["angular_rates"])
    torques_all = np.asarray(data["torques"])
    validation_ids = np.asarray(data["validation_indices"], dtype=int)
    sample_period_s = float(np.asarray(data["sample_period_s"]).item())

    n_intervals = torques_all.shape[1]
    origins = _rollout_origins(n_intervals, maximum_horizon)
    horizon_set = set(horizons)
    angle_errors: dict[int, list[Array]] = {h: [] for h in horizons}
    rate_errors: dict[int, list[Array]] = {h: [] for h in horizons}
    failure_counts: dict[int, int] = {h: 0 for h in horizons}
    expected_windows = len(validation_ids) * len(origins)

    for trajectory_id in validation_ids:
        rotations = rotations_all[trajectory_id]
        rates = rates_all[trajectory_id]
        torques = torques_all[trajectory_id]

        for origin in origins:
            z = attitude_lift(rotations[origin], rates[origin], model.order)
            failed = False
            for step in range(1, maximum_horizon + 1):
                z = model.A @ z + model.B @ torques[origin + step - 1]
                if not np.all(np.isfinite(z)):
                    failed = True
                if failed:
                    for horizon in horizons:
                        if horizon >= step:
                            failure_counts[horizon] += 1
                    break

                if step not in horizon_set:
                    continue
                try:
                    predicted_rotation, predicted_rates = decode_attitude_lift(z)
                    predicted_angles = rotation_to_euler_zyx(predicted_rotation)
                    true_angles = rotation_to_euler_zyx(rotations[origin + step])
                    angle_error = _wrapped_angle_error(predicted_angles, true_angles)
                    rate_error = predicted_rates - rates[origin + step]
                    if not (np.all(np.isfinite(angle_error)) and np.all(np.isfinite(rate_error))):
                        raise FloatingPointError("Decoded physical-state error is non-finite.")
                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    for horizon in horizons:
                        if horizon >= step:
                            failure_counts[horizon] += 1
                    failed = True
                    break
                angle_errors[step].append(angle_error)
                rate_errors[step].append(rate_error)

    results: list[dict[str, Any]] = []
    for horizon in horizons:
        if angle_errors[horizon] and rate_errors[horizon]:
            angle_matrix = np.asarray(angle_errors[horizon], dtype=float).reshape(-1, 3)
            rate_matrix = np.asarray(rate_errors[horizon], dtype=float).reshape(-1, 3)
            all_errors = np.column_stack([angle_matrix, rate_matrix])
            nrmse = float(np.sqrt(np.mean((all_errors / scales) ** 2)))
            attitude_rmse_deg = float(np.rad2deg(np.sqrt(np.mean(angle_matrix**2))))
            angular_rate_rmse = float(np.sqrt(np.mean(rate_matrix**2)))
            status = "ok" if failure_counts[horizon] == 0 else "partial_rollout_failures"
        else:
            nrmse = None
            attitude_rmse_deg = None
            angular_rate_rmse = None
            status = "no_valid_forecasts"

        results.append({
            "horizon_steps": int(horizon),
            "horizon_s": float(horizon * sample_period_s),
            "normalized_prediction_rmse": nrmse,
            "attitude_rmse_deg": attitude_rmse_deg,
            "angular_rate_rmse_rad_s": angular_rate_rmse,
            "expected_forecast_windows": int(expected_windows),
            "valid_forecast_windows": int(len(angle_errors[horizon])),
            "failed_forecast_windows": int(failure_counts[horizon]),
            "status": status,
        })
    return results


def _fit_clean_training_model(data: dict[str, Array], order: int) -> Any:
    train_ids = np.asarray(data["training_indices"], dtype=int)
    return fit_koopman_edmd(
        AttitudeTransitions.from_trajectories(
            np.asarray(data["rotations"])[train_ids],
            np.asarray(data["angular_rates"])[train_ids],
            np.asarray(data["torques"])[train_ids],
        ),
        order=order,
    )


def _aggregate_trial_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(str(row["noise_level"]), int(row["horizon_steps"])) for row in rows})
    metric_names = (
        "normalized_prediction_rmse",
        "attitude_rmse_deg",
        "angular_rate_rmse_rad_s",
        "fit_time_s",
    )
    summary: list[dict[str, Any]] = []
    for noise_level, horizon in keys:
        group = [
            row for row in rows
            if row["noise_level"] == noise_level and int(row["horizon_steps"]) == horizon
        ]
        baseline = noise_level == "clean_reference"
        requested_repetitions = 1 if baseline else len(group)
        successful = [
            row for row in group
            if row.get("normalized_prediction_rmse") is not None
            and np.isfinite(float(row["normalized_prediction_rmse"]))
        ]
        record: dict[str, Any] = {
            "noise_level": noise_level,
            "horizon_steps": horizon,
            "horizon_s": float(group[0]["horizon_s"]),
            "requested_repetitions": int(requested_repetitions),
            "successful_repetitions": int(len(successful)),
            "repetitions_with_rollout_failures": int(sum(
                int(row.get("failed_forecast_windows") or 0) > 0 for row in group
            )),
            "expected_forecast_windows_per_repetition": int(group[0]["expected_forecast_windows"]),
            "mean_valid_forecast_windows": float(np.mean([
                int(row.get("valid_forecast_windows") or 0) for row in group
            ])) if group else None,
        }
        for metric in metric_names:
            values = np.asarray([
                float(row[metric]) for row in successful
                if row.get(metric) is not None and np.isfinite(float(row[metric]))
            ], dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values)) if values.size else None
            if values.size > 1:
                record[f"{metric}_std"] = float(np.std(values, ddof=1))
                record[f"{metric}_variance"] = float(np.var(values, ddof=1))
            elif values.size == 1:
                record[f"{metric}_std"] = 0.0
                record[f"{metric}_variance"] = 0.0
            else:
                record[f"{metric}_std"] = None
                record[f"{metric}_variance"] = None
        summary.append(record)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(
    summary_rows: list[dict[str, Any]],
    path: Path,
    noise_levels: Sequence[NoiseLevel],
) -> tuple[str, ...]:
    """Write one three-panel, logarithmic comparison figure.

    The panels deliberately use separate y-axes because the metrics have
    different units and magnitudes. Means are shown as lines; the shaded band
    is mean +/- one sample standard deviation across noise realizations.
    Since log axes cannot display zero, zero values are placed at a small,
    metric-specific plotting floor only. CSV/JSON values remain unchanged.
    """
    levels = ["clean_reference", *(level.name for level in noise_levels)]
    labels = {
        "clean_reference": "Clean training reference",
        "low": "Low noise",
        "medium": "Medium noise",
        "high": "High noise",
    }
    metric_specs = (
        ("normalized_prediction_rmse", "Normalized prediction RMSE (dimensionless)"),
        ("attitude_rmse_deg", "Attitude RMSE (deg)"),
        ("angular_rate_rmse_rad_s", "Body-rate RMSE (rad/s)"),
    )
    fig, axes = plt.subplots(
        nrows=3, ncols=1, figsize=(10.5, 11.0), sharex=True, constrained_layout=True
    )
    plot_floors: dict[str, float] = {}

    for ax, (metric, ylabel) in zip(axes, metric_specs, strict=True):
        all_positive: list[float] = []
        for row in summary_rows:
            value = row.get(f"{metric}_mean")
            if value is not None and np.isfinite(float(value)) and float(value) > 0:
                all_positive.append(float(value))
        # One decade below the smallest positive value keeps zero-valued points
        # visible without letting a hard-coded tiny floor dominate the plot.
        floor = max(min(all_positive) * 0.1, 1e-12) if all_positive else 1e-12
        plot_floors[metric] = floor

        for level in levels:
            subset = sorted(
                [row for row in summary_rows if row["noise_level"] == level],
                key=lambda row: int(row["horizon_steps"]),
            )
            if not subset:
                continue
            x = np.asarray([float(row["horizon_s"]) for row in subset], dtype=float)
            means_raw = [row.get(f"{metric}_mean") for row in subset]
            stds_raw = [row.get(f"{metric}_std") for row in subset]
            means = np.asarray([
                np.nan if value is None or not np.isfinite(float(value))
                else max(float(value), floor)
                for value in means_raw
            ], dtype=float)
            stds = np.asarray([
                0.0 if value is None or not np.isfinite(float(value))
                else max(float(value), 0.0)
                for value in stds_raw
            ], dtype=float)
            valid = np.isfinite(means)
            if not np.any(valid):
                continue
            ax.plot(x[valid], means[valid], marker="o", linewidth=1.8, markersize=4,
                    label=labels.get(level, level))
            lower = np.maximum(means - stds, floor)
            upper = np.maximum(means + stds, floor)
            ax.fill_between(x, lower, upper, where=valid, alpha=0.12, interpolate=False)

        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Open-loop prediction horizon (s)")
    fig.suptitle("Trajectory measurement-noise sensitivity: three prediction metrics\n(logarithmic y-axes)", fontsize=13)
    fig.text(
        0.5, -0.005,
        "Lower is better. Shading: mean +/- 1 sample SD across noise realizations. "
        "Zero values use a plotting floor on log axes; saved numeric results are unchanged.",
        ha="center", va="top", fontsize=8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    scales = tuple(ax.get_yscale() for ax in axes)
    try:
        fig.savefig(temporary, dpi=200, bbox_inches="tight")
        temporary.replace(path)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    if scales != ("log", "log", "log"):
        raise RuntimeError(f"Expected logarithmic y-axes for all metrics, got {scales}.")
    return scales


def run_trajectory_noise_sensitivity(
    data: dict[str, Array],
    results_root: str | Path,
    *,
    order: int = 5,
    repetitions: int = DEFAULT_REPETITIONS,
    seed: int = DEFAULT_SEED,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    noise_levels: Sequence[NoiseLevel] = DEFAULT_NOISE_LEVELS,
) -> dict[str, Any]:
    """Fit and evaluate clean/noisy Koopman models using a common data split."""
    validate_dataset(data)
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError("order must be an integer >= 2.")
    if not isinstance(repetitions, (int, np.integer)) or repetitions < 2:
        raise ValueError("repetitions must be >= 2 to estimate variance across noise realizations.")
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must contain positive integers.")
    maximum_horizon = max(horizons)
    if np.asarray(data["torques"]).shape[1] < maximum_horizon:
        raise ValueError("The longest prediction horizon exceeds the trajectory length.")
    if not noise_levels:
        raise ValueError("At least one nonzero noise level must be specified.")
    names: set[str] = set()
    for level in noise_levels:
        level.validate()
        if level.name == "clean_reference":
            raise ValueError("Noise-level name 'clean_reference' is reserved for the baseline.")
        if level.name in names:
            raise ValueError(f"Duplicate noise level name: {level.name}")
        names.add(level.name)

    output_dir = Path(results_root) / "trajectory_noise_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_period_s = float(np.asarray(data["sample_period_s"]).item())
    scales = _training_state_scales(data)
    trial_rows: list[dict[str, Any]] = []

    # Baseline: same train/validation split and order, but no measurement noise.
    clean_fit_started_local = datetime.now().astimezone().isoformat(timespec="seconds")
    clean_fit_start = perf_counter()
    clean_model = _fit_clean_training_model(data, order)
    clean_fit_time_s = float(perf_counter() - clean_fit_start)
    clean_forecasts = evaluate_multistep_rollouts(clean_model, data, horizons, scales)
    for metrics in clean_forecasts:
        trial_rows.append({
            "noise_level": "clean_reference",
            "repetition": 0,
            "seed": int(seed),
            "order": int(order),
            "attitude_std_deg": 0.0,
            "angular_rate_std_rad_s": 0.0,
            "torque_reading_std_nm": 0.0,
            "fit_time_s": clean_fit_time_s,
            **metrics,
        })

    for level_index, level in enumerate(noise_levels):
        for repetition_index in range(repetitions):
            trial_seed = int(seed + level_index * 100_000 + repetition_index * 997 + 1)
            rng = np.random.default_rng(trial_seed)
            tic = perf_counter()
            try:
                noisy_training = inject_training_measurement_noise(data, level, rng)
                model = fit_koopman_edmd(noisy_training, order=order)
                fit_time_s = float(perf_counter() - tic)
                forecast_rows = evaluate_multistep_rollouts(model, data, horizons, scales)
                for metrics in forecast_rows:
                    trial_rows.append({
                        "noise_level": level.name,
                        "repetition": int(repetition_index + 1),
                        "seed": trial_seed,
                        "order": int(order),
                        "attitude_std_deg": float(level.attitude_std_deg),
                        "angular_rate_std_rad_s": float(level.angular_rate_std_rad_s),
                        "torque_reading_std_nm": float(level.torque_reading_std_nm),
                        "fit_time_s": fit_time_s,
                        **metrics,
                    })
            except Exception as exc:
                # Keep the failed realization explicit; do not silently discard it.
                for horizon in horizons:
                    trial_rows.append({
                        "noise_level": level.name,
                        "repetition": int(repetition_index + 1),
                        "seed": trial_seed,
                        "order": int(order),
                        "attitude_std_deg": float(level.attitude_std_deg),
                        "angular_rate_std_rad_s": float(level.angular_rate_std_rad_s),
                        "torque_reading_std_nm": float(level.torque_reading_std_nm),
                        "fit_time_s": None,
                        "horizon_steps": int(horizon),
                        "horizon_s": float(horizon * sample_period_s),
                        "normalized_prediction_rmse": None,
                        "attitude_rmse_deg": None,
                        "angular_rate_rmse_rad_s": None,
                        "expected_forecast_windows": None,
                        "valid_forecast_windows": 0,
                        "failed_forecast_windows": None,
                        "status": f"failed_trial:{type(exc).__name__}",
                        "error": str(exc),
                    })

    summary_rows = _aggregate_trial_rows(trial_rows)
    trials_csv = output_dir / "trajectory_noise_sensitivity_trials.csv"
    summary_csv = output_dir / "trajectory_noise_sensitivity_summary.csv"
    plot_path = output_dir / "trajectory_noise_sensitivity_metrics_log.png"
    summary_json = output_dir / "trajectory_noise_sensitivity_summary.json"
    _write_csv(trials_csv, trial_rows)
    _write_csv(summary_csv, summary_rows)
    _plot_summary(summary_rows, plot_path, noise_levels)

    summary: dict[str, Any] = {
        "experiment": "Trajectory-level measurement-noise sensitivity of a Koopman EDMD attitude model",
        "generated_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset": "results/data/attitude_dataset.npz",
        "dataset_properties": {
            "trajectory_count": int(np.asarray(data["rotations"]).shape[0]),
            "training_trajectory_count": int(len(data["training_indices"])),
            "validation_trajectory_count": int(len(data["validation_indices"])),
            "states_per_trajectory": int(np.asarray(data["rotations"]).shape[1]),
            "transitions_per_trajectory": int(np.asarray(data["torques"]).shape[1]),
            "sample_period_s": sample_period_s,
            "validation_trajectories_remain_clean": True,
        },
        "configuration": {
            "koopman_order": int(order),
            "lifted_dimension": int(9 * order),
            "repetitions_per_noise_level": int(repetitions),
            "base_random_seed": int(seed),
            "prediction_horizons_steps": list(horizons),
            "prediction_horizons_s": [float(h * sample_period_s) for h in horizons],
            "noise_levels": [asdict(level) for level in noise_levels],
            "normalization_scales_clean_training_state_std": scales.tolist(),
            "forecast_origin_rule": (
                "Clean validation origins spaced by max(1, floor(maximum_horizon/2)) samples; "
                "forecasts are open-loop within each window."
            ),
        },
        "method": {
            "training": (
                "Fit standard controlled EDMD z[k+1] = A z[k] + B u[k] on clean training trajectories "
                "for the reference, then on noisy training observations for each noise realization."
            ),
            "noise_model": (
                "Independent zero-mean Gaussian additive noise is applied per timestamp to ZYX attitude "
                "readings, body angular-rate measurements, and recorded torque/input telemetry. Attitude "
                "measurements are converted back to valid SO(3) matrices."
            ),
            "ground_truth": (
                "Original trajectories are not modified. All evaluation uses untouched clean held-out "
                "validation trajectories; clean recorded validation inputs are used to isolate training-noise sensitivity."
            ),
            "plot_configuration": {
                "metrics": ["normalized_prediction_rmse", "attitude_rmse_deg", "angular_rate_rmse_rad_s"],
                "y_scale": "logarithmic for each metric panel",
                "zero_display_rule": "zero values are shown at a metric-specific positive plotting floor; CSV/JSON values are not modified",
                "uncertainty_band": "mean plus/minus one sample standard deviation across independent noise realizations",
            },
            "normalized_rmse_definition": (
                "NRMSE_H = sqrt(mean(((euler_angle_error_wrapped, body_rate_error) / s_clean_train)^2)); "
                "s_clean_train is the per-channel standard deviation of clean training physical states [roll,pitch,yaw,p,q,r]."
            ),
            "uncertainty_reporting": (
                "Mean, sample variance (ddof=1), and sample standard deviation of per-realization forecast metrics "
                "are computed for each noise condition and horizon. Failed/partial rollouts are kept visible in the trial table."
            ),
        },
        "limitations": [
            "Dataset trajectories are simulated, not raw physical flight recordings.",
            "The dataset has attitude rotations, body rates, and torque telemetry but no GPS/position or raw IMU channels; position/GPS noise is therefore not injected.",
            "Gaussian white measurement noise is a sensitivity approximation. Sensor bias, drift, temporal correlation, and explicit vibration spectra are not separately modelled.",
            "Torque noise represents uncertainty in the recorded input/torque channel used for identification; it does not change the original plant trajectory or physical applied controls.",
            "Multi-step validation is open-loop and uses the clean held-out input sequence; this isolates training-measurement-noise effects rather than testing a noisy closed-loop controller.",
        ],
        "outputs": [
            "trajectory_noise_sensitivity_trials.csv",
            "trajectory_noise_sensitivity_summary.csv",
            "trajectory_noise_sensitivity_metrics_log.png",
            "trajectory_noise_sensitivity_summary.json",
        ],
        "clean_reference_fit_started_local": clean_fit_started_local,
        "clean_reference_fit_time_s": clean_fit_time_s,
    }
    with summary_json.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)

    return {
        "summary": summary,
        "trial_rows": trial_rows,
        "summary_rows": summary_rows,
        "output_dir": str(output_dir.resolve()),
        "trials_csv": str(trials_csv.resolve()),
        "summary_csv": str(summary_csv.resolve()),
        "plot": str(plot_path.resolve()),
        "summary_json": str(summary_json.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train order-5 Koopman EDMD models with trajectory measurement noise and "
            "evaluate clean held-out multi-step prediction accuracy."
        )
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("results/data/attitude_dataset.npz"),
        help="Existing clean attitude dataset; it is read-only and never regenerated.",
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results"),
        help="Results root (default: results). Study outputs go into trajectory_noise_sensitivity/.",
    )
    parser.add_argument("--order", type=int, default=5, help="Koopman lift order (default: 5).")
    parser.add_argument(
        "--repetitions", type=int, default=DEFAULT_REPETITIONS,
        help="Independent noise realizations per noise level (default: 5; minimum: 2).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Base random seed.")
    parser.add_argument(
        "--horizons", type=_parse_horizons, default=DEFAULT_HORIZONS,
        help="Comma-separated horizons in samples (default: 1,5,10,20,50; sample period is 0.05 s).",
    )
    args = parser.parse_args(argv)
    try:
        data = load_common_dataset(args.dataset)
        result = run_trajectory_noise_sensitivity(
            data,
            args.results_root,
            order=args.order,
            repetitions=args.repetitions,
            seed=args.seed,
            horizons=args.horizons,
        )
    except Exception as exc:
        parser.exit(2, f"Trajectory-noise sensitivity study failed: {type(exc).__name__}: {exc}\n")

    print("Trajectory measurement-noise sensitivity study completed.")
    print(f"Dataset (unchanged): {args.dataset.resolve()}")
    print(f"Order N={args.order}; repetitions per noise level={args.repetitions}")
    print(f"Horizons (samples): {','.join(str(h) for h in args.horizons)}")
    print(f"Output folder: {result['output_dir']}")
    print(f"Trial-level CSV: {result['trials_csv']}")
    print(f"Mean/variance summary CSV: {result['summary_csv']}")
    print(f"Single summary figure: {result['plot']}")
    print(f"Study protocol and limitations: {result['summary_json']}")
    print("Metrics: normalized prediction RMSE, attitude RMSE (deg), and body-rate RMSE (rad/s).")
    print("Figure: three separate logarithmic y-axis panels; shaded bands show +/- one sample standard deviation.")
    print("Zero values are floored for plotting only; CSV/JSON retain the original numeric values.")
    for row in result["summary_rows"]:
        if row["noise_level"] == "clean_reference":
            continue
        if row["normalized_prediction_rmse_mean"] is None:
            mean_std = "no valid forecasts"
        else:
            mean_std = (
                f"mean NRMSE={row['normalized_prediction_rmse_mean']:.5g}, "
                f"std={row['normalized_prediction_rmse_std']:.3g}, "
                f"variance={row['normalized_prediction_rmse_variance']:.3g}"
            )
        print(
            f"  {row['noise_level']:>6} | H={row['horizon_steps']:>2} "
            f"({row['horizon_s']:.2f}s): {mean_std}; "
            f"valid repetitions={row['successful_repetitions']}/{row['requested_repetitions']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
