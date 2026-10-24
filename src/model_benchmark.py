
"""EDMD model-order and DMDc benchmark using the shared attitude dataset.

The primary attitude-model table is left unchanged. This extension report
compares EDMD orders N=3..8 against a six-state controlled-linear DMDc
baseline using common physical-state metrics and multi-step open-loop rollouts.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.attitude_core import attitude_lift, decode_attitude_lift
from src.identification import (
    AttitudeTransitions,
    DMDcModel,
    KoopmanModel,
    fit_dmdc,
    fit_koopman_edmd,
    physical_state,
)

DEFAULT_ORDERS = (3, 4, 5, 6, 7, 8)
DEFAULT_HORIZONS = (1, 5, 10, 20)
REQUIRED_DATA_KEYS = {
    "rotations", "angular_rates", "torques", "training_indices",
    "validation_indices", "sample_period_s", "duration_s", "seed",
    "training_trajectory_count",
}


def _wrap_angles(values: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(values), np.cos(values))


def validate_dataset(data: dict[str, np.ndarray]) -> None:
    missing = REQUIRED_DATA_KEYS.difference(data)
    if missing:
        raise ValueError(f"Dataset is missing required fields: {sorted(missing)}")

    rotations = np.asarray(data["rotations"])
    rates = np.asarray(data["angular_rates"])
    torques = np.asarray(data["torques"])

    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (trajectories, states, 3, 3).")

    n_trajectories, n_states = rotations.shape[:2]

    if rates.shape != (n_trajectories, n_states, 3):
        raise ValueError("angular_rates shape does not match rotations.")

    if torques.shape != (n_trajectories, n_states - 1, 3):
        raise ValueError("torques must contain one sample per state transition.")

    train = np.asarray(data["training_indices"], dtype=int).reshape(-1)
    valid = np.asarray(data["validation_indices"], dtype=int).reshape(-1)

    if np.intersect1d(train, valid).size:
        raise ValueError("Training and validation trajectory IDs overlap.")

    all_ids = np.concatenate([train, valid])
    if len(all_ids) != n_trajectories or not np.array_equal(
        np.sort(all_ids), np.arange(n_trajectories)
    ):
        raise ValueError("Training/validation indices must partition all trajectories.")

    for name, value in data.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Dataset field '{name}' contains non-finite values.")


def load_common_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load the existing project dataset without generating or replacing it."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Shared dataset not found: {path}. "
            "Run 'python -m src.attitude_study' first."
        )

    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}

    validate_dataset(data)
    return data


def transitions_for_split(
    data: dict[str, np.ndarray], split: str
) -> AttitudeTransitions:
    validate_dataset(data)

    if split == "training":
        indices = np.asarray(data["training_indices"], dtype=int)
    elif split == "validation":
        indices = np.asarray(data["validation_indices"], dtype=int)
    else:
        raise ValueError("split must be 'training' or 'validation'.")

    return AttitudeTransitions.from_trajectories(
        data["rotations"][indices],
        data["angular_rates"][indices],
        data["torques"][indices],
    )


def _state_scales(data: dict[str, np.ndarray]) -> np.ndarray:
    """Common normalization scales calculated from held-out physical states."""
    indices = np.asarray(data["validation_indices"], dtype=int)
    rows = []

    for trajectory_id in indices:
        for rotation, rate in zip(
            data["rotations"][trajectory_id],
            data["angular_rates"][trajectory_id],
            strict=True,
        ):
            rows.append(physical_state(rotation, rate))

    states = np.asarray(rows, dtype=float)
    scales = np.std(states, axis=0)
    fallback = np.maximum(np.sqrt(np.mean(states ** 2, axis=0)), 1e-6)

    return np.maximum(np.where(scales > 1e-8, scales, fallback), 1e-6)


def _metrics(
    predictions: Sequence[np.ndarray],
    truths: Sequence[np.ndarray],
    scales: np.ndarray,
) -> dict[str, float]:
    if len(predictions) == 0:
        return {
            "angle_rmse_deg": float("nan"),
            "body_rate_rmse_rad_s": float("nan"),
            "normalized_state_rmse": float("nan"),
        }

    pred = np.asarray(predictions, dtype=float).reshape(-1, 6)
    true = np.asarray(truths, dtype=float).reshape(-1, 6)

    error = np.column_stack([
        _wrap_angles(pred[:, :3] - true[:, :3]),
        pred[:, 3:] - true[:, 3:],
    ])

    channel_rmse = np.sqrt(np.mean(error ** 2, axis=0))

    return {
        "angle_rmse_deg": float(
            np.rad2deg(np.sqrt(np.mean(error[:, :3] ** 2)))
        ),
        "body_rate_rmse_rad_s": float(
            np.sqrt(np.mean(error[:, 3:] ** 2))
        ),
        "normalized_state_rmse": float(
            np.sqrt(np.mean((channel_rmse / scales) ** 2))
        ),
    }


def _predict_one_step_edmd(
    model: KoopmanModel, data: AttitudeTransitions
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []

    for rotation, rate, torque, rotation_next, rate_next in zip(
        data.rotation_current, data.omega_current, data.torque,
        data.rotation_next, data.omega_next, strict=True,
    ):
        z = attitude_lift(rotation, rate, model.order)
        z_next = model.A @ z + model.B @ torque
        rotation_pred, rate_pred = decode_attitude_lift(z_next)

        predictions.append(physical_state(rotation_pred, rate_pred))
        truths.append(physical_state(rotation_next, rate_next))

    return predictions, truths


def _predict_one_step_dmdc(
    model: DMDcModel, data: AttitudeTransitions
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []

    for rotation, rate, torque, rotation_next, rate_next in zip(
        data.rotation_current, data.omega_current, data.torque,
        data.rotation_next, data.omega_next, strict=True,
    ):
        state = physical_state(rotation, rate)
        predictions.append(model.A @ state + model.B @ torque)
        truths.append(physical_state(rotation_next, rate_next))

    return predictions, truths


def _one_step_h_error(
    model: KoopmanModel, data: AttitudeTransitions
) -> float:
    """Compute H-projected RMSE for H=[I_18,0], for EDMD models only."""
    squared_errors = []

    for rotation, rate, torque, rotation_next, rate_next in zip(
        data.rotation_current, data.omega_current, data.torque,
        data.rotation_next, data.omega_next, strict=True,
    ):
        z = attitude_lift(rotation, rate, model.order)
        predicted = model.A @ z + model.B @ torque
        target = attitude_lift(rotation_next, rate_next, model.order)

        squared_errors.append(
            float(np.mean((predicted[:18] - target[:18]) ** 2))
        )

    return float(np.sqrt(np.mean(squared_errors)))


def _rollout_at_horizons(
    model: KoopmanModel | DMDcModel,
    family: str,
    data: dict[str, np.ndarray],
    horizons: tuple[int, ...],
    scales: np.ndarray,
    segment_stride: int | None = None,
    state_magnitude_limit: float = 1e6,
) -> list[dict[str, Any]]:
    """Evaluate open-loop predictions on held-out trajectory windows.

    Each window starts from a measured state. Thereafter, model predictions
    feed subsequent predictions without resetting from plant measurements.
    The recorded validation torque sequence is supplied as the input. This is
    a model-prediction test, not a closed-loop controller experiment.
    """
    horizons = tuple(sorted(set(int(h) for h in horizons)))

    if not horizons or horizons[0] < 1:
        raise ValueError("horizons must be positive integers.")

    max_horizon = max(horizons)
    stride = max_horizon if segment_stride is None else int(segment_stride)

    if stride < 1:
        raise ValueError("segment_stride must be positive.")

    prediction_by_h = {h: [] for h in horizons}
    truth_by_h = {h: [] for h in horizons}
    attempts_by_h = {h: 0 for h in horizons}
    failures_by_h = {h: 0 for h in horizons}

    for trajectory_id in np.asarray(data["validation_indices"], dtype=int):
        rotations = data["rotations"][trajectory_id]
        rates = data["angular_rates"][trajectory_id]
        torques = data["torques"][trajectory_id]
        n_intervals = len(torques)

        if n_intervals < max_horizon:
            continue

        starts = range(0, n_intervals - max_horizon + 1, stride)

        for start in starts:
            if family == "EDMD":
                assert isinstance(model, KoopmanModel)
                state = attitude_lift(
                    rotations[start], rates[start], model.order
                )
            elif family == "DMDc":
                assert isinstance(model, DMDcModel)
                state = physical_state(rotations[start], rates[start])
            else:
                raise ValueError(f"Unknown model family: {family}")

            alive = True

            for step in range(1, max_horizon + 1):
                if alive:
                    control = torques[start + step - 1]

                    with np.errstate(over="ignore", invalid="ignore"):
                        state = model.A @ state + model.B @ control

                    if (
                        not np.all(np.isfinite(state))
                        or np.max(np.abs(state)) > state_magnitude_limit
                    ):
                        alive = False

                if step not in attempts_by_h:
                    continue

                attempts_by_h[step] += 1

                if not alive:
                    failures_by_h[step] += 1
                    continue

                try:
                    if family == "EDMD":
                        rotation_pred, rate_pred = decode_attitude_lift(state)
                        predicted_state = physical_state(rotation_pred, rate_pred)
                    else:
                        predicted_state = np.asarray(state, dtype=float).copy()

                    truth_state = physical_state(
                        rotations[start + step], rates[start + step]
                    )

                    if not np.all(np.isfinite(predicted_state)):
                        failures_by_h[step] += 1
                    else:
                        prediction_by_h[step].append(predicted_state)
                        truth_by_h[step].append(truth_state)

                except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                    failures_by_h[step] += 1
                    alive = False

    rows: list[dict[str, Any]] = []

    for horizon in horizons:
        values = _metrics(
            prediction_by_h[horizon],
            truth_by_h[horizon],
            scales,
        )

        attempts = attempts_by_h[horizon]
        failures = failures_by_h[horizon]

        rows.append({
            "model": (
                f"EDMD_N{model.order}" if family == "EDMD" else "DMDc_linear"
            ),
            "family": family,
            "horizon_steps": horizon,
            "horizon_seconds": float(
                horizon * float(data["sample_period_s"])
            ),
            "windows_attempted": attempts,
            "windows_failed": failures,
            "failure_rate_pct": 100.0 * failures / max(attempts, 1),
            **values,
        })

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))

    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _save_model_order_plot(
    rows: list[dict[str, Any]], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    edmd_rows = [row for row in rows if row["family"] == "EDMD"]

    ax.plot(
        [row["parameter_count"] for row in edmd_rows],
        [row["one_step_normalized_state_rmse"] for row in edmd_rows],
        marker="o",
        label="EDMD lifted models",
    )

    dmdc = next(row for row in rows if row["family"] == "DMDc")
    ax.scatter(
        [dmdc["parameter_count"]],
        [dmdc["one_step_normalized_state_rmse"]],
        marker="s",
        s=75,
        label="DMDc linear baseline",
    )

    ax.annotate(
        "DMDc",
        (dmdc["parameter_count"], dmdc["one_step_normalized_state_rmse"]),
        xytext=(6, 6),
        textcoords="offset points",
    )

    for row in edmd_rows:
        ax.annotate(
            f"N={row['N']}",
            (row["parameter_count"], row["one_step_normalized_state_rmse"]),
            xytext=(4, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("Number of fitted A and B coefficients")
    ax.set_ylabel("One-step normalized physical-state RMSE (lower is better)")
    ax.set_title("DMDc baseline versus lifted EDMD models")
    ax.grid(True, alpha=0.3)
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.tmp{path.suffix}")

    try:
        fig.savefig(temp, dpi=180)
        temp.replace(path)
    finally:
        plt.close(fig)
        if temp.exists():
            temp.unlink()


def _save_rollout_plot(
    rows: list[dict[str, Any]], path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    models = sorted(set(row["model"] for row in rows))

    for name in models:
        subset = sorted(
            (row for row in rows if row["model"] == name),
            key=lambda row: row["horizon_steps"],
        )

        ax.plot(
            [row["horizon_seconds"] for row in subset],
            [row["normalized_state_rmse"] for row in subset],
            marker="o",
            label=name,
        )

    ax.set_xlabel("Open-loop prediction horizon (s)")
    ax.set_ylabel("Normalized physical-state RMSE (lower is better)")
    ax.set_title("Multi-step open-loop prediction error")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.tmp{path.suffix}")

    try:
        fig.savefig(temp, dpi=180)
        temp.replace(path)
    finally:
        plt.close(fig)
        if temp.exists():
            temp.unlink()


def run_benchmark(
    dataset: dict[str, np.ndarray],
    results_dir: str | Path,
    orders: tuple[int, ...] = DEFAULT_ORDERS,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    """Fit EDMD candidates and DMDc, then write comparable result files."""
    validate_dataset(dataset)
    orders = tuple(int(n) for n in orders)
    horizons = tuple(sorted(set(int(h) for h in horizons)))

    if not orders or any(n < 2 for n in orders):
        raise ValueError("orders must contain integers >= 2.")
    if not horizons or min(horizons) < 1:
        raise ValueError("horizons must contain positive integers.")

    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    models_dir = results_dir / "models"

    for folder in (tables_dir, figures_dir, models_dir):
        folder.mkdir(parents=True, exist_ok=True)

    training = transitions_for_split(dataset, "training")
    validation = transitions_for_split(dataset, "validation")
    scales = _state_scales(dataset)

    fit_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    saved: dict[str, np.ndarray] = {}

    for order in orders:
        started = time.perf_counter()
        model = fit_koopman_edmd(training, order=order)
        fit_seconds = time.perf_counter() - started

        predictions, truths = _predict_one_step_edmd(model, validation)
        metrics = _metrics(predictions, truths, scales)
        h_rmse = _one_step_h_error(model, validation)

        fit_rows.append({
            "model": f"EDMD_N{order}",
            "family": "EDMD",
            "N": order,
            "lift_dimension": 9 * order,
            "parameter_count": int(model.A.size + model.B.size),
            "design_rank": model.design_rank,
            "design_condition_number": model.design_condition,
            "fit_time_seconds": fit_seconds,
            "one_step_h_projected_rmse": h_rmse,
            "one_step_angle_rmse_deg": metrics["angle_rmse_deg"],
            "one_step_body_rate_rmse_rad_s": metrics["body_rate_rmse_rad_s"],
            "one_step_normalized_state_rmse": metrics["normalized_state_rmse"],
            "improvement_vs_dmdc_pct": None,
        })

        saved[f"edmd_N{order}_A"] = model.A
        saved[f"edmd_N{order}_B"] = model.B

        rollout_rows.extend(_rollout_at_horizons(
            model, "EDMD", dataset, horizons, scales
        ))

    started = time.perf_counter()
    dmdc_model = fit_dmdc(training)
    dmdc_fit_seconds = time.perf_counter() - started

    predictions, truths = _predict_one_step_dmdc(dmdc_model, validation)
    dmdc_metrics = _metrics(predictions, truths, scales)

    dmdc_row = {
        "model": "DMDc_linear",
        "family": "DMDc",
        "N": 1,
        "lift_dimension": 6,
        "parameter_count": int(dmdc_model.A.size + dmdc_model.B.size),
        "design_rank": dmdc_model.design_rank,
        "design_condition_number": dmdc_model.design_condition,
        "fit_time_seconds": dmdc_fit_seconds,
        "one_step_h_projected_rmse": None,
        "one_step_angle_rmse_deg": dmdc_metrics["angle_rmse_deg"],
        "one_step_body_rate_rmse_rad_s": dmdc_metrics["body_rate_rmse_rad_s"],
        "one_step_normalized_state_rmse": dmdc_metrics["normalized_state_rmse"],
        "improvement_vs_dmdc_pct": 0.0,
    }

    fit_rows.append(dmdc_row)
    saved["dmdc_A"] = dmdc_model.A
    saved["dmdc_B"] = dmdc_model.B

    rollout_rows.extend(_rollout_at_horizons(
        dmdc_model, "DMDc", dataset, horizons, scales
    ))

    dmdc_error = float(dmdc_row["one_step_normalized_state_rmse"])

    for row in fit_rows:
        if row["family"] == "EDMD":
            edmd_error = float(row["one_step_normalized_state_rmse"])
            row["improvement_vs_dmdc_pct"] = (
                100.0 * (dmdc_error - edmd_error) / max(dmdc_error, 1e-12)
            )

    _write_csv(tables_dir / "dmdc_model_benchmark.csv", fit_rows)
    _write_csv(tables_dir / "model_rollout_benchmark.csv", rollout_rows)

    _save_model_order_plot(
        fit_rows, figures_dir / "dmdc_model_benchmark.png"
    )
    _save_rollout_plot(
        rollout_rows, figures_dir / "model_rollout_benchmark.png"
    )

    np.savez_compressed(
        models_dir / "dmdc_edmd_benchmark.npz",
        **saved,
    )

    edmd_fit_rows = [row for row in fit_rows if row["family"] == "EDMD"]
    best_one = min(
        edmd_fit_rows,
        key=lambda row: row["one_step_normalized_state_rmse"],
    )

    longest_horizon = max(horizons)
    edmd_long = [
        row for row in rollout_rows
        if row["family"] == "EDMD"
        and row["horizon_steps"] == longest_horizon
        and np.isfinite(float(row["normalized_state_rmse"]))
    ]

    best_long = (
        min(edmd_long, key=lambda row: row["normalized_state_rmse"])
        if edmd_long else None
    )

    summary: dict[str, Any] = {
        "dataset": {
            "trajectory_count": int(dataset["rotations"].shape[0]),
            "training_trajectory_count": int(len(dataset["training_indices"])),
            "validation_trajectory_count": int(len(dataset["validation_indices"])),
            "training_transition_count": int(len(training)),
            "validation_transition_count": int(len(validation)),
            "sample_period_s": float(dataset["sample_period_s"]),
            "source_dataset_seed": int(dataset["seed"]),
        },
        "method": {
            "DMDc_reference": (
                "Proctor, Brunton and Kutz, SIAM Journal on Applied Dynamical "
                "Systems, 2016, DOI 10.1137/15M1013857"
            ),
            "DMDc_model": (
                "Unlifted controlled least-squares model "
                "x[k+1] = A x[k] + B u[k], with x=[roll,pitch,yaw,body rates]."
            ),
            "EDMD_model": (
                "z[k+1] = A z[k] + B u[k] using the project's "
                "rotation-matrix lifted observables."
            ),
            "common_metrics": (
                "One-step angle RMSE, body-rate RMSE and normalized "
                "physical-state RMSE, evaluated on the same held-out trajectories."
            ),
            "normalized_state_rmse": (
                "Per-channel RMSE divided by validation-set standard deviation "
                "for that channel, then root-mean-squared across six channels."
            ),
            "rollout_method": (
                "Open-loop multi-step prediction using recorded validation "
                "torques; state estimates are not reset within a rollout window."
            ),
            "rollout_start_spacing": (
                f"Non-overlapping windows with stride {max(horizons)} samples."
            ),
            "model_specific_metric": (
                "H-projected RMSE is computed for EDMD models only; it is not "
                "applicable to the six-state DMDc representation."
            ),
        },
        "model_selection": {
            "candidate_orders": list(orders),
            "best_edmd_by_one_step_normalized_state_rmse": best_one["model"],
            "best_edmd_by_longest_rollout_normalized_state_rmse": (
                best_long["model"] if best_long else None
            ),
            "longest_rollout_horizon_steps": longest_horizon,
            "dmdc_one_step_normalized_state_rmse": dmdc_error,
        },
        "fit_results": fit_rows,
        "rollout_results": rollout_rows,
        "outputs": [
            "tables/dmdc_model_benchmark.csv",
            "tables/model_rollout_benchmark.csv",
            "figures/dmdc_model_benchmark.png",
            "figures/model_rollout_benchmark.png",
            "models/dmdc_edmd_benchmark.npz",
        ],
    }

    with (results_dir / "dmdc_benchmark_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(_json_safe(summary), stream, indent=2, allow_nan=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare EDMD lifting orders against a DMDc baseline "
            "using the existing dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/data/attitude_dataset.npz"),
        help="Path to the existing common dataset.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory for extension output files.",
    )

    args = parser.parse_args()
    dataset = load_common_dataset(args.dataset)
    summary = run_benchmark(dataset, args.results_dir)

    print("DMDc/model-order benchmark completed.")
    print(
        f"Trajectories: {summary['dataset']['trajectory_count']} "
        f"(train={summary['dataset']['training_trajectory_count']}, "
        f"validation={summary['dataset']['validation_trajectory_count']})"
    )
    print(
        "Best EDMD by one-step physical-state metric: "
        f"{summary['model_selection']['best_edmd_by_one_step_normalized_state_rmse']}"
    )
    print(
        "Best EDMD by longest rollout metric: "
        f"{summary['model_selection']['best_edmd_by_longest_rollout_normalized_state_rmse']}"
    )
    print(f"Results: {args.results_dir.resolve()}")

    for name in summary["outputs"]:
        print(f"  {args.results_dir / name}")


if __name__ == "__main__":
    main()
