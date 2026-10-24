"""Repeated noise-sensitivity study for controlled lifted attitude models.

Uses the existing dataset and compares:
- clean-data least-squares EDMD reference,
- least-squares EDMD trained on noisy measurements,
- covariance-weighted TLS EDMD trained on the same noisy measurements.

The TLS estimator is the controlled-system adaptation implemented in
src.noise_identification; this module does not assert autonomous-system
convergence results for the controlled formulation.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.attitude_core import attitude_lift
from src.identification import (
    AttitudeTransitions,
    evaluate_koopman,
    fit_koopman_edmd,
)
from src.model_benchmark import load_common_dataset, transitions_for_split
from src.noise_identification import (
    NoiseModel,
    add_measurement_noise,
    fit_tls_edmd_controlled,
)


@dataclass
class NoiseScenario:
    name: str
    angle_std_deg: float
    rate_std_rad_s: float
    torque_reading_std_nm: float

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Noise scenario name cannot be empty.")
        for label, value in (
            ("angle_std_deg", self.angle_std_deg),
            ("rate_std_rad_s", self.rate_std_rad_s),
            ("torque_reading_std_nm", self.torque_reading_std_nm),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative.")
        if self.torque_reading_std_nm <= 0:
            raise ValueError(
                "torque_reading_std_nm must be positive for this full-rank "
                "covariance-weighted TLS formulation."
            )

    def estimator_noise(self) -> NoiseModel:
        self.validate()
        return NoiseModel(
            angle_std_deg=self.angle_std_deg,
            rate_std_rad_s=self.rate_std_rad_s,
            torque_measurement_std=self.torque_reading_std_nm,
        )


DEFAULT_SCENARIOS = (
    NoiseScenario("low", 0.05, 0.003, 0.0001),
    NoiseScenario("medium", 0.15, 0.008, 0.0002),
    NoiseScenario("high", 0.30, 0.015, 0.0005),
)


def _build_nonoverlap_training_transitions(
    data: dict[str, np.ndarray],
) -> AttitudeTransitions:
    """Build disjoint 1-step pairs to avoid reusing noisy snapshots.

    For each trajectory, pairs are (0,1), (2,3), (4,5), ... . Therefore no
    measured state is reused as both the target of one selected pair and the
    regressor of another. This better matches the independent-row noise
    approximation used by the covariance estimation.
    """
    ids = np.asarray(data["training_indices"], dtype=int)
    rotations = np.asarray(data["rotations"])[ids]
    rates = np.asarray(data["angular_rates"])[ids]
    torques = np.asarray(data["torques"])[ids]

    return AttitudeTransitions(
        rotation_current=rotations[:, :-1:2].reshape(-1, 3, 3),
        omega_current=rates[:, :-1:2].reshape(-1, 3),
        torque=torques[:, ::2].reshape(-1, 3),
        rotation_next=rotations[:, 1::2].reshape(-1, 3, 3),
        omega_next=rates[:, 1::2].reshape(-1, 3),
    )


def _h_projected_rmse(
    model: Any,
    validation: AttitudeTransitions,
) -> float:
    """RMSE on the first 18 lift coordinates, H=[I_18, 0]."""
    mse = []
    for R, w, u, Rn, wn in zip(
        validation.rotation_current,
        validation.omega_current,
        validation.torque,
        validation.rotation_next,
        validation.omega_next,
        strict=True,
    ):
        z_now = attitude_lift(R, w, model.order)
        z_prediction = model.A @ z_now + model.B @ u
        z_target = attitude_lift(Rn, wn, model.order)
        mse.append(float(np.mean((z_prediction[:18] - z_target[:18]) ** 2)))
    return float(np.sqrt(np.mean(mse)))


def _relative_operator_deviation(model: Any, reference: Any) -> float:
    current = np.hstack([model.A, model.B])
    baseline = np.hstack([reference.A, reference.B])
    denominator = max(float(np.linalg.norm(baseline, ord="fro")), 1e-12)
    return float(100.0 * np.linalg.norm(current - baseline, ord="fro") / denominator)


def _full_lifted_rmse(
    model: Any, validation: AttitudeTransitions
) -> float:
    errors = []
    for R, w, u, Rn, wn in zip(
        validation.rotation_current,
        validation.omega_current,
        validation.torque,
        validation.rotation_next,
        validation.omega_next,
        strict=True,
    ):
        z_prediction = model.predict_lifted(R, w, u)
        z_target = attitude_lift(Rn, wn, model.order)
        errors.append(float(np.mean((z_prediction - z_target) ** 2)))
    return float(np.sqrt(np.mean(errors)))


def _model_metrics(
    model: Any,
    validation: AttitudeTransitions,
    clean_reference: Any,
) -> dict[str, float]:
    metrics = evaluate_koopman(model, validation)
    return {
        "angle_rmse_deg": float(metrics["angle_rmse_deg"]),
        "body_rate_rmse_rad_s": float(
            metrics["angular_velocity_rmse_rad_s"]
        ),
        "normalized_state_rmse": float(metrics["normalized_state_rmse"]),
        "lifted_rmse": _full_lifted_rmse(model, validation),
        "h_projected_rmse": _h_projected_rmse(model, validation),
        "operator_deviation_from_clean_fit_pct": _relative_operator_deviation(
            model, clean_reference
        ),
    }


def _safe_float(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_float(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_float(v) for v in value]
    if isinstance(value, np.ndarray):
        return _safe_float(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "angle_rmse_deg",
        "body_rate_rmse_rad_s",
        "normalized_state_rmse",
        "lifted_rmse",
        "h_projected_rmse",
        "operator_deviation_from_clean_fit_pct",
        "fit_time_s",
    )
    group_keys = sorted({(r["noise_level"], r["estimator"]) for r in rows})
    output: list[dict[str, Any]] = []

    for level, estimator in group_keys:
        group = [
            r for r in rows
            if r["noise_level"] == level
            and r["estimator"] == estimator
            and r["status"] == "ok"
        ]
        record: dict[str, Any] = {
            "noise_level": level,
            "estimator": estimator,
            "successful_trials": len(group),
            "requested_trials": sum(
                1 for r in rows
                if r["noise_level"] == level and r["estimator"] == estimator
            ),
        }
        for metric in metrics:
            vals = np.asarray(
                [float(r[metric]) for r in group if r.get(metric) is not None],
                dtype=float,
            )
            vals = vals[np.isfinite(vals)]
            record[f"{metric}_mean"] = float(np.mean(vals)) if vals.size else None
            record[f"{metric}_std"] = (
                float(np.std(vals, ddof=1)) if vals.size > 1
                else (0.0 if vals.size == 1 else None)
            )
        output.append(record)
    return output


def _safe_plot_replace(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        fig.savefig(temp, dpi=180, bbox_inches="tight")
        try:
            temp.replace(path)
            return path
        except OSError:
            alternate = path.with_name(f"{path.stem}_new{path.suffix}")
            temp.replace(alternate)
            return alternate
    finally:
        plt.close(fig)
        if temp.exists():
            try:
                temp.unlink()
            except OSError:
                pass


def _plot_noise_metrics(
    summary_rows: list[dict[str, Any]],
    clean_reference: dict[str, float],
    levels: tuple[NoiseScenario, ...],
    path: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)
    x = np.arange(len(levels))
    names = [level.name for level in levels]

    for estimator, label in (
        ("LS-EDMD", "Least-squares EDMD"),
        ("TLS-EDMD", "Covariance-weighted TLS EDMD"),
    ):
        values = []
        errors = []
        for level in names:
            row = next(
                r for r in summary_rows
                if r["noise_level"] == level and r["estimator"] == estimator
            )
            value = row["normalized_state_rmse_mean"]
            std = row["normalized_state_rmse_std"]
            values.append(np.nan if value is None else float(value))
            errors.append(0.0 if std is None else float(std))
        axes[0].errorbar(x, values, yerr=errors, marker="o", capsize=4, label=label)

        values_angle = []
        errors_angle = []
        for level in names:
            row = next(
                r for r in summary_rows
                if r["noise_level"] == level and r["estimator"] == estimator
            )
            value = row["angle_rmse_deg_mean"]
            std = row["angle_rmse_deg_std"]
            values_angle.append(np.nan if value is None else float(value))
            errors_angle.append(0.0 if std is None else float(std))
        axes[1].errorbar(
            x, values_angle, yerr=errors_angle, marker="o", capsize=4, label=label
        )

    axes[0].axhline(
        clean_reference["normalized_state_rmse"],
        linestyle="--", label="Clean-data LS reference",
    )
    axes[1].axhline(
        clean_reference["angle_rmse_deg"],
        linestyle="--", label="Clean-data LS reference",
    )

    axes[0].set_ylabel("Normalized physical-state RMSE (lower is better)")
    axes[1].set_ylabel("Attitude RMSE (degrees, lower is better)")
    for ax in axes:
        ax.set_xticks(x, names)
        ax.set_xlabel("Measurement-noise condition")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Noisy-data identification: least squares versus covariance-weighted TLS")
    return _safe_plot_replace(fig, path)


def _plot_operator_deviation(
    summary_rows: list[dict[str, Any]],
    levels: tuple[NoiseScenario, ...],
    path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    x = np.arange(len(levels))
    for estimator, label in (
        ("LS-EDMD", "Least-squares EDMD"),
        ("TLS-EDMD", "Covariance-weighted TLS EDMD"),
    ):
        vals, errs = [], []
        for level in levels:
            row = next(
                r for r in summary_rows
                if r["noise_level"] == level.name and r["estimator"] == estimator
            )
            vals.append(float(row["operator_deviation_from_clean_fit_pct_mean"]))
            errs.append(float(row["operator_deviation_from_clean_fit_pct_std"] or 0.0))
        ax.errorbar(x, vals, yerr=errs, marker="o", capsize=4, label=label)

    ax.set_xticks(x, [level.name for level in levels])
    ax.set_xlabel("Measurement-noise condition")
    ax.set_ylabel("Operator deviation from clean-data fit (%)")
    ax.set_title("Sensitivity of fitted operators to measurement noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _safe_plot_replace(fig, path)


def run_noise_study(
    data: dict[str, np.ndarray],
    results_root: str | Path,
    *,
    order: int = 5,
    fit_sample_count: int = 360,
    repetitions: int = 3,
    mc_samples: int = 8,
    seed: int = 61018,
    scenarios: tuple[NoiseScenario, ...] = DEFAULT_SCENARIOS,
) -> dict[str, Any]:
    """Run repeated identification trials with identical clean sample pairs.

    The measured noise is applied only to the recorded training observations.
    The underlying simulated plant trajectories and validation data stay clean.
    Each estimator sees the same noisy training observations within a trial.
    """
    if order < 2:
        raise ValueError("order must be >= 2.")
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1.")
    if mc_samples < 3:
        raise ValueError("mc_samples must be >= 3.")
    if not scenarios:
        raise ValueError("At least one noise scenario is required.")
    for scenario in scenarios:
        scenario.validate()

    results_root = Path(results_root)
    output_dir = results_root / "noise_study"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_training_pairs = _build_nonoverlap_training_transitions(data)
    clean_validation = transitions_for_split(data, "validation")
    minimum_columns = 2 * (9 * order) + 3
    if fit_sample_count <= minimum_columns:
        raise ValueError(
            f"fit_sample_count must exceed {minimum_columns} for order {order}."
        )
    if fit_sample_count > len(all_training_pairs):
        raise ValueError(
            f"Requested {fit_sample_count} training pairs but only "
            f"{len(all_training_pairs)} non-overlapping pairs are available."
        )

    sample_rng = np.random.default_rng(seed + 101)
    selected_indices = np.sort(
        sample_rng.choice(len(all_training_pairs), size=fit_sample_count, replace=False)
    )
    clean_training_subset = AttitudeTransitions(
        all_training_pairs.rotation_current[selected_indices],
        all_training_pairs.omega_current[selected_indices],
        all_training_pairs.torque[selected_indices],
        all_training_pairs.rotation_next[selected_indices],
        all_training_pairs.omega_next[selected_indices],
    )

    clean_start = time.perf_counter()
    clean_model = fit_koopman_edmd(clean_training_subset, order=order)
    clean_fit_time = time.perf_counter() - clean_start
    clean_metrics = _model_metrics(clean_model, clean_validation, clean_model)
    clean_metrics["fit_time_s"] = clean_fit_time

    trial_rows: list[dict[str, Any]] = []
    saved_models: dict[str, np.ndarray] = {
        "clean_reference_A": clean_model.A,
        "clean_reference_B": clean_model.B,
    }

    for level_index, scenario in enumerate(scenarios):
        noise = scenario.estimator_noise()
        for repetition_index in range(repetitions):
            trial_seed = seed + 10000 * (level_index + 1) + repetition_index
            noisy_training = add_measurement_noise(
                clean_training_subset, noise=noise, seed=trial_seed
            )

            estimators: list[tuple[str, Any, float, dict[str, Any]]] = []

            start = time.perf_counter()
            ls_model = fit_koopman_edmd(noisy_training, order=order)
            ls_seconds = time.perf_counter() - start
            estimators.append(("LS-EDMD", ls_model, ls_seconds, {}))

            tls_diagnostics: dict[str, Any] = {}
            try:
                start = time.perf_counter()
                tls_model, tls_diagnostics = fit_tls_edmd_controlled(
                    noisy_training,
                    order=order,
                    noise=noise,
                    mc_samples=mc_samples,
                    seed=trial_seed + 1,
                )
                tls_seconds = time.perf_counter() - start
                estimators.append(("TLS-EDMD", tls_model, tls_seconds, tls_diagnostics))
            except Exception as exc:
                trial_rows.append({
                    "noise_level": scenario.name,
                    "estimator": "TLS-EDMD",
                    "repetition": repetition_index + 1,
                    "seed": trial_seed,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "angle_std_deg": scenario.angle_std_deg,
                    "rate_std_rad_s": scenario.rate_std_rad_s,
                    "torque_reading_std_nm": scenario.torque_reading_std_nm,
                    "fit_sample_count": fit_sample_count,
                    "validation_transition_count": len(clean_validation),
                    "order": order,
                    "mc_samples": mc_samples,
                })

            for estimator_name, model, fit_seconds, diagnostics in estimators:
                metrics = _model_metrics(model, clean_validation, clean_model)
                row: dict[str, Any] = {
                    "noise_level": scenario.name,
                    "estimator": estimator_name,
                    "repetition": repetition_index + 1,
                    "seed": trial_seed,
                    "status": "ok",
                    "error": "",
                    "angle_std_deg": scenario.angle_std_deg,
                    "rate_std_rad_s": scenario.rate_std_rad_s,
                    "torque_reading_std_nm": scenario.torque_reading_std_nm,
                    "fit_sample_count": fit_sample_count,
                    "validation_transition_count": len(clean_validation),
                    "order": order,
                    "mc_samples": mc_samples,
                    "fit_time_s": float(fit_seconds),
                    "design_rank": int(model.design_rank),
                    "design_condition_number": float(model.design_condition),
                    **metrics,
                }
                for key, value in diagnostics.items():
                    row[f"tls_{key}"] = value
                trial_rows.append(row)

                prefix = re.sub(r"[^A-Za-z0-9_]+", "_", scenario.name).strip("_")
                suffix = f"trial{repetition_index + 1}_{estimator_name.lower().replace('-', '_')}"
                saved_models[f"{prefix}_{suffix}_A"] = model.A
                saved_models[f"{prefix}_{suffix}_B"] = model.B

    # Add the clean reference as a standalone row so it is not confused with noisy trials.
    trial_rows.append({
        "noise_level": "clean_reference",
        "estimator": "Clean-LS-reference",
        "repetition": 0,
        "seed": seed,
        "status": "ok",
        "error": "",
        "angle_std_deg": 0.0,
        "rate_std_rad_s": 0.0,
        "torque_reading_std_nm": 0.0,
        "fit_sample_count": fit_sample_count,
        "validation_transition_count": len(clean_validation),
        "order": order,
        "mc_samples": 0,
        "design_rank": int(clean_model.design_rank),
        "design_condition_number": float(clean_model.design_condition),
        **clean_metrics,
    })

    summary_rows = _aggregate_rows(trial_rows)

    # Add the clean reference line to the aggregate table for each named condition.
    for scenario in scenarios:
        summary_rows.append({
            "noise_level": scenario.name,
            "estimator": "Clean-LS-reference",
            "successful_trials": 1,
            "requested_trials": 1,
            **{f"{metric}_mean": clean_metrics.get(metric) for metric in (
                "angle_rmse_deg", "body_rate_rmse_rad_s", "normalized_state_rmse",
                "lifted_rmse", "h_projected_rmse", "operator_deviation_from_clean_fit_pct",
                "fit_time_s",
            )},
            **{f"{metric}_std": 0.0 for metric in (
                "angle_rmse_deg", "body_rate_rmse_rad_s", "normalized_state_rmse",
                "lifted_rmse", "h_projected_rmse", "operator_deviation_from_clean_fit_pct",
                "fit_time_s",
            )},
        })

    _write_csv(output_dir / "noise_study_trials.csv", trial_rows)
    _write_csv(output_dir / "noise_study_summary.csv", summary_rows)
    np.savez_compressed(output_dir / "noise_study_models.npz", **saved_models)

    plot1 = _plot_noise_metrics(
        summary_rows, clean_metrics, scenarios,
        output_dir / "noise_metrics_vs_condition.png",
    )
    plot2 = _plot_operator_deviation(
        summary_rows, scenarios,
        output_dir / "operator_deviation_vs_condition.png",
    )

    # Summarize measured performance gains without presuming TLS is superior.
    gains: list[dict[str, Any]] = []
    for scenario in scenarios:
        ls = next(
            row for row in summary_rows
            if row["noise_level"] == scenario.name and row["estimator"] == "LS-EDMD"
        )
        tls = next(
            row for row in summary_rows
            if row["noise_level"] == scenario.name and row["estimator"] == "TLS-EDMD"
        )
        ls_value = ls["normalized_state_rmse_mean"]
        tls_value = tls["normalized_state_rmse_mean"]
        gains.append({
            "noise_level": scenario.name,
            "ls_normalized_state_rmse_mean": ls_value,
            "tls_normalized_state_rmse_mean": tls_value,
            "tls_improvement_over_ls_pct": (
                100.0 * (float(ls_value) - float(tls_value)) / max(float(ls_value), 1e-12)
                if ls_value is not None and tls_value is not None else None
            ),
        })

    summary: dict[str, Any] = {
        "experiment": {
            "order": order,
            "repetitions_per_noise_condition": repetitions,
            "monte_carlo_samples_for_covariance": mc_samples,
            "fit_sample_count": fit_sample_count,
            "validation_transition_count": len(clean_validation),
            "training_pairs_are_nonoverlapping": True,
            "random_seed_base": seed,
        },
        "method_and_assumptions": {
            "clean_reference": (
                "Least-squares EDMD fitted on a clean subset of the same selected training pairs. "
                "It is a reference fit, not the exact unknown Koopman operator."
            ),
            "noisy_estimators": (
                "LS-EDMD and covariance-weighted TLS EDMD are fitted on the same noisy observations "
                "within each trial and evaluated on the same clean held-out trajectories."
            ),
            "noise_construction": (
                "Independent perturbations are applied to Euler-angle readings, angular-rate readings, "
                "and recorded torque telemetry. The torque used to generate the original plant trajectory "
                "is unchanged."
            ),
            "independent_pairs": (
                "Every other transition is selected from each training trajectory, so selected pairs do not "
                "share a state measurement; this is intended to better respect the independent-row noise "
                "approximation used in the covariance calculation."
            ),
            "tls_scope": (
                "This is a covariance-weighted TLS adaptation to the controlled lifted regression. "
                "The autonomous-system theorem in the NREDMD study is not asserted as a guarantee for this adaptation."
            ),
            "primary_metrics": [
                "angle_rmse_deg", "body_rate_rmse_rad_s", "normalized_state_rmse",
                "h_projected_rmse", "operator_deviation_from_clean_fit_pct", "fit_time_s",
            ],
        },
        "noise_conditions": [asdict(scenario) for scenario in scenarios],
        "clean_reference_metrics": clean_metrics,
        "summary_results": summary_rows,
        "tls_relative_improvement": gains,
        "outputs": [
            "noise_study/noise_study_trials.csv",
            "noise_study/noise_study_summary.csv",
            "noise_study/noise_study_models.npz",
            str(plot1.relative_to(results_root)),
            str(plot2.relative_to(results_root)),
        ],
    }

    with (output_dir / "noise_study_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(_safe_float(summary), stream, indent=2, allow_nan=False)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare least-squares EDMD and covariance-weighted TLS under measured noise."
    )
    parser.add_argument(
        "--dataset", type=Path, default=Path("results/data/attitude_dataset.npz"),
        help="Existing dataset path; data are not regenerated.",
    )
    parser.add_argument(
        "--results-root", type=Path, default=Path("results"),
        help="Root directory for existing project results.",
    )
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--fit-samples", type=int, default=360)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mc-samples", type=int, default=8)
    args = parser.parse_args()

    dataset = load_common_dataset(args.dataset)
    summary = run_noise_study(
        dataset,
        args.results_root,
        order=args.order,
        fit_sample_count=args.fit_samples,
        repetitions=args.repetitions,
        mc_samples=args.mc_samples,
    )

    print("Noise-aware identification study completed.")
    print(f"Shared dataset: {args.dataset.resolve()}")
    print(f"Order: {args.order}; repeats per condition: {args.repetitions}")
    print(f"Results: {(args.results_root / 'noise_study').resolve()}")
    for output in summary["outputs"]:
        print(f"  {args.results_root / output}")


if __name__ == "__main__":
    main()
