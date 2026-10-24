from __future__ import annotations

import numpy as np

from src.attitude_core import attitude_lift
from src.attitude_study import StudyConfig, generate_dataset
from src.identification import AttitudeTransitions, fit_koopman_edmd
from src.trajectory_noise_sensitivity import (
    NoiseLevel,
    _training_state_scales,
    evaluate_multistep_rollouts,
    inject_training_measurement_noise,
    run_trajectory_noise_sensitivity,
    _plot_summary,
)


def _small_dataset(seed: int = 2701):
    return generate_dataset(
        StudyConfig(
            trajectory_count=4,
            training_trajectory_count=3,
            duration_s=5.0,
            sample_period_s=0.05,
            integration_substeps=2,
            seed=seed,
            control_duration_s=0.5,
            mpc_horizon=3,
        )
    )


def test_noise_is_reproducible_preserves_source_and_valid_rotations():
    data = _small_dataset()
    clean_rotations = data["rotations"].copy()
    clean_rates = data["angular_rates"].copy()
    level = NoiseLevel("test", 0.15, 0.008, 0.0002)

    first = inject_training_measurement_noise(data, level, np.random.default_rng(17))
    second = inject_training_measurement_noise(data, level, np.random.default_rng(17))

    assert np.allclose(first.rotation_current, second.rotation_current)
    assert np.allclose(first.omega_current, second.omega_current)
    assert np.allclose(first.torque, second.torque)
    assert np.array_equal(data["rotations"], clean_rotations)
    assert np.array_equal(data["angular_rates"], clean_rates)
    for rotation in first.rotation_current[:20]:
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)


def test_multistep_prediction_metrics_are_finite_on_clean_validation():
    data = _small_dataset(seed=2702)
    training_ids = np.asarray(data["training_indices"], dtype=int)
    model = fit_koopman_edmd(
        AttitudeTransitions.from_trajectories(
            data["rotations"][training_ids],
            data["angular_rates"][training_ids],
            data["torques"][training_ids],
        ),
        order=2,
    )
    metrics = evaluate_multistep_rollouts(
        model, data, horizons=(1, 3, 5), state_scales=_training_state_scales(data)
    )

    assert [row["horizon_steps"] for row in metrics] == [1, 3, 5]
    assert all(row["normalized_prediction_rmse"] is not None for row in metrics)
    assert all(np.isfinite(row["normalized_prediction_rmse"]) for row in metrics)
    assert all(row["valid_forecast_windows"] == row["expected_forecast_windows"] for row in metrics)


def test_noise_sensitivity_writes_summary_trials_and_one_figure(tmp_path):
    data = _small_dataset(seed=2703)
    result = run_trajectory_noise_sensitivity(
        data,
        tmp_path / "results",
        order=2,
        repetitions=2,
        seed=991,
        horizons=(1, 3),
        noise_levels=(NoiseLevel("test_low", 0.05, 0.003, 0.0001),),
    )

    out = tmp_path / "results" / "trajectory_noise_sensitivity"
    assert (out / "trajectory_noise_sensitivity_trials.csv").is_file()
    assert (out / "trajectory_noise_sensitivity_summary.csv").is_file()
    assert (out / "trajectory_noise_sensitivity_summary.json").is_file()
    assert (out / "trajectory_noise_sensitivity_metrics_log.png").is_file()
    assert len(result["summary_rows"]) == 4  # 2 horizons x (clean + test level)
    rows = [row for row in result["summary_rows"] if row["noise_level"] == "test_low"]
    assert len(rows) == 2
    assert all(row["successful_repetitions"] == 2 for row in rows)
    assert all(row["normalized_prediction_rmse_variance"] is not None for row in rows)


def test_summary_plot_uses_three_log_metric_panels(tmp_path):
    rows = []
    for noise, scale in (("clean_reference", 1.0), ("test_low", 1.5)):
        for horizon, base in ((1, 0.01), (5, 0.1)):
            rows.append({
                "noise_level": noise,
                "horizon_steps": horizon,
                "horizon_s": horizon * 0.05,
                "normalized_prediction_rmse_mean": base * scale,
                "normalized_prediction_rmse_std": base * 0.1,
                "attitude_rmse_deg_mean": base * 2 * scale,
                "attitude_rmse_deg_std": base * 0.2,
                "angular_rate_rmse_rad_s_mean": base * 0.01 * scale,
                "angular_rate_rmse_rad_s_std": base * 0.001,
            })
    plot_path = tmp_path / "metrics_log.png"
    scales = _plot_summary(
        rows, plot_path, (NoiseLevel("test_low", 0.05, 0.003, 0.0001),)
    )
    assert scales == ("log", "log", "log")
    assert plot_path.is_file() and plot_path.stat().st_size > 10_000
