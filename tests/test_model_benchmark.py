
import numpy as np

from src.attitude_study import StudyConfig, generate_dataset
from src.model_benchmark import (
    load_common_dataset,
    run_benchmark,
    transitions_for_split,
    validate_dataset,
)


def _small_dataset(seed=17):
    config = StudyConfig(
        trajectory_count=4,
        training_trajectory_count=3,
        duration_s=1.5,
        sample_period_s=0.05,
        integration_substeps=2,
        seed=seed,
        control_duration_s=0.5,
        mpc_horizon=3,
    )
    return generate_dataset(config)


def test_dataset_validation_and_split_dimensions():
    data = _small_dataset()
    validate_dataset(data)

    train = transitions_for_split(data, "training")
    validation = transitions_for_split(data, "validation")

    assert len(train) == 3 * 30
    assert len(validation) == 1 * 30


def test_dataset_loader_rejects_missing_path(tmp_path):
    missing = tmp_path / "not_created.npz"

    try:
        load_common_dataset(missing)
    except FileNotFoundError as exc:
        assert "Run 'python -m src.attitude_study' first" in str(exc)
    else:
        raise AssertionError("Missing dataset was not rejected.")


def test_benchmark_outputs_edmd_and_dmdc_with_common_metrics(tmp_path):
    data = _small_dataset(seed=29)

    summary = run_benchmark(
        data,
        tmp_path / "results",
        orders=(2, 3),
        horizons=(1, 3, 5),
    )

    assert len(summary["fit_results"]) == 3
    assert {row["family"] for row in summary["fit_results"]} == {
        "EDMD", "DMDc"
    }
    assert all(
        np.isfinite(row["one_step_normalized_state_rmse"])
        for row in summary["fit_results"]
    )
    assert len(summary["rollout_results"]) == 9

    assert (
        tmp_path / "results" / "tables" / "dmdc_model_benchmark.csv"
    ).is_file()
    assert (
        tmp_path / "results" / "tables" / "model_rollout_benchmark.csv"
    ).is_file()
    assert (
        tmp_path / "results" / "figures" / "dmdc_model_benchmark.png"
    ).is_file()
    assert (
        tmp_path / "results" / "figures" / "model_rollout_benchmark.png"
    ).is_file()


def test_h_projected_metric_is_not_claimed_for_dmdc(tmp_path):
    data = _small_dataset(seed=31)

    summary = run_benchmark(
        data,
        tmp_path / "results",
        orders=(2,),
        horizons=(1, 3),
    )

    dmdc_row = next(
        row for row in summary["fit_results"] if row["family"] == "DMDc"
    )
    edmd_row = next(
        row for row in summary["fit_results"] if row["family"] == "EDMD"
    )

    assert dmdc_row["one_step_h_projected_rmse"] is None
    assert np.isfinite(edmd_row["one_step_h_projected_rmse"])
