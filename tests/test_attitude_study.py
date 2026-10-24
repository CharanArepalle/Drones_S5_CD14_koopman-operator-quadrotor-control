
import numpy as np

from src.attitude_study import (
    StudyConfig,
    generate_dataset,
    transitions_for_split,
    validate_dataset,
)


def _small_config(seed=17):
    return StudyConfig(
        trajectory_count=4,
        training_trajectory_count=3,
        duration_s=0.2,
        sample_period_s=0.05,
        integration_substeps=2,
        seed=seed,
        control_duration_s=0.5,
    )


def test_dataset_uses_expected_shapes_and_split():
    data = generate_dataset(_small_config())
    validate_dataset(data)

    assert data["rotations"].shape == (4, 5, 3, 3)
    assert data["angular_rates"].shape == (4, 5, 3)
    assert data["torques"].shape == (4, 4, 3)
    assert len(data["training_indices"]) == 3
    assert len(data["validation_indices"]) == 1


def test_data_generation_is_repeatable_for_a_seed():
    first = generate_dataset(_small_config(seed=45))
    second = generate_dataset(_small_config(seed=45))

    assert np.allclose(first["rotations"], second["rotations"])
    assert np.allclose(first["torques"], second["torques"])
    assert np.array_equal(
        first["validation_indices"], second["validation_indices"]
    )


def test_transition_splits_do_not_cross_trajectory_boundaries():
    data = generate_dataset(_small_config())
    train = transitions_for_split(data, "training")
    validation = transitions_for_split(data, "validation")

    assert len(train) == 3 * 4
    assert len(validation) == 1 * 4


def test_invalid_split_is_rejected():
    data = generate_dataset(_small_config())

    try:
        transitions_for_split(data, "test")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid split name was accepted.")
