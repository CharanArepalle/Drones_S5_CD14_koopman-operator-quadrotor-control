import numpy as np
import pytest

from src.simulation_contract import (
    SimulationObservation,
    SimulationTrace,
    TorqueCommand,
    validate_rotation,
)
from src.quadrotor_mixer import CF2XMixer


def make_mixer():
    # Plausible test parameters only; production runs must read these from the
    # selected vehicle model and must not silently reuse these fixture values.
    return CF2XMixer(
        mass_kg=0.027,
        arm_length_m=0.0397,
        thrust_coefficient_n_per_rpm2=3.16e-10,
        yaw_moment_coefficient_nm_per_rpm2=7.94e-12,
        max_rpm=25000.0,
    )


def test_valid_identity_observation():
    obs = SimulationObservation(0.0, np.eye(3), np.zeros(3), np.array([0, 0, 1]), np.zeros(3))
    assert obs.time_s == 0.0
    assert np.allclose(obs.rotation_body_to_world, np.eye(3))


def test_observation_rejects_non_rotation():
    with pytest.raises(ValueError, match="orthonormal"):
        validate_rotation(np.diag([1.0, 1.0, 2.0]))


def test_observation_rejects_nonfinite_rate():
    with pytest.raises(ValueError, match="finite"):
        SimulationObservation(0.0, np.eye(3), [0, np.nan, 0], [0, 0, 1], [0, 0, 0])


def test_torque_command_validation():
    cmd = TorqueCommand(np.array([0.001, -0.001, 0.0002]), 0.3)
    assert cmd.body_torque_nm.shape == (3,)
    with pytest.raises(ValueError, match="non-negative"):
        TorqueCommand(np.zeros(3), -0.1)


def test_trace_accepts_consistent_state_and_interval_lengths():
    trace = SimulationTrace(
        backend="unit-test", controller="unit-test",
        time_s=np.array([0.0, 0.1]),
        rotations_body_to_world=np.array([np.eye(3), np.eye(3)]),
        body_rates_rad_s=np.zeros((2, 3)), position_world_m=np.zeros((2, 3)),
        velocity_world_m_s=np.zeros((2, 3)),
        requested_body_torque_nm=np.zeros((1, 3)), applied_body_torque_nm=np.zeros((1, 3)),
        motor_rpm=np.zeros((1, 4)), saturated=np.array([False]),
    )
    trace.validate()


def test_trace_rejects_nonmonotonic_time():
    trace = SimulationTrace(
        backend="unit-test", controller="unit-test", time_s=np.array([0.0, 0.0]),
        rotations_body_to_world=np.array([np.eye(3), np.eye(3)]), body_rates_rad_s=np.zeros((2, 3)),
        position_world_m=np.zeros((2, 3)), velocity_world_m_s=np.zeros((2, 3)),
        requested_body_torque_nm=np.zeros((1, 3)), applied_body_torque_nm=np.zeros((1, 3)),
        motor_rpm=np.zeros((1, 4)), saturated=np.array([False]),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        trace.validate()


def test_trace_rejects_off_by_one_command_shapes():
    trace = SimulationTrace(
        backend="unit-test", controller="unit-test", time_s=np.array([0.0, 0.1, 0.2]),
        rotations_body_to_world=np.array([np.eye(3)] * 3), body_rates_rad_s=np.zeros((3, 3)),
        position_world_m=np.zeros((3, 3)), velocity_world_m_s=np.zeros((3, 3)),
        requested_body_torque_nm=np.zeros((3, 3)), applied_body_torque_nm=np.zeros((3, 3)),
        motor_rpm=np.zeros((3, 4)), saturated=np.array([False, False, False]),
    )
    with pytest.raises(ValueError, match="must have shape"):
        trace.validate()


def test_mixer_hover_is_symmetric_and_unsaturated():
    mixer = make_mixer()
    output = mixer.allocate(np.zeros(3))
    assert not output.saturated
    assert output.saturation_scale == pytest.approx(1.0)
    assert np.allclose(output.motor_rpm, output.motor_rpm[0])
    assert output.achieved_collective_thrust_n == pytest.approx(mixer.mass_kg * mixer.gravity_m_s2)
    assert np.allclose(output.achieved_body_torque_nm, np.zeros(3), atol=1e-12)


def test_mixer_round_trip_torque_below_saturation():
    mixer = make_mixer()
    requested = np.array([2.0e-4, -1.5e-4, 2.0e-5])
    output = mixer.allocate(requested)
    assert not output.saturated
    assert np.allclose(output.achieved_body_torque_nm, requested, rtol=1e-8, atol=1e-10)
    implied = mixer.torque_from_motor_rpm(output.motor_rpm)
    assert np.allclose(implied[1:], requested, rtol=1e-8, atol=1e-10)


def test_mixer_saturation_is_explicit_and_motor_rpm_bounded():
    mixer = make_mixer()
    output = mixer.allocate(np.array([0.2, -0.2, 0.1]))
    assert output.saturated
    assert 0.0 <= output.saturation_scale < 1.0
    assert np.all(output.motor_rpm >= 0.0)
    assert np.all(output.motor_rpm <= mixer.max_rpm + 1e-8)
    assert output.achieved_collective_thrust_n == pytest.approx(mixer.mass_kg * mixer.gravity_m_s2)
    assert np.linalg.norm(output.achieved_body_torque_nm) < np.linalg.norm(output.requested_body_torque_nm)


def test_mixer_rejects_invalid_motor_parameters():
    with pytest.raises(ValueError, match="positive and finite"):
        CF2XMixer(
            mass_kg=0.027, arm_length_m=0.0397,
            thrust_coefficient_n_per_rpm2=0.0,
            yaw_moment_coefficient_nm_per_rpm2=7.94e-12,
            max_rpm=25000.0,
        )
