import numpy as np

from src.attitude_core import euler_zyx_to_rotation, rotation_to_euler_zyx
from src.trajectory_tracking import (
    TrackingConfig,
    acceleration_to_attitude,
    integrate_quadrotor_rk4,
    quadrotor_derivative,
    trajectory_reference,
)


def test_reference_shapes_and_finiteness():
    cfg = TrackingConfig(duration_s=1.0, sample_period_s=0.05)
    t = np.arange(cfg.steps + 1) * cfg.sample_period_s
    ref = trajectory_reference(t, cfg)
    assert ref["position"].shape == (len(t), 3)
    assert ref["velocity"].shape == (len(t), 3)
    assert ref["acceleration"].shape == (len(t), 3)
    assert ref["yaw"].shape == (len(t),)
    assert all(np.all(np.isfinite(v)) for v in ref.values())


def test_hover_acceleration_maps_to_level_attitude_and_weight_thrust():
    mass = 1.2
    thrust, rotation = acceleration_to_attitude(np.zeros(3), 0.0, mass, 9.81)
    assert np.isclose(thrust, mass * 9.81)
    assert np.allclose(rotation, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)


def test_desired_body_z_axis_aligns_with_requested_force():
    acc = np.array([1.0, -0.5, 0.3])
    mass, gravity = 1.2, 9.81
    thrust, rotation = acceleration_to_attitude(acc, yaw=0.4, mass_kg=mass, gravity_m_s2=gravity)
    force = mass * (acc + np.array([0.0, 0.0, gravity]))
    assert np.allclose(rotation[:, 2], force / np.linalg.norm(force), atol=1e-8)
    assert np.isclose(thrust, np.linalg.norm(force))
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)


def test_hover_plant_is_equilibrium_under_weight_thrust():
    mass = 1.2
    inertia = np.array([0.021, 0.024, 0.039])
    state = np.concatenate([np.zeros(3), np.zeros(3), np.eye(3).reshape(9), np.zeros(3)])
    derivative = quadrotor_derivative(state, mass * 9.81, np.zeros(3), mass, inertia)
    assert np.allclose(derivative, 0.0, atol=1e-10)


def test_rk4_preserves_rotation_and_finite_state_with_disturbance():
    p = np.array([0.0, 0.0, 1.0])
    v = np.zeros(3)
    R = euler_zyx_to_rotation(0.05, -0.03, 0.2)
    omega = np.array([0.01, -0.02, 0.03])
    p1, v1, R1, w1 = integrate_quadrotor_rk4(
        p, v, R, omega,
        thrust_n=1.2 * 9.81,
        torque_nm=np.array([0.001, -0.001, 0.0005]),
        mass_kg=1.2,
        inertia=np.array([0.021, 0.024, 0.039]),
        dt=0.01,
        force_disturbance_n=np.array([0.05, -0.02, 0.01]),
        linear_drag_n_per_m_s=0.1,
    )
    assert all(np.all(np.isfinite(x)) for x in (p1, v1, R1, w1))
    assert np.allclose(R1.T @ R1, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(R1), 1.0, atol=1e-8)


def test_invalid_tracking_config_is_rejected():
    cfg = TrackingConfig(duration_s=1.0, sample_period_s=0.03)
    try:
        cfg.validate()
    except ValueError:
        pass
    else:
        raise AssertionError("A non-integral number of sample intervals was accepted.")
