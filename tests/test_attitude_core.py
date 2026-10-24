
import numpy as np

from src.attitude_core import (
    attitude_derivative,
    attitude_lift,
    decode_attitude_lift,
    euler_zyx_to_rotation,
    integrate_attitude_rk4,
    project_rotation,
    rotation_matrix_to_vector,
    rotation_to_euler_zyx,
    rotation_vector_to_matrix,
    skew,
    vee,
)


def test_skew_and_vee_round_trip():
    vector = np.array([0.3, -0.2, 0.7])

    assert np.allclose(vee(skew(vector)), vector)
    assert np.allclose(skew(vector) + skew(vector).T, 0.0)


def test_rotation_vector_round_trip():
    vector = np.array([0.12, -0.08, 0.17])
    rotation = rotation_vector_to_matrix(vector)
    recovered = rotation_matrix_to_vector(rotation)

    assert np.allclose(recovered, vector, atol=1e-8)
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)


def test_euler_round_trip_away_from_singularity():
    angles = np.array([0.2, -0.3, 0.4])
    rotation = euler_zyx_to_rotation(*angles)
    recovered = rotation_to_euler_zyx(rotation)

    assert np.allclose(recovered, angles, atol=1e-10)


def test_lift_dimension_and_decode():
    rotation = euler_zyx_to_rotation(0.12, -0.07, 0.2)
    omega = np.array([0.4, -0.25, 0.1])

    lifted = attitude_lift(rotation, omega, order=5)
    rotation_hat, omega_hat = decode_attitude_lift(lifted)

    assert lifted.shape == (45,)
    assert np.allclose(rotation_hat, rotation, atol=1e-10)
    assert np.allclose(omega_hat, omega, atol=1e-10)


def test_project_rotation_returns_valid_matrix():
    matrix = np.eye(3) + np.array([
        [0.0, 0.03, -0.01],
        [-0.02, 0.0, 0.04],
        [0.01, -0.03, 0.0],
    ])

    rotation = project_rotation(matrix)

    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10)


def test_zero_torque_rotational_derivative():
    inertia = np.array([0.02, 0.025, 0.04])
    rotation = np.eye(3)
    omega = np.zeros(3)

    rotation_dot, omega_dot = attitude_derivative(
        rotation,
        omega,
        np.zeros(3),
        inertia,
    )

    assert np.allclose(rotation_dot, np.zeros((3, 3)))
    assert np.allclose(omega_dot, np.zeros(3))


def test_rk4_keeps_rotation_on_so3():
    inertia = np.array([0.02, 0.025, 0.04])
    rotation = euler_zyx_to_rotation(0.05, -0.03, 0.1)
    omega = np.array([0.2, -0.1, 0.15])

    next_rotation, next_omega = integrate_attitude_rk4(
        rotation,
        omega,
        np.array([0.001, -0.002, 0.0015]),
        inertia,
        0.005,
    )

    assert np.all(np.isfinite(next_omega))
    assert np.allclose(
        next_rotation.T @ next_rotation,
        np.eye(3),
        atol=1e-10,
    )
    assert np.isclose(np.linalg.det(next_rotation), 1.0, atol=1e-10)
