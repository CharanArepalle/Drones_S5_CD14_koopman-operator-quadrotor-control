
"""Core rotation and rigid-body utilities for quadrotor attitude experiments."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def _as_vector3(value: Array, name: str) -> Array:
    out = np.asarray(value, dtype=float).reshape(-1)
    if out.shape != (3,):
        raise ValueError(
            f"{name} must contain exactly 3 values; got shape {out.shape}."
        )
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values.")
    return out


def skew(vector: Array) -> Array:
    """Return the 3x3 skew-symmetric matrix for a 3-vector."""
    x, y, z = _as_vector3(vector, "vector")
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=float)


def vee(matrix: Array) -> Array:
    """Map a skew-symmetric matrix to its associated 3-vector."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(
            f"matrix must have shape (3, 3); got {matrix.shape}."
        )
    return np.array([
        matrix[2, 1],
        matrix[0, 2],
        matrix[1, 0],
    ], dtype=float)


def project_rotation(matrix: Array) -> Array:
    """Project a 3x3 matrix onto SO(3) using singular-value decomposition."""
    matrix = np.asarray(matrix, dtype=float)

    if matrix.shape != (3, 3):
        raise ValueError(
            f"matrix must have shape (3, 3); got {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must contain only finite values.")

    u, _, vt = np.linalg.svd(matrix)

    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(u @ vt)

    return u @ correction @ vt


def rotation_vector_to_matrix(vector: Array) -> Array:
    """Convert an axis-angle rotation vector to a rotation matrix."""
    vector = _as_vector3(vector, "vector")
    angle = float(np.linalg.norm(vector))
    generator = skew(vector)

    if angle < 1e-8:
        return project_rotation(
            np.eye(3) + generator
            + 0.5 * (generator @ generator)
        )

    return (
        np.eye(3)
        + (np.sin(angle) / angle) * generator
        + (
            (1.0 - np.cos(angle)) / (angle * angle)
        ) * (generator @ generator)
    )


def rotation_matrix_to_vector(matrix: Array) -> Array:
    """Convert a rotation matrix to its axis-angle rotation vector."""
    rotation = project_rotation(matrix)

    cosine = float(
        np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    angle = float(np.arccos(cosine))

    if angle < 1e-8:
        return 0.5 * vee(rotation - rotation.T)

    if np.pi - angle < 1e-5:
        # The antisymmetric formula is ill-conditioned near pi.
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis_index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, axis_index])

        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-10:
            raise ValueError(
                "Could not determine the rotation axis near pi radians."
            )

        axis = axis / axis_norm
        anti = vee(rotation - rotation.T)

        if np.linalg.norm(anti) > 1e-8 and np.dot(axis, anti) < 0.0:
            axis = -axis

        return angle * axis

    return (
        angle / (2.0 * np.sin(angle))
    ) * vee(rotation - rotation.T)


def euler_zyx_to_rotation(
    roll: float,
    pitch: float,
    yaw: float,
) -> Array:
    """Convert roll, pitch and yaw angles in radians to a rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ])

    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ])

    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return rz @ ry @ rx


def rotation_to_euler_zyx(matrix: Array) -> Array:
    """Convert a rotation matrix to roll, pitch and yaw in radians."""
    r = project_rotation(matrix)
    sin_pitch = float(np.clip(-r[2, 0], -1.0, 1.0))
    pitch = float(np.arcsin(sin_pitch))

    if abs(np.cos(pitch)) > 1e-7:
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
    else:
        # At gimbal lock, yaw and roll are not independently identifiable.
        roll = 0.0
        yaw = float(np.arctan2(-r[0, 1], r[1, 1]))

    return np.array([roll, pitch, yaw], dtype=float)


def attitude_lift(
    rotation: Array,
    angular_velocity: Array,
    order: int = 5,
) -> Array:
    """Construct the rotation-matrix and angular-rate observable vector.

    Each matrix block is flattened in row-major order.
    The first block is R.T. Subsequent blocks are
    R @ skew(omega)**j for j = 1, ..., order - 1.
    """
    if not isinstance(order, (int, np.integer)) or order < 2:
        raise ValueError(
            "order must be an integer greater than or equal to 2."
        )

    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(
            f"rotation must have shape (3, 3); got {rotation.shape}."
        )

    rotation = project_rotation(rotation)
    angular_velocity = _as_vector3(
        angular_velocity, "angular_velocity"
    )

    omega_hat = skew(angular_velocity)
    blocks = [rotation.T.reshape(-1)]
    power = np.eye(3)

    for _ in range(1, order):
        power = power @ omega_hat
        blocks.append((rotation @ power).reshape(-1))

    result = np.concatenate(blocks)

    if result.shape != (9 * order,):
        raise RuntimeError("Unexpected lifted-state dimension.")

    return result


def decode_attitude_lift(
    lifted_state: Array,
) -> tuple[Array, Array]:
    """Recover rotation and body angular velocity from lifted values."""
    lifted_state = np.asarray(lifted_state, dtype=float).reshape(-1)

    if lifted_state.size < 18 or lifted_state.size % 9 != 0:
        raise ValueError(
            "lifted_state must contain 9*N values for integer N >= 2."
        )

    if not np.all(np.isfinite(lifted_state[:18])):
        raise ValueError("The first 18 lifted values must be finite.")

    raw_rotation_transpose = lifted_state[:9].reshape(3, 3)
    rotation = project_rotation(raw_rotation_transpose.T)

    rotation_omega = lifted_state[9:18].reshape(3, 3)
    omega_hat_raw = rotation.T @ rotation_omega
    omega_hat = 0.5 * (omega_hat_raw - omega_hat_raw.T)
    angular_velocity = vee(omega_hat)

    return rotation, angular_velocity


def attitude_derivative(
    rotation: Array,
    angular_velocity: Array,
    torque: Array,
    inertia: Array,
    disturbance: Array | None = None,
) -> tuple[Array, Array]:
    """Evaluate the rigid-body attitude dynamics."""
    rotation = np.asarray(rotation, dtype=float)

    if rotation.shape != (3, 3):
        raise ValueError(
            f"rotation must have shape (3, 3); got {rotation.shape}."
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must contain only finite values.")

    angular_velocity = _as_vector3(
        angular_velocity, "angular_velocity"
    )
    torque = _as_vector3(torque, "torque")

    inertia = np.asarray(inertia, dtype=float)

    if inertia.shape == (3,):
        inertia = np.diag(inertia)

    if inertia.shape != (3, 3):
        raise ValueError(
            "inertia must be a 3-vector of diagonal values or a 3x3 matrix."
        )

    if (
        not np.all(np.isfinite(inertia))
        or not np.allclose(inertia, inertia.T, atol=1e-10)
    ):
        raise ValueError("inertia must be finite and symmetric.")

    if np.min(np.linalg.eigvalsh(inertia)) <= 0.0:
        raise ValueError("inertia must be positive definite.")

    if disturbance is None:
        disturbance_vector = np.zeros(3, dtype=float)
    else:
        disturbance_vector = _as_vector3(
            disturbance, "disturbance"
        )

    rotation_dot = rotation @ skew(angular_velocity)
    angular_momentum = inertia @ angular_velocity

    angular_acceleration = np.linalg.solve(
        inertia,
        -skew(angular_velocity) @ angular_momentum
        + torque
        + disturbance_vector,
    )

    return rotation_dot, angular_acceleration


def integrate_attitude_rk4(
    rotation: Array,
    angular_velocity: Array,
    torque: Array,
    inertia: Array,
    dt: float,
    disturbance: Array | None = None,
) -> tuple[Array, Array]:
    """Integrate the rigid-body attitude equations for one RK4 time step."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be a positive finite number.")

    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3).")

    rotation = project_rotation(rotation)
    angular_velocity = _as_vector3(
        angular_velocity, "angular_velocity"
    )
    torque = _as_vector3(torque, "torque")

    inertia_array = np.asarray(inertia, dtype=float)

    if disturbance is None:
        disturbance_vector = np.zeros(3, dtype=float)
    else:
        disturbance_vector = _as_vector3(
            disturbance, "disturbance"
        )

    def derivative(r: Array, w: Array) -> tuple[Array, Array]:
        return attitude_derivative(
            r, w, torque, inertia_array, disturbance_vector
        )

    k1_r, k1_w = derivative(rotation, angular_velocity)

    k2_r, k2_w = derivative(
        rotation + 0.5 * dt * k1_r,
        angular_velocity + 0.5 * dt * k1_w,
    )

    k3_r, k3_w = derivative(
        rotation + 0.5 * dt * k2_r,
        angular_velocity + 0.5 * dt * k2_w,
    )

    k4_r, k4_w = derivative(
        rotation + dt * k3_r,
        angular_velocity + dt * k3_w,
    )

    next_rotation = rotation + (dt / 6.0) * (
        k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r
    )

    next_angular_velocity = angular_velocity + (dt / 6.0) * (
        k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w
    )

    return project_rotation(next_rotation), next_angular_velocity
