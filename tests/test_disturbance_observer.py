import numpy as np
import pytest

from src.attitude_core import euler_zyx_to_rotation, integrate_attitude_rk4
from src.disturbance_observer import LowPassTorqueDOB
from src.disturbance_study import disturbance_torque


INERTIA = np.array([0.021, 0.024, 0.039], dtype=float)


def test_observer_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        LowPassTorqueDOB(INERTIA, sample_period_s=0.0, cutoff_hz=1.5)
    with pytest.raises(ValueError):
        LowPassTorqueDOB(INERTIA, sample_period_s=0.01, cutoff_hz=50.0)
    with pytest.raises(ValueError):
        LowPassTorqueDOB(np.array([0.02, -0.01, 0.03]), 0.01, 1.0)


def test_observer_estimates_constant_matched_roll_torque():
    dt = 0.01
    injected = np.array([0.006, 0.0, 0.0])
    observer = LowPassTorqueDOB(INERTIA, dt, cutoff_hz=1.5)
    rotation = euler_zyx_to_rotation(0.0, 0.0, 0.0)
    omega = np.zeros(3)
    torque = np.zeros(3)
    estimate = np.zeros(3)

    for _ in range(500):
        estimate = observer.update(omega)
        observer.record_applied_torque(torque)
        rotation, omega = integrate_attitude_rk4(
            rotation, omega, torque, INERTIA, dt, disturbance=injected
        )

    assert abs(estimate[0] - injected[0]) < 5e-4
    assert np.max(np.abs(estimate[1:])) < 1e-6


def test_observer_has_near_zero_residual_without_disturbance():
    dt = 0.01
    observer = LowPassTorqueDOB(INERTIA, dt, cutoff_hz=1.5)
    rotation = euler_zyx_to_rotation(0.0, 0.0, 0.0)
    omega = np.zeros(3)
    max_estimate = 0.0

    for k in range(300):
        estimate = observer.update(omega)
        max_estimate = max(max_estimate, float(np.max(np.abs(estimate))))

        torque = np.array([
            0.003 * np.sin(0.1 * k),
            0.002 * np.cos(0.07 * k),
            0.001 * np.sin(0.05 * k),
        ])
        observer.record_applied_torque(torque)
        rotation, omega = integrate_attitude_rk4(
            rotation, omega, torque, INERTIA, dt
        )

    assert max_estimate < 1e-7


def test_disturbance_profile_is_finite_and_has_step_changes():
    times = np.linspace(0.0, 10.0, 101)
    values = np.stack([disturbance_torque(t) for t in times])

    assert values.shape == (101, 3)
    assert np.all(np.isfinite(values))
    # The step component causes visible change near t=1.5s.
    before = disturbance_torque(1.49)
    after = disturbance_torque(1.51)
    assert np.linalg.norm(after - before) > 0.01
