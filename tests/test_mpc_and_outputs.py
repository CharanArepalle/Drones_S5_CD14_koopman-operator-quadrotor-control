import numpy as np

from src.attitude_core import euler_zyx_to_rotation
from src.identification import KoopmanModel
from src.koopman_mpc import KoopmanMPC


def make_model(order=5):
    n = 9 * order
    A = 0.96 * np.eye(n)
    B = np.zeros((n, 3), dtype=float)
    B[9:12, :] = 0.05 * np.eye(3)
    fallback_gain = np.zeros((3, n), dtype=float)
    fallback_gain[:, 9:12] = np.eye(3)
    model = KoopmanModel(
        A=A,
        B=B,
        order=order,
        design_rank=n + 3,
        design_condition=1.0,
    )
    return model, fallback_gain


def test_mpc_command_is_finite_and_within_limits():
    model, gain = make_model()
    limits = np.array([0.12, 0.12, 0.06])
    controller = KoopmanMPC(model, limits, gain, horizon=10)

    result = controller.command(
        euler_zyx_to_rotation(0.1, -0.08, 0.04),
        np.array([0.2, -0.1, 0.15]),
    )

    assert result.torque.shape == (3,)
    assert np.all(np.isfinite(result.torque))
    assert np.all(np.abs(result.torque) <= limits + 1e-10)
    assert result.solve_time_ms > 0.0


def test_mpc_warm_start_supports_repeated_control_updates():
    model, gain = make_model()
    controller = KoopmanMPC(model, np.array([0.12, 0.12, 0.06]), gain, horizon=6)
    rotation = np.eye(3)

    first = controller.command(rotation, np.array([0.1, 0.0, 0.0]))
    second = controller.command(rotation, np.array([0.05, 0.0, 0.0]))

    assert np.all(np.isfinite(first.torque))
    assert np.all(np.isfinite(second.torque))
    assert first.success and second.success


def test_mpc_rejects_invalid_limits():
    model, gain = make_model()
    try:
        KoopmanMPC(model, np.array([0.12, 0.0, 0.06]), gain)
    except ValueError:
        pass
    else:
        raise AssertionError("A zero torque limit should be rejected.")


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.attitude_study import _safe_savefig


def test_safe_figure_writer_can_overwrite_existing_output(tmp_path):
    path = tmp_path / "repeatable_plot.png"
    for value in (1.0, 2.0):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, value])
        returned = _safe_savefig(fig, path, dpi=60)
        plt.close(fig)
        assert returned.exists()
        assert returned.stat().st_size > 0
    assert path.exists()
    assert path.stat().st_size > 0
