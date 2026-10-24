"""Focused tests for the live controller runner; no live PyBullet GUI required."""
from __future__ import annotations

import csv
import importlib
import sys
from pathlib import Path
import types
from types import SimpleNamespace

import numpy as np
import pytest


def _rotation_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


def _euler_from_rotation(rotation):
    r = np.asarray(rotation, dtype=float)
    pitch = np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0]))
    roll = np.arctan2(r[2, 1], r[2, 2])
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return np.array([roll, pitch, yaw])


def _lift(rotation, omega, order):
    vector = np.zeros(9 * order, dtype=float)
    vector[:9] = np.asarray(rotation).reshape(-1)
    vector[9:12] = np.asarray(omega).reshape(3)
    return vector


# The user's project supplies these project modules. The focused staging tests
# inject only their documented small API surface so they can run without the
# physical simulator packages installed in this validation workspace.
original_core_module = sys.modules.get("src.attitude_core")
original_mpc_module = sys.modules.get("src.koopman_mpc")
core = types.ModuleType("src.attitude_core")
core.attitude_lift = _lift
core.rotation_to_euler_zyx = _euler_from_rotation
sys.modules["src.attitude_core"] = core

mpc_module = types.ModuleType("src.koopman_mpc")

class FakeKoopmanMPC:
    def __init__(self, model, torque_limits, fallback_gain, horizon=10, **kwargs):
        self.model = model
        self.torque_limits = np.asarray(torque_limits)
        self.fallback_gain = np.asarray(fallback_gain)
        self.horizon = horizon

    def command(self, rotation, omega):
        return SimpleNamespace(torque=-0.02 * _euler_from_rotation(rotation) - 0.01 * omega,
                               solve_time_ms=0.8, success=True, used_fallback=False, iterations=3)

mpc_module.KoopmanMPC = FakeKoopmanMPC
sys.modules["src.koopman_mpc"] = mpc_module

PROJECT_SRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_SRC_ROOT))
import src
src.__path__ = [str(PROJECT_SRC_ROOT / "src")]
demo = importlib.import_module("src.pybullet_controller_demo")
# Do not leave test stubs in sys.modules: other project tests must import the
# real attitude_core and KoopmanMPC implementation.
if original_core_module is not None:
    sys.modules["src.attitude_core"] = original_core_module
else:
    sys.modules.pop("src.attitude_core", None)
if original_mpc_module is not None:
    sys.modules["src.koopman_mpc"] = original_mpc_module
else:
    sys.modules.pop("src.koopman_mpc", None)


class FakeMixerOutput:
    def __init__(self, torque):
        self.motor_rpm = np.array([9000.0, 9100.0, 8900.0, 9050.0])
        self.allocator_implied_body_torque_nm = np.asarray(torque, dtype=float).copy()
        self.achieved_collective_thrust_n = 0.26487
        self.saturation_scale = 1.0
        self.saturated = False


class FakeMapped:
    def __init__(self, torque):
        self.requested_physical_torque_nm = np.asarray(torque, dtype=float).copy()
        self.collective_thrust_requested_n = 0.26487
        self.mixer_output = FakeMixerOutput(torque)


class FakeObservation:
    def __init__(self, time_s, rotation, omega, position=None, velocity=None):
        self.time_s = float(time_s)
        self.rotation_body_to_world = np.asarray(rotation, dtype=float).copy()
        self.body_rates_rad_s = np.asarray(omega, dtype=float).copy()
        self.position_world_m = np.asarray(position if position is not None else [0, 0, 1], dtype=float)
        self.velocity_world_m_s = np.asarray(velocity if velocity is not None else [0, 0, 0], dtype=float)


class FakeAdapter:
    name = "FakeCF2X"

    def __init__(self, dt=0.05):
        self.control_timestep_s = dt
        self.obs = None
        self.closed = False

    def reset(self, *, initial_position_world_m, initial_euler_zyx_rad):
        self.obs = FakeObservation(0.0, _rotation_from_euler(*initial_euler_zyx_rad), [0, 0, 0], initial_position_world_m)
        return self.obs

    def step_model_equivalent_torque(self, torque, *, duration_s, collective_thrust_n=None):
        angles = _euler_from_rotation(self.obs.rotation_body_to_world)
        torque = np.asarray(torque, dtype=float)
        omega = self.obs.body_rates_rad_s + torque * duration_s * 8.0
        next_angles = angles + omega * duration_s
        self.obs = FakeObservation(
            self.obs.time_s + duration_s,
            _rotation_from_euler(*next_angles),
            omega,
            self.obs.position_world_m,
            self.obs.velocity_world_m_s,
        )
        return self.obs, FakeMapped(torque)

    def close(self):
        self.closed = True


def _model():
    n = 45
    return SimpleNamespace(A=np.eye(n) * 0.8, B=np.zeros((n, 3)), order=5)


def _trace_for_metrics():
    dt = 0.05
    angles_deg = np.array([[4.0, -3.0, 2.0], [2.0, -1.5, 1.0], [0.5, -0.2, 0.1]])
    rotations = np.stack([_rotation_from_euler(*np.deg2rad(a)) for a in angles_deg])
    trace_lists = {
        "time_s": np.array([0.0, dt, 2*dt]),
        "rotation_body_to_world": rotations,
        "angles_rad": np.deg2rad(angles_deg),
        "angles_deg": angles_deg,
        "body_rates_rad_s": np.zeros((3, 3)),
        "position_world_m": np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float),
        "velocity_world_m_s": np.zeros((3, 3)),
        "geodesic_attitude_error_deg": np.array([5.0, 2.8, 0.55]),
        "position_target_world_m": np.repeat(np.array([[0.0, 0.0, 1.0]]), 3, axis=0),
        "position_error_world_m": np.zeros((3, 3)),
        "position_error_norm_m": np.zeros(3),
        "desired_attitude_deg": np.zeros((2, 3)),
        "position_hold_acceleration_command_m_s2": np.zeros((2, 3)),
        "requested_model_torque_nm": np.zeros((2, 3)),
        "limited_model_torque_nm": np.zeros((2, 3)),
        "requested_physical_torque_nm": np.zeros((2, 3)),
        "allocator_implied_physical_torque_nm": np.zeros((2, 3)),
        "motor_rpm": np.ones((2, 4)) * 9000,
        "requested_collective_thrust_n": np.ones(2) * 0.26487,
        "achieved_collective_thrust_n": np.ones(2) * 0.26487,
        "saturation_scale": np.ones(2),
        "saturated": np.zeros(2, dtype=bool),
        "mpc_solve_time_ms": np.zeros(2),
        "mpc_success": np.ones(2, dtype=bool),
        "mpc_fallback": np.zeros(2, dtype=bool),
        "mpc_iterations": np.zeros(2, dtype=int),
        "controller": "PID",
        "backend": "FakeCF2X",
        "stop_reason": "duration_complete",
        "control_period_s": dt,
        "wall_duration_s": 0.1,
    }
    return trace_lists


def test_geodesic_attitude_error_zero_for_identity_and_90_degrees_for_quarter_turn():
    assert demo.geodesic_attitude_error_deg(np.eye(3)) == pytest.approx(0.0, abs=1e-8)
    rot = _rotation_from_euler(np.pi / 2, 0, 0)
    assert demo.geodesic_attitude_error_deg(rot) == pytest.approx(90.0, abs=1e-7)


def test_pid_command_counteracts_positive_roll_error_and_respects_model_limit():
    cfg = demo.ControllerDemoConfig()
    rotation = _rotation_from_euler(np.deg2rad(10), 0, 0)
    requested, limited, diagnostics = demo.controller_model_torque(
        "PID", rotation, np.zeros(3), _model(), np.zeros((3, 45)), None, np.zeros(3), 0.05, cfg,
    )
    assert requested[0] < 0.0
    assert np.all(np.abs(limited) <= demo.MODEL_TORQUE_LIMITS_NM + 1e-12)
    assert diagnostics["pid_integral_next"][0] < 0.0


def test_lqr_controller_uses_saved_gain_on_lifted_state():
    cfg = demo.ControllerDemoConfig()
    gain = np.zeros((3, 45)); gain[0, 0] = 0.5; gain[1, 4] = 0.5
    rotation = _rotation_from_euler(0.1, -0.1, 0.0)
    requested, limited, _ = demo.controller_model_torque(
        "Koopman-LQR", rotation, np.zeros(3), _model(), gain, None, np.zeros(3), 0.05, cfg,
    )
    assert requested.shape == (3,)
    assert np.all(np.isfinite(requested))
    assert np.all(np.abs(limited) <= demo.MODEL_TORQUE_LIMITS_NM)


def test_mpc_diagnostics_are_captured_and_torque_is_limited():
    cfg = demo.ControllerDemoConfig()
    fake_mpc = FakeKoopmanMPC(_model(), demo.MODEL_TORQUE_LIMITS_NM, np.zeros((3, 45)))
    requested, limited, diagnostics = demo.controller_model_torque(
        "Koopman-MPC", _rotation_from_euler(0.1, 0.05, -0.04), [0.1, 0.2, -0.1],
        _model(), np.zeros((3, 45)), fake_mpc, np.zeros(3), 0.05, cfg,
    )
    assert requested.shape == (3,)
    assert diagnostics["mpc_success"] is True
    assert diagnostics["mpc_fallback"] is False
    assert diagnostics["mpc_solve_time_ms"] == pytest.approx(0.8)
    assert np.all(np.abs(limited) <= demo.MODEL_TORQUE_LIMITS_NM)


def test_saved_asset_loader_rejects_wrong_gain_shape(tmp_path):
    model_dir = tmp_path / "results" / "models"; model_dir.mkdir(parents=True)
    np.savez_compressed(model_dir / "attitude_models.npz", A_N5=np.eye(45), B_N5=np.zeros((45, 3)),
                        lqr_gain=np.zeros((3, 44)), selected_control_order=np.array(5))
    with pytest.raises(ValueError, match="LQR gain"):
        demo.load_controller_assets(tmp_path, 5)


def test_live_trace_state_and_command_arrays_must_be_n_plus_one_to_n():
    trace = _trace_for_metrics()
    demo.validate_live_trace(trace)
    trace["motor_rpm"] = np.ones((3, 4))
    with pytest.raises(ValueError, match="motor_rpm has shape"):
        demo.validate_live_trace(trace)


def test_metrics_report_settling_only_if_error_stays_inside_tolerance():
    cfg = demo.ControllerDemoConfig(settling_tolerance_deg=2.0)
    metrics = demo.calculate_live_metrics(_trace_for_metrics(), cfg)
    assert metrics["settling_time_to_tolerance_s"] == pytest.approx(0.1)
    assert metrics["state_samples"] == 3
    assert metrics["control_intervals"] == 2
    assert metrics["backend"] == "FakeCF2X"


def test_live_trial_runs_closed_loop_against_adapter_and_returns_aligned_trace(monkeypatch):
    cfg = demo.ControllerDemoConfig(duration_s=0.10, initial_attitude_deg=(2.0, 1.0, 0.5))
    adapter = FakeAdapter(dt=0.05)
    trace = demo.run_live_trial("PID", adapter, _model(), np.zeros((3, 45)), cfg, realtime=False, overlay=False)
    assert trace["time_s"].shape == (3,)
    assert trace["limited_model_torque_nm"].shape == (2, 3)
    assert trace["motor_rpm"].shape == (2, 4)
    assert np.all(np.diff(trace["time_s"]) > 0)
    demo.validate_live_trace(trace)


def test_saved_trace_csv_has_one_row_per_state_and_command_columns(tmp_path):
    cfg = demo.ControllerDemoConfig(settling_tolerance_deg=2.0)
    trace = _trace_for_metrics()
    metrics = demo.save_live_trial(trace, cfg, tmp_path)
    csv_path = tmp_path / metrics["trace_csv"]
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 4  # header + three state samples
    assert all(len(row) == len(rows[0]) for row in rows)
    assert (tmp_path / metrics["trace_npz"]).is_file()


def test_position_hold_at_target_requests_hover_and_level_attitude():
    cfg = demo.ControllerDemoConfig()
    desired_rotation, thrust, acceleration = demo.position_hold_command(
        [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        mass_kg=0.027, gravity_m_s2=9.81, config=cfg,
    )
    assert np.allclose(desired_rotation, np.eye(3), atol=1e-8)
    assert thrust == pytest.approx(0.027 * 9.81, rel=1e-8)
    assert np.allclose(acceleration, np.zeros(3), atol=1e-10)


def test_position_hold_desired_body_z_aligns_with_requested_force():
    cfg = demo.ControllerDemoConfig()
    desired_rotation, thrust, acceleration = demo.position_hold_command(
        [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.5, 0.0, 1.0],
        mass_kg=0.027, gravity_m_s2=9.81, config=cfg,
    )
    force_direction = (acceleration + np.array([0.0, 0.0, 9.81]))
    force_direction /= np.linalg.norm(force_direction)
    assert np.allclose(desired_rotation[:, 2], force_direction, atol=1e-8)
    assert np.linalg.det(desired_rotation) == pytest.approx(1.0, abs=1e-8)
    assert thrust > 0.027 * 9.81


def test_position_hold_lateral_acceleration_and_tilt_are_bounded():
    cfg = demo.ControllerDemoConfig()
    desired_rotation, _, acceleration = demo.position_hold_command(
        [0.0, 0.0, 1.0], [5.0, -5.0, 0.0], [2.0, 2.0, 1.0],
        mass_kg=0.027, gravity_m_s2=9.81, config=cfg,
    )
    assert np.linalg.norm(acceleration[:2]) <= cfg.max_lateral_acceleration_m_s2 + 1e-10
    tilt = np.arccos(np.clip(desired_rotation[2, 2], -1.0, 1.0))
    assert tilt <= np.deg2rad(cfg.max_tilt_deg) + 1e-8


def test_live_trace_contains_position_hold_arrays_and_metrics():
    cfg = demo.ControllerDemoConfig()
    trace = demo._trace_for_metrics() if hasattr(demo, "_trace_for_metrics") else None
    # Build a compact trace by using the established test fixture.
    trace = _trace_for_metrics()
    metrics = demo.calculate_live_metrics(trace, cfg)
    assert metrics["position_hold_enabled"] is False
    assert metrics["position_rmse_3d_m"] == pytest.approx(0.0)
    assert metrics["final_position_error_3d_m"] == pytest.approx(0.0)
