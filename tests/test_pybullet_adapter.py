import numpy as np
import pytest

from src.simulators.pybullet_adapter import (
    PyBulletAdapterConfig,
    map_model_equivalent_torque_to_physical,
    quaternion_xyzw_to_rotation,
    rotation_to_euler_zyx,
)


def test_identity_quaternion_produces_identity_rotation():
    assert np.allclose(quaternion_xyzw_to_rotation([0.0, 0.0, 0.0, 1.0]), np.eye(3))


def test_quaternion_conversion_matches_positive_roll_quarter_turn():
    half = np.sqrt(0.5)
    rotation = quaternion_xyzw_to_rotation([half, 0.0, 0.0, half])
    expected = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    assert np.allclose(rotation, expected, atol=1e-12)
    assert np.allclose(rotation_to_euler_zyx(rotation), [np.pi / 2, 0.0, 0.0], atol=1e-12)


def test_torque_mapping_preserves_input_induced_angular_acceleration_for_diagonal_inertia():
    j_model = np.array([0.021, 0.024, 0.039])
    j_sim = np.array([1.4e-5, 1.4e-5, 2.17e-5])
    tau_model = np.array([0.02, -0.01, 0.005])
    tau_physical = map_model_equivalent_torque_to_physical(tau_model, j_sim, j_model)
    alpha_model = tau_model / j_model
    alpha_sim = tau_physical / j_sim
    assert np.allclose(alpha_sim, alpha_model, rtol=1e-12, atol=1e-12)
    assert np.all(np.abs(tau_physical) < np.abs(tau_model))


@pytest.mark.parametrize(
    "bad_sim,bad_model",
    [([0.0, 1.0, 1.0], [1.0, 1.0, 1.0]), ([1.0, 1.0], [1.0, 1.0, 1.0]), ([1.0, np.nan, 1.0], [1.0, 1.0, 1.0])],
)
def test_torque_mapping_rejects_invalid_inertias(bad_sim, bad_model):
    with pytest.raises(ValueError):
        map_model_equivalent_torque_to_physical([0.1, 0.1, 0.1], bad_sim, bad_model)


def test_torque_mapping_rejects_bad_torque():
    with pytest.raises(ValueError, match="3 values"):
        map_model_equivalent_torque_to_physical([0.1, 0.2], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])


def test_config_enforces_integer_control_to_physics_ratio():
    cfg = PyBulletAdapterConfig(control_frequency_hz=25, physics_frequency_hz=240)
    with pytest.raises(ValueError, match="divisible"):
        cfg.validate()


def test_config_accepts_default_frequency_pair():
    PyBulletAdapterConfig().validate()


def _fake_gym_modules(monkeypatch, *, initial_world_angular_velocity=(0.0, 0.0, 0.0)):
    import sys
    import types

    class FakeEnum:
        CF2X = object()
        PYB = object()

    class FakeCtrlAviary:
        def __init__(self, **kwargs):
            self.M = 0.027
            self.L = 0.0397
            self.KF = 3.16e-10
            self.KM = 7.94e-12
            self.MAX_RPM = 25000.0
            self.G = 9.81
            self.J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
            xyzs = np.asarray(kwargs["initial_xyzs"], dtype=float)
            rpys = np.asarray(kwargs["initial_rpys"], dtype=float)
            self.pos = xyzs.copy()
            self.vel = np.zeros((1, 3), dtype=float)
            self.ang_v = np.asarray([initial_world_angular_velocity], dtype=float)
            # ZYX Euler input [roll,pitch,yaw] converted to quaternion xyzw.
            roll, pitch, yaw = rpys[0]
            cr, sr = np.cos(roll / 2), np.sin(roll / 2)
            cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            self.quat = np.array([[
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy,
            ]], dtype=float)
            self.last_action = np.zeros((1, 4), dtype=float)
            self.closed = False
            self.init_kwargs = kwargs

        def reset(self):
            return {"0": np.zeros(20, dtype=float)}, {}

        def step(self, action):
            self.last_action = np.asarray(action, dtype=float).copy()
            return {"0": np.zeros(20, dtype=float)}, 0.0, False, False, {}

        def close(self):
            self.closed = True

    root = types.ModuleType("gym_pybullet_drones")
    envs = types.ModuleType("gym_pybullet_drones.envs")
    ctrl = types.ModuleType("gym_pybullet_drones.envs.CtrlAviary")
    utils = types.ModuleType("gym_pybullet_drones.utils")
    enums = types.ModuleType("gym_pybullet_drones.utils.enums")
    ctrl.CtrlAviary = FakeCtrlAviary
    enums.DroneModel = FakeEnum
    enums.Physics = FakeEnum
    root.__path__ = []
    envs.__path__ = []
    utils.__path__ = []
    monkeypatch.setitem(sys.modules, "gym_pybullet_drones", root)
    monkeypatch.setitem(sys.modules, "gym_pybullet_drones.envs", envs)
    monkeypatch.setitem(sys.modules, "gym_pybullet_drones.envs.CtrlAviary", ctrl)
    monkeypatch.setitem(sys.modules, "gym_pybullet_drones.utils", utils)
    monkeypatch.setitem(sys.modules, "gym_pybullet_drones.utils.enums", enums)
    return FakeCtrlAviary


def test_adapter_reads_vehicle_parameters_maps_torque_and_steps(monkeypatch):
    from src.simulators.pybullet_adapter import PyBulletDronesAdapter

    _fake_gym_modules(monkeypatch)
    adapter = PyBulletDronesAdapter(PyBulletAdapterConfig())
    try:
        obs = adapter.reset(
            initial_position_world_m=[0.0, 0.0, 1.0],
            initial_euler_zyx_rad=[0.0, 0.0, 0.0],
        )
        assert adapter.mass_kg == pytest.approx(0.027)
        assert np.allclose(adapter.simulator_inertia_kg_m2, [1.4e-5, 1.4e-5, 2.17e-5])
        assert np.allclose(obs.rotation_body_to_world, np.eye(3))
        cmd = adapter.allocate_model_equivalent_torque([0.02, -0.01, 0.005])
        expected = map_model_equivalent_torque_to_physical(
            [0.02, -0.01, 0.005], [1.4e-5, 1.4e-5, 2.17e-5], [0.021, 0.024, 0.039]
        )
        assert np.allclose(cmd.requested_physical_torque_nm, expected)
        assert not cmd.mixer_output.saturated
        assert np.allclose(cmd.mixer_output.achieved_body_torque_nm, expected, atol=1e-10)
        report = adapter.compatibility_report()
        assert report["mixer_physics_mode"] == "PYB"
        assert report["all_single_axis_bounds_feasible"] is True
        obs2 = adapter.step(cmd.mixer_output.motor_rpm, duration_s=0.05)
        assert obs2.time_s == pytest.approx(0.05)
    finally:
        adapter.close()


def test_adapter_transforms_world_angular_velocity_to_body_frame(monkeypatch):
    from src.simulators.pybullet_adapter import PyBulletDronesAdapter

    _fake_gym_modules(monkeypatch, initial_world_angular_velocity=(0.0, 0.0, 1.0))
    adapter = PyBulletDronesAdapter(PyBulletAdapterConfig())
    try:
        obs = adapter.reset(
            initial_position_world_m=[0.0, 0.0, 1.0],
            initial_euler_zyx_rad=[np.pi / 2, 0.0, 0.0],
        )
        assert np.allclose(obs.body_rates_rad_s, [0.0, 1.0, 0.0], atol=1e-10)
    finally:
        adapter.close()


def _build_mixer(mode="PYB"):
    from src.quadrotor_mixer import CF2XMixer
    return CF2XMixer(
        mass_kg=0.027,
        arm_length_m=0.0397,
        thrust_coefficient_n_per_rpm2=3.16e-10,
        yaw_moment_coefficient_nm_per_rpm2=7.94e-12,
        max_rpm=25000.0,
        physics_mode=mode,
    )


def test_cf2x_native_pyb_positive_roll_command_uses_correct_motor_differential():
    mixer = _build_mixer("PYB")
    output = mixer.allocate(np.array([1.0e-5, 0.0, 0.0]))
    # Positive body roll requires motors 0/1 to reduce thrust and motors 2/3
    # to increase thrust for the observed CF2X Physics.PYB sign convention.
    rpm2 = output.motor_rpm ** 2
    assert np.mean(rpm2[:2]) < np.mean(rpm2[2:])
    # This checks the allocation equation only; measured response is checked
    # separately by the live smoke test.
    assert mixer.torque_from_motor_rpm(output.motor_rpm)[1] == pytest.approx(1.0e-5, abs=1e-10)


def test_cf2x_simplified_dynamics_positive_roll_has_same_sign_convention():
    mixer = _build_mixer("DYN")
    output = mixer.allocate(np.array([1.0e-5, 0.0, 0.0]))
    rpm2 = output.motor_rpm ** 2
    assert np.mean(rpm2[:2]) < np.mean(rpm2[2:])
    assert mixer.torque_from_motor_rpm(output.motor_rpm)[1] == pytest.approx(1.0e-5, abs=1e-10)


def test_axis_sign_diagnostics_pass_when_measured_rate_increments_match_torque():
    from src.pybullet_smoke_test import _axis_sign_diagnostics

    torque = np.array([
        [0.02, 0.0, 0.0], [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0], [-0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0], [0.0, 0.02, 0.0], [0.0, -0.02, 0.0], [0.0, -0.02, 0.0],
        [0.0, 0.0, 0.01], [0.0, 0.0, 0.01], [0.0, 0.0, -0.01], [0.0, 0.0, -0.01],
    ])
    alpha = np.array([
        [0.8, 0.0, 0.0], [0.9, 0.0, 0.0], [-0.8, 0.0, 0.0], [-0.9, 0.0, 0.0],
        [0.0, 0.7, 0.0], [0.0, 0.8, 0.0], [0.0, -0.7, 0.0], [0.0, -0.8, 0.0],
        [0.0, 0.0, 0.3], [0.0, 0.0, 0.4], [0.0, 0.0, -0.3], [0.0, 0.0, -0.4],
    ])
    report = _axis_sign_diagnostics(torque, alpha)
    assert report["all_axes_pass"] is True
    assert all(report[name]["positive_negative_signs_match"] for name in ("roll", "pitch", "yaw"))


def test_axis_sign_diagnostics_detects_the_roll_sign_failure_in_uploaded_trace():
    from src.pybullet_smoke_test import _axis_sign_diagnostics

    torque = np.array([
        [0.02, 0.0, 0.0], [-0.02, 0.0, 0.0],
        [0.0, 0.02, 0.0], [0.0, -0.02, 0.0],
        [0.0, 0.0, 0.01], [0.0, 0.0, -0.01],
    ])
    alpha = np.array([
        [-0.9, 0.0, 0.0], [0.9, 0.0, 0.0],  # wrong roll sign
        [0.0, 0.8, 0.0], [0.0, -0.8, 0.0],
        [0.0, 0.0, 0.3], [0.0, 0.0, -0.3],
    ])
    report = _axis_sign_diagnostics(torque, alpha)
    assert report["all_axes_pass"] is False
    assert report["roll"]["positive_negative_signs_match"] is False
    assert report["pitch"]["positive_negative_signs_match"] is True
    assert report["yaw"]["positive_negative_signs_match"] is True


def test_smoke_trace_uses_n_plus_one_state_samples_and_n_commands(monkeypatch, tmp_path):
    from src.pybullet_smoke_test import _state_row, _write_results
    from src.simulators.pybullet_adapter import PyBulletDronesAdapter

    _fake_gym_modules(monkeypatch)
    adapter = PyBulletDronesAdapter(PyBulletAdapterConfig())
    try:
        obs = adapter.reset(
            initial_position_world_m=[0.0, 0.0, 1.0],
            initial_euler_zyx_rad=[0.0, 0.0, 0.0],
        )
        rows = []
        for requested in (np.array([0.01, 0.0, 0.0]), np.array([-0.01, 0.0, 0.0])):
            command = adapter.allocate_model_equivalent_torque(requested)
            rows.append(_state_row(obs, command))
            obs = adapter.step(command.mixer_output.motor_rpm, duration_s=0.05)
        rows.append(_state_row(obs, None))
        summary = _write_results(rows, adapter.compatibility_report(), tmp_path)
        saved = np.load(tmp_path / "pybullet_smoke_trace.npz")
        assert saved["time_s"].shape == (3,)
        assert saved["command_time_s"].shape == (2,)
        assert saved["motor_rpm"].shape == (2, 4)
        assert saved["requested_physical_torque_nm"].shape == (2, 3)
        assert summary["state_sample_count"] == 3
        assert summary["control_interval_count"] == 2
        assert (tmp_path / "pybullet_smoke_trace.csv").is_file()
        assert (tmp_path / "pybullet_smoke_attitude.png").is_file()
    finally:
        adapter.close()
