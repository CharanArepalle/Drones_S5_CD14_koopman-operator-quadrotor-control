from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from src.mujoco_controller_demo import (
    MuJoCoDemoConfig,
    calculate_primary_metrics,
    geodesic_rotation_error_deg,
    load_controller_assets,
)
from src.quadrotor_mixer import CF2XMixer
from src.simulators.mujoco_adapter import (
    MuJoCoAdapterConfig,
    MuJoCoQuadrotorAdapter,
    build_cf2x_mjcf,
    compute_body_wrench_from_rpm,
    euler_zyx_to_quaternion_wxyz,
    rotation_to_euler_zyx,
)


def _rotation_from_euler(angles):
    roll, pitch, yaw = np.asarray(angles, dtype=float)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def test_default_config_matches_shared_platform_contract():
    cfg = MuJoCoAdapterConfig()
    cfg.validate()
    assert cfg.physics_frequency_hz == 240
    assert cfg.control_frequency_hz == 20
    assert cfg.substeps_per_control == 12
    assert cfg.target_position_world_m == (0.0, 0.0, 1.0)
    assert cfg.max_rpm > 14_000


def test_mjcf_is_well_formed_and_contains_hover_and_floor_projection_rings():
    xml = build_cf2x_mjcf()
    root = ET.fromstring(xml)
    assert root.attrib["model"] == "CF2X Koopman controller comparison"
    geoms = {g.attrib.get("name"): g for g in root.findall(".//geom") if "name" in g.attrib}
    upper = [g for name, g in geoms.items() if name.startswith("hover_target_ring_")]
    lower = [g for name, g in geoms.items() if name.startswith("floor_projection_ring_")]
    assert len(upper) == 32
    assert len(lower) == 32
    for geom in upper:
        values = [float(v) for v in geom.attrib["fromto"].split()]
        assert values[2] == pytest.approx(1.0)
        assert values[5] == pytest.approx(1.0)
    for geom in lower:
        values = [float(v) for v in geom.attrib["fromto"].split()]
        assert values[2] == pytest.approx(0.018)
        assert values[5] == pytest.approx(0.018)
    assert "axis_x" in geoms and "axis_y" in geoms and "axis_z" in geoms
    assert root.find(".//freejoint[@name='quadrotor_free']") is not None


def test_hover_motor_commands_generate_exact_nominal_weight_and_zero_moment():
    cfg = MuJoCoAdapterConfig()
    hover_rpm = math.sqrt(cfg.mass_kg * cfg.gravity_m_s2 / (4 * cfg.thrust_coefficient_n_per_rpm2))
    force_body, torque_body = compute_body_wrench_from_rpm(np.full(4, hover_rpm), cfg)
    assert force_body == pytest.approx([0.0, 0.0, cfg.mass_kg * cfg.gravity_m_s2], rel=1e-9, abs=1e-12)
    assert torque_body == pytest.approx(np.zeros(3), abs=1e-12)


def test_mujo_coefficient_and_shared_mixer_have_same_wrench_ordering():
    cfg = MuJoCoAdapterConfig()
    mixer = CF2XMixer(
        mass_kg=cfg.mass_kg,
        arm_length_m=cfg.arm_length_m,
        thrust_coefficient_n_per_rpm2=cfg.thrust_coefficient_n_per_rpm2,
        yaw_moment_coefficient_nm_per_rpm2=cfg.yaw_moment_coefficient_nm_per_rpm2,
        max_rpm=cfg.max_rpm,
        gravity_m_s2=cfg.gravity_m_s2,
        physics_mode="DYN",
    )
    collective = cfg.mass_kg * cfg.gravity_m_s2
    for axis in range(3):
        torque = np.zeros(3)
        torque[axis] = 8e-6
        mixed = mixer.allocate(torque, collective)
        force_body, torque_body = compute_body_wrench_from_rpm(mixed.motor_rpm, cfg)
        assert force_body[2] == pytest.approx(mixed.achieved_collective_thrust_n, rel=1e-9)
        assert torque_body == pytest.approx(mixed.allocator_implied_body_torque_nm, rel=1e-7, abs=1e-11)
        assert torque_body[axis] > 0.0


def test_quaternion_is_scalar_first_unit_norm_and_roundtrips_small_attitude():
    angles = np.deg2rad([10.0, 8.0, 6.0])
    quat = euler_zyx_to_quaternion_wxyz(angles)
    assert quat.shape == (4,)
    assert np.linalg.norm(quat) == pytest.approx(1.0, abs=1e-12)
    w, x, y, z = quat
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    recovered = rotation_to_euler_zyx(rotation)
    assert np.rad2deg(recovered) == pytest.approx([10.0, 8.0, 6.0], abs=1e-8)
    assert geodesic_rotation_error_deg(rotation.T @ rotation) == pytest.approx(0.0, abs=1e-6)


def test_model_loader_checks_saved_order_and_shapes_without_refitting(tmp_path: Path):
    model_path = tmp_path / "results" / "models" / "attitude_models.npz"
    model_path.parent.mkdir(parents=True)
    n = 45
    np.savez_compressed(
        model_path,
        A_N5=np.eye(n), B_N5=np.zeros((n, 3)),
        lqr_gain=np.zeros((3, n)), selected_control_order=np.array(5),
    )
    model, gain, metadata = load_controller_assets(tmp_path, 5)
    assert model.A.shape == (45, 45)
    assert model.B.shape == (45, 3)
    assert gain.shape == (3, 45)
    assert metadata["selected_order"] == 5
    with pytest.raises(ValueError, match="missing|uses order|requested controller order"):
        load_controller_assets(tmp_path, 4)


def test_primary_metrics_are_computed_from_state_traces_only():
    config = MuJoCoDemoConfig(duration_s=0.1)
    times = np.array([0.0, 0.05, 0.1])
    desired = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
    actual = np.stack([
        _rotation_from_euler(np.deg2rad([10.0, 0.0, 0.0])),
        _rotation_from_euler(np.deg2rad([1.0, 0.0, 0.0])),
        np.eye(3),
    ])
    target = np.repeat(np.array([[0.0, 0.0, 1.0]]), 3, axis=0)
    pos = np.array([[0.1, 0.0, 1.0], [0.02, 0.0, 1.0], [0.0, 0.0, 1.0]])
    trace = {
        "controller": "PID", "trial_status": "duration_complete",
        "time_s": times, "rotation_body_to_world": actual,
        "desired_rotation_body_to_world": desired,
        "position_world_m": pos, "position_target_world_m": target,
    }
    metrics = calculate_primary_metrics(trace, config)
    assert metrics["attitude_tracking_rmse_deg"] == pytest.approx(np.sqrt((100 + 1 + 0) / 3), abs=1e-6)
    assert metrics["settling_time_s"] == pytest.approx(0.05)
    assert metrics["position_rmse_3d_m"] == pytest.approx(np.sqrt((.1**2 + .02**2) / 3), abs=1e-9)
    assert set(metrics) == {
        "controller", "attitude_tracking_rmse_deg", "settling_time_s", "position_rmse_3d_m", "trial_status"
    }


def test_mujoco_engine_model_compiles_and_hover_balances_if_package_is_installed():
    pytest.importorskip("mujoco")
    adapter = MuJoCoQuadrotorAdapter()
    try:
        assert adapter.model.nbody >= 2
        assert adapter.model.ngeom >= 70
        hover_rpm = math.sqrt(adapter.mass_kg * adapter.gravity_m_s2 / (4 * adapter.config.thrust_coefficient_n_per_rpm2))
        start = adapter.reset(initial_position_world_m=np.array([0, 0, 1.0]), initial_euler_zyx_rad=np.zeros(3))
        end = adapter.step(np.full(4, hover_rpm), duration_s=0.25)
        assert abs(end.position_world_m[2] - start.position_world_m[2]) < 0.02
        assert np.linalg.norm(end.body_rates_rad_s) < 0.05
    finally:
        adapter.close()
