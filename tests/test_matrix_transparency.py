import json
import numpy as np

from src.attitude_core import euler_zyx_to_rotation, integrate_attitude_rk4
from src.identification import AttitudeTransitions, fit_koopman_edmd
from src.attitude_study import _save_matrix_artifacts


def _make_transitions(n=240, seed=110):
    rng = np.random.default_rng(seed)
    inertia = np.array([0.021, 0.024, 0.039])
    rc, wc, u, rn, wn = [], [], [], [], []
    for _ in range(n):
        r = euler_zyx_to_rotation(*rng.uniform(-0.4, 0.4, size=3))
        w = rng.uniform(-0.8, 0.8, size=3)
        torque = rng.uniform(-0.04, 0.04, size=3)
        r_next, w_next = integrate_attitude_rk4(r, w, torque, inertia, 0.01)
        rc.append(r)
        wc.append(w)
        u.append(torque)
        rn.append(r_next)
        wn.append(w_next)
    return AttitudeTransitions(
        np.asarray(rc), np.asarray(wc), np.asarray(u),
        np.asarray(rn), np.asarray(wn),
    )


def test_matrix_report_exposes_g1_g2_k_a_b_and_lqr(tmp_path):
    transitions = _make_transitions()
    model = fit_koopman_edmd(transitions, order=5)
    n = model.A.shape[0]
    q = np.eye(n)
    r = np.eye(3)
    p = np.eye(n)
    omega_lqr = np.zeros((3, n))

    diagnostics = _save_matrix_artifacts(
        tmp_path / "results", transitions, model, q, r, p, omega_lqr
    )

    output = tmp_path / "results" / "matrices"
    expected = {
        "G1.csv", "G2.csv", "K_operator_moments.csv", "K_operator_lstsq.csv",
        "A_lstsq.csv", "B_lstsq.csv", "Q_lqr.csv", "R_lqr.csv",
        "P_riccati.csv", "Omega_lqr.csv", "attitude_matrices_N5.npz",
        "matrix_diagnostics.json",
    }
    assert expected.issubset({p.name for p in output.iterdir()})
    assert diagnostics["matrix_shapes"]["G1"] == [45, 48]
    assert diagnostics["matrix_shapes"]["G2"] == [48, 48]
    assert diagnostics["matrix_shapes"]["K_operator_moments"] == [45, 48]

    report = json.loads((output / "matrix_diagnostics.json").read_text())
    assert report["K_moment_vs_lstsq_relative_frobenius_difference"] < 1e-6

    with np.load(output / "attitude_matrices_N5.npz", allow_pickle=False) as archive:
        assert archive["A_lstsq"].shape == (45, 45)
        assert archive["B_lstsq"].shape == (45, 3)
        assert archive["Omega_lqr"].shape == (3, 45)
