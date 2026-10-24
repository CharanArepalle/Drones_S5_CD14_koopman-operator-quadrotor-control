"""Bounded finite-horizon Koopman MPC for attitude regulation.

The controller uses a lifted-state deviation model centered at level attitude.
The nominal learned matrices A and B are reused without retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from src.attitude_core import attitude_lift

Array = np.ndarray


@dataclass
class MPCResult:
    torque: Array
    solve_time_ms: float
    success: bool
    iterations: int
    objective: float
    used_fallback: bool


class KoopmanMPC:
    """Receding-horizon MPC with box-constrained body-torque commands.

    The quadratic program is solved as a bound-constrained convex quadratic
    optimization using L-BFGS-B and an analytic gradient. The affine residual
    at the requested equilibrium is removed by expressing predictions in
    deviation coordinates. This equilibrium correction is used by the
    controller only; the model-identification metrics evaluate the unmodified
    learned predictor.
    """

    def __init__(
        self,
        model,
        torque_limits: Array,
        fallback_gain: Array,
        horizon: int = 10,
        q_rotation: float = 5.0,
        q_rate: float = 2.0,
        q_higher: float = 0.2,
        r_weights: tuple[float, float, float] = (0.08, 0.08, 0.05),
        max_iterations: int = 80,
    ) -> None:
        self.A = np.asarray(model.A, dtype=float)
        self.B = np.asarray(model.B, dtype=float)
        self.n = int(self.A.shape[0])
        self.m = int(self.B.shape[1])
        if self.A.shape != (self.n, self.n) or self.B.shape != (self.n, 3):
            raise ValueError("Expected model.A shape (n,n) and model.B shape (n,3).")
        if self.n < 18 or self.n % 9:
            raise ValueError("The model must use a 9*N attitude lift, N >= 2.")
        if horizon < 1:
            raise ValueError("horizon must be at least 1.")
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1.")

        self.order = self.n // 9
        self.horizon = int(horizon)
        self.max_iterations = int(max_iterations)
        self.torque_limits = np.asarray(torque_limits, dtype=float).reshape(3)
        self.fallback_gain = np.asarray(fallback_gain, dtype=float)
        if self.fallback_gain.shape != (3, self.n):
            raise ValueError("fallback_gain must have shape (3,n).")
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.B)):
            raise ValueError("Model matrices must be finite.")
        if not np.all(np.isfinite(self.torque_limits)) or np.any(self.torque_limits <= 0):
            raise ValueError("torque_limits must be positive and finite.")
        weights = (q_rotation, q_rate, q_higher, *r_weights)
        if any(not np.isfinite(w) or w < 0 for w in weights):
            raise ValueError("All cost weights must be finite and non-negative.")
        if not any(w > 0 for w in weights):
            raise ValueError("At least one cost weight must be positive.")

        self.z_equilibrium = attitude_lift(np.eye(3), np.zeros(3), self.order)
        self.q = np.concatenate([
            np.full(9, q_rotation),
            np.full(9, q_rate),
            np.full(self.n - 18, q_higher),
        ])
        self.r = np.asarray(r_weights, dtype=float)
        self._warm_start = np.zeros(self.horizon * self.m, dtype=float)
        self._build_prediction_matrices()

    def _build_prediction_matrices(self) -> None:
        """Build condensed linear predictions X = F*x0 + G*U once."""
        n, m, horizon = self.n, self.m, self.horizon
        powers = [np.eye(n)]
        for _ in range(horizon):
            powers.append(self.A @ powers[-1])

        self.F = np.vstack([powers[k] for k in range(1, horizon + 1)])
        self.G = np.zeros((horizon * n, horizon * m), dtype=float)
        for i in range(horizon):
            for j in range(i + 1):
                self.G[
                    i * n:(i + 1) * n,
                    j * m:(j + 1) * m,
                ] = powers[i - j] @ self.B

        self.q_bar = np.tile(self.q, horizon)
        self.r_bar = np.tile(self.r, horizon)
        self.hessian = self.G.T @ (self.q_bar[:, None] * self.G)
        self.hessian += np.diag(self.r_bar)
        self.lower_bounds = np.tile(-self.torque_limits, horizon)
        self.upper_bounds = np.tile(self.torque_limits, horizon)
        # Numerically symmetrize the positive-semidefinite Hessian.
        self.hessian = 0.5 * (self.hessian + self.hessian.T)

    def command(self, rotation: Array, omega: Array) -> MPCResult:
        """Compute the first torque command and report optimizer diagnostics."""
        start = perf_counter()
        z = attitude_lift(rotation, omega, self.order)
        deviation = z - self.z_equilibrium
        base = self.F @ deviation
        linear_term = self.G.T @ (self.q_bar * base)
        constant = float(base @ (self.q_bar * base))

        def objective(commands: Array) -> float:
            return float(
                commands @ self.hessian @ commands
                + 2.0 * linear_term @ commands
                + constant
            )

        def gradient(commands: Array) -> Array:
            return 2.0 * (self.hessian @ commands + linear_term)

        result = minimize(
            objective,
            self._warm_start,
            jac=gradient,
            method="L-BFGS-B",
            bounds=list(zip(self.lower_bounds, self.upper_bounds, strict=True)),
            options={"maxiter": self.max_iterations, "ftol": 1e-8, "maxls": 20},
        )

        success = bool(result.success and np.all(np.isfinite(result.x)))
        used_fallback = not success
        if success:
            sequence = np.asarray(result.x, dtype=float)
            self._warm_start = np.concatenate([
                sequence[self.m:], sequence[-self.m:]
            ])
            torque = sequence[:self.m]
            objective_value = float(result.fun)
            iterations = int(result.nit)
        else:
            # Use the already-computed LQR feedback if the numerical solver fails.
            torque = -self.fallback_gain @ deviation
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            objective_value = float(objective(self._warm_start))
            iterations = int(getattr(result, "nit", 0))

        torque = np.clip(torque, -self.torque_limits, self.torque_limits)
        return MPCResult(
            torque=torque,
            solve_time_ms=(perf_counter() - start) * 1000.0,
            success=success,
            iterations=iterations,
            objective=objective_value,
            used_fallback=used_fallback,
        )
