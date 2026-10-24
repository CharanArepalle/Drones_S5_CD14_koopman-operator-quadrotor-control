"""Filtered inverse-model torque disturbance observer.

The observer is based on the matched rotational disturbance in
J*w_dot = tau - cross(w, J*w) + d.

It estimates the unknown matched torque d from measured body rates and the
known applied torque. A first-order low-pass Q filter suppresses high-frequency
content in the inverse-model residual. This is a discrete-time, axis-coupled
rotational adaptation of the conventional disturbance-observer structure; it
is not a claim to implement every observer scheme in the cited tutorial.
"""
from __future__ import annotations

import numpy as np

Array = np.ndarray


class LowPassTorqueDOB:
    """First-order filtered inverse-dynamics observer for 3-axis torque."""

    def __init__(
        self,
        inertia: Array,
        sample_period_s: float,
        cutoff_hz: float = 1.5,
    ) -> None:
        self.inertia = np.asarray(inertia, dtype=float).reshape(3).copy()
        self.sample_period_s = float(sample_period_s)
        self.cutoff_hz = float(cutoff_hz)

        if not np.all(np.isfinite(self.inertia)) or np.any(self.inertia <= 0.0):
            raise ValueError("inertia must contain three finite positive values.")
        if not np.isfinite(self.sample_period_s) or self.sample_period_s <= 0.0:
            raise ValueError("sample_period_s must be finite and positive.")
        if not np.isfinite(self.cutoff_hz) or self.cutoff_hz <= 0.0:
            raise ValueError("cutoff_hz must be finite and positive.")
        if self.cutoff_hz >= 0.5 / self.sample_period_s:
            raise ValueError("cutoff_hz must remain below the Nyquist frequency.")

        self.alpha = float(
            np.exp(-2.0 * np.pi * self.cutoff_hz * self.sample_period_s)
        )
        self.reset()

    def reset(self) -> None:
        """Reset the observer for a new independent simulation."""
        self.previous_rate: Array | None = None
        self.previous_applied_torque = np.zeros(3, dtype=float)
        self.estimate = np.zeros(3, dtype=float)
        self.last_raw_residual = np.zeros(3, dtype=float)

    def update(self, measured_body_rate: Array) -> Array:
        """Update the disturbance estimate using the latest measured body rate.

        At time k>0, the residual uses the rate increment over [k-1,k] and
        the torque actually applied over that same interval:

            d_raw = J*(w[k]-w[k-1])/dt
                    + cross(w[k-1], J*w[k-1]) - tau[k-1]

        The filtered estimate is returned for the next control command.
        The first call initializes the observer and returns zero.
        """
        rate = np.asarray(measured_body_rate, dtype=float).reshape(3)
        if not np.all(np.isfinite(rate)):
            raise ValueError("measured_body_rate must contain finite values.")

        if self.previous_rate is None:
            self.previous_rate = rate.copy()
            return self.estimate.copy()

        previous_rate = self.previous_rate
        angular_momentum = self.inertia * previous_rate
        raw = (
            self.inertia * (rate - previous_rate) / self.sample_period_s
            + np.cross(previous_rate, angular_momentum)
            - self.previous_applied_torque
        )

        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("The inverse-dynamics residual is non-finite.")

        self.last_raw_residual = raw.copy()
        self.estimate = self.alpha * self.estimate + (1.0 - self.alpha) * raw
        self.previous_rate = rate.copy()
        return self.estimate.copy()

    def record_applied_torque(self, applied_torque: Array) -> None:
        """Record the saturated torque that was applied during the next plant step."""
        torque = np.asarray(applied_torque, dtype=float).reshape(3)
        if not np.all(np.isfinite(torque)):
            raise ValueError("applied_torque must contain finite values.")
        self.previous_applied_torque = torque.copy()
