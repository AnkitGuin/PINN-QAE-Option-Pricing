"""Risk-neutral terminal distribution used by ALL pricing paths.

For constant parameters (and, exactly, for deterministic time-varying
coefficients expressed as total variance A and total discount R):

    S_T = S_0 exp[ R - A/2 + sqrt(A) Z ],   Z ~ N(0,1)

i.e. lognormal with

    mu_ln = ln(S_0) + R - A/2,   sigma_ln = sqrt(A).

The analytical evaluator, Monte Carlo and the quantum module all consume this
SAME distribution; the quantum module additionally discretizes it (see
discretization.py), which introduces the separately-reported discretization
error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from scipy.stats import norm

__all__ = ["LognormalTerminalDistribution"]


@dataclass
class LognormalTerminalDistribution:
    spot: float
    total_variance: float  # A
    total_discount: float  # R

    def __post_init__(self) -> None:
        self.spot = float(self.spot)
        self.total_variance = float(self.total_variance)
        self.total_discount = float(self.total_discount)
        if not np.isfinite(self.spot) or self.spot <= 0:
            raise ValueError("spot must be finite and positive")
        if not np.isfinite(self.total_variance) or self.total_variance < 0:
            raise ValueError("total_variance must be finite and nonnegative")
        if not np.isfinite(self.total_discount):
            raise ValueError("total_discount must be finite")

    @classmethod
    def from_constant_parameters(cls, spot, tau, rate, sigma):
        return cls(spot=float(spot), total_variance=float(sigma) ** 2 * float(tau),
                   total_discount=float(rate) * float(tau))

    @property
    def mu_ln(self) -> float:
        return np.log(self.spot) + self.total_discount - 0.5 * self.total_variance

    @property
    def sigma_ln(self) -> float:
        return np.sqrt(self.total_variance)

    def ppf(self, q):
        q = np.asarray(q, dtype=float)
        if np.any((q < 0) | (q > 1)):
            raise ValueError("q must lie in [0, 1]")
        if self.total_variance == 0:
            return np.full_like(q, np.exp(self.mu_ln), dtype=float)
        return np.exp(self.mu_ln + self.sigma_ln * norm.ppf(q))

    def cdf(self, s):
        s = np.asarray(s, dtype=float)
        out = np.zeros_like(s)
        pos = s > 0
        if self.total_variance == 0:
            out[pos] = (s[pos] >= np.exp(self.mu_ln)).astype(float)
        else:
            out[pos] = norm.cdf((np.log(s[pos]) - self.mu_ln) / self.sigma_ln)
        return out

    def mean(self) -> float:
        return self.spot * np.exp(self.total_discount)

    def sample(self, n: int, seed=None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return self.spot * np.exp(
            self.total_discount - 0.5 * self.total_variance
            + self.sigma_ln * rng.standard_normal(n)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spot": self.spot,
            "total_variance": self.total_variance,
            "total_discount": self.total_discount,
            "mu_ln": self.mu_ln,
            "sigma_ln": self.sigma_ln,
            "mean": self.mean(),
        }
