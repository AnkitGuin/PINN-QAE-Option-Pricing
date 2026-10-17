"""Payoff encoding for the quantum module.

The discounted payoff is encoded as the amplitude of a single "objective"
qubit:

    h_j = (S_j - K)^+ / P_max   in [0, 1],   P_max >= max_j (S_j - K)^+

Default P_max rule (fixed before evaluation): P_max = S_max_grid - K, so that
h_j <= 1 automatically.  The physical price is recovered as
V = exp(-R) * P_max * a with a the (estimated) expected h.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

__all__ = ["PayoffEncoding", "encode_call_payoff"]


@dataclass
class PayoffEncoding:
    strike: float
    p_max: float
    h: np.ndarray  # normalized payoffs, same length as the grid
    rule: str

    @property
    def max_payoff_grid(self) -> float:
        return float(np.max(self.h) * self.p_max)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strike": self.strike,
            "p_max": self.p_max,
            "h": self.h.tolist(),
            "rule": self.rule,
            "max_h": float(np.max(self.h)),
            "min_h": float(np.min(self.h)),
        }


def encode_call_payoff(grid: np.ndarray, strike: float, p_max: float | None = None) -> PayoffEncoding:
    grid = np.asarray(grid, dtype=float)
    raw = np.maximum(grid - strike, 0.0)
    if p_max is None:
        p_max = float(grid[-1] - strike)  # s_max_minus_k rule
    if p_max <= 0:
        raise ValueError("P_max must be positive (grid maximum must exceed the strike)")
    h = raw / p_max
    if np.any(h > 1.0 + 1e-12) or np.any(h < 0.0):
        raise ValueError("normalized payoff outside [0, 1]; increase P_max")
    return PayoffEncoding(
        strike=float(strike),
        p_max=float(p_max),
        h=h,
        rule="s_max_minus_k" if p_max == float(grid[-1] - strike) else "custom",
    )
