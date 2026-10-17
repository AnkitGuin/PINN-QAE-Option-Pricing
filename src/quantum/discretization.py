"""Discretization of the risk-neutral terminal distribution.

Construction (fixed before evaluation, identical for every input parameter
set):

  * bounds [S_min, S_max] = [q_(alpha/2), q_(1 - alpha/2)] of the lognormal,
  * N = 2^n equally spaced grid points S_1..S_N on [S_min, S_max],
  * bin edges at grid midpoints, with outer edges at S_min and S_max,
  * p_j = F(edge_{j+1}) - F(edge_j) (exact in-range lognormal bin mass),
  * normalize in-range masses to one for QAE and record excluded tail mass.

The QAE distribution is the lognormal conditional on the retained quantile
interval. Excluding and renormalizing the tails contributes to the reported
discretization error against the full continuous Black-Scholes price.

This is a proper probability discretization of the SAME continuous law used by
the analytical and Monte Carlo evaluators; the induced pricing bias is the
discretization error V_exact_discrete - V_BS, reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .distribution import LognormalTerminalDistribution

__all__ = ["DiscretizedDistribution", "discretize"]


@dataclass
class DiscretizedDistribution:
    n_qubits: int
    grid: np.ndarray          # S_1..S_N (ascending)
    probabilities: np.ndarray # conditional in-range bin probabilities; sum to 1
    tail_mass: float
    s_min: float
    s_max: float
    truncation_quantile: float
    mean_continuous: float
    mean_discrete: float

    @property
    def n_points(self) -> int:
        return len(self.grid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_points": self.n_points,
            "s_min": self.s_min,
            "s_max": self.s_max,
            "tail_mass": self.tail_mass,
            "probabilities": self.probabilities.tolist(),
            "grid": self.grid.tolist(),
            "mean_continuous": self.mean_continuous,
            "mean_discrete": self.mean_discrete,
        }


def discretize(
    dist: LognormalTerminalDistribution,
    n_qubits: int,
    truncation_quantile: float = 0.001,
) -> DiscretizedDistribution:
    if not (0 < truncation_quantile < 0.5):
        raise ValueError("truncation_quantile must be in (0, 0.5)")
    n = int(n_qubits)
    if n < 1 or n > 6:
        raise ValueError("n_qubits must be in [1, 6] for simulator budgets")
    N = 2**n
    alpha = float(truncation_quantile)
    s_min = float(dist.ppf(alpha / 2.0))
    s_max = float(dist.ppf(1.0 - alpha / 2.0))
    grid = np.linspace(s_min, s_max, N)

    if N > 1:
        mids = 0.5 * (grid[1:] + grid[:-1])
        edges = np.concatenate([[s_min], mids, [s_max]])
    else:
        edges = np.array([s_min, s_max])
    cdf = dist.cdf(edges)
    probs = np.diff(cdf)
    probs = np.clip(probs, 0.0, None)
    in_range_mass = float(probs.sum())
    if not np.isfinite(in_range_mass) or in_range_mass <= 0.0:
        raise ValueError("distribution has no probability mass inside the truncation interval")
    tail_mass = float(max(0.0, 1.0 - in_range_mass))
    probs /= in_range_mass

    return DiscretizedDistribution(
        n_qubits=n,
        grid=grid,
        probabilities=probs,
        tail_mass=tail_mass,
        s_min=s_min,
        s_max=s_max,
        truncation_quantile=alpha,
        mean_continuous=float(dist.mean()),
        mean_discrete=float(np.sum(probs * grid)),
    )
