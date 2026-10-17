"""Frozen-parameter baseline (analytical Black-Scholes with fixed inputs).

The frozen baseline isolates the value of *adaptation*: parameters estimated
once on the initial training period and never updated.  ``FrozenEngine`` in
``rolling.py`` implements the rolling-protocol version (LS fit on the first
window); this module provides the parameter handoff record used downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..black_scholes.analytical import bs_price
from .least_squares import calibrate_least_squares

__all__ = ["FrozenParameters", "fit_frozen_parameters"]


@dataclass
class FrozenParameters:
    sigma: Optional[float]
    status: str
    objective: Optional[float] = None
    runtime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def fit_frozen_parameters(frame, sigma_bounds=(0.01, 3.0)) -> FrozenParameters:
    """Fit sigma once on the provided (initial) frame."""
    result = calibrate_least_squares(
        frame["spot"].to_numpy(dtype=float),
        frame["strike"].to_numpy(dtype=float),
        frame["tau"].to_numpy(dtype=float),
        frame["rate"].to_numpy(dtype=float),
        frame["market_price"].to_numpy(dtype=float),
        sigma_bounds=sigma_bounds,
    )
    return FrozenParameters(
        sigma=result.sigma,
        status=result.status,
        objective=result.objective,
        runtime_seconds=result.runtime_seconds,
    )
