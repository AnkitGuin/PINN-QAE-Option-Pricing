"""Constrained least-squares volatility calibration (primary comparator).

    sigma_LS = argmin_{sigma>0} sum_i [ C_BS(S_i, K_i, tau_i, r_i, sigma) - C_i ]^2

The optimisation sees EXACTLY the same observations, rates and window as the
inverse PINN (same degrees of freedom: one positive volatility).  Convergence
status, objective value, evaluation count and runtime are always recorded;
failures are reported, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from ..black_scholes.analytical import bs_price
from ..utils.timing import Timer

__all__ = ["LeastSquaresResult", "calibrate_least_squares"]


@dataclass
class LeastSquaresResult:
    sigma: Optional[float]
    objective: Optional[float]
    status: str  # "success" | "failed"
    reason: Optional[str] = None
    n_evaluations: int = 0
    runtime_seconds: float = 0.0
    init: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _objective(sigma: float, S, K, tau, r, prices, weights) -> float:
    model = bs_price(S, K, tau, r, sigma, "call")
    resid = model - prices
    return float(np.sum(weights * resid**2))


def calibrate_least_squares(
    spot,
    strike,
    tau,
    rate,
    prices,
    sigma_bounds=(0.01, 3.0),
    x_tolerance=1e-10,
    max_evaluations=200,
    init: Optional[float] = None,
    weights=None,
) -> LeastSquaresResult:
    """Fit a single positive volatility to observed call prices."""
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    t = np.asarray(tau, dtype=float)
    r = np.asarray(rate, dtype=float)
    p = np.asarray(prices, dtype=float)
    w = np.ones_like(p) if weights is None else np.asarray(weights, dtype=float)

    lo, hi = float(sigma_bounds[0]), float(sigma_bounds[1])
    init = float(init) if init is not None else 0.5 * (lo + hi)
    init = float(np.clip(init, lo, hi))

    timer = Timer().start()
    try:
        res = minimize_scalar(
            lambda s: _objective(s, S, K, t, r, p, w),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": x_tolerance, "maxiter": max_evaluations},
        )
        elapsed = timer.stop()
        sigma_hat = float(res.x)
        objective = float(res.fun)
        n_evaluations = int(getattr(res, "nfev", 0))
        # ``minimize_scalar(method='bounded')`` has no x0 argument. Preserve
        # the rolling estimate as an explicit candidate and keep the better
        # of that candidate and the bounded search result.
        if init is not None:
            init_objective = _objective(init, S, K, t, r, p, w)
            n_evaluations += 1
            if np.isfinite(init_objective) and init_objective < objective:
                sigma_hat, objective = init, init_objective
        ok = (
            bool(res.success)
            and np.isfinite(sigma_hat)
            and np.isfinite(objective)
            and lo <= sigma_hat <= hi
        )
        return LeastSquaresResult(
            sigma=sigma_hat if ok else None,
            objective=objective if ok else None,
            status="success" if ok else "failed",
            reason=None if ok else "optimiser returned invalid sigma",
            n_evaluations=n_evaluations,
            runtime_seconds=elapsed,
            init=init,
        )
    except Exception as exc:  # defensive: record, never hide
        elapsed = timer.stop()
        return LeastSquaresResult(
            sigma=None,
            objective=None,
            status="failed",
            reason=f"optimiser exception: {exc}",
            runtime_seconds=elapsed,
            init=init,
        )
