"""Robust implied-volatility inversion.

Implied volatility is a DIAGNOSTIC in this project.  It is never treated as
ground truth and, in particular, the implied volatility of a held-out premium
is never used as an input to predict that premium.

Method: bracket check against no-arbitrage bounds, then Brent root-finding on
C_BS(sigma) - C_market over a wide bracket, with a Brenner-Subrahmanyam style
initial refinement.  Prices violating static bounds return NaN with a status
flag instead of a spliced value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .analytical import bs_price, price_bounds

__all__ = ["implied_volatility", "ImpliedVolResult"]

_DEFAULT_LO = 1e-8
_DEFAULT_HI = 10.0


@dataclass
class ImpliedVolResult:
    sigma: Optional[float]
    status: str  # "ok" | "below_bounds" | "above_bounds" | "no_convergence" | "degenerate"
    iterations: int
    residual: Optional[float]


def implied_volatility(
    price,
    spot,
    strike,
    tau,
    rate,
    option_type: str = "call",
    sigma_lo: float = _DEFAULT_LO,
    sigma_hi: float = _DEFAULT_HI,
    xtol: float = 1e-12,
    max_iter: int = 200,
) -> ImpliedVolResult:
    """Invert C_BS(sigma) = price for a single contract."""
    price = float(price)
    tau = float(tau)
    if tau <= 1e-12:
        return ImpliedVolResult(None, "degenerate", 0, None)

    lo_bound, hi_bound = price_bounds(spot, strike, tau, rate, option_type=option_type)
    # Small numerical slack around static bounds.
    slack = 1e-10 + 1e-10 * max(hi_bound, 1.0)
    if price < lo_bound - slack:
        return ImpliedVolResult(None, "below_bounds", 0, None)
    if price > hi_bound + slack:
        return ImpliedVolResult(None, "above_bounds", 0, None)

    def f(sig: float) -> float:
        return float(bs_price(spot, strike, tau, rate, sig, option_type)) - price

    f_lo, f_hi = f(sigma_lo), f(sigma_hi)
    if f_lo > 0:  # price below even the tiny-vol limit (deep ITM forward intrinsic)
        return ImpliedVolResult(None, "below_bounds", 0, None)
    if f_hi < 0:
        return ImpliedVolResult(None, "above_bounds", 0, None)

    try:
        root = brentq(f, sigma_lo, sigma_hi, xtol=xtol, rtol=1e-14, maxiter=max_iter, full_output=True)
        sol, r = root[0], root[1]
        return ImpliedVolResult(float(sol), "ok", int(r.iterations), float(f(sol)))
    except (RuntimeError, ValueError):  # pragma: no cover - defensive
        return ImpliedVolResult(None, "no_convergence", max_iter, None)


def implied_volatility_array(prices, spots, strikes, taus, rates, option_type: str = "call", **kwargs):
    """Vectorised wrapper; returns (sigma array with NaN on failures, status array)."""
    n = len(np.atleast_1d(prices))
    sig = np.full(n, np.nan)
    status = np.empty(n, dtype=object)
    for i in range(n):
        res = implied_volatility(
            np.asarray(prices).reshape(-1)[i],
            np.asarray(spots).reshape(-1)[i],
            np.asarray(strikes).reshape(-1)[i],
            np.asarray(taus).reshape(-1)[i],
            np.asarray(rates).reshape(-1)[i],
            option_type,
            **kwargs,
        )
        sig[i] = res.sigma if res.sigma is not None else np.nan
        status[i] = res.status
    return sig, status
