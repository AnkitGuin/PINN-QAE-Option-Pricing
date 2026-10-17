"""Analytical European Black-Scholes pricing.

Vectorised pricing for calls and puts.  The implementation is written in the
*total-variance* form so that deterministic time-varying coefficients are
handled exactly rather than by substituting an instantaneous volatility into a
constant-parameter formula:

    A(t,T) = integral_t^T sigma^2(u) du      (total variance)
    R(t,T) = integral_t^T r(u) du            (total discount)

    S_T = S_t exp( R - A/2 + sqrt(A) Z ),    Z ~ N(0,1)

    C = S N(d1) - K exp(-R) N(d2),
    d1 = (ln(S/K) + R + A/2) / sqrt(A),  d2 = d1 - sqrt(A).

For constant parameters A = sigma^2 * tau and R = r * tau, recovering the
classical formula.  Edge cases (zero maturity, zero variance) are handled
explicitly by their deterministic limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

__all__ = [
    "bs_price",
    "bs_call",
    "bs_put",
    "call_price_total_variance",
    "put_price_total_variance",
    "price_bounds",
    "put_call_parity_check",
    "forward_form_call",
]


def _ndtr(x) -> np.ndarray:
    """Standard normal CDF (scipy.special.ndtr is accurate in the tails)."""
    return ndtr(x)


def call_price_total_variance(spot, strike, total_variance, total_discount, option_type: str = "call"):
    """Price a European option given total variance ``A`` and discount ``R``.

    Parameters
    ----------
    spot, strike : positive scalars or arrays.
    total_variance : A >= 0 (equals sigma^2 * tau for constant parameters).
    total_discount : R (equals r * tau for constant parameters).
    option_type : "call" or "put".

    Returns a float for all-scalar inputs, else a broadcast ndarray.
    """
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    A0 = np.asarray(total_variance, dtype=float)
    R0 = np.asarray(total_discount, dtype=float)

    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(A0 < -1e-12):
        raise ValueError("total variance must be non-negative")
    if np.any(R0 < -0.5):
        raise ValueError("total discount unreasonably negative")

    scalar_input = all(np.ndim(x) == 0 for x in (spot, strike, total_variance, total_discount))
    S, K, A, R = np.broadcast_arrays(S, K, np.clip(A0, 0.0, None), R0)

    sqrtA = np.sqrt(A)
    safe_sqrtA = np.where(sqrtA > 0.0, sqrtA, 1.0)
    ln_m = np.log(S / K)
    d1 = (ln_m + R + 0.5 * A) / safe_sqrtA
    d2 = d1 - sqrtA
    disc = np.exp(-R)

    # Deterministic limit (A ~ 0): S_T = S exp(R) exactly.
    tiny = A <= 1e-14
    if option_type == "call":
        det = np.maximum(S - K * disc, 0.0)
        bs = S * _ndtr(d1) - K * disc * _ndtr(d2)
    elif option_type == "put":
        det = np.maximum(K * disc - S, 0.0)
        bs = K * disc * _ndtr(-d2) - S * _ndtr(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    out = np.where(tiny, det, bs)
    return float(out) if scalar_input else out


def put_price_total_variance(spot, strike, total_variance, total_discount):
    return call_price_total_variance(spot, strike, total_variance, total_discount, "put")


def bs_price(spot, strike, tau, rate, sigma, option_type: str = "call"):
    """Black-Scholes price with constant parameters.

    ``tau`` must be non-negative; at zero maturity the payoff is returned.
    ``sigma`` may be zero (deterministic limit returned).
    """
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    tau_raw = np.asarray(tau, dtype=float)
    if np.any(tau_raw < 0.0):
        raise ValueError("tau must be non-negative")
    tau_ = tau_raw
    r = np.asarray(rate, dtype=float)
    sig = np.asarray(sigma, dtype=float)

    if np.any(sig < -1e-12):
        raise ValueError("sigma must be non-negative")
    if np.any(r < -0.5):
        raise ValueError("rate unreasonably negative")

    return call_price_total_variance(S, K, sig**2 * tau_, r * tau_, option_type=option_type)


def bs_call(spot, strike, tau, rate, sigma):
    return bs_price(spot, strike, tau, rate, sigma, "call")


def bs_put(spot, strike, tau, rate, sigma):
    return bs_price(spot, strike, tau, rate, sigma, "put")


def forward_form_call(spot, strike, total_variance, total_discount):
    """Independent algebraic form used for cross-validation (Black-76 style).

    C = exp(-R) [ F N(d1) - K N(d2) ],  F = S exp(R),
    d1 = (ln(F/K) + A/2)/sqrt(A),  d2 = d1 - sqrt(A).
    """
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    A = np.clip(np.asarray(total_variance, dtype=float), 0.0, None)
    R = np.asarray(total_discount, dtype=float)
    F = S * np.exp(R)
    sqrtA = np.sqrt(A)
    safe = np.where(sqrtA > 0.0, sqrtA, 1.0)
    d1 = (np.log(F / K) + 0.5 * A) / safe
    d2 = d1 - sqrtA
    price = np.exp(-R) * (F * _ndtr(d1) - K * _ndtr(d2))
    deterministic = np.maximum(S - K * np.exp(-R), 0.0)
    return np.where(A <= 1e-14, deterministic, price)


def price_bounds(spot, strike, tau, rate, option_type: str = "call"):
    """No-arbitrage price bounds with no dividends (constant r).

    call: max(S - K e^{-r tau}, 0) <= C <= S
    put : max(K e^{-r tau} - S, 0) <= P <= K e^{-r tau}
    """
    S, K, r, tau_ = np.broadcast_arrays(
        np.asarray(spot, dtype=float), np.asarray(strike, dtype=float),
        np.asarray(rate, dtype=float), np.asarray(tau, dtype=float),
    )
    if np.any(tau_ < 0.0):
        raise ValueError("tau must be non-negative")
    disc = np.exp(-r * tau_)
    if option_type == "call":
        return np.maximum(S - K * disc, 0.0), S.copy()
    if option_type == "put":
        return np.maximum(K * disc - S, 0.0), np.broadcast_to(K * disc, disc.shape).copy()
    raise ValueError("option_type must be 'call' or 'put'")


def put_call_parity_check(spot, strike, tau, rate, call, put):
    """Return C - P - (S - K e^{-r tau}); exactly zero under the model."""
    S = np.asarray(spot, dtype=float)
    K = np.asarray(strike, dtype=float)
    r = np.asarray(rate, dtype=float)
    tau_ = np.clip(np.asarray(tau, dtype=float), 0.0, None)
    return np.asarray(call) - np.asarray(put) - (S - K * np.exp(-r * tau_))


@dataclass
class EuropeanContract:
    """Simple contract record used across the project."""

    spot: float
    strike: float
    tau: float
    rate: float
    option_type: str = "call"

    def price(self, sigma) -> float:
        return float(bs_price(self.spot, self.strike, self.tau, self.rate, sigma, self.option_type))
