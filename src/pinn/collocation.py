"""Collocation point sampling in homogeneous coordinates.

Interior points: x = ln(S/K) uniform over the window's (expanded) moneyness
box, tau in [tau_min, tau_max], r from the window range; S is drawn
log-uniformly over the window spot range and K = S * exp(-x) constructed from
it (the transformed PDE depends on (x, tau, r) only, but the loss API carries
physical (S, K) for boundary/terminal reconstruction).

Boundary points: deep-OTM (u = 0) and deep-ITM (u = 1 - e^{-r tau - x}) in
moneyness space.  Terminal points: tau = 0 with the payoff (the kink is
imposed HERE, never as an interior PDE point).  The terminal payoff may be
softplus-smoothed over a configurable width (documented regularization; the
induced price bias is quantified in tests/test_pinn_derivatives.py).

Training and diagnostic collocation sets use independent seeds so final
residuals are evaluated on points the optimiser never saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

__all__ = ["CollocationConfig", "WindowCollocation", "sample_window_collocation"]


@dataclass
class CollocationConfig:
    n_interior: int = 512
    n_boundary: int = 256
    n_terminal: int = 256
    tau_min: float = 0.05
    s_low_factor: float = 0.30    # S_lo = factor * min(window spot)
    s_high_factor: float = 2.40   # S_hi = factor * max(window spot); deep enough
                                   # that the parity-floor boundary error is
                                   # negligible (recorded per window)
    terminal_smoothing_alpha: float = 0.0  # 0 = exact kink; >0 softplus width in
                                           # price units

    @classmethod
    def from_dict(cls, d: Dict) -> "CollocationConfig":
        return cls(
            n_interior=int(d.get("n_interior", 512)),
            n_boundary=int(d.get("n_boundary", 256)),
            n_terminal=int(d.get("n_terminal", 256)),
            tau_min=float(d.get("tau_min", 0.05)),
            s_low_factor=float(d.get("s_low_factor", 0.30)),
            s_high_factor=float(d.get("s_high_factor", 2.40)),
            terminal_smoothing_alpha=float(d.get("terminal_smoothing_alpha", 0.0)),
        )


@dataclass
class WindowCollocation:
    interior_S: np.ndarray
    interior_tau: np.ndarray
    interior_K: np.ndarray
    interior_r: np.ndarray
    boundary_S: np.ndarray
    boundary_tau: np.ndarray
    boundary_K: np.ndarray
    boundary_r: np.ndarray
    boundary_target: np.ndarray  # physical price units
    terminal_S: np.ndarray
    terminal_K: np.ndarray
    terminal_r: np.ndarray
    terminal_payoff: np.ndarray  # physical price units
    x_low: float
    x_high: float
    boundary_upper_error_estimate: float  # max put/S gap at the upper boundary


def _softplus(x: np.ndarray, alpha: float) -> np.ndarray:
    z = np.clip(alpha * x, -60.0, 60.0)
    return np.where(alpha * x > 60.0, x, np.log1p(np.exp(z)) / alpha)


def sample_window_collocation(
    window: pd.DataFrame,
    cfg: CollocationConfig,
    seed: int,
) -> WindowCollocation:
    rng = np.random.default_rng(seed)

    S_min, S_max = float(window["spot"].min()), float(window["spot"].max())
    K_min, K_max = float(window["strike"].min()), float(window["strike"].max())
    r_min, r_max = float(window["rate"].min()), float(window["rate"].max())
    tau_max = max(float(window["tau"].max()), cfg.tau_min * 2.0)

    S_lo = cfg.s_low_factor * S_min
    S_hi = cfg.s_high_factor * S_max
    x_low = float(np.log(S_lo / K_max))   # deepest OTM corner
    x_high = float(np.log(S_hi / K_min))  # deepest ITM corner

    # Upper-boundary approximation error: the parity floor differs from the
    # true value by the put price at S_hi; estimate its worst case with the
    # smallest plausible vol over the largest maturity.
    _sig_floor = 0.08
    from ..black_scholes.analytical import bs_put

    put_est = float(
        bs_put(S_hi, K_min, tau_max, r_min, _sig_floor)
    )
    boundary_err = put_est / S_hi

    def _sample_S(n: int) -> np.ndarray:
        return np.exp(rng.uniform(np.log(S_min), np.log(S_max), n))

    def _sample_r(n: int) -> np.ndarray:
        if r_max - r_min < 1e-12:
            return np.full(n, r_min)
        return rng.uniform(r_min, r_max, n)

    # Interior
    n = cfg.n_interior
    x_f = rng.uniform(x_low, x_high, n)
    tau_f = rng.uniform(cfg.tau_min, tau_max, n)
    S_f = _sample_S(n)
    K_f = S_f * np.exp(-x_f)
    r_f = _sample_r(n)

    # Boundary: half deep-OTM (V=0), half deep-ITM (parity floor)
    n_b = cfg.n_boundary
    n_lo, n_hi = n_b // 2, n_b - n_b // 2
    x_b = np.concatenate([
        rng.uniform(x_low, x_low + 0.25 * (x_high - x_low), n_lo),
        rng.uniform(x_low + 0.75 * (x_high - x_low), x_high, n_hi),
    ])
    tau_b = rng.uniform(cfg.tau_min, tau_max, n_b)
    S_b = _sample_S(n_b)
    K_b = S_b * np.exp(-x_b)
    r_b = _sample_r(n_b)
    is_upper = np.concatenate([np.zeros(n_lo, dtype=bool), np.ones(n_hi, dtype=bool)])
    g_b = np.where(is_upper, S_b - K_b * np.exp(-r_b * tau_b), 0.0)

    # Terminal: tau = 0, payoff (optionally smoothed)
    n_t = cfg.n_terminal
    x_T = rng.uniform(x_low, x_high, n_t)
    S_T = _sample_S(n_t)
    K_T = S_T * np.exp(-x_T)
    r_T = _sample_r(n_t)
    if cfg.terminal_smoothing_alpha > 0:
        payoff_T = _softplus(S_T - K_T, cfg.terminal_smoothing_alpha)
    else:
        payoff_T = np.maximum(S_T - K_T, 0.0)

    return WindowCollocation(
        interior_S=S_f,
        interior_tau=tau_f,
        interior_K=K_f,
        interior_r=r_f,
        boundary_S=S_b,
        boundary_tau=tau_b,
        boundary_K=K_b,
        boundary_r=r_b,
        boundary_target=g_b,
        terminal_S=S_T,
        terminal_K=K_T,
        terminal_r=r_T,
        terminal_payoff=payoff_T,
        x_low=x_low,
        x_high=x_high,
        boundary_upper_error_estimate=boundary_err,
    )
