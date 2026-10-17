"""Black-Scholes PDE utilities.

The PINN residual (torch, with automatic differentiation) lives in
``src/pinn/losses.py``.  This module provides:

* the PDE definition in words/constants shared across implementations,
* a finite-difference residual of an arbitrary vectorisable pricing function,
  used to cross-validate the automatic-differentiation convention (V_t = -V_tau
  and the normalization chain rule) independently of torch.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["pde_residual_fd"]


def pde_residual_fd(
    price_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    S: np.ndarray,
    tau: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    dS_factor: float = 1e-3,
    dtau: float = 1e-5,
) -> np.ndarray:
    """Finite-difference Black-Scholes PDE residual of ``price_fn``.

    The PDE in maturity time tau (tau = T - t, so V_t = -V_tau):

        f = -V_tau + 0.5 sigma^2 S^2 V_SS + r S V_S - r V

    ``price_fn(S_array, tau_scalar_or_array)`` must return prices for fixed
    contract parameters (K, r held fixed — they are conditioning inputs, not
    PDE variables).
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    dS = dS_factor * S
    v = price_fn(S, tau)
    v_up = price_fn(S + dS, tau)
    v_dn = price_fn(S - dS, tau)
    v_fwd = price_fn(S, tau + dtau)
    v_bwd = price_fn(S, np.maximum(tau - dtau, 0.0))

    v_s = (v_up - v_dn) / (2.0 * dS)
    v_ss = (v_up - 2.0 * v + v_dn) / (dS**2)
    v_tau = (v_fwd - v_bwd) / (dtau + np.where(tau - dtau >= 0, dtau, tau))

    return -v_tau + 0.5 * sigma**2 * S**2 * v_ss + r * S * v_s - r * v
