"""PINN diagnostics: residual maps and independent-collocation evaluation.

A low training residual is not proof of PDE satisfaction: final residuals are
evaluated on freshly sampled points the optimiser never saw.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from ..black_scholes.analytical import bs_price
from ..utils.reproducibility import derive_seed
from .collocation import CollocationConfig, sample_window_collocation
from .losses import pde_residual
from .network import PINNModel

__all__ = ["evaluate_on_independent_collocation", "residual_map", "price_diagnostics"]


def evaluate_on_independent_collocation(
    model: PINNModel,
    frame: pd.DataFrame,
    cfg: CollocationConfig,
    seed: int,
) -> Dict[str, Any]:
    """PDE residual / boundary / terminal errors on an independent sample."""
    coll = sample_window_collocation(frame, cfg, seed=derive_seed(seed, "indep_collocation"))
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64)
    sigma = t(np.full(len(coll.interior_S), float(model.sigma().detach())))
    with torch.enable_grad():
        f_hat = pde_residual(
            model,
            t(coll.interior_S), t(coll.interior_tau), t(coll.interior_K), t(coll.interior_r),
            sigma,
        )
        resid_rms = float(torch.sqrt(torch.mean(f_hat**2)).detach())
        resid_max = float(torch.max(torch.abs(f_hat)).detach())
        u_b = model.u(t(coll.boundary_S), t(coll.boundary_tau), t(coll.boundary_K), t(coll.boundary_r))
        bc_err = float(torch.sqrt(torch.mean((u_b - t(coll.boundary_target) / t(coll.boundary_S)) ** 2)).detach())
        u_T = model.u(t(coll.terminal_S), torch.zeros_like(t(coll.terminal_S)), t(coll.terminal_K), t(coll.terminal_r))
        term_err = float(torch.sqrt(torch.mean((u_T - t(coll.terminal_payoff) / t(coll.terminal_S)) ** 2)).detach())
    return {
        "pde_residual_rms": resid_rms,
        "pde_residual_max": resid_max,
        "boundary_error": bc_err,
        "terminal_error": term_err,
        "n_interior": int(cfg.n_interior),
        "collocation_x_range": [coll.x_low, coll.x_high],
        "boundary_upper_error_estimate": coll.boundary_upper_error_estimate,
    }


def residual_map(
    model: PINNModel,
    frame: pd.DataFrame,
    cfg: CollocationConfig,
    seed: int,
    n_x: int = 40,
    n_tau: int = 30,
    reference_strike_moneyness: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Residual on a regular (x, tau) grid for plotting.

    K is set to reference_strike_moneyness * mean window spot; S = K * e^x.
    """
    scaler = model.scaler
    S_ref = float(frame["spot"].mean())
    K_ref = reference_strike_moneyness * S_ref
    coll = sample_window_collocation(frame, cfg, seed=derive_seed(seed, "map"))
    x_grid = np.linspace(coll.x_low, coll.x_high, n_x)
    tau_grid = np.linspace(cfg.tau_min, float(frame["tau"].max()), n_tau)
    xx, tt = np.meshgrid(x_grid, tau_grid, indexing="ij")
    S = (K_ref * np.exp(xx)).ravel()
    tau = tt.ravel()
    K = np.full_like(S, K_ref)
    r = np.full_like(S, float(frame["rate"].mean()))

    t = lambda a: torch.as_tensor(a, dtype=torch.float64)
    with torch.enable_grad():
        f_hat = pde_residual(model, t(S), t(tau), t(K), t(r), t(np.full_like(S, float(model.sigma().detach()))))
    return {
        "x": x_grid,
        "tau": tau_grid,
        "residual": f_hat.detach().numpy().reshape(n_x, n_tau),
        "sigma": float(model.sigma().detach()),
        "s_ref": S_ref,
        "k_ref": K_ref,
    }


def price_diagnostics(
    model: PINNModel,
    frame: pd.DataFrame,
    sigma_reference: Optional[float] = None,
) -> Dict[str, Any]:
    """Network prices vs analytical BS(model sigma) and vs observed prices."""
    pred = model.price_numpy(frame["spot"], frame["tau"], frame["strike"], frame["rate"])
    observed = frame["market_price"].to_numpy(dtype=float)
    sig = float(model.sigma().detach())
    analytic_self = bs_price(
        frame["spot"].to_numpy(dtype=float), frame["strike"].to_numpy(dtype=float),
        frame["tau"].to_numpy(dtype=float), frame["rate"].to_numpy(dtype=float), sig,
    )
    out = {
        "sigma_model": sig,
        "network_vs_observed_rmse": float(np.sqrt(np.mean((pred - observed) ** 2))),
        "network_vs_bs_own_sigma_rmse": float(np.sqrt(np.mean((pred - analytic_self) ** 2))),
    }
    if sigma_reference is not None and "true_price" in frame.columns:
        out["network_vs_true_rmse"] = float(np.sqrt(np.mean((pred - frame["true_price"]) ** 2)))
        out["bs_own_sigma_vs_true_rmse"] = float(np.sqrt(np.mean((analytic_self - frame["true_price"]) ** 2)))
    return out
