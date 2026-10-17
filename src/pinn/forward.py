"""Forward PINN: solve the pricing PDE with externally known parameters.

Purpose (Experiment 1 / Gate 4): verify that the PINN can approximate the
Black-Scholes solution BEFORE any inverse calibration is trusted.  Sigma is
supplied externally and never trained.  Optionally, sparse analytical price
labels can be added (config ``use_data_labels``); the default is a pure
physics solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.preprocessing import WindowScaler, fit_window_scaler
from ..utils.reproducibility import derive_seed
from .collocation import CollocationConfig, sample_window_collocation
from .losses import compute_losses, pde_residual
from .network import PINNModel
from .training import TrainConfig, train_pinn_window

__all__ = ["ForwardPINNResult", "fit_forward_pinn"]


@dataclass
class ForwardPINNResult:
    sigma: float
    price_rmse: Optional[float]
    price_mae: Optional[float]
    price_max_error: Optional[float]
    pde_residual_rms: float
    boundary_error: float
    terminal_error: float
    runtime_seconds: float
    status: str
    stop_reason: str
    history_len: int

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def fit_forward_pinn(
    frame: pd.DataFrame,
    config: TrainConfig,
    seed: int,
    known_sigma: float,
    use_data_labels: bool = False,
    return_model: bool = False,
) -> Tuple[ForwardPINNResult, Optional[PINNModel]]:
    """Train the price network with a FIXED, externally supplied sigma."""
    cfg = config
    local = TrainConfig(**{**cfg.__dict__, "use_data": use_data_labels})
    result, model = train_pinn_window(
        frame,
        local,
        seed=seed,
        trainable_sigma=False,
        fixed_sigma=known_sigma,
        return_model=True,
    )

    # Independent diagnostics on a fresh collocation set.
    coll = sample_window_collocation(
        frame, cfg.collocation, seed=derive_seed(seed, "diagnostic_collocation")
    )
    import torch

    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64)
    with torch.enable_grad():
        f_hat = pde_residual(
            model,
            t(coll.interior_S), t(coll.interior_tau), t(coll.interior_K), t(coll.interior_r),
            t(np.full(len(coll.interior_S), known_sigma)),
        )
        resid_rms = float(torch.sqrt(torch.mean(f_hat**2)).detach())
        u_b = model.u(t(coll.boundary_S), t(coll.boundary_tau), t(coll.boundary_K), t(coll.boundary_r))
        bc_err = float(torch.sqrt(torch.mean((u_b - t(coll.boundary_target) / t(coll.boundary_S)) ** 2)).detach())
        u_T = model.u(t(coll.terminal_S), torch.zeros_like(t(coll.terminal_S)), t(coll.terminal_K), t(coll.terminal_r))
        term_err = float(torch.sqrt(torch.mean((u_T - t(coll.terminal_payoff) / t(coll.terminal_S)) ** 2)).detach())

    out = ForwardPINNResult(
        sigma=known_sigma,
        price_rmse=None,
        price_mae=None,
        price_max_error=None,
        pde_residual_rms=resid_rms,
        boundary_error=bc_err,
        terminal_error=term_err,
        runtime_seconds=result.runtime_seconds,
        status=result.status,
        stop_reason=result.stop_reason,
        history_len=len(result.history),
    )
    return (out, model if return_model else None)
