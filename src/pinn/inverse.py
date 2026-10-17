"""Inverse PINN: joint price-network + shared-volatility calibration.

One positive effective volatility per calibration window (spec identifiability
restriction).  The calibration output for downstream pricing is the pair
(sigma_hat, V_theta); per the specification, analytical prices computed with
sigma_hat are reported separately from the network's direct prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .training import TrainConfig, TrainResult, train_pinn_window
from .network import PINNModel

__all__ = ["InversePINNResult", "fit_inverse_pinn"]


@dataclass
class InversePINNResult:
    train: TrainResult
    sigma_hat: Optional[float]
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sigma_hat": self.sigma_hat,
            "status": self.status,
            "training": self.train.to_dict(),
        }


def fit_inverse_pinn(
    window: pd.DataFrame,
    config: TrainConfig,
    seed: int,
    sigma_init: Optional[float] = None,
    return_model: bool = False,
) -> Tuple[InversePINNResult, Optional[PINNModel]]:
    """Fit the inverse PINN on one calibration window.

    ``sigma_init`` warm-starts the shared volatility from the previous
    window's estimate (past information only).
    """
    result, model = train_pinn_window(
        window,
        config,
        seed=seed,
        trainable_sigma=True,
        sigma_init_override=sigma_init,
        return_model=True,
    )
    sigma_hat = result.final_sigma if result.status != "failed" else None
    out = InversePINNResult(train=result, sigma_hat=sigma_hat, status=result.status)
    return (out, model if return_model else None)
