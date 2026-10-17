"""Preprocessing: window-local normalization fitted on calibration data only.

The PINN works in homogeneous coordinates u(ln(S/K), tau, r) (see
src/pinn/network.py).  Normalization statistics are computed from the
CALIBRATION WINDOW alone; the fitted scaler is applied unchanged to later
evaluation blocks, and test data never influence any statistic (tested in
tests/test_no_leakage.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

__all__ = ["WindowScaler", "fit_window_scaler"]


@dataclass
class WindowScaler:
    """Normalization statistics for the homogeneous PINN coordinates.

    Features (z-scored):  z = (ln(S/K), tau, r)
    Output:               u is dimensionless; physical price V = S * u.
    """

    mean_log_moneyness: float
    std_log_moneyness: float
    mean_tau: float
    std_tau: float
    mean_rate: float
    std_rate: float
    mean_spot: float  # reference only (not used by the network)

    def transform_inputs(self, S, tau, K, r) -> np.ndarray:
        x = np.log(np.asarray(S, dtype=float) / np.asarray(K, dtype=float))
        z1 = (x - self.mean_log_moneyness) / self.std_log_moneyness
        z2 = (np.asarray(tau, dtype=float) - self.mean_tau) / self.std_tau
        z3 = (np.asarray(r, dtype=float) - self.mean_rate) / self.std_rate
        return np.column_stack([z1, z2, z3])

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def _safe_std(x: np.ndarray, fallback: float) -> float:
    s = float(np.std(x))
    if not np.isfinite(s) or s < 1e-8:
        return fallback
    return s


def fit_window_scaler(frame: pd.DataFrame) -> WindowScaler:
    """Fit normalization statistics on a calibration window (and nothing else)."""
    S = frame["spot"].to_numpy(dtype=float)
    K = frame["strike"].to_numpy(dtype=float)
    tau = frame["tau"].to_numpy(dtype=float)
    r = frame["rate"].to_numpy(dtype=float)
    log_m = np.log(S / K)

    return WindowScaler(
        mean_log_moneyness=float(np.mean(log_m)),
        std_log_moneyness=_safe_std(log_m, fallback=0.10),
        mean_tau=float(np.mean(tau)),
        std_tau=_safe_std(tau, fallback=0.10),
        mean_rate=float(np.mean(r)),
        std_rate=_safe_std(r, fallback=0.005),
        mean_spot=float(np.mean(S)),
    )
