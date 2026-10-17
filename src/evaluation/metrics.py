"""Pricing metrics (spec section 10).

    MAE = mean |e_i|
    MSE = mean e_i^2
    RMSE = sqrt(MSE)
    Relative error = mean |e_i| / max(|V_ref|, epsilon_P)   (guarded)
    Bias = mean e_i

with e_i = V_hat_i - V_ref_i.  The reference is the analytical true price for
synthetic experiments and the observed premium for real-market experiments.
Date-aggregated variants respect within-date dependence.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["pricing_metrics", "group_metrics"]


def pricing_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    price_floor: float = 0.05,
    timestamps: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    e = predicted - reference
    n = len(e)
    out: Dict[str, Any] = {
        "n": int(n),
        "mae": float(np.mean(np.abs(e))),
        "mse": float(np.mean(e**2)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "relative_error": float(np.mean(np.abs(e) / np.maximum(np.abs(reference), price_floor))),
        "bias": float(np.mean(e)),
    }
    near_zero = np.abs(reference) < price_floor
    out["n_near_zero_price_contracts"] = int(np.sum(near_zero))
    if n > 0 and np.any(~near_zero):
        out["relative_error_excluding_near_zero"] = float(
            np.mean(np.abs(e[~near_zero]) / np.maximum(np.abs(reference[~near_zero]), price_floor))
        )

    if timestamps is not None and n > 1:
        ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
        per_date = pd.Series(e).groupby(ts.values).agg(["mean", "std", "count"])
        # date-aggregated RMSE: sqrt of the mean squared date-mean error
        out["date_aggregated_rmse"] = float(np.sqrt(np.mean(per_date["mean"] ** 2)))
        out["n_dates"] = int(len(per_date))
    return out


def group_metrics(
    frame: pd.DataFrame,
    predicted_col: str,
    reference_col: str,
    group_col: str,
    price_floor: float = 0.05,
) -> pd.DataFrame:
    """Pricing metrics per group (maturity bucket, moneyness bucket, regime)."""
    rows = []
    for g, sub in frame.groupby(group_col):
        e = sub[predicted_col].to_numpy(dtype=float) - sub[reference_col].to_numpy(dtype=float)
        rows.append(
            {
                group_col: g,
                "n": int(len(sub)),
                "mae": float(np.mean(np.abs(e))),
                "rmse": float(np.sqrt(np.mean(e**2))),
                "bias": float(np.mean(e)),
                "relative_error": float(
                    np.mean(np.abs(e) / np.maximum(np.abs(sub[reference_col].to_numpy(dtype=float)), price_floor))
                ),
            }
        )
    return pd.DataFrame(rows)


def maturity_moneyness_buckets(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach maturity and moneyness buckets used across evaluations."""
    out = frame.copy()
    tau = out["tau"].to_numpy(dtype=float)
    out["maturity_bucket"] = pd.cut(
        tau, bins=[0, 0.15, 0.35, 0.7, np.inf],
        labels=["short_(<0.15)", "mid_(0.15-0.35)", "long_(0.35-0.7)", "vlong_(>0.7)"],
    )
    m = out["strike"].to_numpy(dtype=float) / out["spot"].to_numpy(dtype=float)
    out["moneyness_bucket"] = pd.cut(
        m, bins=[0, 0.92, 0.98, 1.02, 1.08, np.inf],
        labels=["deep_otm_(<0.92)", "otm_(0.92-0.98)", "atm_(0.98-1.02)", "itm_(1.02-1.08)", "deep_itm_(>1.08)"],
    )
    return out
