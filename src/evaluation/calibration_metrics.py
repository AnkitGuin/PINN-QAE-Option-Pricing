"""Calibration-quality metrics (spec section 10).

Parameter estimation error against known synthetic truth, volatility
reconstruction error, update stability, adaptation lag, and physics-consistency
diagnostics.  Two reference definitions are reported for time-varying
scenarios (they coincide for constant volatility):

  sigma_instantaneous : schedule value at the calibration date
  sigma_effective     : sqrt of the quote-mean integrated variance A/tau over
                        the evaluation-block contracts
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = ["sigma_error_metrics", "stability_metrics", "adaptation_lag"]


def sigma_error_metrics(
    sigma_hats: List[Optional[float]],
    references: Dict[str, List[Optional[float]]],
) -> Dict[str, Any]:
    """Absolute / relative sigma errors against each reference definition."""
    out: Dict[str, Any] = {}
    arr = np.array([s if s is not None else np.nan for s in sigma_hats], dtype=float)
    out["n_windows"] = int(len(arr))
    out["n_failed"] = int(np.sum(~np.isfinite(arr)))
    for name, ref in references.items():
        ref_arr = np.array([r if r is not None else np.nan for r in ref], dtype=float)
        both = np.isfinite(arr) & np.isfinite(ref_arr)
        if np.any(both):
            err = arr[both] - ref_arr[both]
            out[f"abs_error_vs_{name}"] = {
                "mean": float(np.mean(np.abs(err))),
                "median": float(np.median(np.abs(err))),
                "max": float(np.max(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
            }
            out[f"rel_error_vs_{name}"] = {
                "mean": float(np.mean(np.abs(err) / ref_arr[both])),
            }
    return out


def stability_metrics(sigma_hats: List[Optional[float]]) -> Dict[str, Any]:
    """Update stability across consecutive calibration events."""
    arr = np.array([s if s is not None else np.nan for s in sigma_hats], dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        return {"n": int(len(finite))}
    diffs = np.abs(np.diff(finite))
    return {
        "n": int(len(finite)),
        "std_across_windows": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "median_abs_change": float(np.median(diffs)),
        "max_abs_change": float(np.max(diffs)),
        "mean_abs_change": float(np.mean(diffs)),
    }


def adaptation_lag(
    sigma_hats: List[Optional[float]],
    true_instantaneous: List[float],
    event_index: List[int],
) -> Dict[str, Any]:
    """Lag of estimates behind a known instantaneous path (synthetic only).

    Reports the mean absolute error between the estimate and the instantaneous
    truth at each event, plus the error against the truth ``k`` events later
    (how long the estimate trails a change).
    """
    est = np.array([s if s is not None else np.nan for s in sigma_hats], dtype=float)
    truth = np.asarray(true_instantaneous, dtype=float)
    idx = np.asarray(event_index, dtype=int)
    if len(est) != len(truth):
        return {}
    mae_now = float(np.nanmean(np.abs(est - truth)))
    # error vs truth 1 and 2 events ahead (if the estimate tracked perfectly,
    # these quantify how much change was about to happen)
    mae_next = float(np.nanmean(np.abs(est[:-1] - truth[1:]))) if len(est) > 1 else mae_now
    return {
        "mae_vs_current_instantaneous": mae_now,
        "mae_vs_next_event_instantaneous": mae_next,
        "events": idx.tolist(),
        "estimates": [None if not np.isfinite(v) else float(v) for v in est],
        "truth_instantaneous": truth.tolist(),
    }
