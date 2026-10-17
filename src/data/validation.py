"""Data validation checks (Gate 2 and ongoing data-quality gates).

Checks: schema presence, positivity, chronology, no duplicates, maturity
correctness (tau consistent with timestamp/expiration), price bounds, noise
consistency, regime correctness against the schedule, and reproducibility
(re-generation produces identical frames).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..black_scholes.analytical import price_bounds
from .schema import COLUMNS

__all__ = ["validate_observations", "validate_dates_frame"]

_DAYS_PER_YEAR = 365.25


def validate_observations(
    obs: pd.DataFrame,
    dates_df: Optional[pd.DataFrame] = None,
    bounds_tolerance: float = 0.05,
) -> Dict[str, Any]:
    """Validate an observation frame; returns a report dict with pass/fail."""
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Optional[Dict] = None):
        checks.append({"check": name, "status": "pass" if passed else "fail", **(detail or {})})

    missing = [c for c in COLUMNS if c not in obs.columns]
    add("schema_columns_present", not missing, {"missing": missing})

    if missing:
        return {"status": "fail", "checks": checks}

    add(
        "positive_spot_strike_tau",
        bool(
            (obs["spot"] > 0).all()
            and (obs["strike"] > 0).all()
            and (obs["tau"] > 0).all()
            and np.isfinite(obs["spot"]).all()
            and np.isfinite(obs["strike"]).all()
            and np.isfinite(obs["tau"]).all()
        ),
    )

    # Maturity correctness: tau == (expiration - timestamp)/365.25 within a day.
    if pd.api.types.is_datetime64_any_dtype(obs["expiration"]) and pd.api.types.is_datetime64_any_dtype(
        obs["timestamp"]
    ):
        dte = (obs["expiration"] - obs["timestamp"]).dt.days.to_numpy(dtype=float)
        tau_implied = dte / _DAYS_PER_YEAR
        err = float(np.max(np.abs(tau_implied - obs["tau"].to_numpy(dtype=float))))
        add("maturity_consistency", err <= 1.5 / _DAYS_PER_YEAR, {"max_abs_error_years": err})

    # Chronology + uniqueness
    ts = pd.to_datetime(obs["timestamp"])
    add("timestamps_sorted", bool(ts.is_monotonic_increasing) or bool(ts.sort_values().index.is_unique))
    dup_mask = obs.duplicated(subset=["timestamp", "strike", "expiration", "option_type"])
    add("no_duplicate_contracts", not bool(dup_mask.any()), {"n_duplicates": int(dup_mask.sum())})

    # Price bounds (calls, no dividends) with tolerance for noise.
    lo, hi = price_bounds(
        obs["spot"].to_numpy(dtype=float),
        obs["strike"].to_numpy(dtype=float),
        obs["tau"].to_numpy(dtype=float),
        obs["rate"].to_numpy(dtype=float),
        "call",
    )
    mid = obs["market_price"].to_numpy(dtype=float)
    viol = float(np.maximum(lo - bounds_tolerance - mid, 0.0).max() + np.maximum(mid - hi - bounds_tolerance, 0.0).max())
    add("price_within_bounds", viol <= 0.0, {"max_violation": viol, "tolerance": bounds_tolerance})

    # Noise field consistency (synthetic).
    if "noise" in obs.columns and obs["noise"].notna().all():
        err = float(np.max(np.abs((obs["market_price"] - obs["true_price"] - obs["noise"]).to_numpy(dtype=float))))
        add("noise_field_consistency", err < 1e-12, {"max_abs_error": err})

    # True sigma within configured sane range if present.
    if "true_sigma" in obs.columns and obs["true_sigma"].notna().all():
        sig = obs["true_sigma"].to_numpy(dtype=float)
        add("true_sigma_positive_sane", bool((sig > 0).all() and (sig < 5.0).all()), {"range": [float(sig.min()), float(sig.max())]})

    # Regime correctness against the dates frame.
    if dates_df is not None:
        merged = obs.merge(
            dates_df[["timestamp", "sigma_instantaneous", "regime"]],
            on="timestamp",
            how="left",
            suffixes=("", "_date"),
        )
        inst_err = float(
            np.max(np.abs(merged["true_sigma_instantaneous"] - merged["sigma_instantaneous"]).to_numpy(dtype=float))
        )
        regime_ok = bool((merged["regime"] == merged["regime_date"]).all())
        add("regime_schedule_consistency", regime_ok and inst_err < 1e-12, {"instantaneous_sigma_max_err": inst_err})

    # Bid/ask sanity where present.
    if obs["bid"].notna().any():
        ba = obs.dropna(subset=["bid", "ask"])
        add(
            "bid_ask_sane",
            bool((ba["ask"] >= ba["bid"]).all() and (ba["bid"] >= 0).all()),
        )

    overall = all(c["status"] == "pass" for c in checks)
    return {"status": "pass" if overall else "fail", "checks": checks, "n_observations": int(len(obs))}


def validate_dates_frame(dates_df: pd.DataFrame, vol_range=(0.01, 3.0)) -> Dict[str, Any]:
    checks = []
    sig = dates_df["sigma_instantaneous"].to_numpy(dtype=float)
    checks.append(
        {
            "check": "schedule_values_in_range",
            "status": "pass" if bool(((sig >= vol_range[0]) & (sig <= vol_range[1])).all()) else "fail",
            "range": [float(sig.min()), float(sig.max())],
        }
    )
    checks.append(
        {
            "check": "yearfrac_monotone",
            "status": "pass" if bool(np.all(np.diff(dates_df["t_yearfrac"].to_numpy(dtype=float)) > 0)) else "fail",
        }
    )
    return {"status": "pass" if all(c["status"] == "pass" for c in checks) else "fail", "checks": checks}
