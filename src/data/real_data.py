"""Real-market data adapter (Experiment 3, conditional).

Implements ingestion of historical European option chains following the
dataset design document (bid/ask mid premiums, maturity-matched rates,
synchronized spot, OTM-focused filters, documented exclusions).  The experiment
executes ONLY when a suitable dataset is actually available; when it is not,
``load_real_data`` raises ``RealDataUnavailable`` and the experiment records an
honest skipped status.  Unsuitable data is never substituted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .schema import COLUMNS, enforce_schema

__all__ = ["RealDataUnavailable", "RealDataEligibility", "load_real_data", "apply_quality_filters"]

_DAYS_PER_YEAR = 365.25


class RealDataUnavailable(Exception):
    """Raised when no suitable real dataset is accessible."""


@dataclass
class RealDataEligibility:
    """Checks required before real data may be used (spec section 9)."""

    exercise_style_ok: bool = False
    bid_ask_ok: bool = False
    timestamps_ok: bool = False
    positive_maturity_ok: bool = False
    rate_compatible_ok: bool = False
    dividends_ok: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return all(
            [
                self.exercise_style_ok,
                self.bid_ask_ok,
                self.timestamps_ok,
                self.positive_maturity_ok,
                self.rate_compatible_ok,
                self.dividends_ok,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible,
            "checks": {
                "european_exercise": self.exercise_style_ok,
                "valid_bid_ask": self.bid_ask_ok,
                "timestamps": self.timestamps_ok,
                "positive_maturity": self.positive_maturity_ok,
                "rate_compatible": self.rate_compatible_ok,
                "no_material_dividends": self.dividends_ok,
            },
            "notes": self.notes,
        }


def check_eligibility(frame: pd.DataFrame, cfg: Dict[str, Any]) -> RealDataEligibility:
    """Run eligibility checks on a candidate real dataset."""
    el = RealDataEligibility()
    notes: List[str] = []

    if "exercise_style" in frame.columns:
        styles = set(frame["exercise_style"].dropna().astype(str).str.lower())
        el.exercise_style_ok = styles.issubset({"european", ""})
        if not el.exercise_style_ok:
            notes.append(f"non-European exercise styles present: {sorted(styles)}")
    else:
        notes.append("exercise_style column missing; cannot confirm European exercise")

    has_bid_ask = {"bid", "ask"}.issubset(frame.columns) and frame[["bid", "ask"]].notna().all().all()
    valid_ba = has_bid_ask and bool((frame["bid"] > 0).all() and (frame["ask"] >= frame["bid"]).all())
    el.bid_ask_ok = bool(valid_ba)
    if not valid_ba:
        notes.append("valid bid/ask not present for all rows (mid premium preferred)")

    el.timestamps_ok = "timestamp" in frame.columns and bool(
        frame["timestamp"].notna().all()
    )
    if not el.timestamps_ok:
        notes.append("timestamps missing or invalid")

    if {"timestamp", "expiration"}.issubset(frame.columns):
        dte = (pd.to_datetime(frame["expiration"]) - pd.to_datetime(frame["timestamp"])).dt.days
        el.positive_maturity_ok = bool((dte >= cfg.get("eligibility", {}).get("min_maturity_days", 7)).all())
        if not el.positive_maturity_ok:
            notes.append("non-positive or too-short maturities present")
    else:
        notes.append("timestamp/expiration missing; cannot verify maturity")

    el.rate_compatible_ok = "rate" in frame.columns and bool(
        frame["rate"].notna().all() and np.isfinite(frame["rate"].to_numpy(dtype=float)).all()
    )
    if not el.rate_compatible_ok:
        notes.append("maturity-compatible rates missing")

    # Dividend assumption: requires either a documented zero-dividend underlying
    # or a dividend yield column that the core BS model can incorporate. The
    # core experiments assume no dividends.
    if "dividend_yield" in frame.columns:
        max_q = float(np.nanmax(np.abs(frame["dividend_yield"].to_numpy(dtype=float))))
        el.dividends_ok = max_q <= 1e-4
        if not el.dividends_ok:
            notes.append(f"material dividends present (max |q| = {max_q}); core BS model assumes none")
    else:
        el.dividends_ok = False
        notes.append("dividend information unavailable; cannot confirm no-material-dividends assumption")

    el.notes = notes
    return el


def load_real_data(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load a real dataset or raise ``RealDataUnavailable``.

    The dataset must be a CSV/Parquet file at ``real_data.source_path``
    implementing the schema in docs/data_dictionary.md.  In this environment
    no eligible provider is accessible, so the default configuration raises
    immediately with the documented reason.
    """
    rd = cfg.get("real_data", {})
    if not rd.get("enabled", False):
        raise RealDataUnavailable(rd.get("reason_unavailable", "real_data.enabled is false"))

    path = rd.get("source_path")
    if not path or not os.path.exists(path):
        raise RealDataUnavailable(
            f"configured source_path does not exist: {path}. "
            "Suitable historical European option chains require a licensed or "
            "keyed provider (OptionMetrics/WRDS, CBOE DataShop, Alpha Vantage "
            "premium, ...); none is accessible here."
        )

    if str(path).endswith(".parquet"):
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)

    frame = enforce_schema(raw)
    el = check_eligibility(frame, rd)
    if not el.eligible:
        raise RealDataUnavailable(f"dataset failed eligibility checks: {el.to_dict()}")
    return frame.sort_values("timestamp").reset_index(drop=True)


def apply_quality_filters(
    frame: pd.DataFrame, filters: Dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the dataset-design QC filters; returns (kept, exclusions_log)."""
    import logging

    log = logging.getLogger("real_data.filters")
    kept = frame.copy()
    exclusions: List[Dict[str, Any]] = []

    def drop(mask: pd.Series, rule: str, reason: str):
        nonlocal kept
        n_drop = int(mask.sum())
        if n_drop:
            exclusions.append({"rule": rule, "reason": reason, "n_dropped": n_drop})
            log.info("filter %s dropped %d rows (%s)", rule, n_drop, reason)
        kept = kept.loc[~mask]

    ts = pd.to_datetime(kept["timestamp"])
    dte = (pd.to_datetime(kept["expiration"]) - ts).dt.days

    if filters.get("drop_duplicates", True):
        dup = kept.duplicated(subset=["timestamp", "strike", "expiration", "option_type"])
        drop(dup, "duplicates", "duplicate contract records")

    drop(kept["market_price"] < filters.get("min_mid_price", 0.05), "min_mid_price", "mid below minimum")
    rel_spread = (kept["ask"] - kept["bid"]) / kept["market_price"]
    drop(rel_spread > filters.get("max_relative_spread", 0.25), "max_relative_spread", "spread too wide")

    if "open_interest" in kept.columns and kept["open_interest"].notna().any():
        drop(
            kept["open_interest"] < filters.get("min_open_interest", 10),
            "min_open_interest",
            "insufficient open interest",
        )
    if "volume" in kept.columns and kept["volume"].notna().any():
        drop(kept["volume"] < filters.get("min_volume", 1), "min_volume", "no volume")

    dte_lo, dte_hi = filters.get("dte_range", [7, 180])
    drop((dte < dte_lo) | (dte > dte_hi), "dte_range", f"dte outside [{dte_lo}, {dte_hi}]")

    log_m = np.log(kept["strike"].to_numpy(dtype=float) / kept["spot"].to_numpy(dtype=float))
    drop(np.abs(log_m) > filters.get("max_abs_log_moneyness", 0.10), "log_moneyness", "|ln(K/S)| too large")

    if "implied_volatility" in kept.columns and kept["implied_volatility"].notna().any():
        iv_lo, iv_hi = filters.get("iv_range", [0.02, 2.0])
        drop(
            (kept["implied_volatility"] < iv_lo) | (kept["implied_volatility"] > iv_hi),
            "iv_range",
            "vendor IV outside sane range",
        )

    return kept.reset_index(drop=True), pd.DataFrame(exclusions)
