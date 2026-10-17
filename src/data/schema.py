"""Canonical observation schema shared by synthetic and real data.

Fields
------
timestamp            observation date/time
underlying           asset identifier
spot                 underlying price at timestamp (positive)
strike               strike (positive)
expiration           expiry date
tau                  time to maturity in year fractions (positive)
option_type          call | put
exercise_style       european | american
currency             currency code
rate                 continuously compounded risk-free rate used for this contract
bid, ask             quote fields (NaN when unavailable)
market_price         premium used as observation (mid when bid/ask valid)
implied_volatility   diagnostic only; NEVER ground truth, NEVER an input feature
volume, open_interest liquidity fields (NaN when unavailable)
true_price           synthetic only: exact model price
true_sigma           synthetic only: effective (integrated) volatility of the contract
true_sigma_instantaneous synthetic only: schedule value at the observation date
observed_price       market_price alias kept for clarity in analysis
noise                synthetic only: observed - true
regime               synthetic only: volatility regime label
source               provenance tag
quality_flag         0 = pass; bitmask of issues otherwise

SYNTHETIC_ONLY_FIELDS must never be used as model inputs or calibration
targets; they exist for recovery assessment and diagnostics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

COLUMNS: List[str] = [
    "timestamp",
    "underlying",
    "spot",
    "strike",
    "expiration",
    "tau",
    "option_type",
    "exercise_style",
    "currency",
    "rate",
    "bid",
    "ask",
    "market_price",
    "implied_volatility",
    "volume",
    "open_interest",
    "true_price",
    "true_sigma",
    "true_sigma_instantaneous",
    "observed_price",
    "noise",
    "regime",
    "source",
    "quality_flag",
]

SYNTHETIC_ONLY_FIELDS: List[str] = [
    "true_price",
    "true_sigma",
    "true_sigma_instantaneous",
    "noise",
    "regime",
]

# Fields that may legitimately be used as calibration/pricing INPUTS.
MODEL_INPUT_FIELDS: List[str] = ["spot", "strike", "tau", "rate"]

QUALITY_FLAGS: Dict[int, str] = {
    0: "pass",
    1: "negative_or_zero_price_clipped",
    2: "implied_vol_inversion_failed",
    4: "outside_no_arbitrage_bounds",
}

DATE_LEVEL_COLUMNS: List[str] = [
    "timestamp",
    "spot",
    "rate",
    "sigma_instantaneous",
    "regime",
    "t_yearfrac",
    "source",
]


def enforce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder/select canonical columns, filling missing ones with NaN."""
    out = pd.DataFrame(index=df.index)
    for col in COLUMNS:
        out[col] = df[col] if col in df.columns else np.nan
    return out[COLUMNS]


def empty_frame() -> pd.DataFrame:
    return enforce_schema(pd.DataFrame())
