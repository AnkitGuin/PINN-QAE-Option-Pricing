"""Feature engineering: build model matrices from observation frames.

Only MODEL_INPUT_FIELDS (spot, strike, tau, rate) plus the observed premium
are extracted.  Synthetic-only ground-truth columns and held-out implied
volatilities are structurally excluded (see tests/test_no_leakage.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .schema import SYNTHETIC_ONLY_FIELDS

__all__ = ["ObservationMatrix", "build_observation_matrix", "INPUT_COLUMNS", "TARGET_COLUMN"]

INPUT_COLUMNS = ["spot", "strike", "tau", "rate"]
TARGET_COLUMN = "market_price"


@dataclass
class ObservationMatrix:
    spot: np.ndarray
    strike: np.ndarray
    tau: np.ndarray
    rate: np.ndarray
    price: np.ndarray  # observed premium (mid)
    index: np.ndarray  # positional indices into the source frame
    timestamps: pd.DatetimeIndex

    def __len__(self) -> int:
        return len(self.spot)


def build_observation_matrix(
    frame: pd.DataFrame,
    include_synthetic_truth: bool = False,
) -> ObservationMatrix:
    """Extract the calibration/pricing inputs from an observation frame.

    ``include_synthetic_truth`` adds nothing to the INPUTS; it only controls
    whether synthetic reference columns are attached for evaluation code paths
    (never for calibration).  Inputs are always the four MODEL_INPUT_FIELDS.
    """
    for col in INPUT_COLUMNS:
        if col not in frame.columns:
            raise KeyError(f"required input column missing: {col}")
    if TARGET_COLUMN not in frame.columns:
        raise KeyError(f"target column missing: {TARGET_COLUMN}")

    return ObservationMatrix(
        spot=frame["spot"].to_numpy(dtype=float),
        strike=frame["strike"].to_numpy(dtype=float),
        tau=frame["tau"].to_numpy(dtype=float),
        rate=frame["rate"].to_numpy(dtype=float),
        price=frame[TARGET_COLUMN].to_numpy(dtype=float),
        index=frame.index.to_numpy(),
        timestamps=pd.DatetimeIndex(pd.to_datetime(frame["timestamp"])),
    )


def strip_synthetic_fields(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without synthetic-only ground-truth columns.

    Used to prove that no calibration path can even see them.
    """
    return frame.drop(columns=[c for c in SYNTHETIC_ONLY_FIELDS if c in frame.columns])
