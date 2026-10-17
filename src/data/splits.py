"""Chronological split management: walk-forward windows with no leakage.

The walk-forward protocol (spec section 5, "Update protocol"):

    for each calibration date c (every ``step`` dates after the warmup):
        calibration window = the ``window`` most recent dates <= c
        evaluation block   = the ``horizon`` dates immediately after c
        -> calibrate on window, price the block BEFORE ingesting its premiums.

Guarantees enforced structurally (and tested in tests/test_no_leakage.py):
  * every calibration date strictly precedes every evaluation date;
  * scalers/hyperparameters are fitted on window data only;
  * evaluation blocks are disjoint across calibration events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import pandas as pd

__all__ = ["WalkForwardSplitter", "WindowSplit"]


@dataclass
class WindowSplit:
    calibration_dates: List[pd.Timestamp]
    evaluation_dates: List[pd.Timestamp]
    calibration_index: np.ndarray  # positional indices into the sorted frame
    evaluation_index: np.ndarray
    event_id: int

    @property
    def cutoff(self) -> pd.Timestamp:
        return max(self.calibration_dates)


class WalkForwardSplitter:
    """Yields (calibration, evaluation) index pairs over a dated frame."""

    def __init__(
        self,
        timestamps,
        window_dates: int,
        step_dates: int,
        horizon_dates: int,
        min_window_observations: int = 0,
        counts: np.ndarray | None = None,
    ):
        ts = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
        if not ts.is_monotonic_increasing:
            raise ValueError("timestamps must be sorted ascending to preserve positional indices")
        self.unique_dates = ts.unique().sort_values()
        self.window = int(window_dates)
        self.step = int(step_dates)
        self.horizon = int(horizon_dates)
        self.min_obs = int(min_window_observations)
        if self.window < 1 or self.step < 1 or self.horizon < 1:
            raise ValueError("window_dates, step_dates, and horizon_dates must be positive")
        if self.step < self.horizon:
            raise ValueError("step_dates must be at least horizon_dates to keep evaluation blocks disjoint")
        if self.min_obs < 0:
            raise ValueError("min_window_observations must be non-negative")
        if counts is not None:
            self.counts = np.asarray(counts)
        else:
            self.counts = ts.value_counts().reindex(self.unique_dates).to_numpy()
        if len(self.counts) != len(self.unique_dates) or np.any(self.counts < 0) or int(self.counts.sum()) != len(ts):
            raise ValueError("counts must contain one non-negative count per unique date and sum to input length")
        # positional start index of each unique date in a frame sorted by time
        self._starts = np.concatenate([[0], np.cumsum(self.counts)])

    def _date_positions(self, date_lo: pd.Timestamp, date_hi: pd.Timestamp) -> np.ndarray:
        """Positional indices for unique-date positions lo..hi inclusive."""
        lo = int(np.searchsorted(self.unique_dates, date_lo))
        hi = int(np.searchsorted(self.unique_dates, date_hi, side="right"))
        return np.arange(self._starts[lo], self._starts[hi])

    def __iter__(self) -> Iterator[WindowSplit]:
        n = len(self.unique_dates)
        event = 0
        # first calibration date index: end of the initial window
        c = self.window - 1
        while c < n:
            cal_lo_idx = max(0, c - self.window + 1)
            cal_dates = self.unique_dates[cal_lo_idx : c + 1]
            eval_lo, eval_hi = c + 1, c + self.horizon + 1
            if eval_lo >= n or eval_hi > n:
                break
            eval_dates = self.unique_dates[eval_lo:eval_hi]
            cal_pos = self._date_positions(cal_dates[0], cal_dates[-1])
            eval_pos = self._date_positions(eval_dates[0], eval_dates[-1])
            if len(cal_pos) >= self.min_obs:
                yield WindowSplit(
                    calibration_dates=list(cal_dates),
                    evaluation_dates=list(eval_dates),
                    calibration_index=cal_pos,
                    evaluation_index=eval_pos,
                    event_id=event,
                )
                event += 1
            c += self.step

    def events(self) -> List[WindowSplit]:
        return list(self)


def chronological_train_val_split(
    frame: pd.DataFrame, validation_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a calibration window into fit/val subsets.

    When the window contains multiple dates, the split is chronological (last
    fraction of dates = validation) to respect time dependence.  For
    single-date windows it falls back to a seeded random split.  Used only for
    early stopping inside the calibration window — never for test evaluation.
    """
    ts = pd.to_datetime(frame["timestamp"])
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    uniq = pd.DatetimeIndex(ts.unique()).sort_values()
    if len(uniq) > 1:
        n_val_dates = max(1, int(round(len(uniq) * validation_fraction)))
        val_dates = set(uniq[-n_val_dates:])
        val_mask = ts.isin(val_dates).to_numpy()
        # Keep at least one date for fitting.
        if val_mask.all():
            val_mask = np.zeros(len(frame), dtype=bool)
            val_mask[: max(1, int(len(frame) * validation_fraction))] = True
        return np.where(~val_mask)[0], np.where(val_mask)[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frame))
    n_val = max(1, int(len(idx) * validation_fraction))
    return idx[n_val:], idx[:n_val]
