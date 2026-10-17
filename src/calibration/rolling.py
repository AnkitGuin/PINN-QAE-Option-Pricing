"""Rolling (walk-forward) and frozen calibration engines.

Rolling: at each calibration date, fit only on the trailing window of quotes,
then price the following out-of-sample block before it is ingested.
Frozen: fit once on the initial window and never update (measures the value of
adaptation).

Both classical engines and the PINN use the identical ``WalkForwardSplitter``
so windows, observations, rates and degrees of freedom are matched exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..black_scholes.analytical import bs_price
from ..data.splits import WalkForwardSplitter, WindowSplit
from .least_squares import LeastSquaresResult, calibrate_least_squares

__all__ = ["RollingLSEngine", "FrozenEngine", "CalibrationRecord"]


@dataclass
class CalibrationRecord:
    event_id: int
    cutoff: pd.Timestamp
    n_window_observations: int
    sigma: Optional[float]
    method: str
    runtime_seconds: float
    status: str
    objective: Optional[float] = None
    extra: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out["cutoff"] = str(self.cutoff)
        return out


class RollingLSEngine:
    """Rolling constrained least-squares calibration over walk-forward windows."""

    method_name = "least_squares"

    def __init__(self, calibration_cfg: Dict[str, Any]):
        rcfg = calibration_cfg["rolling"]
        self.window_dates = int(rcfg["window_dates"])
        self.step_dates = int(rcfg["step_dates"])
        self.horizon_dates = int(rcfg["horizon_dates"])
        self.min_obs = int(rcfg.get("min_window_observations", 0))
        self.sigma_bounds = tuple(calibration_cfg["sigma_bounds"])
        ls_cfg = calibration_cfg.get("ls", {})
        self.x_tolerance = float(ls_cfg.get("x_tolerance", 1e-10))
        self.max_evaluations = int(ls_cfg.get("max_evaluations", 200))
        self.warm_start = bool(ls_cfg.get("warm_start", True))
        self._last_sigma: Optional[float] = None

    def splitter(self, frame: pd.DataFrame) -> WalkForwardSplitter:
        return WalkForwardSplitter(
            frame["timestamp"],
            window_dates=self.window_dates,
            step_dates=self.step_dates,
            horizon_dates=self.horizon_dates,
            min_window_observations=self.min_obs,
        )

    def calibrate_window(self, window_frame: pd.DataFrame) -> LeastSquaresResult:
        init = self._last_sigma if (self.warm_start and self._last_sigma is not None) else None
        result = calibrate_least_squares(
            window_frame["spot"].to_numpy(dtype=float),
            window_frame["strike"].to_numpy(dtype=float),
            window_frame["tau"].to_numpy(dtype=float),
            window_frame["rate"].to_numpy(dtype=float),
            window_frame["market_price"].to_numpy(dtype=float),
            sigma_bounds=self.sigma_bounds,
            x_tolerance=self.x_tolerance,
            max_evaluations=self.max_evaluations,
            init=init,
        )
        if result.status == "success":
            self._last_sigma = result.sigma
        return result

    def run(self, frame: pd.DataFrame) -> List[CalibrationRecord]:
        records: List[CalibrationRecord] = []
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        for split in self.splitter(frame):
            window = frame.iloc[split.calibration_index]
            result = self.calibrate_window(window)
            records.append(
                CalibrationRecord(
                    event_id=split.event_id,
                    cutoff=split.cutoff,
                    n_window_observations=int(len(window)),
                    sigma=result.sigma,
                    method=self.method_name,
                    runtime_seconds=result.runtime_seconds,
                    status=result.status,
                    objective=result.objective,
                    extra={"n_evaluations": result.n_evaluations, "init": result.init},
                )
            )
        return records


class FrozenEngine(RollingLSEngine):
    """Least-squares fit on the FIRST calibration window only; never updated."""

    method_name = "frozen"

    def __init__(self, calibration_cfg: Dict[str, Any]):
        super().__init__(calibration_cfg)
        self._frozen_sigma: Optional[float] = None

    def run(self, frame: pd.DataFrame) -> List[CalibrationRecord]:
        records: List[CalibrationRecord] = []
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        for split in self.splitter(frame):
            if self._frozen_sigma is None:
                window = frame.iloc[split.calibration_index]
                result = self.calibrate_window(window)
                if result.status == "success":
                    self._frozen_sigma = result.sigma
                records.append(
                    CalibrationRecord(
                        event_id=split.event_id,
                        cutoff=split.cutoff,
                        n_window_observations=int(len(window)),
                        sigma=result.sigma,
                        method=self.method_name,
                        runtime_seconds=result.runtime_seconds,
                        status=result.status,
                        objective=result.objective,
                    )
                )
            else:
                records.append(
                    CalibrationRecord(
                        event_id=split.event_id,
                        cutoff=split.cutoff,
                        n_window_observations=0,
                        sigma=self._frozen_sigma,
                        method=self.method_name,
                        runtime_seconds=0.0,
                        status="reused_frozen",
                    )
                )
        return records


def price_with_sigma(frame: pd.DataFrame, sigma: Optional[float]) -> np.ndarray:
    """Price every contract with the SAME analytical evaluator and one sigma."""
    if sigma is None or not np.isfinite(sigma):
        return np.full(len(frame), np.nan)
    return bs_price(
        frame["spot"].to_numpy(dtype=float),
        frame["strike"].to_numpy(dtype=float),
        frame["tau"].to_numpy(dtype=float),
        frame["rate"].to_numpy(dtype=float),
        sigma,
        "call",
    )
