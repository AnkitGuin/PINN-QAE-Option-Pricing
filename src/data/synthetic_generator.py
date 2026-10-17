"""Fully reproducible synthetic European-option observation generator.

Ground-truth model
------------------
The latent pricing volatility is a step function on the business-date axis:

    sigma(u) = sigma_d   for u in [t_d, t_{d+1}),   t_d = business date d,

so the total variance between observation time t_i and expiry u_T is EXACT:

    A(t_i, u_T) = sum_d sigma_d^2 * length([t_d, t_{d+1}) cap [t_i, u_T]).

Rates are observed per quote date and held constant to expiry (documented
convention: "core runs use a known rate held fixed for each contract's pricing
run"), so R(t_i, u_T) = r_i * tau.

True prices use the total-variance Black-Scholes form; the naive substitution
of an instantaneous sigma into a constant-parameter formula is deliberately
NOT used (see src/black_scholes/analytical.py).

Latent ground truth (true_price, true_sigma, sigma_instantaneous, regime) is
preserved in dedicated columns that never feed calibration.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..black_scholes.analytical import call_price_total_variance
from ..black_scholes.implied_vol import implied_volatility
from ..utils.reproducibility import derive_seed
from .schema import COLUMNS, DATE_LEVEL_COLUMNS

__all__ = ["generate_synthetic_dataset", "integrated_variance"]

DAYS_PER_YEAR = 365.25


# ---------------------------------------------------------------------------
# Volatility / rate schedules (step functions on the business-date axis)
# ---------------------------------------------------------------------------
class _Schedule:
    """Step-function schedule on business dates with exact cumulative integrals."""

    def __init__(self, values: np.ndarray, times: np.ndarray, regimes: List[str]):
        # values: sigma_d on [t_d, t_{d+1}); extended constant beyond the ends.
        self.values = np.asarray(values, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.regimes = regimes
        # cumulative integral of sigma^2 from times[0]
        dt = np.diff(self.times)
        self._cum = np.concatenate([[0.0], np.cumsum(self.values[:-1] ** 2 * dt)])
        self._total = self.values[-1] ** 2

    def value_at(self, t: float) -> float:
        idx = int(np.searchsorted(self.times, t, side="right") - 1)
        idx = min(max(idx, 0), len(self.values) - 1)
        return float(self.values[idx])

    def regime_at(self, t: float) -> str:
        idx = int(np.searchsorted(self.times, t, side="right") - 1)
        idx = min(max(idx, 0), len(self.regimes) - 1)
        return self.regimes[idx]

    def integral_squared(self, t0: float, t1: float) -> float:
        """Exact integral of sigma(u)^2 du over [t0, t1] for the step function."""
        if t1 <= t0:
            return 0.0
        return self._cum_integral(t1) - self._cum_integral(t0)

    def _cum_integral(self, t: float) -> float:
        # integral from times[0] to t
        if t <= self.times[0]:
            return self.values[0] ** 2 * (t - self.times[0])
        if t >= self.times[-1]:
            return self._cum[-1] + self._total * (t - self.times[-1])
        idx = int(np.searchsorted(self.times, t, side="right") - 1)
        base = self._cum[idx]
        return base + self.values[idx] ** 2 * (t - self.times[idx])


def _build_schedules(cfg: Dict[str, Any], times: np.ndarray, dates: pd.DatetimeIndex):
    n = len(times)
    vol_cfg = cfg["vol_schedule"]
    kind = vol_cfg["kind"]
    if kind == "constant":
        sigma = np.full(n, float(vol_cfg["value"]))
        regimes = ["constant"] * n
    elif kind == "linear_ramp":
        # Ramp between the first and last observation date (step function on
        # business days; the daily steps are the ground truth, not an
        # approximation of a smoother latent path).
        a = float(vol_cfg["from"])
        b = float(vol_cfg["to"])
        frac = times / times[-1]
        sigma = a + (b - a) * frac
        regimes = [f"ramp_{i:03d}" for i in range(n)]
    elif kind == "piecewise_constant":
        sigma = np.empty(n)
        regimes_list: List[str] = []
        segs = vol_cfg["segments"]
        total_seg = sum(int(s["n_dates"]) for s in segs)
        if total_seg != n:
            raise ValueError(
                f"piecewise_constant segment dates ({total_seg}) must sum to n_dates ({n})"
            )
        pos = 0
        for seg in segs:
            for _ in range(int(seg["n_dates"])):
                sigma[pos] = float(seg["sigma"])
                regimes_list.append(str(seg.get("regime", "segment")))
                pos += 1
        regimes = regimes_list
    else:
        raise ValueError(f"unknown vol_schedule kind: {kind}")

    rate_cfg = cfg["rate_schedule"]
    if rate_cfg["kind"] == "constant":
        rate = np.full(n, float(rate_cfg["value"]))
    else:
        raise ValueError(f"unknown rate_schedule kind: {rate_cfg['kind']}")

    return _Schedule(sigma, times, regimes), rate


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------
def integrated_variance(schedule: _Schedule, t0: float, t1: float) -> float:
    return schedule.integral_squared(t0, t1)


def generate_synthetic_dataset(
    synthetic_cfg: Dict[str, Any],
    seed: int,
    master_seed: int = 42,
    source_tag: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Generate one synthetic dataset (observations + date-level truth).

    Returns (observations_df, dates_df, generation_metadata).
    """
    cfg = synthetic_cfg
    n_dates = int(cfg["n_dates"])
    start = pd.Timestamp(cfg["start_date"])
    dates = pd.bdate_range(start=start, periods=n_dates)
    # Year-fraction axis (ACT/365.25 from the first date).
    times = np.array([(d - dates[0]).days / DAYS_PER_YEAR for d in dates])

    rng = np.random.default_rng(derive_seed(master_seed, "synthetic", cfg["scenario"], seed))

    # --- underlying spot process (independent of the pricing vol schedule) ---
    spot_cfg = cfg["spot"]
    s0 = float(spot_cfg["s0"])
    mu = float(spot_cfg["drift_annual"])
    spot_vol = float(spot_cfg["realized_vol_annual"])
    dt = np.diff(times)
    log_ret = (mu - 0.5 * spot_vol**2) * dt + spot_vol * np.sqrt(dt) * rng.standard_normal(n_dates - 1)
    spot = np.empty(n_dates, dtype=float)
    spot[0] = s0
    if n_dates > 1:
        spot[1:] = s0 * np.exp(np.cumsum(log_ret))

    vol_schedule, rate = _build_schedules(cfg, times, dates)

    # --- contract grid ---
    maturities = [float(m) for m in cfg["maturities"]]
    mny = cfg["moneyness"]
    strikes_rel = np.linspace(float(mny["min"]), float(mny["max"]), int(mny["n"]))

    noise_cfg = cfg.get("noise", {})
    noise_mode = noise_cfg.get("mode", "none")
    noise_std = float(noise_cfg.get("relative_std", 0.0))
    spread_cfg = cfg.get("spreads", {})
    spread_rel = float(spread_cfg.get("relative", 0.01))
    spread_abs = float(spread_cfg.get("absolute", 0.05))
    dropout = float(cfg.get("observation_dropout", 0.0) or 0.0)

    rows: List[Dict[str, Any]] = []
    for i in range(n_dates):
        t_i = times[i]
        s_i = float(spot[i])
        r_i = float(rate[i])
        sigma_inst = vol_schedule.value_at(t_i)
        regime = vol_schedule.regime_at(t_i)
        for tau_target in maturities:
            expiry_date = dates[i] + pd.Timedelta(days=int(round(tau_target * DAYS_PER_YEAR)))
            u_T = (expiry_date - dates[0]).days / DAYS_PER_YEAR
            tau = u_T - t_i
            if tau <= 0:
                continue
            A = vol_schedule.integral_squared(t_i, u_T)
            R = r_i * tau
            sigma_eff = math.sqrt(A / tau)
            for m in strikes_rel:
                K = s_i * float(m)
                true_price = float(call_price_total_variance(s_i, K, A, R, "call"))

                # observation noise
                flag = 0
                if noise_mode == "relative_gaussian" and noise_std > 0:
                    observed = true_price * (1.0 + noise_std * float(rng.standard_normal()))
                    if observed <= 0.0:
                        observed = 0.01
                        flag |= 1
                elif noise_mode == "none":
                    observed = true_price
                else:
                    raise ValueError(f"unknown noise mode: {noise_mode}")

                spread = max(spread_abs, spread_rel * true_price)
                bid = max(observed - spread / 2.0, 0.0)
                ask = observed + spread / 2.0
                mid = 0.5 * (bid + ask)  # equals observed by construction

                iv = implied_volatility(mid, s_i, K, tau, r_i)
                if iv.status != "ok":
                    flag |= 2
                if not (0.0 < mid <= s_i + 1e-9):
                    flag |= 4

                rows.append(
                    {
                        "timestamp": dates[i],
                        "underlying": cfg.get("underlying", "SYNTH"),
                        "spot": s_i,
                        "strike": K,
                        "expiration": expiry_date,
                        "tau": tau,
                        "option_type": cfg.get("option_type", "call"),
                        "exercise_style": "european",
                        "currency": cfg.get("currency", "USD"),
                        "rate": r_i,
                        "bid": bid,
                        "ask": ask,
                        "market_price": mid,
                        "implied_volatility": iv.sigma if iv.sigma is not None else np.nan,
                        "volume": np.nan,
                        "open_interest": np.nan,
                        "true_price": true_price,
                        "true_sigma": sigma_eff,
                        "true_sigma_instantaneous": sigma_inst,
                        "observed_price": mid,
                        "noise": mid - true_price,
                        "regime": regime,
                        "source": source_tag or f"synthetic:{cfg['scenario']}",
                        "quality_flag": flag,
                    }
                )

    obs = pd.DataFrame(rows, columns=COLUMNS)

    # optional dropout of observations (sparsity stress)
    if dropout > 0:
        keep = rng.random(len(obs)) >= dropout
        obs = obs.loc[keep].reset_index(drop=True)

    dates_df = pd.DataFrame(
        {
            "timestamp": dates,
            "spot": spot,
            "rate": rate,
            "sigma_instantaneous": [vol_schedule.value_at(t) for t in times],
            "regime": [vol_schedule.regime_at(t) for t in times],
            "t_yearfrac": times,
            "source": source_tag or f"synthetic:{cfg['scenario']}",
        }
    )

    meta = {
        "scenario": cfg["scenario"],
        "seed": seed,
        "master_seed": master_seed,
        "n_dates": n_dates,
        "n_observations": int(len(obs)),
        "n_expiries_per_date": len(maturities),
        "n_strikes_per_expiry": len(strikes_rel),
        "noise_mode": noise_mode,
        "noise_relative_std": noise_std if noise_mode != "none" else 0.0,
        "observation_dropout": dropout,
        "dropout_realized": float(1.0 - len(obs) / (n_dates * len(maturities) * len(strikes_rel))),
        "quality_flag_counts": obs["quality_flag"].value_counts().to_dict(),
        "dates": [str(d.date()) for d in dates[[0, -1]]],
    }
    return obs, dates_df, meta
