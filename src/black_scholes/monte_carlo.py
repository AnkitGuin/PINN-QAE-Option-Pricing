"""Classical Monte Carlo for the same risk-neutral model as the analytics.

    S_T = S_0 exp[ (r - sigma^2/2) tau + sigma sqrt(tau) Z ]   (constant params)
    S_T = S_0 exp[ R - A/2 + sqrt(A) Z ]                       (total variance form)

    V = exp(-R) E[(S_T - K)^+]

Supports configurable path counts, fixed seeds, antithetic variates, analytic
standard errors, t-based confidence intervals, repeated runs and convergence
curves.  Convergence toward the analytical price is verified in Experiment 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .analytical import bs_price

__all__ = ["mc_price", "mc_convergence_curve", "MCResult"]


@dataclass
class MCResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    n_paths: int
    seed: Optional[int]
    antithetic: bool
    undiscounted_payoff_mean: float

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def _t_crit_95(df: int) -> float:
    """Approximate two-sided 95% t-quantile (adequate for df >= 2)."""
    from scipy import stats

    return float(stats.t.ppf(0.975, df))


def mc_price(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: Optional[float] = None,
    n_paths: int = 100_000,
    seed: Optional[int] = None,
    antithetic: bool = True,
    option_type: str = "call",
    total_variance: Optional[float] = None,
    total_discount: Optional[float] = None,
    ci_level: float = 0.95,
) -> MCResult:
    """Monte Carlo price of a European option under the risk-neutral model.

    Either pass constant parameters (``sigma``) or the total-variance form
    (``total_variance`` / ``total_discount``) matching the analytical module.
    """
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if tau < 0:
        raise ValueError("tau must be non-negative")
    if n_paths < 2 or (antithetic and n_paths == 2):
        min_paths = 3 if antithetic else 2
        raise ValueError(f"n_paths must be at least {min_paths} for the selected sampling mode")
    if not 0.0 < ci_level < 1.0:
        raise ValueError("ci_level must be in (0, 1)")
    if total_variance is None:
        if sigma is None or sigma < 0:
            raise ValueError("sigma must be non-negative when total_variance is omitted")
        A = sigma**2 * tau
    else:
        A = float(total_variance)
    R = rate * tau if total_discount is None else float(total_discount)
    if not np.isfinite(A) or A < 0 or not np.isfinite(R):
        raise ValueError("total_variance must be non-negative and total_discount finite")

    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2 != 0:
            n_paths += 1
        half = n_paths // 2
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    s_t = spot * np.exp(R - 0.5 * A + np.sqrt(A) * z)
    if option_type == "call":
        payoff = np.maximum(s_t - strike, 0.0)
    elif option_type == "put":
        payoff = np.maximum(strike - s_t, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    disc = np.exp(-R)
    samples = disc * payoff
    price = float(samples.mean())
    # Standard error accounts for antithetic pairing (pair means are i.i.d.).
    if antithetic:
        pair_means = 0.5 * (samples[: half] + samples[half:])
        se = float(pair_means.std(ddof=1) / np.sqrt(half)) if half > 1 else float("nan")
    else:
        se = float(samples.std(ddof=1) / np.sqrt(n_paths)) if n_paths > 1 else float("nan")

    ci_df = half - 1 if antithetic else n_paths - 1
    if ci_level == 0.95:
        crit = _t_crit_95(ci_df)
    else:
        from scipy import stats

        crit = float(stats.t.ppf(0.5 + ci_level / 2.0, ci_df))

    return MCResult(
        price=price,
        std_error=se,
        ci_low=price - crit * se,
        ci_high=price + crit * se,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
        undiscounted_payoff_mean=float(payoff.mean()),
    )


def mc_convergence_curve(
    spot: float,
    strike: float,
    tau: float,
    rate: float,
    sigma: float,
    path_counts: List[int],
    n_repeats: int = 20,
    seed: Optional[int] = None,
    option_type: str = "call",
) -> Dict:
    """Repeated MC estimates for a sweep of path counts, with errors vs BS."""
    analytic = float(bs_price(spot, strike, tau, rate, sigma, option_type))
    curve = {"path_counts": list(path_counts), "analytical": analytic, "repeats": []}
    for j, n in enumerate(path_counts):
        prices, ses = [], []
        for rep in range(n_repeats):
            sub_seed = None if seed is None else seed + 1000 * j + rep
            res = mc_price(spot, strike, tau, rate, sigma, n_paths=n, seed=sub_seed, option_type=option_type)
            prices.append(res.price)
            ses.append(res.std_error)
        prices = np.asarray(prices)
        curve["repeats"].append(
            {
                "n_paths": n,
                "mean_price": float(prices.mean()),
                "std_across_repeats": float(prices.std(ddof=1)) if n_repeats > 1 else 0.0,
                "mean_abs_error": float(np.abs(prices - analytic).mean()),
                "rmse": float(np.sqrt(((prices - analytic) ** 2).mean())),
                "mean_std_error": float(np.mean(ses)),
                "prices": prices.tolist(),
            }
        )
    return curve
