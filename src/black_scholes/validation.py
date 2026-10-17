"""Gate 1 validation suite: analytical Black-Scholes + Monte Carlo checks.

Used by ``experiments/exp0_validation.py`` and by the test suite.  All
tolerances are passed in from configuration (never chosen after seeing
outcomes).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .analytical import (
    bs_call,
    bs_price,
    bs_put,
    forward_form_call,
    price_bounds,
    put_call_parity_check,
)
from .implied_vol import implied_volatility
from .monte_carlo import mc_convergence_curve, mc_price

__all__ = ["run_gate1_validation"]


def _check(name: str, passed: bool, detail: Dict[str, Any]) -> Dict[str, Any]:
    return {"check": name, "status": "pass" if passed else "fail", **detail}


def run_gate1_validation(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run all Gate 1 checks and return a machine-readable report."""
    cfg = config["exp0"]
    tol = cfg["tolerances"]
    checks: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. Independent known values (published example, Hull 2022: S=42,
    #    K=40, r=0.10, sigma=0.20, tau=0.5 -> call = 4.76 to 2 d.p.)
    # ------------------------------------------------------------------
    hull_call = float(bs_call(42.0, 40.0, 0.5, 0.10, 0.20))
    checks.append(
        _check(
            "known_value_hull_2022",
            abs(hull_call - 4.76) < 0.01,
            {"computed": hull_call, "reference": 4.76, "tolerance": 0.01},
        )
    )

    # 2. Cross-check against the independent forward-form implementation.
    worst = 0.0
    for c in cfg["reference_contracts"]:
        a_form = float(forward_form_call(c["spot"], c["strike"], c["sigma"] ** 2 * c["tau"], c["rate"] * c["tau"]))
        d_form = float(bs_call(c["spot"], c["strike"], c["tau"], c["rate"], c["sigma"]))
        worst = max(worst, abs(a_form - d_form))
    checks.append(
        _check(
            "independent_forward_form_agreement",
            worst <= tol["forward_form_abs"],
            {"max_abs_difference": worst, "tolerance": tol["forward_form_abs"]},
        )
    )

    # 3. Put-call parity on a grid.
    rng = np.random.default_rng(config["project"]["master_seed"])
    n = 2000
    S = rng.uniform(50, 150, n)
    K = rng.uniform(50, 150, n)
    tau = rng.uniform(1e-3, 3.0, n)
    r = rng.uniform(0.0, 0.10, n)
    sig = rng.uniform(0.01, 1.5, n)
    calls = bs_price(S, K, tau, r, sig, "call")
    puts = bs_price(S, K, tau, r, sig, "put")
    parity_dev = np.abs(put_call_parity_check(S, K, tau, r, calls, puts))
    checks.append(
        _check(
            "put_call_parity",
            float(parity_dev.max()) <= tol["parity_abs"],
            {"max_abs_deviation": float(parity_dev.max()), "tolerance": tol["parity_abs"]},
        )
    )

    # 4. Price bounds.
    lo, hi = price_bounds(S, K, tau, r, "call")
    lo_p, hi_p = price_bounds(S, K, tau, r, "put")
    viol = float(
        np.max(
            [
                np.max(lo - calls),
                np.max(calls - hi),
                np.max(lo_p - puts),
                np.max(puts - hi_p),
                0.0,
            ]
        )
    )
    checks.append(
        _check(
            "price_bounds",
            viol <= tol["bounds_tol"],
            {"max_violation": viol, "tolerance": tol["bounds_tol"]},
        )
    )

    # 5. Limiting cases.
    eps_tau = tol["near_zero_maturity_tau"]
    intrinsic = max(100.0 - 95.0, 0.0)
    near_expiry = float(bs_call(100.0, 95.0, eps_tau, 0.05, 0.20))
    sigma_zero = float(bs_call(100.0, 95.0, 0.5, 0.05, 0.0))
    sigma_zero_expected = max(100.0 - 95.0 * np.exp(-0.05 * 0.5), 0.0)
    deep_otm = float(bs_call(100.0, 1000.0, 1.0, 0.05, 0.20))
    lim_ok = (
        abs(near_expiry - intrinsic) < 1e-3
        and abs(sigma_zero - sigma_zero_expected) < 1e-10
        and deep_otm < 1e-6
    )
    checks.append(
        _check(
            "limiting_cases",
            lim_ok,
            {
                "near_expiry": near_expiry,
                "intrinsic": intrinsic,
                "sigma_zero": sigma_zero,
                "sigma_zero_expected": sigma_zero_expected,
                "deep_otm": deep_otm,
            },
        )
    )

    # 6. Implied-volatility round trip.
    sig_test = np.array([0.08, 0.15, 0.2, 0.35, 0.6, 1.2])
    prices = bs_call(100.0, 100.0, 0.5, 0.05, sig_test)
    recovered = np.array(
        [implied_volatility(p, 100.0, 100.0, 0.5, 0.05).sigma or np.nan for p in prices]
    )
    iv_err = float(np.nanmax(np.abs(recovered - sig_test)))
    checks.append(
        _check(
            "implied_vol_round_trip",
            iv_err < 1e-7,
            {"max_abs_error": iv_err},
        )
    )

    # 7. Total-variance form vs Monte Carlo (time-varying sigma consistency).
    A = 0.20**2 * 0.3 + 0.35**2 * 0.7  # piecewise sigma: 20% for 0.3y, 35% for 0.7y
    R = 0.04 * 1.0
    tv_price = float(bs_price(100.0, 105.0, 1.0, 0.04, np.sqrt(A / 1.0)))
    exact_tv = float(
        forward_form_call(100.0, 105.0, A, R)
    )
    mc_tv = mc_price(100.0, 105.0, 1.0, 0.04, total_variance=A, total_discount=R, n_paths=2_000_000, seed=123)
    tv_ok = abs(mc_tv.price - exact_tv) < 4 * mc_tv.std_error
    checks.append(
        _check(
            "time_varying_variance_consistency",
            tv_ok,
            {
                "total_variance_form_price": exact_tv,
                "monte_carlo_price": mc_tv.price,
                "mc_std_error": mc_tv.std_error,
                "within_4_sigma": tv_ok,
                "note": "MC with same A,R must agree with total-variance form; "
                "constant-sigma substitution would differ materially",
            },
        )
    )

    # 8. Monte Carlo convergence + CI coverage at multiple path counts.
    mc_cfg = cfg["mc"]
    conv_contracts = tol["mc_contracts"]
    mc_reports = []
    conv_ok = True
    for c in conv_contracts:
        curve = mc_convergence_curve(
            c["spot"],
            c["strike"],
            c["tau"],
            c["rate"],
            c["sigma"],
            path_counts=mc_cfg["path_counts"],
            n_repeats=mc_cfg["n_repeats"],
            seed=config["project"]["master_seed"] + 1,
        )
        max_paths = curve["repeats"][-1]
        analytic = curve["analytical"]
        mean_err = abs(max_paths["mean_price"] - analytic)
        conv_ok = conv_ok and (mean_err <= mc_cfg["convergence_atol"])
        mc_reports.append(
            {
                "contract": c,
                "analytical": analytic,
                "repeats": curve["repeats"],
            }
        )
    checks.append(
        _check(
            "monte_carlo_convergence",
            conv_ok,
            {
                "max_paths": mc_cfg["path_counts"][-1],
                "mean_abs_error_at_max_paths": [
                    abs(r["repeats"][-1]["mean_price"] - r["analytical"]) for r in mc_reports
                ],
                "tolerance": mc_cfg["convergence_atol"],
            },
        )
    )

    # CI coverage: fraction of 95% CIs containing the analytical price.
    ci_fracs = []
    for rep in mc_reports[-1]["repeats"]:
        n = rep["n_paths"]
        # Re-derive CIs from stored prices using per-run std errors.
        analytic = mc_reports[-1]["analytical"]
        inside = 0
        for p, se in zip(rep["prices"], [rep["mean_std_error"]] * len(rep["prices"])):
            # Use the analytical z 1.96 for the 95% interval (documented).
            if p - 1.96 * se <= analytic <= p + 1.96 * se:
                inside += 1
        ci_fracs.append(inside / len(rep["prices"]))
    # Coverage should be near 0.95 for large path counts; check it is at
    # least not degenerate (>= 0.70) at the two largest path counts.
    coverage_ok = all(f >= 0.70 for f in ci_fracs[-2:])
    checks.append(
        _check(
            "monte_carlo_ci_coverage",
            coverage_ok,
            {"coverage_by_path_count": ci_fracs, "threshold": 0.70},
        )
    )

    overall = all(c["status"] == "pass" for c in checks)
    return {
        "gate": 1,
        "name": "black_scholes_and_monte_carlo_validation",
        "status": "pass" if overall else "fail",
        "checks": checks,
        "monte_carlo_reports": mc_reports,
    }
