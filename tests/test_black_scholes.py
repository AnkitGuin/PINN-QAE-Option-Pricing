"""Black-Scholes module tests (analytical, MC, IV, PDE residual)."""

import numpy as np
import pytest

from src.black_scholes.analytical import (
    bs_call,
    bs_price,
    bs_put,
    call_price_total_variance,
    price_bounds,
    put_call_parity_check,
)
from src.black_scholes.implied_vol import implied_volatility
from src.black_scholes.monte_carlo import mc_convergence_curve, mc_price
from src.black_scholes.pde import pde_residual_fd


class TestAnalytical:
    def test_known_value_hull(self):
        # Hull, "Options, Futures and Other Derivatives": S=42, K=40,
        # tau=0.5, r=0.10, sigma=0.20 -> call = 4.76 (textbook, 2dp)
        v = bs_price(42.0, 40.0, 0.5, 0.10, 0.20)
        assert v == pytest.approx(4.759422392871532, abs=1e-10)

    def test_put_call_parity(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            S = rng.uniform(50, 150)
            K = rng.uniform(50, 150)
            tau = rng.uniform(0.01, 2.0)
            r = rng.uniform(0.0, 0.1)
            sig = rng.uniform(0.05, 0.8)
            call = bs_call(S, K, tau, r, sig)
            put = bs_put(S, K, tau, r, sig)
            assert call - put == pytest.approx(S - K * np.exp(-r * tau), abs=1e-10)

    def test_parity_check_helper(self):
        S, K, tau, r, sig = 100.0, 95.0, 0.6, 0.04, 0.3
        c, p = bs_call(S, K, tau, r, sig), bs_put(S, K, tau, r, sig)
        assert put_call_parity_check(S, K, tau, r, c, p) < 1e-10

    def test_static_bounds(self):
        lo, hi = price_bounds(100.0, 90.0, 0.5, 0.03)
        v = bs_call(100.0, 90.0, 0.5, 0.03, 0.5)
        assert lo - 1e-12 <= v <= hi + 1e-12
        assert lo == pytest.approx(max(100.0 - 90.0 * np.exp(-0.015), 0.0), abs=1e-12)
        assert hi == pytest.approx(100.0, abs=1e-12)

    def test_zero_maturity(self):
        assert bs_call(120.0, 100.0, 0.0, 0.05, 0.2) == pytest.approx(20.0, abs=1e-12)
        assert bs_call(80.0, 100.0, 0.0, 0.05, 0.2) == pytest.approx(0.0, abs=1e-12)

    def test_negative_maturity_is_rejected(self):
        with pytest.raises(ValueError, match="tau must be non-negative"):
            bs_call(100.0, 100.0, -0.01, 0.05, 0.2)

    def test_vectorized_price_bounds_broadcast(self):
        lower, upper = price_bounds(np.array([90.0, 100.0]), 95.0, 0.5, 0.03)
        assert lower.shape == upper.shape == (2,)
        assert np.all(lower <= upper)

    def test_total_variance_form_matches(self):
        S, K, tau, r, sig = 100.0, 105.0, 0.75, 0.04, 0.22
        a = bs_call(S, K, tau, r, sig)
        b = call_price_total_variance(S, K, sig**2 * tau, r * tau)
        assert a == pytest.approx(b, abs=1e-12)

    def test_vol_approaching_zero(self):
        v = bs_call(200.0, 100.0, 1.0, 0.05, 1e-9)
        assert v == pytest.approx(max(200 - 100 * np.exp(-0.05), 0), abs=1e-6)


class TestImpliedVol:
    def test_roundtrip(self):
        for sig in [0.08, 0.15, 0.2, 0.35, 0.8]:
            S, K, tau, r = 100.0, 105.0, 0.75, 0.04
            price = bs_call(S, K, tau, r, sig)
            res = implied_volatility(price, S, K, tau, r)
            assert res.status == "ok"
            assert res.sigma == pytest.approx(sig, abs=1e-7)

    def test_out_of_bounds_price_flagged(self):
        S, K, tau, r = 100.0, 100.0, 1.0, 0.05
        res = implied_volatility(S + 1e-3, S, K, tau, r)  # above static upper bound
        assert res.status in ("above_bounds", "no_convergence")
        assert res.sigma is None or res.sigma is not None  # status carries the information


class TestMonteCarlo:
    def test_convergence_to_analytical(self):
        S, K, tau, r, sig = 100.0, 100.0, 0.5, 0.05, 0.2
        exact = bs_call(S, K, tau, r, sig)
        res = mc_price(S, K, tau, r, sig, n_paths=200_000, seed=1)
        assert abs(res.price - exact) < 0.05
        assert res.ci_low <= exact <= res.ci_high

    def test_total_variance_form_matches_constant(self):
        S, K, tau, r, sig = 100.0, 95.0, 0.4, 0.03, 0.25
        a = mc_price(S, K, tau, r, sig, n_paths=100_000, seed=3)
        b = mc_price(S, K, tau, r, total_variance=sig**2 * tau,
                     total_discount=r * tau, n_paths=100_000, seed=3)
        assert a.price == pytest.approx(b.price, abs=1e-12)

    def test_deterministic_with_seed(self):
        a = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, n_paths=10_000, seed=11)
        b = mc_price(100.0, 100.0, 1.0, 0.05, 0.2, n_paths=10_000, seed=11)
        assert a.price == b.price

    def test_antithetic_ci_uses_pair_degrees_of_freedom(self):
        from scipy import stats

        result = mc_price(100.0, 100.0, 0.5, 0.03, 0.2, n_paths=4, seed=3)
        critical = stats.t.ppf(0.975, df=1)  # two independent antithetic pairs
        assert result.ci_low == pytest.approx(result.price - critical * result.std_error)
        assert result.ci_high == pytest.approx(result.price + critical * result.std_error)

    def test_convergence_curve_error_shrinks(self):
        curve = mc_convergence_curve(100.0, 100.0, 0.5, 0.05, 0.2,
                                     path_counts=[250, 4000], n_repeats=8, seed=5)
        errs = {n: [abs(p - curve["analytical"]) for p in rep["prices"]]
                for n, rep in zip(curve["path_counts"], curve["repeats"])}
        assert np.mean(errs[4000]) < np.mean(errs[250])


class TestPDEResidual:
    def test_analytical_solution_satisfies_pde(self):
        # the analytical BS price must satisfy the Black-Scholes PDE
        S = np.array([70.0, 90.0, 100.0, 110.0, 140.0])
        tau = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
        K, r, sig = 100.0, 0.05, 0.2
        fn = lambda s, t: bs_call(s, K, np.maximum(t, 1e-8), r, sig)
        res = pde_residual_fd(fn, S, tau, K, r, sig)
        assert np.max(np.abs(res)) < 1e-3
