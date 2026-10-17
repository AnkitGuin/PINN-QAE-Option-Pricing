"""Calibration module tests: LS recovery, bounds, rolling/frozen engines."""

import numpy as np
import pytest

from experiments._common import scenario_synthetic_config
from src.black_scholes.analytical import bs_call
from src.calibration.frozen import fit_frozen_parameters
from src.calibration.least_squares import calibrate_least_squares
from src.calibration.rolling import FrozenEngine, RollingLSEngine, price_with_sigma
from src.data.synthetic_generator import generate_synthetic_dataset


@pytest.fixture(scope="module")
def dataset():
    cfg = scenario_synthetic_config("synthetic/constant.yaml")
    obs, _, _ = generate_synthetic_dataset(cfg, seed=0, master_seed=42)
    return obs


CAL_CFG = {
    "sigma_bounds": [0.01, 3.0],
    "ls": {"x_tolerance": 1e-10, "max_evaluations": 200, "warm_start": True},
    "rolling": {"window_dates": 20, "step_dates": 5, "horizon_dates": 5,
                "min_window_observations": 10},
    "frozen": {},
    "references": {},
}


class TestLeastSquares:
    def test_exact_recovery_on_clean_constant_data(self, dataset):
        window = dataset.head(1000)
        res = calibrate_least_squares(
            window["spot"], window["strike"], window["tau"], window["rate"],
            window["market_price"], sigma_bounds=(0.01, 3.0),
        )
        assert res.sigma == pytest.approx(0.2, abs=1e-8)

    def test_bounds_respected(self, dataset):
        window = dataset.head(500)
        lo, hi = 0.19, 0.21
        res = calibrate_least_squares(
            window["spot"], window["strike"], window["tau"], window["rate"],
            window["market_price"], sigma_bounds=(lo, hi),
        )
        assert lo - 1e-12 <= res.sigma <= hi + 1e-12

    def test_recovers_sigma_used_to_generate_prices(self):
        # prices generated with sigma=0.35 (not the default 0.2)
        rng = np.random.default_rng(0)
        S = rng.uniform(90, 110, 60)
        K = rng.uniform(85, 115, 60)
        tau = rng.uniform(0.1, 1.0, 60)
        r = np.full(60, 0.05)
        prices = bs_call(S, K, tau, r, 0.35)
        res = calibrate_least_squares(S, K, tau, r, prices, sigma_bounds=(0.01, 3.0))
        assert res.sigma == pytest.approx(0.35, abs=1e-7)


class TestRollingEngine:
    def test_walk_forward_calibrates_every_window(self, dataset):
        engine = RollingLSEngine(CAL_CFG)
        splitter = engine.splitter(dataset)
        events = splitter.events()
        assert len(events) >= 10
        sigmas = []
        for ev in events:
            window = dataset.iloc[ev.calibration_index]
            rec = engine.calibrate_window(window)
            assert rec.sigma is not None
            sigmas.append(rec.sigma)
        # constant scenario: every window recovers the true sigma
        assert np.max(np.abs(np.array(sigmas) - 0.2)) < 1e-6

    def test_warm_start_reproducible(self, dataset):
        engine = RollingLSEngine(CAL_CFG)
        splitter = engine.splitter(dataset)
        events = splitter.events()
        w0 = dataset.iloc[events[0].calibration_index]
        w1 = dataset.iloc[events[1].calibration_index]
        e2 = RollingLSEngine(CAL_CFG)
        e2.calibrate_window(w0)
        s1 = e2.calibrate_window(w1).sigma
        e3 = RollingLSEngine(CAL_CFG)
        e3.calibrate_window(w0)
        s2 = e3.calibrate_window(w1).sigma
        assert s1 == pytest.approx(s2, abs=1e-15)


class TestFrozenEngine:
    def test_frozen_sigma_constant_across_windows(self, dataset):
        frozen = FrozenEngine(CAL_CFG)
        records = frozen.run(dataset)
        assert len(records) >= 10
        fitted = [r for r in records if r.status == "success"]
        reused = [r for r in records if r.status == "reused_frozen"]
        assert len(fitted) == 1  # exactly one fit (first window)
        assert len(reused) == len(records) - 1
        assert all(r.sigma == fitted[0].sigma for r in reused)

    def test_frozen_parameters_fit(self, dataset):
        window = dataset.head(500)
        fp = fit_frozen_parameters(window, sigma_bounds=(0.01, 3.0))
        assert fp.sigma == pytest.approx(0.2, abs=1e-6)


class TestPriceWithSigma:
    def test_matches_analytical(self, dataset):
        block = dataset.tail(20)
        got = price_with_sigma(block, 0.2)
        want = bs_call(block["spot"], block["strike"], block["tau"], block["rate"], 0.2)
        assert np.allclose(got, want, atol=1e-12)
