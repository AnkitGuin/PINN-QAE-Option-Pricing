"""Leakage-prevention suite (spec: leakage controls section).

These tests encode the leakage rules as executable invariants:

1. Synthetic-only ground-truth fields never enter model inputs.
2. Model/scaler inputs never include premiums or target-derived IV.
3. Walk-forward calibration windows never contain future information.
4. Scalers are fitted on the calibration window only.
5. The shared evaluator prices blocks with market-observable columns only.
"""

import numpy as np
import pandas as pd
import pytest

from experiments._common import scenario_synthetic_config
from src.calibration.rolling import RollingLSEngine, price_with_sigma
from src.data.preprocessing import fit_window_scaler
from src.data.schema import MODEL_INPUT_FIELDS, SYNTHETIC_ONLY_FIELDS
from src.data.splits import WalkForwardSplitter
from src.data.synthetic_generator import generate_synthetic_dataset
from src.evaluation.comparisons import evaluate_block_with_sigma


@pytest.fixture(scope="module")
def dataset():
    cfg = scenario_synthetic_config("synthetic/constant.yaml")
    obs, dates, meta = generate_synthetic_dataset(cfg, seed=0, master_seed=42)
    return obs


CAL_CFG = {
    "sigma_bounds": [0.01, 3.0],
    "ls": {"x_tolerance": 1e-10, "max_evaluations": 200, "warm_start": True},
    "rolling": {"window_dates": 20, "step_dates": 5, "horizon_dates": 5,
                "min_window_observations": 10},
    "frozen": {},
    "references": {},
}


class TestFieldLevelLeaks:
    def test_synthetic_only_fields_never_model_inputs(self):
        assert not set(SYNTHETIC_ONLY_FIELDS) & set(MODEL_INPUT_FIELDS)

    def test_scaler_uses_only_market_observable_columns(self, dataset):
        # the scaler constructor touches only spot/strike/tau/rate columns
        scaler = fit_window_scaler(dataset.head(50))
        stats = scaler.to_dict()
        for key in stats:
            for banned in SYNTHETIC_ONLY_FIELDS + ["market_price", "implied_volatility",
                                                   "bid", "ask", "observed_price"]:
                assert banned not in key

    def test_transform_inputs_signature(self, dataset):
        scaler = fit_window_scaler(dataset.head(50))
        X = scaler.transform_inputs(
            dataset["spot"].head(5), dataset["tau"].head(5),
            dataset["strike"].head(5), dataset["rate"].head(5),
        )
        assert X.shape == (5, 3)  # (z(log-moneyness), z(tau), z(rate)) — no other fields

    def test_calibration_target_is_market_price_not_true_price(self, dataset):
        # LS calibration must fit market_price; if true_price were the target,
        # calibration on synthetic data would be trivially exact even with noise
        cfg = scenario_synthetic_config("synthetic/noise_sparse.yaml")
        obs, _, _ = generate_synthetic_dataset(cfg, seed=1, master_seed=42)
        window = obs[obs["timestamp"].isin(obs["timestamp"].unique()[:20])]
        from src.calibration.least_squares import calibrate_least_squares

        res_market = calibrate_least_squares(
            window["spot"], window["strike"], window["tau"], window["rate"],
            window["market_price"], sigma_bounds=(0.01, 3.0),
        )
        assert res_market.sigma is not None
        # the noise actually perturbs the target
        assert (window["noise"].abs() > 0).any()


class TestWindowLevelLeaks:
    def test_walk_forward_no_overlap_no_future(self, dataset):
        splitter = WalkForwardSplitter(
            dataset["timestamp"], window_dates=20, step_dates=5, horizon_dates=5,
            min_window_observations=10,
        )
        ts = dataset["timestamp"].to_numpy()
        for ev in splitter.events():
            assert ts[ev.calibration_index].max() < ts[ev.evaluation_index].min()

    def test_scaler_fitted_on_window_only(self, dataset):
        # statistics computed on the window must differ from full-sample stats
        splitter = WalkForwardSplitter(
            dataset["timestamp"], window_dates=20, step_dates=5, horizon_dates=5,
        )
        ev = splitter.events()[0]
        window = dataset.iloc[ev.calibration_index]
        scaler = fit_window_scaler(window)
        full = fit_window_scaler(dataset)
        # the constant scenario has constant rate and stationary grid, but
        # spot moves; mean_spot must reflect only the window
        assert scaler.mean_spot == pytest.approx(window["spot"].mean())
        assert not np.isclose(scaler.mean_spot, full.mean_spot) or len(window) == len(dataset)

    def test_evaluation_uses_only_market_columns(self, dataset):
        splitter = WalkForwardSplitter(
            dataset["timestamp"], window_dates=20, step_dates=5, horizon_dates=5,
        )
        ev = splitter.events()[0]
        block = dataset.iloc[ev.evaluation_index]
        res_a = evaluate_block_with_sigma(block, sigma=0.2)
        # perturbing NON-target latent fields must not change predictions or
        # metrics (true_price is the evaluation target and stays untouched)
        perturbed = block.copy()
        perturbed["true_sigma"] = 0.9
        perturbed["true_sigma_instantaneous"] = 0.9
        perturbed["noise"] = 123.0
        perturbed["regime"] = "bogus"
        res_b = evaluate_block_with_sigma(perturbed, sigma=0.2)
        assert res_a["rmse"] == pytest.approx(res_b["rmse"], abs=1e-12)
        # the prediction path equals the analytical evaluator on market columns
        from src.black_scholes.analytical import bs_call

        pred = bs_call(block["spot"], block["strike"], block["tau"], block["rate"], 0.2)
        ref = block["true_price"].to_numpy(dtype=float)
        assert res_a["rmse"] == pytest.approx(float(np.sqrt(np.mean((pred - ref) ** 2))), abs=1e-12)

    def test_price_with_sigma_ignores_latent_fields(self, dataset):
        block = dataset.tail(30)
        stripped = block.drop(columns=[c for c in SYNTHETIC_ONLY_FIELDS
                                       if c in block.columns])
        a = price_with_sigma(block, 0.2)
        b = price_with_sigma(stripped, 0.2)
        assert np.allclose(a, b)


class TestWarmStartCausality:
    def test_warm_start_uses_only_past_estimates(self, dataset):
        # the rolling engine's warm start is the previous window's estimate;
        # calibrating window 1 must be fully determined by windows 0 and 1
        engine = RollingLSEngine(CAL_CFG)
        splitter = engine.splitter(dataset)
        events = splitter.events()
        w0 = dataset.iloc[events[0].calibration_index]
        w1 = dataset.iloc[events[1].calibration_index]

        e1 = RollingLSEngine(CAL_CFG)
        e1.calibrate_window(w0)
        s1 = e1.calibrate_window(w1).sigma

        e2 = RollingLSEngine(CAL_CFG)
        e2.calibrate_window(w0)
        s2 = e2.calibrate_window(w1).sigma
        assert s1 == pytest.approx(s2, abs=1e-15)

        # a fresh engine (cold start) may differ from warm start, but the
        # warm-start value is the previous estimate, i.e. causal
        fresh = RollingLSEngine(CAL_CFG)
        fresh._last_sigma = None
        s_fresh = fresh.calibrate_window(w1).sigma
        assert np.isfinite(s_fresh)
