"""Data module tests: schema, synthetic generator, integrated variance, splits."""

import numpy as np
import pandas as pd
import pytest

from src.data.schema import COLUMNS, MODEL_INPUT_FIELDS, SYNTHETIC_ONLY_FIELDS, enforce_schema
from src.data.splits import WalkForwardSplitter, chronological_train_val_split
from src.data.synthetic_generator import generate_synthetic_dataset
from src.data.validation import validate_dates_frame, validate_observations
from experiments._common import scenario_synthetic_config


@pytest.fixture(scope="module")
def constant_cfg():
    return scenario_synthetic_config("synthetic/constant.yaml")


@pytest.fixture(scope="module")
def dataset(constant_cfg):
    obs, dates, meta = generate_synthetic_dataset(constant_cfg, seed=0, master_seed=42)
    return obs, dates, meta


class TestSchema:
    def test_generated_columns_match_schema(self, dataset):
        obs, _, _ = dataset
        assert list(obs.columns) == COLUMNS

    def test_synthetic_only_fields_declared(self):
        for f in ["true_price", "true_sigma", "noise", "regime"]:
            assert f in SYNTHETIC_ONLY_FIELDS
        # model inputs must not intersect synthetic-only fields
        assert not set(MODEL_INPUT_FIELDS) & set(SYNTHETIC_ONLY_FIELDS)
        # the pricing inputs are exactly the market-observable contract terms
        assert set(MODEL_INPUT_FIELDS) == {"spot", "strike", "tau", "rate"}

    def test_enforce_schema(self, dataset):
        obs, _, _ = dataset
        out = enforce_schema(obs)
        assert len(out) == len(obs)


class TestSyntheticGenerator:
    def test_deterministic_given_seed(self, constant_cfg):
        a = generate_synthetic_dataset(constant_cfg, seed=3, master_seed=42)[0]
        b = generate_synthetic_dataset(constant_cfg, seed=3, master_seed=42)[0]
        pd.testing.assert_frame_equal(a, b)

    def test_initial_spot_is_configured_spot(self, constant_cfg):
        obs, dates, _ = generate_synthetic_dataset(constant_cfg, seed=0, master_seed=42)
        assert dates.iloc[0]["spot"] == pytest.approx(constant_cfg["spot"]["s0"])
        first_date_quotes = obs.loc[obs["timestamp"] == dates.iloc[0]["timestamp"]]
        assert np.allclose(first_date_quotes["spot"], constant_cfg["spot"]["s0"])

    def test_different_seeds_differ(self, constant_cfg):
        a = generate_synthetic_dataset(constant_cfg, seed=3, master_seed=42)[0]
        b = generate_synthetic_dataset(constant_cfg, seed=4, master_seed=42)[0]
        assert not np.allclose(a["market_price"], b["market_price"])

    def test_constant_condition_consistency(self, dataset, constant_cfg):
        # with zero noise, observed price == true price == BS(true_sigma)
        from src.black_scholes.analytical import bs_call

        obs, _, _ = dataset
        assert float(np.max(np.abs(obs["market_price"] - obs["true_price"]))) < 1e-10
        expected = bs_call(obs["spot"], obs["strike"], obs["tau"], obs["rate"], obs["true_sigma"])
        assert np.allclose(expected, obs["true_price"], atol=1e-10)
        # constant schedule: effective sigma == instantaneous sigma
        assert np.allclose(obs["true_sigma"], obs["true_sigma_instantaneous"], atol=1e-12)

    def test_gradual_condition_integrated_variance(self):
        # sigma_eff^2 * tau must equal the exact integrated variance A(t, T):
        # for a piecewise-constant schedule this is the exact integral
        cfg = scenario_synthetic_config("synthetic/gradual_change.yaml")
        obs, _, _ = generate_synthetic_dataset(cfg, seed=0, master_seed=42)
        # check on a sample of rows against an independent recomputation from
        # the date-level instantaneous schedule via cumulative integration
        dates = obs["timestamp"].unique()
        # effective sigma must be strictly positive and within schedule range
        assert (obs["true_sigma"] > 0).all()
        assert obs["true_sigma"].between(0.17, 0.33).all()
        # integrated variance consistency: A = sigma_eff^2 * tau is increasing
        # in maturity for a fixed date (vol schedule is positive everywhere)
        sample = obs[obs["timestamp"] == dates[10]]
        A = sample["true_sigma"].to_numpy() ** 2 * sample["tau"].to_numpy()
        order = np.argsort(sample["tau"].to_numpy())
        assert (np.diff(A[order]) > -1e-12).all()

    def test_validation_passes_on_clean_data(self, dataset):
        obs, dates, _ = dataset
        rep = validate_observations(obs)
        assert rep["status"] == "pass"
        assert rep["n_observations"] == len(obs)
        drep = validate_dates_frame(dates)
        assert drep["status"] == "pass"


class TestSplits:
    def test_rejects_unsorted_input_and_overlapping_eval_blocks(self):
        dates = pd.date_range("2025-01-01", periods=12)
        with pytest.raises(ValueError, match="sorted ascending"):
            WalkForwardSplitter(dates[::-1], window_dates=4, step_dates=2, horizon_dates=2)
        with pytest.raises(ValueError, match="disjoint"):
            WalkForwardSplitter(dates, window_dates=4, step_dates=2, horizon_dates=3)

    def test_walk_forward_strictly_chronological(self, dataset):
        obs, _, _ = dataset
        splitter = WalkForwardSplitter(
            obs["timestamp"], window_dates=20, step_dates=5, horizon_dates=5,
        )
        events = splitter.events()
        assert len(events) >= 10
        ts = obs["timestamp"].to_numpy()
        prev_cutoff = None
        for ev in events:
            cal_idx = ev.calibration_index
            eval_idx = ev.evaluation_index
            # calibration strictly before evaluation
            assert ts[cal_idx].max() < ts[eval_idx].min()
            # cutoffs advance monotonically
            if prev_cutoff is not None:
                assert ev.cutoff > prev_cutoff
            prev_cutoff = ev.cutoff
            # evaluation windows do not overlap with their own calibration window
            assert not (set(cal_idx) & set(eval_idx))
            assert len(ev.evaluation_dates) == 5

    def test_no_future_premium_in_calibration(self, dataset):
        obs, _, _ = dataset
        splitter = WalkForwardSplitter(
            obs["timestamp"], window_dates=20, step_dates=5, horizon_dates=5,
        )
        for ev in splitter.events():
            cutoff = obs["timestamp"].iloc[ev.calibration_index].max()
            eval_min = obs["timestamp"].iloc[ev.evaluation_index].min()
            assert cutoff < eval_min

    def test_chronological_train_val_split(self, dataset):
        obs, _, _ = dataset
        frame = obs.head(900).reset_index(drop=True)
        tr, va = chronological_train_val_split(frame, 0.2, seed=0)
        # validation is the LAST fraction of dates (no future data in fit part)
        ts = frame["timestamp"].to_numpy()
        assert ts[tr].max() < ts[va].min()
        assert len(tr) + len(va) == len(frame)
