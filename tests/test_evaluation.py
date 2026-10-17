"""Evaluation module tests: metrics, uncertainty, comparisons, reporting."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.calibration_metrics import (
    adaptation_lag,
    sigma_error_metrics,
    stability_metrics,
)
from src.evaluation.comparisons import evaluate_block_with_sigma
from src.evaluation.error_decomposition import decompose_quantum_errors
from src.evaluation.metrics import maturity_moneyness_buckets, pricing_metrics
from src.evaluation.reporting import comparison_statement, hypothesis_report
from src.evaluation.uncertainty import paired_block_differences, seed_variability


class TestMetrics:
    def test_pricing_metrics_values(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        ref = np.array([1.5, 2.0, 2.5, 5.0])
        m = pricing_metrics(pred, ref)
        e = pred - ref
        assert m["rmse"] == pytest.approx(float(np.sqrt(np.mean(e**2))))
        assert m["mae"] == pytest.approx(float(np.mean(np.abs(e))))
        assert m["bias"] == pytest.approx(float(np.mean(e)))

    def test_evaluate_block_with_sigma(self):
        from src.black_scholes.analytical import bs_call

        block = pd.DataFrame({
            "spot": [100.0, 100.0, 100.0],
            "strike": [95.0, 100.0, 105.0],
            "tau": [0.5, 0.5, 0.5],
            "rate": [0.05, 0.05, 0.05],
            "true_price": bs_call([100.0] * 3, [95.0, 100.0, 105.0], 0.5, 0.05, 0.25),
        })
        res = evaluate_block_with_sigma(block, sigma=0.25)
        assert res["rmse"] == pytest.approx(0.0, abs=1e-12)
        assert res["n_contracts"] == 3

    def test_evaluate_block_with_invalid_sigma(self):
        block = pd.DataFrame({
            "spot": [100.0], "strike": [100.0], "tau": [0.5],
            "rate": [0.05], "true_price": [5.0],
        })
        res = evaluate_block_with_sigma(block, sigma=float("nan"))
        assert np.isnan(res["rmse"])

    def test_buckets_partition_contracts(self):
        n = 90
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            "spot": 100.0,
            "strike": rng.uniform(80, 120, n),
            "tau": np.tile([0.1, 0.25, 0.5, 0.75, 1.0, 1.5], 15),
            "rate": 0.05,
            "true_price": rng.uniform(1, 10, n),
            "market_price": rng.uniform(1, 10, n),
        })
        out = maturity_moneyness_buckets(frame)
        assert len(out) == n
        assert {"maturity_bucket", "moneyness_bucket"}.issubset(out.columns)
        assert out["maturity_bucket"].notna().all()
        assert out["moneyness_bucket"].notna().all()


class TestCalibrationMetrics:
    def test_sigma_error_metrics_zero_for_perfect(self):
        m = sigma_error_metrics([0.2, 0.2, 0.2], {"instantaneous": [0.2, 0.2, 0.2],
                                                  "effective_block": [0.2, 0.2, 0.2]})
        assert m["abs_error_vs_instantaneous"]["mean"] == pytest.approx(0.0, abs=1e-15)
        assert m["n_windows"] == 3

    def test_sigma_error_nonzero_for_wrong(self):
        m = sigma_error_metrics([0.3, 0.3], {"instantaneous": [0.2, 0.2],
                                             "effective_block": [0.2, 0.2]})
        assert m["abs_error_vs_instantaneous"]["mean"] == pytest.approx(0.1)
        assert m["rel_error_vs_instantaneous"]["mean"] == pytest.approx(0.5)

    def test_stability_metrics(self):
        s = stability_metrics([0.2, 0.2, 0.2, 0.2])
        assert s["std_across_windows"] == pytest.approx(0.0)
        assert s["max_abs_change"] == pytest.approx(0.0)

    def test_adaptation_lag(self):
        est = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.35, 0.35]
        ref = [0.2, 0.2, 0.2, 0.2, 0.2, 0.35, 0.35, 0.35, 0.35]
        out = adaptation_lag(est, ref, list(range(len(est))))
        assert "mae_vs_current_instantaneous" in out
        assert "mae_vs_next_event_instantaneous" in out
        assert out["mae_vs_current_instantaneous"] > 0  # estimates trail the truth
        # for a perfectly tracking estimator both MAEs are zero
        perfect = adaptation_lag(ref, ref, list(range(len(ref))))
        assert perfect["mae_vs_current_instantaneous"] == pytest.approx(0.0)


class TestUncertainty:
    def test_paired_differences_zero_for_identical(self):
        a = [0.1, 0.2, 0.3, 0.4]
        out = paired_block_differences(a, a)
        assert out["mean_difference"] == pytest.approx(0.0)
        assert out["mean_diff_ci95_normal_approx"][0] <= 0.0 \
            <= out["mean_diff_ci95_normal_approx"][1]

    def test_paired_differences_sign(self):
        a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        b = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        out = paired_block_differences(a, b)
        assert out["mean_difference"] == pytest.approx(-0.1)
        assert out["mean_diff_ci95_normal_approx"][1] < 0.0  # significantly below zero
        assert out["fraction_blocks_a_larger"] == 0.0

    def test_seed_variability(self):
        out = seed_variability([1.0, 1.1, 0.9, 1.05])
        assert out["std"] == pytest.approx(float(np.std([1.0, 1.1, 0.9, 1.05], ddof=1)))
        assert "mean" in out and "min" in out and "max" in out


class TestReporting:
    def test_hypothesis_report_structure(self):
        rep = hypothesis_report(
            hypothesis="H1: PINN RMSE < LS RMSE under changing volatility",
            observed={"pinn_rmse": 0.12, "ls_rmse": 0.15},
            interpretation="Observed difference favours PINN in this setting.",
            limitation="Chronological blocks, small n, one data-generating process.",
            supports=True,
        )
        assert rep["hypothesis"].startswith("H1")
        assert rep["observed_result"]["pinn_rmse"] == 0.12
        assert rep["supports_hypothesis"] is True
        assert "limitation" in rep

    def test_comparison_statement_no_unevidenced_claim(self):
        s = comparison_statement(
            metric_a=0.12, metric_b=0.15, name_a="PINN", name_b="LS",
            metric_name="pooled RMSE",
        )
        assert "PINN" in s and "LS" in s
        assert "superior" not in s.lower()
        assert "observed result" in s

    def test_error_decomposition_identity(self):
        dec = decompose_quantum_errors(7.0, 7.1, 7.2, 7.15)
        assert dec.calibration_error == pytest.approx(0.1)
        assert dec.discretization_error == pytest.approx(0.1)
        assert dec.qae_estimation_error == pytest.approx(-0.05)
        assert dec.total_error == pytest.approx(0.15)
