"""Quantum module tests (Gate 6).

The pure-mathematical parts (distribution, discretization, payoff encoding,
likelihood, MLE, error decomposition) run in-process — they never import
Qiskit.  The circuit-level checks (A operator, Grover rotation, canonical-QAE
distribution and estimation) run in a dedicated SUBPROCESS
(tests/_quantum_runner.py) because PyTorch and Qiskit-Aer cannot both load
in one process on this platform (static-TLS/libgomp conflict); the quantum
runner imports Qiskit only, and the rest of the suite imports PyTorch only.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.black_scholes.analytical import bs_call
from src.evaluation.error_decomposition import decompose_quantum_errors
from src.quantum import (
    LognormalTerminalDistribution,
    discretize,
    encode_call_payoff,
    exact_discrete_expectation,
)
from src.quantum.qae import canonical_qae_probability, mle_amplitude

S0, K, TAU, RATE, SIGMA = 100.0, 100.0, 0.5, 0.05, 0.20
REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def setup():
    dist = LognormalTerminalDistribution.from_constant_parameters(S0, TAU, RATE, SIGMA)
    disc = discretize(dist, n_qubits=3, truncation_quantile=0.001)
    pay = encode_call_payoff(disc.grid, K)
    ex = exact_discrete_expectation(disc, pay, dist.total_discount)
    return dist, disc, pay, ex


class TestDistribution:
    def test_moments(self, setup):
        dist, _, _, _ = setup
        assert dist.mean() == pytest.approx(S0 * np.exp(RATE * TAU), rel=1e-12)
        assert dist.total_variance == pytest.approx(SIGMA**2 * TAU, rel=1e-12)
        assert dist.total_discount == pytest.approx(RATE * TAU, rel=1e-12)

    def test_cdf_at_zero_and_below(self, setup):
        dist, _, _, _ = setup
        assert float(np.asarray(dist.cdf(0.0))) == 0.0
        assert float(np.asarray(dist.cdf(-5.0))) == 0.0
        assert 0.0 < float(np.asarray(dist.cdf(S0))) < 1.0


class TestDiscretization:
    def test_probability_conservation(self, setup):
        _, disc, _, _ = setup
        assert disc.probabilities.sum() == pytest.approx(1.0, abs=1e-12)
        assert disc.tail_mass == pytest.approx(0.001, abs=1e-10)
        assert (disc.probabilities >= 0).all()

    def test_grid_matches_qubit_count(self, setup):
        _, disc, _, _ = setup
        assert len(disc.grid) == 2**disc.n_qubits == len(disc.probabilities)

    def test_qubit_bounds_enforced(self, setup):
        dist, _, _, _ = setup
        with pytest.raises(ValueError):
            discretize(dist, n_qubits=7)
        with pytest.raises(ValueError):
            discretize(dist, n_qubits=0)


class TestPayoff:
    def test_normalization_in_unit_interval(self, setup):
        _, disc, pay, _ = setup
        assert pay.h.min() >= 0.0
        assert pay.h.max() <= 1.0 + 1e-12
        # call payoff is monotone in S
        assert (np.diff(pay.h) >= -1e-12).all()

    def test_p_max_rule(self, setup):
        _, disc, pay, _ = setup
        assert pay.p_max == pytest.approx(disc.grid[-1] - K)
        # a P_max smaller than the maximum grid payoff must be rejected
        with pytest.raises(ValueError):
            encode_call_payoff(disc.grid, 90.0, p_max=1.0)

    def test_exact_discrete_expectation(self, setup):
        _, disc, pay, ex, = setup
        a = float(np.dot(disc.probabilities, pay.h))
        assert ex.a_exact == pytest.approx(a, rel=1e-12)
        assert ex.price == pytest.approx(np.exp(-RATE * TAU) * pay.p_max * a, rel=1e-12)
        # the discretized price is close to (but distinct from) BS
        assert abs(ex.price - bs_call(S0, K, TAU, RATE, SIGMA)) < 0.5


class TestLikelihoodMath:
    """Pure-numpy checks of the exact finite-sum likelihood (no circuits)."""

    def test_probability_normalisation(self, setup):
        _, _, _, ex = setup
        theta = float(np.arcsin(np.sqrt(ex.a_exact)))
        for m in (2, 3, 4):
            p = sum(canonical_qae_probability(k, theta, m) for k in range(2**m))
            assert p == pytest.approx(1.0, abs=1e-10)

    def test_peak_near_expected_theta(self, setup):
        # the likelihood should peak near the outcome closest to 2^m*theta/pi
        _, _, _, ex = setup
        m = 3
        theta = float(np.arcsin(np.sqrt(ex.a_exact)))
        probs = [canonical_qae_probability(k, theta, m) for k in range(2**m)]
        k_peak = int(np.argmax(probs))
        expected = 2**m * theta / np.pi
        # mirrored peak at 2^m - k is equally valid (both map to same a)
        assert min(abs(k_peak - expected), abs(2**m - k_peak - expected)) <= 1.0

    def test_mle_recovers_theta_on_exact_distribution(self, setup):
        _, _, _, ex = setup
        m = 3
        theta = float(np.arcsin(np.sqrt(ex.a_exact)))
        counts = {k: canonical_qae_probability(k, theta, m) * 1000 for k in range(2**m)}
        est = mle_amplitude(counts, m, grid_points=5001)
        assert est["a_hat"] == pytest.approx(ex.a_exact, abs=1e-3)

    def test_mle_flat_counts_still_finite(self):
        est = mle_amplitude({k: 10.0 for k in range(8)}, 3, grid_points=2001)
        assert 0.0 <= est["a_hat"] <= 1.0


class TestErrorDecomposition:
    def test_components_sum_to_total(self):
        dec = decompose_quantum_errors(
            v_reference=6.89, v_model=6.71, v_exact_discrete=6.98, v_qae=6.95,
            sigma_reference=0.20, sigma_input=0.22,
        )
        assert dec.calibration_error + dec.discretization_error + dec.qae_estimation_error \
            == pytest.approx(dec.total_error, abs=1e-12)
        assert dec.total_error == pytest.approx(6.95 - 6.89, abs=1e-12)


class TestCircuitsViaSubprocess:
    """Circuit-level Gate-6 checks executed in a Qiskit-only subprocess."""

    @pytest.fixture(scope="class")
    def report(self, tmp_path_factory):
        out = tmp_path_factory.mktemp("qrunner") / "report.json"
        proc = subprocess.run(
            [sys.executable, "-m", "tests._quantum_runner", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
        )
        assert out.exists(), f"runner produced no report: {proc.stdout}\n{proc.stderr}"
        return json.loads(out.read_text())

    def test_runner_completed(self, report):
        assert report["status"] == "ok", report.get("error", "") + report.get("traceback", "")

    def test_all_circuit_checks_pass(self, report):
        failed = [c for c in report["checks"] if not c["passed"]]
        assert not failed, json.dumps(failed, indent=2)

    def test_expected_checks_present(self, report):
        names = {c["name"] for c in report["checks"]}
        for expected in (
            "a_operator_exact_state",
            "grover_rotation_identity",
            "global_sign_convention_brasard",
            "qae_statevector_recovers_a_m3",
            "qae_shots_within_noise",
            "qae_shots_deterministic_given_seed",
            "qubit_layout_recorded",
        ):
            assert expected in names
