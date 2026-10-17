"""Subprocess runner for circuit-level quantum checks (Gate 6).

Executed as ``python -m tests._quantum_runner <output.json>`` in its own
process.  This process imports Qiskit but never PyTorch, avoiding the
static-TLS (libgomp) conflict that arises when both frameworks load in one
process.  The parent test module (tests/test_quantum.py) asserts on the
JSON report this runner writes.

Each check returns (name, passed, detail).  The runner exits non-zero if any
check fails, so it can also be used standalone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def checks() -> list[tuple[str, bool, str]]:
    import numpy as np

    from src.quantum import (
        LognormalTerminalDistribution,
        build_qae_circuits,
        discretize,
        encode_call_payoff,
        exact_discrete_expectation,
        qa_outcome_distribution,
        run_qae,
    )
    from src.quantum.qae import build_a_operator_statevector, build_q_operator

    results: list[tuple[str, bool, str]] = []

    def record(name, cond, detail=""):
        results.append((name, bool(cond), str(detail)))

    S0, K, TAU, RATE, SIGMA = 100.0, 100.0, 0.5, 0.05, 0.20
    dist = LognormalTerminalDistribution.from_constant_parameters(S0, TAU, RATE, SIGMA)
    disc = discretize(dist, n_qubits=3, truncation_quantile=0.001)
    pay = encode_call_payoff(disc.grid, K)
    ex = exact_discrete_expectation(disc, pay, dist.total_discount)

    # --- A operator prepares the exact amplitudes ---
    from qiskit.quantum_info import Operator, Statevector

    A = build_a_operator_statevector(disc.n_qubits, disc.probabilities, pay.h)
    vec = np.asarray(Statevector(A).data)
    intended = np.zeros(2 * 2**disc.n_qubits, dtype=complex)
    for j in range(2**disc.n_qubits):
        intended[j] = np.sqrt(disc.probabilities[j] * (1 - pay.h[j]))
        intended[j + 2**disc.n_qubits] = np.sqrt(disc.probabilities[j] * pay.h[j])
    record("a_operator_exact_state", np.allclose(vec, intended, atol=1e-12),
           f"max|diff|={np.max(np.abs(vec - intended)):.2e}")

    # --- Q = -A S0 A^dagger S_chi rotates by exactly 2 theta ---
    Q = Operator(build_q_operator(A)).data
    theta = np.arcsin(np.sqrt(ex.a_exact))
    state = vec
    N = 2**disc.n_qubits
    worst = 0.0
    for t in range(4):
        good = float(np.sum(np.abs(state[N:]) ** 2))
        worst = max(worst, abs(good - np.sin((2 * t + 1) * theta) ** 2))
        state = Q @ state
    record("grover_rotation_identity", worst < 1e-10, f"max err={worst:.2e}")

    # --- global sign convention: circuit distribution matches the exact
    #     finite-sum likelihood derived for Brassard's Q (with the -1) ---
    from src.quantum.qae import canonical_qae_probability

    worst_cf = 0.0
    for m in (2, 3):
        bundle = build_qae_circuits(disc.n_qubits, disc.probabilities, pay.h, m)
        probs = qa_outcome_distribution(bundle)
        cf = np.array([canonical_qae_probability(k, theta, m) for k in range(2**m)])
        worst_cf = max(worst_cf, float(np.max(np.abs(probs - cf))))
        record(f"likelihood_matches_circuit_m{m}",
               float(np.max(np.abs(probs - cf))) < 1e-10, f"max diff={np.max(np.abs(probs - cf)):.2e}")
    record("global_sign_convention_brasard", worst_cf < 1e-10,
           "circuit implements canonical Brassard QAE (verified vs likelihood)")

    # --- statevector-mode QAE recovers the exact discrete amplitude ---
    for m in (2, 3, 4):
        res = run_qae(disc.probabilities, pay.h, n_evaluation=m, mode="statevector",
                      a_exact=ex.a_exact)
        record(f"qae_statevector_recovers_a_m{m}", abs(res.a_hat - ex.a_exact) < 1e-4,
               f"|a_hat - a|={abs(res.a_hat - ex.a_exact):.2e}")

    # --- shots-mode within shot noise and deterministic given seed ---
    res = run_qae(disc.probabilities, pay.h, n_evaluation=3, mode="shots",
                  shots=4096, simulator_seed=7, a_exact=ex.a_exact)
    record("qae_shots_within_noise", abs(res.a_hat - ex.a_exact) < 0.02,
           f"|a_hat - a|={abs(res.a_hat - ex.a_exact):.2e}, method={res.sampling_method}")
    again = run_qae(disc.probabilities, pay.h, n_evaluation=3, mode="shots",
                    shots=4096, simulator_seed=7, a_exact=ex.a_exact)
    record("qae_shots_deterministic_given_seed", res.a_hat == again.a_hat,
           f"{res.a_hat:.6f} vs {again.a_hat:.6f}")

    # --- qubit layout recorded ---
    bundle = build_qae_circuits(disc.n_qubits, disc.probabilities, pay.h, 3)
    lay = bundle.layout
    record("qubit_layout_recorded",
           lay["objective_qubit"] == 3 + disc.n_qubits and len(lay["evaluation_qubits"]) == 3,
           str(lay))

    return results


def main() -> int:
    out_path = Path(sys.argv[1])
    try:
        res = checks()
        payload = {
            "status": "ok",
            "checks": [{"name": n, "passed": p, "detail": d} for n, p, d in res],
        }
    except Exception as exc:  # pragma: no cover
        import traceback

        payload = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["status"]))
    if payload["status"] != "ok" or not all(c["passed"] for c in payload["checks"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
