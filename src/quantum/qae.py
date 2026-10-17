"""Canonical Quantum Amplitude Estimation (Brassard et al.) on a simulator.

Construction
------------
A-operator (n grid qubits + 1 objective qubit), implemented exactly:

    A|0...0> = sum_j sqrt(p_j) |j> ( sqrt(1-h_j)|0> + sqrt(h_j)|1> )

so the objective-qubit amplitude equals a = sum_j p_j h_j EXACTLY for the
given discretization and payoff (any payoff shape; no piecewise-linear
restriction).  The cost of this state-preparation-based encoding is reported
honestly through transpiled resource accounting (see resources.py).

Grover operator:

    Q = A S_0 A^dagger S_chi,
    S_chi = Z on the objective qubit (sign-flip of good states),
    S_0   = X^{n+1} . MCZ . X^{n+1} (sign-flip of the all-zero state).

Canonical QAE with m evaluation qubits: H layer, A, controlled-Q^{2^k}
(k = 0..m-1), inverse QFT, measurement.  Outcome k has the closed-form
likelihood (Brassard et al. 2002):

    P(k | theta) = sin^2(2^m theta) / ( 2^{2m} sin^2(theta - pi k / 2^m) )

with a = sin^2(theta).  Post-processing is maximum likelihood over a theta
grid (deterministic given counts).

Modes:
    statevector : exact circuit outcome probabilities (deterministic; used for
                  Gate 6 validation and as the noiseless reference)
    shots       : finite-shot sampling.  Counts are drawn either by the ideal
                  Aer simulator (seeded) or, when Aer sampling of the deep
                  controlled-StatePreparation circuit is impractically slow,
                  by seeded multinomial sampling from the exact statevector
                  distribution (mathematically identical to an ideal
                  noiseless simulator; the mode actually used is recorded).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .qiskit_compat import import_qiskit

__all__ = ["QAECircuitBundle", "build_qae_circuits", "qa_outcome_distribution", "mle_amplitude", "run_qae"]

_TINY = 1e-300


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------
def build_a_operator_statevector(n_qubits: int, probabilities: np.ndarray, h: np.ndarray):
    """Return a QuantumCircuit implementing A on (grid, objective) qubits."""
    qiskit = import_qiskit()
    from qiskit.circuit.library import StatePreparation

    p = np.asarray(probabilities, dtype=float)
    h = np.asarray(h, dtype=float)
    N = 2**n_qubits
    if len(p) != N or len(h) != N:
        raise ValueError("probabilities and h must have length 2^n_qubits")
    # Qiskit little-endian convention: state index = j + b * 2^n_qubits, i.e.
    # qubits 0..n-1 carry the grid integer j and qubit n is the objective.
    amp_good = np.sqrt(p * h)
    amp_bad = np.sqrt(p * (1.0 - h))
    state = np.zeros(2 * N, dtype=complex)
    state[:N] = amp_bad    # objective qubit = 0  -> indices [0, N)
    state[N:] = amp_good   # objective qubit = 1  -> indices [N, 2N)
    norm = np.linalg.norm(state)
    if abs(norm - 1.0) > 1e-9:
        state = state / norm
    qc = qiskit.QuantumCircuit(n_qubits + 1, name="A")
    qc.append(StatePreparation(state), range(n_qubits + 1))
    return qc


def build_q_operator(a_circuit):
    """Q = -A S0 A^dagger S_chi (Brassard et al.) on (grid, objective) qubits.

    The global minus sign is essential: without it the operator's eigenphases
    are pi +/- 2theta instead of +/- 2theta, and canonical QAE would estimate
    1 - a.  It is included via the circuit's global phase, which Qiskit
    preserves through ``to_gate().control()`` (verified in
    tests/test_quantum.py).
    """
    qiskit = import_qiskit()
    n_total = a_circuit.num_qubits  # n grid + 1 objective
    n_grid = n_total - 1
    obj = n_total - 1

    qc = qiskit.QuantumCircuit(n_total, name="Q")
    # S_chi: flip sign of good states (objective = |1>)
    qc.z(obj)
    # A^dagger
    qc.compose(a_circuit.inverse(), inplace=True)
    # S_0: flip sign of |0...0>
    qc.x(range(n_total))
    qc.h(obj)
    qc.mcx(list(range(n_grid)), obj)
    qc.h(obj)
    qc.x(range(n_total))
    # A
    qc.compose(a_circuit, inplace=True)
    # global minus sign (Brassard convention)
    qc.global_phase = np.pi
    return qc


# ---------------------------------------------------------------------------
# Canonical QAE circuit
# ---------------------------------------------------------------------------
@dataclass
class QAECircuitBundle:
    a_circuit: Any
    q_circuit: Any
    full_circuit: Any
    n_grid: int
    n_evaluation: int
    layout: Dict[str, Any]

    @property
    def total_qubits(self) -> int:
        return self.full_circuit.num_qubits


def build_qae_circuits(
    n_grid: int,
    probabilities: np.ndarray,
    h: np.ndarray,
    n_evaluation: int,
) -> QAECircuitBundle:
    qiskit = import_qiskit()
    from qiskit.circuit.library import QFT

    a = build_a_operator_statevector(n_grid, probabilities, h)
    q = build_q_operator(a)

    m = int(n_evaluation)
    n_total = n_grid + 1
    total = m + n_total
    # layout: eval qubits [0, m), grid [m, m+n_grid), objective [m+n_grid]
    qc = qiskit.QuantumCircuit(total, m)
    qc.h(range(m))
    qc.compose(a, qubits=list(range(m, total)), inplace=True)

    q_gate = q.to_gate(label="Q")
    for k in range(m):
        power = 2**k
        if power == 1:
            controlled = q_gate.control(1)
        else:
            q_power = qiskit.QuantumCircuit(n_total, name=f"Q^{power}")
            for _ in range(power):
                q_power.append(q_gate, range(n_total))
            controlled = q_power.to_gate(label=f"Q^{power}").control(1)
        qc.append(controlled, [k] + list(range(m, total)))

    qc.compose(QFT(m, inverse=True, do_swaps=True), qubits=list(range(m)), inplace=True)
    for k in range(m):
        qc.measure(k, k)

    return QAECircuitBundle(
        a_circuit=a,
        q_circuit=q,
        full_circuit=qc,
        n_grid=n_grid,
        n_evaluation=m,
        layout={
            "evaluation_qubits": list(range(m)),
            "grid_qubits": list(range(m, m + n_grid)),
            "objective_qubit": m + n_grid,
            "total_qubits": total,
        },
    )


# ---------------------------------------------------------------------------
# Outcome distributions and estimation
# ---------------------------------------------------------------------------
def qa_outcome_distribution(bundle: QAECircuitBundle) -> np.ndarray:
    """Exact probability distribution over the m-bit measured register."""
    qiskit = import_qiskit()
    from qiskit.quantum_info import Statevector

    no_meas = bundle.full_circuit.remove_final_measurements(inplace=False)
    sv = Statevector(no_meas)
    m = bundle.n_evaluation
    probs = np.asarray(sv.probabilities(qargs=list(range(m))), dtype=float)
    return probs  # index = measured integer k


_PHASE_MATRIX_CACHE: Dict[int, np.ndarray] = {}


def _phase_matrix(m: int) -> np.ndarray:
    """E[k, l] = exp(-2 pi i k l / 2^m) (cached)."""
    if m not in _PHASE_MATRIX_CACHE:
        L = 2**m
        k = np.arange(L)[:, None]
        l = np.arange(L)[None, :]
        _PHASE_MATRIX_CACHE[m] = np.exp(-2j * np.pi * k * l / L)
    return _PHASE_MATRIX_CACHE[m]


def canonical_qae_probability(k: int, theta: float, m: int) -> float:
    """Exact P(k | theta) for canonical QAE, derived from the circuit.

    After the H layer and A, the evaluation register holds
    sin((2l+1)theta)|good> + cos((2l+1)theta)|bad> in branch |l> (Grover
    amplification identity, verified numerically in tests/test_quantum.py).
    The inverse QFT maps branch l to outcome k with phase e^{-2 pi i k l/2^m},
    so

        P(k | theta) = |S_k|^2 + |C_k|^2,
        S_k = (1/2^m) sum_l sin((2l+1)theta) e^{-2 pi i k l / 2^m}
        C_k = (1/2^m) sum_l cos((2l+1)theta) e^{-2 pi i k l / 2^m}

    This exact finite-sum form matches the simulated circuit distribution to
    ~1e-14 (tests/test_quantum.py) and is what the MLE uses.
    """
    L = 2**m
    l = np.arange(L)
    phase = np.exp(-2j * np.pi * k * l / L)
    S = np.sum(np.sin((2 * l + 1) * theta) * phase) / L
    C = np.sum(np.cos((2 * l + 1) * theta) * phase) / L
    return float(abs(S) ** 2 + abs(C) ** 2)


def _likelihood_matrix(m: int, thetas: np.ndarray) -> np.ndarray:
    """P[k, t] over outcomes k x theta grid, exact finite-sum form."""
    L = 2**m
    E = _phase_matrix(m)  # (L, L): E[k, l]
    l = np.arange(L)
    # ang[l, t]
    ang = (2 * l + 1)[:, None] * thetas[None, :]
    # This contraction is tiny (L <= 64 for the supported QAE budgets). Use
    # NumPy's generic einsum loop instead of dispatching to vendor BLAS: on
    # some Windows builds, BLAS loaded after PyTorch's MKL runtime aborts the
    # process even though the QAE code itself does not use PyTorch.
    S = np.einsum("kl,lt->kt", E, np.sin(ang), optimize=False) / L
    C = np.einsum("kl,lt->kt", E, np.cos(ang), optimize=False) / L
    P = np.abs(S) ** 2 + np.abs(C) ** 2
    return np.clip(P, 1e-300, None)


def mle_amplitude(counts: Dict[int, float], m: int, grid_points: int = 20001) -> Dict[str, Any]:
    """Maximum-likelihood amplitude from (possibly fractional) outcome counts."""
    thetas = np.linspace(1e-9, np.pi / 2 - 1e-9, grid_points)
    P = _likelihood_matrix(m, thetas)  # (2^m, T)
    P = P / P.sum(axis=0, keepdims=True)

    c = np.zeros(2**m)
    for k, v in counts.items():
        c[int(k)] += v
    total = c.sum()
    if total <= 0:
        raise ValueError("empty counts")
    loglik = np.einsum("k,kt->t", c / total, np.log(P), optimize=False)
    theta_hat = float(thetas[int(np.argmax(loglik))])
    a_hat = float(np.sin(theta_hat) ** 2)
    return {
        "theta_hat": theta_hat,
        "a_hat": a_hat,
        "m": m,
        "grid_points": grid_points,
        "counts": {int(k): float(v) for k, v in counts.items()},
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------
@dataclass
class QAERunResult:
    a_hat: float
    a_exact: Optional[float]
    mode: str
    shots: Optional[int]
    counts: Dict[int, float]
    m: int
    n_grid: int
    runtime_seconds: float
    sampling_method: str  # "statevector_exact" | "aer_simulator" | "multinomial_from_statevector"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def run_qae(
    probabilities: np.ndarray,
    h: np.ndarray,
    n_evaluation: int,
    mode: str = "shots",
    shots: int = 4096,
    simulator_seed: int = 7,
    theta_grid_points: int = 20001,
    a_exact: Optional[float] = None,
    return_bundle: bool = False,
):
    """Run canonical QAE and return the estimated amplitude a_hat."""
    n_grid = int(np.log2(len(probabilities)))
    t0 = time.perf_counter()
    bundle = build_qae_circuits(n_grid, probabilities, h, n_evaluation)
    t_build = time.perf_counter() - t0

    notes = [f"circuit build: {t_build:.3f}s"]

    if mode == "statevector":
        probs = qa_outcome_distribution(bundle)
        counts = {k: float(v) for k, v in enumerate(probs)}
        sampling = "statevector_exact"
        est = mle_amplitude(counts, n_evaluation, theta_grid_points)
        runtime = time.perf_counter() - t0
    elif mode == "shots":
        # Ideal (noiseless) Aer simulator with a fixed seed; falls back to
        # seeded multinomial sampling from the exact statevector outcome
        # distribution (mathematically identical to an ideal simulator) if
        # Aer is unavailable.  The method actually used is recorded.
        counts = None
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile as _transpile

            sim = AerSimulator(method="statevector")
            compiled = _transpile(bundle.full_circuit, backend=sim, optimization_level=0)
            result = sim.run(compiled, shots=shots, seed_simulator=simulator_seed).result()
            raw = result.get_counts()
            counts = {}
            for k, v in raw.items():
                counts[int(k.replace(" ", ""), 2)] = float(v)
            sampling = "aer_simulator"
        except Exception as exc:  # pragma: no cover
            notes.append(f"Aer unavailable ({exc}); using multinomial fallback")
        if counts is None:
            probs = qa_outcome_distribution(bundle)
            rng = np.random.default_rng(simulator_seed)
            draw = rng.multinomial(shots, probs / probs.sum())
            counts = {k: float(v) for k, v in enumerate(draw) if v > 0}
            sampling = "multinomial_from_statevector"
        est = mle_amplitude(counts, n_evaluation, theta_grid_points)
        runtime = time.perf_counter() - t0
    else:
        raise ValueError(f"unknown QAE mode: {mode}")

    result = QAERunResult(
        a_hat=est["a_hat"],
        a_exact=a_exact,
        mode=mode,
        shots=shots if mode == "shots" else None,
        counts=est["counts"],
        m=n_evaluation,
        n_grid=n_grid,
        runtime_seconds=runtime,
        sampling_method=sampling,
        notes=notes,
    )
    if return_bundle:
        return result, bundle
    return result
