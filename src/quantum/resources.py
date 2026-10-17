"""Quantum resource accounting.

Reports circuit resources for state preparation (A), the Grover operator (Q)
and the full canonical-QAE circuit, plus oracle/query counts and wall-clock
times.  Resource accounting covers state preparation and payoff encoding (both
live inside A here), not only amplitude queries.  The StatePreparation-based A
operator is exact for arbitrary payoffs but NOT shallow; this cost is reported
transparently rather than hidden.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from .qiskit_compat import import_qiskit
from .qae import QAECircuitBundle, build_qae_circuits

__all__ = ["resource_report", "aer_smoke_run"]


def _transpiled_stats(circuit, optimization_level: int = 1) -> Dict[str, Any]:
    qiskit = import_qiskit()
    from qiskit import transpile

    t0 = time.perf_counter()
    # transpile to a concrete gate basis so custom gates (StatePreparation,
    # controlled-Q powers) are fully decomposed and depth/size are meaningful
    tq = transpile(
        circuit,
        basis_gates=["u", "cx"],
        optimization_level=optimization_level,
        seed_transpiler=7,
    )
    dt = time.perf_counter() - t0
    ops = dict(tq.count_ops())
    return {
        "qubits": int(tq.num_qubits),
        "depth": int(tq.depth()),
        "size": int(tq.size()),
        "gate_counts": {str(k): int(v) for k, v in ops.items()},
        "transpilation_seconds": dt,
    }


def resource_report(
    bundle: QAECircuitBundle,
    shots: int,
    optimization_level: int = 1,
) -> Dict[str, Any]:
    m = bundle.n_evaluation
    grover_applications_per_shot = 2**m - 1
    # Each shot applies A once to prepare the state and each Grover iterate
    # contains one A and one A-dagger. The inverse calls have the same
    # state-preparation cost and must be included in resource totals.
    a_calls_per_shot = 1 + 2 * grover_applications_per_shot
    return {
        "layout": bundle.layout,
        "n_grid_qubits": bundle.n_grid,
        "n_evaluation_qubits": m,
        "total_circuit_qubits": bundle.total_qubits,
        "state_preparation_a": _transpiled_stats(bundle.a_circuit, optimization_level),
        "grover_q": _transpiled_stats(bundle.q_circuit, optimization_level),
        "full_qae_circuit": _transpiled_stats(bundle.full_circuit, optimization_level),
        "shots": int(shots),
        "grover_applications_per_shot": grover_applications_per_shot,
        "total_grover_queries": int(shots * grover_applications_per_shot),
        "initial_a_applications": int(shots),
        "total_a_or_a_dagger_applications": int(shots * a_calls_per_shot),
        "theoretical_theta_resolution": float(np.pi / 2 ** (m + 1)),
        "note": (
            "A embeds distribution loading AND payoff encoding exactly via "
            "StatePreparation; its transpiled cost is reported and is large. "
            "Simulator resources do not establish hardware quantum advantage."
        ),
    }


def aer_smoke_run(bundle: QAECircuitBundle, shots: int, seed: int) -> Dict[str, Any]:
    """Execute the full QAE circuit on the ideal Aer simulator (seeded).

    Used for resource/timing accounting and to demonstrate that the circuit
    as constructed executes on the simulator stack.
    """
    import_qiskit()
    try:
        from qiskit_aer import AerSimulator
    except Exception as exc:  # pragma: no cover
        return {"status": "unavailable", "reason": str(exc)}

    t0 = time.perf_counter()
    sim = AerSimulator(method="statevector")
    sim.set_options(seed_simulator=seed)
    compiled = sim.transpile(bundle.full_circuit) if hasattr(sim, "transpile") else bundle.full_circuit
    from qiskit import transpile as _t

    compiled = _t(bundle.full_circuit, backend=sim, optimization_level=0)
    result = sim.run(compiled, shots=shots, seed_simulator=seed).result()
    counts = result.get_counts()
    dt = time.perf_counter() - t0
    parsed = {}
    for k, v in counts.items():
        parsed[int(k.replace(" ", ""), 2)] = v
    return {
        "status": "ok",
        "runtime_seconds": dt,
        "shots": shots,
        "counts": {str(k): int(v) for k, v in sorted(parsed.items())},
    }
