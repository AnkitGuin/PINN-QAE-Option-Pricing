# Physics-Informed Neural Networks for Adaptive Quantum Option Pricing under Time-Varying Market Conditions

Reference implementation of the full research pipeline: synthetic
European-option panels with time-varying volatility, classical (constrained
least-squares) and inverse-PINN volatility calibration under identical
leakage-safe walk-forward evaluation, and a simulator-based canonical
quantum amplitude estimation (QAE) pricing module fed by either calibration
method, with full error decomposition and resource accounting.

**Primary methodological source of truth:** [`docs/methodology.md`](docs/methodology.md)
and [`docs/experiment_protocol.md`](docs/experiment_protocol.md). Deviations,
audit history, and known limitations are documented in [`docs/limitations.md`](docs/limitations.md)
and [`REPORT.md`](REPORT.md).

## Research question

Under identical rolling data windows and Black–Scholes assumptions, does
inverse-PINN estimation of effective volatility reduce out-of-sample
European-call pricing RMSE relative to constrained least-squares volatility
calibration, particularly under changing volatility — and does any
calibration difference persist when both estimates feed the same simulated
QAE pricing module?

## Previously reported results

The results in `REPORT.md` and `results/` predate the audit corrections
described there and are **superseded pending a full experiment rerun**. Do
not treat their numerical tables as validated against the current code. The
exp2 runner now rejects stale run artifacts and recomputes them.

The prior result summary is retained for provenance:

On the completed conditions, the inverse PINN recovers the true volatility on
clean constant data (Gate 5, median |σ error| 0.0027) but its pooled
out-of-sample pricing RMSE is **worse than rolling least squares** in the
constant-data conditions (e.g. constant_clean: PINN 0.086 vs LS ≈ 0.000; the
generator is exactly invertible by LS, so this is expected on clean data) —
the scientifically relevant comparisons are the noisy/sparse/changing-vol
conditions, reported without favourable spin. The quantum module reproduces
the exact discrete expectation to ~1e-4 (statevector mode) and to shot-noise
level with 4096 shots, and error decomposition separates calibration,
discretization and QAE estimation effects. No quantum advantage is claimed.

## Repository layout

```
configs/            all YAML configuration (no hard-coded experimental values)
  base.yaml, calibration.yaml, pinn.yaml, quantum.yaml, real_data.yaml
  synthetic/*.yaml           scenario definitions
  experiments/exp0..exp4     experiment configurations
src/
  data/          synthetic generator, schema, splits, preprocessing, validation
  black_scholes/ analytical, implied vol, Monte Carlo, PDE residual
  calibration/   constrained LS, rolling/frozen engines
  pinn/          homogeneous-formulation PINN (network, losses, collocation,
                 staged training, forward/inverse fits, diagnostics)
  quantum/       terminal distribution, discretization, payoff encoding,
                 exact discrete expectation, canonical QAE, resources
  evaluation/    metrics, calibration metrics, uncertainty, comparisons,
                 error decomposition, reporting
  utils/         config, reproducibility, logging, timing, serialization
experiments/      exp0..exp4 CLI modules + shared helpers
tests/            7 test modules incl. the leakage-prevention suite
scripts/          make_figures.py
docs/             data dictionary, methodology, experiment protocol, limitations
notebooks/        analysis notebooks (see notebooks/README.md)
results/          machine-readable artifacts per experiment (committed)
figures/          generated figures + manifest
data/             synthetic datasets + manifests (regenerated deterministically)
```

## Setup

Python ≥ 3.11. Install pinned dependencies:

```bash
pip install torch qiskit qiskit-aer numpy scipy pandas matplotlib pyyaml pytest
```

(pins used in development: torch 2.14.1+cpu, qiskit 2.6.0, qiskit-aer 0.17.2 —
see `pyproject.toml`).

### Platform note (torch + qiskit-aer in one process)

Both packages bundle their own libgomp; loading both in one process can fail
with "cannot allocate memory in static TLS block". The experiment scripts
never import both in one process (exp0–exp3 never import qiskit; exp4 never
imports torch), and the test suite runs circuit-level quantum checks in a
subprocess, so `python -m pytest tests/` works as-is.

## Running everything

```bash
python -m experiments.generate_synthetic --config configs/synthetic/constant.yaml --seeds 0,1,2,3,4
python -m experiments.exp0_validation   --config configs/experiments/exp0_validation.yaml
python -m experiments.exp1_forward_pinn --config configs/experiments/exp1_forward_pinn.yaml
python -m experiments.exp2_inverse_pinn --config configs/experiments/exp2_inverse_pinn.yaml
python -m experiments.exp3_real_market  --config configs/experiments/exp3_real_market.yaml   # records not_executed
python -m experiments.exp4_quantum      --config configs/experiments/exp4_quantum.yaml
python scripts/make_figures.py
python -m pytest tests/        # 87 collected tests, including simulator circuit checks
```

The audit run passed 82 non-circuit tests before the final edge-case and
documentation edits. The full suite, including Qiskit Aer circuit checks, has
not been rerun against the final tree; run the command above before relying on
the implementation for new results.

exp2 supports `--conditions a,b`, `--seeds 0,1`, and resumes completed runs;
`--aggregate-only` rebuilds the cross-condition summary from disk. exp4
requires the exp2 constant_clean seed-0 artifacts.

## Historical validation gates (pre-audit)

| Gate | Check | Result |
|---|---|---|
| 1 | Analytical/MC core (exp0) | PASS 9/9 |
| 3 | LS σ recovery on clean data | PASS (≤ 3.4e-10) |
| 4 | Forward PINN reproduces BS (exp1) | PASS |
| 5 | Inverse PINN constant-σ recovery | PASS (0.0027 ≤ 0.02) |
| 6 | Quantum module correctness | PASS |

## Scientific-integrity commitments

- No fabricated data or numbers; no retries until favourable results; no
  silent seed removal (failed runs would be recorded with status/reason).
- Comparators share observations, windows, evaluation contracts and the
  analytical evaluator; PINN outputs are analytical prices at the calibrated
  σ̂.
- Synthetic-only fields (true σ, true price, noise, regime) are never model
  inputs; enforced by `tests/test_no_leakage.py`.
- Time-varying volatility always enters through integrated variance
  A(t,T)=∫σ²du (never instantaneous σ in a constant-vol formula).
- Unfavourable results are reported as first-class findings.
- No unevidenced claims of PINN/quantum superiority, quantum advantage, or
  speedup.
