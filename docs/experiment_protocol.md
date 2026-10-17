# Experiment Protocol

All experiments run from the repository root as:

```
python -m experiments.generate_synthetic --config configs/synthetic/constant.yaml --seeds 0,1,2,3,4
python -m experiments.exp0_validation   --config configs/experiments/exp0_validation.yaml
python -m experiments.exp1_forward_pinn --config configs/experiments/exp1_forward_pinn.yaml
python -m experiments.exp2_inverse_pinn --config configs/experiments/exp2_inverse_pinn.yaml
python -m experiments.exp3_real_market  --config configs/experiments/exp3_real_market.yaml
python -m experiments.exp4_quantum      --config configs/experiments/exp4_quantum.yaml
python scripts/make_figures.py
python -m pytest tests/
```

Everything numeric (seeds, windows, tolerances, qubit counts, path counts)
comes from the YAML configs; no hard-coded experimental values in code.

## Experiment 0 — Black–Scholes + Monte Carlo validation (Gate 1)

**Question:** is the analytical/monte-carlo core correct before anything is
built on it?

Analytical checks against independent references (textbook value, put–call
parity, static bounds, total-variance form, zero-maturity limits); implied-vol
round-trip (1e-7); MC convergence to analytical (≤ 0.15 at 10⁶ paths) and
95% CI coverage (≥ 0.70 of repeats); PDE residual of the analytical solution.

**Acceptance:** all checks pass (Gate 1). Result: PASS, 9/9 checks
(`results/experiment_0/gate1_report.json`).

## Experiment 1 — Forward PINN reproduces Black–Scholes (Gate 4)

**Question:** can the PINN architecture/training learn the pricing PDE at all
(forward problem, σ known = 0.2)?

5 seeds on the constant scenario grid; evaluate on an independent grid with
the analytical solution; independent-collocation PDE residual.

**Acceptance:** mean price RMSE ≤ 0.25, max per-seed RMSE ≤ 0.25 (strict
form: max seed), mean max error ≤ 2.0, independent PDE residual RMS ≤ 0.05.
Result: PASS (mean RMSE 0.1431, max-seed 0.1992, mean max error 1.151,
residual RMS 3.9e-4).

## Experiment 2 — Inverse PINN vs LS calibration (core experiment)

**Research question:** under identical rolling data windows and
Black–Scholes assumptions, does inverse-PINN estimation of effective
volatility reduce out-of-sample European-call pricing RMSE relative to
constrained least-squares volatility calibration, particularly under changing
volatility?

**Design:** 6 conditions × 5 seeds × 20 walk-forward events × 4 methods.

- Conditions: constant_clean, constant_noise (2% relative price noise),
  constant_sparse (sparse strike/maturity grid), constant_noise_sparse,
  gradual_change (0.18→0.32), abrupt_change (0.20/0.35/0.18 steps).
- Methods: `least_squares` (rolling, warm start), `pinn` (rolling, warm
  start of σ), `frozen` (first-window LS, never updated), `data_only_nn`
  (same network, all physics terms off, direct network prices).
- Walk-forward: 20 business days calibration window, step 5 days, horizon 5
  days; evaluation blocks priced by the common analytical evaluator
  (data-only network priced directly; PINN direct-network RMSE recorded
  separately).
- σ references: instantaneous at cutoff and block-effective
  √(mean σ_i²); regime per block.
- Gate 5 (constant recovery): median |σ̂_PINN − σ_true| ≤ 0.02 on
  constant_clean.

**Outputs:** per-run `block_table.csv`, `sigma_history.csv`, `metrics.json`,
`training_history_first_window.csv`; cross-condition `summary.json`,
`method_summary.csv`, `all_block_tables.csv`, `all_sigma_histories.csv`.
Runs resume on re-execution (completed runs are skipped); aggregation is
always rebuilt from disk (`--aggregate-only` rebuilds without running).

## Experiment 3 — Real market data (conditional, NOT EXECUTED)

Requires a suitable historical European-option dataset (bid/ask, European
exercise, aligned spot/rates). No eligible provider is accessible in this
environment (no API credentials, no licensed vendor files). Per the research
specification, unsuitable data is not substituted: the script records
`status: not_executed` with the reason and exits 0
(`results/experiment_3/status.json`). The enabled path (dataset provided via
`configs/real_data.yaml`) implements schema validation, quality filters and
exclusion logging, but has not been validated end-to-end here.

## Experiment 4 — Quantum pricing (canonical QAE, simulator)

**Question:** does the calibration difference between methods propagate
through the quantum pricing module, and what are the error components?

**Design (fixed before evaluation):**

- Contracts: from exp2 constant_clean seed 0, first evaluation block,
  nearest-ATM strike per maturity {0.25, 0.50, 1.00}.
- σ inputs: known (true effective), least_squares, pinn, frozen — taken from
  the exp2 artifacts for that block (not recomputed).
- Identical quantum settings for all inputs (n = 4 grid qubits, α = 0.001,
  m = 3 evaluation qubits, 4096 shots, MLE post-processing, seed 7), so
  output differences are attributable to calibration inputs only.
- Error decomposition per (input, contract): calibration / discretization /
  QAE estimation / total.
- Sweeps (known σ only): m ∈ {2,3,4,5} at n = 4; n ∈ {2,3,4,5} at m = 3.
- Classical MC convergence on the same contract: path counts
  {10³,10⁴,10⁵,10⁶} × 20 repeats.
- Resource accounting: transpiled circuit depth/size, Grover queries, initial
  A preparations, and all A/A† calls inside Grover iterations.

**Outputs:** `qae_main.csv`, `sweep_m.csv`, `sweep_n.csv`,
`mc_convergence.csv`, `resources.json`, `summary.json`.

## Gates

| Gate | Content | Status |
|---|---|---|
| 1 | BS/MC core validation (exp0) | PASS |
| 2 | (internal) synthetic data + schema checks | PASS (tests) |
| 3 | LS σ recovery on clean data | PASS (σ err ≤ 3.4e-10) |
| 4 | Forward PINN reproduces BS (exp1) | PASS |
| 5 | Inverse PINN constant-σ recovery (exp2) | PASS (median 0.0027 ≤ 0.02) |
| 6 | Quantum module correctness | PASS (tests/test_quantum.py + subprocess checks) |
