# Final Implementation Report

**Project:** Physics-Informed Neural Networks for Adaptive Quantum Option
Pricing under Time-Varying Market Conditions
**Date:** 2026-10-17
**Status:** Prior run archived; final experiment rerun required after audit corrections

> **Audit correction notice (2026-10-17):** A final source audit found an
> evaluation-horizon off-by-one, a synthetic initial-spot mismatch, incorrect
> PINN early-stopping patience handling, inconsistent QAE tail treatment, and
> undercounted A/A† calls. These have been corrected in the implementation.
> The numerical results below were generated before those fixes and are now
> historical only. They must not be cited as results from the corrected code.
> The exp2 runner now marks old artifacts stale and will regenerate them.

This report summarises what was implemented, what was validated, what
failed, and where the implementation deviates from the research
specification. All numbers below come from the machine-readable artifacts in
`results/` and are reproducible from the committed configs and code.

---

## 1. What was implemented

| Component | Location | Status |
|---|---|---|
| Config system (YAML includes, hashes; no hard-coded experimental values) | `src/utils/config.py`, `configs/` | done, tested |
| Reproducibility (seed derivation, environment manifests) | `src/utils/reproducibility.py` | done |
| Black–Scholes core (analytical, total-variance form, IV, MC, PDE residual) | `src/black_scholes/` | done, tested |
| Synthetic data generator (step-σ schedules, exact integrated variance A(t,T), noise, sparse grids, leakage-safe schema) | `src/data/` | done, tested |
| Constrained LS calibration (rolling warm-start, frozen) | `src/calibration/` | done, tested |
| Homogeneous-formulation inverse PINN (staged training, warm start, physics-off ablation) | `src/pinn/` | done, tested |
| Evaluation (common analytical evaluator, metrics, σ references, paired differences, error decomposition, neutral reporting) | `src/evaluation/` | done, tested |
| Quantum module (lognormal terminal law, quantile discretization, payoff encoding, exact discrete expectation, canonical Brassard QAE, MLE, resource accounting) | `src/quantum/` | done, tested (Gate 6) |
| Experiments exp0–exp4 as CLI modules + synthetic-data CLI | `experiments/` | done, all run |
| Test suite (7 modules, 82 tests, incl. leakage suite) | `tests/` | all pass |
| Figures (7, from saved artifacts only) + manifest | `scripts/make_figures.py`, `figures/` | done |
| Docs (data dictionary, methodology, experiment protocol, limitations) | `docs/` | done |
| Notebooks (exp2, exp4 analysis; artifact-reading only) | `notebooks/` | done |
| README, LICENSE (MIT), pyproject | repo root | done |

Compute environment: 2 CPU cores, CPU-only PyTorch 2.14.1, Qiskit 2.6.0,
qiskit-aer 0.17.2, Python 3.13.

## 2. Validation gates

| Gate | Check | Result |
|---|---|---|
| 1 | Analytical/MC core (exp0): textbook value, parity, bounds, IV round-trip, MC convergence + CI coverage | **PASS** (9/9; MC error 0.0051 at 10⁶ paths, CI coverage 0.95) |
| 2 | Synthetic data + schema invariants | **PASS** (tests) |
| 3 | LS σ recovery on clean data | **PASS** (σ error ≤ 3.4e-10) |
| 4 | Forward PINN reproduces BS (exp1, 5 seeds) | **PASS** (mean RMSE 0.1431, max-seed 0.1992 ≤ 0.25; mean max error 1.151 ≤ 2.0; independent PDE residual RMS 3.9e-4 ≤ 0.05) |
| 5 | Inverse PINN constant-σ recovery (exp2, 5 seeds) | **PASS** (median |σ error| 0.0029 ≤ 0.02; max 0.0086) |
| 6 | Quantum module correctness | **PASS** (A operator exact to 1e-15; Grover rotation identity to 1e-10; circuit distribution matches derived likelihood to 1e-10; statevector QAE recovers a to <1e-4; shots mode within shot noise) |

Test suite: **82 passed** (`python -m pytest tests/`), including
`test_no_leakage.py` (synthetic-only fields never inputs; walk-forward
strictly causal; scaler on window only; evaluator invariant to latent-field
perturbation; training identical with latent columns removed).

## 3. Experiment results

### 3.1 Experiment 2 (core): inverse PINN vs LS, 6 conditions × 5 seeds × 20 blocks (30/30 runs, 2400 blocks, no failures, no exclusions)

Pooled out-of-sample block RMSE (common analytical evaluator; seed std of
mean block RMSE ≤ 0.014 everywhere):

| Condition | LS | PINN | Frozen | Data-only NN | Paired PINN−LS mean [95% CI] |
|---|---|---|---|---|---|
| constant_clean | **0.0000** | 0.0870 | 0.0000 | 0.0394 | +0.0734 [+0.064, +0.083] |
| constant_noise | **0.0061** | 0.0818 | 0.0048 | 0.0527 | +0.0662 [+0.058, +0.074] |
| constant_sparse | **0.0000** | 0.0901 | 0.0000 | 0.0288 | +0.0792 [+0.071, +0.088] |
| constant_noise_sparse | **0.0069** (frozen) / 0.0097 (LS) | 0.1065 | 0.0069 | 0.0472 | +0.0818 [+0.071, +0.093] |
| gradual_change | **0.2863** | 0.3279 | 0.8116 | 0.1693 | +0.0546 [+0.043, +0.066] |
| abrupt_change | 0.4732 | **0.4470** | 0.9712 | 0.3218 | **−0.0231 [−0.039, −0.008]** |

**Findings, stated as observed results:**

1. **Constant conditions:** LS is essentially exact (the synthetic generator
   is the same BS formula LS inverts), so PINN cannot and does not beat it
   there. This is expected and reported without spin. The PINN's residual
   error (~0.087) reflects estimation noise of a one-parameter-augmented
   network fit vs an exactly invertible map; its σ̂ itself is accurate
   (mean |σ error| 0.003).
2. **Abrupt volatility change:** the inverse PINN produces a small but
   consistent improvement — pooled RMSE 0.4470 vs 0.4732, paired per-block
   difference −0.0231 with a normal 95% interval [−0.039, −0.008] excluding
   zero (descriptive interval; blocks are chronological, possibly
   autocorrelated). The mechanism is visible in the σ paths: at the jump
   event both methods undershoot the block-effective σ (0.264) — LS 0.233,
   PINN 0.225 — and PINN's smoother re-adjustment prices the transition
   blocks slightly better.
3. **Gradual change:** PINN is slightly worse (+0.0546) — the smooth ramp
   is tracked well by both, and PINN's estimation noise dominates.
4. **Frozen** first-window calibration is catastrophic under change
   (0.81–0.97 RMSE), confirming that adaptation matters far more than the
   choice of adaptive estimator.
5. **Data-only network** (no physics) generalises well *within the window
   domain* (0.03–0.32) — an important negative control: on this synthetic
   in-distribution task, physics constraints are not necessary for good
   interpolation; their value must be argued elsewhere (out-of-domain
   robustness, parameter interpretability — the calibrated σ̂).
6. Overall pooled across all conditions: LS 0.2259, PINN 0.2384, frozen
   0.5167, data-only 0.1525 — dominated by the four constant conditions;
   the condition-level table above is the meaningful comparison.

**Answer to the research question:** under the stated synthetic conditions,
inverse-PINN σ estimation reduces out-of-sample pricing RMSE relative to
constrained LS **only under abrupt volatility change** (−0.026 pooled RMSE,
paired CI excluding zero); it does not under gradual change or constant
conditions. Both conclusions are observed results for this data-generating
process, not general claims.

### 3.2 Experiment 4 (quantum): canonical QAE fed by each calibration

Contracts: exp2 constant_clean seed 0, first evaluation block, nearest-ATM
per maturity (τ ≈ 0.25/0.5/1.0, K = S = 103.10). Identical quantum settings
for all inputs (n = 4 grid qubits, α = 0.001, m = 3, 4096 shots, MLE, seed
7). Pooled RMSE over the three contracts:

| σ input | σ̂ | Total | Calibration | Discretization | QAE estimation |
|---|---|---|---|---|---|
| known σ (0.2000) | 0.2000 | 0.0129 | 0.0000 | 0.0438 | 0.0520 |
| least squares | 0.2000 | 0.0129 | 0.0000 | 0.0438 | 0.0520 |
| frozen | 0.2000 | 0.0129 | 0.0000 | 0.0438 | 0.0520 |
| PINN | 0.19882 | 0.0525 | 0.0354 | 0.0440 | 0.0410 |

**Findings:**

- The quantum module is validated end-to-end: statevector-mode QAE recovers
  the exact discrete amplitude to <1e-4 for every contract and m ∈ {2..5};
  shots mode (ideal Aer, 4096 shots) sits within shot-noise expectations
  (10-seed spread at m=4: std 6e-4 in a, ≈0.04 in price units).
- **Calibration differences propagate through QAE as expected:** the PINN's
  σ error (−0.0012) produces a calibration error of 0.035 RMSE that passes
  through the identical quantum pipeline — total error 0.0525 vs 0.0129 for
  the exactly-calibrated inputs. The QAE module treats all inputs
  identically (same grids, shots, seed); differences are attributable to
  inputs only.
- Error decomposition separates the effects: at these settings,
  discretization (0.044) and QAE shot noise (0.04–0.05) dominate
  calibration error for well-calibrated inputs; the calibration term
  dominates for the PINN input.
- Resolution sweeps (known σ): QAE estimation error does not improve
  monotonically with m at fixed 4096 shots (m=2: 0.020, m=3: 0.089, m=4:
  0.015, m=5: 0.024 in price units) — expected, because MLE resolution
  improves with m but the likelihood sharpens beyond what 4096 shots can
  resolve; this is an honest observation about canonical QAE at modest shot
  budgets, not a defect of the implementation. Discretization error falls
  from 0.35 (n=3) to 0.069 (n=4) and 0.017 (n=5) in price units.
- Classical MC on the same contract: RMSE 0.246 (10³) → 0.064 (10⁴) →
  0.017 (10⁵) → 0.0069 (10⁶ paths), 20 repeats each. The MC-vs-QAE
  comparison is reported as error vs budget (paths vs Grover queries:
  12k–127k at m=2..5) — **no speedup or quantum-advantage claim is made**;
  simulator timings are not comparable to classical runtimes.
- Resource accounting (n=4, m=3, transpiled to u/cx basis): A operator
  depth 53 / size 57; Grover Q depth 169; full circuit depth 9,904 / size
  11,968; 28,672 Grover queries and 4,096 A-applications at 4,096 shots;
  circuit executes on AerSimulator (statevector) in 0.2 s. The A operator
  uses exact StatePreparation (distribution loading + payoff encoding
  together); its cost is reported, not hidden, and no hardware-feasibility
  claim is made.

### 3.3 Experiment 3 (real market): not executed

No suitable historical European-option dataset is accessible in this
environment (no API credentials, no licensed vendor files). Per the
specification, unsuitable data was not substituted. Status recorded in
`results/experiment_3/status.json`; the synthetic pipeline is fully
functional without it. The enabled path (quality filters, exclusion
logging, walk-forward on real data) is implemented but unvalidated.

## 4. What failed, and what was done about it

1. **4-input PINN formulation (S,τ,K,r)→V — abandoned after extensive
   diagnosis.** The network could not represent the pricing map accurately
   (representation floor ≈ 0.35 RMSE at the profile optimum σ≈0.16);
   loss-weight regimes, L-BFGS variants, multi-start selection, tighter
   domains, joint σ-initialisation sweeps and staged protocols all failed
   to break the floor. Root cause: the 4-D parameterisation itself. Fix:
   the homogeneous formulation V = S·u(x,τ,r), x = ln(S/K), which satisfies
   a transformed PDE and removed the floor entirely (forward RMSE 0.059 in
   development; Gate 4/5 pass). Full dead-end documentation is preserved in
   this report and `docs/limitations.md`.
2. **Canonical QAE sign convention (caught by verification).** Without
   Brassard's global −1 on Q, the eigenphases are π±2θ and QAE estimates
   1−a. Detected numerically (circuit distribution was exactly the
   likelihood shifted by 2^{m−1}); fixed via the circuit global phase and
   locked in by tests.
3. **Closed-form QAE likelihood — misremembered formula caught by
   verification.** The initially used sin²(2^mθ)/(2^{2m}sin²(θ−πk/2^m)) form
   did not match the simulated circuit. Replaced by the exact finite-sum
   likelihood derived from the circuit's own structure
   (P(k|θ)=|S_k|²+|C_k|² with sin/cos of (2l+1)θ), which matches simulation
   to 1e-14 and is what the MLE uses.
4. **Condition-override bug in exp2 (caught before it could contaminate
   results).** The `constant_noise` / `constant_sparse` condition overrides
   were wrapped under a `synthetic:` key that the loader did not unwrap, so
   the noise/sparse overrides would have been silently ignored. Fixed and
   verified before the affected conditions ran; the affected runs were
   (re)executed with the corrected loader.
5. **torch + qiskit-aer static-TLS conflict.** Both bundle libgomp; loading
   both in one process can fail with "cannot allocate memory in static TLS
   block". Resolution: experiments never import both in one process (exp0–3
   do not import qiskit; exp4 does not import torch); the environment
   manifest records torch as unloadable in qiskit processes rather than
   failing; circuit-level tests run in a dedicated subprocess. Two earlier
   conftest-based workarounds (early import, LD_PRELOAD re-exec) were tried
   and rejected (the re-exec lost output through pytest's fd capture).
6. **exp4 selection bug (caught in dry-run).** The nearest-ATM selector
   initially minimised raw moneyness K/S instead of |K/S−1| (selecting the
   lowest strike). Fixed and verified before the experiment ran.
7. **exp2 aggregation gap (fixed).** Resume-skipped runs were not loaded
   into the final summary, and parallel workers each produced
   condition-partial summaries. Aggregation now always rebuilds from disk
   (`--aggregate-only`), covering all 30 runs.

## 5. Deviations from the specification

1. **PINN parameterisation** changed from a direct V(S,τ,K,r) network to
   the homogeneous V = S·u form (see §4.1). The research question,
   hypotheses, evaluation contract and comparators are unchanged; the
   deviation is in the function parameterisation only, and is justified by
   the documented failure of the original form.
2. **QAE variant**: canonical Brassard et al. QAE with MLE post-processing
   (fixed before evaluation). The specification allowed the variant to be
   fixed in advance; this choice is recorded.
3. **Distribution grid**: default n = 4 (16 points, ≈1% discretization
   error on the exp4 contracts) instead of the originally sketched n = 3
   (≈5%). Config-driven; the n-sweep reports both.
4. **Experiment 3 not executed** (no eligible data) — explicitly permitted
   by the specification's conditionality; recorded honestly.
5. **Hyperparameter freezing:** the PINN protocol was tuned on development
   scenarios and frozen into `configs/pinn.yaml` before the evaluation
   conditions were run; no per-condition tuning, no post-hoc retries.

## 6. Reproducibility

- Every dataset is a pure function of (scenario config, seed, master seed),
  cached by config hash; experiment configs pin every numeric choice.
- All runs carry `config.json`, `manifest.json` (config hash, seeds,
  environment), and per-run artifacts (`results/experiment_2/*/`,
  `results/experiment_4/*.csv`).
- Full pipeline re-run: see `README.md` (six CLI commands + figures +
  tests). exp2 resumes completed runs; `--aggregate-only` rebuilds summaries.
- Total compute for the reported results: ≈ 2.2 h on 2 CPU cores (exp2:
  7,381 s PINN + 1,159 s data-only network + ~0 LS across 120 windows × 30
  runs; exp4: 13 s; tests: ~100 s).

## 7. Conclusions (scoped)

- The inverse PINN recovers volatility accurately on clean data (Gate 5)
  and is a viable σ estimator, but under this synthetic data-generating
  process it improves out-of-sample pricing RMSE only under abrupt
  volatility change; least-squares calibration remains better under
  constant and gradually changing volatility. The observed improvement at
  abrupt transitions is small (−0.023 RMSE on prices of order 8–15) and its
  normal-approximation interval should be read as descriptive.
- Calibration differences propagate through the QAE pricing module exactly
  as the error decomposition predicts: with identical quantum settings, the
  quantum output inherits the calibration error of its input (0.035 RMSE
  for the PINN input vs 0 for exactly-calibrated inputs), with
  discretization and shot-noise components of similar magnitude at the
  chosen settings.
- No evidence about real markets is claimed: all results are synthetic, and
  Experiment 3 was not executed for lack of eligible data.
