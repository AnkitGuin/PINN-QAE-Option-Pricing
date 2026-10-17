# Methodology

This document describes what is actually implemented. The primary
methodological source of truth is `uploads/research_specification.md`; where
this implementation deviates (with reasons), the deviation is listed in
`docs/limitations.md` and the final report.

## 1. Data generation (synthetic)

Synthetic European-call panels are generated per scenario
(`src/data/synthetic_generator.py`):

- Business-day calendar from `start_date` (`n_dates` dates, ACT/365.25 year
  fractions).
- Spot path: geometric process with configured drift and realized volatility
  — **independent** of the pricing volatility schedule (avoids contaminating
  calibration with spot-path information).
- Time-varying volatility as **step schedules**; the exact integrated
  variance A(t,T) = ∫σ²(u)du is computed from the schedule by cumulative
  summation over the daily grid (piecewise-constant segments), so
  `true_sigma = σ_eff = sqrt(A/τ)` is exact, not approximated by an
  instantaneous substitution.
- Risk-neutral pricing uses the total-variance form
  `S_T = S_t·exp(R − A/2 + √A·Z)`, `R(t,T) = ∫r(u)du`, with the analytical
  Black–Scholes call evaluated at (σ_eff, r) — mathematically identical to
  the closed form under piecewise-constant schedules (validated to 1e-10 in
  Experiment 0 and `tests/`).
- Observation noise: relative Gaussian (`market_price = true_price·(1+ε)`,
  ε ~ N(0, rel_std)) when configured; zero otherwise.
- Determinism: every dataset is a pure function of (scenario config, seed,
  master_seed) and is cached by config hash (`experiments/_common.py`).

## 2. Classical benchmark: constrained least squares

`src/calibration/least_squares.py` fits a single constant σ per window by
minimising Σ_i (BS(σ; S_i,K_i,τ_i,r_i) − market_price_i)² over
σ ∈ [σ_lo, σ_hi] (bounded scalar minimisation; the prior window's estimate
is checked as an additional candidate because SciPy's bounded scalar solver
has no initial-point parameter). `RollingLSEngine` carries the preceding
estimate as that candidate; `FrozenEngine` fits the first window and never
updates. Both price out-of-sample blocks with the **same analytical
evaluator** used for every other σ-based method.

## 3. Inverse PINN (homogeneous formulation)

The network does not learn V(S,τ,K,r) directly. Using the homogeneity of the
Black–Scholes call in (S, K), it learns the dimensionless function

    u(x, τ, r),  x = ln(S/K) (z-scored),  V(S,τ,K,r) = S · u(x,τ,r)

which satisfies the transformed PDE

    f̃ = −u_τ + ½σ²(u_xx + u_x) + r·u_x = 0.

This parameterisation was adopted after a 4-input formulation
(S,τ,K,r)→V hit a representation floor (see `docs/limitations.md`); the
homogeneous form removes the floor entirely. Boundary/terminal conditions:

- terminal: u(x,0) = (1 − e^(−x))⁺
- lower boundary: u(x_lo,·) = 0
- upper boundary: u(x_hi,·) = 1 − e^(−rτ−x_hi)

Collocation points are sampled log-uniformly in S over a window-derived box
x ∈ [ln(S_lo/K_max), ln(S_hi/K_min)], S_hi chosen (factor 2.4 of window max
spot) so the parity-floor boundary error is negligible (recorded per
window).

**Training protocol (frozen before evaluation):** two stages. Stage A trains
the network with σ frozen (Adam with decay, then L-BFGS, chronological
early stopping on a validation split inside the window). Stage B releases a
single shared σ parameter (softplus parameterisation) and refines jointly
(Adam with separate network/σ learning rates, then L-BFGS). The calibrated
output is **σ̂ (and analytical prices computed from it)**, not direct network
prices; direct network prices are recorded separately as diagnostics. All
hyperparameters live in `configs/pinn.yaml`. The same architecture with all
physics terms disabled and stage B skipped serves as the **data-only neural
comparator** (direct network prices).

**Leakage controls:** inputs are (spot, strike, τ, rate) only; the scaler is
fitted on the calibration window only; walk-forward windows are strictly
chronological (`tests/test_no_leakage.py`).

## 4. Evaluation contracts

- All σ-based methods (LS, PINN, frozen) are priced on evaluation blocks by
  the **same analytical BS evaluator** (`evaluate_block_with_sigma`);
  the data-only network is priced directly by the network (recorded with
  `price_source`).
- Blocks are strictly out-of-sample: after each calibration window cutoff,
  the next `horizon_dates` business days form the evaluation block.
- Metrics: RMSE, MAE, bias, guarded relative error (floor 0.05); pooled
  across blocks and dates; per-method, per-condition.
- σ references: instantaneous schedule value at cutoff **and** block-effective
  σ = sqrt(mean σ_i²) (integrated variance over the block).
- Uncertainty: paired per-block RMSE differences (PINN − LS) with a normal
  95% interval (blocks are chronological, possibly autocorrelated — the
  interval is reported with that caveat, not as a rigorous test), and seed
  variability (5 seeds per condition).
- No seed is excluded; failed runs would be recorded with status and reason
  (none occurred in the completed runs).

## 5. Quantum module (canonical QAE, simulator)

`src/quantum/` implements the full pipeline:

1. **Terminal distribution**: lognormal S_T with μ_ln = ln S0 + R − A/2,
   σ_ln = √A (total-variance form; supports time-varying volatility through
   A(t,T), not instantaneous σ).
2. **Discretization** (`discretization.py`): N = 2^n uniform grid on
   [q_{α/2}, q_{1−α/2}] (α = 0.001 default), midpoint bins, conditional
   in-range probabilities normalized to one, and excluded tail mass recorded.
   Tail truncation and grid approximation both contribute to the reported
   discretization error. n is configurable (default 4 → 16 points).
3. **Payoff encoding** (`payoff.py`): h_j = (S_j − K)⁺/P_max with
   P_max = S_max − K, so h ∈ [0,1]; V = e^{−R}·P_max·a.
4. **Exact discrete expectation** (`exact_discrete.py`): a_exact = Σ p_j h_j
   computed before any QAE run, separating discretization error from
   estimation error.
5. **A operator**: exact `StatePreparation` of
   Σ_j √p_j |j⟩(√(1−h_j)|0⟩ + √h_j|1⟩) — arbitrary payoff shapes are exact;
   the cost of this state-preparation encoding is reported honestly in
   resource accounting (it is not shallow).
6. **Grover operator** Q = −A S₀ A† S_χ (Brassard et al.): S_χ = Z on the
   objective qubit, S₀ = X^{⊗}·MCZ·X^{⊗}. The **global −1 sign is essential**
   (verified numerically; without it canonical QAE estimates 1−a).
7. **Canonical QAE circuit**: m evaluation qubits, controlled-Q^{2^k} by gate
   repetition, inverse QFT. Outcome likelihood (exact finite-sum form,
   derived from the circuit and verified against simulation to ~1e-14):

       P(k|θ) = |S_k|² + |C_k|²,
       S_k = (1/2^m) Σ_l sin((2l+1)θ) e^{−2πikl/2^m},
       C_k = (1/2^m) Σ_l cos((2l+1)θ) e^{−2πikl/2^m},  a = sin²θ.

8. **Estimation**: maximum likelihood over a θ grid (20001 points,
   deterministic given counts). Modes: `statevector` (exact, deterministic;
   validation) and `shots` (ideal Aer simulator with fixed seed; falls back
   to seeded multinomial from the exact distribution if Aer is unavailable,
   with the method recorded).
9. **Resource accounting** (`resources.py`): transpiled (u, cx basis)
   depth/size for A, Q, and the full circuit; Grover queries = shots·(2^m−1);
   Initial A preparations and all A/A† calls inside Grover iterations are
   counted; Aer execution smoke run.
10. **Error decomposition** (`src/evaluation/error_decomposition.py`):
    calibration (input σ vs reference σ, both through analytical BS),
    discretization (exact-discrete vs analytical at the same σ), QAE
    estimation (QAE vs exact-discrete), total (QAE vs reference). Each
    component is reported separately; none is hidden in a total.

## 6. Statistical reporting discipline

- Comparisons are stated as observed results under the stated conditions
  (`comparison_statement`), never as general superiority claims.
- Hypothesis reports record the hypothesis, the observed numbers, an
  interpretation, and a limitation.
- No data or numbers are fabricated; unfavourable results are preserved and
  reported (see Experiment 2 results, where the inverse PINN underperforms
  least squares on pooled RMSE in the completed conditions).
