# Notebooks

- `exp2_analysis.ipynb` — interactive exploration of the exp2 artifacts
  (pooled RMSE table, sigma paths vs truth, paired block differences,
  Gate 5). Reads `results/experiment_2/` only; no recomputation.
- `exp4_quantum_analysis.ipynb` — exp4 artifacts (error decomposition,
  m/n sweeps, MC vs QAE budgets, resource accounting). Reads
  `results/experiment_4/` only.

Both notebooks expect to be launched from the repository root:
`jupyter notebook notebooks/`. All figures are also produced by
`python scripts/make_figures.py` (into `figures/`).
