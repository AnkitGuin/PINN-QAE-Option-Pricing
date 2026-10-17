"""Generate all figures from saved experiment artifacts.

Reads only from results/ (never recomputes experiments) so figures are
exactly consistent with the recorded numbers.  Each figure is saved as PNG
plus a JSON manifest (path, what it shows, data source).

Usage:
    python scripts/make_figures.py [--output figures]

Figures:
  fig1_exp0_mc_convergence   Monte Carlo error vs path count, with CI coverage
  fig2_exp1_price_surface    forward-PINN learned surface vs analytical BS
  fig3_exp2_rmse_by_condition pooled RMSE by condition and method
  fig4_exp2_sigma_paths      rolling sigma estimates vs truth (one condition)
  fig5_exp2_paired_diffs     paired per-block PINN-LS RMSE differences
  fig6_exp4_error_decomposition stacked error components by sigma input
  fig7_exp4_convergence      MC RMSE vs budget with QAE error for comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT

METHOD_COLORS = {
    "least_squares": "#1f77b4",
    "pinn": "#d62728",
    "frozen": "#7f7f7f",
    "data_only_nn": "#2ca02c",
    "known_sigma": "#9467bd",
}
METHOD_LABELS = {
    "least_squares": "Least squares",
    "pinn": "Inverse PINN",
    "frozen": "Frozen",
    "data_only_nn": "Data-only NN",
    "known_sigma": "Known $\\sigma$",
}
CONDITION_ORDER = [
    "constant_clean", "constant_noise", "constant_sparse", "constant_noise_sparse",
    "gradual_change", "abrupt_change",
]


def _save(fig, out: Path, name: str, description: str, source: str, manifests: List[Dict[str, Any]]) -> Path:
    p = out / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    manifests.append({"figure": str(p), "description": description, "data_source": source})
    return p


def fig_exp0(out: Path, manifests) -> None:
    res = json.loads((PROJECT_ROOT / "results/experiment_0/gate1_report.json").read_text())
    reports = res.get("monte_carlo_reports") or []
    if not reports:
        return
    rep = reports[0]
    # RMSE across repeats vs path count
    counts = [r["n_paths"] for r in rep["repeats"]]
    rmse = []
    for r in rep["repeats"]:
        errs = [abs(p - rep["analytical"]) for p in r["prices"]]
        rmse.append(float(np.sqrt(np.mean(np.square(errs)))))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(counts, rmse, "o-", label="MC RMSE vs analytical")
    ref = [rmse[0] * np.sqrt(counts[0] / n) for n in counts]
    ax.loglog(counts, ref, "k--", alpha=0.5, label="$N^{-1/2}$ reference")
    ax.set_xlabel("Monte Carlo paths")
    ax.set_ylabel("RMSE (price units)")
    ax.set_title(f"Experiment 0: MC convergence (S={rep['contract']['spot']}, "
                 f"K={rep['contract']['strike']}, τ={rep['contract']['tau']}, σ={rep['contract']['sigma']})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out, "fig1_exp0_mc_convergence",
          "Monte Carlo pricing error vs path count with 1/sqrt(N) reference",
          "results/experiment_0/gate1_report.json", manifests)


def fig_exp1(out: Path, manifests) -> None:
    summary = json.loads((PROJECT_ROOT / "results/experiment_1/summary.json").read_text())
    runs = summary["runs"]
    seeds = [r["seed"] for r in runs]
    rmse = [r["price_rmse"] for r in runs]
    resid = [r["pde_residual_rms_independent"] for r in runs]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar([str(s) for s in seeds], rmse, color="#d62728", alpha=0.85)
    ax1.axhline(summary["acceptance"]["price_rmse_max"], color="k", ls="--",
                label=f"tolerance {summary['acceptance']['price_rmse_max']}")
    ax1.set_xlabel("seed"); ax1.set_ylabel("price RMSE vs BS")
    ax1.set_title("Forward PINN: price RMSE by seed")
    ax1.legend(); ax1.grid(True, axis="y", alpha=0.3)
    ax2.bar([str(s) for s in seeds], resid, color="#1f77b4", alpha=0.85)
    ax2.axhline(summary["acceptance"]["pde_residual_rms_max"], color="k", ls="--",
                label=f"tolerance {summary['acceptance']['pde_residual_rms_max']}")
    ax2.set_xlabel("seed"); ax2.set_ylabel("independent PDE residual RMS")
    ax2.set_title("Forward PINN: physics residual by seed")
    ax2.legend(); ax2.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Experiment 1 (gate {summary['gate']}: {summary['name']})")
    fig.tight_layout()
    _save(fig, out, "fig2_exp1_forward_pinn",
          "Forward PINN price RMSE and independent PDE residual per seed with acceptance tolerances",
          "results/experiment_1/summary.json", manifests)


def fig_exp2_rmse(out: Path, manifests) -> None:
    table = pd.read_csv(PROJECT_ROOT / "results/experiment_2/all_block_tables.csv")
    conds = [c for c in CONDITION_ORDER if c in set(table["condition"])]
    methods = [m for m in ["least_squares", "pinn", "frozen", "data_only_nn"] if m in set(table["method"])]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.8 / len(methods)
    for i, meth in enumerate(methods):
        xs, ys, es = [], [], []
        for j, cond in enumerate(conds):
            sub = table[(table["condition"] == cond) & (table["method"] == meth)]
            seed_rmse = sub.groupby("seed")["rmse"].mean()
            xs.append(j + (i - len(methods) / 2 + 0.5) * width)
            ys.append(float(np.sqrt(np.mean(sub["rmse"] ** 2))))
            es.append(float(seed_rmse.std(ddof=1)) if len(seed_rmse) > 1 else 0.0)
        ax.bar(xs, ys, width=width * 0.92, color=METHOD_COLORS[meth],
               label=METHOD_LABELS[meth], yerr=es, capsize=2)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylabel("Pooled block RMSE (price units)")
    ax.set_title("Experiment 2: out-of-sample pricing RMSE by condition and method")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out, "fig3_exp2_rmse_by_condition",
          "Pooled out-of-sample RMSE per condition/method (error bars: seed std of mean block RMSE)",
          "results/experiment_2/all_block_tables.csv", manifests)


def fig_exp2_sigma(out: Path, manifests) -> None:
    sig = pd.read_csv(PROJECT_ROOT / "results/experiment_2/all_sigma_histories.csv")
    sig["cutoff"] = pd.to_datetime(sig["cutoff"])
    conditions = [c for c in ["abrupt_change", "gradual_change"] if c in set(sig["condition"])]
    if not conditions:
        conditions = [sig["condition"].iloc[0]]
    fig, axes = plt.subplots(1, len(conditions), figsize=(6.2 * len(conditions), 4), squeeze=False)
    for ax, cond in zip(axes[0], conditions):
        sub = sig[(sig["condition"] == cond) & (sig["seed"] == 0)].sort_values("cutoff")
        ax.plot(sub["cutoff"], sub["sigma_ref_instantaneous"], "k-", lw=2, label="true inst.")
        ax.plot(sub["cutoff"], sub["sigma_ref_effective_block"], "k:", lw=2, label="true block-eff.")
        ax.plot(sub["cutoff"], sub["sigma_ls"], color=METHOD_COLORS["least_squares"], label="LS")
        ax.plot(sub["cutoff"], sub["sigma_pinn"], color=METHOD_COLORS["pinn"], label="PINN")
        ax.plot(sub["cutoff"], sub["sigma_frozen"], color=METHOD_COLORS["frozen"], label="frozen")
        ax.set_title(f"{cond} (seed 0)")
        ax.set_ylabel("volatility σ")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30); lbl.set_ha("right")
    fig.suptitle("Experiment 2: rolling σ estimates vs generating truth")
    fig.tight_layout()
    _save(fig, out, "fig4_exp2_sigma_paths",
          "Rolling sigma estimates against instantaneous and block-effective truth (seed 0)",
          "results/experiment_2/all_sigma_histories.csv", manifests)


def fig_exp2_paired(out: Path, manifests) -> None:
    table = pd.read_csv(PROJECT_ROOT / "results/experiment_2/all_block_tables.csv")
    conds = [c for c in CONDITION_ORDER if c in set(table["condition"])]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for cond in conds:
        sub = table[(table["condition"] == cond) & (table["seed"] == 0)]
        p = sub[sub["method"] == "pinn"].sort_values("event_id")
        l = sub[sub["method"] == "least_squares"].sort_values("event_id")
        n = min(len(p), len(l))
        if n:
            ax.plot(p["event_id"].to_numpy()[:n], (p["rmse"].to_numpy()[:n] - l["rmse"].to_numpy()[:n]),
                    "o-", ms=3, label=cond)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("walk-forward event")
    ax.set_ylabel("RMSE(PINN) − RMSE(LS)")
    ax.set_title("Experiment 2: paired per-block difference, PINN vs LS (seed 0)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, out, "fig5_exp2_paired_diffs",
          "Per-block RMSE difference PINN minus LS (positive: PINN worse)",
          "results/experiment_2/all_block_tables.csv", manifests)


def fig_exp4_decomposition(out: Path, manifests) -> None:
    df = pd.read_csv(PROJECT_ROOT / "results/experiment_4/qae_main.csv")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    inputs = list(dict.fromkeys(df["sigma_input"]))
    maturities = sorted(df["maturity"].unique())
    width = 0.8 / len(maturities)
    for j, tau in enumerate(maturities):
        xs, cal, disc, qae = [], [], [], []
        for i, inp in enumerate(inputs):
            sub = df[(df["sigma_input"] == inp) & (df["maturity"] == tau)].iloc[0]
            xs.append(i + (j - len(maturities) / 2 + 0.5) * width)
            cal.append(abs(sub["calibration_error"]))
            disc.append(abs(sub["discretization_error"]))
            qae.append(abs(sub["qae_estimation_error"]))
        ax.bar(xs, cal, width * 0.92, label=f"calibration (τ={tau})" if j == 0 else None,
               color="#1f77b4", alpha=0.9)
        ax.bar(xs, disc, width * 0.92, bottom=cal, label="discretization" if j == 0 else None,
               color="#ff7f0e", alpha=0.9)
        ax.bar(xs, qae, width * 0.92, bottom=np.array(cal) + np.array(disc),
               label="QAE estimation" if j == 0 else None, color="#2ca02c", alpha=0.9)
    ax.set_xticks(range(len(inputs)))
    ax.set_xticklabels([METHOD_LABELS.get(i, i) for i in inputs], fontsize=9)
    ax.set_ylabel("|error| (price units)")
    ax.set_title("Experiment 4: QAE pricing error decomposition by sigma input")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out, "fig6_exp4_error_decomposition",
          "Absolute error components (calibration/discretization/QAE) per sigma input and maturity",
          "results/experiment_4/qae_main.csv", manifests)


def fig_exp4_convergence(out: Path, manifests) -> None:
    mc = pd.read_csv(PROJECT_ROOT / "results/experiment_4/mc_convergence.csv")
    sweep = pd.read_csv(PROJECT_ROOT / "results/experiment_4/sweep_m.csv")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.loglog(mc["n_paths"], mc["rmse"], "o-", color="#1f77b4", label="classical MC (RMSE, 20 repeats)")
    ax.loglog(sweep["grover_queries"], sweep["total_error"].abs(), "s-", color="#d62728",
              label="QAE total |error| (shots=4096)")
    ax.set_xlabel("oracle / query budget (MC paths, QAE Grover queries)")
    ax.set_ylabel("|pricing error| (price units)")
    ax.set_title("Experiment 4: error vs budget — MC and QAE (not a speedup claim)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    _save(fig, out, "fig7_exp4_convergence",
          "Error vs computational budget for classical MC and canonical QAE",
          "results/experiment_4/{mc_convergence,sweep_m}.csv", manifests)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate figures from saved results")
    parser.add_argument("--output", default="figures")
    args = parser.parse_args()
    out = PROJECT_ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)

    manifests: List[Dict[str, Any]] = []
    steps = [
        ("exp0", fig_exp0), ("exp1", fig_exp1),
        ("exp2", fig_exp2_rmse), ("exp2", fig_exp2_sigma), ("exp2", fig_exp2_paired),
        ("exp4", fig_exp4_decomposition), ("exp4", fig_exp4_convergence),
    ]
    for name, fn in steps:
        try:
            fn(out, manifests)
        except Exception as exc:
            print(f"[skip] {fn.__name__}: {exc}")
    (out / "figures_manifest.json").write_text(json.dumps(manifests, indent=2))
    print(f"{len(manifests)} figures written to {out}")
    for m in manifests:
        print("  -", m["figure"])


if __name__ == "__main__":
    main()
