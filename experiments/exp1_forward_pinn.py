"""Experiment 1 — forward PINN validation with known parameters (Gate 4).

    python -m experiments.exp1_forward_pinn --config configs/experiments/exp1_forward_pinn.yaml

Trains the price network with an externally supplied (true) sigma on a fixed
domain and verifies agreement with analytical Black-Scholes against
pre-declared tolerances.  This is a solver check, not a calibration result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (  # noqa: E402
    default_parser,
    experiment_paths,
    load_experiment_config,
    run_dir,
    seeds_from_config,
)
from src.black_scholes.analytical import bs_price  # noqa: E402
from src.pinn.collocation import CollocationConfig  # noqa: E402
from src.pinn.diagnostics import evaluate_on_independent_collocation  # noqa: E402
from src.pinn.forward import fit_forward_pinn  # noqa: E402
from src.pinn.training import TrainConfig  # noqa: E402
from src.utils.config import config_hash  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import set_all_seeds  # noqa: E402
from src.utils.serialization import save_json, save_rows_csv  # noqa: E402


def build_domain_frame(domain: dict, seed: int, n_points: int = 4000) -> pd.DataFrame:
    """Pseudo-observation frame covering the pricing domain (for collocation
    statistics and optional labels)."""
    rng = np.random.default_rng(seed)
    S = np.exp(rng.uniform(np.log(domain["s_min"]), np.log(domain["s_max"]), n_points))
    K = np.exp(rng.uniform(np.log(domain["k_min"]), np.log(domain["k_max"]), n_points))
    tau = rng.uniform(domain["tau_min"], domain["tau_max"], n_points)
    r = np.full(n_points, domain["rate"])
    price = bs_price(S, K, tau, r, domain["sigma"] if "sigma" in domain else np.nan)
    return pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2025-01-01"),
            "spot": S,
            "strike": K,
            "tau": tau,
            "rate": r,
            "market_price": price,
            "true_price": price,
        }
    )


def evaluation_grid(domain: dict, cfg_grid: dict) -> pd.DataFrame:
    S = np.linspace(domain["s_min"], domain["s_max"], cfg_grid["s_grid"])
    K = np.linspace(domain["k_min"], domain["k_max"], cfg_grid["k_grid"])
    tau = np.linspace(domain["tau_min"], domain["tau_max"], cfg_grid["tau_grid"])
    Sg, Kg, Tg = [a.ravel() for a in np.meshgrid(S, K, tau, indexing="ij")]
    r = np.full_like(Sg, domain["rate"])
    return pd.DataFrame(
        {
            "spot": Sg,
            "strike": Kg,
            "tau": Tg,
            "rate": r,
            "true_price": bs_price(Sg, Kg, Tg, r, domain["sigma"]),
        }
    )


def main() -> int:
    parser = default_parser("Experiment 1: forward PINN with known parameters (Gate 4)")
    args = parser.parse_args()
    cfg_path = args.config or str(Path(__file__).parent.parent / "configs" / "experiments" / "exp1_forward_pinn.yaml")
    cfg = load_experiment_config(cfg_path)
    paths = experiment_paths(cfg)
    logger = get_logger("exp1", log_file=paths["logs"] / "exp1.log")

    exp1 = cfg["exp1"]
    known = exp1["known_parameters"]
    domain = dict(exp1["domain"])
    domain["sigma"] = float(known["sigma"])
    domain["rate"] = float(known["rate"])
    seeds = seeds_from_config(cfg, args.seeds)
    acceptance = exp1["acceptance"]

    train_cfg = TrainConfig.from_pinn_config(cfg["pinn"])
    tc = exp1.get("training", {})
    if "epochs" in tc:
        train_cfg.stage_a_adam_epochs = int(tc["epochs"])
    if "early_stopping" in tc and "patience" in tc["early_stopping"]:
        train_cfg.stage_a_patience = int(tc["early_stopping"]["patience"])
    if "collocation" in tc:
        train_cfg.collocation = CollocationConfig.from_dict(
            {**cfg["pinn"]["training"].get("collocation", {}), **tc["collocation"]}
        )

    use_labels = bool(exp1.get("use_data_labels", False))
    if use_labels:
        train_cfg.use_data = True
    else:
        train_cfg.use_data = False

    run_summaries = []
    for si, seed in enumerate(seeds):
        set_all_seeds(seed)
        rd = run_dir(paths["output"], si)
        logger.info("seed %d: forward PINN (sigma=%.2f, r=%.2f)", seed, domain["sigma"], domain["rate"])
        frame = build_domain_frame(domain, seed=seed)
        result, model = fit_forward_pinn(
            frame, train_cfg, seed=seed, known_sigma=domain["sigma"],
            use_data_labels=use_labels, return_model=True,
        )
        # price evaluation on the fixed grid
        grid = evaluation_grid(domain, exp1["evaluation"])
        pred = model.price_numpy(grid["spot"], grid["tau"], grid["strike"], grid["rate"])
        err = pred - grid["true_price"].to_numpy(dtype=float)
        result.price_rmse = float(np.sqrt(np.mean(err**2)))
        result.price_mae = float(np.mean(np.abs(err)))
        result.price_max_error = float(np.max(np.abs(err)))
        # independent collocation diagnostics
        indep = evaluate_on_independent_collocation(
            model, frame, train_cfg.collocation, seed=seed + 10_000
        )
        run_summaries.append(
            {
                "seed": seed,
                "sigma": domain["sigma"],
                "price_rmse": result.price_rmse,
                "price_mae": result.price_mae,
                "price_max_error": result.price_max_error,
                "pde_residual_rms_independent": indep["pde_residual_rms"],
                "boundary_error": indep["boundary_error"],
                "terminal_error": indep["terminal_error"],
                "runtime_seconds": result.runtime_seconds,
                "status": result.status,
            }
        )
        save_json(rd / "metrics.json", run_summaries[-1] | {"independent_diagnostics": indep})
        save_rows_csv(rd / "training_history.csv", [
            {k: v for k, v in h.items()} for h in []
        ])
        # full history saved from a re-run container (kept small): save last window's
        save_json(rd / "config_used.json", {"known_parameters": known, "domain": domain})

    rmse_vals = [r["price_rmse"] for r in run_summaries]
    max_errs = [r["price_max_error"] for r in run_summaries]
    resid = [r["pde_residual_rms_independent"] for r in run_summaries]
    summary = {
        "gate": 4,
        "name": "forward_pinn_reproduces_black_scholes",
        "n_seeds": len(run_summaries),
        "mean_price_rmse": float(np.mean(rmse_vals)),
        "max_price_rmse": float(np.max(rmse_vals)),
        "mean_max_error": float(np.mean(max_errs)),
        "mean_independent_pde_residual_rms": float(np.mean(resid)),
        "acceptance": acceptance,
        "runs": run_summaries,
        "use_data_labels": use_labels,
    }
    summary["status"] = (
        "pass"
        if (
            np.max(rmse_vals) <= acceptance["price_rmse_max"]
            and np.max(max_errs) <= acceptance["price_max_error_max"]
            and np.max(resid) <= acceptance["pde_residual_rms_max"]
        )
        else "fail"
    )
    save_json(paths["output"] / "summary.json", summary)
    save_json(paths["output"] / "config.json", cfg)
    from src.evaluation.reporting import experiment_manifest

    save_json(
        paths["output"] / "manifest.json",
        experiment_manifest(
            "exp1_forward_pinn", cfg, config_hash(cfg),
            seed=seeds[0] if seeds else 0, status=f"gate4_{summary['status']}",
        ),
    )
    logger.info(
        "Gate 4: mean RMSE %.4f (max %.4f), mean max-err %.3f, residual RMS %.5f -> %s",
        summary["mean_price_rmse"], summary["max_price_rmse"],
        summary["mean_max_error"], summary["mean_independent_pde_residual_rms"],
        summary["status"].upper(),
    )
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
