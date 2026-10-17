"""Experiment 4: simulator-based QAE option pricing (quantum experiment).

Pipeline: from the exp2 constant_clean seed-0 artifacts (sigma histories and
the first evaluation block), price three near-ATM contracts through the
quantum module under four sigma inputs (known, least-squares, PINN, frozen),
with a full error decomposition:

  calibration error     V_BS(sigma_input) - V_BS(sigma_reference)
  discretization error  V_exact_discrete(sigma_input) - V_BS(sigma_input)
  QAE estimation error  V_QAE - V_exact_discrete(sigma_input)
  total error           V_QAE - V_BS(sigma_reference)

All quantum settings (grid size, truncation, m, shots, seed, estimator) are
held fixed across sigma inputs so output differences are attributable to
calibration inputs only.  Known-sigma sweeps over m and n, classical Monte
Carlo convergence, and resource accounting complete the picture.  No quantum
advantage or speedup is claimed anywhere; simulator timings and query counts
are reported as accounting facts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from experiments._common import (
    default_parser,
    experiment_paths,
    generate_or_load_dataset,
    load_experiment_config,
    scenario_synthetic_config,
)
from src.calibration.rolling import RollingLSEngine
from src.data.splits import WalkForwardSplitter
from src.evaluation.error_decomposition import decompose_quantum_errors
from src.evaluation.reporting import experiment_manifest
from src.quantum import (
    LognormalTerminalDistribution,
    aer_smoke_run,
    build_qae_circuits,
    discretize,
    encode_call_payoff,
    exact_discrete_expectation,
    resource_report,
    run_qae,
)
from src.utils.logging import get_logger
from src.utils.reproducibility import derive_seed
from src.black_scholes.analytical import bs_price
from src.black_scholes.monte_carlo import mc_price

SIGMA_INPUTS = ["known_sigma", "least_squares", "pinn", "frozen"]


def load_first_block_contracts(cfg: Dict[str, Any], paths: Dict[str, Path], logger) -> tuple[pd.DataFrame, str]:
    """Regenerate the source dataset and extract the first evaluation block.

    The condition definition (scenario + overrides) is taken from the exp2
    experiment config so the dataset is generated exactly as in exp2.
    """
    exp4 = cfg["exp4"]
    sel = exp4["selection"]
    cond_name = sel["source_condition"]
    seed = int(sel["source_seed"])

    exp2_cfg = load_experiment_config("configs/experiments/exp2_inverse_pinn.yaml")
    conditions = exp2_cfg["exp2"]["conditions"]
    cond = next(c for c in conditions if c["name"] == cond_name)
    syn_cfg = scenario_synthetic_config(cond["scenario"], cond.get("overrides"))

    obs, dates, meta, obs_path = generate_or_load_dataset(
        syn_cfg, seed=seed, master_seed=cfg["project"]["master_seed"], cache_dir=paths["data"] / "synthetic", logger=logger
    )

    cal_cfg = cfg["calibration"]
    rolling = RollingLSEngine(cal_cfg)
    splitter = WalkForwardSplitter(
        obs["timestamp"],
        window_dates=rolling.window_dates,
        step_dates=rolling.step_dates,
        horizon_dates=rolling.horizon_dates,
        min_window_observations=rolling.min_obs,
    )
    events = splitter.events()
    split = events[0]  # first evaluation block (selection rule: block=first)
    block = obs.iloc[split.evaluation_index].copy()

    # nearest-ATM contract per requested maturity
    maturities = list(sel["maturities"])
    rows = []
    for tau in maturities:
        sub = block[np.isclose(block["tau"], tau)]
        if len(sub) == 0:
            # fall back to the closest available maturity (recorded)
            avail = block["tau"].unique()
            tau = float(avail[np.argmin(np.abs(avail - tau))])
            sub = block[np.isclose(block["tau"], tau)]
        m = (sub["strike"] / sub["spot"] - 1.0).abs().idxmin()  # nearest ATM
        rows.append(sub.loc[m])
    contracts = pd.DataFrame(rows).reset_index(drop=True)
    contracts["tau"] = contracts["tau"].astype(float)
    return contracts, obs_path


def load_exp2_sigmas(paths: Dict[str, Path], cfg: Dict[str, Any]) -> Dict[str, float]:
    """Load the first-window sigma estimates saved by exp2."""
    sel = cfg["exp4"]["selection"]
    exp2_dir = paths["root"] / "results" / "experiment_2" / f"{sel['source_condition']}__seed{sel['source_seed']}"
    sh_path = exp2_dir / "sigma_history.csv"
    if not sh_path.exists():
        raise FileNotFoundError(
            f"exp2 artifacts not found at {sh_path}; run experiments/exp2_inverse_pinn.py first"
        )
    sh = pd.read_csv(sh_path)
    first = sh.iloc[0]
    return {
        "least_squares": float(first["sigma_ls"]),
        "pinn": float(first["sigma_pinn"]),
        "frozen": float(first["sigma_frozen"]),
        "true_effective": float(first["sigma_ref_effective_block"]),
    }


def price_contract_quantum(
    S0: float, K: float, tau: float, rate: float, sigma: float,
    qcfg: Dict[str, Any], m: int | None = None, n: int | None = None,
) -> Dict[str, Any]:
    """Price one contract through discretization + QAE at a given sigma."""
    dist = LognormalTerminalDistribution.from_constant_parameters(S0, tau, rate, sigma)
    disc = discretize(
        dist,
        n_qubits=int(n if n is not None else qcfg["distribution"]["n_qubits"]),
        truncation_quantile=float(qcfg["distribution"]["truncation_quantile"]),
    )
    pay = encode_call_payoff(disc.grid, K)
    ex = exact_discrete_expectation(disc, pay, dist.total_discount)
    qaecfg = qcfg["qae"]
    mode = str(qaecfg.get("mode", "shots"))
    res = run_qae(
        disc.probabilities,
        pay.h,
        n_evaluation=int(m if m is not None else qaecfg["evaluation_qubits"]),
        mode=mode,
        shots=int(qaecfg["shots"]),
        simulator_seed=int(qaecfg["simulator_seed"]),
        theta_grid_points=int(qaecfg["theta_grid_points"]),
        a_exact=ex.a_exact,
    )
    discount = float(np.exp(-dist.total_discount))
    v_qae = discount * pay.p_max * res.a_hat
    v_disc = ex.price
    v_model = float(bs_price(S0, K, tau, rate, sigma))
    return {
        "v_qae": v_qae,
        "v_exact_discrete": v_disc,
        "v_model": v_model,
        "a_hat": res.a_hat,
        "a_exact": ex.a_exact,
        "sampling_method": res.sampling_method,
        "qae_runtime_seconds": res.runtime_seconds,
        "tail_mass": disc.tail_mass,
        "grid": disc.grid,
        "probabilities": disc.probabilities,
        "h": pay.h,
        "p_max": pay.p_max,
    }


def main() -> None:
    parser = default_parser("Experiment 4: simulator-based QAE option pricing")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    paths = experiment_paths(cfg)
    logger = get_logger("exp4", paths["logs"] / "exp4.log")
    t_start = time.perf_counter()

    qcfg = cfg["quantum"]
    exp4 = cfg["exp4"]
    out = paths["output"]

    # ---------------- inputs ----------------
    contracts, obs_path = load_first_block_contracts(cfg, paths, logger)
    sigmas = load_exp2_sigmas(paths, cfg)
    sigma_map = {
        "known_sigma": sigmas["true_effective"],
        "least_squares": sigmas["least_squares"],
        "pinn": sigmas["pinn"],
        "frozen": sigmas["frozen"],
    }
    logger.info(
        "sigmas: known=%.6f ls=%.6f pinn=%.6f frozen=%.6f",
        sigma_map["known_sigma"], sigma_map["least_squares"], sigma_map["pinn"], sigma_map["frozen"],
    )

    main_rows: List[Dict[str, Any]] = []
    for _, c in contracts.iterrows():
        S0, K, tau, rate = float(c["spot"]), float(c["strike"]), float(c["tau"]), float(c["rate"])
        sigma_ref = sigma_map["known_sigma"]
        v_ref = float(bs_price(S0, K, tau, rate, sigma_ref))
        for inp in exp4["inputs"]:
            sigma_in = sigma_map[inp]
            q = price_contract_quantum(S0, K, tau, rate, sigma_in, qcfg)
            dec = decompose_quantum_errors(
                v_reference=v_ref,
                v_model=q["v_model"],
                v_exact_discrete=q["v_exact_discrete"],
                v_qae=q["v_qae"],
                sigma_reference=sigma_ref,
                sigma_input=sigma_in,
            )
            main_rows.append({
                "maturity": tau,
                "strike": K,
                "spot": S0,
                "sigma": sigma_in,
                "v_reference": v_ref,
                "v_model_bs": q["v_model"],
                "v_exact_discrete": q["v_exact_discrete"],
                "v_qae": q["v_qae"],
                "a_exact": q["a_exact"],
                "a_hat": q["a_hat"],
                **dec.to_dict(),
                # explicit input label AFTER dec.to_dict(): the decomposition
                # also carries a `sigma_input` field (the float sigma), which
                # must not overwrite the label used for grouping
                "sigma_input": inp,
                "sampling_method": q["sampling_method"],
                "qae_runtime_seconds": q["qae_runtime_seconds"],
                "tail_mass": q["tail_mass"],
            })
            logger.info(
                "tau=%.2f K=%.0f %s: total_err=%+.4f (cal=%+.4f disc=%+.4f qae=%+.4f)",
                tau, K, inp, dec.total_error, dec.calibration_error, dec.discretization_error, dec.qae_estimation_error,
            )
    main_df = pd.DataFrame(main_rows)
    main_df.to_csv(out / "qae_main.csv", index=False)

    # ---------------- sweeps (known sigma) ----------------
    sweeps = exp4["sweeps"]
    m_rows, n_rows = [], []
    c0 = contracts.iloc[len(contracts) // 2]  # middle maturity for sweeps
    S0, K, tau, rate = float(c0["spot"]), float(c0["strike"]), float(c0["tau"]), float(c0["rate"])
    sigma_ref = sigma_map["known_sigma"]
    v_ref = float(bs_price(S0, K, tau, rate, sigma_ref))

    for m in sweeps["evaluation_qubits"]:
        q = price_contract_quantum(S0, K, tau, rate, sigma_ref, qcfg, m=int(m))
        m_rows.append({
            "m": int(m),
            "v_qae": q["v_qae"],
            "total_error": q["v_qae"] - v_ref,
            "qae_estimation_error": q["v_qae"] - q["v_exact_discrete"],
            "a_hat": q["a_hat"], "a_exact": q["a_exact"],
            "shots": int(qcfg["qae"]["shots"]),
            "grover_queries": int(qcfg["qae"]["shots"] * (2 ** int(m) - 1)),
            "a_applications": int(qcfg["qae"]["shots"]),
            "qae_runtime_seconds": q["qae_runtime_seconds"],
        })
    for n in sweeps["distribution_qubits"]:
        q = price_contract_quantum(S0, K, tau, rate, sigma_ref, qcfg, n=int(n))
        n_rows.append({
            "n": int(n),
            "grid_points": 2 ** int(n),
            "v_exact_discrete": q["v_exact_discrete"],
            "discretization_error": q["v_exact_discrete"] - q["v_model"],
            "v_qae": q["v_qae"],
            "total_error": q["v_qae"] - v_ref,
            "tail_mass": q["tail_mass"],
        })
    m_df, n_df = pd.DataFrame(m_rows), pd.DataFrame(n_rows)
    m_df.to_csv(out / "sweep_m.csv", index=False)
    n_df.to_csv(out / "sweep_n.csv", index=False)

    # ---------------- classical Monte Carlo convergence ----------------
    mc_rows = []
    mc_cfg = exp4["monte_carlo"]
    for n_paths in mc_cfg["path_counts"]:
        errs = []
        for rep in range(int(mc_cfg["n_repeats"])):
            seed = derive_seed(cfg["project"]["master_seed"], "exp4_mc", str(n_paths), str(rep))
            res = mc_price(S0, K, tau, rate, sigma=sigma_ref, n_paths=int(n_paths), seed=seed)
            errs.append(res.price - v_ref)
        errs = np.array(errs)
        mc_rows.append({
            "n_paths": int(n_paths),
            "rmse": float(np.sqrt(np.mean(errs**2))),
            "mean_error": float(np.mean(errs)),
            "std_error": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
            "n_repeats": int(mc_cfg["n_repeats"]),
        })
    mc_df = pd.DataFrame(mc_rows)
    mc_df.to_csv(out / "mc_convergence.csv", index=False)

    # ---------------- resource accounting ----------------
    dist = LognormalTerminalDistribution.from_constant_parameters(S0, tau, rate, sigma_map["known_sigma"])
    disc = discretize(
        dist,
        n_qubits=int(qcfg["distribution"]["n_qubits"]),
        truncation_quantile=float(qcfg["distribution"]["truncation_quantile"]),
    )
    pay = encode_call_payoff(disc.grid, K)
    bundle = build_qae_circuits(
        disc.n_qubits, disc.probabilities, pay.h, int(qcfg["qae"]["evaluation_qubits"])
    )
    resources = resource_report(
        bundle,
        shots=int(qcfg["qae"]["shots"]),
        optimization_level=int(qcfg["resources"]["transpile_optimization_level"]),
    )
    smoke = aer_smoke_run(bundle, shots=int(qcfg["qae"]["shots"]), seed=int(qcfg["qae"]["simulator_seed"]))
    resources["aer_execution"] = smoke
    (out / "resources.json").write_text(json.dumps(resources, indent=2, default=str))

    # ---------------- summary ----------------
    def pooled(inp: str) -> Dict[str, float]:
        sub = main_df[main_df["sigma_input"] == inp]
        return {
            "rmse_total": float(np.sqrt(np.mean(sub["total_error"] ** 2))),
            "rmse_calibration": float(np.sqrt(np.mean(sub["calibration_error"] ** 2))),
            "rmse_discretization": float(np.sqrt(np.mean(sub["discretization_error"] ** 2))),
            "rmse_qae": float(np.sqrt(np.mean(sub["qae_estimation_error"] ** 2))),
        }

    summary = {
        "selection": exp4["selection"],
        "contracts": [
            {"tau": float(c["tau"]), "strike": float(c["strike"]), "spot": float(c["spot"])}
            for _, c in contracts.iterrows()
        ],
        "sigma_inputs": sigma_map,
        "pooled_by_input": {inp: pooled(inp) for inp in exp4["inputs"]},
        "sweep_m": m_df.to_dict(orient="records"),
        "sweep_n": n_df.to_dict(orient="records"),
        "mc_convergence": mc_df.to_dict(orient="records"),
        "quantum_settings": {
            "n_qubits": int(qcfg["distribution"]["n_qubits"]),
            "truncation_quantile": float(qcfg["distribution"]["truncation_quantile"]),
            "m": int(qcfg["qae"]["evaluation_qubits"]),
            "mode": qcfg["qae"]["mode"],
            "shots": int(qcfg["qae"]["shots"]),
            "post_processing": qcfg["qae"]["post_processing"],
        },
        "notes": [
            "All sigma inputs use identical quantum settings; differences in QAE output are attributable to inputs.",
            "A embeds distribution loading and payoff encoding exactly (StatePreparation); its transpiled cost is reported in resources.json.",
            "Classical MC uses antithetic sampling of the same lognormal law; comparison is by error vs sample/query budget, not runtime speedup.",
            "No quantum advantage is claimed; simulator resource counts are accounting facts.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    from src.utils.config import config_hash

    manifest = experiment_manifest(
        experiment=cfg["experiment"]["id"],
        config=cfg,
        config_hash=config_hash(cfg),
        seed=int(cfg["project"]["master_seed"]),
        status="completed",
        dataset_version=obs_path,
        extra={
            "runtime_seconds": time.perf_counter() - t_start,
            "n_contracts": int(len(contracts)),
            "n_inputs": len(exp4["inputs"]),
            "depends_on": "results/experiment_2 (constant_clean seed 0 artifacts)",
        },
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (out / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    logger.info("exp4 complete in %.1fs", time.perf_counter() - t_start)
    print("\n=== Experiment 4: QAE pricing (pooled RMSE by input) ===")
    print(main_df.groupby("sigma_input")[
        ["total_error", "calibration_error", "discretization_error", "qae_estimation_error"]
    ].apply(lambda g: g.pow(2).mean().pow(0.5)).round(4).to_string())
    print("\nMC convergence:")
    print(mc_df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
