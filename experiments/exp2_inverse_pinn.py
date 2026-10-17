"""Experiment 2 — rolling inverse-PINN calibration vs matched comparators.

    python -m experiments.exp2_inverse_pinn --config configs/experiments/exp2_inverse_pinn.yaml
    python -m experiments.exp2_inverse_pinn --config ... --conditions constant_clean --seeds 0,1

The core experiment: under IDENTICAL walk-forward windows, compare
  least_squares : rolling constrained LS calibration (primary comparator)
  pinn          : rolling inverse PINN (physics-first staged training)
  frozen        : LS fit on the first window only (adaptation ablation)
  data_only_nn  : capacity-matched network without physics losses (H3 ablation)

All sigma-based methods price evaluation blocks with the SAME analytical
evaluator; the data-only network prices directly.  Direct PINN network prices
are reported separately from BS(sigma_PINN) prices.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import (  # noqa: E402
    EXPERIMENT_IMPLEMENTATION_VERSION,
    default_parser,
    experiment_paths,
    generate_or_load_dataset,
    load_experiment_config,
    run_dir,
    scenario_synthetic_config,
    seeds_from_config,
)
from src.calibration.least_squares import calibrate_least_squares  # noqa: E402
from src.calibration.rolling import FrozenEngine, RollingLSEngine, price_with_sigma  # noqa: E402
from src.data.splits import WalkForwardSplitter  # noqa: E402
from src.evaluation.calibration_metrics import (  # noqa: E402
    adaptation_lag,
    sigma_error_metrics,
    stability_metrics,
)
from src.evaluation.comparisons import (  # noqa: E402
    BlockEvaluation,
    aggregate_method_metrics,
    assemble_comparison_table,
    evaluate_block_with_network,
    evaluate_block_with_sigma,
)
from src.evaluation.metrics import maturity_moneyness_buckets  # noqa: E402
from src.evaluation.reporting import experiment_manifest  # noqa: E402
from src.evaluation.uncertainty import paired_block_differences, seed_variability  # noqa: E402
from src.pinn.diagnostics import price_diagnostics  # noqa: E402
from src.pinn.training import TrainConfig, train_pinn_window  # noqa: E402
from src.utils.config import config_hash  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import derive_seed, set_all_seeds  # noqa: E402
from src.utils.serialization import save_json, save_rows_csv  # noqa: E402


def run_condition_seed(
    condition: dict,
    seed: int,
    cfg: dict,
    paths: dict,
    logger,
) -> dict:
    """Run one (condition, seed) walk-forward experiment; returns run summary."""
    master_seed = cfg["project"]["master_seed"]
    syn_cfg = scenario_synthetic_config(condition["scenario"], condition.get("overrides"))
    obs, dates_df, meta, dataset_path = generate_or_load_dataset(
        syn_cfg, seed=seed, master_seed=master_seed, cache_dir=paths["data"] / "synthetic",
        logger=logger,
    )
    obs = obs.sort_values("timestamp").reset_index(drop=True)

    cal_cfg = cfg["calibration"]
    rolling = RollingLSEngine(cal_cfg)
    frozen = FrozenEngine(cal_cfg)
    splitter = WalkForwardSplitter(
        obs["timestamp"],
        window_dates=rolling.window_dates,
        step_dates=rolling.step_dates,
        horizon_dates=rolling.horizon_dates,
        min_window_observations=rolling.min_obs,
    )
    events = splitter.events()

    pinn_cfg = TrainConfig.from_pinn_config(cfg["pinn"])
    nn_cfg = TrainConfig.from_pinn_config(cfg["pinn"])
    nn_cfg.use_pde = False
    nn_cfg.use_boundary = False
    nn_cfg.use_terminal = False
    nn_cfg.stage_b_adam_epochs = 0
    nn_cfg.stage_b_lbfgs_iterations = 0

    rows: list[BlockEvaluation] = []
    sigma_records = []
    first_history = None
    pinn_warm_sigma = None
    pinn_models_runtime = 0.0
    nn_runtime = 0.0

    inst_by_date = dict(zip(dates_df["timestamp"], dates_df["sigma_instantaneous"]))

    for split in events:
        window = obs.iloc[split.calibration_index]
        block = obs.iloc[split.evaluation_index]

        # ---------------- classical calibrators ----------------
        ls_res = rolling.calibrate_window(window)
        if frozen._frozen_sigma is None:
            frozen_fresh = calibrate_least_squares(
                window["spot"], window["strike"], window["tau"], window["rate"],
                window["market_price"],
                sigma_bounds=tuple(cal_cfg["sigma_bounds"]),
            )
            frozen._frozen_sigma = frozen_fresh.sigma
            frozen_runtime = frozen_fresh.runtime_seconds
        else:
            frozen_runtime = 0.0
        frozen_sigma = frozen._frozen_sigma

        # ---------------- inverse PINN (warm start) ----------------
        t0 = time.perf_counter()
        pinn_res, pinn_model = train_pinn_window(
            window, pinn_cfg, seed=derive_seed(master_seed, "pinn", condition["name"], seed, split.event_id),
            trainable_sigma=True, sigma_init_override=pinn_warm_sigma, return_model=True,
        )
        pinn_rt = time.perf_counter() - t0
        pinn_models_runtime += pinn_rt
        if pinn_res.status != "failed" and pinn_res.final_sigma is not None:
            pinn_warm_sigma = pinn_res.final_sigma

        # ---------------- data-only network ----------------
        t0 = time.perf_counter()
        nn_res, nn_model = train_pinn_window(
            window, nn_cfg, seed=derive_seed(master_seed, "nn", condition["name"], seed, split.event_id),
            trainable_sigma=False, sigma_init_override=None, return_model=True,
        )
        nn_rt = time.perf_counter() - t0
        nn_runtime += nn_rt

        if first_history is None and pinn_res.history:
            first_history = pinn_res.history

        # ---------------- references for sigma error ----------------
        cutoff = split.cutoff
        inst_ref = float(inst_by_date.get(cutoff, np.nan))
        block_eff = float(np.sqrt(np.mean(block["true_sigma"].to_numpy(dtype=float) ** 2)))
        regime = str(block["regime"].iloc[0])

        for method, sigma, rt in [
            ("least_squares", ls_res.sigma, ls_res.runtime_seconds),
            ("pinn", pinn_res.final_sigma if pinn_res.status != "failed" else None, pinn_rt),
            ("frozen", frozen_sigma, frozen_runtime),
        ]:
            ev = evaluate_block_with_sigma(block, sigma)
            direct = None
            if method == "pinn" and pinn_model is not None:
                direct = evaluate_block_with_network(pinn_model, block)["rmse"]
            rows.append(
                BlockEvaluation(
                    event_id=split.event_id, condition=condition["name"], seed=seed,
                    method=method, sigma=sigma, rmse=ev["rmse"], mae=ev["mae"],
                    bias=ev["bias"], relative_error=ev["relative_error"],
                    n_contracts=ev["n_contracts"], runtime_seconds=rt,
                    price_source="bs_with_calibrated_sigma", direct_network_rmse=direct,
                    regime=regime, cutoff=str(cutoff.date()),
                )
            )
        # data-only network prices directly
        nn_ev = evaluate_block_with_network(nn_model, block)
        rows.append(
            BlockEvaluation(
                event_id=split.event_id, condition=condition["name"], seed=seed,
                method="data_only_nn", sigma=None, rmse=nn_ev["rmse"], mae=nn_ev["mae"],
                bias=nn_ev["bias"], relative_error=nn_ev["relative_error"],
                n_contracts=nn_ev["n_contracts"], runtime_seconds=nn_rt,
                price_source="network_direct", regime=regime, cutoff=str(cutoff.date()),
            )
        )

        sigma_records.append(
            {
                "event_id": split.event_id,
                "cutoff": str(cutoff.date()),
                "sigma_ls": ls_res.sigma,
                "sigma_pinn": pinn_res.final_sigma if pinn_res.status != "failed" else None,
                "sigma_frozen": frozen_sigma,
                "sigma_ref_instantaneous": inst_ref,
                "sigma_ref_effective_block": block_eff,
                "regime": regime,
                "pinn_status": pinn_res.status,
                "pinn_stage_a_sigma": pinn_res.stage_a_sigma,
                "pinn_final_data_loss": pinn_res.final_losses.get("data"),
                "pinn_final_pde_loss": pinn_res.final_losses.get("pde"),
                "pinn_direct_network_rmse": rows[-2].direct_network_rmse if method == "pinn" else None,
            }
        )

    table = assemble_comparison_table(rows)
    sig_df = pd.DataFrame(sigma_records)

    # ---------------- per-run metrics ----------------
    per_method = {}
    for method, sub in table.groupby("method"):
        per_method[method] = {
            "pooled_rmse": float(np.sqrt(np.mean(sub["rmse"] ** 2))),
            "mean_block_rmse": float(sub["rmse"].mean()),
            "mean_runtime_seconds": float(sub["runtime_seconds"].mean()),
            "n_failed": int(sub["sigma"].isna().sum()),
        }
        if method == "pinn":
            per_method[method]["direct_network_pooled_rmse"] = float(
                np.sqrt(np.nanmean(sub["direct_network_rmse"] ** 2))
            )

    sigma_err = sigma_error_metrics(
        sig_df["sigma_pinn"].tolist(),
        {
            "instantaneous": sig_df["sigma_ref_instantaneous"].tolist(),
            "effective_block": sig_df["sigma_ref_effective_block"].tolist(),
        },
    )
    sigma_err_ls = sigma_error_metrics(
        sig_df["sigma_ls"].tolist(),
        {
            "instantaneous": sig_df["sigma_ref_instantaneous"].tolist(),
            "effective_block": sig_df["sigma_ref_effective_block"].tolist(),
        },
    )

    return {
        "condition": condition["name"],
        "seed": seed,
        "n_events": len(events),
        "n_observations": int(len(obs)),
        "dataset_path": dataset_path,
        "methods": per_method,
        "sigma_error_pinn": sigma_err,
        "sigma_error_least_squares": sigma_err_ls,
        "stability_pinn": stability_metrics(sig_df["sigma_pinn"].tolist()),
        "stability_ls": stability_metrics(sig_df["sigma_ls"].tolist()),
        "pinn_total_runtime": pinn_models_runtime,
        "nn_total_runtime": nn_runtime,
        "table": table,
        "sigma_history": sig_df,
        "first_training_history": first_history,
        "generation_meta": meta,
    }


def load_runs_from_disk(output: Path, conditions: list, seeds: list, logger) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Load every completed run's artifacts from disk into pooled frames.

    This makes aggregation independent of which worker process produced which
    runs, and makes a bare re-run (all completed) still rebuild summary.json.
    """
    tables, sigmas, n_runs = [], [], 0
    for condition in conditions:
        for seed in seeds:
            run_name = f"{condition['name']}__seed{seed}"
            rd = output / run_name
            bt, sh = rd / "block_table.csv", rd / "sigma_history.csv"
            if not (rd / "metrics.json").exists():
                logger.warning("run %s missing metrics.json (incomplete?) — excluded from summary", run_name)
                continue
            metrics = json.loads((rd / "metrics.json").read_text())
            if metrics.get("implementation_version") != EXPERIMENT_IMPLEMENTATION_VERSION:
                logger.warning("run %s was produced by an older implementation; excluded", run_name)
                continue
            if bt.exists() and sh.exists():
                bt_df = pd.read_csv(bt)
                sh_df = pd.read_csv(sh)
                if "condition" not in sh_df.columns:
                    sh_df = sh_df.assign(condition=condition["name"], seed=seed)
                if "condition" not in bt_df.columns:
                    bt_df = bt_df.assign(condition=condition["name"], seed=seed)
                tables.append(bt_df)
                sigmas.append(sh_df)
                n_runs += 1
            else:
                logger.warning("run %s has metrics.json but missing tables", run_name)
    if tables:
        return pd.concat(tables, ignore_index=True), pd.concat(sigmas, ignore_index=True), n_runs
    return pd.DataFrame(), pd.DataFrame(), 0


def aggregate_from_disk(paths: dict, cfg: dict, conditions: list, seeds: list, logger) -> dict | None:
    """Rebuild the cross-condition summary from all completed run dirs."""
    exp2 = cfg["exp2"]
    table, sigma_all, n_runs = load_runs_from_disk(paths["output"], conditions, seeds, logger)
    if n_runs == 0:
        logger.info("no completed runs found on disk; nothing to aggregate")
        return None

    table.to_csv(paths["output"] / "all_block_tables.csv", index=False)
    sigma_all.to_csv(paths["output"] / "all_sigma_histories.csv", index=False)
    logger.info("aggregating %d completed runs (%d blocks) from disk", n_runs, len(table))

    agg = aggregate_method_metrics(table)
    agg.to_csv(paths["output"] / "method_summary.csv", index=False)

    summary: dict = {
        "conditions": [c["name"] for c in conditions],
        "seeds": list(seeds),
        "methods": exp2["methods"],
        "n_completed_runs": n_runs,
        "n_expected_runs": len(conditions) * len(seeds),
    }
    # per condition: pooled metrics + paired block differences (PINN vs LS)
    per_condition = {}
    for cond, sub in table.groupby("condition"):
        entry = {}
        for method, msub in sub.groupby("method"):
            seed_rmse = msub.groupby("seed")["rmse"].mean()
            entry[method] = {
                "pooled_rmse": float(np.sqrt(np.mean(msub["rmse"] ** 2))),
                "mean_block_rmse": float(msub["rmse"].mean()),
                "seed_rmse_std": float(seed_rmse.std(ddof=1)) if len(seed_rmse) > 1 else 0.0,
                "mean_bias": float(msub["bias"].mean()),
            }
        # paired differences per seed (chronological blocks)
        diffs_by_seed = []
        for sd in sub["seed"].unique():
            ssub = sub[sub["seed"] == sd]
            p = ssub[ssub["method"] == "pinn"].sort_values("event_id")["rmse"].to_numpy()
            l = ssub[ssub["method"] == "least_squares"].sort_values("event_id")["rmse"].to_numpy()
            n = min(len(p), len(l))
            if n:
                diffs_by_seed.append(np.mean(p[:n] - l[:n]))
        entry["paired_block_diff_pinn_minus_ls"] = paired_block_differences(
            sub[sub["method"] == "pinn"].sort_values(["seed", "event_id"])["rmse"].tolist(),
            sub[sub["method"] == "least_squares"].sort_values(["seed", "event_id"])["rmse"].tolist(),
        )
        if diffs_by_seed:
            entry["seed_mean_diff_variability"] = seed_variability(diffs_by_seed)
        # sigma errors
        cond_sigma = sigma_all[sigma_all["condition"] == cond]
        entry["sigma_error_pinn"] = sigma_error_metrics(
            cond_sigma["sigma_pinn"].tolist(),
            {"instantaneous": cond_sigma["sigma_ref_instantaneous"].tolist(),
             "effective_block": cond_sigma["sigma_ref_effective_block"].tolist()},
        )
        entry["sigma_error_ls"] = sigma_error_metrics(
            cond_sigma["sigma_ls"].tolist(),
            {"instantaneous": cond_sigma["sigma_ref_instantaneous"].tolist(),
             "effective_block": cond_sigma["sigma_ref_effective_block"].tolist()},
        )
        per_condition[cond] = entry

    summary["per_condition"] = per_condition

    # Gate 5: constant-volatility recovery check
    gate5 = None
    if "constant_clean" in per_condition:
        tol = float(exp2["gates"]["constant_recovery_abs_tol"])
        cond_sigma = sigma_all[sigma_all["condition"] == "constant_clean"]
        errs = (cond_sigma["sigma_pinn"] - cond_sigma["sigma_ref_instantaneous"]).abs()
        median_err = float(np.nanmedian(errs))
        gate5 = {
            "tolerance": tol,
            "median_abs_sigma_error": median_err,
            "max_abs_sigma_error": float(np.nanmax(errs)),
            "status": "pass" if median_err <= tol else "fail",
        }
    summary["gate5_constant_recovery"] = gate5

    save_json(paths["output"] / "summary.json", summary)
    save_json(paths["output"] / "config.json", cfg)
    save_json(
        paths["output"] / "manifest.json",
        experiment_manifest(
            "exp2_inverse_pinn", cfg, config_hash(cfg), seed=seeds[0] if seeds else 0,
            status="completed",
            extra={"n_runs": int(n_runs), "n_blocks": int(len(table))},
        ),
    )

    logger.info("=== aggregate pooled RMSE by condition/method:")
    for _, row in agg.iterrows():
        logger.info("  %-22s %-14s pooled_rmse=%.4f seed_std=%.4f",
                    row["condition"], row["method"], row["pooled_rmse"], row["seed_rmse_std"])
    if gate5:
        logger.info("Gate 5 (constant recovery): %s (median |sigma err| = %.4f)",
                    gate5["status"].upper(), gate5["median_abs_sigma_error"])
    return summary


def main() -> int:
    parser = default_parser("Experiment 2: rolling inverse-PINN vs LS calibration")
    parser.add_argument("--conditions", default=None, help="Comma-separated condition filter.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Skip running; rebuild summary.json from completed runs on disk.")
    args = parser.parse_args()
    cfg_path = args.config or str(Path(__file__).parent.parent / "configs" / "experiments" / "exp2_inverse_pinn.yaml")
    cfg = load_experiment_config(cfg_path)
    paths = experiment_paths(cfg)
    logger = get_logger("exp2", log_file=paths["logs"] / "exp2.log")

    exp2 = cfg["exp2"]
    # aggregate over the FULL configured condition set (not the CLI filter) so
    # the summary always covers every condition that has completed runs
    all_conditions = exp2["conditions"]
    conditions = all_conditions
    if args.conditions:
        keep = set(args.conditions.split(","))
        conditions = [c for c in conditions if c["name"] in keep]
    seeds = seeds_from_config(cfg, args.seeds)

    if not args.aggregate_only:
        for condition in conditions:
            for seed in seeds:
                run_name = f"{condition['name']}__seed{seed}"
                rd = paths["output"] / run_name
                rd.mkdir(parents=True, exist_ok=True)
                metrics_path = rd / "metrics.json"
                if metrics_path.exists():
                    old_metrics = json.loads(metrics_path.read_text())
                    if old_metrics.get("implementation_version") == EXPERIMENT_IMPLEMENTATION_VERSION:
                        logger.info("skipping completed run %s (resume)", rd.name)
                        continue
                    logger.info("recomputing stale run %s from older implementation", rd.name)
                logger.info("=== condition=%s seed=%d", condition["name"], seed)
                t0 = time.perf_counter()
                result = run_condition_seed(condition, seed, cfg, paths, logger)
                elapsed = time.perf_counter() - t0

                # persist run artefacts
                result["table"].to_csv(rd / "block_table.csv", index=False)
                result["sigma_history"].to_csv(rd / "sigma_history.csv", index=False)
                if result["first_training_history"]:
                    save_rows_csv(rd / "training_history_first_window.csv", result["first_training_history"])
                payload = {k: v for k, v in result.items()
                           if k not in ("table", "sigma_history", "first_training_history")}
                payload["condition"] = condition["name"]
                payload["runtime_seconds"] = elapsed
                payload["implementation_version"] = EXPERIMENT_IMPLEMENTATION_VERSION
                save_json(rd / "metrics.json", payload)
                save_json(rd / "condition_config.json", condition)

                logger.info("condition=%s seed=%d done in %.0fs", condition["name"], seed, elapsed)

    # ALWAYS rebuild the summary from disk (covers runs skipped by resume and
    # runs produced by other worker processes)
    aggregate_from_disk(paths, cfg, all_conditions, seeds, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
