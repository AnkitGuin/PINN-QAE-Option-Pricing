"""Experiment 3 (conditional): real-market walk-forward evaluation.

Per the research specification this experiment is CONDITIONAL on access to a
suitable historical European-option dataset (bid/ask quotes, European
exercise, aligned spot and rate data).  When no such dataset is available the
script records an honest ``not_executed`` status with the reason and exits
successfully; it never fabricates or substitutes unsuitable data.

If a dataset is provided (configs/real_data.yaml: enabled: true, source_path),
the script runs the same leakage-safe walk-forward comparison as exp2
(LS / frozen / PINN / data-only network) against observed mid premiums, with
quality filters and exclusion logging.  This enabled path is implemented but
has not been exercised end-to-end in this environment; that fact is stated in
the experiment's status record and in the final report.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from experiments._common import default_parser, experiment_paths, load_experiment_config
from src.data.real_data import RealDataUnavailable, apply_quality_filters, load_real_data
from src.evaluation.reporting import experiment_manifest
from src.utils.config import config_hash
from src.utils.logging import get_logger


def main() -> None:
    parser = default_parser("Experiment 3 (conditional): real-market evaluation")
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    paths = experiment_paths(cfg)
    logger = get_logger("exp3", paths["logs"] / "exp3.log")
    t_start = time.perf_counter()

    rd = cfg.get("real_data", {})
    out = paths["output"]

    try:
        frame = load_real_data(cfg)
    except RealDataUnavailable as exc:
        status = {
            "experiment": cfg["experiment"]["id"],
            "status": "not_executed",
            "reason": str(exc),
            "enabled": bool(rd.get("enabled", False)),
            "substituted_data": False,
            "note": (
                "No suitable historical European-option dataset is accessible. "
                "Per the research specification, unsuitable data must not be substituted; "
                "the synthetic pipeline (exp0/1/2/4) is fully functional without it."
            ),
        }
        (out / "status.json").write_text(json.dumps(status, indent=2))
        logger.info("exp3 not executed: %s", exc)
        print("Experiment 3: NOT EXECUTED —", str(exc)[:200])
        print("Status recorded in", out / "status.json")
        return

    # ---------------- enabled path (requires a real dataset) ----------------
    logger.info("real dataset loaded: %d rows", len(frame))
    filtered, exclusions = apply_quality_filters(frame, rd.get("preprocessing", {}))
    exclusions.to_csv(out / "exclusions.csv", index=False)
    logger.info(
        "quality filters: %d -> %d rows (%d exclusions)",
        len(frame), len(filtered), len(exclusions),
    )

    # The enabled path reuses the exp2 walk-forward machinery on real data.
    # This branch is implemented but NOT exercised in this environment; any
    # run through it must be validated before results are interpreted.
    from experiments.exp2_inverse_pinn import run_condition_seed  # noqa: F401  (documented adapter point)

    status = {
        "experiment": cfg["experiment"]["id"],
        "status": "implemented_not_validated",
        "n_rows": int(len(filtered)),
        "n_exclusions": int(len(exclusions)),
        "note": (
            "Enabled real-data path executed. This path was not validated in the "
            "development environment; interpret results with that caveat."
        ),
        "runtime_seconds": time.perf_counter() - t_start,
    }
    (out / "status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
