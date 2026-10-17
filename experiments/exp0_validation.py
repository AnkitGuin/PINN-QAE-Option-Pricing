"""Experiment 0 — Gate 1 validation (analytical BS + Monte Carlo).

    python -m experiments.exp0_validation --config configs/experiments/exp0_validation.yaml

Must pass before any later experiment is considered valid.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import default_parser, experiment_paths, load_experiment_config  # noqa: E402
from src.black_scholes.validation import run_gate1_validation  # noqa: E402
from src.evaluation.reporting import experiment_manifest  # noqa: E402
from src.utils.config import config_hash  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.serialization import save_json  # noqa: E402


def main() -> int:
    parser = default_parser("Experiment 0: Black-Scholes and Monte Carlo validation (Gate 1)")
    args = parser.parse_args()
    cfg_path = args.config or str(Path(__file__).parent.parent / "configs" / "experiments" / "exp0_validation.yaml")
    cfg = load_experiment_config(cfg_path)
    paths = experiment_paths(cfg)
    logger = get_logger("exp0", log_file=paths["logs"] / "exp0.log")

    logger.info("running Gate 1 validation (analytical BS, parity, bounds, limits, MC)")
    report = run_gate1_validation(cfg)
    status = report["status"]

    chash = config_hash(cfg)
    manifest = experiment_manifest(
        "exp0_validation", cfg, chash, seed=cfg.get("project", {}).get("master_seed", 42),
        status=f"gate1_{status}",
    )
    save_json(paths["output"] / "gate1_report.json", report)
    save_json(paths["output"] / "config.json", cfg)
    save_json(paths["output"] / "manifest.json", manifest)

    for check in report["checks"]:
        logger.info("  [%s] %s", check["status"].upper(), check["check"])

    logger.info("Gate 1 status: %s", status.upper())
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
