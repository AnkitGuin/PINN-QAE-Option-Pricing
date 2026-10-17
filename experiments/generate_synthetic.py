"""Synthetic dataset generation CLI.

Examples
--------
    python -m experiments.generate_synthetic --config configs/synthetic/constant.yaml
    python -m experiments.generate_synthetic --config configs/synthetic/abrupt_change.yaml --seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments._common import scenario_synthetic_config  # noqa: E402
from src.data.synthetic_generator import generate_synthetic_dataset  # noqa: E402
from src.data.validation import validate_dates_frame, validate_observations  # noqa: E402
from src.utils.config import PROJECT_ROOT, config_hash, load_yaml  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402
from src.utils.reproducibility import dataset_version, environment_manifest, write_manifest  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Generate synthetic European-option datasets.")
    p.add_argument("--config", required=True, help="Scenario config, e.g. configs/synthetic/constant.yaml")
    p.add_argument("--seeds", default="0", help="Comma-separated seeds (default 0).")
    p.add_argument("--master-seed", type=int, default=42)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    logger = get_logger("generate_synthetic")
    cfg = load_yaml(args.config)
    syn_cfg = cfg.get("synthetic", {})
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = PROJECT_ROOT / "data" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    chash = config_hash(syn_cfg)
    all_ok = True
    for seed in [int(s) for s in args.seeds.split(",")]:
        obs, dates, meta = generate_synthetic_dataset(
            syn_cfg, seed=seed, master_seed=args.master_seed
        )
        suffix = f"{syn_cfg['scenario']}_seed{seed}"
        obs_path = out_dir / f"{suffix}_observations.csv"
        dates_path = out_dir / f"{suffix}_dates.csv"
        obs.to_csv(obs_path, index=False)
        dates.to_csv(dates_path, index=False)

        rep = validate_observations(obs, dates)
        rep_d = validate_dates_frame(dates)
        ok = rep["status"] == "pass" and rep_d["status"] == "pass"
        all_ok = all_ok and ok

        manifest = {
            "dataset": suffix,
            "scenario": syn_cfg["scenario"],
            "seed": seed,
            "master_seed": args.master_seed,
            "config_hash": chash,
            "dataset_version": dataset_version(obs_path, chash),
            "n_observations": int(len(obs)),
            "n_dates": int(len(dates)),
            "validation": {"observations": rep, "dates": rep_d},
            "generation_meta": meta,
            "environment": environment_manifest(seed=seed),
            "files": {"observations": str(obs_path), "dates": str(dates_path)},
        }
        write_manifest(manifest_dir / f"{suffix}_manifest.json", manifest)
        logger.info(
            "%s: %d observations, %d dates, validation=%s",
            suffix, len(obs), len(dates), "pass" if ok else "FAIL",
        )

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
