"""Shared helpers for experiment scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import (
    PROJECT_ROOT,
    _deep_merge,
    config_hash,
    load_config,
    load_yaml,
    resolve_includes,
)
from src.utils.logging import get_logger
from src.utils.reproducibility import derive_seed
from src.data.synthetic_generator import generate_synthetic_dataset

import pandas as pd

# Bump these when result-generating semantics change. Include the generator
# version in the cache key so old panels cannot silently survive a correction.
SYNTHETIC_GENERATOR_VERSION = 2
EXPERIMENT_IMPLEMENTATION_VERSION = 2


def default_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        default=None,
        help="Path to the experiment YAML (defaults per experiment script).",
    )
    p.add_argument("--output-dir", default=None, help="Override output directory.")
    p.add_argument("--seeds", default=None, help="Comma-separated seed list override.")
    return p


def load_experiment_config(path: str) -> Dict[str, Any]:
    cfg = load_config(path)
    return cfg


def experiment_paths(cfg: Dict[str, Any], repo_root: Optional[Path] = None) -> Dict[str, Path]:
    root = repo_root or PROJECT_ROOT
    out = Path(cfg.get("experiment", {}).get("output_dir", "results/experiment"))
    if not out.is_absolute():
        out = root / out
    out.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "output": out,
        "data": root / cfg.get("paths", {}).get("data_dir", "data"),
        "figures": root / cfg.get("paths", {}).get("figures_dir", "figures"),
        "logs": root / cfg.get("paths", {}).get("logs_dir", "logs"),
    }


def scenario_synthetic_config(scenario_file: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a synthetic scenario file and apply deep overrides.

    ``overrides`` may either be the synthetic block itself or wrapped under a
    ``"synthetic"`` key (the form used in experiment condition definitions,
    which mirrors the YAML structure).  Both are accepted.
    """
    cfg = load_yaml(PROJECT_ROOT / "configs" / scenario_file)
    resolved = resolve_includes(cfg, PROJECT_ROOT / "configs")
    syn = resolved.get("synthetic", {})
    if overrides:
        if set(overrides.keys()) == {"synthetic"}:
            overrides = overrides["synthetic"]
        syn = _deep_merge(syn, overrides)
    return syn


def generate_or_load_dataset(
    synthetic_cfg: Dict[str, Any],
    seed: int,
    master_seed: int,
    cache_dir: Path,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], str]:
    """Generate (and cache) a synthetic dataset deterministically."""
    scenario = synthetic_cfg["scenario"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = json.dumps(
        {"generator_version": SYNTHETIC_GENERATOR_VERSION, "config": synthetic_cfg},
        sort_keys=True,
        default=str,
    )
    import hashlib

    h = hashlib.sha256(tag.encode()).hexdigest()[:12]
    obs_path = cache_dir / f"{scenario}_seed{seed}_{h}_observations.csv"
    dates_path = cache_dir / f"{scenario}_seed{seed}_{h}_dates.csv"
    meta_path = cache_dir / f"{scenario}_seed{seed}_{h}_meta.json"
    if obs_path.exists() and dates_path.exists() and meta_path.exists():
        if logger:
            logger.info("loading cached dataset %s", obs_path.name)
        obs = pd.read_csv(obs_path, parse_dates=["timestamp", "expiration"])
        dates = pd.read_csv(dates_path, parse_dates=["timestamp"])
        meta = json.loads(meta_path.read_text())
    else:
        if logger:
            logger.info("generating dataset scenario=%s seed=%s", scenario, seed)
        obs, dates, meta = generate_synthetic_dataset(
            synthetic_cfg, seed=seed, master_seed=master_seed
        )
        obs.to_csv(obs_path, index=False)
        dates.to_csv(dates_path, index=False)
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    return obs, dates, meta, str(obs_path)


def seeds_from_config(cfg: Dict[str, Any], cli_seeds: Optional[str]) -> List[int]:
    if cli_seeds:
        return [int(s) for s in cli_seeds.split(",")]
    exp = cfg.get("experiment", {})
    for key in ("seeds",):
        if key in exp:
            return [int(s) for s in exp[key]]
    # each experiment section may carry its own seeds
    for section in cfg.values():
        if isinstance(section, dict) and "seeds" in section:
            return [int(s) for s in section["seeds"]]
    return [0]


def run_dir(output: Path, idx: int) -> Path:
    d = output / f"run_{idx:03d}"
    d.mkdir(parents=True, exist_ok=True)
    return d
