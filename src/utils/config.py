"""Configuration loading and layering utilities.

Configuration system
--------------------
Experiments are driven by YAML files in ``configs/``.  An experiment config may
list other config files under ``includes:``; those are loaded first (relative to
the ``configs/`` directory) and deep-merged in order, with later layers and the
top-level file taking precedence.  The fully resolved configuration is hashed
(sha256 over a canonical JSON serialisation) and stored with every result so a
run can always be traced back to its exact settings.

Nothing scientific (volatilities, seeds, architectures, loss weights, windows,
qubit counts, ...) is hard-coded in the source; it lives in these files.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import yaml

# Default repository root: parent of the directory containing this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at top level")
    return data


def resolve_includes(
    config: Mapping[str, Any],
    config_dir: str | os.PathLike = CONFIG_DIR,
    _seen: tuple = (),
) -> Dict[str, Any]:
    """Resolve the ``includes:`` chain of a config, deep-merging layers in order.

    Later layers override earlier ones; the file containing the ``includes``
    key overrides everything it includes.
    """
    includes: Iterable[str] = config.get("includes", []) or []
    if isinstance(includes, str):
        includes = [includes]
    merged: Dict[str, Any] = {}
    for inc in includes:
        inc_key = str(inc)
        if inc_key in _seen:
            raise ValueError(f"Circular config include detected: {inc_key}")
        inc_path = Path(config_dir) / inc_key
        inc_cfg = load_yaml(inc_path)
        inc_resolved = resolve_includes(inc_cfg, config_dir, _seen + (inc_key,))
        merged = _deep_merge(merged, inc_resolved)
    own = {k: v for k, v in config.items() if k != "includes"}
    return _deep_merge(merged, own)


def load_config(
    path: str | os.PathLike,
    config_dir: str | os.PathLike = CONFIG_DIR,
) -> Dict[str, Any]:
    """Load a config file and resolve its include chain."""
    cfg = load_yaml(path)
    return resolve_includes(cfg, config_dir)


def config_hash(config: Mapping[str, Any]) -> str:
    """Stable sha256 hash of the canonical JSON serialisation of a config."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Fetch a nested value with dotted notation, e.g. ``pinn.architecture.hidden``."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def require_keys(config: Mapping[str, Any], dotted_keys: List[str]) -> None:
    """Fail fast with a clear message when required configuration is missing."""
    missing = [k for k in dotted_keys if get_nested(config, k) is None]
    if missing:
        raise KeyError(f"Missing required configuration keys: {missing}")


def config_to_json(config: Mapping[str, Any]) -> str:
    """Pretty JSON serialisation used to freeze configs next to results."""
    return json.dumps(config, indent=2, sort_keys=True, default=str)
