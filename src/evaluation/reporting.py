"""Structured reporting: hypothesis-style summaries with neutral language.

Distinguishes observed result / expected hypothesis / interpretation /
limitation, and never claims superiority that the measurements do not support.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["hypothesis_report", "comparison_statement", "experiment_manifest"]


def hypothesis_report(
    hypothesis: str,
    observed: Dict[str, Any],
    interpretation: str,
    limitation: str,
    supports: Optional[bool] = None,
) -> Dict[str, Any]:
    return {
        "hypothesis": hypothesis,
        "observed_result": observed,
        "supports_hypothesis": supports,
        "interpretation": interpretation,
        "limitation": limitation,
    }


def comparison_statement(
    metric_a: float,
    metric_b: float,
    name_a: str,
    name_b: str,
    metric_name: str = "RMSE",
    ci: Optional[List[float]] = None,
) -> str:
    """Neutral comparative sentence; no superiority claim beyond the numbers."""
    diff = metric_a - metric_b
    if not np.isfinite(diff):
        return f"{name_a} could not be compared to {name_b} on {metric_name} (missing values)."
    rel = abs(diff) / max(abs(metric_b), 1e-12)
    if rel < 0.02:
        qualifier = "similar"
    elif diff < 0:
        qualifier = "lower"
    else:
        qualifier = "higher"
    s = (
        f"In this experiment, {name_a} produced {qualifier} {metric_name} than {name_b} "
        f"({metric_a:.4f} vs {metric_b:.4f}; difference {diff:+.4f})."
    )
    if ci is not None:
        s += f" Block-level 95% interval on the difference: [{ci[0]:.4f}, {ci[1]:.4f}]."
    s += " This is an observed result under the stated conditions, not a general claim."
    return s


def experiment_manifest(
    experiment: str,
    config: Dict[str, Any],
    config_hash: str,
    seed: int,
    status: str,
    dataset_version: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    from ..utils.reproducibility import environment_manifest

    manifest = {
        "experiment": experiment,
        "dataset_version": dataset_version,
        "config_hash": config_hash,
        "seed": seed,
        "device": config.get("project", {}).get("device", "cpu"),
        "status": status,
        "environment": environment_manifest(seed=seed),
    }
    if extra:
        manifest.update(extra)
    return manifest
