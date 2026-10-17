"""Uncertainty quantification across seeds and chronological blocks.

Respects date dependence: paired differences between methods are computed per
evaluation block, and uncertainty is reported at the block level rather than
treating every quote as independent.  Seed variability is reported as the
standard deviation across repeated fits on identical data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["paired_block_differences", "seed_variability"]


def paired_block_differences(
    block_errors_a: List[float],
    block_errors_b: List[float],
    labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Paired per-block differences d = err_a - err_b (e.g. PINN - LS).

    Positive d means method a has LARGER block error.  Reports mean, median,
    std and a normal-approximation 95% interval on the mean, plus the full
    distribution.  Statistical significance is NOT overstated: blocks are
    chronological, potentially autocorrelated, and n is small.
    """
    a = np.asarray(block_errors_a, dtype=float)
    b = np.asarray(block_errors_b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    d = a - b
    mean = float(np.mean(d)) if n else float("nan")
    se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    out = {
        "n_blocks": int(n),
        "mean_difference": mean,
        "median_difference": float(np.median(d)) if n else float("nan"),
        "std_difference": float(np.std(d, ddof=1)) if n > 1 else float("nan"),
        "mean_diff_ci95_normal_approx": (
            [mean - 1.96 * se, mean + 1.96 * se] if n > 1 and np.isfinite(se) else None
        ),
        "fraction_blocks_a_larger": float(np.mean(d > 0)) if n else float("nan"),
        "block_differences": d.tolist(),
    }
    if labels is not None:
        out["block_labels"] = list(labels[:n])
    return out


def seed_variability(values: List[float]) -> Dict[str, Any]:
    """Spread of a scalar metric across repeated seeds on identical data."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return {"n_seeds": 0}
    return {
        "n_seeds": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "values": arr.tolist(),
    }
