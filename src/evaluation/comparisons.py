"""Method comparisons: assemble matched per-block results across methods.

Every method prices the SAME evaluation blocks with the SAME analytical
evaluator and its own calibrated sigma (or, for the data-only network, direct
network prices).  Nothing method-specific enters the evaluation contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..black_scholes.analytical import bs_price

__all__ = ["BlockEvaluation", "assemble_comparison_table", "aggregate_method_metrics"]


@dataclass
class BlockEvaluation:
    event_id: int
    condition: str
    seed: int
    method: str
    sigma: Optional[float]
    rmse: float
    mae: float
    bias: float
    relative_error: float
    n_contracts: int
    runtime_seconds: float
    price_source: str  # "bs_with_calibrated_sigma" | "network_direct"
    direct_network_rmse: Optional[float] = None
    regime: Optional[str] = None
    cutoff: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def evaluate_block_with_sigma(
    block: pd.DataFrame, sigma: Optional[float], reference_col: str = "true_price"
) -> Dict[str, Any]:
    """Price an evaluation block with the common analytical evaluator."""
    if sigma is None or not np.isfinite(sigma):
        return {"rmse": float("nan"), "mae": float("nan"), "bias": float("nan"),
                "relative_error": float("nan"), "n_contracts": int(len(block))}
    pred = bs_price(
        block["spot"].to_numpy(dtype=float),
        block["strike"].to_numpy(dtype=float),
        block["tau"].to_numpy(dtype=float),
        block["rate"].to_numpy(dtype=float),
        sigma,
        "call",
    )
    ref = block[reference_col].to_numpy(dtype=float)
    e = pred - ref
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "relative_error": float(np.mean(np.abs(e) / np.maximum(np.abs(ref), 0.05))),
        "n_contracts": int(len(block)),
    }


def evaluate_block_with_network(model, block: pd.DataFrame, reference_col: str = "true_price") -> Dict[str, Any]:
    """Direct network prices on an evaluation block (reported separately)."""
    pred = model.price_numpy(
        block["spot"].to_numpy(dtype=float),
        block["tau"].to_numpy(dtype=float),
        block["strike"].to_numpy(dtype=float),
        block["rate"].to_numpy(dtype=float),
    )
    ref = block[reference_col].to_numpy(dtype=float)
    e = pred - ref
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "relative_error": float(np.mean(np.abs(e) / np.maximum(np.abs(ref), 0.05))),
        "n_contracts": int(len(block)),
    }


def assemble_comparison_table(evaluations: List[BlockEvaluation]) -> pd.DataFrame:
    return pd.DataFrame([e.to_dict() for e in evaluations])


def aggregate_method_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, method): pooled and per-seed metrics with seed spread."""
    rows = []
    for (condition, method), sub in table.groupby(["condition", "method"]):
        seed_rmse = sub.groupby("seed")["rmse"].mean()
        rows.append(
            {
                "condition": condition,
                "method": method,
                "n_blocks": int(len(sub)),
                "pooled_rmse": float(np.sqrt(np.mean(sub["rmse"] ** 2))),
                "mean_block_rmse": float(sub["rmse"].mean()),
                "mean_block_mae": float(sub["mae"].mean()),
                "mean_block_bias": float(sub["bias"].mean()),
                "mean_block_relative_error": float(sub["relative_error"].mean()),
                "seed_rmse_mean": float(seed_rmse.mean()),
                "seed_rmse_std": float(seed_rmse.std(ddof=1)) if len(seed_rmse) > 1 else 0.0,
                "mean_runtime_seconds": float(sub["runtime_seconds"].mean()),
                "n_failed_sigma": int(sub["sigma"].isna().sum()),
            }
        )
    return pd.DataFrame(rows)
