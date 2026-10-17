"""Exact discrete expectation (computed BEFORE any QAE run).

    a_exact   = sum_j p_j h_j
    V_exact_discrete = exp(-R) * P_max * a_exact

This separates the pipeline into:

    continuous Black-Scholes -> discretization -> exact discrete expectation
                              -> QAE estimation

with each error component measurable independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from .discretization import DiscretizedDistribution
from .payoff import PayoffEncoding

__all__ = ["ExactDiscreteResult", "exact_discrete_expectation"]


@dataclass
class ExactDiscreteResult:
    a_exact: float
    price: float  # exp(-R) * P_max * a_exact
    discount: float

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def exact_discrete_expectation(
    dist_disc: DiscretizedDistribution,
    payoff: PayoffEncoding,
    total_discount: float,
) -> ExactDiscreteResult:
    a = float(np.dot(dist_disc.probabilities, payoff.h))
    return ExactDiscreteResult(
        a_exact=a,
        price=float(np.exp(-total_discount) * payoff.p_max * a),
        discount=float(total_discount),
    )
