"""Error decomposition for the quantum pipeline (spec section 25).

Separates, for a single contract priced through the quantum module:

  calibration error      : V_ref(sigma_input) - V_ref(sigma_reference)
                           (effect of the input parameter on the model price)
  discretization error   : V_exact_discrete - V_BS          (same sigma)
  qae estimation error   : V_QAE - V_exact_discrete         (same sigma)
  total error            : V_QAE - V_reference

Each component is computed and reported separately; none is hidden inside a
total.  For synthetic runs the reference is the true generating parameter's
analytical price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["QuantumErrorDecomposition", "decompose_quantum_errors"]


@dataclass
class QuantumErrorDecomposition:
    calibration_error: float
    discretization_error: float
    qae_estimation_error: float
    total_error: float
    v_reference: float
    v_model: float
    v_exact_discrete: float
    v_qae: float
    sigma_reference: Optional[float] = None
    sigma_input: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def decompose_quantum_errors(
    v_reference: float,
    v_model: float,
    v_exact_discrete: float,
    v_qae: float,
    sigma_reference: Optional[float] = None,
    sigma_input: Optional[float] = None,
) -> QuantumErrorDecomposition:
    return QuantumErrorDecomposition(
        calibration_error=v_model - v_reference,
        discretization_error=v_exact_discrete - v_model,
        qae_estimation_error=v_qae - v_exact_discrete,
        total_error=v_qae - v_reference,
        v_reference=v_reference,
        v_model=v_model,
        v_exact_discrete=v_exact_discrete,
        v_qae=v_qae,
        sigma_reference=sigma_reference,
        sigma_input=sigma_input,
    )
