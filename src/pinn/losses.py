"""PINN loss components in homogeneous coordinates (exact chain rules).

PDE (maturity-time form, tau = T - t, hence V_t = -V_tau):

    -V_tau + 0.5 sigma^2 S^2 V_SS + r S V_S - r V = 0

With V = S * u(x, tau, r), x = ln(S/K), the exact chain rule gives

    V_S  = u + u_x
    V_SS = (u_x + u_xx) / S
    V_tau= S * u_tau

so the PDE reduces to the dimensionless form

    f~ = -u_tau + 0.5 sigma^2 (u_x + u_xx) + r * u_x = 0.

Normalizations and chain rule for the network inputs
----------------------------------------------------
Inputs z = (z1, z2, z3) with

    z1 = (x - m_x)/s_x,  z2 = (tau - m_tau)/s_tau,  z3 = (r - m_r)/s_r,

so (partial derivatives of the network output u w.r.t. z, written u_1, u_2):

    u_x   = u_1 / s_x
    u_xx  = u_11 / s_x^2
    u_tau = u_2 / s_tau

The residual is scaled by s_tau for O(1) magnitude:

    f_hat = s_tau * f~

All loss components are dimensionless:
    data     : mean((u_i - y_i/S_i)^2)
    pde      : mean(f_hat^2)
    boundary : mean((u - g/P)^2) with g the physical boundary value
    terminal : mean((u - payoff/S)^2)

Boundary and terminal conditions (European call, no dividends):
    u(x, 0)      = (1 - e^{-x})^+                    (terminal payoff / S)
    u(x_lo, tau) = 0                                  (deep OTM: V/S ~ 0)
    u(x_hi, tau) = 1 - e^{-r tau - x_hi}              (large-S parity floor;
                                                      error = put/S at x_hi,
                                                      negligible for x_hi
                                                      deep enough - recorded)

The payoff kink at tau = 0 (x = 0) is NEVER treated as an interior PDE point;
terminal samples are imposed separately from interior collocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .network import PINNModel

__all__ = ["LossComponents", "compute_losses", "pde_residual"]


@dataclass
class LossComponents:
    data: float
    pde: float
    boundary: float
    terminal: float
    total: float
    sigma: float
    pde_residual_rms: float

    def to_dict(self) -> Dict:
        return dict(self.__dict__)


def pde_residual(
    model: PINNModel,
    S: torch.Tensor,
    tau: torch.Tensor,
    K: torch.Tensor,
    r: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Scaled dimensionless PDE residual f_hat at interior collocation points.

    x = ln(S/K) and tau are differentiated (PDE variables); K enters only
    through x (the model is homogeneous, so K's absolute level is not a PDE
    variable), and r is a conditioning input held fixed per contract.
    """
    sc = model.scaler
    s_x = sc.std_log_moneyness
    s_tau = sc.std_tau

    z = model.inputs_from_physical(S, tau, K, r)
    z_in = z.clone().requires_grad_(True)

    u = model.net(z_in).squeeze(-1)
    grad_u = torch.autograd.grad(u.sum(), z_in, create_graph=True, retain_graph=True)[0]
    u_1 = grad_u[:, 0]
    u_2 = grad_u[:, 1]
    u_11 = torch.autograd.grad(u_1.sum(), z_in, create_graph=True, retain_graph=True)[0][:, 0]

    u_x = u_1 / s_x
    u_xx = u_11 / (s_x**2)
    u_tau = u_2 / s_tau

    f_tilde = -u_tau + 0.5 * sigma**2 * (u_x + u_xx) + r * u_x
    return s_tau * f_tilde


def compute_losses(
    model: PINNModel,
    data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    interior: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    boundary: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    terminal: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    weights: Dict[str, float],
    use_data: bool = True,
    use_pde: bool = True,
    use_boundary: bool = True,
    use_terminal: bool = True,
) -> Tuple[torch.Tensor, LossComponents]:
    """Weighted total loss with all components logged separately."""
    loss_data = torch.zeros((), dtype=torch.float64)
    loss_pde = torch.zeros((), dtype=torch.float64)
    loss_bc = torch.zeros((), dtype=torch.float64)
    loss_terminal = torch.zeros((), dtype=torch.float64)
    resid_rms = torch.zeros((), dtype=torch.float64)

    sigma = model.sigma()

    if use_data and data is not None:
        S, tau, K, r, y = data
        u_hat = model.u(S, tau, K, r)
        u_obs = y / S  # dimensionless observed price
        loss_data = torch.mean((u_hat - u_obs) ** 2)

    if use_pde:
        S_f, tau_f, K_f, r_f = interior
        f_hat = pde_residual(model, S_f, tau_f, K_f, r_f, sigma)
        loss_pde = torch.mean(f_hat**2)
        resid_rms = torch.sqrt(loss_pde.detach())

    if use_boundary and boundary is not None:
        S_b, tau_b, K_b, r_b, g_b = boundary  # g_b in physical price units
        u_b = model.u(S_b, tau_b, K_b, r_b)
        loss_bc = torch.mean((u_b - g_b / S_b) ** 2)

    if use_terminal and terminal is not None:
        S_T, K_T, r_T, payoff_T = terminal  # payoff in physical price units
        tau_zero = torch.zeros_like(S_T)
        u_T = model.u(S_T, tau_zero, K_T, r_T)
        loss_terminal = torch.mean((u_T - payoff_T / S_T) ** 2)

    total = (
        weights.get("lambda_data", 1.0) * loss_data
        + weights.get("lambda_pde", 1.0) * loss_pde
        + weights.get("lambda_boundary", 1.0) * loss_bc
        + weights.get("lambda_terminal", 1.0) * loss_terminal
    )

    components = LossComponents(
        data=float(loss_data.detach()),
        pde=float(loss_pde.detach()),
        boundary=float(loss_bc.detach()),
        terminal=float(loss_terminal.detach()),
        total=float(total.detach()),
        sigma=float(sigma.detach()),
        pde_residual_rms=float(resid_rms),
    )
    return total, components
