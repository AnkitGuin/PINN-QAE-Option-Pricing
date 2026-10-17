"""PINN price network and volatility parameterization.

Formulation (homogeneous coordinates)
-------------------------------------
The Black-Scholes call price is exactly degree-1 homogeneous in (S, K):

    C(lambda*S, lambda*K, tau, r) = lambda * C(S, K, tau, r)

for any lambda > 0 (no dividends).  The network therefore represents the
DIMENSIONLESS function

    u(x, tau, r),   x = ln(S/K),

and the physical price is reconstructed exactly as

    V = S * u(x, tau, r).

Under this reparameterization the Black-Scholes PDE (tau-form) becomes

    f~ = -u_tau + 0.5 sigma^2 (u_x + u_xx) + r u_x = 0,

which is derived by the chain rule (see src/pinn/losses.py) and verified
numerically against the analytical solution in tests/test_pinn_derivatives.py.
This is an exact reformulation of the specification's (S, t, K, T, r) input
structure (the specification permits equivalent implementations with tau in
place of t); the homogeneity is exact for the assumed Black-Scholes model and
is documented as a model-scope restriction in docs/methodology.md.

Shared volatility parameter (one per calibration window):

    sigma_hat_w = softplus(a_w) + epsilon_sigma

Inputs are z-scored with window statistics; the output u is already
dimensionless in [0, ~1.5].
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..data.preprocessing import WindowScaler

__all__ = ["PriceNetwork", "SharedSigma", "PINNModel", "inverse_softplus"]


def inverse_softplus(y: float) -> float:
    """Numerically stable inverse of softplus: x = y + log(1 - exp(-y))."""
    y = float(y)
    if y <= 0.0:
        raise ValueError("inverse_softplus requires y > 0")
    if y > 40.0:
        return y
    return y + math.log1p(-math.exp(-y))


class PriceNetwork(nn.Module):
    """MLP mapping normalized (x, tau, r) to the dimensionless price u."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_units: Sequence[int] = (32, 32, 32),
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        act = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}.get(activation)
        if act is None:
            raise ValueError(f"unsupported activation: {activation}")
        layers = []
        prev = input_dim
        for h in hidden_units:
            layers += [nn.Linear(prev, h, dtype=dtype), act()]
            prev = h
        layers += [nn.Linear(prev, 1, dtype=dtype)]
        nn.init.normal_(layers[-1].weight, std=0.1)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SharedSigma(nn.Module):
    """sigma = softplus(a) + epsilon_sigma, a single shared trainable scalar."""

    def __init__(self, init_sigma: float, epsilon_sigma: float = 1e-6):
        super().__init__()
        if init_sigma <= epsilon_sigma:
            raise ValueError("init_sigma must exceed epsilon_sigma")
        a0 = inverse_softplus(init_sigma - epsilon_sigma)
        self.a = nn.Parameter(torch.tensor(a0, dtype=torch.float64))
        self.epsilon_sigma = float(epsilon_sigma)

    def forward(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.a) + self.epsilon_sigma

    @property
    def value(self) -> float:
        with torch.no_grad():
            return float(self.forward())

    def set_value(self, sigma: float) -> None:
        with torch.no_grad():
            self.a.fill_(inverse_softplus(sigma - self.epsilon_sigma))


class PINNModel(nn.Module):
    """Price network + (optionally trainable) shared volatility.

    Coordinates: x = ln(S/K) z-scored, tau z-scored, r z-scored.  The stored
    ``scaler`` holds the window statistics.  Physical price V = S * u.
    """

    def __init__(
        self,
        scaler: WindowScaler,
        hidden_units: Sequence[int] = (32, 32, 32),
        activation: str = "tanh",
        sigma_init: float = 0.20,
        epsilon_sigma: float = 1e-6,
        trainable_sigma: bool = True,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.net = PriceNetwork(3, hidden_units, activation, dtype=dtype)
        self.sigma_module = SharedSigma(sigma_init, epsilon_sigma)
        if not trainable_sigma:
            for p in self.sigma_module.parameters():
                p.requires_grad_(False)
        self.scaler = scaler
        self.trainable_sigma = trainable_sigma

    def sigma(self) -> torch.Tensor:
        return self.sigma_module()

    def inputs_from_physical(self, S, tau, K, r) -> torch.Tensor:
        sc = self.scaler
        x = torch.log(S / K)
        x1 = (x - sc.mean_log_moneyness) / sc.std_log_moneyness
        x2 = (tau - sc.mean_tau) / sc.std_tau
        x3 = (r - sc.mean_rate) / sc.std_rate
        return torch.stack([x1, x2, x3], dim=1)

    def u(self, S, tau, K, r) -> torch.Tensor:
        return self.net(self.inputs_from_physical(S, tau, K, r)).squeeze(-1)

    def price(self, S, tau, K, r) -> torch.Tensor:
        """Physical price V = S * u (differentiable)."""
        return S * self.u(S, tau, K, r)

    @torch.no_grad()
    def price_numpy(self, S, tau, K, r) -> np.ndarray:
        t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=torch.float64)
        out = self.price(t(S), t(tau), t(K), t(r))
        return out.detach().numpy()

    def to_dict_meta(self) -> Dict:
        return {
            "hidden_units": [int(m.out_features) for m in self.net.net if isinstance(m, nn.Linear)],
            "n_parameters": self.net.n_parameters(),
            "trainable_sigma": self.trainable_sigma,
            "formulation": "homogeneous_u(x=log(S/K), tau, r), V = S*u",
            "scaler": self.scaler.to_dict(),
        }
