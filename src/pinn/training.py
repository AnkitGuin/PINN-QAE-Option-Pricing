"""PINN training: staged optimisation with structured logging.

Protocol (chosen on the constant-volatility development scenario, Gates 4/5,
and frozen before test conditions were evaluated):

  Stage A ("physics-first", sigma FROZEN at its initial value):
      Adam -> optional L-BFGS on the network only, with data + PDE + boundary
      + terminal losses.  Early stopping on a held-out fraction of the
      CALIBRATION WINDOW (never test data).
  Stage B (sigma released):
      short joint Adam -> optional joint L-BFGS with a smaller learning rate.

Rationale (documented in docs/methodology.md): joint gradient descent from a
released sigma falls into compensated (sigma, network) states; freezing sigma
while the network settles onto the PDE-consistent solution family first, then
releasing it, avoids the worst of those local minima.

Logs per epoch: total loss, data loss, PDE loss, boundary loss, terminal loss,
sigma estimate, learning rate, validation metric, stage.  At the end: best
epoch, final sigma, final losses, training time, convergence status, seed.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ..data.preprocessing import WindowScaler, fit_window_scaler
from ..data.splits import chronological_train_val_split
from ..utils.reproducibility import derive_seed
from .collocation import CollocationConfig, sample_window_collocation
from .losses import LossComponents, compute_losses
from .network import PINNModel

__all__ = ["TrainConfig", "TrainResult", "train_pinn_window"]


@dataclass
class TrainConfig:
    # architecture
    hidden_units: tuple = (32, 32, 32)
    activation: str = "tanh"
    epsilon_sigma: float = 1e-6
    init_sigma: float = 0.20
    warm_start: bool = True
    weights: Dict[str, float] = field(default_factory=lambda: {
        "lambda_data": 1.0, "lambda_pde": 100.0, "lambda_boundary": 1.0, "lambda_terminal": 5.0,
    })
    collocation: CollocationConfig = field(default_factory=CollocationConfig)
    torch_threads: int = 1
    dtype: torch.dtype = torch.float64
    # stage A: network only, sigma frozen
    stage_a_adam_epochs: int = 900
    stage_a_adam_lr: float = 5e-3
    stage_a_lr_decay: float = 0.9995
    stage_a_lbfgs_iterations: int = 350
    stage_a_patience: int = 300
    early_stopping_enabled: bool = True
    validation_fraction: float = 0.15
    # stage B: joint refinement, sigma released
    stage_b_adam_epochs: int = 400
    stage_b_adam_lr_net: float = 1e-3
    stage_b_adam_lr_sigma: float = 3e-3
    stage_b_lbfgs_iterations: int = 350
    stage_b_lbfgs_lr: float = 0.3
    # ablation switches (data-only network: all physics off)
    use_data: bool = True
    use_pde: bool = True
    use_boundary: bool = True
    use_terminal: bool = True

    @classmethod
    def from_pinn_config(cls, cfg: Dict[str, Any]) -> "TrainConfig":
        t = cfg.get("training", {})
        w = cfg.get("loss_weights", {})
        sp = cfg.get("sigma_parameterization", {})
        arch = cfg.get("architecture", {})
        es = t.get("early_stopping", {})
        sa = t.get("stage_a", {}) or {}
        sb = t.get("stage_b", {}) or {}
        rf = t.get("refinement", {}) or {}
        return cls(
            hidden_units=tuple(int(h) for h in arch.get("hidden_units", [32, 32, 32])),
            activation=str(arch.get("activation", "tanh")),
            epsilon_sigma=float(sp.get("epsilon_sigma", 1e-6)),
            init_sigma=float(sp.get("init_sigma", 0.20)),
            warm_start=bool(sp.get("warm_start", True)),
            weights={k: float(v) for k, v in w.items()},
            collocation=CollocationConfig.from_dict(t.get("collocation", {})),
            torch_threads=int(t.get("torch_threads", 1)),
            stage_a_adam_epochs=int(sa.get("epochs", 900)),
            stage_a_adam_lr=float(sa.get("lr", 5e-3)),
            stage_a_lr_decay=float(sa.get("lr_decay", 0.9995)),
            stage_a_lbfgs_iterations=int(rf.get("iterations", sa.get("lbfgs_iterations", 350))),
            stage_a_patience=int(es.get("patience", 300)),
            early_stopping_enabled=bool(es.get("enabled", True)),
            validation_fraction=float(t.get("validation_fraction", 0.15)),
            stage_b_adam_epochs=int(sb.get("epochs", 400)),
            stage_b_adam_lr_net=float(sb.get("lr_net", 1e-3)),
            stage_b_adam_lr_sigma=float(sb.get("lr_sigma", 3e-3)),
            stage_b_lbfgs_iterations=int(sb.get("lbfgs_iterations", 350)),
            stage_b_lbfgs_lr=float(sb.get("lbfgs_lr", 0.3)),
        )


@dataclass
class TrainResult:
    status: str
    best_epoch: int
    final_sigma: Optional[float]
    final_losses: Dict[str, float]
    val_metric: Optional[float]
    runtime_seconds: float
    history: List[Dict[str, Any]] = field(default_factory=list)
    scaler: Optional[Dict] = None
    n_parameters: Optional[int] = None
    n_fit_observations: int = 0
    n_val_observations: int = 0
    stop_reason: str = ""
    stage_a_sigma: Optional[float] = None

    def sigma_history(self) -> np.ndarray:
        return np.array([h["sigma"] for h in self.history])

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.__dict__)
        out.pop("history", None)
        return out


def train_pinn_window(
    window: pd.DataFrame,
    config: TrainConfig,
    seed: int,
    trainable_sigma: bool = True,
    sigma_init_override: Optional[float] = None,
    fixed_sigma: Optional[float] = None,
    return_model: bool = False,
) -> Tuple[TrainResult, Optional[PINNModel]]:
    """Fit the (inverse or forward) PINN on one calibration window.

    ``fixed_sigma`` pins the shared volatility (forward / fixed-parameter
    ablation: stage B is skipped entirely).  ``sigma_init_override`` warm-starts
    the trainable parameter (rolling warm start uses past estimates only).
    """
    torch.set_num_threads(max(1, config.torch_threads))
    torch.manual_seed(seed)

    scaler = fit_window_scaler(window)
    init_sigma = fixed_sigma if fixed_sigma is not None else (
        sigma_init_override if sigma_init_override is not None else config.init_sigma
    )
    model = PINNModel(
        scaler,
        hidden_units=config.hidden_units,
        activation=config.activation,
        sigma_init=init_sigma,
        epsilon_sigma=config.epsilon_sigma,
        trainable_sigma=trainable_sigma and fixed_sigma is None,
        dtype=config.dtype,
    )

    # --- window-internal fit/val split for early stopping (no test data) ---
    val_seed = derive_seed(seed, "val_split")
    fit_idx, val_idx = chronological_train_val_split(window, config.validation_fraction, val_seed)
    if len(fit_idx) < 5 or len(val_idx) < 2:
        fit_idx, val_idx = np.arange(len(window)), np.arange(0)

    wf = window.iloc[fit_idx]
    wv = window.iloc[val_idx]

    def _t(a):
        return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=config.dtype)

    data_fit = (
        _t(wf["spot"]), _t(wf["tau"]), _t(wf["strike"]), _t(wf["rate"]), _t(wf["market_price"])
    ) if len(wf) else None
    data_val = (
        _t(wv["spot"]), _t(wv["tau"]), _t(wv["strike"]), _t(wv["rate"]), _t(wv["market_price"])
    ) if len(wv) else None

    coll = sample_window_collocation(window, config.collocation, seed=derive_seed(seed, "collocation"))
    interior = (_t(coll.interior_S), _t(coll.interior_tau), _t(coll.interior_K), _t(coll.interior_r))
    boundary = (
        _t(coll.boundary_S), _t(coll.boundary_tau), _t(coll.boundary_K),
        _t(coll.boundary_r), _t(coll.boundary_target),
    )
    terminal = (_t(coll.terminal_S), _t(coll.terminal_K), _t(coll.terminal_r), _t(coll.terminal_payoff))

    net_params = list(model.net.parameters())
    sigma_params = list(model.sigma_module.parameters())

    def _full_losses(use_flags=None):
        flags = use_flags or dict(
            use_data=config.use_data, use_pde=config.use_pde,
            use_boundary=config.use_boundary, use_terminal=config.use_terminal,
        )
        return compute_losses(
            model, data_fit, interior, boundary, terminal,
            weights=config.weights, **flags,
        )

    def _val_metric():
        if data_val is None:
            return None
        with torch.no_grad():
            S_v, tau_v, K_v, r_v, y_v = data_val
            u_hat = model.u(S_v, tau_v, K_v, r_v)
            return float(torch.mean((u_hat - y_v / S_v) ** 2))

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None
    status = "completed"
    stop_reason = ""
    start = time.perf_counter()
    epoch_counter = 0

    def _record(comp: LossComponents, stage: str, lr_val: float):
        nonlocal best_val, best_epoch, best_state
        val_metric = _val_metric()
        history.append(
            {
                "epoch": epoch_counter,
                "stage": stage,
                "total": comp.total,
                "data": comp.data,
                "pde": comp.pde,
                "boundary": comp.boundary,
                "terminal": comp.terminal,
                "sigma": comp.sigma,
                "lr": lr_val,
                "val": val_metric if val_metric is not None else float("nan"),
            }
        )
        score = val_metric if val_metric is not None else comp.total
        if np.isfinite(score) and score < best_val:
            best_val = score
            best_epoch = epoch_counter
            best_state = copy.deepcopy(model.state_dict())

    # ------------------------- Stage A: sigma frozen -------------------------
    if not (trainable_sigma and fixed_sigma is None):
        for p in sigma_params:
            p.requires_grad_(False)
    elif config.stage_b_adam_epochs + config.stage_b_lbfgs_iterations > 0:
        for p in sigma_params:
            p.requires_grad_(False)  # frozen during stage A

    opt_a = torch.optim.Adam(net_params, lr=config.stage_a_adam_lr)
    sched_a = torch.optim.lr_scheduler.ExponentialLR(opt_a, gamma=config.stage_a_lr_decay) \
        if config.stage_a_lr_decay != 1.0 else None
    patience_left = config.stage_a_patience

    for epoch in range(config.stage_a_adam_epochs):
        epoch_counter = epoch
        opt_a.zero_grad()
        total, comp = _full_losses()
        if not torch.isfinite(total):
            status = "failed"
            stop_reason = f"non-finite loss in stage A at epoch {epoch}"
            break
        total.backward()
        opt_a.step()
        if sched_a is not None:
            sched_a.step()
        prior_best = best_val
        _record(comp, "adam_a", opt_a.param_groups[0]["lr"])

        score = history[-1]["val"] if data_val is not None else comp.total
        improved = np.isfinite(score) and best_val < prior_best
        if improved:
            patience_left = config.stage_a_patience
        else:
            patience_left -= 1
            if config.early_stopping_enabled and patience_left <= 0:
                status = "converged_stage_a"
                stop_reason = f"stage A early stop at epoch {epoch} (best {best_epoch})"
                break

    stage_a_sigma = None
    if status != "failed" and config.stage_a_lbfgs_iterations > 0:
        if best_state is not None:
            model.load_state_dict(best_state)
        lbfgs_a = torch.optim.LBFGS(
            net_params, lr=0.5, max_iter=config.stage_a_lbfgs_iterations,
            history_size=50, tolerance_change=1e-16, tolerance_grad=1e-16,
            line_search_fn="strong_wolfe",
        )

        def closure_a():
            lbfgs_a.zero_grad()
            total, _ = _full_losses()
            total.backward()
            return total

        try:
            lbfgs_a.step(closure_a)
            epoch_counter = len(history)
            _, comp = _full_losses()
            _record(comp, "lbfgs_a", 0.5)
        except Exception as exc:  # pragma: no cover - defensive
            status = "failed"
            stop_reason = f"stage A L-BFGS exception: {exc}"

    stage_a_sigma = float(model.sigma())

    # ---------------------- Stage B: joint refinement ------------------------
    if status != "failed" and trainable_sigma and fixed_sigma is None \
            and (config.stage_b_adam_epochs > 0 or config.stage_b_lbfgs_iterations > 0):
        for p in sigma_params:
            p.requires_grad_(True)
        all_params = net_params + sigma_params
        opt_b = torch.optim.Adam([
            {"params": net_params, "lr": config.stage_b_adam_lr_net},
            {"params": sigma_params, "lr": config.stage_b_adam_lr_sigma},
        ])
        for epoch in range(config.stage_b_adam_epochs):
            epoch_counter = len(history)
            opt_b.zero_grad()
            total, comp = _full_losses()
            if not torch.isfinite(total):
                status = "failed"
                stop_reason = f"non-finite loss in stage B at epoch {epoch}"
                break
            total.backward()
            opt_b.step()
            _record(comp, "adam_b", opt_b.param_groups[0]["lr"])

        if status != "failed" and config.stage_b_lbfgs_iterations > 0:
            lbfgs_b = torch.optim.LBFGS(
                all_params, lr=config.stage_b_lbfgs_lr, max_iter=config.stage_b_lbfgs_iterations,
                history_size=50, tolerance_change=1e-16, tolerance_grad=1e-16,
                line_search_fn="strong_wolfe",
            )

            def closure_b():
                lbfgs_b.zero_grad()
                total, _ = _full_losses()
                total.backward()
                return total

            try:
                lbfgs_b.step(closure_b)
                epoch_counter = len(history)
                _, comp = _full_losses()
                _record(comp, "lbfgs_b", config.stage_b_lbfgs_lr)
            except Exception as exc:  # pragma: no cover - defensive
                status = "failed"
                stop_reason = f"stage B L-BFGS exception: {exc}"

    runtime = time.perf_counter() - start

    # final reported state: end of training (stage B end, or stage A end when
    # sigma is fixed).  Val-best state is recorded in the history for
    # diagnostics; the calibration output is the final joint state, matching
    # the "Adam warmup -> L-BFGS to convergence, take final" protocol.
    final_sigma = None
    with torch.no_grad():
        final_sigma = float(model.sigma()) if (fixed_sigma is not None or trainable_sigma) else None

    # final losses on the FULL window (fit + val) for reporting
    data_full = (
        _t(window["spot"]), _t(window["tau"]), _t(window["strike"]), _t(window["rate"]),
        _t(window["market_price"]),
    )
    _, comp_final = compute_losses(
        model, data_full, interior, boundary, terminal,
        weights={"lambda_data": 0.0, "lambda_pde": 0.0, "lambda_boundary": 0.0, "lambda_terminal": 0.0},
        use_data=config.use_data, use_pde=config.use_pde,
        use_boundary=config.use_boundary, use_terminal=config.use_terminal,
    )
    val_final = _val_metric()

    result = TrainResult(
        status=status if status != "failed" else "failed",
        best_epoch=best_epoch,
        final_sigma=final_sigma,
        final_losses={
            "data": comp_final.data,
            "pde": comp_final.pde,
            "boundary": comp_final.boundary,
            "terminal": comp_final.terminal,
        },
        val_metric=val_final,
        runtime_seconds=runtime,
        history=history,
        scaler=scaler.to_dict(),
        n_parameters=model.net.n_parameters(),
        n_fit_observations=int(len(wf)),
        n_val_observations=int(len(wv)),
        stop_reason=stop_reason or "completed all stages",
        stage_a_sigma=stage_a_sigma,
    )
    return (result, model if return_model else None)
