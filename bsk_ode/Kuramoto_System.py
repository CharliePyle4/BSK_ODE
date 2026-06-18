import os
os.environ["KERAS_BACKEND"] = "torch"  # before importing keras_sig / keras

import random

import torch
torch.set_default_dtype(torch.float64)
active_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", active_device, "torch", torch.__version__)


import time
import math
import numpy as np
import torchmin
from torch import nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torch import cumulative_trapezoid  # or torch.cumulative_trapezoid in newer versions


import keras_sig

import math
from typing import Tuple
import torch

from .stochastic.processes.continuous import FractionalBrownianMotion

# Cell 3 - seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





def generate_noise(
    N_paths: int,
    N_points: int,
    t0: float = 0.0,
    T: float = 1.0,
    hurst: float = 0.2,
    method: str = "daviesharte",  # kept for API compatibility, not used here
) -> torch.Tensor:
    """
    Generates N_paths paths of N_points observations of fractional Gaussian noise.

    Returns
    -------
    eta : (N_paths, N_points)
        Noise rate dB/dt at each point.
    """
    length = T - t0
    dt = length / N_points

    # One RNG for reproducible but independent paths
    rng = np.random.default_rng(SEED)

    eta = torch.empty(N_paths, N_points,
                      device=active_device,
                      dtype=torch.get_default_dtype())

    for i in range(N_paths):
        fbm_gen = FractionalBrownianMotion(
            hurst=hurst,
            t=length,
            rng=rng,
        )
        # N_points+1 samples over [t0, T]
        fbm_sample = fbm_gen.sample(n=N_points)
        B = torch.tensor(fbm_sample,
                         device=active_device,
                         dtype=torch.get_default_dtype())
        dB = B[1:] - B[:-1]      # (N_points,)
        eta[i] = dB / dt         # fractional Gaussian noise rate

    return eta


def _coupling_term(theta: torch.Tensor, K: float) -> torch.Tensor:
    """
    Calculate the coupling term
    c_i(theta) = (K/N) sum_j sin(theta_j - theta_i)
    shape: (N,)
    """
    N    = theta.shape[0]
    diff = theta.unsqueeze(0) - theta.unsqueeze(1)
    return (K / N) * torch.sum(torch.sin(diff), dim=1)


def _drift(
    theta: torch.Tensor,
    omega_vec: torch.Tensor,
    K: float,
) -> torch.Tensor:
    return omega_vec + _coupling_term(theta, K)

def generate_kuramoto_data(
    N_paths: int,
    N_points: int,
    t0: float = 0.0,
    T: float = 1.0,
    K: float = 1.0,
    omega: float | torch.Tensor | None = None,
    hurst: float = 0.7,
    sigma: float | torch.Tensor = 1.0,
    theta0: torch.Tensor | None = None,
    method: str = "daviesharte",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    fBm-driven Kuramoto system:
        dtheta_i(t) = [omega_i + (K/N) sum_j sin(theta_j - theta_i)] dt + sigma_i dB_i^H(t)

    Returns
    -------
    dtheta_dt : (N_paths, N_points)
        Discrete effective derivative, left-endpoint aligned:
        dtheta_dt[:,k] = (theta[:,k+1] - theta[:,k]) / dt  — all N real finite diffs
    theta : (N_paths, N_points)
        Phase paths (first N_points values; theta[:,N_points] used internally only).
    eta : (N_paths, N_points)
        Discrete fractional Gaussian noise rate:
        eta[:,k] ≈ sigma_i * ΔB_i^H(t_k) / dt
    lhs : (N_paths, N_points)
        lhs[:,k] = dtheta_dt[:,k] - omega_i - coupling(theta[:,k]),
        which matches eta[:,k] exactly.
    """
    dt = (T - t0) / N_points

    if omega is None:
        omega_vec = torch.zeros(N_paths, device=active_device)
    elif isinstance(omega, (int, float)):
        omega_vec = torch.full((N_paths,), float(omega), device=active_device)
    else:
        omega_vec = omega.to(device=active_device).reshape(N_paths)

    if isinstance(sigma, (int, float)):
        sigma_vec = torch.full((N_paths,), float(sigma), device=active_device)
    else:
        sigma_vec = sigma.to(device=active_device).reshape(N_paths)

    if theta0 is None:
        theta0 = 2.0 * math.pi * torch.rand(N_paths, device=active_device)
    else:
        theta0 = theta0.to(device=active_device).reshape(N_paths)

    # generate noise: eta has shape (N_paths, N_points), all observations real
    eta = sigma_vec.unsqueeze(1) * generate_noise(
        N_paths=N_paths,
        N_points=N_points,
        t0=t0,
        T=T,
        hurst=hurst,
        method=method,
    )  # (N_paths, N_points)
    dB = eta * dt  # recover increments for the Euler step

    # left-endpoint Euler-Maruyama update over N_points steps → N_points+1 values
    # theta[:,N_points] is the extra value needed for the last finite difference
    theta_full          = torch.empty(N_paths, N_points + 1, device=active_device)
    theta_full[:, 0]    = theta0
    for k in range(N_points):                              # ← N_points steps, not N_points-1
        drift_k              = _drift(theta_full[:, k], omega_vec, K)
        theta_full[:, k + 1] = theta_full[:, k] + drift_k * dt + dB[:, k]

    # all N_points finite differences are now real — no special last-point case needed
    dtheta_dt = torch.empty(N_paths, N_points, device=active_device)
    lhs       = torch.empty(N_paths, N_points, device=active_device)
    for k in range(N_points):
        drift_k           = _drift(theta_full[:, k], omega_vec, K)
        dtheta_dt[:, k]   = (theta_full[:, k + 1] - theta_full[:, k]) / dt
        lhs[:, k]         = dtheta_dt[:, k] - drift_k

    # sanity check: lhs should equal eta up to floating point error
    max_diff = (lhs - eta).abs().max().item()
    print(f"max |lhs - eta|: {max_diff:.3e}")

    return dtheta_dt, theta_full[:, :N_points], eta, lhs


import torch
from torchmin import minimize


def forcing_loss(
    true_forcing: torch.Tensor,
    approximated_forcing: torch.Tensor
) -> torch.Tensor:
    residual = true_forcing - approximated_forcing
    return torch.mean(residual**2)


'''def cumtrapz_torch(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    y: (T, M), x: (T,)
    returns integral along dim=0 with zero at x[0], shape (T, M)
    """
    dx = x[1:] - x[:-1]
    area = 0.5 * (y[1:] + y[:-1]) * dx[:, None]
    out = torch.zeros_like(y)
    out[1:] = torch.cumsum(area, dim=0)
    return out'''

# NOTE !!! THIS IS LEFT RIEMANN NOT TRAPEZOIDAL
def cumtrapz_torch(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    y: (T, M), x: (T,)
    returns integral along dim=0 with zero at x[0], shape (T, M)
    Left-rectangle cumulative sum matching Euler-Maruyama generation scheme.
    """
    dt = x[1] - x[0]
    out = torch.zeros_like(y)
    out[1:] = torch.cumsum(y[:-1], dim=0) * dt
    return out

def normalize_kernel_matrix(
    Z: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Column-wise robust normalization using median and IQR.
    Z: (T, D)
    """
    med = Z.median(dim=0, keepdim=True).values
    q25 = Z.quantile(0.25, dim=0, keepdim=True)
    q75 = Z.quantile(0.75, dim=0, keepdim=True)
    iqr = q75 - q25
    return (Z - med) / (iqr + eps)


def compute_signatures(path: torch.Tensor,
                       depth: int) -> torch.Tensor:
    """
    path: (T, d) or (1, T, d) torch tensor.
    Returns: (T, D) prefix-signature features as torch.Tensor.
    """
    if path.dim() == 2:
        path = path.unsqueeze(0)          # (1, T, d)

    # Prepend basepoint, emulating Signatory's basepoint behaviour
    basepoint = path[:, 0:1, :]           # (1, 1, d)
    path_bp = torch.cat([basepoint, path], dim=1)  # (1, T+1, d)

    # Streaming signatures on the basepoint-augmented path:
    # for length = T+1, stream=True → output length-1 = T
    sigs = keras_sig.signature(
        path_bp,
        depth=depth,
        stream=True,
        gpu_optimized=True
    )                                      # (1, T, D)

    return sigs.squeeze(0)                # (T, D)

def build_kernel_from_signatures(
    sigs_flat: torch.Tensor,
    sigma: float = 1.0,
    kernel_type: str = "rbf"
) -> torch.Tensor:
    """
    Build a kernel matrix from signature features.

    sigs_flat: (T, D)
    """
    norms = None
    gram = None
    d2 = None

    if kernel_type == "linear":
        Ker = sigs_flat @ sigs_flat.T

    elif kernel_type == "rbf":
        norms = (sigs_flat ** 2).sum(dim=1, keepdim=True)
        gram = sigs_flat @ sigs_flat.T
        d2 = norms + norms.T - 2 * gram
        Ker = torch.exp(-d2 / (2 * sigma**2))

    else:
        raise ValueError(f"Unknown kernel_type {kernel_type}")

    return Ker


def build_kernel_operators(
    Ksig: torch.Tensor,
    x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    First-order kernel representation:
      dtheta_dt = Kp @ beta
      theta     = theta0 + K @ beta
    """
    Kp = Ksig.clone()
    K = cumtrapz_torch(Kp, x)
    return Kp, K


def kuramoto_coupling(theta: torch.Tensor, K_coupling: float) -> torch.Tensor:
    """
    theta: (T, N)
    returns: (T, N)

    coupling[t, i] = (K/N) sum_j sin(theta[t, j] - theta[t, i])
    """
    T, N = theta.shape
    diff = theta[:, None, :] - theta[:, :, None]
    coupling = (K_coupling / N) * torch.sin(diff).sum(dim=2)
    return coupling

def solve_betas_kuramoto(
    Ksig: torch.Tensor,
    forcing_true: torch.Tensor,          # (T, N), e.g. sigma_i * eta_i^H(t_k)
    x: torch.Tensor,                     # (T,)
    theta0: torch.Tensor,                # (N,)
    omega: float | torch.Tensor,         # scalar or (N,)
    K_coupling: float,
    beta_init: torch.Tensor | None = None,   # (T, N)
    max_iter: int = 500,
    method: str = "l-bfgs",              # kept for API compatibility
):
    """
    Returns:
      beta_opt     : (T, N)
      theta        : (T, N)
      dtheta_dt    : (T, N)
      lhs          : (T, N)
      forcing_pred : (T, N)

    Model:
      dtheta_dt    = Kp @ beta
      theta        = theta0 + K @ beta
      lhs          = dtheta_dt - omega - coupling(theta)
      forcing_pred = lhs
    """
    if method.lower() != "l-bfgs":
        raise ValueError(f"Only 'l-bfgs' is supported, got {method!r}")

    dtype = Ksig.dtype
    device = Ksig.device

    Ksig = Ksig.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)
    forcing_true = forcing_true.to(device=device, dtype=dtype)

    T = Ksig.shape[0]
    N = forcing_true.shape[1]

    theta0 = theta0.to(device=device, dtype=dtype).reshape(N)

    if isinstance(omega, (int, float)):
        omega_vec = torch.full((N,), float(omega), dtype=dtype, device=device)
    else:
        omega_vec = omega.to(device=device, dtype=dtype).reshape(N)

    Kp, K = build_kernel_operators(Ksig, x)  # Kp,K: (T,T)

    # ----- initialization / warm start -----
    if beta_init is None:
        beta0 = torch.zeros((T, N), dtype=dtype, device=device)
    else:
        beta0 = beta_init.clone().detach().to(device=device, dtype=dtype).reshape(T, N)

    # LBFGS works on a 1D parameter vector; we reshape inside the closure
    beta_vec = beta0.reshape(-1).clone().detach().requires_grad_(True)

    def forward(beta_mat: torch.Tensor):
        # beta_mat: (T, N)
        dtheta_dt = Kp @ beta_mat                       # (T, N)
        theta     = theta0.unsqueeze(0) + K @ beta_mat  # (T, N)
        coupling  = kuramoto_coupling(theta, K_coupling)
        lhs       = dtheta_dt - omega_vec.unsqueeze(0) - coupling
        forcing_pred = lhs
        return theta, dtheta_dt, lhs, forcing_pred

    def closure():
        # Called by LBFGS; recomputes loss and gradients
        optimizer.zero_grad()
        beta_mat = beta_vec.view(T, N)
        _, _, _, forcing_pred = forward(beta_mat)
        r = forcing_pred - forcing_true
        loss = 0.5 * (r**2).sum() 
        loss.backward()
        return loss

    if max_iter > 0:
        optimizer = torch.optim.LBFGS(
            [beta_vec],
            max_iter=max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=20,
            line_search_fn="strong_wolfe",
        )
        optimizer.step(closure)
    # else: no optimization; just use beta0 as-is

    # Get final beta matrix
    beta_opt = beta_vec.detach().view(T, N) if max_iter > 0 else beta0

    # Final forward pass (no grad)
    with torch.no_grad():
        theta, dtheta_dt, lhs, forcing_pred = forward(beta_opt)

    return beta_opt, theta, dtheta_dt, lhs, forcing_pred

def init_path_extension_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class PathExtension(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims=(512, 256, 128, 64, 32, 16),
                 activation_cls=nn.Tanh):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation_cls())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def shuffle_loss_function(X_bar: torch.Tensor) -> torch.Tensor:
    dX = X_bar[1:, :] - X_bar[:-1, :]
    X_left = X_bar[:-1, :]
    I = torch.einsum('ka,kb->ab', X_left, dX)
    deltas = X_bar[-1, :] - X_bar[0, :]
    R = deltas[:, None] * deltas[None, :] - (I + I.T)
    return torch.triu(R.pow(2), diagonal=0).sum()

def solve_signature_kernel_branched(
    t_grid: torch.Tensor,
    forcing_true: torch.Tensor,          # (N, T)
    theta0: torch.Tensor,                # (N,)
    depth: int,
    rbf_sigma: float,
    omega: float | torch.Tensor,
    K_coupling: float,
    kernel_type: str = "rbf",
    normalize: bool = True,
    max_beta_iter: int = 500,
    beta_method: str = "l-bfgs",
    hidden_dims=(512, 256, 128, 64, 32, 16),
    activation_cls=nn.Tanh,
    extensions: int = 2,
    adam_iters: int = 1000,
    adam_lr: float = 1e-3,
    adam_lambda_model: float = 10.0,
    adam_lambda_shuffle: float = 1e-3,
    adam_use_scheduler: bool = True,
    adam_sched_factor: float = 0.5,
    adam_sched_patience: int = 1000,
    num_snapshots: int = 10,
    grad_clip: float | None = 1.0,
    verbose: bool = True,
    beta_solve_every: int = 1,
    beta_min_iterations: int = 20,
    beta_ramp_portion: float = 0.3,
):
    N, T = forcing_true.shape
    assert theta0.shape[0] == N, "theta0 must have shape (N,)"
    assert t_grid.numel() == T, "t_grid must have length T"
    assert extensions >= 0, "extensions must be nonnegative"

    device = t_grid.device
    dtype = t_grid.dtype

    t_grid = t_grid.to(device=device, dtype=dtype)
    forcing_true = forcing_true.to(device=device, dtype=dtype)
    theta0 = theta0.to(device=device, dtype=dtype).reshape(N)

    if isinstance(omega, (int, float)):
        omega_vec = torch.full((N,), float(omega), dtype=dtype, device=device)
    else:
        omega_vec = omega.to(device=device, dtype=dtype).reshape(N)

    # Base path: time + forcing
    X = torch.cat([t_grid.unsqueeze(1), forcing_true.T], dim=1)   # (T, 1+N)


    # ------------------------ branched (NN) case ------------------------ #
    path_ext = PathExtension(
        input_dim=X.shape[1],
        output_dim=extensions,
        hidden_dims=hidden_dims,
        activation_cls=activation_cls,
    ).to(device=device, dtype=dtype)

    path_ext.apply(init_path_extension_weights)

    path_ext = torch.compile(path_ext)

    optimizer = torch.optim.Adam(
        list(path_ext.parameters()),
        lr=adam_lr,
    )

    scheduler = None
    if adam_use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=adam_sched_factor,
            patience=adam_sched_patience,
        )

    if num_snapshots > 0:
        snapshot_iters = sorted(set(
            int(i) for i in torch.linspace(1, adam_iters, num_snapshots)
        ))
        if 1 in snapshot_iters:
            snapshot_iters.remove(1)
    else:
        snapshot_iters = []

    snapshots = []
    training_history = []
    beta_prev = None

    for it in range(1, adam_iters + 1):
        optimizer.zero_grad()

        out_ext = path_ext(X)                         # (T, extensions)
        X_bar = torch.cat([X, out_ext], dim=1)        # (T, 1+N+extensions)

        X_sig = compute_signatures(X_bar, depth)
        if normalize:
            X_sig = normalize_kernel_matrix(X_sig)

        Ksig = build_kernel_from_signatures(
            X_sig,
            sigma=rbf_sigma,
            kernel_type=kernel_type,
        )


        # Decide whether to re-solve beta this iteration
        if (it == 1) or (it % beta_solve_every == 0):
            # Ramp LBFGS iterations from beta_min_iterations -> beta_iterations
            progress = min(1.0, it / (adam_iters * beta_ramp_portion))
            lbfgs_iters = int(
                beta_min_iterations
                + (max_beta_iter - beta_min_iterations) * progress
            )

            # Beta solve on detached K_stack — avoids double-backward
            Ksig_detached = Ksig.detach()
            beta_w, _, _, _, _= solve_betas_kuramoto(
                Ksig=Ksig_detached,
                forcing_true=forcing_true.T,
                x=t_grid,
                theta0=theta0,
                omega=omega_vec,
                K_coupling=K_coupling,
                beta_init=beta_prev,
                max_iter=lbfgs_iters,
                method=beta_method,
            )
            beta_prev = beta_w.detach().clone()

        else:
            # Reuse previous beta; no LBFGS this step
            beta_w = beta_prev 


        Kp, K = build_kernel_operators(Ksig, t_grid)  # Kp,K: (T,T)
        dtheta_dt = Kp @ beta_w                       # (T, N)
        theta     = theta0.unsqueeze(0) + K @ beta_w   # (T, N)
        coupling  = kuramoto_coupling(theta, K_coupling)
        lhs       = dtheta_dt - omega_vec.unsqueeze(0) - coupling
        forcing_pred = lhs
        

        # transpose so its fine
        theta_fit = theta.T
        dtheta_fit = dtheta_dt.T
        lhs_fit = lhs.T
        forcing_fit = forcing_pred.T

        model_loss   = forcing_loss(forcing_true, forcing_fit)
        shuffle_loss = shuffle_loss_function(out_ext)
        total_loss   = adam_lambda_model * model_loss + adam_lambda_shuffle * shuffle_loss

        total_loss.backward()


        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(path_ext.parameters(), max_norm=grad_clip)

        optimizer.step()

        if scheduler is not None:
            scheduler.step(total_loss.detach())

        current_lr = optimizer.param_groups[0]["lr"]

        training_history.append({
            "iter": it,
            "lr": float(current_lr),
            "total_weighted_loss": float(total_loss.detach().item()),
            "model_loss": float(model_loss.detach().item()),
            "shuffle_loss": float(shuffle_loss.detach().item()),
        })

        if verbose and (it % 50 == 0 or it == 1):
            print(
                f"[Adam {it:04d}] "
                f"loss={total_loss.item():.3e}, "
                f"model={model_loss.item():.3e}, "
                f"shuffle={shuffle_loss.item():.3e}"
            )

        if it in snapshot_iters:
            snapshots.append({
                "phase": "Adam",
                "iter": it,
                "X_bar": X_bar.detach().clone(),
                "theta_fit": theta_fit.detach().clone(),
                "forcing_fit": forcing_fit.detach().clone(),
            })

    # ------------------------ final evaluation ------------------------ #
    with torch.no_grad():
        out_ext_final = path_ext(X)
        X_bar_final = torch.cat([X, out_ext_final], dim=1)

        X_sig_final = compute_signatures(X_bar_final, depth)
        if normalize:
            X_sig_final = normalize_kernel_matrix(X_sig_final)

        Ksig_final = build_kernel_from_signatures(
            X_sig_final,
            sigma=rbf_sigma,
            kernel_type=kernel_type,
        )

    beta, theta_fit_T, dtheta_fit_T, lhs_fit_T, forcing_fit_T = solve_betas_kuramoto(
        Ksig=Ksig_final,
        forcing_true=forcing_true.T,
        x=t_grid,
        theta0=theta0,
        omega=omega_vec,
        K_coupling=K_coupling,
        beta_init=beta_prev,
        max_iter=max_beta_iter,
        method=beta_method,
    )

    theta_fit   = theta_fit_T.T
    dtheta_fit  = dtheta_fit_T.T
    lhs_fit     = lhs_fit_T.T
    forcing_fit = forcing_fit_T.T

    final_loss = forcing_loss(forcing_true, forcing_fit)
    print(f"branched model forcing match loss: {final_loss.item():.3e}")

    snapshots.append({
        "phase": "final",
        "iter": adam_iters,
        "X_bar": X_bar_final.detach().clone(),
        "theta_fit": theta_fit.detach().clone(),
        "forcing_fit": forcing_fit.detach().clone(),
    })

    return (
        theta_fit,
        dtheta_fit,
        lhs_fit,
        forcing_fit,
        beta,
        X_bar_final.detach(),
        path_ext,
        snapshots,
        training_history
    )


    

    

def solve_signature_kernel_non_branched(
    t_grid: torch.Tensor,
    forcing_true: torch.Tensor,          # (N, T), e.g. sigma_i * eta_i^H(t_k)
    theta0: torch.Tensor,                # (N,)
    depth: int,
    rbf_sigma: float,
    omega: float | torch.Tensor,
    K_coupling: float,
    kernel_type: str = "rbf",
    normalize: bool = True,
    beta_reg: float = 1e-10,
    max_beta_iter: int = 500,
    beta_method: str = "l-bfgs",
):
    N, T = forcing_true.shape
    assert forcing_true.shape == (N, T), "forcing_true must have shape (N, T)"
    assert theta0.shape[0] == N, "theta0 must have shape (N,)"
    assert t_grid.numel() == T, "t_grid must have length T"

    theta0 = theta0.to(device=t_grid.device, dtype=t_grid.dtype).reshape(N)

    # Observed path: time + forcing
    X = torch.cat(
        [t_grid.unsqueeze(1), forcing_true.T],
        dim=1
    )  # (T, 1+N)

    X_sig = compute_signatures(X, depth)

    if normalize:
        X_sig = normalize_kernel_matrix(X_sig)

    Ksig = build_kernel_from_signatures(
        X_sig,
        sigma=rbf_sigma,
        kernel_type=kernel_type
    )

    beta, theta_fit_T, dtheta_fit_T, lhs_fit_T, forcing_fit_T = solve_betas_kuramoto(
        Ksig=Ksig,
        forcing_true=forcing_true.T,   # (T, N)
        x=t_grid,
        theta0=theta0,
        omega=omega,
        K_coupling=K_coupling,
        max_iter=max_beta_iter,
        method=beta_method,
    )

    theta_fit = theta_fit_T.T
    dtheta_fit = dtheta_fit_T.T
    lhs_fit = lhs_fit_T.T
    forcing_fit = forcing_fit_T.T

    final_loss = forcing_loss(forcing_true, forcing_fit)
    print(f"non-branched model forcing match loss: {final_loss.item():.3e}")

    return theta_fit, dtheta_fit, lhs_fit, forcing_fit, beta






import torch
import matplotlib.pyplot as plt

def plot_training_history(training_history, use_log_scale=True):
    """
    Plot:
      - learning rate in its own figure
      - total loss, model loss, and shuffle loss in a 1x3 grid

    training_history: list of dicts with keys
        'iter', 'lr', 'total_weighted_loss', 'model_loss', 'shuffle_loss'
    """
    if len(training_history) == 0:
        print("training_history is empty.")
        return

    iters = [row["iter"] for row in training_history]
    lrs = [row["lr"] for row in training_history]
    total_losses = [row["total_weighted_loss"] for row in training_history]
    model_losses = [row["model_loss"] for row in training_history]
    shuffle_losses = [row["shuffle_loss"] for row in training_history]

    plt.figure(figsize=(7, 4))
    plt.plot(iters, lrs, color="purple", linewidth=2)
    plt.title("Learning Rate")
    plt.xlabel("Iteration")
    plt.ylabel("LR")
    plt.grid(True, alpha=0.3)
    if use_log_scale:
        plt.yscale("log")
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

    axes[0].plot(iters, total_losses, color="black", linewidth=2)
    axes[0].set_title("Total Weighted Loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    if use_log_scale:
        axes[0].set_yscale("log")

    axes[1].plot(iters, model_losses, color="blue", linewidth=2)
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, alpha=0.3)
    if use_log_scale:
        axes[1].set_yscale("log")

    axes[2].plot(iters, shuffle_losses, color="red", linewidth=2)
    axes[2].set_title("Shuffle Loss")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Loss")
    axes[2].grid(True, alpha=0.3)
    if use_log_scale:
        axes[2].set_yscale("log")

    fig.tight_layout()
    plt.show()


def plot_kuramoto_thetas_and_signals(
    Thetas: torch.Tensor,    # (N, T)
    Values: torch.Tensor,    # (N, T) e.g. DTheta, Zeta, or LHS
    t_grid: torch.Tensor,    # (T,)
    value_label: str = r's_i(t)',
    value_title: str = 'Quantity',
    suptitle: str = r'Kuramoto Phase Paths $\theta_i(t)$ and Companion Quantity',
):
    """
    Plot, for each oscillator i:
      left  : theta_i(t)
      right : a companion quantity Values_i(t), e.g. dtheta/dt, zeta, or lhs
    """
    assert Thetas.shape == Values.shape, "Thetas and Values must have same shape"
    N, T = Thetas.shape
    assert t_grid.numel() == T, "t_grid length must match tensor time dimension"

    t = t_grid.detach().cpu().numpy()
    th = Thetas.detach().cpu().numpy()
    val = Values.detach().cpu().numpy()

    fig, axes = plt.subplots(
        N, 2,
        figsize=(12, 2.5 * N),
        sharex='col'
    )

    if N == 1:
        axes = axes.reshape(1, 2)

    for i in range(N):
        ax_theta = axes[i, 0]
        ax_theta.plot(t, th[i], lw=1.2, color=f'C{i}')
        ax_theta.set_ylabel(rf'$\theta_{{{i}}}(t)$')
        ax_theta.set_title(f'Oscillator {i} — Phase')
        ax_theta.grid(True, alpha=0.3)

        ax_val = axes[i, 1]
        ax_val.plot(t, val[i], lw=1.2, color=f'C{i}')
        ax_val.set_ylabel(rf'${value_label}$')
        ax_val.set_title(f'Oscillator {i} — {value_title}')
        ax_val.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_kuramoto_true_vs_fitted(
    Thetas_true: torch.Tensor,   # (N, T)
    Values_true: torch.Tensor,   # (N, T)
    Thetas_fit: torch.Tensor,    # (N, T)
    Values_fit: torch.Tensor,    # (N, T)
    t_grid: torch.Tensor,        # (T,)
    value_label: str = r's_i(t)',
    value_title: str = 'Quantity',
    suptitle: str = r'Kuramoto: True vs Fitted Phases and Companion Quantity',
):
    """
    Plot, for each oscillator i:
      left  : theta_true_i(t) vs theta_fit_i(t)
      right : value_true_i(t) vs value_fit_i(t)

    Values can be DTheta, Zeta, LHS, etc.
    """
    assert Thetas_true.shape == Values_true.shape, "True Thetas and Values must match in shape"
    assert Thetas_fit.shape == Values_fit.shape, "Fitted Thetas and Values must match in shape"
    assert Thetas_true.shape == Thetas_fit.shape, "True and fitted tensors must have same shape"

    N, T = Thetas_true.shape
    assert t_grid.numel() == T, "t_grid length must match tensor time dimension"

    t = t_grid.detach().cpu().numpy()
    th_t = Thetas_true.detach().cpu().numpy()
    th_f = Thetas_fit.detach().cpu().numpy()
    val_t = Values_true.detach().cpu().numpy()
    val_f = Values_fit.detach().cpu().numpy()

    fig, axes = plt.subplots(
        N, 2,
        figsize=(12, 2.5 * N),
        sharex='col'
    )

    if N == 1:
        axes = axes.reshape(1, 2)

    for i in range(N):
        ax_theta = axes[i, 0]
        ax_theta.plot(t, th_t[i], lw=1.5, color=f'C{i}', label='true')
        ax_theta.plot(t, th_f[i], lw=1.2, color='k', ls='--', label='fitted')
        ax_theta.set_ylabel(rf'$\theta_{{{i}}}(t)$')
        ax_theta.set_title(f'Oscillator {i} — Phase (true vs fitted)')
        ax_theta.grid(True, alpha=0.3)
        ax_theta.legend(fontsize=8)

        ax_val = axes[i, 1]
        ax_val.plot(t, val_t[i], lw=1.5, color=f'C{i}', label='true')
        ax_val.plot(t, val_f[i], lw=1.2, color='k', ls='--', label='fitted')
        ax_val.set_ylabel(rf'${value_label}$')
        ax_val.set_title(f'Oscillator {i} — {value_title} (true vs fitted)')
        ax_val.grid(True, alpha=0.3)
        ax_val.legend(fontsize=8)

    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_kuramoto_signals(
    Thetas: torch.Tensor,       # (N, T)
    DTheta: torch.Tensor,       # (N, T)
    Eta: torch.Tensor,          # (N, T)
    Lhs: torch.Tensor,          # (N, T)
    t_grid: torch.Tensor,       # (T,)
    suptitle: str = r'Kuramoto: $\theta_i(t)$, $\dot{\theta}_i(t)$, $\eta_i(t)$, and LHS',
):
    """
    Plot, for each oscillator i, four panels:
      col 0 : theta_i(t)
      col 1 : dtheta_dt_i(t)  vs  eta_i(t)  vs  lhs_i(t)  — all overlaid
    """
    assert Thetas.shape == DTheta.shape == Eta.shape == Lhs.shape, \
        "Thetas, DTheta, Eta, and Lhs must all have the same shape"
    N, T = Thetas.shape
    assert t_grid.numel() == T, "t_grid length must match tensor time dimension"

    t      = t_grid.detach().cpu().numpy()
    th     = Thetas.detach().cpu().numpy()
    dth    = DTheta.detach().cpu().numpy()
    eta    = Eta.detach().cpu().numpy()
    lhs    = Lhs.detach().cpu().numpy()

    fig, axes = plt.subplots(
        N, 2,
        figsize=(12, 2.5 * N),
        sharex='col',
    )

    if N == 1:
        axes = axes.reshape(1, 2)

    for i in range(N):
        # left panel: phase
        ax_theta = axes[i, 0]
        ax_theta.plot(t, th[i], lw=1.2, color=f'C{i}')
        ax_theta.set_ylabel(rf'$\theta_{{{i}}}(t)$')
        ax_theta.set_title(f'Oscillator {i} — Phase')
        ax_theta.grid(True, alpha=0.3)

        # right panel: dtheta, eta, lhs overlaid
        ax_sig = axes[i, 1]
        ax_sig.plot(t, dth[i], lw=1.2, color=f'C{i}',  label=r'$\dot{\theta}_i$')
        ax_sig.plot(t, eta[i], lw=1.2, color='k',       label=r'$\eta_i$',  ls='--')
        ax_sig.plot(t, lhs[i], lw=1.0, color='tomato',  label=r'$\mathrm{lhs}_i$', ls=':')
        ax_sig.set_ylabel(rf'signals (oscillator {i})')
        ax_sig.set_title(f'Oscillator {i} — $\dot{{\\theta}}$, $\\eta$, LHS')
        ax_sig.grid(True, alpha=0.3)
        ax_sig.legend(fontsize=8)

    axes[-1, 0].set_xlabel('Time')
    axes[-1, 1].set_xlabel('Time')

    fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()