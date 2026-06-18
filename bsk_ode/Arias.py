import random
import os
os.environ["KERAS_BACKEND"] = "torch"  # before importing keras_sig / keras


import torch
torch.set_default_dtype(torch.float64)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, "torch", torch.__version__)

import time
import math
import numpy as np
import torchmin
from torch import nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torchmin import least_squares
from torchmin import minimize
from torch import cumulative_trapezoid
import torch.nn.functional as F


import keras_sig
from keras_sig import SigLayer

from .stochastic.processes.continuous import FractionalBrownianMotion



# Cell 3 - seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def f_forcing_fbm(x: torch.Tensor, hurst: float = 0.2) -> torch.Tensor:
    """Fractional Brownian motion on [a,b] using the stochastic library."""
    N = x.numel()
    a_ = float(x[0])
    b_ = float(x[-1])
    length = b_ - a_

    rng = np.random.default_rng(SEED)
    fbm_gen = FractionalBrownianMotion(hurst=hurst, t=length, rng=rng)
    fbm_sample = fbm_gen.sample(n=N - 1)

    return torch.tensor(fbm_sample, dtype=x.dtype, device=x.device)

def solve_linear_ivp(x_grid: torch.Tensor,
                    forcing_torch: torch.Tensor,
                    a: float, b: float,
                    ya: float, ypa: float,
                    a_fun, b_fun, c_fun):
    """
    Solve a(x) u'' + b(x) u' + c(x) u = f(x)
    rewritten as u'' = (f - b(x) u' - c(x) u) / a(x)
    """
    # Work with numpy time grid
    t_eval = x_grid.cpu().numpy()
    f_np = forcing_torch.cpu().numpy()
    dx = t_eval[1] - t_eval[0]

    # Precompute coefficient values on the grid using the torch functions
    # a_fun, b_fun, c_fun are assumed to accept a torch tensor and return a tensor
    a_vec = a_fun(x_grid).cpu().numpy()
    b_vec = b_fun(x_grid).cpu().numpy()
    c_vec = c_fun(x_grid).cpu().numpy()

    def interp(arr, t):
        # nearest‑neighbor lookup on the grid
        idx = int(round((t - t_eval[0]) / dx))
        idx = max(0, min(len(arr) - 1, idx))
        return arr[idx]

    def forcing(t):
        return interp(f_np, t)

    def a_val(t):
        return interp(a_vec, t)

    def b_val(t):
        return interp(b_vec, t)

    def c_val(t):
        return interp(c_vec, t)

    def fun(t, y):
        u, up = y
        f_v = forcing(t)
        av = a_val(t)
        bv = b_val(t)
        cv = c_val(t)
        du_dt = up
        dup_dt = (f_v - bv * up - cv * u) / av
        return [du_dt, dup_dt]

    y0 = [ya, ypa]
    sol = solve_ivp(fun, (a, b), y0, t_eval=t_eval,
                    method="BDF", rtol=1e-10, atol=1e-15, max_step=0.05)
    u_ivp = sol.y[0]
    return (torch.tensor(t_eval, dtype=torch.float64),
            torch.tensor(u_ivp, dtype=torch.float64))


# === Integration and kernel helpers ===
#integrate kernel matrix
def cumtrapz_torch(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dx   = x[1:] - x[:-1]
    area = 0.5 * (y[1:] + y[:-1]) * dx
    out  = torch.zeros_like(y)
    out[1:] = torch.cumsum(area, dim=0)
    return out

def double_integrate(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return cumtrapz_torch(cumtrapz_torch(y, x), x)


def forcing_loss(true_forcing, approximated_forcing):
    residual = true_forcing - approximated_forcing
    loss = torch.mean(residual**2)
    return loss

def l2_err(y: torch.Tensor, yref: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.trapz((y - yref)**2, x))




def compute_signatures(path: torch.Tensor, depth: int) -> torch.Tensor:
    # path: (T, d) on CUDA
    if path.dim() == 2:
        path = path.unsqueeze(0)  # (1, T, d)

    # keep on GPU
    pathbp = torch.cat([path[:, :1, :], path], dim=1)  # basepoint on GPU

    sigsraw = keras_sig.signature(
        pathbp,
        depth=depth,
        stream=True,
        gpu_optimized=True,  
    ).to(path.dtype)
    # convert back to torch if needed; but with Keras 3 + torch backend, this should already be a torch tensor
    if not isinstance(sigsraw, torch.Tensor):
        sigsraw = torch.tensor(sigsraw, dtype=path.dtype, device=path.device)

    return sigsraw.squeeze(0).to(path.device)



def build_kernel_from_signatures(sigs_flat: torch.Tensor,
                                 sigma: float = 1.0,
                                 kernel_type: str = "rbf") -> torch.Tensor:
    """
    Build a kernel matrix from signature features.
    sigs_flat: (T,D)
    """
    if kernel_type == "linear":
        Ker = sigs_flat @ sigs_flat.T
    elif kernel_type == "rbf":
        norms = (sigs_flat ** 2).sum(dim=1, keepdim=True)
        d2 = norms + norms.T - 2 * (sigs_flat @ sigs_flat.T)
        Ker = torch.exp(-d2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown kernel_type {kernel_type}")

    return Ker

# --- Normalisation functions on signatures ---
def normalize_none(sigs_flat: torch.Tensor,
                   depth: int,
                   dim: int,
                   **kwargs) -> torch.Tensor:
    """No normalization."""
    return sigs_flat


def normalize_rowwise_unit_norm(Z: torch.Tensor,
                                depth: int,
                                dim: int,
                                eps: float = 1e-8,
                                **kwargs) -> torch.Tensor:
    """
    Row-wise unit-norm scaling.
    For each time t, divide by the row's Euclidean norm:
      Z[t, :] -> Z[t, :] / ||Z[t, :]||_2
    """
    norms = Z.norm(dim=1, keepdim=True)   # (T,1)
    return Z / (norms + eps)


def normalize_colwise_unit_norm(Z: torch.Tensor,
                                depth: int,
                                dim: int,
                                eps: float = 1e-8,
                                **kwargs) -> torch.Tensor:
    """
    Column-wise unit-norm scaling.
    For each feature j, divide by the column's Euclidean norm:
      Z[:, j] -> Z[:, j] / ||Z[:, j]||_2
    """
    norms = Z.norm(dim=0, keepdim=True)   # (1,D)
    return Z / (norms + eps)


def normalize_rowwise_zscore(Z: torch.Tensor,
                             depth: int,
                             dim: int,
                             eps: float = 1e-8,
                             **kwargs) -> torch.Tensor:
    """
    Row-wise z-score normalization.
    For each time t, subtract row mean and divide by row std:
      Z[t, :] -> (Z[t, :] - mean_t) / std_t
    """
    mean = Z.mean(dim=1, keepdim=True)              # (T,1)
    std  = Z.std(dim=1, unbiased=False, keepdim=True)
    return (Z - mean) / (std + eps)


def normalize_colwise_zscore(Z: torch.Tensor,
                             depth: int,
                             dim: int,
                             eps: float = 1e-8,
                             **kwargs) -> torch.Tensor:
    """
    Column-wise z-score normalization.
    For each feature j, subtract feature mean and divide by feature std:
      Z[:, j] -> (Z[:, j] - mu_j) / sigma_j
    """
    mean = Z.mean(dim=0, keepdim=True)              # (1,D)
    std  = Z.std(dim=0, unbiased=False, keepdim=True)
    return (Z - mean) / (std + eps)


def normalize_rowwise_robust(Z: torch.Tensor,
                             depth: int,
                             dim: int,
                             eps: float = 1e-8,
                             **kwargs) -> torch.Tensor:
    """
    Row-wise robust normalization using median and IQR.
    For each time t:
      med_t = median_j Z[t,j]
      q25_t, q75_t = 25th, 75th percentiles over features
      iqr_t = q75_t - q25_t
      Z[t,:] -> (Z[t,:] - med_t) / (iqr_t + eps)
    """
    med = Z.median(dim=1, keepdim=True).values      # (T,1)
    q25 = Z.quantile(0.25, dim=1, keepdim=True)
    q75 = Z.quantile(0.75, dim=1, keepdim=True)
    iqr = q75 - q25
    return (Z - med) / (iqr + eps)


def normalize_colwise_robust(Z: torch.Tensor,
                             depth: int,
                             dim: int,
                             eps: float = 1e-8,
                             **kwargs) -> torch.Tensor:
    """
    Column-wise robust normalization using median and IQR.
    For each feature j:
      med_j = median_t Z[t,j]
      q25_j, q75_j = 25th, 75th percentiles over time
      iqr_j = q75_j - q25_j
      Z[:,j] -> (Z[:,j] - med_j) / (iqr_j + eps)
    """
    med = Z.median(dim=0, keepdim=True).values      # (1,D)
    q25 = Z.quantile(0.25, dim=0, keepdim=True)
    q75 = Z.quantile(0.75, dim=0, keepdim=True)
    iqr = q75 - q25
    return (Z - med) / (iqr + eps)


# Dispatcher
def apply_signature_normalization(sigs_flat: torch.Tensor,
                                  depth: int,
                                  dim: int,
                                  scheme: str,
                                  **kwargs) -> torch.Tensor:
    """
    scheme:
      'none'
      'row_unit'     : row-wise unit-norm
      'col_unit'     : column-wise unit-norm
      'row_zscore'   : row-wise mean/std
      'col_zscore'   : column-wise mean/std
      'row_robust'   : row-wise median/IQR
      'col_robust'   : column-wise median/IQR
    """
    if scheme == "none":
        return normalize_none(sigs_flat, depth, dim, **kwargs)
    elif scheme == "row_unit":
        return normalize_rowwise_unit_norm(sigs_flat, depth, dim, **kwargs)
    elif scheme == "col_unit":
        return normalize_colwise_unit_norm(sigs_flat, depth, dim, **kwargs)
    elif scheme == "row_zscore":
        return normalize_rowwise_zscore(sigs_flat, depth, dim, **kwargs)
    elif scheme == "col_zscore":
        return normalize_colwise_zscore(sigs_flat, depth, dim, **kwargs)
    elif scheme == "row_robust":
        return normalize_rowwise_robust(sigs_flat, depth, dim, **kwargs)
    elif scheme == "col_robust":
        return normalize_colwise_robust(sigs_flat, depth, dim, **kwargs)
    else:
        raise ValueError(f"Unknown normalization scheme '{scheme}'")




def build_kernel_operators(Ksig: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    I1  = torch.cumulative_trapezoid(Ksig, x, dim=0)
    Kup = F.pad(I1, (0, 0, 1, 0))
    I2  = torch.cumulative_trapezoid(Kup, x, dim=0)
    Ku  = F.pad(I2, (0, 0, 1, 0))
    return torch.stack([Ksig, Kup, Ku])   # (3, N, N)


def solve_betas(Ksig: torch.Tensor,
                f: torch.Tensor,
                x: torch.Tensor,
                ua: float,
                upa: float,
                a_fun,
                b_fun,
                c_fun,
                **kwargs):
    K_stack = build_kernel_operators(Ksig, x)   # (3, N, N)
    N  = Ksig.shape[0]
    x0 = x[0]

    a_vec = a_fun(x)
    b_vec = b_fun(x)
    c_vec = c_fun(x)

    # A[i,:] = a(x_i)*K[0,i,:] + b(x_i)*K[1,i,:] + c(x_i)*K[2,i,:]
    A = (a_vec[:, None] * K_stack[0]
       + b_vec[:, None] * K_stack[1]
       + c_vec[:, None] * K_stack[2])

    rhs = f - b_vec * upa - c_vec * (ua + upa * (x - x0))

    beta = torch.linalg.solve(A.T @ A, A.T @ rhs)

    z    = K_stack @ beta          # (3, N)
    u_dd = z[0]
    u_p  = upa + z[1]
    u    = ua + upa * (x - x0) + z[2]

    f_pred = a_vec * u_dd + b_vec * u_p + c_vec * u
    return beta, u, f_pred


def solve_rbf_kernel_plain(
    x: torch.Tensor,        # (N,)
    f: torch.Tensor,        # (N,)
    a_fun,                  # a_fun(t)
    b_fun,                  # b_fun(t)
    c_fun,                  # c_fun(t)
    ua: float,              # u(a)
    upa: float,             # u'(a)
    rbfsigma: float,        # RBF length-scale σ
):
    """
    Baseline solver using a plain pointwise RBF kernel on the time grid.

    K(t_i, t_j) = exp(-|t_i - t_j|^2 / (2 * σ^2))

    No signatures, no path extensions.
    """
    with torch.no_grad():
        t0 = time.time()

        x64 = x.to(torch.float64)
        f64 = f.to(torch.float64)

        # Plain RBF kernel on time
        diff = x64.unsqueeze(0) - x64.unsqueeze(1)   # (N, N)
        d2   = diff ** 2
        Ksig = torch.exp(-d2 / (2.0 * rbfsigma ** 2))  # (N, N)

        print("finding betas (plain RBF kernel)...")
        t1 = time.time()
        beta, u, fpred = solve_betas(
            Ksig=Ksig,
            f=f64,
            x=x64,
            ua=ua,
            upa=upa,
            a_fun=a_fun,
            b_fun=b_fun,
            c_fun=c_fun,
        )
        t2 = time.time()
        print(f"time solve betas (RBF): {t2 - t1:.3f} s")

        final_loss = forcing_loss(f64, fpred)
        print(f"plain RBF model forcing match loss: {final_loss.item():.3e}")
        print(f"total time RBF-only: {t2 - t0:.3f} s")

    return u, fpred



# === Non‑branched (no extension) signature‑kernel solver ===
def solve_signature_kernel_non_branched(x, f,
                                        a_fun, b_fun, c_fun,
                                        ua, upa,
                                        depth, rbf_sigma,
                                        kernel_type: str = "rbf",
                                        norm_scheme: str = "none",
                                        norm_kwargs: dict | None = None):

    if norm_kwargs is None:
        norm_kwargs = {}

    with torch.no_grad():

        X = torch.stack([x, f], dim=1)           # (T,2)

   
        X_sig = compute_signatures(X, depth)     # (T,D)
   
   
        X_sig_norm = apply_signature_normalization(
            X_sig, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs
        )
 
       
        Ksig = build_kernel_from_signatures(
            X_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )
   
        beta_w, u, f_pred_final = solve_betas(
          Ksig=Ksig,
          f=f,
          x=x,
          ua=ua,
          upa=upa,
          a_fun=a_fun,
          b_fun=b_fun,
          c_fun=c_fun,
        )
  

        final_loss = forcing_loss(f, f_pred_final)
 
        print(f"non-branched model forcing match loss: {final_loss.item():.3e}")

    return u, f_pred_final

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

def shuffle_loss_residual(X_bar: torch.Tensor) -> torch.Tensor:
    """
    Return the full shuffle residual matrix R for the extended path X_bar (T,E):
    """
    dX = X_bar[1:, :] - X_bar[:-1, :]    # (T-1, E)
    X_left = X_bar[:-1, :]               # (T-1, E)
    I = torch.einsum('ka,kb->ab', X_left, dX)  # ∑ X^i ΔX^j over k
    deltas = X_bar[-1, :] - X_bar[0, :]        # (E,)
    R = deltas[:, None] * deltas[None, :] - (I + I.T)
    return R


# === Branched (extended‑path) signature‑kernel solver ===
def solve_signature_kernel_branched(x, f,
                                    a_fun, b_fun, c_fun,
                                    ua, upa,
                                    adam_iters, adam_lr,
                                    ADAM_lambda_model, ADAM_lambda_shuffle,
                                    snapshot_count, hidden_dims,
                                    activation_cls, extensions,
                                    adam_use_scheduler, adam_sched_factor,
                                    adam_sched_patience,
                                    depth, rbf_sigma,
                                    kernel_type: str = "rbf",
                                    norm_scheme: str = "none",
                                    norm_kwargs: dict | None = None):

    if norm_kwargs is None:
        norm_kwargs = {}

    #Build extension network
    path_ext = PathExtension(
        input_dim=2,
        output_dim=extensions,
        hidden_dims=hidden_dims,
        activation_cls=activation_cls
    ).to(device)
    path_ext = torch.compile(path_ext)


    # Optimizer (Adam) over both extension net and beta_w
    snapshots = []
    training_history = []
    opt = torch.optim.Adam(
        list(path_ext.parameters()),
        lr=adam_lr
    )

    # Optional LR scheduler
    scheduler = None
    if adam_use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=adam_sched_factor,
            patience=adam_sched_patience,
        )

    # Build the base path
    X = torch.stack([x, f], dim=1)

    # Snapshot schedule for Adam phase
    num_snaps_adam = snapshot_count
    adam_snapshot_epochs = sorted(set(
        int(i) for i in torch.linspace(1, adam_iters, num_snaps_adam)
    ))
    if 1 in adam_snapshot_epochs:
        adam_snapshot_epochs.remove(1)


    # ---------- Adam phase (jointly train extension + beta_w) ----------
    for it in range(1, adam_iters + 1):
        opt.zero_grad()

        # Compute extensions and shuffle loss
        out_ext = path_ext(X)
        shuffle_loss = shuffle_loss_function(out_ext)

        # Build extended path
        X_ext = torch.cat([X, out_ext], dim=1)   # (T, 2+extensions)

        # signatures of extended path
        X_ext_sig = compute_signatures(X_ext, depth)

        # normalization on signatures of extended path (if any)
        X_ext_sig_norm = apply_signature_normalization(
            X_ext_sig, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )



        # kernel from normalized extended signatures
        Ksig = build_kernel_from_signatures(
            X_ext_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )


        beta_w, u_tmp, f_pred = solve_betas(
            Ksig=Ksig,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            a_fun=a_fun,
            b_fun=b_fun,
            c_fun=c_fun,
        )

        pde_loss = forcing_loss(f, f_pred)

        # Total loss (PDE + shuffle)
        loss = ADAM_lambda_model * pde_loss + ADAM_lambda_shuffle * shuffle_loss
        loss.backward()
        opt.step()

        # Scheduler step
        if scheduler is not None:
            scheduler.step(loss.detach())

        # Logging and snapshots
        current_lr = opt.param_groups[0]["lr"]

        training_history.append({
            "iter": it,
            "lr": float(current_lr),
            "total_weighted_loss": float(loss.detach().item()),
            "model_loss": float(pde_loss.detach().item()),
            "shuffle_loss": float(shuffle_loss.detach().item()),
        })

        if it % 50 == 0:
            print(f"[Adam {it:04d}] loss={loss.item():.3e}, "
                  f"PDE={pde_loss.item():.3e}, "
                  f"shuffle={shuffle_loss.item():.3e}")

        if it in adam_snapshot_epochs:
            snapshots.append({
                "phase": "Adam",
                "iter": it,
                "u": u_tmp.detach().clone(),
                "f_pred": f_pred.detach().clone(),
            })


    # ---------- Final evaluation ----------
    with torch.no_grad():
        X_final = torch.stack([x, f], dim=1)
        out_ext_final = path_ext(X_final)


        X_ext_final = torch.cat([X_final, out_ext_final], dim=1)


        X_ext_sig_final = compute_signatures(X_ext_final, depth)
        X_ext_sig_final_norm = apply_signature_normalization(
            X_ext_sig_final, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )

        Ksig_final = build_kernel_from_signatures(
            X_ext_sig_final_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )



        beta_w,u,f_pred_final = solve_betas(
            Ksig=Ksig_final,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            a_fun=a_fun,
            b_fun=b_fun,
            c_fun=c_fun,
        )

        final_loss = forcing_loss(f, f_pred_final)
        print(f"Overall true final loss={final_loss:.3e}")

    return u, snapshots, f_pred_final, path_ext, training_history

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

    # --- Plot 1: learning rate ---
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

    # --- Plot 2: losses in 1x3 grid ---
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


def plot_reference_forcing_and_solution(x, f_true, u_ref):
    """
    Two plots on one row:
      left  - forcing f_true(x)
      right - reference solution u_ref(x)
    """
    t = x.detach().cpu().numpy()
    f_np = f_true.detach().cpu().numpy()
    u_np = u_ref.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # Forcing
    axes[0].plot(t, f_np, 'k-')
    axes[0].set_title("Forcing f(x)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")

    # Reference solution
    axes[1].plot(t, u_np, 'k-')
    axes[1].set_title("Reference solution u(x)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("u(x)")

    plt.tight_layout()
    plt.show()

def plot_final_comparison_all(x,
                              f_true,
                              u_ref,
                              u_rbf, f_rbf,
                              u_nb,  f_nb,
                              u_sig, f_sig):
    """
    2x3 figure:

      row 0: forcing comparisons vs true
      row 1: solution comparisons vs reference

      col 0: plain RBF (no signatures)
      col 1: non-branched signature kernel
      col 2: branched signature kernel
    """
    t = x.detach().cpu().numpy()
    f_true_np = f_true.detach().cpu().numpy()
    u_ref_np  = u_ref.detach().cpu().numpy()

    u_rbf_np = u_rbf.detach().cpu().numpy()
    f_rbf_np = f_rbf.detach().cpu().numpy()

    u_nb_np  = u_nb.detach().cpu().numpy()
    f_nb_np  = f_nb.detach().cpu().numpy()

    u_sig_np = u_sig.detach().cpu().numpy()
    f_sig_np = f_sig.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex='col')

    # --- Forcing ---
    ax = axes[0, 0]
    ax.plot(t, f_true_np, 'k-', label="true f(x)")
    ax.plot(t, f_rbf_np,  'g--', label="rbf pred f")
    ax.set_title("Forcing: RBF vs true")
    ax.set_ylabel("f(x)")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(t, f_true_np, 'k-', label="true f(x)")
    ax.plot(t, f_nb_np,   'b--', label="non-branched pred f")
    ax.set_title("Forcing: non-branched vs true")
    ax.legend()

    ax = axes[0, 2]
    ax.plot(t, f_true_np, 'k-', label="true f(x)")
    ax.plot(t, f_sig_np,  'r--', label="branched pred f")
    ax.set_title("Forcing: branched vs true")
    ax.legend()

    # --- Solution ---
    ax = axes[1, 0]
    ax.plot(t, u_ref_np, 'k-', label="reference u(x)")
    ax.plot(t, u_rbf_np, 'g--', label="rbf pred u")
    ax.set_title("Solution: RBF vs reference")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t, u_ref_np, 'k-', label="reference u(x)")
    ax.plot(t, u_nb_np,  'b--', label="non-branched pred u")
    ax.set_title("Solution: non-branched vs reference")
    ax.set_xlabel("x")
    ax.legend()

    ax = axes[1, 2]
    ax.plot(t, u_ref_np, 'k-', label="reference u(x)")
    ax.plot(t, u_sig_np, 'r--', label="branched pred u")
    ax.set_title("Solution: branched vs reference")
    ax.set_xlabel("x")
    ax.legend()

    plt.tight_layout()
    plt.show()

def compute_mse_and_relative_mse(y_true, y_pred):
    y_true_cpu = y_true.detach().cpu()
    y_pred_cpu = y_pred.detach().cpu()

    mse = torch.mean((y_pred_cpu - y_true_cpu) ** 2).item()
    rel_mse = mse / (torch.mean(y_true_cpu ** 2).item() + 1e-12)
    return mse, rel_mse

def plot_final_comparison_2x2(x,
                              f_true,
                              u_ref,
                              u_sig, f_sig,
                              u_nb,  f_nb):
    """
    2x2 figure:

      [0,0]: forcing (branched vs true)
      [0,1]: forcing (non-branched vs true)
      [1,0]: solution (branched vs reference)
      [1,1]: solution (non-branched vs reference)
    """
    t = x.detach().cpu().numpy()
    f_true_np = f_true.detach().cpu().numpy()
    u_ref_np  = u_ref.detach().cpu().numpy()
    u_sig_np  = u_sig.detach().cpu().numpy()
    f_sig_np  = f_sig.detach().cpu().numpy()
    u_nb_np   = u_nb.detach().cpu().numpy()
    f_nb_np   = f_nb.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col')

    # Forcing: branched vs true
    ax = axes[0, 0]
    ax.plot(t, f_true_np, 'k-', label="true f(x)")
    ax.plot(t, f_sig_np,  'r--', label="branched pred f")
    ax.set_title("Forcing: branched vs true")
    ax.set_ylabel("f(x)")
    ax.legend()

    # Forcing: non-branched vs true
    ax = axes[0, 1]
    ax.plot(t, f_true_np, 'k-', label="true f(x)")
    ax.plot(t, f_nb_np,   'b--', label="non-branched pred f")
    ax.set_title("Forcing: non-branched vs true")
    ax.legend()

    # Solution: branched vs reference
    ax = axes[1, 0]
    ax.plot(t, u_ref_np, 'k-', label="reference u(x)")
    ax.plot(t, u_sig_np, 'r--', label="branched pred u")
    ax.set_title("Solution: branched vs reference")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend()

    # Solution: non-branched vs reference
    ax = axes[1, 1]
    ax.plot(t, u_ref_np, 'k-', label="reference u(x)")
    ax.plot(t, u_nb_np,  'b--', label="non-branched pred u")
    ax.set_title("Solution: non-branched vs reference")
    ax.set_xlabel("x")
    ax.legend()

    plt.tight_layout()
    plt.show()

def plot_snapshot_evolution_branched(x,
                                     f_true,
                                     u_ref,
                                     snapshots_branched):
    """
    1x2 figure with evolution for the branched model only:

      [0]: forcing snapshots (branched) vs true
      [1]: solution snapshots (branched) vs reference
    """
    t = x.detach().cpu().numpy()
    f_true_np = f_true.detach().cpu().numpy()
    u_ref_np  = u_ref.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex='col')

    # Forcing evolution: branched
    ax = axes[0]
    ax.plot(t, f_true_np, 'k--', linewidth=2, label="true f(x)")
    for snap in snapshots_branched:
        f_pred = snap["f_pred"].detach().cpu().numpy()
        label = f"{snap['phase']} {snap['iter']}"
        ax.plot(t, f_pred, alpha=0.7, label=label)
    ax.set_title("Forcing evolution (branched)")
    ax.set_ylabel("f(x)")
    ax.legend(fontsize=8)

    # Solution evolution: branched
    ax = axes[1]
    ax.plot(t, u_ref_np, 'k--', linewidth=2, label="reference u(x)")
    for snap in snapshots_branched:
        u_hat = snap["u"].detach().cpu().numpy()
        label = f"{snap['phase']} {snap['iter']}"
        ax.plot(t, u_hat, alpha=0.7, label=label)
    ax.set_title("Solution evolution (branched)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()




def plot_learned_extensions(x, f_true, X_ext, title_prefix="Learned extensions"):
    """
    X_ext is the full extended path (T, 2+E) or just extensions (T, E).
    Plots forcing on the first row, each extension on its own row below.
    """
    t = x.detach().cpu().numpy()
    f_np = f_true.detach().cpu().numpy()

    # If X_ext includes base channels, strip them; otherwise assume it's only extensions.
    if X_ext.shape[1] > 2:
        ext_np = X_ext[:, 2:].detach().cpu().numpy()
    else:
        ext_np = X_ext.detach().cpu().numpy()

    E = ext_np.shape[1]
    nrows = 1 + E   # 1 for forcing, E for extensions

    fig, axes = plt.subplots(
        nrows, 1,
        figsize=(10, 2.0 * nrows),
        sharex=True
    )

    if nrows == 1:
        axes = [axes]

    # Row 0: forcing
    ax0 = axes[0]
    ax0.plot(t, f_np, 'k-')
    ax0.set_ylabel("f(x)")
    ax0.set_title(f"{title_prefix}: forcing")

    # Rows 1..E: each extension
    for i in range(E):
        ax = axes[i + 1]
        ax.plot(t, ext_np[:, i], 'b-')
        ax.set_ylabel(f"ext {i+1}")

    axes[-1].set_xlabel("x")

    fig.tight_layout()
    plt.show()


def print_errors(label, u_ref, u_pred, f_true, f_pred,
                 ref_rel_u=None, ref_rel_f=None):
    """
    Print MSE, absolute (L2), and relative (L2) errors
    of solution and forcing vs IVP reference.
    """

    # Move predictions to CPU to match u_ref and f_true (which come from SciPy/CPU)
    u_pred_cpu = u_pred.detach().cpu()
    f_pred_cpu = f_pred.detach().cpu()
    u_ref_cpu  = u_ref.detach().cpu()
    f_true_cpu = f_true.detach().cpu()

    # solution errors
    mse_u = torch.mean((u_pred_cpu - u_ref_cpu) ** 2).item()
    abs_u = torch.sqrt(torch.mean((u_pred_cpu - u_ref_cpu) ** 2)).item()
    rel_u = abs_u / (torch.sqrt(torch.mean(u_ref_cpu ** 2)).item() + 1e-12)

    # forcing errors
    mse_f = torch.mean((f_pred_cpu - f_true_cpu) ** 2).item()
    abs_f = torch.sqrt(torch.mean((f_pred_cpu - f_true_cpu) ** 2)).item()
    rel_f = abs_f / (torch.sqrt(torch.mean(f_true_cpu ** 2)).item() + 1e-12)

    print(f"\n[{label}] solution errors vs IVP:")
    print(f"  MSE(u)      = {mse_u:.3e}")
    print(f"  Abs L2(u)   = {abs_u:.3e}")
    print(f"  Rel L2(u)   = {rel_u:.3e}")

    if ref_rel_u is not None:
        imp_u = (ref_rel_u - rel_u) / (ref_rel_u + 1e-12)
        print(f"  Rel improv vs non branched model = {100*imp_u:.2f}%")

    print(f"[{label}] forcing errors vs true f:")
    print(f"  MSE(f)      = {mse_f:.3e}")
    print(f"  Abs L2(f)   = {abs_f:.3e}")
    print(f"  Rel L2(f)   = {rel_f:.3e}")

    if ref_rel_f is not None:
        imp_f = (ref_rel_f - rel_f) / (ref_rel_f + 1e-12)
        print(f"  Rel improv vs non branched model = {100*imp_f:.2f}%")

    return rel_u, rel_f



def plot_shuffle_residual_matrix(x, f_true, path_ext, title="Shuffle Product Loss (Residual Matrix)"):
    """
    Compute the extension path for (x, f_true) using path_ext, then
    plot the shuffle residual matrix R as a heatmap:

      R_{ij} = ΔX^i ΔX^j - (∫ X^i dX^j + ∫ X^j dX^i)

    where X are the extension channels.
    """
    path_ext.eval()
    with torch.no_grad():
        X = torch.stack([x, f_true], dim=1)      # (T, 2)
        X_bar = path_ext(X)                      # (T, E)

        R = shuffle_loss_residual(X_bar)         # (E, E)
        R_np = R.detach().cpu().numpy()

    C = R_np.shape[0]
    v = np.abs(R_np).max() + 1e-12

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(R_np, cmap="bwr", vmin=-v, vmax=v,
                   interpolation="nearest", aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        r"$R_{ij}=\Delta X^i\,\Delta X^j - \left(\int X^i\,dX^j + \int X^j\,dX^i\right)$",
        rotation=90, labelpad=12
    )

    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")

    positions = np.arange(C)
    labels = [str(i+1) for i in range(C)]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)

    ax.set_xlim(-0.5, C-0.5)
    ax.set_ylim(C-0.5, -0.5)

    plt.show()

def plot_rbf_final_comparison(x, f_true, u_ref, urbf, fpred_rbf):
    """
    1x2 figure for plain RBF (no signatures, no extensions):

      left  - forcing RBF vs true
      right - solution RBF vs reference
    """
    t = x.detach().cpu().numpy()
    f_true_np = f_true.detach().cpu().numpy()
    u_ref_np = u_ref.detach().cpu().numpy()
    urbf_np = urbf.detach().cpu().numpy()
    fpred_rbf_np = fpred_rbf.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    ax = axes[0]
    ax.plot(t, f_true_np, 'k-', label="true f(t)")
    ax.plot(t, fpred_rbf_np, 'g--', label="rbf pred f")
    ax.set_title("Forcing RBF vs true")
    ax.set_xlabel("t")
    ax.set_ylabel("f(t)")
    ax.legend()

    ax = axes[1]
    ax.plot(t, u_ref_np, 'k-', label="reference u(t)")
    ax.plot(t, urbf_np, 'g--', label="rbf pred u")
    ax.set_title("Solution RBF vs reference")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.legend()

    plt.tight_layout()
    plt.show()


def compute_mse_and_relative_mse(y_true, y_pred):
    y_true_cpu = y_true.detach().cpu()
    y_pred_cpu = y_pred.detach().cpu()

    mse = torch.mean((y_pred_cpu - y_true_cpu) ** 2).item()
    rel_mse = mse / (torch.mean(y_true_cpu ** 2).item() + 1e-12)
    return mse, rel_mse

def plot_overlay_reference_rbf_branched_1x2(
    x,
    f_true,
    u_ref,
    f_rbf,
    u_rbf,
    f_sig,
    u_sig
):
    """
    1x2 overlay figure:
      left  - forcing: true, RBF no-sig, branched
      right - solution: reference, RBF no-sig, branched
    """
    t = x.detach().cpu().numpy()

    f_true_np = f_true.detach().cpu().numpy()
    u_ref_np  = u_ref.detach().cpu().numpy()

    f_rbf_np = f_rbf.detach().cpu().numpy()
    u_rbf_np = u_rbf.detach().cpu().numpy()

    f_sig_np = f_sig.detach().cpu().numpy()
    u_sig_np = u_sig.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharex=True)

    # Forcing overlay
    ax = axes[0]
    ax.plot(t, f_true_np, 'k-', linewidth=2, alpha=0.8, zorder=2, label='true forcing')
    ax.plot(t, f_rbf_np, color='limegreen', linestyle='--', linewidth=3,
            marker='o', markevery=20, markersize=3, zorder=5, label='rbf no sig')
    ax.plot(t, f_sig_np, 'r-.', linewidth=2, alpha=0.85, zorder=3, label='branched')
    ax.set_title('Forcing overlay')
    ax.set_xlabel('t')
    ax.set_ylabel('f(t)')
    ax.legend()

    # Solution overlay
    ax = axes[1]
    ax.plot(t, u_ref_np, 'k-', linewidth=2, alpha=0.8, zorder=2, label='reference solution')
    ax.plot(t, u_rbf_np, color='limegreen', linestyle='--', linewidth=3,
            marker='o', markevery=20, markersize=3, zorder=5, label='rbf no sig')
    ax.plot(t, u_sig_np, 'r-.', linewidth=2, alpha=0.85, zorder=3, label='branched')
    ax.set_title('Solution overlay')
    ax.set_xlabel('t')
    ax.set_ylabel('u(t)')
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_forcing_overlay_reference_rbf_branched(
    x,
    f_true,
    f_rbf,
    f_sig
):
    """
    1x1 overlay figure for forcing only:
      true, RBF no-sig, branched
    """
    t = x.detach().cpu().numpy()

    f_true_np = f_true.detach().cpu().numpy()
    f_rbf_np  = f_rbf.detach().cpu().numpy()
    f_sig_np  = f_sig.detach().cpu().numpy()

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, f_true_np, 'k-', linewidth=2, alpha=0.8, zorder=2, label='true forcing')
    plt.plot(t, f_rbf_np, color='limegreen', linestyle='--', linewidth=3,
             marker='o', markevery=20, markersize=3, zorder=5, label='rbf no sig')
    plt.plot(t, f_sig_np, 'r-.', linewidth=2, alpha=0.85, zorder=3, label='branched')
    plt.title('Forcing overlay')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend()
    plt.tight_layout()
    plt.show()
