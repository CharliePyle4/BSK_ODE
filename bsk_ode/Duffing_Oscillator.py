import os
os.environ["KERAS_BACKEND"] = "torch"  # before importing keras_sig / keras

SEED = 42


import random

import torch
torch.set_default_dtype(torch.float64)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device, "torch", torch.__version__)

import time
import math
import numpy as np
import torchmin
from torch import nn
from scipy.integrate import solve_ivp
from torchmin import least_squares
from torchmin import minimize
from torch import cumulative_trapezoid
from stochastic.processes.continuous import FractionalBrownianMotion

import keras_sig
from keras_sig import SigLayer

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
import random
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
# ─────────────────────────────────────────────────────────────────────────────



def f_forcing_fbm(x: torch.Tensor, hurst: float = 0.2) -> torch.Tensor:
    """Fractional Brownian motion on [a,b] using the stochastic library."""
    N = x.numel()
    a_ = float(x[0])
    b_ = float(x[-1])
    length = b_ - a_

    # Create a seeded random number generator for reproducibility.
    # The 'stochastic' library uses the new numpy Generator API, which is not
    # seeded by the legacy np.random.seed(). We create a new generator from
    # the global SEED constant.
    rng = np.random.default_rng(SEED)

    fbm_gen = FractionalBrownianMotion(hurst=hurst, t=length, rng=rng)
    # sample(N-1) generates N points for the interval [0, length].
    fbm_sample = fbm_gen.sample(n=N-1)

    return torch.tensor(fbm_sample, dtype=x.dtype, device=x.device)


def solve_duffing_ivp(x_grid: torch.Tensor,
                      forcing_torch: torch.Tensor,
                      a: float, b: float,
                      ya: float, ypa: float,
                      k1: float, k2: float, k3: float):
    """
    Solve u'' + k1 u' + k2 u + k3 u^3 = f(x)
    with u(a)=ya, u'(a)=ypa on grid x_grid, using forcing_torch on that grid.
    """
    t_eval = x_grid.cpu().numpy()
    f_np   = forcing_torch.cpu().numpy()
    dx = t_eval[1] - t_eval[0]

    def forcing(t):
        idx = int(round((t - t_eval[0]) / dx))
        idx = max(0, min(len(f_np) - 1, idx))
        return f_np[idx]

    def fun(t, y):
        u, up = y
        f_val = forcing(t)
        du_dt  = up
        dup_dt = -k1 * up - k2 * u - k3 * u**3 + f_val
        return [du_dt, dup_dt]

    y0 = [ya, ypa]
    sol = solve_ivp(fun, (a, b), y0, t_eval=t_eval,
                    method="BDF", rtol=1e-6, atol=1e-9, max_step=0.05)
    u_ivp = sol.y[0]
    return torch.tensor(t_eval, dtype=torch.float64), torch.tensor(u_ivp, dtype=torch.float64)

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
'''
def compute_signatures(path: torch.Tensor,
                       depth: int) -> torch.Tensor:
    """
    path: (T, d) or (1, T, d) torch tensor.
    Returns: (T, D) prefix-signature features as torch.Tensor,
             on the same device as the input.
    """
    orig_device = path.device
    orig_dtype  = path.dtype

    if path.dim() == 2:
        path = path.unsqueeze(0)          # (1, T, d)

    # keras_sig calls .numpy() internally — must be on CPU
    path_cpu = path.detach().cpu()

    # Prepend basepoint, emulating Signatory's basepoint behaviour
    basepoint = path_cpu[:, 0:1, :]
    path_bp   = torch.cat([basepoint, path_cpu], dim=1)  # (1, T+1, d)

    # keras_sig may return a TF or torch tensor depending on the backend
    # that was active when keras was first imported — convert via numpy
    # to be safe in all cases.
    sigs_raw = keras_sig.signature(
        path_bp,
        depth=depth,
        stream=True,
        gpu_optimized=True
    )                                           # (1, T, D) — unknown tensor type

    # Convert to torch regardless of backend
    if not isinstance(sigs_raw, torch.Tensor):
        import numpy as np
        sigs_raw = torch.tensor(np.array(sigs_raw), dtype=orig_dtype)
    else:
        sigs_raw = sigs_raw.detach().to(dtype=orig_dtype)

    return sigs_raw.squeeze(0).to(orig_device)  # (T, D)
'''
def computesignatures(path: torch.Tensor, depth: int) -> torch.Tensor:
    # path: (T, d) on CUDA
    if path.dim() == 2:
        path = path.unsqueeze(0)  # (1, T, d)

    # keep on GPU
    pathbp = torch.cat([path[:, :1, :], path], dim=1)  # basepoint on GPU

    sigsraw = keras_sig.signature(
        pathbp,
        depth=depth,
        stream=True,
        gpu_optimized=True,  # now really uses GPU kernels
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
        d2 = torch.clamp(d2, min=0.0)
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
    



'''def build_kernel_operators_method1(Ksig: torch.Tensor,
                           x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized construction of Ku2, Kup, Ku from Ksig.
    """
    # Ksig: (N, N), integrate along rows (dim=0) w.r.t. x (N,)
    Ku2 = Ksig.clone()  # u''

    # first integral: shape (N-1, N), then pad a leading zero row
    I1 = cumulative_trapezoid(Ku2, x, dim=0)          # (N-1, N)
    I1 = torch.vstack([torch.zeros_like(I1[0:1, :]), I1])  # (N, N)
    Kup = I1

    # second integral: integrate I1 again along x
    I2 = cumulative_trapezoid(I1, x, dim=0)           # (N-1, N)
    I2 = torch.vstack([torch.zeros_like(I2[0:1, :]), I2])  # (N, N)
    Ku = I2

    return Ku2, Kup, Ku
'''

import torch.nn.functional as F

def build_kernel_operators_method1(Ksig, x):
    I1  = torch.cumulative_trapezoid(Ksig, x, dim=0)
    Kup = F.pad(I1, (0, 0, 1, 0))
    I2  = torch.cumulative_trapezoid(Kup, x, dim=0)
    Ku  = F.pad(I2, (0, 0, 1, 0))
    return torch.stack([Ksig, Kup, Ku])   # (3, N, N)
'''
def solve_betas_method1(Ksig: torch.Tensor,
                f: torch.Tensor,
                x: torch.Tensor,
                ua: float,
                upa: float,
                k1: float,
                k2: float,
                k3: float,
                beta_init: torch.Tensor | None = None,
                max_iter: int = 500,
                method: str = "l-bfgs") -> tuple[torch.Tensor,
                                                    torch.Tensor,
                                                    torch.Tensor]:
    """
    Returns (beta_opt, u, u_p, u_dd, f_pred).
    Mapping:
      Ku2, Kup, Ku from build_kernel_operators,
      u_dd = Ku2 @ beta,
      u_p  = upa + Kup @ beta,
      u    = ua + upa * (x - x[0]) + Ku @ beta,
      f_pred = u_dd + k1 u_p + k2 u + k3 u^3.
    """

    Ku2, Kup, Ku = build_kernel_operators_method1(Ksig, x)   # (N,N)
    N = Ksig.shape[0]

    if beta_init is None:
        beta0 = torch.zeros(N, dtype=Ksig.dtype, device=Ksig.device)
    else:
        beta0 = beta_init.clone().detach().to(Ksig.device).reshape(-1)

    """
    def forward(beta: torch.Tensor):

        z_u2 = Ku2 @ beta
        z_u  = Ku  @ beta
        z_up = Kup @ beta

        u_dd = z_u2
        u_p  = upa + z_up
        u    = ua + upa * (x - x[0]) + z_u

        linear_term    = u_dd + k1 * u_p + k2 * u
        nonlinear_term = k3 * u**3
        f_pred = linear_term + nonlinear_term
        return u, u_p, u_dd, f_pred
    """
    
    def forward(beta):
        # Stack operators: (3, N, N) @ (N,) -> (3, N) in one CUDA call
        z = torch.stack([Ku2, Kup, Ku]) @ beta   # (3, N)
        u_dd = z[0]
        u_p  = upa + z[1]
        u    = ua + upa * (x - x[0]) + z[2]
        f_pred = u_dd + k1 * u_p + k2 * u + k3 * u**3
        return u, u_p, u_dd, f_pred

    def residual(beta: torch.Tensor) -> torch.Tensor:
        _, _, _, f_pred = forward(beta)
        return (f_pred - f).reshape(-1)

    def objective(beta: torch.Tensor) -> torch.Tensor:
        r = residual(beta)
        return 0.5 * (r * r).sum()

    res = minimize(
        objective,
        beta0,
        method=method,
        max_iter=max_iter,
        tol=1e-8,
    )




    beta_opt = res.x
    u, u_p, u_dd, f_pred = forward(beta_opt)
    return beta_opt, u, f_pred

'''

def solve_betas_method1(K_stack, f, x, ua, upa, k1, k2, k3,
                        beta_init=None, max_iter=500, method="l-bfgs"):
    # K_stack: (3, N, N)
    N = K_stack.shape[-1]
    x_shifted = x - x[0]

    beta = (torch.zeros(N, dtype=x.dtype, device=x.device)
            if beta_init is None
            else beta_init.to(x.device).reshape(-1).clone())
    beta.requires_grad_(True)

    optimizer = torch.optim.LBFGS(
        [beta], max_iter=max_iter,
        tolerance_grad=1e-9, tolerance_change=1e-12,
        history_size=20, line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        z = K_stack @ beta          # (3, N)
        u_dd = z[0]
        u_p  = upa + z[1]
        u    = ua + upa * x_shifted + z[2]
        loss = 0.5 * ((u_dd + k1*u_p + k2*u + k3*u**3 - f) ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        z = K_stack @ beta.detach()
        u_dd = z[0]
        u_p  = upa + z[1]
        u    = ua + upa * x_shifted + z[2]
        f_pred = u_dd + k1*u_p + k2*u + k3*u**3

    return beta.detach(), u, f_pred

def build_kernel_from_different_signatures(
    sigs_flat1: torch.Tensor,
    sigs_flat2: torch.Tensor,
    sigma: float = 1.0,
    kernel_type: str = "rbf",
) -> torch.Tensor:
    """
    Build a cross-kernel matrix from two sets of signature features.

    sigs_flat1: (T1, D)
    sigs_flat2: (T2, D)
    Returns:
        Ker: (T1, T2)
    """
    if kernel_type == "linear":
        # K_ij = <sigs_flat1[i], sigs_flat2[j]>
        Ker = sigs_flat1 @ sigs_flat2.T
    elif kernel_type == "rbf":
        # pairwise squared distances between rows of sigs_flat1 and sigs_flat2
        norms1 = (sigs_flat1 ** 2).sum(dim=1, keepdim=True)    # (T1, 1)
        norms2 = (sigs_flat2 ** 2).sum(dim=1, keepdim=True)    # (T2, 1)
        # d2_ij = ||x_i||^2 + ||y_j||^2 - 2 <x_i, y_j>
        d2 = norms1 + norms2.T - 2 * (sigs_flat1 @ sigs_flat2.T)  # (T1, T2)
        Ker = torch.exp(-d2 / (2 * sigma**2))
    else:
        raise ValueError(f"Unknown kernel_type {kernel_type}")

    return Ker

def apply_signature_normalization_pair(sigs_train: torch.Tensor,
                                       sigs_full: torch.Tensor,
                                       depth: int,
                                       dim: int,
                                       scheme: str,
                                       **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize both train and full signatures using TRAINING statistics only.
    Use this in the no-retrain test functions instead of calling
    apply_signature_normalization twice.

    Returns:
        (sigs_train_norm, sigs_full_norm)
    """
    eps = kwargs.get("eps", 1e-8)

    if scheme == "none":
        return sigs_train, sigs_full

    elif scheme == "col_unit":
        norms = sigs_train.norm(dim=0, keepdim=True)
        return sigs_train / (norms + eps), sigs_full / (norms + eps)

    elif scheme == "col_zscore":
        mean = sigs_train.mean(dim=0, keepdim=True)
        std  = sigs_train.std(dim=0, unbiased=False, keepdim=True)
        return (sigs_train - mean) / (std + eps), (sigs_full - mean) / (std + eps)

    elif scheme == "col_robust":
        med = sigs_train.median(dim=0, keepdim=True).values
        q25 = sigs_train.quantile(0.25, dim=0, keepdim=True)
        q75 = sigs_train.quantile(0.75, dim=0, keepdim=True)
        iqr = q75 - q25
        return (sigs_train - med) / (iqr + eps), (sigs_full - med) / (iqr + eps)

    else:
        raise ValueError(f"Unknown normalization scheme '{scheme}'")

'''def evaluate_solution_from_beta_method1(
    Ku2: torch.Tensor,
    Kup: torch.Tensor,
    Ku: torch.Tensor,
    x: torch.Tensor,
    beta: torch.Tensor,
    ua: float,
    upa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given kernel operators Ku2, Kup, Ku and coefficients beta,
    reconstruct u, u', u'' on the grid x.
    """
    x0 = x[0]

    u_dd = Ku2 @ beta
    u_p  = upa + Kup @ beta
    u    = ua + upa * (x - x0) + Ku @ beta

    return u, u_p, u_dd'''

def evaluate_solution_from_beta_method1(K_stack, x, beta, ua, upa):
    z    = K_stack @ beta          # (3, N) — one batched GEMV
    u_dd = z[0]
    u_p  = upa + z[1]
    u    = ua + upa * (x - x[0]) + z[2]
    return u, u_p, u_dd

def evaluate_forcing_from_solution_method1(
    u: torch.Tensor,
    u_p: torch.Tensor,
    u_dd: torch.Tensor,
    k1: float,
    k2: float,
    k3: float,
) -> torch.Tensor:
    """
    Compute predicted forcing for the Duffing equation:
    f_pred = u'' + k1*u' + k2*u + k3*u^3
    """
    f_pred = u_dd + k1 * u_p + k2 * u + k3 * u**3
    return f_pred




class PathExtension(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 #hidden_dims=(512, 256, 128, 64, 32, 16),
                 hidden_dims=(16, 32, 64, 128, 64, 32, 16),
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

# === Non‑branched (no extension) signature‑kernel solver ===
def solve_signature_kernel_non_branched_method1(x, f,
                                        k1, k2, k3,
                                        ua, upa,
                                        depth, rbf_sigma,
                                        beta_opt_method: str = "l-bfgs",
                                        beta_iterations: int = 500,
                                        kernel_type: str = "rbf",
                                        norm_scheme: str = "none",
                                        norm_kwargs: dict | None = None):

    if norm_kwargs is None:
        norm_kwargs = {}

    with torch.no_grad():

        X = torch.stack([x, f], dim=1)           # (T,2)

        X_sig = computesignatures(X, depth)      # (T,D)

        X_sig_norm = apply_signature_normalization(X_sig, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)

        Ksig = build_kernel_from_signatures(X_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type)

        K_stack = build_kernel_operators_method1(Ksig, x)

        beta_w, u, f_pred_final = solve_betas_method1(
            K_stack=K_stack,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3,
            beta_init=None,
            max_iter=beta_iterations,
            method=beta_opt_method,
        )

        final_loss = forcing_loss(f, f_pred_final)
        print(f"method 1, non-branched model forcing match loss: {final_loss.item():.3e}")

    return u, f_pred_final


'''# === Branched (extended‑path) signature‑kernel solver ===
def solve_signature_kernel_branched_method1(x, f,
                                    k1, k2, k3,
                                    ua, upa,
                                    adam_iters, adam_lr,
                                    ADAM_lambda_model, ADAM_lambda_shuffle,
                                    snapshot_count, hidden_dims,
                                    activation_cls, extensions,
                                    adam_use_scheduler, adam_sched_factor,
                                    adam_sched_patience,
                                    depth, rbf_sigma,
                                    beta_opt_method: str = "l-bfgs",
                                    beta_iterations: int = 500,
                                    kernel_type: str = "rbf",
                                    norm_scheme: str = "none",
                                    norm_kwargs: dict | None = None):

    if norm_kwargs is None:
        norm_kwargs = {}

    # Build extension network
    path_ext = PathExtension(
        input_dim=2,
        output_dim=extensions,
        hidden_dims=hidden_dims,
        activation_cls=activation_cls
    ).to(device, dtype=torch.float64)

    path_ext = torch.compile(path_ext)

    # Optimizer (Adam) over extension net parameters only
    snapshots = []
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

    beta_prev = None  # warm-start: reuse beta from previous Adam iteration

    # ---------- Adam phase (jointly train extension + beta_w) ----------
    for it in range(1, adam_iters + 1):
        opt.zero_grad()

        # Compute extensions and shuffle loss
        out_ext = path_ext(X)
        shuffle_loss = shuffle_loss_function(out_ext)

        # Build extended path
        X_ext = torch.cat([X, out_ext], dim=1)  # (T, 2+extensions) directly

        # Signatures of extended path
        X_ext_sig = computesignatures(X_ext, depth)

        # Normalization on signatures of extended path (if any)
        X_ext_sig_norm = apply_signature_normalization(
            X_ext_sig, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )


        # Kernel from normalized extended signatures
        Ksig = build_kernel_from_signatures(
            X_ext_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )

        # Build operators once
        K_stack = build_kernel_operators_method1(Ksig, x)
        K_stack_detached = K_stack.detach()

        # Beta solve on detached K_stack — avoids double-backward
        beta_w, u_tmp, _ = solve_betas_method1(
            K_stack=K_stack_detached,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3,
            beta_init=beta_prev,
            max_iter=beta_iterations,
            method=beta_opt_method,
        )

        beta_prev = beta_w.detach().clone()

        # Recompute f_pred through live K_stack so pde_loss trains path_ext
        z = K_stack @ beta_w           # uses non-detached K_stack
        u_tmp = ua + upa * (x - x[0]) + z[2]
        f_pred = z[0] + k1 * (upa + z[1]) + k2 * u_tmp + k3 * u_tmp**3

        pde_loss = forcing_loss(f, f_pred)


        # Total loss (PDE + shuffle)
        loss = ADAM_lambda_model * pde_loss + ADAM_lambda_shuffle * shuffle_loss
        loss.backward()
        opt.step()

        # Scheduler step
        if scheduler is not None:
            scheduler.step(loss.detach())

        # Logging and snapshots
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

        X_ext_sig_final = computesignatures(X_ext_final, depth)
        X_ext_sig_final_norm = apply_signature_normalization(
            X_ext_sig_final, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )



        Ksig_final = build_kernel_from_signatures(
            X_ext_sig_final_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )
        K_stack_final = build_kernel_operators_method1(Ksig_final, x)

        beta_w, u, f_pred_final = solve_betas_method1(
            K_stack=K_stack_final,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3,
            beta_init=None,
            max_iter=beta_iterations,
            method=beta_opt_method,
        )

        final_loss = forcing_loss(f, f_pred_final)
        print(f"Overall true final loss={final_loss:.3e}")

    return u, snapshots, f_pred_final, path_ext
'''


# === Branched (extended‑path) signature‑kernel solver ===
def solve_signature_kernel_branched_method1(x, f,
                                    k1, k2, k3,
                                    ua, upa,
                                    adam_iters, adam_lr,
                                    ADAM_lambda_model, ADAM_lambda_shuffle,
                                    snapshot_count, hidden_dims,
                                    activation_cls, extensions,
                                    adam_use_scheduler, adam_sched_factor,
                                    adam_sched_patience,
                                    depth, rbf_sigma,
                                    beta_opt_method: str = "l-bfgs",
                                    beta_iterations: int = 500,
                                    kernel_type: str = "rbf",
                                    norm_scheme: str = "none",
                                    norm_kwargs: dict | None = None,
                                    beta_solve_every: int = 1,
                                    beta_min_iterations: int = 20,
                                    beta_ramp_portion: float = 0.3,
                                    ):

    if norm_kwargs is None:
        norm_kwargs = {}

    # Build extension network
    path_ext = PathExtension(
        input_dim=2,
        output_dim=extensions,
        hidden_dims=hidden_dims,
        activation_cls=activation_cls
    ).to(device, dtype=torch.float64)

    path_ext = torch.compile(path_ext)

    # Optimizer (Adam) over extension net parameters only
    snapshots = []
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

    beta_prev = None  # warm-start: reuse beta from previous Adam iteration

    # ---------- Adam phase (jointly train extension + beta_w) ----------
    for it in range(1, adam_iters + 1):
        opt.zero_grad()

        # Compute extensions and shuffle loss
        out_ext = path_ext(X)
        shuffle_loss = shuffle_loss_function(out_ext)

        # Build extended path
        X_ext = torch.cat([X, out_ext], dim=1)  # (T, 2+extensions) directly

        # Signatures of extended path
        X_ext_sig = computesignatures(X_ext, depth)

        # Normalization on signatures of extended path (if any)
        X_ext_sig_norm = apply_signature_normalization(
            X_ext_sig, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )

        # Kernel from normalized extended signatures
        Ksig = build_kernel_from_signatures(
            X_ext_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )

        # Build operators once
        K_stack = build_kernel_operators_method1(Ksig, x)

        # Decide whether to re-solve beta this iteration
        if (it == 1) or (it % beta_solve_every == 0):
            # Ramp LBFGS iterations from beta_min_iterations -> beta_iterations
            progress = min(1.0, it / (adam_iters * beta_ramp_portion))
            lbfgs_iters = int(
                beta_min_iterations
                + (beta_iterations - beta_min_iterations) * progress
            )

            # Beta solve on detached K_stack — avoids double-backward
            K_stack_detached = K_stack.detach()
            beta_w, u_tmp, _ = solve_betas_method1(
                K_stack=K_stack_detached,
                f=f,
                x=x,
                ua=ua,
                upa=upa,
                k1=k1,
                k2=k2,
                k3=k3,
                beta_init=beta_prev,
                max_iter=lbfgs_iters,
                method=beta_opt_method,
            )

            beta_prev = beta_w.detach().clone()
        else:
            # Reuse previous beta; no LBFGS this step
            beta_w = beta_prev

        # Recompute f_pred through live K_stack so pde_loss trains path_ext
        z = K_stack @ beta_w           # uses non-detached K_stack
        u_tmp = ua + upa * (x - x[0]) + z[2]
        f_pred = z[0] + k1 * (upa + z[1]) + k2 * u_tmp + k3 * u_tmp**3

        pde_loss = forcing_loss(f, f_pred)

        # Total loss (PDE + shuffle)
        loss = ADAM_lambda_model * pde_loss + ADAM_lambda_shuffle * shuffle_loss
        loss.backward()
        opt.step()

        # Scheduler step
        if scheduler is not None:
            scheduler.step(loss.detach())

        # Logging and snapshots
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

        X_ext_sig_final = computesignatures(X_ext_final, depth)
        X_ext_sig_final_norm = apply_signature_normalization(
            X_ext_sig_final, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )



        Ksig_final = build_kernel_from_signatures(
            X_ext_sig_final_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )
        K_stack_final = build_kernel_operators_method1(Ksig_final, x)

        beta_w, u, f_pred_final = solve_betas_method1(
            K_stack=K_stack_final,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3,
            beta_init=None,
            max_iter=beta_iterations,
            method=beta_opt_method,
        )

        final_loss = forcing_loss(f, f_pred_final)
        print(f"Overall true final loss={final_loss:.3e}")

    return u, snapshots, f_pred_final, path_ext


def test_nonbranched_method1(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth, rbf_sigma,
    beta_opt_method: str = "l-bfgs",
    beta_iterations: int = 500,
    kernel_type: str = "rbf",
    norm_scheme: str = "none",
    norm_kwargs: dict | None = None,
    retrain_every: int = 10,
):
    if norm_kwargs is None:
        norm_kwargs = {}

    u_pred_full = []
    f_pred_full = []

    with torch.no_grad():
        # First train on training data
        X_train = torch.stack([x_train, f_train], dim=1)
        X_sig_train = computesignatures(X_train, depth)
        X_sig_norm_train = apply_signature_normalization(
            X_sig_train, depth=depth, dim=2,
            scheme=norm_scheme, **norm_kwargs
        )
        Ksig_train = build_kernel_from_signatures(
            X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type
        )
        K_stack_train = build_kernel_operators_method1(Ksig_train, x_train)

        beta_w, u_pred_train, f_pred_train = solve_betas_method1(
            K_stack=K_stack_train, f=f_train, x=x_train, ua=ua, upa=upa,
            k1=k1, k2=k2, k3=k3,
            beta_init=None,
            max_iter=beta_iterations,
            method=beta_opt_method,
        )
        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        N_test = x_test.numel()
        N_train = x_train.numel()

        for j in range(1, N_test + 1):

            # Retrain if at retraining step
            if (j % retrain_every) == 0:
                x_retrain = torch.cat([x_train, x_test[:j]], dim=0)
                f_retrain = torch.cat([f_train, f_test[:j]], dim=0)
                X_train = torch.stack([x_retrain, f_retrain], dim=1)
                X_sig_train = computesignatures(X_train, depth)
                X_sig_norm_train = apply_signature_normalization(
                    X_sig_train, depth=depth, dim=2,
                    scheme=norm_scheme, **norm_kwargs
                )
                
                Ksig_train = build_kernel_from_signatures(
                    X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type
                )

                # ---- zero-pad (or truncate) beta_w to match new N ----
                N_new = Ksig_train.shape[0]
                beta_old = beta_w.reshape(-1)
                beta_init = torch.zeros(
                    N_new, dtype=beta_old.dtype, device=beta_old.device
                )
                n_copy = min(N_new, beta_old.numel())
                beta_init[:n_copy] = beta_old[:n_copy]
                # ------------------------------------------------------

                K_stack_train = build_kernel_operators_method1(Ksig_train, x_retrain)

                beta_w, u_pred_train, f_pred_train = solve_betas_method1(
                    K_stack=K_stack_train, f=f_retrain, x=x_retrain, ua=ua, upa=upa,
                    k1=k1, k2=k2, k3=k3,
                    beta_init=beta_init,          # padded warm start
                    max_iter=beta_iterations,
                    method=beta_opt_method,
                )

            # Build current path [train | first j test points]
            x_curr = torch.cat([x_train, x_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)
            X_curr = torch.stack([x_curr, f_curr], dim=1)

            # Compute signatures and normalize using training stats
            X_sig_curr = computesignatures(X_curr, depth)
            X_sig_train_norm_pair, X_sig_curr_norm = apply_signature_normalization_pair(
                X_sig_train, X_sig_curr,
                depth=depth, dim=2,
                scheme=norm_scheme, **norm_kwargs
            )

            # Build cross-kernel and operators (use train side from pair)
            Ksig_curr_train = build_kernel_from_different_signatures(
                X_sig_curr_norm, X_sig_train_norm_pair,
                sigma=rbf_sigma, kernel_type=kernel_type
            )

            K_stack_curr = build_kernel_operators_method1(Ksig_curr_train, x_curr)

            # Evaluate on current grid
            u_curr, u_p_curr, u_dd_curr = evaluate_solution_from_beta_method1(
                K_stack_curr, x_curr, beta_w, ua, upa
            )



            f_curr_pred = evaluate_forcing_from_solution_method1(
                u_curr, u_p_curr, u_dd_curr, k1, k2, k3
            )

            # Append only the new test point
            u_pred_full = torch.cat(
                [u_pred_full, u_curr[N_train + j - 1 : N_train + j]], dim=0
            )
            f_pred_full = torch.cat(
                [f_pred_full, f_curr_pred[N_train + j - 1 : N_train + j]], dim=0
            )

    x_all = torch.cat([x_train, x_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, last beta): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full




def test_branched_NN_method1(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth, rbf_sigma,
    pathextension,
    beta_opt_method: str = "l-bfgs",
    beta_iterations: int = 500,
    kernel_type: str = "rbf",
    norm_scheme: str = "none",
    norm_kwargs: dict | None = None,
    retrain_every: int = 10,
):
    if norm_kwargs is None:
        norm_kwargs = {}

    u_pred_full = []
    f_pred_full = []

    with torch.no_grad():
        # First train on training data
        X_train = torch.stack([x_train, f_train], dim=1)
        X_branched_train = torch.cat([X_train, pathextension(X_train)], dim=1)
        X_sig_train = computesignatures(X_branched_train, depth)
        X_sig_norm_train = apply_signature_normalization(
            X_sig_train, depth=depth, dim=2,
            scheme=norm_scheme, **norm_kwargs
        )

        Ksig_train = build_kernel_from_signatures(
            X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type
        )
        K_stack_train = build_kernel_operators_method1(Ksig_train, x_train)

        beta_w, u_pred_train, f_pred_train = solve_betas_method1(
            K_stack=K_stack_train, f=f_train, x=x_train, ua=ua, upa=upa,
            k1=k1, k2=k2, k3=k3,
            beta_init=None,
            max_iter=beta_iterations,
            method=beta_opt_method,
        )



        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        N_test = x_test.numel()
        N_train = x_train.numel()

        for j in range(1, N_test + 1):

            # Retrain if at retraining step
            if (j % retrain_every) == 0:
                x_retrain = torch.cat([x_train, x_test[:j]], dim=0)
                f_retrain = torch.cat([f_train, f_test[:j]], dim=0)
                X_train = torch.stack([x_retrain, f_retrain], dim=1)
                X_branched_train = torch.cat([X_train, pathextension(X_train)], dim=1)
                X_sig_train = computesignatures(X_branched_train, depth)
                X_sig_norm_train = apply_signature_normalization(
                    X_sig_train, depth=depth, dim=2,
                    scheme=norm_scheme, **norm_kwargs
                )
                
                Ksig_train = build_kernel_from_signatures(
                    X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type
                )

                # ---- zero-pad (or truncate) beta_w to match new N ----
                N_new = Ksig_train.shape[0]
                beta_old = beta_w.reshape(-1)
                beta_init = torch.zeros(
                    N_new, dtype=beta_old.dtype, device=beta_old.device
                )
                n_copy = min(N_new, beta_old.numel())
                beta_init[:n_copy] = beta_old[:n_copy]
                # ------------------------------------------------------

                K_stack_train = build_kernel_operators_method1(Ksig_train, x_retrain)

                beta_w, u_pred_train, f_pred_train = solve_betas_method1(
                    K_stack=K_stack_train, f=f_retrain, x=x_retrain, ua=ua, upa=upa,
                    k1=k1, k2=k2, k3=k3,
                    beta_init=beta_init,        # padded warm start
                    max_iter=beta_iterations,
                    method=beta_opt_method,
                )

            # Build current path [train | first j test points]
            x_curr = torch.cat([x_train, x_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)
            X_curr = torch.stack([x_curr, f_curr], dim=1)
            X_curr_branched = torch.cat([X_curr, pathextension(X_curr)], dim=1)

            # Compute signatures and normalize using training stats
            X_sig_curr = computesignatures(X_curr_branched, depth)
            X_sig_train_norm_pair, X_sig_curr_norm = apply_signature_normalization_pair(
                X_sig_train, X_sig_curr,
                depth=depth, dim=2,
                scheme=norm_scheme, **norm_kwargs
            )

            # Build cross-kernel and operators (use train side from pair)
            Ksig_curr_train = build_kernel_from_different_signatures(
                X_sig_curr_norm, X_sig_train_norm_pair,
                sigma=rbf_sigma, kernel_type=kernel_type
            )

            K_stack_curr = build_kernel_operators_method1(Ksig_curr_train, x_curr)

            # Evaluate on current grid
            u_curr, u_p_curr, u_dd_curr = evaluate_solution_from_beta_method1(
                K_stack_curr, x_curr, beta_w, ua, upa
            )


            f_curr_pred = evaluate_forcing_from_solution_method1(
                u_curr, u_p_curr, u_dd_curr, k1, k2, k3
            )

            # Append only the new test point
            u_pred_full = torch.cat(
                [u_pred_full, u_curr[N_train + j - 1 : N_train + j]], dim=0
            )
            f_pred_full = torch.cat(
                [f_pred_full, f_curr_pred[N_train + j - 1 : N_train + j]], dim=0
            )

    x_all = torch.cat([x_train, x_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, last beta): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full


import matplotlib.pyplot as plt
import torch


# ── Helper ────────────────────────────────────────────────────────────────────
def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


def _add_split_vline(ax, x_full, N_train, color="gray", linestyle="--", linewidth=1.0):
    """Draw vertical line at the train/test split x = x_full[N_train-1]."""
    xn = _to_np(x_full)
    if N_train is None or N_train <= 0 or N_train >= len(xn):
        return
    x_split = xn[N_train - 1]
    ax.axvline(x_split, color=color, linestyle=linestyle, linewidth=linewidth)


# ── 1. Reference forcing and solution ────────────────────────────────────────
def plot_reference_forcing_and_solution(x, f_true, u_ref):
    xn = _to_np(x)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(xn, _to_np(f_true), color="black")
    axes[0].set_title("True forcing f(t)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("f(t)")
    axes[1].plot(xn, _to_np(u_ref), color="black")
    axes[1].set_title("Reference solution u(t)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("u(t)")
    plt.tight_layout(); plt.show()


# ── 2. 2×2 final comparison (generic) ───────────────────────────────────────
def plot_final_comparison_2x2_generic(
    x, target_true, u_ref,
    u_sig, target_sig,
    u_nb,  target_nb,
    target_name="forcing", title_prefix="",
):
    xn = _to_np(x)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(xn, _to_np(u_ref), "k-",  label="Reference")
    axes[0, 0].plot(xn, _to_np(u_nb),  "r--", label="Non-branched")
    axes[0, 0].set_title(f"{title_prefix}Non-branched u(t)")
    axes[0, 0].legend()

    axes[0, 1].plot(xn, _to_np(target_true), "k-",  label=f"True {target_name}")
    axes[0, 1].plot(xn, _to_np(target_nb),   "r--", label=f"Pred {target_name}")
    axes[0, 1].set_title(f"{title_prefix}Non-branched {target_name}")
    axes[0, 1].legend()

    axes[1, 0].plot(xn, _to_np(u_ref), "k-",  label="Reference")
    axes[1, 0].plot(xn, _to_np(u_sig), "b--", label="Branched NN")
    axes[1, 0].set_title(f"{title_prefix}Branched NN u(t)")
    axes[1, 0].legend()

    axes[1, 1].plot(xn, _to_np(target_true), "k-",  label=f"True {target_name}")
    axes[1, 1].plot(xn, _to_np(target_sig),  "b--", label=f"Pred {target_name}")
    axes[1, 1].set_title(f"{title_prefix}Branched NN {target_name}")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("t")
    plt.tight_layout(); plt.show()


# ── 3. Extension channels (branched only) ────────────────────────────────────
def plot_all_extensions(x, f_true, path_ext_m1):
    xn = _to_np(x)
    X = torch.stack([x, f_true], dim=1)
    with torch.no_grad():
        ext_m1 = _to_np(path_ext_m1(X))
    n_ext = ext_m1.shape[1]
    fig, axes = plt.subplots(1, n_ext, figsize=(5 * n_ext, 4))
    if n_ext == 1:
        axes = [axes]
    for i in range(n_ext):
        axes[i].plot(xn, ext_m1[:, i])
        axes[i].set_title(f"Branched ext channel {i+1}")
        axes[i].set_xlabel("t")
    plt.suptitle("Learned path extension channels", fontsize=13)
    plt.tight_layout(); plt.show()


# ── 4. Shuffle residual matrix (branched only) ───────────────────────────────
def plot_shuffle_residual_matrix(
    x, f_true, path_ext,
    title="Shuffle Product Loss (Residual Matrix)",
):
    path_ext.eval()
    with torch.no_grad():
        X = torch.stack([x, f_true], dim=1)
        X_bar = path_ext(X)
        R = shuffle_loss_residual(X_bar)
        R_np = _to_np(R)

    C = R_np.shape[0]
    v = abs(R_np).max() + 1e-12

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.imshow(
        R_np, cmap="bwr", vmin=-v, vmax=v,
        interpolation="nearest", aspect="equal"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        r"$R_{ij}=\Delta X^i\,\Delta X^j - \left(\int X^i\,dX^j + \int X^j\,dX^i\right)$",
        rotation=90, labelpad=12
    )
    ax.set_title(title)
    ax.set_xlabel("j"); ax.set_ylabel("i")
    positions = list(range(C))
    labels = [str(i + 1) for i in range(C)]
    ax.set_xticks(positions); ax.set_xticklabels(labels)
    ax.set_yticks(positions); ax.set_yticklabels(labels)
    ax.set_xlim(-0.5, C - 0.5); ax.set_ylim(C - 0.5, -0.5)
    plt.show()


# ── 5. Error helpers ─────────────────────────────────────────────────────────
def get_errors(u_pred, f_pred, u_ref, f_true):
    from torch import mean
    eps = 1e-12
    u_p, f_p = u_pred.detach().cpu(), f_pred.detach().cpu()
    u_r, f_t = u_ref.detach().cpu(),  f_true.detach().cpu()
    mse_u = mean((u_p - u_r) ** 2).item()
    mse_f = mean((f_p - f_t) ** 2).item()
    return {
        "mse_u": mse_u,
        "rel_u": mse_u / (mean(u_r ** 2).item() + eps),
        "mse_f": mse_f,
        "rel_f": mse_f / (mean(f_t ** 2).item() + eps),
    }


def print_variant_errors(label, u_pred, f_pred, u_ref, f_true, x):
    from torch import mean
    eps = 1e-12
    u_p, f_p = u_pred.detach().cpu(), f_pred.detach().cpu()
    u_r, f_t = u_ref.detach().cpu(),  f_true.detach().cpu()
    mse_u = mean((u_p - u_r) ** 2).item()
    mse_f = mean((f_p - f_t) ** 2).item()
    rel_u = mse_u / (mean(u_r ** 2).item() + eps)
    rel_f = mse_f / (mean(f_t ** 2).item() + eps)
    print(
        f"  [{label}]  MSE(u)={mse_u:.3e}  RelMSE(u)={rel_u:.3e}  "
        f"MSE(f)={mse_f:=.3e}  RelMSE(f)={rel_f:.3e}"
    )


def print_pct_improvement(label, errs, baseline):
    def pct(b, v): return 100.0 * (b - v) / (abs(b) + 1e-12)
    print(
        f"  [{label}]"
        f"  MSE(u)={pct(baseline['mse_u'], errs['mse_u']):+.1f}%"
        f"  RelMSE(u)={pct(baseline['rel_u'], errs['rel_u']):+.1f}%"
        f"  MSE(f)={pct(baseline['mse_f'], errs['mse_f']):+.1f}%"
        f"  RelMSE(f)={pct(baseline['rel_f'], errs['rel_f']):+.1f}%"
    )


# ── 6. Structured error printing (full and test) ─────────────────────────────
def print_train_test_errors(
    x_full, u_ref_full, f_true_full,
    u_nb_m1_full, f_pred_nb_m1_full,
    u_sig_m1_full, f_pred_sig_m1_full,
    label_prefix="FULL (train+test)",
):
    print(f"\n{'=' * 80}")
    print(f"{label_prefix}: errors vs reference")
    print(f"{'=' * 80}")
    print_variant_errors(
        f"{label_prefix} — Non-branched",
        u_pred=u_nb_m1_full,  f_pred=f_pred_nb_m1_full,
        u_ref=u_ref_full, f_true=f_true_full, x=x_full,
    )
    print_variant_errors(
        f"{label_prefix} — Branched NN",
        u_pred=u_sig_m1_full, f_pred=f_pred_sig_m1_full,
        u_ref=u_ref_full, f_true=f_true_full, x=x_full,
    )


def print_test_only_errors(
    x_test, u_ref_full, f_true_full,
    u_nb_m1_full, f_pred_nb_m1_full,
    u_sig_m1_full, f_pred_sig_m1_full,
    N_train,
    label_prefix="TEST ONLY",
):
    u_ref_test      = u_ref_full[N_train:]
    f_true_test     = f_true_full[N_train:]
    u_nb_test       = u_nb_m1_full[N_train:]
    f_pred_nb_test  = f_pred_nb_m1_full[N_train:]
    u_sig_test      = u_sig_m1_full[N_train:]
    f_pred_sig_test = f_pred_sig_m1_full[N_train:]

    print(f"\n{'=' * 80}")
    print(f"{label_prefix}: errors vs reference (TEST portion)")
    print(f"{'=' * 80}")
    print_variant_errors(
        f"{label_prefix} — Non-branched (test)",
        u_pred=u_nb_test,  f_pred=f_pred_nb_test,
        u_ref=u_ref_test, f_true=f_true_test, x=x_test,
    )
    print_variant_errors(
        f"{label_prefix} — Branched NN (test)",
        u_pred=u_sig_test, f_pred=f_pred_sig_test,
        u_ref=u_ref_test, f_true=f_true_test, x=x_test,
    )


# ── 7. Full train+test path comparison (with split line) ─────────────────────
def plot_all_variants_full_train_test(
    x_full, f_true_full, u_ref_full,
    u_nb_m1_full,  f_pred_nb_m1_full,
    u_sig_m1_full, f_pred_sig_m1_full,
    N_train=None,
):
    xn = _to_np(x_full)
    variants = [
        ("Non-branched", u_nb_m1_full,  f_pred_nb_m1_full),
        ("Branched NN",  u_sig_m1_full, f_pred_sig_m1_full),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (name, u, fp) in enumerate(variants):
        axes[row, 0].plot(xn, _to_np(u_ref_full), "k-", lw=1.2, label="Reference")
        axes[row, 0].plot(xn, _to_np(u),          "r--", lw=1,   label="Predicted")
        axes[row, 0].set_title(f"{name} — u(t) [full]")
        axes[row, 0].legend(fontsize=7)
        _add_split_vline(axes[row, 0], x_full, N_train)

        axes[row, 1].plot(xn, _to_np(f_true_full), "k-", lw=1.2, label="True f")
        axes[row, 1].plot(xn, _to_np(fp),          "r--", lw=1,   label="Pred f")
        axes[row, 1].set_title(f"{name} — f(t) [full]")
        axes[row, 1].legend(fontsize=7)
        _add_split_vline(axes[row, 1], x_full, N_train)

    for ax in axes.flat:
        ax.set_xlabel("t")
    plt.suptitle("All variants — full train+test path", fontsize=14)
    plt.tight_layout(); plt.show()


# ── 8. Test portion only comparison ──────────────────────────────────────────
def plot_all_variants_test_only(
    x_test, u_ref_full, f_true_full,
    u_nb_m1_full,  f_pred_nb_m1_full,
    u_sig_m1_full, f_pred_sig_m1_full,
    N_train,
):
    xn = _to_np(x_test)

    def sl(t): return _to_np(t)[N_train:]

    variants = [
        ("Non-branched", sl(u_nb_m1_full),  sl(f_pred_nb_m1_full)),
        ("Branched NN",  sl(u_sig_m1_full), sl(f_pred_sig_m1_full)),
    ]
    u_ref_test  = sl(u_ref_full)
    f_true_test = sl(f_true_full)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (name, u, fp) in enumerate(variants):
        axes[row, 0].plot(xn, u_ref_test, "k-", lw=1.2, label="Reference")
        axes[row, 0].plot(xn, u,          "r--", lw=1,   label="Predicted")
        axes[row, 0].set_title(f"{name} — u(t) [test]")
        axes[row, 0].legend(fontsize=7)

        axes[row, 1].plot(xn, f_true_test, "k-", lw=1.2, label="True f")
        axes[row, 1].plot(xn, fp,          "r--", lw=1,   label="Pred f")
        axes[row, 1].set_title(f"{name} — f(t) [test]")
        axes[row, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel("t")
    plt.suptitle("All variants — test portion only", fontsize=14)
    plt.tight_layout(); plt.show()


def plot_learned_extensions(x, f_true, X_ext, title_prefix="Learned extensions"):
    """
    X_ext is the full extended path (T, 2+E) or just extensions (T, E).
    Plots forcing on the first row, each extension on its own row below.
    """
    t = x.detach().cpu().numpy()
    f_np = f_true.detach().cpu().numpy()

    if X_ext.shape[1] > 2:
        ext_np = X_ext[:, 2:].detach().cpu().numpy()
    else:
        ext_np = X_ext.detach().cpu().numpy()

    E = ext_np.shape[1]
    nrows = 1 + E

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
    ax0.set_ylabel("f(t)")
    ax0.set_title(f"{title_prefix}: forcing")

    # Rows 1..E: each extension
    for i in range(E):
        ax = axes[i + 1]
        ax.plot(t, ext_np[:, i], 'b-')
        ax.set_ylabel(f"ext {i+1}")

    axes[-1].set_xlabel("t")
    fig.tight_layout()
    plt.show()


# ── 9. Calibration: 2×2 non-branched vs branched ─────────────────────────────
def plot_calibration_2x2(
    x_train, f_train, u_ref_train,
    u_nb, f_pred_nb,
    u_sig, f_pred_sig,
    title_prefix="Calibration — ",
):
    """
    2×2 plot for the training (calibration) split.
    Row 0: Non-branched  |  Row 1: Branched NN
    Col 0: solution u(t) vs reference
    Col 1: forcing f(t) vs true
    """
    xn = _to_np(x_train)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Row 0: Non-branched
    axes[0, 0].plot(xn, _to_np(u_ref_train), "k-",  label="Reference")
    axes[0, 0].plot(xn, _to_np(u_nb),        "r--", label="Non-branched")
    axes[0, 0].set_title(f"{title_prefix}Non-branched — u(t)")
    axes[0, 0].legend()

    axes[0, 1].plot(xn, _to_np(f_train),   "k-",  label="True f")
    axes[0, 1].plot(xn, _to_np(f_pred_nb), "r--", label="Pred f")
    axes[0, 1].set_title(f"{title_prefix}Non-branched — f(t)")
    axes[0, 1].legend()

    # Row 1: Branched NN
    axes[1, 0].plot(xn, _to_np(u_ref_train), "k-",  label="Reference")
    axes[1, 0].plot(xn, _to_np(u_sig),       "b--", label="Branched NN")
    axes[1, 0].set_title(f"{title_prefix}Branched NN — u(t)")
    axes[1, 0].legend()

    axes[1, 1].plot(xn, _to_np(f_train),    "k-",  label="True f")
    axes[1, 1].plot(xn, _to_np(f_pred_sig), "b--", label="Pred f")
    axes[1, 1].set_title(f"{title_prefix}Branched NN — f(t)")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("t")
    plt.suptitle("Calibration (training split)", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── 10. Calibration: 1×2 branched only ───────────────────────────────────────
def plot_calibration_branched_1x2(
    x_train, f_train, u_ref_train,
    u_sig, f_pred_sig,
    title_prefix="Calibration — ",
):
    """
    1×2 plot for the training (calibration) split, branched model only.
    Col 0: solution u(t) vs reference
    Col 1: forcing f(t) vs true
    """
    xn = _to_np(x_train)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(xn, _to_np(u_ref_train), "k-",  label="Reference")
    axes[0].plot(xn, _to_np(u_sig),       "b--", label="Branched NN")
    axes[0].set_title(f"{title_prefix}Branched NN — u(t)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("u(t)")
    axes[0].legend()

    axes[1].plot(xn, _to_np(f_train),    "k-",  label="True f")
    axes[1].plot(xn, _to_np(f_pred_sig), "b--", label="Pred f")
    axes[1].set_title(f"{title_prefix}Branched NN — f(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("f(t)")
    axes[1].legend()

    plt.suptitle("Calibration — Branched NN (training split)", fontsize=13)
    plt.tight_layout()
    plt.show()


# ── 11. 2×2 forcing-only, training (left col) vs testing (right col) ─────────
def plot_forcing_train_test_2x2(
    x_full, f_true_full,
    f_pred_nb_m1_full, f_pred_sig_m1_full,
    N_train,
    title_prefix="Forcing — ",
):
    """
    2×2 forcing-only plot.
    Rows: Non-branched, Branched
    Col 0: training forcing vs true
    Col 1: testing forcing vs true
    """
    xn_full = _to_np(x_full)
    x_train = xn_full[:N_train]
    x_test  = xn_full[N_train:]

    f_true_train = _to_np(f_true_full[:N_train])
    f_true_test  = _to_np(f_true_full[N_train:])
    f_nb_train   = _to_np(f_pred_nb_m1_full[:N_train])
    f_nb_test    = _to_np(f_pred_nb_m1_full[N_train:])
    f_sig_train  = _to_np(f_pred_sig_m1_full[:N_train])
    f_sig_test   = _to_np(f_pred_sig_m1_full[N_train:])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Row 0: Non-branched
    axes[0, 0].plot(x_train, f_true_train, "k-",  label="True f (train)")
    axes[0, 0].plot(x_train, f_nb_train,   "r--", label="Pred f (train)")
    axes[0, 0].set_title(f"{title_prefix}Non-branched — train")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].plot(x_test, f_true_test, "k-",  label="True f (test)")
    axes[0, 1].plot(x_test, f_nb_test,   "r--", label="Pred f (test)")
    axes[0, 1].set_title(f"{title_prefix}Non-branched — test")
    axes[0, 1].legend(fontsize=7)

    # Row 1: Branched
    axes[1, 0].plot(x_train, f_true_train, "k-",  label="True f (train)")
    axes[1, 0].plot(x_train, f_sig_train,  "b--", label="Pred f (train)")
    axes[1, 0].set_title(f"{title_prefix}Branched NN — train")
    axes[1, 0].legend(fontsize=7)

    axes[1, 1].plot(x_test, f_true_test, "k-",  label="True f (test)")
    axes[1, 1].plot(x_test, f_sig_test,  "b--", label="Pred f (test)")
    axes[1, 1].set_title(f"{title_prefix}Branched NN — test")
    axes[1, 1].legend(fontsize=7)

    for ax in axes.flat:
        ax.set_xlabel("t")
    plt.suptitle("Forcing — train vs test, non-branched vs branched", fontsize=14)
    plt.tight_layout()
    plt.show()


# ── 12. 1×2 full path (forcing + solution) for branched only ─────────────────
def plot_branched_full_1x2(
    x_full, u_ref_full, f_true_full,
    u_sig_m1_full, f_pred_sig_m1_full,
    N_train,
    title_prefix="Branched — full path",
):
    """
    1×2 plot: full train+test for branched model only, with vertical split line.
    Left: u(t) vs reference. Right: f(t) vs true.
    """
    xn = _to_np(x_full)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(xn, _to_np(u_ref_full),    "k-",  label="Reference")
    axes[0].plot(xn, _to_np(u_sig_m1_full), "b--", label="Branched NN")
    axes[0].set_title(f"{title_prefix} — u(t)")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("u(t)")
    _add_split_vline(axes[0], x_full, N_train)
    axes[0].legend()

    axes[1].plot(xn, _to_np(f_true_full),        "k-",  label="True f")
    axes[1].plot(xn, _to_np(f_pred_sig_m1_full), "b--", label="Pred f")
    axes[1].set_title(f"{title_prefix} — f(t)")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("f(t)")
    _add_split_vline(axes[1], x_full, N_train)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ── 13. 1×2 forcing-only, non-branched (left) vs branched (right) — TRAIN ────
def plot_forcing_train_1x2(
    x_full, f_true_full,
    f_pred_nb_m1_full, f_pred_sig_m1_full,
    N_train,
    title_prefix="Forcing — train",
):
    xn = _to_np(x_full)
    x_train = xn[:N_train]

    f_true_train = _to_np(f_true_full[:N_train])
    f_nb_train   = _to_np(f_pred_nb_m1_full[:N_train])
    f_sig_train  = _to_np(f_pred_sig_m1_full[:N_train])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: non-branched
    axes[0].plot(x_train, f_true_train, "k-",  label="True f (train)")
    axes[0].plot(x_train, f_nb_train,   "r--", label="Pred f (train)")
    axes[0].set_title(f"{title_prefix} — Non-branched")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("f(t)")
    axes[0].legend()

    # Right: branched
    axes[1].plot(x_train, f_true_train, "k-",  label="True f (train)")
    axes[1].plot(x_train, f_sig_train,  "b--", label="Pred f (train)")
    axes[1].set_title(f"{title_prefix} — Branched NN")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("f(t)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ── 14. 1×2 forcing-only, non-branched (left) vs branched (right) — TEST ─────
def plot_forcing_test_1x2(
    x_full, f_true_full,
    f_pred_nb_m1_full, f_pred_sig_m1_full,
    N_train,
    title_prefix="Forcing — test",
):
    xn = _to_np(x_full)
    x_test = xn[N_train:]

    f_true_test = _to_np(f_true_full[N_train:])
    f_nb_test   = _to_np(f_pred_nb_m1_full[N_train:])
    f_sig_test  = _to_np(f_pred_sig_m1_full[N_train:])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: non-branched
    axes[0].plot(x_test, f_true_test, "k-",  label="True f (test)")
    axes[0].plot(x_test, f_nb_test,   "r--", label="Pred f (test)")
    axes[0].set_title(f"{title_prefix} — Non-branched")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("f(t)")
    axes[0].legend()

    # Right: branched
    axes[1].plot(x_test, f_true_test, "k-",  label="True f (test)")
    axes[1].plot(x_test, f_sig_test,  "b--", label="Pred f (test)")
    axes[1].set_title(f"{title_prefix} — Branched NN")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("f(t)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
