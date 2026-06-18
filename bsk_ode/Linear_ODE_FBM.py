import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import random
import time, math, random
import numpy as np
import torchmin
from torch import nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from torchmin import least_squares, minimize
from torch import cumulative_trapezoid
import keras_sig
from keras_sig import SigLayer


# Cell 3 - seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from .stochastic.processes.continuous import FractionalBrownianMotion

def f_forcing_fbm(x: torch.Tensor, hurst: float = 0.2) -> torch.Tensor:
    """Fractional Brownian motion on [a,b] using stochastic (Davies–Harte)."""
    N = x.numel()
    length = float(x[-1]) - float(x[0])

    rng = np.random.default_rng(SEED)
    fbm_gen = FractionalBrownianMotion(hurst=hurst, t=length, rng=rng)

    # sample returns N+1 points: B[0]=0, B[1]...B[N]
    fbm_sample = fbm_gen.sample(n=N)
    B = torch.tensor(fbm_sample, dtype=x.dtype, device=x.device)

    # return the full path (N+1 points) trimmed to N to match x
    return B[:N]

def solve_linear_ivp(x_grid: torch.Tensor,
                  forcing_torch: torch.Tensor,
                  a: float, b: float,
                  ya: float, ypa: float,
                  k1: float, k2: float, k3: float):
  """
  Reference Solver for the Differential Equation using Scipy
  Solve  k1*u'' + k2*u' + k*u = f(x)
  rewritten as  u'' = (f - k2*u' - k*u) / k1
  with u(interval_a)=ya, u'(interval_a)=ypa.
  """
  t_eval = x_grid.cpu().numpy()
  f_np   = forcing_torch.cpu().numpy()

  def forcing(t):
      return np.interp(t, t_eval, f_np)


  def fun(t, y):
      u, up = y
      f_val = forcing(t)
      du_dt  = up
      dup_dt = (f_val - k2 * up - k3 * u) / k1
      return [du_dt, dup_dt]

  y0 = [ya, ypa]
  sol = solve_ivp(fun, (a, b), y0, t_eval=t_eval,
                  method="BDF", rtol=1e-6, atol=1e-9, max_step=0.05)
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

def compute_signatures(path: torch.Tensor,
                       depth: int) -> torch.Tensor:
    """
    path: (T, d) or (1, T, d) torch tensor.
    Returns: (T, D) prefix-signature features as torch.Tensor,
             on the same device as the input.
    """

    if path.dim() == 2:
        path = path.unsqueeze(0)          


    # Prepend basepoint
    basepoint = path[:, 0:1, :]
    path_bp   = torch.cat([basepoint, path], dim=1)  

    #Compute Signature
    sigs_raw = keras_sig.signature(
        path_bp,
        depth=depth,
        stream=True,
        gpu_optimized=True
    )                                           

    return sigs_raw.squeeze(0).to(device=device, dtype=torch.float64)

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
    

def build_kernel_operators_method1(Ksig: torch.Tensor,
                           x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized construction of Ku2, Kup, Ku from Ksig.
    """
    dtype = Ksig.dtype
    device = Ksig.device
    x = x.to(device=device, dtype=dtype)

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


def solve_betas_method1(Ksig: torch.Tensor,
                f: torch.Tensor,
                x: torch.Tensor,
                ua: float,
                upa: float,
                k1: float,
                k2: float,
                k3: float,
                reg: float = 1e-10,
                **kwargs):

    # 1) Fix dtype/device from Ksig
    dtype = torch.float64
    device = Ksig.device

    Ksig = Ksig.to(device=device, dtype=dtype)
    x    = x.to(device=device, dtype=dtype)
    f    = f.to(device=device, dtype=dtype)

    # 2) Build Ku2, Kup, Ku
    Ku2, Kup, Ku = build_kernel_operators_method1(Ksig, x)
    N  = Ksig.shape[0]
    x0 = x[0]

    # 3) Build A and rhs
    A   = (k1 * Ku2 + k2 * Kup + k3 * Ku).to(device=device, dtype=dtype)
    rhs = (f - k2 * upa - k3 * (ua + upa * (x - x0))).to(device=device, dtype=dtype)

    # 4) Tikhonov solve in
    I     = torch.eye(N, dtype=dtype, device=device)
    ATA   = A.T @ A + reg * I
    ATrhs = A.T @ rhs
    beta  = torch.linalg.solve(ATA, ATrhs).to(device=device, dtype=dtype)

    # 5) Reconstruct
    u      = ua + upa * (x - x0) + Ku  @ beta
    u_p    = upa + Kup @ beta
    u_dd   = Ku2 @ beta
    f_pred = k1 * u_dd + k2 * u_p + k3 * u

    return beta, u, f_pred


def buildkerneloperators_method2(Ksig: torch.Tensor, x: torch.Tensor):
    dtype = Ksig.dtype
    device = Ksig.device
    x = x.to(device=device, dtype=dtype)

    K0 = Ksig.clone()  # K
    I1 = cumulative_trapezoid(K0, x, dim=0)
    I1 = torch.vstack([torch.zeros_like(I1[:1]), I1])  # I K
    I2 = cumulative_trapezoid(I1, x, dim=0)
    I2 = torch.vstack([torch.zeros_like(I2[:1]), I2])  # I^2 K
    return K0, I1, I2


def rhs_method2(f: torch.Tensor, x: torch.Tensor,
                ua: float, upa: float,
                k1: float, k2: float):
    x0 = x[0]
    q = k1 * ua + (k1 * upa + k2 * ua) * (x - x0)
    return double_integrate(f, x) + q

def solvebetasmethod2(Ksig: torch.Tensor,
                      f: torch.Tensor,
                      x: torch.Tensor,
                      ua: float,
                      upa: float,
                      k1: float,
                      k2: float,
                      k3: float,
                      reg: float = 1e-10):
    dtype = torch.float64
    device = Ksig.device

    Ksig = Ksig.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)
    f = f.to(device=device, dtype=dtype)

    K0, IK, I2K = buildkerneloperators_method2(Ksig, x)

    A = k1 * K0 + k2 * IK + k3 * I2K
    rhs = rhs_method2(f, x, ua, upa, k1, k2).to(device=device, dtype=dtype)

    N = Ksig.shape[0]
    Ireg = torch.eye(N, dtype=dtype, device=device)
    beta = torch.linalg.solve(A.T @ A + reg * Ireg, A.T @ rhs)

    u = K0 @ beta
    Iu = IK @ beta
    I2u = I2K @ beta
    rhs_pred = k1 * u + k2 * Iu + k3 * I2u

    return beta, u, rhs_pred, rhs

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

def evaluate_solution_from_beta_method1(
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
    Compute predicted forcing f_pred = k1*u'' + k2*u' + k3*u.
    """
    f_pred = k1 * u_dd + k2 * u_p + k3 * u
    return f_pred


def evaluate_solution_from_beta_method2(
    K0: torch.Tensor,
    IK: torch.Tensor,
    I2K: torch.Tensor,
    x: torch.Tensor,
    beta: torch.Tensor,
    ua: float,
    upa: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Given kernel operators K0, IK, I2K and coefficients beta,
    reconstruct u, Iu, I^2 u on the grid x for method 2.
    """

    u   = K0  @ beta   # u      ~ K0  beta  (same as in solvebetasmethod2)
    Iu  = IK  @ beta   # ∫u     ~ IK  beta
    I2u = I2K @ beta   # ∫∫u    ~ I2K beta

    # Maintain the output order expected by the test code:
    return u, Iu, I2u


def evaluate_forcing_from_solution_method2(
    u: torch.Tensor,      # really u
    Iu: torch.Tensor,     # ∫u
    I2u: torch.Tensor,    # ∫∫u
    k1: float,
    k2: float,
    k3: float,
) -> torch.Tensor:
    """
    Method 2 integrated target:
        rhs_pred(x) = k1 * u(x) + k2 * ∫u + k3 * ∫∫u.
    """
    return k1 * u + k2 * Iu + k3 * I2u



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

def tlift(X, holder_value):
    """
    Time-lift a path by appending x^(2H) as an extra channel.

    Parameters
    ----------
    X : torch.Tensor, shape (T, d)
        Path whose first column is assumed to be the time/grid variable x.
    holder_value : float
        Hölder exponent H.

    Returns
    -------
    X_tlift : torch.Tensor, shape (T, d+1)
        Original path with appended channel x^(2H).
    """
    if X.ndim != 2:
        raise ValueError("X must have shape (T, d)")

    x = X[:, 0]
    H = holder_value

    x_lift = torch.pow(x, 2.0 * H).unsqueeze(1)
    X_tlift = torch.cat([X, x_lift], dim=1)
    return X_tlift


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

def init_path_extension_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)



# === Non‑branched (no extension) signature‑kernel solver ===
def solve_signature_kernel_non_branched_method1(x, f,
                                        k1, k2, k3,
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


        X_sig_norm = apply_signature_normalization(X_sig, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)


        Ksig = build_kernel_from_signatures(X_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type)


        beta_w, u, f_pred_final = solve_betas_method1(
            Ksig=Ksig,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3,
        )


        final_loss = forcing_loss(f, f_pred_final)

        print(f"method 1, non-branched model forcing match loss: {final_loss.item():.3e}")


    return u, f_pred_final

# === Non‑branched (no extension) signature‑kernel solver ===
def solve_signature_kernel_tlift_method1(x, f,
                                        k1, k2, k3,
                                        ua, upa,
                                        depth, rbf_sigma,
                                        holder_value,
                                        kernel_type: str = "rbf",
                                        norm_scheme: str = "none",
                                        norm_kwargs: dict | None = None):

    if norm_kwargs is None:
        norm_kwargs = {}

    with torch.no_grad():

        X = torch.stack([x, f], dim=1)           # (T,2)

        X_tlift = tlift(X, holder_value)

        X_sig = compute_signatures(X_tlift, depth)     # (T,D)


        X_sig_norm = apply_signature_normalization(X_sig, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)


        Ksig = build_kernel_from_signatures(X_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type)


        beta_w, u, f_pred_final = solve_betas_method1(
            Ksig=Ksig,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3,
        )


        final_loss = forcing_loss(f, f_pred_final)

        print(f"method 1, t-lift model forcing match loss: {final_loss.item():.3e}")


    return u, f_pred_final

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

    path_ext.apply(init_path_extension_weights)

    path_ext = torch.compile(path_ext)


    # Optimizer (Adam) over both extension net and beta_w
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


    # ---------- Adam phase (jointly train extension + beta_w) ----------
    for it in range(1, adam_iters + 1):
        opt.zero_grad()

        # Compute extensions and shuffle loss
        out_ext = path_ext(X)
        shuffle_loss = shuffle_loss_function(out_ext)

        # Build extended path
        stack = torch.cat([X.unsqueeze(0), out_ext.unsqueeze(0)], dim=2)
        X_ext = stack.squeeze(0)

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


        beta_w, u_tmp, f_pred = solve_betas_method1(
            Ksig=Ksig,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3
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

        stack_final = torch.cat(
            [X_final.unsqueeze(0), out_ext_final.unsqueeze(0)], dim=2
        )
        X_ext_final = stack_final.squeeze(0)

        X_ext_sig_final = compute_signatures(X_ext_final, depth)
        X_ext_sig_final_norm = apply_signature_normalization(
            X_ext_sig_final, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )

        Ksig_final = build_kernel_from_signatures(
            X_ext_sig_final_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )

        beta_w,u,f_pred_final = solve_betas_method1(
            Ksig=Ksig_final,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3
        )

        final_loss = forcing_loss(f, f_pred_final)
        print(f"Overall true final loss={final_loss:.3e}")

    return u, snapshots, f_pred_final, path_ext

# === Non-branched (no extension) signature-kernel solver: method 2 ===
def solve_signature_kernel_non_branched_method2(x, f,
                                                k1, k2, k3,
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

        beta_w, u, f_pred_final, rhs_true = solvebetasmethod2(
            Ksig=Ksig,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3,
        )

        final_loss = forcing_loss(rhs_true, f_pred_final)
        print(f"method 2, non-branched integrated-target loss: {final_loss.item():.3e}")

    return u, f_pred_final

# === Non-branched (no extension) signature-kernel solver: method 2 ===
def solve_signature_kernel_tlift_method2(x, f,
                                                k1, k2, k3,
                                                ua, upa,
                                                depth, rbf_sigma,
                                                holder_value,
                                                kernel_type: str = "rbf",
                                                norm_scheme: str = "none",
                                                norm_kwargs: dict | None = None):

    if norm_kwargs is None:
        norm_kwargs = {}

    with torch.no_grad():

        X = torch.stack([x, f], dim=1)           # (T,2)

        X_tlift = tlift(X, holder_value)

        X_sig = compute_signatures(X_tlift, depth)     # (T,D)

        X_sig_norm = apply_signature_normalization(
            X_sig, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs
        )

        Ksig = build_kernel_from_signatures(
            X_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )

        beta_w, u, f_pred_final, rhs_true = solvebetasmethod2(
            Ksig=Ksig,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3,
        )

        final_loss = forcing_loss(rhs_true, f_pred_final)
        print(f"method 2, t-lift integrated-target loss: {final_loss.item():.3e}")

    return u, f_pred_final

# === Branched (extended-path) signature-kernel solver: method 2 ===
def solve_signature_kernel_branched_method2(x, f,
                                            k1, k2, k3,
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

    path_ext = PathExtension(
        input_dim=2,
        output_dim=extensions,
        hidden_dims=hidden_dims,
        activation_cls=activation_cls
    ).to(device)

    path_ext.apply(init_path_extension_weights)

    path_ext = torch.compile(path_ext)

    snapshots = []
    opt = torch.optim.Adam(
        list(path_ext.parameters()),
        lr=adam_lr
    )

    scheduler = None
    if adam_use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=adam_sched_factor,
            patience=adam_sched_patience,
        )

    X = torch.stack([x, f], dim=1)

    num_snaps_adam = snapshot_count
    adam_snapshot_epochs = sorted(set(
        int(i) for i in torch.linspace(1, adam_iters, num_snaps_adam)
    ))
    if 1 in adam_snapshot_epochs:
        adam_snapshot_epochs.remove(1)

    # ---------- Adam phase ----------
    for it in range(1, adam_iters + 1):
        opt.zero_grad()

        out_ext = path_ext(X)
        shuffle_loss = shuffle_loss_function(out_ext)

        stack = torch.cat([X.unsqueeze(0), out_ext.unsqueeze(0)], dim=2)
        X_ext = stack.squeeze(0)

        X_ext_sig = compute_signatures(X_ext, depth)

        X_ext_sig_norm = apply_signature_normalization(
            X_ext_sig, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )

        Ksig = build_kernel_from_signatures(
            X_ext_sig_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )

        beta_w, u_tmp, f_pred, rhs_true = solvebetasmethod2(
            Ksig=Ksig,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3
        )

        pde_loss = forcing_loss(rhs_true, f_pred)

        loss = ADAM_lambda_model * pde_loss + ADAM_lambda_shuffle * shuffle_loss
        loss.backward()
        opt.step()

        if scheduler is not None:
            scheduler.step(loss.detach())

        if it % 50 == 0:
            print(
                f"[Adam {it:04d}] loss={loss.item():.3e}, "
                f"PDE={pde_loss.item():.3e}, "
                f"shuffle={shuffle_loss.item():.3e}"
            )

        if it in adam_snapshot_epochs:
            snapshots.append({
                "phase": "Adam",
                "iter": it,
                "u": u_tmp.detach().clone(),
                "f_pred": f_pred.detach().clone(),   # keep same key for compatibility
            })

    # ---------- Final evaluation ----------
    with torch.no_grad():
        X_final = torch.stack([x, f], dim=1)
        out_ext_final = path_ext(X_final)

        stack_final = torch.cat(
            [X_final.unsqueeze(0), out_ext_final.unsqueeze(0)], dim=2
        )
        X_ext_final = stack_final.squeeze(0)

        X_ext_sig_final = compute_signatures(X_ext_final, depth)
        X_ext_sig_final_norm = apply_signature_normalization(
            X_ext_sig_final, depth=depth, dim=2 + extensions,
            scheme=norm_scheme, **norm_kwargs
        )

        Ksig_final = build_kernel_from_signatures(
            X_ext_sig_final_norm, sigma=rbf_sigma, kernel_type=kernel_type
        )

        beta_w, u, f_pred_final, rhs_true_final = solvebetasmethod2(
            Ksig=Ksig_final,
            f=f,
            x=x,
            ua=ua,
            upa=upa,
            k1=k1,
            k2=k2,
            k3=k3
        )

        final_loss = forcing_loss(rhs_true_final, f_pred_final)
        print(f"Overall true final method 2 loss={final_loss.item():.3e}")

    return u, snapshots, f_pred_final, path_ext


def test_nonbranched_method1(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth, rbf_sigma,
    kernel_type: str = "rbf",
    norm_scheme: str = "none",
    norm_kwargs: dict | None = None,
    retrain_every: int = 10,
):
    """
    Non-branched testing with periodic retraining.
    """
    if norm_kwargs is None:
        norm_kwargs = {}

    u_pred_full = []
    f_pred_full = []

    with torch.no_grad():
        #First train on training data and add to upred, fpred, full
        X_train = torch.stack([x_train, f_train], dim=1)  # (T,2)
        X_sig_train = compute_signatures(X_train, depth)  # (T,D)
        X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
        Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
        beta_w, u_pred_train, f_pred_train = solve_betas_method1(Ksig=Ksig_train, f=f_train, x=x_train, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)
        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        # Loop over test points; at step j we evaluate on [train | first j test]
        N_test = x_test.numel()
        N_train = x_train.numel()
        for j in range(1, N_test + 1):

            #Retrain if at retraining step
            if (j % retrain_every) == 0:
                x_retrain = torch.cat([x_train, x_test[:j]], dim=0)
                f_retrain = torch.cat([f_train, f_test[:j]], dim=0)
                X_train = torch.stack([x_retrain, f_retrain], dim=1)  # (T,2)
                X_sig_train = compute_signatures(X_train, depth)  # (T,D)
                X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
                Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
                beta_w, u_pred_train, f_pred_train = solve_betas_method1(Ksig=Ksig_train, f=f_retrain, x=x_retrain, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)

            #Build new training path by adding new points onto current training path
            x_curr = torch.cat([x_train, x_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)
            X_curr = torch.stack([x_curr, f_curr], dim=1)

            #Compute Signatures of new path and normalize
            X_sig_curr = compute_signatures(X_curr, depth)
            _, X_sig_curr_norm = apply_signature_normalization_pair(X_sig_train, X_sig_curr, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)

            #Build Kernels and Operatprs
            Ksig_curr_train = build_kernel_from_different_signatures(X_sig_curr_norm, X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
            Ku2_curr, Kup_curr, Ku_curr = build_kernel_operators_method1(Ksig_curr_train, x_curr)

            # Evaluate on current grid
            u_curr, u_p_curr, u_dd_curr = evaluate_solution_from_beta_method1(Ku2_curr, Kup_curr, Ku_curr, x_curr, beta_w, ua, upa)
            f_curr_pred = evaluate_forcing_from_solution_method1(u_curr, u_p_curr, u_dd_curr, k1, k2, k3)

            # append ONLY the new test part onto the end of the tensors
            u_pred_full = torch.cat([u_pred_full, u_curr[N_train + j - 1:N_train + j]], dim=0)
            f_pred_full = torch.cat([f_pred_full, f_curr_pred[N_train + j - 1:N_train + j]], dim=0)

    #Testing done, print accuracy of forcing
    x_all = torch.cat([x_train, x_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, last beta): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full

def test_branched_tlift_method1(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth, rbf_sigma,
    holder_value,
    kernel_type: str = "rbf",
    norm_scheme: str = "none",
    norm_kwargs: dict | None = None,
    retrain_every: int = 10,
):
    """
    Non-branched testing with periodic retraining.
    """
    if norm_kwargs is None:
        norm_kwargs = {}

    u_pred_full = []
    f_pred_full = []

    with torch.no_grad():
        #First train on training data and add to upred, fpred, full
        X_train = torch.stack([x_train, f_train], dim=1)  # (T,2)
        X_tlift_train = tlift(X_train, holder_value)
        X_sig_train = compute_signatures(X_tlift_train, depth)  # (T,D)
        X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
        Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
        beta_w, u_pred_train, f_pred_train = solve_betas_method1(Ksig=Ksig_train, f=f_train, x=x_train, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)
        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        # Loop over test points; at step j we evaluate on [train | first j test]
        N_test = x_test.numel()
        N_train = x_train.numel()
        for j in range(1, N_test + 1):

            #Retrain if at retraining step
            if (j % retrain_every) == 0:
                x_retrain = torch.cat([x_train, x_test[:j]], dim=0)
                f_retrain = torch.cat([f_train, f_test[:j]], dim=0)
                X_train = torch.stack([x_retrain, f_retrain], dim=1)  # (T,2)
                X_tlift_train = tlift(X_train, holder_value)
                X_sig_train = compute_signatures(X_tlift_train, depth)  # (T,D)
                X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
                Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
                beta_w, u_pred_train, f_pred_train = solve_betas_method1(Ksig=Ksig_train, f=f_retrain, x=x_retrain, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)

            #Build new training path by adding new points onto current training path
            x_curr = torch.cat([x_train, x_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)
            X_curr = torch.stack([x_curr, f_curr], dim=1)
            X_curr_tlift = tlift(X_curr, holder_value)

            #Compute Signatures of new path and normalize
            X_sig_curr = compute_signatures(X_curr_tlift, depth)
            _, X_sig_curr_norm = apply_signature_normalization_pair(X_sig_train, X_sig_curr, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)

            #Build Kernels and Operatprs
            Ksig_curr_train = build_kernel_from_different_signatures(X_sig_curr_norm, X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
            Ku2_curr, Kup_curr, Ku_curr = build_kernel_operators_method1(Ksig_curr_train, x_curr)

            # Evaluate on current grid
            u_curr, u_p_curr, u_dd_curr = evaluate_solution_from_beta_method1(Ku2_curr, Kup_curr, Ku_curr, x_curr, beta_w, ua, upa)
            f_curr_pred = evaluate_forcing_from_solution_method1(u_curr, u_p_curr, u_dd_curr, k1, k2, k3)

            # append ONLY the new test part onto the end of the tensors
            u_pred_full = torch.cat([u_pred_full, u_curr[N_train + j - 1:N_train + j]], dim=0)
            f_pred_full = torch.cat([f_pred_full, f_curr_pred[N_train + j - 1:N_train + j]], dim=0)

    #Testing done, print accuracy of forcing
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
    kernel_type: str = "rbf",
    norm_scheme: str = "none",
    norm_kwargs: dict | None = None,
    retrain_every: int = 10,
):
    """
    Non-branched testing with periodic retraining.
    """
    if norm_kwargs is None:
        norm_kwargs = {}

    u_pred_full = []
    f_pred_full = []

    with torch.no_grad():
        #First train on training data and add to upred, fpred, full
        X_train = torch.stack([x_train, f_train], dim=1)  # (T,2)                
        out_ext_train = pathextension(X_train)
        X_branched_train = torch.cat([X_train, out_ext_train], dim=1)
        X_sig_train = compute_signatures(X_branched_train, depth)  # (T,D)
        X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
        Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
        beta_w, u_pred_train, f_pred_train = solve_betas_method1(Ksig=Ksig_train, f=f_train, x=x_train, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)
        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        # Loop over test points; at step j we evaluate on [train | first j test]
        N_test = x_test.numel()
        N_train = x_train.numel()
        for j in range(1, N_test + 1):

            #Retrain if at retraining step
            if (j % retrain_every) == 0:
                x_retrain = torch.cat([x_train, x_test[:j]], dim=0)
                f_retrain = torch.cat([f_train, f_test[:j]], dim=0)
                X_train = torch.stack([x_retrain, f_retrain], dim=1)  # (T,2)
                out_ext_train = pathextension(X_train)
                X_branched_train = torch.cat([X_train, out_ext_train], dim=1)
                X_sig_train = compute_signatures(X_branched_train, depth)  # (T,D)
                X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
                Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
                beta_w, u_pred_train, f_pred_train = solve_betas_method1(Ksig=Ksig_train, f=f_retrain, x=x_retrain, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)

            #Build new training path by adding new points onto current training path
            x_curr = torch.cat([x_train, x_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)
            X_curr = torch.stack([x_curr, f_curr], dim=1)
            out_ext_curr = pathextension(X_curr)
            X_curr_branched = torch.cat([X_curr, out_ext_curr], dim=1)

            #Compute Signatures of new path and normalize
            X_sig_curr = compute_signatures(X_curr_branched, depth)
            _, X_sig_curr_norm = apply_signature_normalization_pair(X_sig_train, X_sig_curr, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)

            #Build Kernels and Operatprs
            Ksig_curr_train = build_kernel_from_different_signatures(X_sig_curr_norm, X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
            Ku2_curr, Kup_curr, Ku_curr = build_kernel_operators_method1(Ksig_curr_train, x_curr)

            # Evaluate on current grid
            u_curr, u_p_curr, u_dd_curr = evaluate_solution_from_beta_method1(Ku2_curr, Kup_curr, Ku_curr, x_curr, beta_w, ua, upa)
            f_curr_pred = evaluate_forcing_from_solution_method1(u_curr, u_p_curr, u_dd_curr, k1, k2, k3)

            # append ONLY the new test part onto the end of the tensors
            u_pred_full = torch.cat([u_pred_full, u_curr[N_train + j - 1:N_train + j]], dim=0)
            f_pred_full = torch.cat([f_pred_full, f_curr_pred[N_train + j - 1:N_train + j]], dim=0)

    #Testing done, print accuracy of forcing
    x_all = torch.cat([x_train, x_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, last beta): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full

def test_nonbranched_method2(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth, rbf_sigma,
    kernel_type: str = "rbf",
    norm_scheme: str = "none",
    norm_kwargs: dict | None = None,
    retrain_every: int = 10,
):
    """
    Non-branched testing with periodic retraining.
    """
    if norm_kwargs is None:
        norm_kwargs = {}

    u_pred_full = []
    f_pred_full = []

    with torch.no_grad():
        #First train on training data and add to upred, fpred, full
        X_train = torch.stack([x_train, f_train], dim=1)  # (T,2)
        X_sig_train = compute_signatures(X_train, depth)  # (T,D)
        X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
        Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
        beta_w, u_pred_train, f_pred_train, _ = solvebetasmethod2(Ksig=Ksig_train, f=f_train, x=x_train, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)
        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        # Loop over test points; at step j we evaluate on [train | first j test]
        N_test = x_test.numel()
        N_train = x_train.numel()
        for j in range(1, N_test + 1):

            #Retrain if at retraining step
            if (j % retrain_every) == 0:
                x_retrain = torch.cat([x_train, x_test[:j]], dim=0)
                f_retrain = torch.cat([f_train, f_test[:j]], dim=0)
                X_train = torch.stack([x_retrain, f_retrain], dim=1)  # (T,2)
                X_sig_train = compute_signatures(X_train, depth)  # (T,D)
                X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
                Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
                beta_w, u_pred_train, f_pred_train, _ = solvebetasmethod2(Ksig=Ksig_train, f=f_retrain, x=x_retrain, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)

            #Build new training path by adding new points onto current training path
            x_curr = torch.cat([x_train, x_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)
            X_curr = torch.stack([x_curr, f_curr], dim=1)

            #Compute Signatures of new path and normalize
            X_sig_curr = compute_signatures(X_curr, depth)
            _, X_sig_curr_norm = apply_signature_normalization_pair(X_sig_train, X_sig_curr, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)

            #Build Kernels and Operatprs
            Ksig_curr_train = build_kernel_from_different_signatures(X_sig_curr_norm, X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
            Ku2_curr, Kup_curr, Ku_curr = buildkerneloperators_method2(Ksig_curr_train, x_curr)

            # Evaluate on current grid
            u_curr, u_p_curr, u_dd_curr = evaluate_solution_from_beta_method2(Ku2_curr, Kup_curr, Ku_curr, x_curr, beta_w, ua, upa)
            f_curr_pred = evaluate_forcing_from_solution_method2(u_curr, u_p_curr, u_dd_curr, k1, k2, k3)

            # append ONLY the new test part onto the end of the tensors
            u_pred_full = torch.cat([u_pred_full, u_curr[N_train + j - 1:N_train + j]], dim=0)
            f_pred_full = torch.cat([f_pred_full, f_curr_pred[N_train + j - 1:N_train + j]], dim=0)

    #Testing done, print accuracy of forcing
    x_all = torch.cat([x_train, x_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, last beta): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full

def test_branched_tlift_method2(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth, rbf_sigma,
    holder_value,
    kernel_type: str = "rbf",
    norm_scheme: str = "none",
    norm_kwargs: dict | None = None,
    retrain_every: int = 10,
):
    """
    Non-branched testing with periodic retraining.
    """
    if norm_kwargs is None:
        norm_kwargs = {}

    u_pred_full = []
    f_pred_full = []

    with torch.no_grad():
        #First train on training data and add to upred, fpred, full
        X_train = torch.stack([x_train, f_train], dim=1)  # (T,2)
        X_tlift_train = tlift(X_train, holder_value)
        X_sig_train = compute_signatures(X_tlift_train, depth)  # (T,D)
        X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
        Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
        beta_w, u_pred_train, f_pred_train, _ = solvebetasmethod2(Ksig=Ksig_train, f=f_train, x=x_train, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)
        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        # Loop over test points; at step j we evaluate on [train | first j test]
        N_test = x_test.numel()
        N_train = x_train.numel()
        for j in range(1, N_test + 1):

            #Retrain if at retraining step
            if (j % retrain_every) == 0:
                x_retrain = torch.cat([x_train, x_test[:j]], dim=0)
                f_retrain = torch.cat([f_train, f_test[:j]], dim=0)
                X_train = torch.stack([x_retrain, f_retrain], dim=1)  # (T,2)
                X_tlift_train = tlift(X_train, holder_value)
                X_sig_train = compute_signatures(X_tlift_train, depth)  # (T,D)
                X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
                Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
                beta_w, u_pred_train, f_pred_train, _ = solvebetasmethod2(Ksig=Ksig_train, f=f_retrain, x=x_retrain, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)

            #Build new training path by adding new points onto current training path
            x_curr = torch.cat([x_train, x_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)
            X_curr = torch.stack([x_curr, f_curr], dim=1)
            X_curr_tlift = tlift(X_curr, holder_value)

            #Compute Signatures of new path and normalize
            X_sig_curr = compute_signatures(X_curr_tlift, depth)
            _, X_sig_curr_norm = apply_signature_normalization_pair(X_sig_train, X_sig_curr, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)

            #Build Kernels and Operatprs
            Ksig_curr_train = build_kernel_from_different_signatures(X_sig_curr_norm, X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
            Ku2_curr, Kup_curr, Ku_curr = buildkerneloperators_method2(Ksig_curr_train, x_curr)

            # Evaluate on current grid
            u_curr, u_p_curr, u_dd_curr = evaluate_solution_from_beta_method2(Ku2_curr, Kup_curr, Ku_curr, x_curr, beta_w, ua, upa)
            f_curr_pred = evaluate_forcing_from_solution_method2(u_curr, u_p_curr, u_dd_curr, k1, k2, k3)

            # append ONLY the new test part onto the end of the tensors
            u_pred_full = torch.cat([u_pred_full, u_curr[N_train + j - 1:N_train + j]], dim=0)
            f_pred_full = torch.cat([f_pred_full, f_curr_pred[N_train + j - 1:N_train + j]], dim=0)

    #Testing done, print accuracy of forcing
    x_all = torch.cat([x_train, x_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, last beta): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full


def test_branched_NN_method2(
    x_train: torch.Tensor,
    x_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth, rbf_sigma,
    pathextension,
    kernel_type: str = "rbf",
    norm_scheme: str = "none",
    norm_kwargs: dict | None = None,
    retrain_every: int = 10,
):
    """
    Non-branched testing with periodic retraining.
    """
    if norm_kwargs is None:
        norm_kwargs = {}

    u_pred_full = []
    f_pred_full = []

    with torch.no_grad():
        #First train on training data and add to upred, fpred, full
        X_train = torch.stack([x_train, f_train], dim=1)  # (T,2)
        out_ext_train = pathextension(X_train)
        X_branched_train = torch.cat([X_train, out_ext_train], dim=1)
        X_sig_train = compute_signatures(X_branched_train, depth)  # (T,D)
        X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
        Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
        beta_w, u_pred_train, f_pred_train, _ = solvebetasmethod2(Ksig=Ksig_train, f=f_train, x=x_train, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)
        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        # Loop over test points; at step j we evaluate on [train | first j test]
        N_test = x_test.numel()
        N_train = x_train.numel()
        for j in range(1, N_test + 1):

            #Retrain if at retraining step
            if (j % retrain_every) == 0:
                x_retrain = torch.cat([x_train, x_test[:j]], dim=0)
                f_retrain = torch.cat([f_train, f_test[:j]], dim=0)
                X_train = torch.stack([x_retrain, f_retrain], dim=1)  # (T,2)
                out_ext_train = pathextension(X_train)
                X_branched_train = torch.cat([X_train, out_ext_train], dim=1)
                X_sig_train = compute_signatures(X_branched_train, depth)  # (T,D)
                X_sig_norm_train = apply_signature_normalization(X_sig_train, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)
                Ksig_train = build_kernel_from_signatures(X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
                beta_w, u_pred_train, f_pred_train, _ = solvebetasmethod2(Ksig=Ksig_train, f=f_retrain, x=x_retrain, ua=ua, upa=upa, k1=k1, k2=k2, k3=k3)

            #Build new training path by adding new points onto current training path
            x_curr = torch.cat([x_train, x_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)
            X_curr = torch.stack([x_curr, f_curr], dim=1)
            out_ext_curr = pathextension(X_curr)
            X_curr_branched = torch.cat([X_curr, out_ext_curr], dim=1)

            #Compute Signatures of new path and normalize
            X_sig_curr = compute_signatures(X_curr_branched, depth)
            _, X_sig_curr_norm = apply_signature_normalization_pair(X_sig_train, X_sig_curr, depth=depth, dim=2, scheme=norm_scheme, **norm_kwargs)

            #Build Kernels and Operatprs
            Ksig_curr_train = build_kernel_from_different_signatures(X_sig_curr_norm, X_sig_norm_train, sigma=rbf_sigma, kernel_type=kernel_type)
            Ku2_curr, Kup_curr, Ku_curr = buildkerneloperators_method2(Ksig_curr_train, x_curr)

            # Evaluate on current grid
            u_curr, u_p_curr, u_dd_curr = evaluate_solution_from_beta_method2(Ku2_curr, Kup_curr, Ku_curr, x_curr, beta_w, ua, upa)
            f_curr_pred = evaluate_forcing_from_solution_method2(u_curr, u_p_curr, u_dd_curr, k1, k2, k3)

            # append ONLY the new test part onto the end of the tensors
            u_pred_full = torch.cat([u_pred_full, u_curr[N_train + j - 1:N_train + j]], dim=0)
            f_pred_full = torch.cat([f_pred_full, f_curr_pred[N_train + j - 1:N_train + j]], dim=0)

    #Testing done, print accuracy of forcing
    x_all = torch.cat([x_train, x_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, last beta): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full

def plot_final_comparison_2x2_generic(x,
                                      target_true,
                                      u_ref,
                                      u_sig, target_sig,
                                      u_nb, target_nb,
                                      target_name="forcing",
                                      title_prefix=""):
    """
    2x2 figure:

      [0,0]: target (branched vs true)
      [0,1]: target (non-branched vs true)
      [1,0]: solution (branched vs reference)
      [1,1]: solution (non-branched vs reference)
    """
    t = x.detach().cpu().numpy()
    target_true_np = target_true.detach().cpu().numpy()
    u_ref_np = u_ref.detach().cpu().numpy()
    u_sig_np = u_sig.detach().cpu().numpy()
    target_sig_np = target_sig.detach().cpu().numpy()
    u_nb_np = u_nb.detach().cpu().numpy()
    target_nb_np = target_nb.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex='col')

    # target: branched vs true
    ax = axes[0, 0]
    ax.plot(t, target_true_np, 'k-', label=f"true {target_name}")
    ax.plot(t, target_sig_np, 'r--', label=f"branched pred {target_name}")
    ax.set_title(f"{title_prefix}{target_name.capitalize()}: branched vs true")
    ax.set_ylabel(target_name)
    ax.legend()

    # target: non-branched vs true
    ax = axes[0, 1]
    ax.plot(t, target_true_np, 'k-', label=f"true {target_name}")
    ax.plot(t, target_nb_np, 'b--', label=f"non-branched pred {target_name}")
    ax.set_title(f"{title_prefix}{target_name.capitalize()}: non-branched vs true")
    ax.legend()

    # solution: branched vs reference
    ax = axes[1, 0]
    ax.plot(t, u_ref_np, 'k-', label="reference u(t)")
    ax.plot(t, u_sig_np, 'r--', label="branched pred u")
    ax.set_title(f"{title_prefix}Solution: branched vs reference")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.legend()

    # solution: non-branched vs reference
    ax = axes[1, 1]
    ax.plot(t, u_ref_np, 'k-', label="reference u(t)")
    ax.plot(t, u_nb_np, 'b--', label="non-branched pred u")
    ax.set_title(f"{title_prefix}Solution: non-branched vs reference")
    ax.set_xlabel("t")
    ax.legend()

    plt.tight_layout()
    plt.show()


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


def plot_reference_forcing_and_solution(x, f_true, u_ref):
    """
    Two plots on one row:
      left  - forcing f_true(t)
      right - reference solution u_ref(t)
    """
    t = x.detach().cpu().numpy()
    f_np = f_true.detach().cpu().numpy()
    u_np = u_ref.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    # Forcing
    axes[0].plot(t, f_np, 'k-')
    axes[0].set_title("Forcing f(t)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("f(t)")

    # Reference solution
    axes[1].plot(t, u_np, 'k-')
    axes[1].set_title("Reference solution u(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("u(t)")

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


def print_pct_improvement(label, errs, baseline):
    """Print % improvement of errs vs baseline (positive = better, negative = worse)."""
    def pct(b, v): return 100.0 * (b - v) / (abs(b) + 1e-12)
    print(f"  [{label}]"
          f"  MSE(u)={pct(baseline['mse_u'], errs['mse_u']):+.1f}%"
          f"  Rel(u)={pct(baseline['rel_u'], errs['rel_u']):+.1f}%"
          f"  MSE(f)={pct(baseline['mse_f'], errs['mse_f']):+.1f}%"
          f"  Rel(f)={pct(baseline['rel_f'], errs['rel_f']):+.1f}%")

# ── Helper: print error table for a single variant ──
def print_variant_errors(label, u_pred, f_pred, u_ref, f_true, x):
    from torch import mean, sqrt
    eps = 1e-12
    u_p, f_p = u_pred.detach().cpu(), f_pred.detach().cpu()
    u_r, f_t = u_ref.detach().cpu(),  f_true.detach().cpu()
    mse_u = mean((u_p - u_r)**2).item()
    rel_u = (sqrt(mean((u_p - u_r)**2)) / (sqrt(mean(u_r**2)) + eps)).item()
    mse_f = mean((f_p - f_t)**2).item()
    rel_f = (sqrt(mean((f_p - f_t)**2)) / (sqrt(mean(f_t**2)) + eps)).item()
    print(f"  [{label}]  MSE(u)={mse_u:.3e}  Rel(u)={rel_u:.3e}  MSE(f)={mse_f:.3e}  Rel(f)={rel_f:.3e}")


def plot_all_extensions(
        x, f_true,
        path_ext_m1, path_ext_m2,
):
    """
    Plots learned branched extensions only.
    Rows: one per extension channel
    Cols: [0] Method 1 branched  [1] Method 2 branched
    """
    t      = x.detach().cpu().numpy()
    X_base = torch.stack([x, f_true], dim=1)

    with torch.no_grad():
        ext_m1 = path_ext_m1(X_base).detach().cpu().numpy()  # (T, n_ext)
        ext_m2 = path_ext_m2(X_base).detach().cpu().numpy()

    n_ext = ext_m1.shape[1]

    fig, axes = plt.subplots(n_ext, 2, figsize=(12, 4 * n_ext), sharex=True)
    if n_ext == 1:
        axes = axes[None, :]  # ensure 2D indexing
    fig.suptitle("Branched: Learned Path Extensions", fontsize=14, fontweight='bold')

    axes[0, 0].set_title("Method 1 — Learned Extensions", fontsize=11, fontweight='bold')
    axes[0, 1].set_title("Method 2 — Learned Extensions", fontsize=11, fontweight='bold')

    for ch in range(n_ext):
        for j, (ext, label) in enumerate([(ext_m1, "M1"), (ext_m2, "M2")]):
            ax = axes[ch, j]
            ax.plot(t, ext[:, ch], lw=1.5)
            ax.set_ylabel(f"Extension {ch}", fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)

    for j in range(2):
        axes[-1, j].set_xlabel("t")

    plt.tight_layout()
    plt.show()


def plot_all_shuffle_matrices(
        x, f_true,
        path_ext_m1, path_ext_m2,
):
    """
    1 row x 2 cols: shuffle residual matrix for Method 1 and Method 2 branched models.
    """
    X_base = torch.stack([x, f_true], dim=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Shuffle Product Residual Matrices", fontsize=14, fontweight='bold')

    for j, (pext, label) in enumerate([(path_ext_m1, "Method 1 — Branched"),
                                        (path_ext_m2, "Method 2 — Branched")]):
        with torch.no_grad():
            out = pext(X_base)
            R = shuffle_loss_residual(out).detach().cpu().numpy()

        ax = axes[j]
        im = ax.imshow(R, aspect='auto', cmap='RdBu',
                       vmin=-abs(R).max(), vmax=abs(R).max())
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel("extension channel j")
        ax.set_ylabel("extension channel i")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def print_train_test_errors(
    x_full: torch.Tensor,
    u_ref_full: torch.Tensor,
    f_true_full: torch.Tensor,
    u_nb_m1_full: torch.Tensor,
    f_pred_nb_m1_full: torch.Tensor,
    u_tl_m1_full: torch.Tensor,
    f_pred_tl_m1_full: torch.Tensor,
    u_sig_m1_full: torch.Tensor,
    f_pred_sig_m1_full: torch.Tensor,
    label_prefix: str = "FULL (train+test)"
):
    """
    Print error table (MSE/Rel for u and f) for all Method‑1 variants
    on the FULL path (train+test).
    Uses print_variant_errors for consistent formatting.
    """
    print(f"\n{'=' * 80}")
    print(f"{label_prefix}: Method 1 errors vs reference")
    print(f"{'=' * 80}")

    print_variant_errors(
        f"{label_prefix} — Non‑branched (M1)",
        u_pred=u_nb_m1_full, f_pred=f_pred_nb_m1_full,
        u_ref=u_ref_full, f_true=f_true_full, x=x_full,
    )
    print_variant_errors(
        f"{label_prefix} — t‑lift (M1)",
        u_pred=u_tl_m1_full, f_pred=f_pred_tl_m1_full,
        u_ref=u_ref_full, f_true=f_true_full, x=x_full,
    )
    print_variant_errors(
        f"{label_prefix} — Branched NN (M1)",
        u_pred=u_sig_m1_full, f_pred=f_pred_sig_m1_full,
        u_ref=u_ref_full, f_true=f_true_full, x=x_full,
    )


def print_test_only_errors(
    x_test: torch.Tensor,
    u_ref_full: torch.Tensor,
    f_true_full: torch.Tensor,
    u_nb_m1_full: torch.Tensor,
    f_pred_nb_m1_full: torch.Tensor,
    u_tl_m1_full: torch.Tensor,
    f_pred_tl_m1_full: torch.Tensor,
    u_sig_m1_full: torch.Tensor,
    f_pred_sig_m1_full: torch.Tensor,
    N_train: int,
    label_prefix: str = "TEST ONLY"
):
    """
    Print error table for all Method‑1 variants on the TEST portion only.
    We slice all tensors from N_train: to align with x_test.
    """
    u_ref_test = u_ref_full[N_train:]
    f_true_test = f_true_full[N_train:]

    u_nb_m1_test = u_nb_m1_full[N_train:]
    f_pred_nb_m1_test = f_pred_nb_m1_full[N_train:]

    u_tl_m1_test = u_tl_m1_full[N_train:]
    f_pred_tl_m1_test = f_pred_tl_m1_full[N_train:]

    u_sig_m1_test = u_sig_m1_full[N_train:]
    f_pred_sig_m1_test = f_pred_sig_m1_full[N_train:]

    print(f"\n{'=' * 80}")
    print(f"{label_prefix}: Method 1 errors vs reference (TEST portion)")
    print(f"{'=' * 80}")

    print_variant_errors(
        f"{label_prefix} — Non‑branched (M1, test)",
        u_pred=u_nb_m1_test, f_pred=f_pred_nb_m1_test,
        u_ref=u_ref_test, f_true=f_true_test, x=x_test,
    )
    print_variant_errors(
        f"{label_prefix} — t‑lift (M1, test)",
        u_pred=u_tl_m1_test, f_pred=f_pred_tl_m1_test,
        u_ref=u_ref_test, f_true=f_true_test, x=x_test,
    )
    print_variant_errors(
        f"{label_prefix} — Branched NN (M1, test)",
        u_pred=u_sig_m1_test, f_pred=f_pred_sig_m1_test,
        u_ref=u_ref_test, f_true=f_true_test, x=x_test,
    )


def print_errors_against_target(label,
                                u_ref, u_pred,
                                target_true, target_pred,
                                target_name="forcing",
                                ref_rel_u=None, ref_rel_target=None):
    eps = 1e-12

    u_pred_cpu      = u_pred.detach().cpu()
    target_pred_cpu = target_pred.detach().cpu()
    u_ref_cpu       = u_ref.detach().cpu()
    target_true_cpu = target_true.detach().cpu()

    # solution errors
    mse_u = torch.mean((u_pred_cpu - u_ref_cpu) ** 2).item()
    rel_u = mse_u / (torch.mean(u_ref_cpu ** 2).item() + eps)           # Rel MSE

    # target errors
    mse_t = torch.mean((target_pred_cpu - target_true_cpu) ** 2).item()
    rel_t = mse_t / (torch.mean(target_true_cpu ** 2).item() + eps)     # Rel MSE

    print(f"\n[{label}] solution errors vs IVP:")
    print(f"  MSE(u)        = {mse_u:.3e}")
    print(f"  Rel MSE(u)    = {rel_u:.3e}")

    if ref_rel_u is not None:
        imp_u = (ref_rel_u - rel_u) / (ref_rel_u + eps)
        print(f"  Rel improv vs non-branched model = {100 * imp_u:.2f}%")

    print(f"[{label}] {target_name} errors vs true {target_name}:")
    print(f"  MSE({target_name})        = {mse_t:.3e}")
    print(f"  Rel MSE({target_name})    = {rel_t:.3e}")

    if ref_rel_target is not None:
        imp_t = (ref_rel_target - rel_t) / (ref_rel_target + eps)
        print(f"  Rel improv vs non-branched model = {100 * imp_t:.2f}%")

    return {
        "mse_u":      mse_u,
        "rel_u":      rel_u,
        "mse_target": mse_t,
        "rel_target": rel_t,
    }


def compare_branched_nonbranched_method1_vs_method2(
    x, f_true, u_ref,
    u_nb_m1, f_pred_nb_m1,
    u_sig_m1, f_pred_sig_m1,
    u_nb_m2, f_pred_nb_m2,
    u_sig_m2, f_pred_sig_m2,
    ua, upa, k1, k2,
    plot_results=True
):
    eps = 1e-12

    def pct_improvement(err_m1, err_m2):
        return 100.0 * (err_m1 - err_m2) / (err_m1 + eps)

    def print_solution_side_by_side(case_label, m1_stats, m2_stats):
        imp_mse = pct_improvement(m1_stats["mse_u"], m2_stats["mse_u"])
        imp_rel = pct_improvement(m1_stats["rel_u"], m2_stats["rel_u"])

        print("\n" + "-" * 80)
        print(f"{case_label}: METHOD 1 vs METHOD 2 solution-fit comparison")
        print("-" * 80)
        print(f"{'Metric':<16}{'Method 1':>16}{'Method 2':>16}{'% improv M2 vs M1':>22}")
        print(f"{'MSE(u)':<16}{m1_stats['mse_u']:>16.3e}{m2_stats['mse_u']:>16.3e}{imp_mse:>22.2f}")
        print(f"{'Rel MSE(u)':<16}{m1_stats['rel_u']:>16.3e}{m2_stats['rel_u']:>16.3e}{imp_rel:>22.2f}")

    rhs_true_m2 = rhs_method2(f=f_true, x=x, ua=ua, upa=upa, k1=k1, k2=k2)

    print("\n" + "=" * 80)
    print("METHOD 1: differential / forcing fit")
    print("=" * 80)

    stats_nb_m1 = print_errors_against_target(
        label="Method 1 Non-branched",
        u_ref=u_ref, u_pred=u_nb_m1,
        target_true=f_true, target_pred=f_pred_nb_m1,
        target_name="forcing"
    )
    stats_sig_m1 = print_errors_against_target(
        label="Method 1 Branched",
        u_ref=u_ref, u_pred=u_sig_m1,
        target_true=f_true, target_pred=f_pred_sig_m1,
        target_name="forcing",
        ref_rel_u=stats_nb_m1["rel_u"],
        ref_rel_target=stats_nb_m1["rel_target"]
    )

    print("\n" + "=" * 80)
    print("METHOD 2: integrated / solution fit")
    print("=" * 80)

    stats_nb_m2 = print_errors_against_target(
        label="Method 2 Non-branched",
        u_ref=u_ref, u_pred=u_nb_m2,
        target_true=rhs_true_m2, target_pred=f_pred_nb_m2,
        target_name="integrated target"
    )
    stats_sig_m2 = print_errors_against_target(
        label="Method 2 Branched",
        u_ref=u_ref, u_pred=u_sig_m2,
        target_true=rhs_true_m2, target_pred=f_pred_sig_m2,
        target_name="integrated target",
        ref_rel_u=stats_nb_m2["rel_u"],
        ref_rel_target=stats_nb_m2["rel_target"]
    )

    print("\n" + "=" * 80)
    print("SIDE-BY-SIDE SOLUTION FITS: METHOD 1 vs METHOD 2")
    print("=" * 80)

    print_solution_side_by_side("Non-branched", stats_nb_m1, stats_nb_m2)
    print_solution_side_by_side("Branched",     stats_sig_m1, stats_sig_m2)

    imp_nb_rel  = pct_improvement(stats_nb_m1["rel_u"],  stats_nb_m2["rel_u"])
    imp_sig_rel = pct_improvement(stats_sig_m1["rel_u"], stats_sig_m2["rel_u"])

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Method 1 solution Rel MSE: non-branched={stats_nb_m1['rel_u']:.3e}, branched={stats_sig_m1['rel_u']:.3e}")
    print(f"Method 1 forcing  Rel MSE: non-branched={stats_nb_m1['rel_target']:.3e}, branched={stats_sig_m1['rel_target']:.3e}")
    print(f"Method 2 solution Rel MSE: non-branched={stats_nb_m2['rel_u']:.3e}, branched={stats_sig_m2['rel_u']:.3e}")
    print(f"Method 2 target   Rel MSE: non-branched={stats_nb_m2['rel_target']:.3e}, branched={stats_sig_m2['rel_target']:.3e}")

    print("\nMethod 2 vs Method 1 solution Rel MSE improvement:")
    print(f"  Non-branched = {imp_nb_rel:.2f}%")
    print(f"  Branched     = {imp_sig_rel:.2f}%")

    if plot_results:
        plot_final_comparison_2x2_generic(
            x=x, target_true=f_true, u_ref=u_ref,
            u_sig=u_sig_m1, target_sig=f_pred_sig_m1,
            u_nb=u_nb_m1, target_nb=f_pred_nb_m1,
            target_name="forcing", title_prefix="Method 1: "
        )
        plot_final_comparison_2x2_generic(
            x=x, target_true=rhs_true_m2, u_ref=u_ref,
            u_sig=u_sig_m2, target_sig=f_pred_sig_m2,
            u_nb=u_nb_m2, target_nb=f_pred_nb_m2,
            target_name="integrated target", title_prefix="Method 2: "
        )

    return {
        "method1": {"non_branched": stats_nb_m1, "branched": stats_sig_m1},
        "method2": {"non_branched": stats_nb_m2, "branched": stats_sig_m2},
        "method2_vs_method1_solution_improvement_percent": {
            "non_branched": {
                "mse_u": pct_improvement(stats_nb_m1["mse_u"], stats_nb_m2["mse_u"]),
                "rel_u": pct_improvement(stats_nb_m1["rel_u"], stats_nb_m2["rel_u"]),
            },
            "branched": {
                "mse_u": pct_improvement(stats_sig_m1["mse_u"], stats_sig_m2["mse_u"]),
                "rel_u": pct_improvement(stats_sig_m1["rel_u"], stats_sig_m2["rel_u"]),
            }
        }
    }


def get_errors(u_pred, f_pred, u_ref, f_true):
    from torch import mean
    eps = 1e-12
    u_p, f_p = u_pred.detach().cpu(), f_pred.detach().cpu()
    u_r, f_t = u_ref.detach().cpu(),  f_true.detach().cpu()
    mse_u = mean((u_p - u_r)**2).item()
    mse_f = mean((f_p - f_t)**2).item()
    return {
        "mse_u": mse_u,
        "rel_u": mse_u / (mean(u_r**2).item() + eps),    # Rel MSE
        "mse_f": mse_f,
        "rel_f": mse_f / (mean(f_t**2).item() + eps),    # Rel MSE
    }


def print_pct_improvement(label, errs, baseline):
    def pct(b, v): return 100.0 * (b - v) / (abs(b) + 1e-12)
    print(f"  [{label}]"
          f"  MSE(u)={pct(baseline['mse_u'], errs['mse_u']):+.1f}%"
          f"  RelMSE(u)={pct(baseline['rel_u'], errs['rel_u']):+.1f}%"
          f"  MSE(f)={pct(baseline['mse_f'], errs['mse_f']):+.1f}%"
          f"  RelMSE(f)={pct(baseline['rel_f'], errs['rel_f']):+.1f}%")


def print_variant_errors(label, u_pred, f_pred, u_ref, f_true, x):
    from torch import mean
    eps = 1e-12
    u_p, f_p = u_pred.detach().cpu(), f_pred.detach().cpu()
    u_r, f_t = u_ref.detach().cpu(),  f_true.detach().cpu()
    mse_u = mean((u_p - u_r)**2).item()
    mse_f = mean((f_p - f_t)**2).item()
    rel_u = mse_u / (mean(u_r**2).item() + eps)
    rel_f = mse_f / (mean(f_t**2).item() + eps)
    print(f"  [{label}]  MSE(u)={mse_u:.3e}  RelMSE(u)={rel_u:.3e}  MSE(f)={mse_f:.3e}  RelMSE(f)={rel_f:.3e}")



# ── Helper ────────────────────────────────────────────────────────────────────
def _to_np(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return t


# ── 1. Reference forcing and solution ─────────────────────────────────────────
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


# ── 2. All variants × both methods (training domain) ─────────────────────────
def plot_all_variants_comparison(
    x, f_true, u_ref,
    f_pred_nb_m1, u_nb_m1,
    f_pred_tl_m1, u_tl_m1,
    f_pred_sig_m1, u_sig_m1,
    f_pred_nb_m2, u_nb_m2,
    f_pred_tl_m2, u_tl_m2,
    f_pred_sig_m2, u_sig_m2,
    f_target_nb_m2, f_target_tl_m2, f_target_sig_m2,
):
    xn = _to_np(x)
    variants = [
        ("Non-branched", u_nb_m1, f_pred_nb_m1, u_nb_m2, f_pred_nb_m2, f_target_nb_m2),
        ("T-lift",       u_tl_m1, f_pred_tl_m1, u_tl_m2, f_pred_tl_m2, f_target_tl_m2),
        ("Branched NN",  u_sig_m1, f_pred_sig_m1, u_sig_m2, f_pred_sig_m2, f_target_sig_m2),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for row, (name, u_m1, fp_m1, u_m2, fp_m2, ft_m2) in enumerate(variants):
        # M1 solution
        axes[row,0].plot(xn, _to_np(u_ref),  "k-",  label="Reference")
        axes[row,0].plot(xn, _to_np(u_m1),   "r--", label="Predicted")
        axes[row,0].set_title(f"M1 {name} — u(t)")
        axes[row,0].legend(fontsize=7)
        # M1 forcing
        axes[row,1].plot(xn, _to_np(f_true), "k-",  label="True f")
        axes[row,1].plot(xn, _to_np(fp_m1),  "r--", label="Pred f")
        axes[row,1].set_title(f"M1 {name} — f(t)")
        axes[row,1].legend(fontsize=7)
        # M2 solution
        axes[row,2].plot(xn, _to_np(u_ref),  "k-",  label="Reference")
        axes[row,2].plot(xn, _to_np(u_m2),   "b--", label="Predicted")
        axes[row,2].set_title(f"M2 {name} — u(t)")
        axes[row,2].legend(fontsize=7)
        # M2 integrated target
        axes[row,3].plot(xn, _to_np(ft_m2),  "k-",  label="True target")
        axes[row,3].plot(xn, _to_np(fp_m2),  "b--", label="Pred target")
        axes[row,3].set_title(f"M2 {name} — integrated target")
        axes[row,3].legend(fontsize=7)
    for ax in axes.flat:
        ax.set_xlabel("t")
    plt.suptitle("All variants — training domain", fontsize=14)
    plt.tight_layout(); plt.show()


# ── 3. Extension channels ──────────────────────────────────────────────────────
def plot_all_extensions(x, f_true, path_ext_m1, path_ext_m2):
    xn = _to_np(x)
    X = torch.stack([x, f_true], dim=1)
    with torch.no_grad():
        ext_m1 = _to_np(path_ext_m1(X))
        ext_m2 = _to_np(path_ext_m2(X))
    n_m1 = ext_m1.shape[1]
    n_m2 = ext_m2.shape[1]
    fig, axes = plt.subplots(2, max(n_m1, n_m2), figsize=(5 * max(n_m1, n_m2), 6))
    if max(n_m1, n_m2) == 1:
        axes = axes.reshape(2, 1)
    for i in range(n_m1):
        axes[0, i].plot(xn, ext_m1[:, i])
        axes[0, i].set_title(f"M1 ext channel {i+1}")
        axes[0, i].set_xlabel("t")
    for i in range(n_m2):
        axes[1, i].plot(xn, ext_m2[:, i])
        axes[1, i].set_title(f"M2 ext channel {i+1}")
        axes[1, i].set_xlabel("t")
    for ax in axes.flat:
        ax.set_visible(False)
    for i in range(n_m1): axes[0, i].set_visible(True)
    for i in range(n_m2): axes[1, i].set_visible(True)
    plt.suptitle("Learned path extension channels", fontsize=13)
    plt.tight_layout(); plt.show()


# ── 4. Shuffle residual matrices ──────────────────────────────────────────────
def plot_all_shuffle_matrices(x, f_true, path_ext_m1, path_ext_m2):
    X = torch.stack([x, f_true], dim=1)
    with torch.no_grad():
        ext_m1 = path_ext_m1(X)
        ext_m2 = path_ext_m2(X)
    R_m1 = _to_np(shuffle_loss_residual(ext_m1))
    R_m2 = _to_np(shuffle_loss_residual(ext_m2))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    im0 = axes[0].imshow(R_m1, aspect="auto", cmap="RdBu_r")
    axes[0].set_title("Shuffle residual R — Method 1")
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(R_m2, aspect="auto", cmap="RdBu_r")
    axes[1].set_title("Shuffle residual R — Method 2")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout(); plt.show()


# ── 5. 2×2 final comparison (generic) ─────────────────────────────────────────
def plot_final_comparison_2x2_generic(
    x, target_true, u_ref,
    u_sig, target_sig,
    u_nb,  target_nb,
    target_name="forcing", title_prefix="",
):
    xn = _to_np(x)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes[0,0].plot(xn, _to_np(u_ref),     "k-",  label="Reference")
    axes[0,0].plot(xn, _to_np(u_nb),      "r--", label="Non-branched")
    axes[0,0].set_title(f"{title_prefix}Non-branched u(t)")
    axes[0,0].legend()
    axes[0,1].plot(xn, _to_np(target_true),"k-",  label=f"True {target_name}")
    axes[0,1].plot(xn, _to_np(target_nb),  "r--", label=f"Pred {target_name}")
    axes[0,1].set_title(f"{title_prefix}Non-branched {target_name}")
    axes[0,1].legend()
    axes[1,0].plot(xn, _to_np(u_ref),     "k-",  label="Reference")
    axes[1,0].plot(xn, _to_np(u_sig),     "b--", label="Branched NN")
    axes[1,0].set_title(f"{title_prefix}Branched NN u(t)")
    axes[1,0].legend()
    axes[1,1].plot(xn, _to_np(target_true),"k-",  label=f"True {target_name}")
    axes[1,1].plot(xn, _to_np(target_sig), "b--", label=f"Pred {target_name}")
    axes[1,1].set_title(f"{title_prefix}Branched NN {target_name}")
    axes[1,1].legend()
    for ax in axes.flat: ax.set_xlabel("t")
    plt.tight_layout(); plt.show()


# ── 6. Full train+test path — all variants × both methods ─────────────────────
def plot_all_variants_full_train_test(
    x_full, f_true_full, u_ref_full,
    u_nb_m1_full,  f_pred_nb_m1_full,
    u_tl_m1_full,  f_pred_tl_m1_full,
    u_sig_m1_full, f_pred_sig_m1_full,
    u_nb_m2_full,  f_pred_nb_m2_full,
    u_tl_m2_full,  f_pred_tl_m2_full,
    u_sig_m2_full, f_pred_sig_m2_full,
    f_target_nb_m2_full, f_target_tl_m2_full, f_target_sig_m2_full,
):
    xn = _to_np(x_full)
    variants = [
        ("Non-branched", u_nb_m1_full, f_pred_nb_m1_full,
                         u_nb_m2_full, f_pred_nb_m2_full, f_target_nb_m2_full),
        ("T-lift",       u_tl_m1_full, f_pred_tl_m1_full,
                         u_tl_m2_full, f_pred_tl_m2_full, f_target_tl_m2_full),
        ("Branched NN",  u_sig_m1_full, f_pred_sig_m1_full,
                         u_sig_m2_full, f_pred_sig_m2_full, f_target_sig_m2_full),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    for row, (name, u_m1, fp_m1, u_m2, fp_m2, ft_m2) in enumerate(variants):
        axes[row,0].plot(xn, _to_np(u_ref_full),    "k-",  lw=1.2, label="Reference")
        axes[row,0].plot(xn, _to_np(u_m1),          "r--", lw=1,   label="Predicted")
        axes[row,0].set_title(f"M1 {name} — u(t) [full]"); axes[row,0].legend(fontsize=7)

        axes[row,1].plot(xn, _to_np(f_true_full),   "k-",  lw=1.2, label="True f")
        axes[row,1].plot(xn, _to_np(fp_m1),         "r--", lw=1,   label="Pred f")
        axes[row,1].set_title(f"M1 {name} — f(t) [full]"); axes[row,1].legend(fontsize=7)

        axes[row,2].plot(xn, _to_np(u_ref_full),    "k-",  lw=1.2, label="Reference")
        axes[row,2].plot(xn, _to_np(u_m2),          "b--", lw=1,   label="Predicted")
        axes[row,2].set_title(f"M2 {name} — u(t) [full]"); axes[row,2].legend(fontsize=7)

        axes[row,3].plot(xn, _to_np(ft_m2),         "k-",  lw=1.2, label="True target")
        axes[row,3].plot(xn, _to_np(fp_m2),         "b--", lw=1,   label="Pred target")
        axes[row,3].set_title(f"M2 {name} — target [full]"); axes[row,3].legend(fontsize=7)

    for ax in axes.flat: ax.set_xlabel("t")
    plt.suptitle("All variants — full train+test path", fontsize=14, y=1.01)
    plt.tight_layout(); plt.show()


# ── 7. Test portion only — all variants × both methods ────────────────────────
def plot_all_variants_test_only(
    x_test, u_ref_full, f_true_full,
    u_nb_m1_full,  f_pred_nb_m1_full,
    u_tl_m1_full,  f_pred_tl_m1_full,
    u_sig_m1_full, f_pred_sig_m1_full,
    u_nb_m2_full,  f_pred_nb_m2_full,
    u_tl_m2_full,  f_pred_tl_m2_full,
    u_sig_m2_full, f_pred_sig_m2_full,
    N_train,
    f_target_nb_m2_full, f_target_tl_m2_full, f_target_sig_m2_full,
):
    xn = _to_np(x_test)
    # slice to test region
    def sl(t): return _to_np(t)[N_train:]

    variants = [
        ("Non-branched", sl(u_nb_m1_full),  sl(f_pred_nb_m1_full),
                         sl(u_nb_m2_full),  sl(f_pred_nb_m2_full),  sl(f_target_nb_m2_full)),
        ("T-lift",       sl(u_tl_m1_full),  sl(f_pred_tl_m1_full),
                         sl(u_tl_m2_full),  sl(f_pred_tl_m2_full),  sl(f_target_tl_m2_full)),
        ("Branched NN",  sl(u_sig_m1_full), sl(f_pred_sig_m1_full),
                         sl(u_sig_m2_full), sl(f_pred_sig_m2_full), sl(f_target_sig_m2_full)),
    ]
    u_ref_test    = sl(u_ref_full)
    f_true_test   = sl(f_true_full)

    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    for row, (name, u_m1, fp_m1, u_m2, fp_m2, ft_m2) in enumerate(variants):
        axes[row,0].plot(xn, u_ref_test, "k-",  lw=1.2, label="Reference")
        axes[row,0].plot(xn, u_m1,       "r--", lw=1,   label="Predicted")
        axes[row,0].set_title(f"M1 {name} — u(t) [test]"); axes[row,0].legend(fontsize=7)

        axes[row,1].plot(xn, f_true_test,"k-",  lw=1.2, label="True f")
        axes[row,1].plot(xn, fp_m1,      "r--", lw=1,   label="Pred f")
        axes[row,1].set_title(f"M1 {name} — f(t) [test]"); axes[row,1].legend(fontsize=7)

        axes[row,2].plot(xn, u_ref_test, "k-",  lw=1.2, label="Reference")
        axes[row,2].plot(xn, u_m2,       "b--", lw=1,   label="Predicted")
        axes[row,2].set_title(f"M2 {name} — u(t) [test]"); axes[row,2].legend(fontsize=7)

        axes[row,3].plot(xn, ft_m2,      "k-",  lw=1.2, label="True target")
        axes[row,3].plot(xn, fp_m2,      "b--", lw=1,   label="Pred target")
        axes[row,3].set_title(f"M2 {name} — target [test]"); axes[row,3].legend(fontsize=7)

    for ax in axes.flat: ax.set_xlabel("t")
    plt.suptitle("All variants — test portion only", fontsize=14, y=1.01)
    plt.tight_layout(); plt.show()

def tonp(t):
        import torch
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t


def plot_m1_forcing_calibration_sidebyside(x, ftrue, fpred_nb, fpred_sig):
    """
    Side-by-side plot of Method 1 forcing calibration on testing data:
      Left  — Non-branched predicted forcing vs. true forcing
      Right — Branched NN predicted forcing vs. true forcing
    """
    xn       = tonp(x)
    ftruenp  = tonp(ftrue)
    fprednb  = tonp(fpred_nb)
    fpredsig = tonp(fpred_sig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].plot(xn, ftruenp, 'k-',  lw=1.5, label='True $f(t)$')
    axes[0].plot(xn, fprednb, 'r--', lw=1.2, label='Non-branched pred')
    axes[0].set_title('Non-branched Forcing Prediction')
    axes[0].set_xlabel('$t$')
    axes[0].set_ylabel('$f(t)$')
    axes[0].legend()

    axes[1].plot(xn, ftruenp,  'k-',  lw=1.5, label='True $f(t)$')
    axes[1].plot(xn, fpredsig, 'b--', lw=1.2, label='Branched NN pred')
    axes[1].set_title('Branched Forcing Prediction (NN lift)')
    axes[1].set_xlabel('$t$')
    axes[1].legend()

    plt.suptitle('Method 1 Forcing Fit Comparison', fontsize=13)
    plt.tight_layout()
    plt.show()

def plot_m1_branched_full_sidebyside(x_full, forcing_full, u_ref_full,
                                      f_pred_sig_m1_full, u_sig_m1_full,
                                      N_train):
    """
    Side-by-side plot for Method 1 Branched NN on the full path (train + test):
      Left  — Forcing fit: true f(t) vs. predicted f(t)
      Right — Solution fit: reference u(t) vs. predicted u(t)
    Vertical dashed line marks the train/test split.
    """
    def to_np(t):
        import torch
        return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t

    xn        = to_np(x_full)
    ftruenp   = to_np(forcing_full)
    fprednp   = to_np(f_pred_sig_m1_full)
    urefnp    = to_np(u_ref_full)
    uprednp   = to_np(u_sig_m1_full)

    x_split = xn[N_train]  # x-value at the train/test boundary

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: forcing fit ────────────────────────────────────────────────────
    axes[0].plot(xn, ftruenp, 'k-',  lw=1.5, label='True $f(t)$')
    axes[0].plot(xn, fprednp, 'b--', lw=1.2, label='Branched NN pred')
    axes[0].axvline(x_split, color='gray', linestyle=':', lw=1.5,
                    label='Train/test split')
    axes[0].set_title('Method 1 Branched NN — Forcing Fit (Full Path)')
    axes[0].set_xlabel('$t$')
    axes[0].set_ylabel('$f(t)$')
    axes[0].legend()

    # ── Right: solution fit ──────────────────────────────────────────────────
    axes[1].plot(xn, urefnp,  'k-',  lw=1.5, label='Reference $u(t)$')
    axes[1].plot(xn, uprednp, 'b--', lw=1.2, label='Branched NN pred')
    axes[1].axvline(x_split, color='gray', linestyle=':', lw=1.5,
                    label='Train/test split')
    axes[1].set_title('Method 1 Branched NN — Solution Fit (Full Path)')
    axes[1].set_xlabel('$t$')
    axes[1].set_ylabel('$u(t)$')
    axes[1].legend()

    plt.suptitle('Method 1 Branched NN — Full Path (Train + Test)', fontsize=13)
    plt.tight_layout()
    plt.show()




# ============================================================
# Full-path assembly + error reporting
# ============================================================

def assemble_full_path_results(
    N_train,
    x, x_test, forcing,
    u_nb_m1,  f_pred_nb_m1,  u_test_nb_m1,  f_test_pred_nb_m1,
    u_tl_m1,  f_pred_tl_m1,  u_test_tl_m1,  f_test_pred_tl_m1,
    u_sig_m1, f_pred_sig_m1, u_test_sig_m1, f_test_pred_sig_m1,
    u_nb_m2,  f_pred_nb_m2,  u_test_nb_m2,  f_test_pred_nb_m2,
    u_tl_m2,  f_pred_tl_m2,  u_test_tl_m2,  f_test_pred_tl_m2,
    u_sig_m2, f_pred_sig_m2, u_test_sig_m2, f_test_pred_sig_m2,
    ya, ypa, k1, k2,
):
    """
    Concatenate train + test tensors into full-path tensors and
    compute the true Method 2 RHS targets for both the full and
    test-only grids.

    Returns a dict with keys:
        u_nb_m1, f_pred_nb_m1,  u_tl_m1, f_pred_tl_m1,
        u_sig_m1, f_pred_sig_m1,
        u_nb_m2, f_pred_nb_m2,  u_tl_m2, f_pred_tl_m2,
        u_sig_m2, f_pred_sig_m2,
        rhs_m2_full, rhs_m2_test
    """
    def cat(train, test):
        return torch.cat([train, test[N_train:]], dim=0)

    return dict(
        u_nb_m1        = cat(u_nb_m1,   u_test_nb_m1),
        f_pred_nb_m1   = cat(f_pred_nb_m1,  f_test_pred_nb_m1),
        u_tl_m1        = cat(u_tl_m1,   u_test_tl_m1),
        f_pred_tl_m1   = cat(f_pred_tl_m1,  f_test_pred_tl_m1),
        u_sig_m1       = cat(u_sig_m1,  u_test_sig_m1),
        f_pred_sig_m1  = cat(f_pred_sig_m1, f_test_pred_sig_m1),
        u_nb_m2        = cat(u_nb_m2,   u_test_nb_m2),
        f_pred_nb_m2   = cat(f_pred_nb_m2,  f_test_pred_nb_m2),
        u_tl_m2        = cat(u_tl_m2,   u_test_tl_m2),
        f_pred_tl_m2   = cat(f_pred_tl_m2,  f_test_pred_tl_m2),
        u_sig_m2       = cat(u_sig_m2,  u_test_sig_m2),
        f_pred_sig_m2  = cat(f_pred_sig_m2, f_test_pred_sig_m2),
        rhs_m2_full    = rhs_method2(f=forcing,            x=x,      ua=ya, upa=ypa, k1=k1, k2=k2),
        rhs_m2_test    = rhs_method2(f=forcing[N_train:],  x=x_test, ua=ya, upa=ypa, k1=k1, k2=k2),
    )


def print_all_errors(
    split_label,
    x, u_ref, forcing,
    u_nb_m1,  f_pred_nb_m1,
    u_tl_m1,  f_pred_tl_m1,
    u_sig_m1, f_pred_sig_m1,
    u_nb_m2,  f_pred_nb_m2,
    u_tl_m2,  f_pred_tl_m2,
    u_sig_m2, f_pred_sig_m2,
    rhs_m2,
):
    """
    Print the full error table (variant errors, % improvement vs
    non-branched baseline, Method 1 vs Method 2 solution improvement)
    for a given split.  split_label is e.g. 'FULL (train+test)' or
    'TEST ONLY'.

    Returns a dict with keys nb_m1, tl_m1, sig_m1, nb_m2, tl_m2,
    sig_m2, each being the get_errors dict for that variant.
    """
    print(f"\nComparing ALL variants on {split_label}: "
          "non-branched, t-lift, branched | method 1 vs method 2")

    print(f"\n--- Method 1 ({split_label}) ---")
    print_variant_errors("Non-branched", u_nb_m1,  f_pred_nb_m1,  u_ref, forcing, x)
    print_variant_errors("t-lift      ", u_tl_m1,  f_pred_tl_m1,  u_ref, forcing, x)
    print_variant_errors("Branched    ", u_sig_m1, f_pred_sig_m1, u_ref, forcing, x)

    print(f"\n--- Method 2 ({split_label}) ---")
    print_variant_errors("Non-branched", u_nb_m2,  f_pred_nb_m2,  u_ref, rhs_m2, x)
    print_variant_errors("t-lift      ", u_tl_m2,  f_pred_tl_m2,  u_ref, rhs_m2, x)
    print_variant_errors("Branched    ", u_sig_m2, f_pred_sig_m2, u_ref, rhs_m2, x)

    nb_m1  = get_errors(u_nb_m1,  f_pred_nb_m1,  u_ref, forcing)
    tl_m1  = get_errors(u_tl_m1,  f_pred_tl_m1,  u_ref, forcing)
    sig_m1 = get_errors(u_sig_m1, f_pred_sig_m1, u_ref, forcing)

    print(f"\n--- Method 1 ({split_label}): % improvement vs Non-branched (positive = better) ---")
    print("  [Non-branched]  (baseline)")
    print_pct_improvement("t-lift      ", tl_m1,  nb_m1)
    print_pct_improvement("Branched    ", sig_m1, nb_m1)

    nb_m2  = get_errors(u_nb_m2,  f_pred_nb_m2,  u_ref, rhs_m2)
    tl_m2  = get_errors(u_tl_m2,  f_pred_tl_m2,  u_ref, rhs_m2)
    sig_m2 = get_errors(u_sig_m2, f_pred_sig_m2, u_ref, rhs_m2)

    print(f"\n--- Method 2 ({split_label}): % improvement vs Non-branched (positive = better) ---")
    print("  [Non-branched]  (baseline)")
    print_pct_improvement("t-lift      ", tl_m2,  nb_m2)
    print_pct_improvement("Branched    ", sig_m2, nb_m2)

    print(f"\n--- Method 1 vs Method 2 ({split_label}): % solution improvement "
          "(positive = Method 1 better) ---")
    for lbl, e_m1, e_m2 in [
        ("Non-branched", nb_m1,  nb_m2),
        ("t-lift      ", tl_m1,  tl_m2),
        ("Branched    ", sig_m1, sig_m2),
    ]:
        def pct(v1, v2): return 100.0 * (v2 - v1) / (abs(v2) + 1e-12)
        print(f"  [{lbl}]"
              f"  MSE(u)={pct(e_m1['mse_u'], e_m2['mse_u']):+.1f}%"
              f"  Rel(u)={pct(e_m1['rel_u'], e_m2['rel_u']):+.1f}%")

    return dict(nb_m1=nb_m1, tl_m1=tl_m1, sig_m1=sig_m1,
                nb_m2=nb_m2, tl_m2=tl_m2, sig_m2=sig_m2)