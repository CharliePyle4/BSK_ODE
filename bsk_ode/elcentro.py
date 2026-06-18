import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
import torch
import math
import matplotlib.pyplot as plt
import random
import keras_sig
from scipy.integrate import solve_ivp
from torch import cumulative_trapezoid



def trapezoidal_cols(M, dt):
    """Cumulative trapezoidal integral along dim=0, works for 1D or 2D tensors."""
    trap = (M[:-1] + M[1:]) / 2 * dt
    out = torch.zeros_like(M)
    out[1:] = torch.cumsum(trap, dim=0)
    return out

def forcing_loss(true_forcing, approximated_forcing):
    residual = true_forcing - approximated_forcing
    loss = torch.mean(residual**2)
    return loss

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

def build_kernel_from_signatures(sigs_flat: torch.Tensor) -> torch.Tensor:
    """
    Build a kernel matrix from signature features.
    sigs_flat: (T,D)
    """
    Ker = sigs_flat @ sigs_flat.T
    return Ker

def normalize_signatures(Z: torch.Tensor,
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

def buildkerneloperators(Ksig: torch.Tensor, x: torch.Tensor):
    dtype = Ksig.dtype
    device = Ksig.device
    x = x.to(device=device, dtype=dtype)

    K0 = Ksig.clone()  # K
    I1 = cumulative_trapezoid(K0, x, dim=0)
    I1 = torch.vstack([torch.zeros_like(I1[:1]), I1])  # I K
    I2 = cumulative_trapezoid(I1, x, dim=0)
    I2 = torch.vstack([torch.zeros_like(I2[:1]), I2])  # I^2 K
    return K0, I1, I2

def double_integrate(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return cumtrapz_torch(cumtrapz_torch(y, x), x)

def build_rhs(f: torch.Tensor, x: torch.Tensor,
                ua: float, upa: float,
                k1: float, k2: float):
    x0 = x[0]
    q = k1 * ua + (k1 * upa + k2 * ua) * (x - x0)
    return double_integrate(f, x) + q

def solvebetas(Ksig: torch.Tensor,
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

    K0, IK, I2K = buildkerneloperators(Ksig, x)

    A = k1 * K0 + k2 * IK + k3 * I2K
    rhs = build_rhs(f, x, ua, upa, k1, k2).to(device=device, dtype=dtype)

    N = Ksig.shape[0]
    Ireg = torch.eye(N, dtype=dtype, device=device)
    beta = torch.linalg.solve(A.T @ A + reg * Ireg, A.T @ rhs)

    u = K0 @ beta
    Iu = IK @ beta
    I2u = I2K @ beta
    rhs_pred = k1 * u + k2 * Iu + k3 * I2u

    return beta, u, rhs_pred, rhs


def solve_signature_kernel(x, f,
                                                k1, k2, k3,
                                                ua, upa,
                                                depth,
                                                normalize = True):

    with torch.no_grad():

        X = torch.stack([x, f], dim=1)           # (T,2)

        X_sig = compute_signatures(X, depth)     # (T,D)

        if(normalize == True):
            X_sig = normalize_signatures()

        Ksig = build_kernel_from_signatures(X_sig)

        beta_w, u, f_pred_final, rhs_true = solvebetas(
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
        print(f"non-branched integrated-target loss: {final_loss.item():.3e}")

    return u, f_pred_final




def plot_calibration(time, F_star, F_hat, U_true, U_hat):
    """
    Plot calibration results.

    Parameters
    ----------
    time   : array-like  (T,)  — time grid
    F_star : tensor/array (T,) — reference double-integrated forcing
    F_hat  : tensor/array (T,) — model-predicted double-integrated forcing
    U_true : tensor/array (T,) — reference solution u(t)
    U_hat  : tensor/array (T,) — reconstructed solution u(t)
    """
    # Convert tensors to plain lists for matplotlib
    def _to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return list(x)

    time_list   = _to_list(time)
    F_star_list = _to_list(F_star)
    F_hat_list  = _to_list(F_hat)
    U_true_list = _to_list(U_true)
    U_hat_list  = _to_list(U_hat)        # ← was incorrectly K_A

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Forcing
    axes[0].plot(time_list, F_star_list,
                 color="black", linewidth=1.5, label="Reference Forcing")
    axes[0].plot(time_list, F_hat_list,
                 color="blue", linestyle="--", linewidth=1.5, label="Calibrated Forcing")
    axes[0].set_title("Double Integral Forcing Fit")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("$\\iint f(t)$")
    axes[0].legend(frameon=True, fancybox=False, edgecolor="black")
    axes[0].grid(True, alpha=0.3)

    # Right: Solution
    axes[1].plot(time_list, U_true_list,
                 color="black", linewidth=1.5, label="Reference $u(t)$")
    axes[1].plot(time_list, U_hat_list,          # ← fixed: was K_A
                 color="blue", linestyle="--", linewidth=1.5, label="Reconstructed $u(t)$")
    axes[1].set_title("Solution Reconstruction")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("$u(t)$")
    axes[1].legend(frameon=True, fancybox=False, edgecolor="black")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()