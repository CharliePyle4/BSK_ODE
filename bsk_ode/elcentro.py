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
import numpy as np



def trapezoidal_cols(M, dt):
    """Cumulative trapezoidal integral along dim=0, works for 1D or 2D tensors."""
    trap = (M[:-1] + M[1:]) / 2 * dt
    out = torch.zeros_like(M)
    out[1:] = torch.cumsum(trap, dim=0)
    return out


def forcing_loss(true_forcing, approximated_forcing):
    # Move both to CPU so subtraction is always valid
    true_forcing = true_forcing.detach().to("cpu")
    approximated_forcing = approximated_forcing.detach().to("cpu")
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



def solvebetas(
    Ksig: torch.Tensor,
    f: torch.Tensor,
    x: torch.Tensor,
    ua: float,
    upa: float,
    k1: float,
    k2: float,
    k3: float,
    reg: float = 1e-10,
):
    """
    GPU-native replacement for torch.linalg.lstsq(..., driver='gelsd').

    Psi is square (T×T), so ridge-regularized LU solve is 3-5x faster
    than full SVD on a T4 and avoids float64 SVD overhead entirely.
    Mathematically equivalent to gelsd with small rcond when reg is small.
    """
    dtype = torch.float64
    device = Ksig.device

    Ksig = Ksig.to(device=device, dtype=dtype)
    x    = x.to(device=device, dtype=dtype).flatten()
    f    = f.to(device=device, dtype=dtype).flatten()

    dt = x[1] - x[0]

    K0  = Ksig
    IK  = trapezoidal_cols(K0, dt)
    I2K = trapezoidal_cols(IK, dt)

    Psi    = k1 * K0 + k2 * IK + k3 * I2K
    F_star = trapezoidal_cols(trapezoidal_cols(f, dt), dt)

    if ua != 0.0 or upa != 0.0:
        print(
            "Warning: solvebetas assumes zero initial conditions; "
            "nonzero ua/upa are ignored."
        )

    # Ridge-regularized solve: (Psi + lam*I) beta = F_star
    # lam scales with the diagonal mean so it is invariant to problem scale.
    T   = Psi.shape[0]
    lam = max(float(reg), float(torch.finfo(dtype).eps) * float(Psi.diagonal().abs().mean()))
    beta = torch.linalg.solve(
        Psi + lam * torch.eye(T, dtype=dtype, device=device),
        F_star,
    )



    if not torch.isfinite(beta).all():
        raise ValueError("beta contains NaN/Inf after solve. Increase reg.")

    u        = K0  @ beta
    Iu       = IK  @ beta
    I2u      = I2K @ beta
    rhs_pred = k1 * u + k2 * Iu + k3 * I2u

    return beta, u, rhs_pred, F_star



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


def normalize_signatures(
    Z: torch.Tensor,
    depth: int,
    dim: int,
    eps: float = 1e-10,
    return_stats: bool = False,
    **kwargs,
):
    """
    Column-wise robust normalization using median and IQR.

    If return_stats=False:
        returns Z_norm

    If return_stats=True:
        returns Z_norm, med, iqr
    """
    med = Z.median(dim=0, keepdim=True).values
    q25 = Z.quantile(0.25, dim=0, keepdim=True)
    q75 = Z.quantile(0.75, dim=0, keepdim=True)
    iqr = q75 - q25

    Z_norm = (Z - med) / (iqr + eps)

    if return_stats:
        return Z_norm, med, iqr
    return Z_norm


def apply_signature_normalization(
    Z: torch.Tensor,
    med: torch.Tensor,
    iqr: torch.Tensor,
    eps: float = 1e-10,
):
    """
    Apply precomputed normalization stats to signatures.
    """
    return (Z - med) / (iqr + eps)



def solve_signature_kernel_calibration(x, f,
                        k1, k2, k3,
                        ua, upa,
                        depth,
                        normalize=True,
                        reg=1e-10,
                        norm_eps=1e-10,      
                        use_tlift=False,
                        holder_value=None):

    with torch.no_grad():
        X = torch.stack([x, f], dim=1)

        if use_tlift:
            if holder_value is None:
                raise ValueError("holder_value must be provided when use_tlift=True")
            X = tlift(X, holder_value)

        X_sig = compute_signatures(X, depth)

        med, iqr = None, None
        if normalize:
            X_sig, med, iqr = normalize_signatures(
                Z=X_sig,
                depth=depth,
                dim=X.shape[1],
                eps=norm_eps,               # ← pass it
                return_stats=True,
            )

        Ksig = build_kernel_from_signatures(X_sig)

        alpha, u, f_pred_final, rhs_true = solvebetas(
            Ksig=Ksig,
            f=f, x=x,
            ua=ua, upa=upa,
            k1=k1, k2=k2, k3=k3,
            reg=reg
        )

    return u, f_pred_final, alpha, X_sig, med, iqr, norm_eps



# -------------------------------------------------------
# Signature + normalization helpers (rolling version)
# -------------------------------------------------------
'''
def signature_of_path(path, depth: int = 8) -> torch.Tensor:
    """Prefix signature of a single path (T, d) using keras_sig."""
    if not isinstance(path, torch.Tensor):
        path = torch.tensor(path, dtype=torch.float64)
    if path.dim() == 2:
        path = path.unsqueeze(0)                          # (1, T, d)
    basepoint = path[:, 0:1, :]
    path_bp   = torch.cat([basepoint, path], dim=1)       # prepend basepoint
    sigs_raw  = keras_sig.signature(
        path_bp,
        depth=depth,
        stream=False,                                     # single sig per path
        gpu_optimized=True,
    )
    return sigs_raw.squeeze(0).to(device=device, dtype=torch.float64).detach()


# -------------------------------------------------------
# Rolling online prediction functions
# -------------------------------------------------------
def build_state(paths,
                n0: int,
                signature_level: int,
                m: float, c: float, k: float,
                dt: float, N: int,
                F_star: torch.Tensor,
                t_vals: torch.Tensor,
                u_true_interp: torch.Tensor,
                reg: float = 1e-10,
                norm_eps = 1e-10) -> dict:
    """
    Train on the first n0+1 paths and return a state dict
    ready for rolling_online_predict.
    """

    # --- move inputs to device once ---
    F_star = F_star.to(device)
    t_vals = t_vals.to(device)
    u_true_interp = u_true_interp.to(device)

    # signatures for first n0+1 paths
    S0_raw = torch.stack([
        signature_of_path(paths[i], depth=signature_level)
        for i in range(n0 + 1)
    ])  # signature_of_path already returns on `device`


    S0, med, iqr = normalize_signatures(
    Z=S0_raw, depth=signature_level, dim=S0_raw.shape[1],
    eps=norm_eps,           
    return_stats=True,
    )

    K0   = S0 @ S0.T                    # (n0+1, n0+1), on device
    K1_0 = trapezoidal_cols(K0, dt)
    K2_0 = trapezoidal_cols(K1_0, dt)

    Psi0 = m * K0 + c * K1_0 + k * K2_0

    # Ridge-regularized solve (replaces gelsd which is CPU-only)
    T   = Psi0.shape[0]
    lam = max(float(reg), float(torch.finfo(torch.float64).eps) * float(Psi0.diagonal().abs().mean()))
    alpha0 = torch.linalg.solve(
        Psi0 + lam * torch.eye(T, dtype=torch.float64, device=device),
        F_star[:n0 + 1],
    )

    F_pred_train = Psi0 @ alpha0
    u_pred_train = K0  @ alpha0

    return {
        "m": m, "c": c, "k": k,
        "dt": dt, "n0": n0, "N": N,
        "paths": paths,
        "signature_level": signature_level,
        "med": med, "iqr": iqr,
        "F_star": F_star,
        "t_vals": t_vals,
        "u_true_interp": u_true_interp,
        "alpha0": alpha0,
        "S_hist": S0.clone(),
        "K_prev": K0[n0, :].clone(),
        "I1": K1_0[n0, :].clone(),
        "I2": K2_0[n0, :].clone(),
        "F_pred_train": F_pred_train,
        "u_pred_train": u_pred_train,
    }


def rolling_online_predict(state: dict,
                           retrain_every: int = 5,
                           max_steps: int | None = None,
                           reg: float = 1e-10,
                           norm_eps: float = 1e-10,
                        ) -> dict:
    """
    Online sequential prediction with periodic full retraining.
    """
    m, c, k = state["m"], state["c"], state["k"]
    dt    = state["dt"]
    n0    = state["n0"]
    N     = state["N"]
    paths = state["paths"]
    depth = state["signature_level"]
    med, iqr = state["med"], state["iqr"]
    F_star   = state["F_star"]
    dev      = F_star.device

    end_idx = N - 1 if max_steps is None else min(N - 1, n0 + max_steps)

    S_hist = state["S_hist"].clone()
    alphas = torch.zeros(end_idx + 1, dtype=torch.float64, device=dev)
    alphas[:n0 + 1] = state["alpha0"]

    K_prev = state["K_prev"].clone()
    I1 = state["I1"].clone()
    I2 = state["I2"].clone()

    F_pred = torch.zeros(end_idx + 1, dtype=torch.float64, device=dev)
    u_pred = torch.zeros(end_idx + 1, dtype=torch.float64, device=dev)

    F_pred[:n0 + 1] = state["F_pred_train"]
    u_pred[:n0 + 1] = state["u_pred_train"]

    retrain_indices = []

    for i in range(n0 + 1, end_idx + 1):
        s_raw = signature_of_path(paths[i], depth=depth)
        s_new = apply_signature_normalization(s_raw, med, iqr, eps=norm_eps).squeeze(0)



        k_row_old = S_hist @ s_new
        k_ii      = float(torch.dot(s_new, s_new).item())

        I1_new = I1 + 0.5 * (K_prev + k_row_old) * dt
        I2_new = I2 + 0.5 * (I1 + I1_new) * dt
        I1, I2 = I1_new, I2_new
        K_prev = k_row_old

        col_i   = torch.cat([k_row_old, torch.tensor([k_ii], dtype=torch.float64, device=dev)])
        inner_i = trapezoidal_cols(col_i, dt)
        outer_i = trapezoidal_cols(inner_i, dt)

        I1 = torch.cat([I1, torch.tensor([float(inner_i[-1])], dtype=torch.float64, device=dev)])
        I2 = torch.cat([I2, torch.tensor([float(outer_i[-1])], dtype=torch.float64, device=dev)])
        K_prev = torch.cat([K_prev, torch.tensor([k_ii], dtype=torch.float64, device=dev)])

        psi_row_old = m * k_row_old + c * I1[:i] + k * I2[:i]
        psi_diag    = m * k_ii + c * float(I1[i]) + k * float(I2[i])

        residual = F_star[i] - torch.dot(psi_row_old, alphas[:i])
        alphas[i] = residual / (psi_diag + norm_eps)

        F_pred[i] = torch.dot(psi_row_old, alphas[:i]) + psi_diag * alphas[i]
        u_pred[i] = torch.dot(k_row_old,   alphas[:i]) + k_ii      * alphas[i]

        S_hist = torch.vstack([S_hist, s_new.unsqueeze(0)])

        if (i - n0) % retrain_every == 0:
            retrain_indices.append(i)

            K      = S_hist @ S_hist.T
            K1     = trapezoidal_cols(K, dt)
            K2     = trapezoidal_cols(K1, dt)
            Psi    = m * K + c * K1 + k * K2
            Psi_bl = Psi[:i + 1, :i + 1]
            F_bl   = F_star[:i + 1]

            lam = max(float(reg), float(torch.finfo(torch.float64).eps) * float(Psi_bl.diagonal().abs().mean()))
            I_mat  = torch.eye(i + 1, dtype=torch.float64, device=dev)
            alphas[:i + 1] = torch.linalg.solve(Psi_bl + lam * I_mat, F_bl)


    return {
        "F_pred": F_pred,
        "u_pred": u_pred,
        "retrain_indices": retrain_indices,
        "end_idx": end_idx,
    }


def solve_signature_kernel_rolling_retrain(
    t_train: torch.Tensor,
    t_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua=0.0, upa=0.0,
    depth=8,
    normalize: bool = True,
    reg: float = 1e-10,
    norm_eps: float = 1e-10,
    retrain_every: int = 10,
    n0: int = 200,
    use_tlift: bool = False,
    holder_value=None,
):
    """
    Rolling retrain using partition prefix paths + incremental Gauss-Seidel update.

    n0 controls the initial training window size (number of points used for the
    first solve). Rolling prediction then runs from n0+1 through all N points.
    This matches the reference implementation where n0=200 is independent of
    the train/test split.

    n0 is from legacy code, always set it to Ntrain
    """
    if use_tlift and holder_value is None:
        raise ValueError("holder_value must be provided when use_tlift=True")

    with torch.no_grad():
        t_all = torch.cat([t_train, t_test], dim=0).to(dtype=torch.float64)
        f_all = torch.cat([f_train, f_test], dim=0).to(dtype=torch.float64)
        N  = t_all.numel()
        dt = float((t_all[1] - t_all[0]).item())

        if n0 >= N:
            raise ValueError(f"n0={n0} must be less than total points N={N}")

        t_cpu = t_all.cpu()
        f_cpu = f_all.cpu()

        # Build prefix paths: paths[i] is the path from index 0..i  (shape (i+1, d))
        if use_tlift:
            lift_cpu = t_cpu ** (2 * holder_value)
            paths = [
                torch.stack([t_cpu[:i+1], f_cpu[:i+1], lift_cpu[:i+1]], dim=1)
                for i in range(N)
            ]
        else:
            paths = [
                torch.stack([t_cpu[:i+1], f_cpu[:i+1]], dim=1)
                for i in range(N)
            ]

        # Double-integrated forcing — regression target over all N points
        F_star = trapezoidal_cols(trapezoidal_cols(f_all.to(device), dt), dt)


        state = build_state(
        paths=paths,
        n0=n0,
        signature_level=depth,
        m=k1, c=k2, k=k3,
        dt=dt, N=N,
        F_star=F_star,
        t_vals=t_all,
        u_true_interp=torch.zeros(N, dtype=torch.float64),
        reg=reg,
        norm_eps=norm_eps,
        )

        res = rolling_online_predict(
            state,
            retrain_every=retrain_every,
            reg=reg,
            norm_eps=norm_eps,
        )

    u_pred = res["u_pred"]
    F_pred = res["F_pred"]

    final_loss = forcing_loss(F_star, F_pred)
    print(f"final forcing loss (train+test, rolling): {final_loss.item():.3e}")

    return u_pred, F_pred

'''

# -------------------------------------------------------
# Signature + normalization helpers (rolling version)
# -------------------------------------------------------

def compute_stream_signatures(
    t_all: torch.Tensor,
    f_all: torch.Tensor,
    depth: int,
    use_tlift: bool = False,
    holder_value: float | None = None,
) -> torch.Tensor:
    """Compute prefix signatures for the full path using keras_sig with stream=True.

    Returns:
        S_all_raw: Tensor of shape (N, D) where row i is the signature of the
            prefix path from index 0..i (with a basepoint prepended).
    """
    if use_tlift:
        if holder_value is None:
            raise ValueError("holder_value must be provided when use_tlift=True")
        lift = t_all ** (2 * holder_value)
        X_full = torch.stack([t_all, f_all, lift], dim=1)  # (N, d=3)
    else:
        X_full = torch.stack([t_all, f_all], dim=1)        # (N, d=2)

    # Add batch dimension
    X_full = X_full.unsqueeze(0)  # (1, N, d)

    # Prepend basepoint as in your original code
    basepoint = X_full[:, 0:1, :]          # (1, 1, d)
    X_bp = torch.cat([basepoint, X_full], dim=1)  # (1, N+1, d)

    # Streaming signatures over the whole path
    sigs_stream = keras_sig.signature(
        X_bp,
        depth=depth,
        stream=True,
        gpu_optimized=True,
    )  # shape ~ (1, N+1, D)

    # Drop the basepoint-only prefix; keep N prefixes
    S_all_raw = sigs_stream[:, 1:, :].squeeze(0)  # (N, D)
    return S_all_raw.to(device=device, dtype=torch.float64).detach()


# -------------------------------------------------------
# Rolling online prediction functions
# -------------------------------------------------------

def build_state(
    S_all_raw: torch.Tensor,
    n0: int,
    signature_level: int,
    m: float,
    c: float,
    k: float,
    dt: float,
    N: int,
    F_star: torch.Tensor,
    t_vals: torch.Tensor,
    u_true_interp: torch.Tensor,
    reg: float = 1e-10,
    norm_eps: float = 1e-10,
) -> dict:
    """
    Train on the first n0+1 points and return a state dict
    ready for rolling_online_predict.
    """

    # --- move inputs to device once ---
    F_star = F_star.to(device)
    t_vals = t_vals.to(device)
    u_true_interp = u_true_interp.to(device)

    # signatures for first n0+1 points (prefix signatures already)
    S0_raw = S_all_raw[: n0 + 1]  # (n0+1, D)

    S0, med, iqr = normalize_signatures(
        Z=S0_raw,
        depth=signature_level,
        dim=S0_raw.shape[1],
        eps=norm_eps,
        return_stats=True,
    )

    K0 = S0 @ S0.T                   # (n0+1, n0+1), on device
    K1_0 = trapezoidal_cols(K0, dt)
    K2_0 = trapezoidal_cols(K1_0, dt)

    Psi0 = m * K0 + c * K1_0 + k * K2_0

    # Ridge-regularized solve (replaces gelsd which is CPU-only)
    T = Psi0.shape[0]
    lam = max(
        float(reg),
        float(torch.finfo(torch.float64).eps)
        * float(Psi0.diagonal().abs().mean()),
    )
    alpha0 = torch.linalg.solve(
        Psi0 + lam * torch.eye(T, dtype=torch.float64, device=device),
        F_star[: n0 + 1],
    )

    F_pred_train = Psi0 @ alpha0
    u_pred_train = K0 @ alpha0

    return {
        "m": m,
        "c": c,
        "k": k,
        "dt": dt,
        "n0": n0,
        "N": N,
        "signature_level": signature_level,
        "med": med,
        "iqr": iqr,
        "F_star": F_star,
        "t_vals": t_vals,
        "u_true_interp": u_true_interp,
        "alpha0": alpha0,
        "S_hist": S0.clone(),         # history of normalized sigs
        "S_all_raw": S_all_raw,       # all (unnormalized) prefix sigs
        "K_prev": K0[n0, :].clone(),
        "I1": K1_0[n0, :].clone(),
        "I2": K2_0[n0, :].clone(),
        "F_pred_train": F_pred_train,
        "u_pred_train": u_pred_train,
    }


def rolling_online_predict(
    state: dict,
    retrain_every: int = 5,
    max_steps: int | None = None,
    reg: float = 1e-10,
    norm_eps: float = 1e-10,
) -> dict:
    """
    Online sequential prediction with periodic full retraining.
    """
    m, c, k = state["m"], state["c"], state["k"]
    dt = state["dt"]
    n0 = state["n0"]
    N = state["N"]
    depth = state["signature_level"]
    med, iqr = state["med"], state["iqr"]
    F_star = state["F_star"]
    dev = F_star.device
    S_all_raw = state["S_all_raw"]

    end_idx = N - 1 if max_steps is None else min(N - 1, n0 + max_steps)

    S_hist = state["S_hist"].clone()
    alphas = torch.zeros(end_idx + 1, dtype=torch.float64, device=dev)
    alphas[: n0 + 1] = state["alpha0"]

    K_prev = state["K_prev"].clone()
    I1 = state["I1"].clone()
    I2 = state["I2"].clone()

    F_pred = torch.zeros(end_idx + 1, dtype=torch.float64, device=dev)
    u_pred = torch.zeros(end_idx + 1, dtype=torch.float64, device=dev)

    F_pred[: n0 + 1] = state["F_pred_train"]
    u_pred[: n0 + 1] = state["u_pred_train"]

    retrain_indices = []

    for i in range(n0 + 1, end_idx + 1):
        # Take precomputed (unnormalized) prefix signature at index i
        s_raw = S_all_raw[i]  # (D,)
        s_new = apply_signature_normalization(
            s_raw.unsqueeze(0), med, iqr, eps=norm_eps
        ).squeeze(0)  # (D,)

        k_row_old = S_hist @ s_new
        k_ii = float(torch.dot(s_new, s_new).item())

        I1_new = I1 + 0.5 * (K_prev + k_row_old) * dt
        I2_new = I2 + 0.5 * (I1 + I1_new) * dt
        I1, I2 = I1_new, I2_new
        K_prev = k_row_old

        col_i = torch.cat(
            [k_row_old, torch.tensor([k_ii], dtype=torch.float64, device=dev)]
        )
        inner_i = trapezoidal_cols(col_i, dt)
        outer_i = trapezoidal_cols(inner_i, dt)

        I1 = torch.cat(
            [I1, torch.tensor([float(inner_i[-1])], dtype=torch.float64, device=dev)]
        )
        I2 = torch.cat(
            [I2, torch.tensor([float(outer_i[-1])], dtype=torch.float64, device=dev)]
        )
        K_prev = torch.cat(
            [K_prev, torch.tensor([k_ii], dtype=torch.float64, device=dev)]
        )

        psi_row_old = m * k_row_old + c * I1[:i] + k * I2[:i]
        psi_diag = m * k_ii + c * float(I1[i]) + k * float(I2[i])

        residual = F_star[i] - torch.dot(psi_row_old, alphas[:i])
        alphas[i] = residual / (psi_diag + norm_eps)

        F_pred[i] = torch.dot(psi_row_old, alphas[:i]) + psi_diag * alphas[i]
        u_pred[i] = torch.dot(k_row_old, alphas[:i]) + k_ii * alphas[i]

        S_hist = torch.vstack([S_hist, s_new.unsqueeze(0)])

        if (i - n0) % retrain_every == 0:
            retrain_indices.append(i)

            K = S_hist @ S_hist.T
            K1 = trapezoidal_cols(K, dt)
            K2 = trapezoidal_cols(K1, dt)
            Psi = m * K + c * K1 + k * K2
            Psi_bl = Psi[: i + 1, : i + 1]
            F_bl = F_star[: i + 1]

            lam = max(
                float(reg),
                float(torch.finfo(torch.float64).eps)
                * float(Psi_bl.diagonal().abs().mean()),
            )
            I_mat = torch.eye(i + 1, dtype=torch.float64, device=dev)
            alphas[: i + 1] = torch.linalg.solve(Psi_bl + lam * I_mat, F_bl)

    return {
        "F_pred": F_pred,
        "u_pred": u_pred,
        "retrain_indices": retrain_indices,
        "end_idx": end_idx,
    }


def solve_signature_kernel_rolling_retrain(
    t_train: torch.Tensor,
    t_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1,
    k2,
    k3,
    ua=0.0,
    upa=0.0,
    depth=8,
    normalize: bool = True,
    reg: float = 1e-10,
    norm_eps: float = 1e-10,
    retrain_every: int = 10,
    n0: int = 200,
    use_tlift: bool = False,
    holder_value=None,
):
    """
    Rolling retrain using streamed prefix signatures + incremental Gauss-Seidel update.
    """
    if use_tlift and holder_value is None:
        raise ValueError("holder_value must be provided when use_tlift=True")

    with torch.no_grad():
        t_all = torch.cat([t_train, t_test], dim=0).to(dtype=torch.float64)
        f_all = torch.cat([f_train, f_test], dim=0).to(dtype=torch.float64)
        N = t_all.numel()
        dt = float((t_all[1] - t_all[0]).item())

        if n0 >= N:
            raise ValueError(f"n0={n0} must be less than total points N={N}")

        # Precompute all prefix signatures in one streaming call
        S_all_raw = compute_stream_signatures(
            t_all,
            f_all,
            depth=depth,
            use_tlift=use_tlift,
            holder_value=holder_value,
        )

        # Double-integrated forcing — regression target over all N points
        F_star = trapezoidal_cols(trapezoidal_cols(f_all.to(device), dt), dt)

        state = build_state(
            S_all_raw=S_all_raw,
            n0=n0,
            signature_level=depth,
            m=k1,
            c=k2,
            k=k3,
            dt=dt,
            N=N,
            F_star=F_star,
            t_vals=t_all,
            u_true_interp=torch.zeros(N, dtype=torch.float64),
            reg=reg,
            norm_eps=norm_eps,
        )

        res = rolling_online_predict(
            state,
            retrain_every=retrain_every,
            reg=reg,
            norm_eps=norm_eps,
        )

    u_pred = res["u_pred"]
    F_pred = res["F_pred"]

    final_loss = forcing_loss(F_star, F_pred)
    print(f"final forcing loss (train+test, rolling): {final_loss.item():.3e}")

    return u_pred, F_pred



def print_errors_calibration(F_pred, F_star, U_pred, U_true):
    # Ensure predictions on same device as references
    F_pred = F_pred.to(F_star.device)
    U_pred = U_pred.to(U_true.device)

    # Forcing errors
    abs_mse_F = mse(F_pred, F_star)
    rel_mse_F = rel_mse(F_pred, F_star)

    # Solution errors
    abs_mse_u = mse(U_pred, U_true)
    rel_mse_u = rel_mse(U_pred, U_true)

    print("\n==============================")
    print("Model Error Summary")
    print("==============================")
    print(f"{'Quantity':20s} {'Absolute MSE':>18s} {'Relative MSE':>18s} {'Relative MSE (%)':>20s}")
    print("-" * 80)
    print(f"{'Forcing F*':20s} {abs_mse_F:>18.6e} {rel_mse_F:>18.6e} {100 * rel_mse_F:>19.4f}%")
    print(f"{'Solution u(t)':20s} {abs_mse_u:>18.6e} {rel_mse_u:>18.6e} {100 * rel_mse_u:>19.4f}%")

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



def plot_normal_vs_tlift(
    t_vals: torch.Tensor,
    TRAIN_FRAC: float,
    u_pred_full: torch.Tensor,
    f_pred_full: torch.Tensor,
    u_pred_full_tlift: torch.Tensor,
    f_pred_full_tlift: torch.Tensor,
    U_ref: torch.Tensor,
    F_star: torch.Tensor,
):
    """
    2x2 plot: forcing and solution, normal vs t-lift,
    with a vertical line at the train/test split.
    """
    # Indices and split
    N        = t_vals.numel()
    N_train  = int(N * TRAIN_FRAC)
    idx_train = torch.arange(0, N_train)
    idx_test  = torch.arange(N_train, N)

    t_train = t_vals[idx_train]
    t_test  = t_vals[idx_test]
    t_split = float(t_vals[N_train - 1].item())

    # Convert to lists for matplotlib
    def _tolist(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return list(x)

    t_train_l = _tolist(t_train)
    t_test_l  = _tolist(t_test)

    F_star_train_l = _tolist(F_star[idx_train])
    F_star_test_l  = _tolist(F_star[idx_test])

    f_norm_train_l = _tolist(f_pred_full[idx_train])
    f_norm_test_l  = _tolist(f_pred_full[idx_test])

    f_tlift_train_l = _tolist(f_pred_full_tlift[idx_train])
    f_tlift_test_l  = _tolist(f_pred_full_tlift[idx_test])

    u_ref_train_l = _tolist(U_ref[idx_train])
    u_ref_test_l  = _tolist(U_ref[idx_test])

    u_norm_train_l = _tolist(u_pred_full[idx_train])
    u_norm_test_l  = _tolist(u_pred_full[idx_test])

    u_tlift_train_l = _tolist(u_pred_full_tlift[idx_train])
    u_tlift_test_l  = _tolist(u_pred_full_tlift[idx_test])

    # Plot style (reuse style from other helpers)
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 15,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.1,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # --- Top-left: Forcing, normal ---
    ax = axes[0, 0]
    ax.plot(t_train_l, F_star_train_l,
            color="black", linewidth=1.5, label="True F*")
    ax.plot(t_train_l, f_norm_train_l,
            color="red", linestyle="--", linewidth=1.5)
    ax.plot(t_test_l,  F_star_test_l,
            color="black", linewidth=1.5)
    ax.plot(t_test_l,  f_norm_test_l,
            color="red", linestyle="--", linewidth=1.5,
            label="normal pred F*")
    ax.axvline(x=t_split, color="gray", linestyle=":", linewidth=1.4,
               label="Train/test split")
    ax.set_title("Forcing: normal vs true")
    ax.set_xlabel("t")
    ax.set_ylabel("F*(t)")
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    # --- Top-right: Forcing, t-lift ---
    ax = axes[0, 1]
    ax.plot(t_train_l, F_star_train_l,
            color="black", linewidth=1.5, label="True F*")
    ax.plot(t_train_l, f_tlift_train_l,
            color="blue", linestyle="--", linewidth=1.5)
    ax.plot(t_test_l,  F_star_test_l,
            color="black", linewidth=1.5)
    ax.plot(t_test_l,  f_tlift_test_l,
            color="blue", linestyle="--", linewidth=1.5,
            label="t-lift pred F*")
    ax.axvline(x=t_split, color="gray", linestyle=":", linewidth=1.4,
               label="Train/test split")
    ax.set_title("Forcing: t-lift vs true")
    ax.set_xlabel("t")
    ax.set_ylabel("F*(t)")
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    # --- Bottom-left: Solution, normal ---
    ax = axes[1, 0]
    ax.plot(t_train_l, u_ref_train_l,
            color="black", linewidth=1.5, label="reference u(t)")
    ax.plot(t_train_l, u_norm_train_l,
            color="red", linestyle="--", linewidth=1.5)
    ax.plot(t_test_l,  u_ref_test_l,
            color="black", linewidth=1.5)
    ax.plot(t_test_l,  u_norm_test_l,
            color="red", linestyle="--", linewidth=1.5,
            label="normal pred u(t)")
    ax.axvline(x=t_split, color="gray", linestyle=":", linewidth=1.2,
               label="Train/test split")
    ax.set_title("Solution: normal vs reference")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    # --- Bottom-right: Solution, t-lift ---
    ax = axes[1, 1]
    ax.plot(t_train_l, u_ref_train_l,
            color="black", linewidth=1.5, label="reference u(t)")
    ax.plot(t_train_l, u_tlift_train_l,
            color="blue", linestyle="--", linewidth=1.5)
    ax.plot(t_test_l,  u_ref_test_l,
            color="black", linewidth=1.5)
    ax.plot(t_test_l,  u_tlift_test_l,
            color="blue", linestyle="--", linewidth=1.5,
            label="t-lift pred u(t)")
    ax.axvline(x=t_split, color="gray", linestyle=":", linewidth=1.2,
               label="Train/test split")
    ax.set_title("Solution: t-lift vs reference")
    ax.set_xlabel("t")
    ax.set_ylabel("u(t)")
    ax.legend(frameon=True, fancybox=False, edgecolor="black")

    fig.tight_layout()
    plt.show()


def print_normal_vs_tlift(
    t_vals: torch.Tensor,
    TRAIN_FRAC: float,
    u_pred_full: torch.Tensor,
    f_pred_full: torch.Tensor,
    u_pred_full_tlift: torch.Tensor,
    f_pred_full_tlift: torch.Tensor,
    U_ref: torch.Tensor,
    F_star: torch.Tensor,
):
    N        = t_vals.numel()
    N_train  = int(N * TRAIN_FRAC)
    idx_train = torch.arange(0, N_train)
    idx_test  = torch.arange(N_train, N)
    idx_all   = torch.arange(0, N)

    def rel_mse(pred, true):
        pred = pred.to(true.device)
        return torch.mean((pred - true) ** 2).item() / torch.mean(true ** 2).item()

    def pct_imp(nb, b):
        return (nb - b) / abs(nb) * 100 if nb != 0 else float("nan")

    rows = [
        ("Training forcing",   idx_train, f_pred_full, f_pred_full_tlift, F_star),
        ("Training solution",  idx_train, u_pred_full, u_pred_full_tlift, U_ref),
        ("Testing forcing",    idx_test,  f_pred_full, f_pred_full_tlift, F_star),
        ("Testing solution",   idx_test,  u_pred_full, u_pred_full_tlift, U_ref),
        ("Train+Test forcing", idx_all,   f_pred_full, f_pred_full_tlift, F_star),
        ("Train+Test solution",idx_all,   u_pred_full, u_pred_full_tlift, U_ref),
    ]

    print(f"\n{'':25s} {'Normal':>15} {'t-lift':>16} {'% Improvement':>14}")
    print("-" * 72)
    prev_section = "Training"
    for label, idx, pred_norm, pred_tlift, ref in rows:
        # Print separator between train/test/full sections
        section = label.split()[0]
        if section != prev_section:
            print("-" * 72)
            prev_section = section

        norm  = rel_mse(pred_norm[idx],  ref[idx])
        tlift = rel_mse(pred_tlift[idx], ref[idx])
        print(
            f"{label:25s} "
            f"{norm:>15.4e} {tlift:>16.4e} "
            f"{pct_imp(norm, tlift):>13.2f}%"
        )



## Compare: Predict-Retrain vs Rolling Retrain

def mse(pred, true):
    pred = pred.to(true.device)
    return torch.mean((pred - true) ** 2).item()


def rel_mse(pred, true):
    pred = pred.to(true.device)
    return (torch.mean((pred - true) ** 2) / torch.mean(true ** 2)).item()

