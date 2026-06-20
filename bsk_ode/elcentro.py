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

def cumtrapz_torch(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dx   = x[1:] - x[:-1]
    area = 0.5 * (y[1:] + y[:-1]) * dx
    out  = torch.zeros_like(y)
    out[1:] = torch.cumsum(area, dim=0)
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


def build_kernel_from_different_signatures(
    sigs_flat1: torch.Tensor,
    sigs_flat2: torch.Tensor
    ) -> torch.Tensor:
    """
    Build a cross-kernel matrix from two sets of signature features.

    sigs_flat1: (T1, D)
    sigs_flat2: (T2, D)
    Returns:
        Ker: (T1, T2)
    """
    Ker = sigs_flat1 @ sigs_flat2.T
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

def apply_signature_normalization_pair(
    sigs_train: torch.Tensor,
    sigs_full: torch.Tensor,
    depth: int,
    dim: int,
    **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalize both train and full signatures using training statistics only.
    Uses the same median/IQR scheme as normalize_signatures, but
    estimates med/iqr on sigs_train and applies them to both.

    Returns:
        (sigs_train_norm, sigs_full_norm)
    """
    eps = kwargs.get("eps", 1e-8)

    # Compute robust column-wise stats on the *training* signatures
    med = sigs_train.median(dim=0, keepdim=True).values
    q25 = sigs_train.quantile(0.25, dim=0, keepdim=True)
    q75 = sigs_train.quantile(0.75, dim=0, keepdim=True)
    iqr = q75 - q25

    sigs_train_norm = (sigs_train - med) / (iqr + eps)
    sigs_full_norm  = (sigs_full  - med) / (iqr + eps)

    return sigs_train_norm, sigs_full_norm
 


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
    dtype = torch.float64
    device = Ksig.device

    Ksig = Ksig.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)
    f = f.to(device=device, dtype=dtype)

    K0, IK, I2K = buildkerneloperators(Ksig, x)

    # Design/operator matrix
    A = k1 * K0 + k2 * IK + k3 * I2K

    # Target
    rhs = build_rhs(
        f, x, ua, upa, k1, k2
    ).to(device=device, dtype=dtype)

    # Ridge regression:
    # beta = argmin ||A beta - rhs||^2 + reg ||beta||^2
    lam = max(float(reg), 1e-2)

    p = A.shape[1]
    Ireg = torch.eye(p, dtype=dtype, device=device)

    lhs = A.T @ A + lam * Ireg
    rhs_ridge = A.T @ rhs

    beta = torch.linalg.solve(lhs, rhs_ridge)

    if not torch.isfinite(beta).all():
        raise ValueError(
            f"Non-finite beta after ridge solve. Increase reg; current reg={lam}"
        )

    u = K0 @ beta
    Iu = IK @ beta
    I2u = I2K @ beta

    rhs_pred = k1 * u + k2 * Iu + k3 * I2u

    return beta, u, rhs_pred, rhs


def evaluate_solution_from_beta(
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


def evaluate_forcing_from_solution(
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

def _matrix_stats(M, name, reg=0):
    dtype = torch.float64
    device = M.device
    M = M.to(dtype=dtype, device=device)

    try:
        s = torch.linalg.svdvals(M)
        smax = s.max()
        smin = s.min()

        cond = (smax / torch.clamp(smin, min=reg)).item()
        rcond = (torch.clamp(smin, min=reg) / torch.clamp(smax, min=reg)).item()
        rank = torch.linalg.matrix_rank(M).item()

    except RuntimeError:
        MtM = M.T @ M
        evals = torch.linalg.eigvalsh(MtM)
        evals = torch.clamp(evals, min=reg)
        s = torch.sqrt(evals)
        smax = s.max()
        smin = s.min()

        cond = (smax / torch.clamp(smin, min=reg)).item()
        rcond = (torch.clamp(smin, min=reg) / torch.clamp(smax, min=reg)).item()
        rank = torch.linalg.matrix_rank(M).item()

    print(f"{name:10s} | shape={tuple(M.shape)} | rank={rank:4d} | rcond={rcond:.3e} | cond={cond:.3e}")


def diagnose_conditioning(x, f,
                        k1, k2, k3,
                        ua, upa,
                        depth,
                        reg = 1e-10,
                        use_tlift=False,
                        holder_value=None):
    
    dtype = torch.float64
    device = torch.device("cuda")

    with torch.no_grad():

        #build path
        X = torch.stack([x, f], dim=1)           # (T,2)
        if use_tlift:
            if holder_value is None:
                raise ValueError("holder_value must be provided when use_tlift=True")
            X = tlift(X, holder_value)

        #non normalized
        X_sig = compute_signatures(X, depth)
        Ksig = build_kernel_from_signatures(X_sig)
        K0, IK, I2K = buildkerneloperators(Ksig, x)
        A = k1 * K0 + k2 * IK + k3 * I2K

        #normalized
        X_sig_norm = normalize_signatures(Z=X_sig,depth=depth,dim=X.shape[1])
        Ksig_norm = build_kernel_from_signatures(X_sig_norm)
        K0_norm, IK_norm, I2K_norm = buildkerneloperators(Ksig_norm, x)
        A_norm = k1 * K0_norm + k2 * IK_norm + k3 * I2K_norm

        #Diagnose and print conditioning number, rcond, and rank for each
        
        print("\n--- non-normalized ---")
        _matrix_stats(X_sig,   "X_sig")
        _matrix_stats(Ksig,    "Ksig")
        _matrix_stats(K0,      "K0")
        _matrix_stats(IK,      "IK")
        _matrix_stats(I2K,     "I2K")
        _matrix_stats(A,       "A")

        print("\n--- normalized ---")
        _matrix_stats(X_sig_norm, "X_sig_n")
        _matrix_stats(Ksig_norm,  "Ksig_n")
        _matrix_stats(K0_norm,    "K0_n")
        _matrix_stats(IK_norm,    "IK_n")
        _matrix_stats(I2K_norm,   "I2K_n")
        _matrix_stats(A_norm,     "A_n")

        return 



def solve_signature_kernel_calibration(x, f,
                        k1, k2, k3,
                        ua, upa,
                        depth,
                        normalize = True,
                        reg = 1e-10,
                        use_tlift=False,
                        holder_value=None):

    with torch.no_grad():

        X = torch.stack([x, f], dim=1)           # (T,2)

        if use_tlift:

            if holder_value is None:
                raise ValueError("holder_value must be provided when use_tlift=True")
            X = tlift(X, holder_value)

        X_sig = compute_signatures(X, depth)

        if(normalize == True):
            X_sig = normalize_signatures(Z=X_sig,depth=depth,dim=X.shape[1],)

        Ksig = build_kernel_from_signatures(X_sig)

        alpha, u, f_pred_final, rhs_true = solvebetas(
            Ksig=Ksig,
            f=f,x=x,
            ua=ua, upa=upa,
            k1=k1,k2=k2,k3=k3,
            reg = reg
        )


    return u, f_pred_final, alpha

def predict_signature_kernel(
    x_train, f_train,
    x_eval, f_eval,
    alpha,
    k1, k2, k3,
    ua, upa,
    depth,
    normalize=True,
    use_tlift=False,
    holder_value=None,
):
    """
    Predict u and f on x_eval using a model calibrated on (x_train, f_train).
    """

    with torch.no_grad():
        if use_tlift and holder_value is None:
            raise ValueError("holder_value must be provided when use_tlift=True")

        # Build training path
        X_train = torch.stack([x_train, f_train], dim=1)
        if use_tlift:
            X_train = tlift(X_train, holder_value)

        X_sig_train = compute_signatures(X_train, depth)
        if normalize:
            X_sig_train = normalize_signatures(
                Z=X_sig_train,
                depth=depth,
                dim=X_train.shape[1],
            )

        # Build evaluation path
        X_eval = torch.stack([x_eval, f_eval], dim=1)
        if use_tlift:
            X_eval = tlift(X_eval, holder_value)

        X_sig_eval = compute_signatures(X_eval, depth)
        if normalize:
            _, X_sig_eval = apply_signature_normalization_pair(
                X_sig_train,
                X_sig_eval,
                depth=depth,
                dim=X_train.shape[1],
            )

        # Cross-kernel operators
        Ksig_eval_train = build_kernel_from_different_signatures(
            X_sig_eval, X_sig_train
        )
        Ku2_eval, Kup_eval, Ku_eval = buildkerneloperators(
            Ksig_eval_train, x_eval
        )

        # Evaluate solution and forcing
        # Correct operator ordering: K0=u operator, IK=∫u operator, I2K=∫∫u operator
        u_eval, u_p_eval, u_dd_eval = evaluate_solution_from_beta(
            Ku_eval, Kup_eval, Ku2_eval, x_eval, alpha, ua, upa
        )
        f_eval_pred = evaluate_forcing_from_solution(
            u_eval, u_p_eval, u_dd_eval, k1, k2, k3
        )

    return u_eval, f_eval_pred

def solve_signature_kernel_predict_retrain(
    t_train: torch.Tensor,
    t_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth,
    normalize=True,
    reg=1e-10,
    retrain_every: int = 10,
    use_tlift=False,
    holder_value=None,
):
    """
    Non-branched testing with periodic retraining.
    """

    if use_tlift and holder_value is None:
        raise ValueError("holder_value must be provided when use_tlift=True")

    with torch.no_grad():
        u_pred_train, f_pred_train, alpha = solve_signature_kernel_calibration(
            x=t_train,
            f=f_train,
            k1=k1, k2=k2, k3=k3,
            ua=ua, upa=upa,
            depth=depth,
            normalize=normalize,
            reg=reg,
            use_tlift=use_tlift,
            holder_value=holder_value,
        )

        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        # Current fitted set starts as the original train set
        t_fit = t_train.clone()
        f_fit = f_train.clone()

        N_train = t_train.numel()
        N_test = t_test.numel()

        for j in range(1, N_test + 1):
            if (j % retrain_every) == 0:
                t_fit = torch.cat([t_train, t_test[:j]], dim=0)
                f_fit = torch.cat([f_train, f_test[:j]], dim=0)

                _, _, alpha = solve_signature_kernel_calibration(
                    x=t_fit,
                    f=f_fit,
                    k1=k1, k2=k2, k3=k3,
                    ua=ua, upa=upa,
                    depth=depth,
                    normalize=normalize,
                    reg=reg,
                    use_tlift=use_tlift,
                    holder_value=holder_value,
                )

            x_curr = torch.cat([t_train, t_test[:j]], dim=0)
            f_curr = torch.cat([f_train, f_test[:j]], dim=0)

            u_curr, f_curr_pred = predict_signature_kernel(
                x_train=t_fit,
                f_train=f_fit,
                x_eval=x_curr,
                f_eval=f_curr,
                alpha=alpha,
                k1=k1, k2=k2, k3=k3,
                ua=ua, upa=upa,
                depth=depth,
                normalize=normalize,
                use_tlift=use_tlift,
                holder_value=holder_value,
            )

            u_pred_full = torch.cat(
                [u_pred_full, u_curr[N_train + j - 1:N_train + j]], dim=0
            )
            f_pred_full = torch.cat(
                [f_pred_full, f_curr_pred[N_train + j - 1:N_train + j]], dim=0
            )

    t_all = torch.cat([t_train, t_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, last beta): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full





# -------------------------------------------------------
# Signature + normalization helpers (rolling version)
# -------------------------------------------------------

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


def robust_fit(S: torch.Tensor, eps: float = 1e-5):
    """Return (median, IQR+eps) column-wise over S (N, D)."""
    q75 = torch.quantile(S.double(), 0.75, dim=0)
    q25 = torch.quantile(S.double(), 0.25, dim=0)
    iqr = q75 - q25
    N   = S.shape[0]
    S_sorted = torch.sort(S, dim=0).values
    med = S_sorted[N // 2] if N % 2 == 1 else (S_sorted[N // 2 - 1] + S_sorted[N // 2]) / 2.0
    return med, iqr + eps


def robust_apply(x: torch.Tensor, med: torch.Tensor, iqr: torch.Tensor) -> torch.Tensor:
    return (x - med) / iqr


# -------------------------------------------------------
# old Rolling online prediction functions
# -------------------------------------------------------
'''
def build_paths(F_t: pd.DataFrame,
                num_partitions: int,
                t_lift_exp: float = 0.3):
    """
    Build prefix paths for branched (t-lift) and non-branched variants.

    Returns:
        paths_branched    : list of (T_i, 3) tensors  [t, f, t^exp]
        paths_nonbranched : list of (T_i, 2) tensors  [t, f]
    """
    total_points   = len(F_t)
    partition_size = total_points // num_partitions

    paths_branched    = []
    paths_nonbranched = []

    for i in range(num_partitions):
        end_index = (i + 1) * partition_size
        if i == num_partitions - 1:
            end_index = total_points

        path_b  = []
        path_nb = []
        for j in range(end_index):
            t_val = F_t.iloc[j, 0]
            f_val = F_t.iloc[j, 1]
            path_b.append([t_val, f_val, t_val ** t_lift_exp])
            path_nb.append([t_val, f_val])

        paths_branched.append(torch.tensor(path_b,  dtype=torch.float64))
        paths_nonbranched.append(torch.tensor(path_nb, dtype=torch.float64))

    return paths_branched, paths_nonbranched


def build_state(paths,
                n0: int,
                signature_level: int,
                m: float, c: float, k: float,
                dt: float, N: int,
                F_star: torch.Tensor,
                t_vals: torch.Tensor,
                u_true_interp: torch.Tensor) -> dict:
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

    med, iqr = robust_fit(S0_raw)
    S0 = robust_apply(S0_raw, med, iqr)

    K0   = S0 @ S0.T                    # (n0+1, n0+1), on device
    K1_0 = trapezoidal_cols(K0, dt)
    K2_0 = trapezoidal_cols(K1_0, dt)

    Psi0 = m * K0 + c * K1_0 + k * K2_0

    rcond = torch.finfo(torch.float64).eps
    alpha0 = torch.linalg.lstsq(
        Psi0, F_star[:n0 + 1], rcond=rcond, driver="gelsd"
    ).solution

    F_pred_train = Psi0 @ alpha0
    u_pred_train = K0  @ alpha0

    return {
        "m": m, "c": c, "k": k,
        "dt": dt, "n0": n0, "N": N,
        "paths": paths,                     # lists of tensors; signature_of_path handles device
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
                           max_steps: int | None = None) -> dict:
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

    end_idx = N - 1 if max_steps is None else min(N - 1, n0 + max_steps)

    S_hist = state["S_hist"].clone()
    alphas = torch.zeros(end_idx + 1, dtype=torch.float64)
    alphas[:n0 + 1] = state["alpha0"]

    K_prev = state["K_prev"].clone()
    I1 = state["I1"].clone()
    I2 = state["I2"].clone()

    F_pred = torch.zeros(end_idx + 1, dtype=torch.float64)
    u_pred = torch.zeros(end_idx + 1, dtype=torch.float64)

    F_pred[:n0 + 1] = state["F_pred_train"]
    u_pred[:n0 + 1] = state["u_pred_train"]

    retrain_indices = []
    eps = 1e-3

    for i in range(n0 + 1, end_idx + 1):
        s_raw = signature_of_path(paths[i], depth=depth)
        s_new = robust_apply(s_raw, med, iqr)

        k_row_old = S_hist @ s_new
        k_ii      = float(torch.dot(s_new, s_new).item())

        I1_new = I1 + 0.5 * (K_prev + k_row_old) * dt
        I2_new = I2 + 0.5 * (I1 + I1_new) * dt
        I1, I2 = I1_new, I2_new
        K_prev = k_row_old

        col_i   = torch.cat([k_row_old, torch.tensor([k_ii], dtype=torch.float64)])
        inner_i = trapezoidal_cols(col_i, dt)
        outer_i = trapezoidal_cols(inner_i, dt)

        I1 = torch.cat([I1, torch.tensor([float(inner_i[-1])], dtype=torch.float64)])
        I2 = torch.cat([I2, torch.tensor([float(outer_i[-1])], dtype=torch.float64)])
        K_prev = torch.cat([K_prev, torch.tensor([k_ii], dtype=torch.float64)])

        psi_row_old = m * k_row_old + c * I1[:i] + k * I2[:i]
        psi_diag    = m * k_ii + c * float(I1[i]) + k * float(I2[i])

        residual = F_star[i] - torch.dot(psi_row_old, alphas[:i])
        alphas[i] = residual / (psi_diag + eps)

        F_pred[i] = torch.dot(psi_row_old, alphas[:i]) + psi_diag * alphas[i]
        u_pred[i] = torch.dot(k_row_old,   alphas[:i]) + k_ii      * alphas[i]

        S_hist = torch.vstack([S_hist, s_new.unsqueeze(0)])

        if (i - n0) % retrain_every == 0:
            print(f"[Retrain] at index {i}")
            retrain_indices.append(i)

            K      = S_hist @ S_hist.T
            K1     = trapezoidal_cols(K, dt)
            K2     = trapezoidal_cols(K1, dt)
            Psi    = m * K + c * K1 + k * K2
            Psi_bl = Psi[:i + 1, :i + 1]
            F_bl   = F_star[:i + 1]

            lam    = 1e-13 * torch.mean(torch.diag(Psi_bl))
            I_mat  = torch.eye(i + 1, dtype=torch.float64)
            alphas[:i + 1] = torch.linalg.solve(Psi_bl + lam * I_mat, F_bl)

            F_pred[:i + 1] = Psi_bl @ alphas[:i + 1]
            u_pred[:i + 1] = K[:i + 1, :i + 1] @ alphas[:i + 1]

    return {
        "F_pred": F_pred,
        "u_pred": u_pred,
        "retrain_indices": retrain_indices,
        "end_idx": end_idx,
    }
'''

def rolling_update_step(
    beta_prev: torch.Tensor,
    X_sig_train: torch.Tensor,
    t_curr: torch.Tensor,
    f_curr: torch.Tensor,
    k1: float,
    k2: float,
    k3: float,
    ua: float,
    upa: float,
    depth: int,
    normalize: bool = True,
    use_tlift: bool = False,
    holder_value=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform one rolling update step (no full retrain).

    Returns:
        u_new    : predicted u at t_curr[-1]
        f_new    : predicted forcing at t_curr[-1]
        beta_new : updated coefficient vector including alpha_new
    """
    if use_tlift and holder_value is None:
        raise ValueError("holder_value must be provided when use_tlift=True")

    # Build current path and signatures
    X_curr = torch.stack([t_curr, f_curr], dim=1)

    if use_tlift:
        X_curr = tlift(X_curr, holder_value)

    X_sig_cur = compute_signatures(X_curr, depth)

    if normalize:
        _, X_sig_cur = apply_signature_normalization_pair(
            X_sig_train,
            X_sig_cur,
            depth=depth,
            dim=X_curr.shape[1],
        )

    # Full kernel on current points
    Ksig_full = build_kernel_from_signatures(X_sig_cur)
    K0, I1K, I2K = buildkerneloperators(Ksig_full, t_curr)

    # Row for the new point vs all previous points
    phi_K0_row = K0[-1, :-1]
    I1_row = I1K[-1, :-1]
    I2_row = I2K[-1, :-1]

    # Diagonal entries for the new point
    k_diag = K0[-1, -1]
    I1_diag = I1K[-1, -1]
    I2_diag = I2K[-1, -1]

    # Operator row/diag at the new point
    phi_row_old = k1 * phi_K0_row + k2 * I1_row + k3 * I2_row
    phi_diag = k1 * k_diag + k2 * I1_diag + k3 * I2_diag

    # One-step residual update at the new point
    eps_denom = 1e-12
    target_new = f_curr[-1]
    alpha_new = (target_new - torch.dot(phi_row_old, beta_prev)) / (phi_diag + eps_denom)

    # Extend coefficients
    beta_new = torch.cat([beta_prev, alpha_new.unsqueeze(0)], dim=0)

    # Evaluate full solution/forcing and take last entries
    u_all, Iu, I2u = evaluate_solution_from_beta(
        K0, I1K, I2K, t_curr, beta_new, ua, upa
    )
    f_pred_all = evaluate_forcing_from_solution(u_all, Iu, I2u, k1, k2, k3)

    u_new = u_all[-1]
    f_new = f_pred_all[-1]

    return u_new, f_new, beta_new

def solve_signature_kernel_rolling_retrain(
    t_train: torch.Tensor,
    t_test: torch.Tensor,
    f_train: torch.Tensor,
    f_test: torch.Tensor,
    k1, k2, k3,
    ua, upa,
    depth,
    normalize: bool = True,
    reg: float = 1e-10,
    retrain_every: int = 10,
    use_tlift: bool = False,
    holder_value=None,
):
    """
    Non-branched rolling testing with periodic retraining.
    """

    if use_tlift and holder_value is None:
        raise ValueError("holder_value must be provided when use_tlift=True")

    with torch.no_grad():
        # Initial fit
        u_pred_train, f_pred_train, beta = solve_signature_kernel_calibration(
            x=t_train,
            f=f_train,
            k1=k1, k2=k2, k3=k3,
            ua=ua, upa=upa,
            depth=depth,
            normalize=normalize,
            reg=reg,
            use_tlift=use_tlift,
            holder_value=holder_value,
        )

        u_pred_full = u_pred_train.clone()
        f_pred_full = f_pred_train.clone()

        # Current fitted set
        t_fit = t_train.clone()
        f_fit = f_train.clone()

        X_fit = torch.stack([t_fit, f_fit], dim=1)
        if use_tlift:
            X_fit = tlift(X_fit, holder_value)

        X_sig_train = compute_signatures(X_fit, depth)
        if normalize:
            X_sig_train = normalize_signatures(
                Z=X_sig_train,
                depth=depth,
                dim=X_fit.shape[1],
            )

        N_test = t_test.numel()

        for j in range(1, N_test + 1):
            # Full retrain step
            if (j % retrain_every) == 0:
                t_fit = torch.cat([t_train, t_test[:j]], dim=0)
                f_fit = torch.cat([f_train, f_test[:j]], dim=0)

                u_fit, f_fit_pred, beta = solve_signature_kernel_calibration(
                    x=t_fit,
                    f=f_fit,
                    k1=k1, k2=k2, k3=k3,
                    ua=ua, upa=upa,
                    depth=depth,
                    normalize=normalize,
                    reg=reg,
                    use_tlift=use_tlift,
                    holder_value=holder_value,
                )

                X_fit = torch.stack([t_fit, f_fit], dim=1)
                if use_tlift:
                    X_fit = tlift(X_fit, holder_value)

                X_sig_train = compute_signatures(X_fit, depth)
                if normalize:
                    X_sig_train = normalize_signatures(
                        Z=X_sig_train,
                        depth=depth,
                        dim=X_fit.shape[1],
                    )

                # calibration already includes the newest point
                u_new = u_fit[-1]
                f_new = f_fit_pred[-1]

            else:
                # one-step extension beyond current fitted set
                t_curr = torch.cat([t_fit, t_test[j-1:j]], dim=0)
                f_curr = torch.cat([f_fit, f_test[j-1:j]], dim=0)

                u_new, f_new, beta = rolling_update_step(
                    beta_prev=beta,
                    X_sig_train=X_sig_train,
                    t_curr=t_curr,
                    f_curr=f_curr,
                    k1=k1,
                    k2=k2,
                    k3=k3,
                    ua=ua,
                    upa=upa,
                    depth=depth,
                    normalize=normalize,
                    use_tlift=use_tlift,
                    holder_value=holder_value,
                )

                # rolling state advances by one point
                t_fit = t_curr
                f_fit = f_curr

            u_pred_full = torch.cat([u_pred_full, u_new.unsqueeze(0)], dim=0)
            f_pred_full = torch.cat([f_pred_full, f_new.unsqueeze(0)], dim=0)

    t_all = torch.cat([t_train, t_test], dim=0)
    f_all = torch.cat([f_train, f_test], dim=0)
    final_loss = forcing_loss(f_all, f_pred_full)
    print(f"final forcing loss (train+test, rolling): {final_loss.item():.3e}")

    return u_pred_full, f_pred_full


def mse(pred, true):
    pred = pred.to(true.device)
    return torch.mean((pred - true) ** 2).item()

def rel_mse(pred, true):
    pred = pred.to(true.device)
    return torch.mean((pred - true) ** 2).item() / torch.mean(true ** 2).item()



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

def rel_mse(pred, true):
    pred = pred.to(true.device)
    return (torch.mean((pred - true) ** 2) / torch.mean(true ** 2)).item()


def print_method_comparison(t_vals, TRAIN_FRAC,
                             f_pred_full, u_pred_full,
                             f_pred_full_tlift, u_pred_full_tlift,
                             f_pred_full_rolling, u_pred_full_rolling,
                             f_pred_full_tlift_rolling, u_pred_full_tlift_rolling,
                             F_star, U_ref):
    N_train  = int(len(t_vals) * TRAIN_FRAC)
    idx_all   = torch.arange(0, len(t_vals))
    idx_train = torch.arange(0, N_train)
    idx_test  = torch.arange(N_train, len(t_vals))

    methods = [
        ("Predict-Retrain (normal)",    f_pred_full,               u_pred_full),
        ("Predict-Retrain (t-lift)",    f_pred_full_tlift,         u_pred_full_tlift),
        ("Rolling-Retrain (normal)",    f_pred_full_rolling,       u_pred_full_rolling),
        ("Rolling-Retrain (t-lift)",    f_pred_full_tlift_rolling, u_pred_full_tlift_rolling),
    ]

    header = f"{'Method':<35} {'Split':<8} {'Forcing Rel MSE':>16} {'Solution Rel MSE':>18}"
    print("=" * len(header))
    print(header)

    prev = None
    for label, f_pred, u_pred in methods:
        print("-" * len(header))
        for split_name, idx in [("Train", idx_train), ("Test", idx_test), ("All", idx_all)]:
            f_err = rel_mse(f_pred[idx], F_star[idx])
            u_err = rel_mse(u_pred[idx], U_ref[idx])
            row_label = label if split_name == "Train" else ""
            print(f"{row_label:<35} {split_name:<8} {f_err:>16.4e} {u_err:>18.4e}")

    print("=" * len(header))


def plot_method_comparison(t_vals, TRAIN_FRAC,
                           f_pred_full, u_pred_full,
                           f_pred_full_tlift, u_pred_full_tlift,
                           f_pred_full_rolling, u_pred_full_rolling,
                           f_pred_full_tlift_rolling, u_pred_full_tlift_rolling,
                           F_star, U_ref):
    N       = len(t_vals)
    N_train = int(N * TRAIN_FRAC)
    t_split = float(t_vals[N_train - 1].item())

    def to_list(x):
        return x.detach().cpu().tolist() if isinstance(x, torch.Tensor) else list(x)

    t_list  = to_list(t_vals)
    fs_list = to_list(F_star)
    ur_list = to_list(U_ref)

    methods = [
        ("Predict-Retrain\n(normal)",  f_pred_full,               u_pred_full),
        ("Predict-Retrain\n(t-lift)",  f_pred_full_tlift,         u_pred_full_tlift),
        ("Rolling-Retrain\n(normal)",  f_pred_full_rolling,       u_pred_full_rolling),
        ("Rolling-Retrain\n(t-lift)",  f_pred_full_tlift_rolling, u_pred_full_tlift_rolling),
    ]

    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 9,  "figure.dpi": 130,
    })

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey="row")

    for col, (label, f_pred, u_pred) in enumerate(methods):
        f_list = to_list(f_pred)
        u_list = to_list(u_pred)

        # --- Row 0: Forcing ---
        ax = axes[0, col]
        ax.plot(t_list, fs_list, color="black",  linewidth=1.2, label="Reference")
        ax.plot(t_list, f_list,  color="tab:blue", linewidth=1.0, linestyle="--", label="Predicted")
        ax.axvline(t_split, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("t")
        if col == 0:
            ax.set_ylabel("F(t)")
            ax.legend(frameon=True, fancybox=False, edgecolor="black")

        # --- Row 1: Solution ---
        ax = axes[1, col]
        ax.plot(t_list, ur_list, color="black",   linewidth=1.2, label="Reference")
        ax.plot(t_list, u_list,  color="tab:red",  linewidth=1.0, linestyle="--", label="Predicted")
        ax.axvline(t_split, color="gray", linestyle=":", linewidth=1.0, label="Train/test split")
        ax.set_xlabel("t")
        if col == 0:
            ax.set_ylabel("u(t)")
            ax.legend(frameon=True, fancybox=False, edgecolor="black")

    fig.suptitle("Method Comparison — Forcing (top) and Solution (bottom)", fontsize=13, y=1.01)
    fig.tight_layout()
    plt.show()


