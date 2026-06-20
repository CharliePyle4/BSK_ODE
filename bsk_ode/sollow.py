import os
os.environ["KERAS_BACKEND"] = "torch"

import torch
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import keras_sig
from scipy.integrate import solve_ivp



# Cell 3 - seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_default_dtype(torch.float64)

def solve_solow_ivp(
    t_grid:        torch.Tensor,
    F_torch:       torch.Tensor,
    y0:            float = 3.1,
    lambda_econ:   float = 0.05,
    forcing_scale: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Solve the first-order Solow ODE using SciPy (BDF method).

        dy/dt + lambda_econ * y(t) = forcing_scale * F(t),   y(0) = y0

    rewritten as:
        dy/dt = forcing_scale * F(t) - lambda_econ * y(t)

    Parameters
    ----------
    t_grid        : 1-D torch.Tensor, normalised time grid
    F_torch       : 1-D torch.Tensor, raw GDP forcing F(t) on t_grid
    y0            : initial condition y(0)
    lambda_econ   : decay coefficient (default 0.05)
    forcing_scale : multiplier on F(t) (default 0.2)

    Returns
    -------
    t_out : torch.Tensor  (same grid as t_grid)
    y_out : torch.Tensor  solution y(t)
    """
    t_eval = t_grid.cpu().numpy()
    F_np   = F_torch.cpu().numpy()

    def F_interp(t):
        return np.interp(t, t_eval, F_np)

    def fun(t, y):
        dydt = forcing_scale * F_interp(t) - lambda_econ * y[0]
        return [dydt]

    sol = solve_ivp(
        fun,
        (t_eval[0], t_eval[-1]),
        [y0],
        t_eval   = t_eval,
        method   = "BDF",
        rtol     = 1e-6,
        atol     = 1e-9,
        max_step = float(t_eval[1] - t_eval[0]) * 2,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    return (
        torch.tensor(t_eval,    dtype=torch.float64),
        torch.tensor(sol.y[0],  dtype=torch.float64),
    )

def trapezoidal_array(F_vals, dt):
    integrated = [0.0]
    total = 0.0
    for i in range(1, len(F_vals)):
        trap = (F_vals[i - 1] + F_vals[i]) / 2
        total += trap * dt
        integrated.append(total)
    return integrated


def trapezoidal_cols(M, dt):
    trap = (M[:-1] + M[1:]) / 2 * dt
    out = torch.zeros_like(M)
    out[1:] = torch.cumsum(trap, dim=0)
    return out




def signature_of_path(path, depth: int = 3) -> torch.Tensor:
    """
    Prefix signature of a single path (T, d) using keras_sig.
    """
    # Ensure tensor
    if not isinstance(path, torch.Tensor):
        path = torch.tensor(path, dtype=torch.float64)

    # Make shape (1, T, d)
    if path.dim() == 2:
        path = path.unsqueeze(0)

    # Prepend basepoint as in your other code
    basepoint = path[:, 0:1, :]               # (1, 1, d)
    path_bp   = torch.cat([basepoint, path], dim=1)  # (1, T+1, d)

    # Single signature per path → stream=False
    sig_raw = keras_sig.signature(
        path_bp,
        depth         = depth,
        stream        = False,   # single sig for the whole path
        gpu_optimized = True,
    )

    # (1, D) → (D,)
    return sig_raw.squeeze(0).detach()

def interpolate_to_grid(t_source, y_source, t_target):
    idx = torch.searchsorted(t_source.contiguous(), t_target.contiguous())
    idx = idx.clamp(1, len(t_source) - 1)
    t0 = t_source[idx - 1]
    t1 = t_source[idx]
    y0 = y_source[idx - 1]
    y1 = y_source[idx]
    weight = (t_target - t0) / (t1 - t0)
    return y0 + weight * (y1 - y0)


def set_professional_plot_style():
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


def format_axes(ax):
    ax.grid(True, which="major", alpha=0.35, linewidth=0.8)
    ax.grid(True, which="minor", alpha=0.18, linewidth=0.5)
    ax.minorticks_on()
    ax.legend(frameon=True, fancybox=False, edgecolor="black", loc="best")
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def rel_mse_np(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    return np.mean((pred - true) ** 2) / np.mean(true ** 2)

def pct_imp(nb, b):
    return (nb - b) / abs(nb) * 100 if nb != 0 else np.nan


def build_state_nonbranched(paths_nb, n0, signature_level, lambda_econ, dt, t_vals,
                            F_star_torch, y_true_interp):
    S0_raw = torch.stack([
        signature_of_path(paths_nb[i], depth=signature_level)
        for i in range(n0 + 1)
    ])

    K0 = S0_raw @ S0_raw.T
    K1_0 = trapezoidal_cols(K0, dt)

    Psi0 = K0 + lambda_econ * K1_0

    rcond = torch.finfo(torch.float64).eps
    alpha0 = torch.linalg.lstsq(
        Psi0,
        F_star_torch[:n0 + 1],
        rcond=rcond,
        driver="gelsd"
    ).solution

    F_pred_train = Psi0 @ alpha0
    y_pred_train = K0 @ alpha0

    return {
        "lambda_econ": lambda_econ,
        "dt": dt,
        "n0": n0,
        "N": len(t_vals),
        "paths": paths_nb,
        "signature_level": signature_level,
        "F_star": F_star_torch,
        "t_vals": t_vals,
        "y_true_interp": y_true_interp,
        "alpha0": alpha0,
        "S_hist": S0_raw.clone(),
        "K_prev": K0[n0, :].clone(),
        "I1": K1_0[n0, :].clone(),
        "F_pred_train": F_pred_train,
        "y_pred_train": y_pred_train,
    }

# Rolling Online Prediction — non-branched 

def rolling_online_predict_econ_nonbranched(state, retrain_every=5, max_steps=None):
    lambda_econ = state["lambda_econ"]
    dt = state["dt"]
    n0 = state["n0"]
    N = state["N"]
    paths = state["paths"]
    depth = state["signature_level"]
    F_star = state["F_star"]

    if max_steps is None:
        end_idx = N - 1
    else:
        end_idx = min(N - 1, n0 + max_steps)

    S_hist = state["S_hist"].clone()

    alphas = torch.zeros(end_idx + 1, dtype=torch.float64)
    alphas[:n0 + 1] = state["alpha0"]

    K_prev = state["K_prev"].clone()
    I1 = state["I1"].clone()

    F_pred = torch.zeros(end_idx + 1, dtype=torch.float64)
    y_pred = torch.zeros(end_idx + 1, dtype=torch.float64)

    F_pred[:n0 + 1] = state["F_pred_train"]
    y_pred[:n0 + 1] = state["y_pred_train"]

    retrain_indices = []

    eps = 1e-3

    for i in range(n0 + 1, end_idx + 1):

        s_new = signature_of_path(paths[i], depth=depth)

        k_row_old = S_hist @ s_new
        k_ii = torch.dot(s_new, s_new)

        I1_new = I1 + 0.5 * (K_prev + k_row_old) * dt
        I1 = I1_new
        K_prev = k_row_old

        col_i = torch.cat([
            k_row_old,
            torch.tensor([k_ii.item()], dtype=torch.float64)
        ])

        inner_i = trapezoidal_cols(col_i, dt)
        I1_i = inner_i[-1]

        I1 = torch.cat([
            I1,
            torch.tensor([I1_i.item()], dtype=torch.float64)
        ])

        K_prev = torch.cat([
            K_prev,
            torch.tensor([k_ii.item()], dtype=torch.float64)
        ])

        psi_row_old = k_row_old + lambda_econ * I1[:i]
        psi_diag = k_ii + lambda_econ * I1[i]

        residual = F_star[i] - torch.dot(psi_row_old, alphas[:i])
        alphas[i] = residual / (psi_diag + eps)

        F_pred[i] = torch.dot(psi_row_old, alphas[:i]) + psi_diag * alphas[i]
        y_pred[i] = torch.dot(k_row_old, alphas[:i]) + k_ii * alphas[i]

        S_hist = torch.vstack([S_hist, s_new.unsqueeze(0)])

        if (i - n0) % retrain_every == 0:
            print(f"[Retrain] at index {i}")
            retrain_indices.append(i)

            K = S_hist @ S_hist.T
            K1 = trapezoidal_cols(K, dt)
            Psi = K + lambda_econ * K1

            Psi_block = Psi[:i + 1, :i + 1]
            F_block = F_star[:i + 1]

            scale = torch.mean(torch.diag(Psi_block))
            lam = 1e-13 * scale
            I_mat = torch.eye(i + 1, dtype=torch.float64)

            alphas[:i + 1] = torch.linalg.solve(
                Psi_block + lam * I_mat,
                F_block
            )

            F_pred[:i + 1] = Psi_block @ alphas[:i + 1]
            y_pred[:i + 1] = K @ alphas[:i + 1]

    return {
        "F_pred": F_pred,
        "y_pred": y_pred,
        "alphas": alphas,
        "retrain_indices": retrain_indices,
        "end_idx": end_idx,
    }

def mse_torch(pred, true):
    return torch.mean((pred - true) ** 2).item()


def rel_mse_torch(pred, true):
    return torch.mean((pred - true) ** 2).item() / torch.mean(true ** 2).item()

def rel_mse_torch(pred, true):
    """
    Relative MSE:
        mean((pred - true)^2) / mean(true^2)
    """
    return torch.mean((pred - true) ** 2).item() / torch.mean(true ** 2).item()

def mse_np(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)
    return float(np.mean((pred - true) ** 2))

def rel_mse_np(pred: np.ndarray, true: np.ndarray) -> float:
    pred = np.asarray(pred)
    true = np.asarray(true)
    num  = np.mean((pred - true) ** 2)
    den  = np.mean(true ** 2)
    return float(num / den)

def build_paths_nonbranched(
    forcing_data: pd.DataFrame,
    t_lift_exp: float = 1.0,
) -> list:
    """
    Build prefix paths [t^exp, f] for the non-branched variant.
    Each paths_nb[i] is a list of i+1 points, i.e. the prefix ending at row i.
    """
    total_points    = len(forcing_data)
    num_partitions  = total_points           # one partition per time step (prefix paths)
    partition_size  = 1

    paths_nb = []
    for i in range(num_partitions):
        end_index = (i + 1) * partition_size
        if i == num_partitions - 1:
            end_index = total_points
        path_nb = []
        for j in range(end_index):
            t_val = forcing_data.iloc[j, 0]
            f_val = forcing_data.iloc[j, 1]
            path_nb.append([float(np.power(t_val, t_lift_exp)), float(f_val)])
        paths_nb.append(path_nb)

    return paths_nb

def run_full_batch(
    paths_nb:        list,
    F_star:          np.ndarray,
    dt:              float,
    signature_level: int = 3,
    lambda_econ:     float = 0.05,
) -> dict:
    """
    Full-batch (non-rolling) signature-kernel calibration.
    Returns F_hat (forcing fit) and K_A (solution reconstruction).
    """
    path_tensor = torch.tensor(
        np.array(paths_nb[-1]), dtype=torch.float64
    ).unsqueeze(0)

    basepoint   = path_tensor[:, 0:1, :]
    path_bp     = torch.cat([basepoint, path_tensor], dim=1)

    sigs = keras_sig.signature(
        path_bp,
        depth        = signature_level,
        stream       = True,
        gpu_optimized = True,
    ).squeeze(0).detach().cpu().numpy()

    S   = sigs
    N_full = len(S)

    K_nb  = np.array([[np.dot(S[i], S[j]) for j in range(N_full)]
                       for i in range(N_full)], dtype=np.float64)

    K1_nb = np.zeros((N_full, N_full), dtype=np.float64)
    for j in range(N_full):
        integrated = trapezoidal_array(K_nb[:, j], dt)
        K1_nb[:, j] = integrated

    Psi_nb = K_nb + lambda_econ * K1_nb    # note: pass lambda_econ as param (see below)
    A_nb   = np.linalg.lstsq(Psi_nb, F_star, rcond=None)[0]
    F_hat  = Psi_nb @ A_nb
    K_A    = K_nb   @ A_nb

    return {"F_hat": F_hat, "K_A": K_A, "A": A_nb, "K": K_nb, "K1": K1_nb}


def print_error_summary(
    label:        str,
    F_pred,       # numpy or torch
    F_true,       # numpy or torch
    y_pred,       # numpy or torch
    y_true,       # numpy or torch
    idx_train,
    idx_test,
    idx_full,
    use_torch: bool = True,
):
    """
    Print a 3-row (train / test / full) MSE table for forcing and solution.
    """
    mse_fn     = mse_torch     if use_torch else mse_np
    rel_mse_fn = rel_mse_torch if use_torch else rel_mse_np

    rows = [
        ("Training forcing",  idx_train, F_pred, F_true),
        ("Training solution", idx_train, y_pred, y_true),
        ("Testing forcing",   idx_test,  F_pred, F_true),
        ("Testing solution",  idx_test,  y_pred, y_true),
        ("Full forcing",      idx_full,  F_pred, F_true),
        ("Full solution",     idx_full,  y_pred, y_true),
    ]

    print(f"\n{'=' * 86}")
    print(f"{label} Error Summary")
    print(f"{'=' * 86}")
    print(f"{'Quantity':25s} {'MSE':>18s} {'Relative MSE':>18s} {'Relative MSE (%)':>20s}")
    print(f"{'-' * 86}")

    prev_section = None
    for name, idx, pred, true in rows:
        section = name.split()[0]
        if prev_section is not None and section != prev_section:
            print(f"{'-' * 86}")
        prev_section = section
        mse_val     = mse_fn(pred[idx], true[idx])
        rel_mse_val = rel_mse_fn(pred[idx], true[idx])
        print(f"{name:25s} {mse_val:>18.6e} {rel_mse_val:>18.6e} {100 * rel_mse_val:>19.4f}%")

    print(f"{'=' * 86}")

def plot_rolling_results(
    t_vals:          torch.Tensor,
    n0:              int,
    F_star_torch:    torch.Tensor,
    y_true_interp:   torch.Tensor,
    res:             dict,
    title_prefix:    str = "",
):
    """
    1x2 plot: forcing fit (left) and solution fit (right).
    Draws train region, test region, train/test split line,
    and faint retrain markers.
    """
    idx_train = torch.arange(0, n0 + 1)
    idx_test  = torch.arange(n0 + 1, res["end_idx"] + 1)
    t_train   = t_vals[idx_train]
    t_test    = t_vals[idx_test]
    t_split   = float(t_vals[n0].item())

    def _to_list(t):
        return t.detach().cpu().tolist() if isinstance(t, torch.Tensor) else list(t)

    set_professional_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)

    for ax, (true_series, pred_series, ylabel, title, pred_color) in zip(axes, [
        (F_star_torch,  res["F_pred"], "f(t)",  "Forcing Fit",  "red"),
        (y_true_interp, res["y_pred"], "y(t)",  "Solution Fit", "blue"),
    ]):
        # True
        ax.plot(_to_list(t_train), _to_list(true_series[idx_train]),
                color="black", linewidth=1.6, label="True")
        ax.plot(_to_list(t_test),  _to_list(true_series[idx_test]),
                color="black", linewidth=1.6)
        # Predicted
        ax.plot(_to_list(t_train), _to_list(pred_series[idx_train]),
                color=pred_color, linestyle="--", linewidth=1.6, label="Predicted")
        ax.plot(_to_list(t_test),  _to_list(pred_series[idx_test]),
                color=pred_color, linestyle="--", linewidth=1.6)
        # Split + retrain markers
        ax.axvline(t_split, color="gray", linestyle=":", linewidth=1.5, label="Train/test split")
        for r_idx in res.get("retrain_indices", []):
            ax.axvline(float(t_vals[r_idx].item()), color="gray",
                       linestyle=":", linewidth=0.7, alpha=0.25)
        ax.set_title(f"{title_prefix}{title}")
        ax.set_xlabel("Normalized time t")
        ax.set_ylabel(ylabel)
        format_axes(ax)

    fig.tight_layout()
    plt.show()



def plot_calibration(time, F_star, F_hat, U_true, U_hat):
    """
    1x2 plot:
      left: integrated forcing (true vs calibrated)
      right: solution y(t) (true vs reconstructed)
    """
    # Convert to numpy for plotting
    t_np      = time.detach().cpu().numpy() if isinstance(time, torch.Tensor) else np.asarray(time)
    F_star_np = np.asarray(F_star)
    F_hat_np  = np.asarray(F_hat)
    U_true_np = np.asarray(U_true)
    U_hat_np  = np.asarray(U_hat)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: forcing
    axes[0].plot(t_np, F_star_np, color="black", linewidth=1.5, label="Reference ∫0.2F")
    axes[0].plot(t_np, F_hat_np,  color="blue",  linestyle="--", linewidth=1.5, label="Calibrated")
    axes[0].set_title("Integrated Forcing Fit")
    axes[0].set_xlabel("Normalized time t")
    axes[0].set_ylabel("∫ 0.2 F(t) dt")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: solution
    axes[1].plot(t_np, U_true_np, color="black", linewidth=1.5, label="Reference y(t)")
    axes[1].plot(t_np, U_hat_np,  color="blue",  linestyle="--", linewidth=1.5, label="Reconstructed y(t)")
    axes[1].set_title("Solution Reconstruction")
    axes[1].set_xlabel("Normalized time t")
    axes[1].set_ylabel("y(t)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()