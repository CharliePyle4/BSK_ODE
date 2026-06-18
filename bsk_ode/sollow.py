import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import random
import keras_sig


# Cell 3 - seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_default_dtype(torch.float64)

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


def signature_of_path(path, depth=3):
    if not isinstance(path, torch.Tensor):
        path = torch.tensor(path, dtype=torch.float64)
    path_tensor = path.unsqueeze(0)
    sig = signatory.signature(
        path_tensor,
        depth=depth,
        basepoint=path_tensor[:, 0, :],
        scalar_term=True
    ).squeeze(0)
    return sig.detach()


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
