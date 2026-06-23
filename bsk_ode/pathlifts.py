import torch
from torch import nn


def tlift(X, holder_value):
    """Time-lift a path by appending a power of time as an extra channel.

    The first column of `X` is assumed to be a time/grid variable `x`.
    This function appends an extra channel `x**(2H)` where `H` is the
    provided Hölder exponent.

    Args:
        X: Tensor of shape (T, d). First column is the time/grid variable.
        holder_value: Hölder exponent H used in the power `x**(2H)`.

    Returns:
        Tensor of shape (T, d + 1) containing the original path with
        an extra channel `x**(2H)` concatenated as the last column.

    Raises:
        ValueError: If `X` does not have shape (T, d) (i.e., `X.ndim != 2`).
    """
    if X.ndim != 2:
        raise ValueError("X must have shape (T, d)")

    x = X[:, 0]
    H = holder_value

    x_lift = torch.pow(x, 2.0 * H).unsqueeze(1)
    X_tlift = torch.cat([X, x_lift], dim=1)
    return X_tlift


def init_path_extension_weights(m):
    """Initialize weights for linear layers in a path-extension network.

    Applies Xavier uniform initialization to the weights and zeros to
    the biases of any ``nn.Linear`` module passed in. This is intended
    to be used with ``nn.Module.apply(init_path_extension_weights)`` so
    that it walks the module tree and initializes all linear layers.

    Args:
        m: Module to (potentially) initialize. If ``m`` is an instance
            of ``nn.Linear``, its weights and bias are modified in-place.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)


class PathExtension(nn.Module):
    """Feed-forward network to learn non-linear path extensions.

    This module builds a multi-layer perceptron that maps input path
    channels to an extended representation, which can be concatenated
    back to the original path or used as additional features.

    Args:
        input_dim: Dimension of the input features.
        output_dim: Dimension of the extended/output features.
        hidden_dims: Sequence of hidden-layer sizes used to construct
            the intermediate fully connected layers.
        activation_cls: Activation module class (e.g. ``nn.Tanh``) used
            between linear layers.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims=(8, 16, 32, 16, 8),
        activation_cls=nn.Tanh,
    ):
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
        """Compute extended features for the given input batch.

        Args:
            x: Input tensor of shape ``(..., input_dim)`` containing
                path features or channels.

        Returns:
            Tensor of shape ``(..., output_dim)`` with the learned
            extended features.
        """
        return self.net(x)


def shuffle_loss_function(X_bar: torch.Tensor) -> torch.Tensor:
    """Compute scalar shuffle loss enforcing Chen's relation on an extended path.

    For an extended path ``X_bar`` of shape (T, E), this loss penalizes
    violations of the shuffle (Chen) relation between first- and
    second-level iterated integrals across channels. It sums the squared
    entries of the upper-triangular part of the residual matrix.

    Args:
        X_bar: Extended path tensor of shape (T, E), where T is the
            number of time steps and E is the number of channels.

    Returns:
        Scalar tensor containing the sum of squared upper-triangular
        residuals of the shuffle constraint.
    """
    dX = X_bar[1:, :] - X_bar[:-1, :]
    X_left = X_bar[:-1, :]
    I = torch.einsum("ka,kb->ab", X_left, dX)
    deltas = X_bar[-1, :] - X_bar[0, :]
    R = deltas[:, None] * deltas[None, :] - (I + I.T)
    return torch.triu(R.pow(2), diagonal=0).sum()


def shuffle_loss_residual(X_bar: torch.Tensor) -> torch.Tensor:
    """Compute the full shuffle residual matrix for an extended path.

    The residual matrix R measures violation of the Chen (shuffle)
    relation between channels of an extended path. For channels i, j,
    the ideal relation is roughly
    ΔXᶦ ΔXʲ ≈ ∑ Xᶦ dXʲ + ∑ Xʲ dXᶦ,
    and R encodes how far the path deviates from this identity.

    Args:
        X_bar: Extended path tensor of shape (T, E), where T is the
            number of time steps and E is the number of channels.

    Returns:
        Tensor of shape (E, E) containing the shuffle residual matrix R,
        where R[i, j] is the residual for the (i, j) channel pair.
    """
    dX = X_bar[1:, :] - X_bar[:-1, :]    # (T-1, E)
    X_left = X_bar[:-1, :]               # (T-1, E)
    I = torch.einsum("ka,kb->ab", X_left, dX)  # ∑ X^i ΔX^j over k
    deltas = X_bar[-1, :] - X_bar[0, :]        # (E,)
    R = deltas[:, None] * deltas[None, :] - (I + I.T)
    return R