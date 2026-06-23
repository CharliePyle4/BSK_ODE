import torch

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
