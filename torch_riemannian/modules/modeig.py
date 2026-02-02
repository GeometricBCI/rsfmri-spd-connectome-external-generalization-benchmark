import torch
import torch.nn as nn

from ..functional import sym_reeig, sym_logm


class ReEig(nn.Module):
    """ReEig layer from [1]_.

    This class added non-linearity to the network by
    applying a rectified linear unit to the eigenvalues
    of a symmetric matrix. If threshold > 0, the matrix
    is non-negative and positive.

    Parameters
    ----------
    threshold : float
        Threshold for the rectified linear unit

    References
    ----------
    .. [1] Zhiwu Huang and Luc Van G, 2016,
        A Riemannian Network for SPD Matrix Learning
        AAAI
    """

    def __init__(self, threshold=1e-4, device=None, dtype=None):
        super().__init__()
        self.register_buffer(
            "threshold_", torch.tensor(threshold, device=device, dtype=dtype)
        )

    def forward(self, X):
        return sym_reeig.apply(X, self.threshold_)


class LogEig(nn.Module):
    """LogEig layer from [1]_.

    This class performs Riemannian projection into a flat space
    by applying the logarithm to the eigenvalues of a symmetric matrix.
    The output is flattened to obtain a vector representation of the matrix.

    Parameters
    ----------
    dim : int
        Dimension of the symmetric matrix
    tril : bool
        If True, only the lower triangular part of the matrix is used

    References
    ----------
    .. [1] Huang, Z., & Van Gool, L. (2017). A riemannian network for spd matrix
       learning. In Proceedings of the AAAI conference on artificial intelligence
       (Vol. 31, No. 1).
    """

    def __init__(self, dim, tril=True, device=None, dtype=torch.long):
        super().__init__()
        self.tril = tril
        self.dtype = dtype
        self.device = device

        if self.tril:
            idx_lower = torch.tril_indices(dim, dim, offset=-1, device=device)
            idx_diag = torch.arange(start=0, end=dim, device=device)
            self.idx = torch.cat((idx_diag[None, :].tile((2, 1)), idx_lower), dim=1)
        self.dim = dim

    def forward(self, X):
        #return self.embed(sym_logm.apply(X)).to(device=self.device)
        return sym_logm.apply(X).to(device=self.device)
    
    def embed(self, X):
        if self.tril:
            x_vec = X[:, self.idx[0], self.idx[1]]
            x_vec[:, self.dim :] *= 2**0.5
        else:
            x_vec = X.flatten(start_dim=1)
        return x_vec
