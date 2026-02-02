import warnings
import torch
from math import sqrt
from torch.autograd import Function

from .utils import modeig_backward, modeig_forward


class sym_logm(Function):
    """
    Matrix logarithm of a symmetric matrix.

    This class computes the matrix logarithm of a symmetric matrix X.
    It also adapts the backpropagation according to the chain rule [1]_, [2]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Matrix logarithm of X

    References
    ----------
    .. [1] Ionescu, C., Vantzos, O., & Sminchisescu, C. (2015). Matrix
        backpropagation for deep networks with structured layers. In Proceedings
        of the IEEE international conference on computer vision (pp. 2965-2973).
    .. [2] Huang, Z., & Van Gool, L. (2017). A riemannian network for spd
        matrix learning. In Proceedings of the AAAI conference on artificial
        intelligence (Vol. 31, No. 1).

    """

    @staticmethod
    def applied_fct(s):
        return s.clamp(min=torch.finfo(s.dtype).eps).log()

    @staticmethod
    def derivative(s):
        s_deriv = s.reciprocal()
        # pick subgradient 0 for clamped eigenvalues
        s_deriv[s <= torch.finfo(s.dtype).eps] = 0
        return s_deriv

    @staticmethod
    def forward(ctx, X):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Symmetric matrix of shape (batch_size, n_channels, n_channels).

        Returns
        -------
        torch.Tensor
            Matrix logarithm of X.
        """
        output, s, U, s_modified = modeig_forward(X, sym_logm.applied_fct)
        # Check if the threshold is valid
        min_eigenvalue = s.min()
        if torch.finfo(s.dtype).eps > min_eigenvalue:
            warnings.warn(
                f"The eps of {s.dtype} is larger than the smallest eigenvalue ({min_eigenvalue}) "
                "of X. This might lead to inaccurate results.",
                UserWarning,
            )
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, sym_logm.derivative)


class sym_expm(Function):
    """
    Matrix exponential of a symmetric matrix.

    This class computes the matrix exponential of a symmetric matrix X.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Matrix logarithm of X

    """

    @staticmethod
    def applied_fct(s):
        return s.exp()

    @staticmethod
    def derivative(s):
        return s.exp()

    @staticmethod
    def forward(ctx, X):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Symmetric matrix of shape (batch_size, n_channels, n_channels).

        Returns
        -------
        torch.Tensor
            Matrix exponential of X.
        """
        output, s, U, s_modified = modeig_forward(X, sym_expm.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, sym_expm.derivative)


class sym_reeig(Function):
    """
    Rectification of the eignvalues of a symmetric matrix.

    Computes the regularized matrix logarithm of a symmetric matrix X.
    It also adapts the backpropagation according to the chain rule [1]_ and [2]_.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Rank deficient matrix logarithm of X

    References
    ----------
    .. [1] Ionescu, C., Vantzos, O., & Sminchisescu, C. (2015). Matrix
        backpropagation for deep networks with structured layers. In Proceedings
        of the IEEE international conference on computer vision (pp. 2965-2973).
    .. [2] Huang, Z., & Van Gool, L. (2017). A riemannian network for spd
        matrix learning. In Proceedings of the AAAI conference on artificial
        intelligence (Vol. 31, No. 1).
    """

    @staticmethod
    def applied_fct(s, threshold):
        return s.clamp(min=threshold)

    @staticmethod
    def derivative(s, threshold):
        s_deriv = torch.zeros_like(s)
        s_deriv[s > threshold] = 1
        return s_deriv

    @staticmethod
    def forward(ctx, X, threshold):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Symmetric matrix of shape (batch_size, n_channels, n_channels).
        threshold : float
            Threshold for numerical stability.

        Returns
        -------
        torch.Tensor
            Regularized matrix.
        """
        output, s, U, s_modified = modeig_forward(X, sym_reeig.applied_fct, threshold)
        ctx.save_for_backward(s, U, s_modified)
        ctx.threshold = threshold
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        s, U, s_modified = ctx.saved_tensors
        threshold = ctx.threshold
        return modeig_backward(
            grad_output, s, U, s_modified, sym_reeig.derivative, threshold
        ), None


class sym_abseig(Function):
    """
    Apply abs function to the eigenvalues of a symmetric matrix.

    This class applies the abs function to the eigenvalues of a symmetric matrix and return the
    modified matrix.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Modified matrix whose eignvalues are abs(eig(X))

    """

    @staticmethod
    def applied_fct(s):
        return s.abs()

    @staticmethod
    def derivative(s):
        return s.sign()

    @staticmethod
    def forward(ctx, X):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Symmetric matrix of shape (batch_size, n_channels, n_channels).

        Returns
        -------
        torch.Tensor
            Modified matrix whose eignvalues are abs(eig(X))
        """
        output, s, U, s_modified = modeig_forward(X, sym_abseig.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, sym_abseig.derivative)


class sym_powm(Function):
    """
    Compute matrix power.

    Computes the matrix power of a symmetric matrix X via Hermitian eigen decomposition.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels)
    exponent : float
        Exponent to raise the matrix to.

    Returns
    -------
    torch.Tensor
        X ** exponent

    """

    @staticmethod
    def applied_fct(s, exponent):
        return s.pow(exponent=exponent)

    @staticmethod
    def derivative(s, exponent):
        return exponent * s.pow(exponent=exponent - 1.0)

    @staticmethod
    def forward(ctx, X, exponent):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Symmetric matrix of shape (batch_size, n_channels, n_channels).
        exponent : float
            Exponent to raise the matrix to.

        Returns
        -------
        torch.Tensor
            X ** exponent
        """
        output, s, U, s_modified = modeig_forward(X, sym_powm.applied_fct, exponent)
        ctx.save_for_backward(s, U, s_modified)
        ctx.exponent = exponent
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        s, U, s_modified = ctx.saved_tensors
        exponent = ctx.exponent
        return modeig_backward(
            grad_output, s, U, s_modified, sym_powm.derivative, exponent
        ), None


class sym_sqrtm(Function):
    """
    Matrix square root

    This class computes the matrix square root of a symmetric positive definite matrix X via
    Hermitian eigen decomposition.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric positive definite matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Matrix square root of X.

    """

    @staticmethod
    def applied_fct(s):
        return s.clamp(min=torch.finfo(s.dtype).eps).sqrt()

    @staticmethod
    def derivative(s):
        sder = s.rsqrt() / 2
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= torch.finfo(s.dtype).eps] = 0
        return sder

    @staticmethod
    def forward(ctx, X):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Symmetric positive definite matrix of shape (batch_size, n_channels, n_channels).

        Returns
        -------
        torch.Tensor
            Matrix square root of X.
        """
        output, s, U, s_modified = modeig_forward(X, sym_sqrtm.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, sym_sqrtm.derivative)


class sym_invsqrtm(Function):
    """
    Inverse matrix square root

    This class computes the inverse of the matrix square root of a symmetric positive definite
    matrix X via Hermitian eigen decomposition.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric positive definite matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Inverse matrix square root of X.

    """

    @staticmethod
    def applied_fct(s):
        return s.clamp(min=torch.finfo(s.dtype).eps).rsqrt()

    @staticmethod
    def derivative(s):
        sder = -0.5 * s.pow(-1.5)
        # pick subgradient 0 for clamped eigenvalues
        sder[s <= torch.finfo(s.dtype).eps] = 0
        return sder

    @staticmethod
    def forward(ctx, X):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Symmetric positive definite matrix of shape (batch_size, n_channels, n_channels).

        Returns
        -------
        torch.Tensor
            Inverse matrix square root of X.
        """
        output, s, U, s_modified = modeig_forward(X, sym_invsqrtm.applied_fct)
        ctx.save_for_backward(s, U, s_modified)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of the loss with respect to the output.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        s, U, s_modified = ctx.saved_tensors
        return modeig_backward(grad_output, s, U, s_modified, sym_invsqrtm.derivative)


class sym_invsqrtm2(Function):
    """
    Matrix square root and inverse matrix square root

    This class computes the matrix square root and its inverse for a symmetric positive definite
    matrix X via Hermitian eigen decomposition.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric positive definite matrix of shape (batch_size, n_channels, n_channels)

    Returns
    -------
    torch.Tensor
        Matrix square root of X.
    torch.Tensor
        Inverse of the matrix square root of X.

    """

    @staticmethod
    def forward(ctx, X):
        """
        Forward pass.

        Parameters
        ----------
        X : torch.Tensor
            Symmetric positive definite matrix of shape (batch_size, n_channels, n_channels).

        Returns
        -------
        torch.Tensor
            Matrix square root of X.
        torch.Tensor
            Inverse of the matrix square root of X.
        """
        output_sqrt, s, U, s_sqrt = modeig_forward(X, sym_sqrtm.applied_fct)
        s_invsqrt = sym_invsqrtm.applied_fct(s)
        output_invsqrt = U @ torch.diag_embed(s_invsqrt) @ U.transpose(-1, -2)
        ctx.save_for_backward(s, U, s_sqrt, s_invsqrt)
        return output_sqrt, output_invsqrt

    @staticmethod
    def backward(ctx, grad_output_sqrt, grad_output_invsqrt):
        """
        Backward pass.

        Parameters
        ----------
        grad_output_sqrt : torch.Tensor
            Gradient of the loss with respect to the output_sqrt.
        grad_output_invsqrt : torch.Tensor
            Gradient of the loss with respect to the output_invsqrt.

        Returns
        -------
        torch.Tensor
            Gradient of the loss with respect to the input.
        """
        s, U, s_sqrt, s_invsqrt = ctx.saved_tensors
        return modeig_backward(
            grad_output_sqrt, s, U, s_sqrt, sym_sqrtm.derivative
        ) + modeig_backward(
            grad_output_invsqrt, s, U, s_invsqrt, sym_invsqrtm.derivative
        )


def geodesic_interpolation_spdairm(A, B, t):
    """Geodesic interpolation between points A and B on the SPD manifold equipped with AIRM.
    Parameters
    ----------
    A : torch.Tensor
        Interpolation starting point
    B : torch.Tensor
        Interpolation end point
    B : float
        Interpolation step size.  For t = 0, the output will be equal to A.
        For t = 1, the output will be equal to B.
    -------
    Returns : torch.Tensor
        Interpolated point the the SPD manifold
    """
    rm_sq, rm_invsq = sym_invsqrtm2.apply(A)
    return (
        rm_sq @ sym_powm.apply(rm_invsq @ B @ rm_invsq, torch.tensor(t).to(A)) @ rm_sq
    )


def geodesic_distance_spdairm(A, B):
    """Compute the geodesic distance between points A and B on the SPD manifold equipped with the
    affine invariant Riemannian metric (AIRM).
    Leading dimension of A and B must be able to broadcasted.
    Parameters
    ----------
    A : torch.Tensor
        SPD matrices with dimension (..., N, N)
    B : torch.Tensor
        SPD matrices with dimension (..., N, N)
    -------
    Returns : torch.Tensor
        Distances between A and B with dimension (...)
    """
    Ainvsqrt = sym_invsqrtm.apply(A)
    return torch.linalg.eigvalsh(Ainvsqrt @ B @ Ainvsqrt).square().sum(dim=-1).sqrt()


def sym_to_upper(X):
    """Takes upper triangular elements along last 2 dimensions.
    Off-diagonal elements are multiplied to preserve the norm.
    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrices with dimension (..., N, N)
    -------
    Returns : torch.Tensor
        Vectorized upper triangular part of X with dimension (..., N*(N+1)/2)
    """
    assert X.ndim >= 2
    assert X.shape[-1] == X.shape[-2]
    ndim = X.shape[-1]
    ixs = torch.triu_indices(ndim, ndim, offset=0)
    x_vec = X[..., ixs[0], ixs[1]]
    # multiply off-diagonal elements to preserve the norm
    x_vec[..., ixs[0] != ixs[1]] *= sqrt(2)
    return x_vec


def upper_to_sym(x_vec):
    """Converts elements of vectors with dimensions (..., N*(N+1)/2) to upper triangular matrices
    with dimension (..., N, N). Off-diagonal elements are multiplied to preserve the norm.
    Parameters
    ----------
    X : torch.Tensor
        Vectorized upper triangular matrices with dimension (..., N*(N+1)/2)
    -------
    Returns : torch.Tensor
        Symmetric matrices with dimension (..., N, N)
    """

    ndim = (sqrt(1 + 8 * x_vec.shape[-1]) - 1) / 2
    assert ndim == int(ndim)
    ndim = int(ndim)

    ixs = torch.triu_indices(ndim, ndim, offset=0)
    od_mask = ixs[0] != ixs[1]

    X = torch.empty(
        (*x_vec.shape[:-1], ndim, ndim), device=x_vec.device, dtype=x_vec.dtype
    )
    X[..., ixs[0], ixs[1]] = x_vec

    # multiply offdiagonal elements to preserve the norm
    X[..., ixs[0, od_mask], ixs[1, od_mask]] /= sqrt(2)
    X[..., ixs[1, od_mask], ixs[0, od_mask]] = X[..., ixs[0, od_mask], ixs[1, od_mask]]
    return X
