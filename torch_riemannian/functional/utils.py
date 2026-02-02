import torch


def ensure_sym(matrix):
    """Ensures that the last two dimensions of the tensor are symmetric.
    Parameters
    ----------
    A : torch.Tensor
        with the last two dimensions being identical
    -------
    Returns : torch.Tensor
    """
    return (matrix + matrix.mT) / 2


def modeig_forward(X, applied_fct, *args):
    """
    Forward pass for the modified eigenvalue of a symmetric matrix X.

    Parameters
    ----------
    X : torch.Tensor
        Symmetric matrix of shape (batch_size, n_channels, n_channels).
    applied_fct : callable
        Function to apply to the eigenvalues.
    *args : tuple
        Additional arguments for the applied function.

    Returns
    -------
    output : torch.Tensor
        Modified matrix after applying the function to the eigenvalues.
    s : torch.Tensor
        Eigenvalues of X.
    U : torch.Tensor
        Eigenvectors of X.
    s_modified : torch.Tensor
        Modified eigenvalues after applying the function.
    """
    s, U = torch.linalg.eigh(X)
    s_modified = applied_fct(s, *args)
    output = U @ torch.diag_embed(s_modified) @ U.transpose(-1, -2)
    return output, s, U, s_modified


def modeig_backward(grad_output, s, U, s_modified, derivative, *args):
    """
    Backward pass for the modified eigenvalue of a symmetric matrix X.

    Parameters
    ----------
    grad_output : torch.Tensor
        Gradient of the loss with respect to the output.
    s : torch.Tensor
        Eigenvalues of X.
    U : torch.Tensor
        Eigenvectors of X.
    s_modified : torch.Tensor
        Modified eigenvalues after applying the function.
    derivative : callable
        Derivative of the applied function with respect to the eigenvalues.
    *args : tuple
        Additional arguments for the derivative of the applied function.

    Returns
    -------
    grad_input : torch.Tensor
        Gradient of the loss with respect to the input.
    """

    # Compute Loewner matrix
    denominator = s.unsqueeze(-1) - s.unsqueeze(-1).transpose(-1, -2)
    is_eq = denominator.abs() < torch.finfo(s.dtype).eps
    denominator[is_eq] = 1.0

    # Case: sigma_i != sigma_j
    numerator = s_modified.unsqueeze(-1) - s_modified.unsqueeze(-1).transpose(-1, -2)

    # Case: sigma_i == sigma_j
    s_derivative = derivative(s, *args)
    numerator[is_eq] = (
        0.5
        * (s_derivative.unsqueeze(-1) + s_derivative.unsqueeze(-1).transpose(-1, -2))[
            is_eq
        ]
    )
    L = numerator / denominator

    grad_input = (
        U
        @ (L * (U.transpose(-1, -2) @ ensure_sym(grad_output) @ U))
        @ U.transpose(-1, -2)
    )

    return grad_input
