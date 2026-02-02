import torch


def covariance(input: torch.Tensor) -> torch.Tensor:
    """
    Computes the covariance matrix of multivariate data.

    The input tensor is assumed to have shape (..., n_channels, n_times), where
    ... represents any number of leading dimensions.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor with EEG windows of shape (..., n_channels, n_times).

    Returns
    -------
    torch.Tensor
        Covariance matrices with shape (..., n_channels, n_channels).
    """
    # Center the data by subtracting the mean over the time dimension
    input_centered = input - input.mean(dim=-1, keepdim=True)

    # Compute covariance using Einstein summation: summing over the time dimension
    covariances = torch.einsum(
        "...ik,...jk->...ij", input_centered, input_centered
    ) / input.size(-1)

    return covariances
