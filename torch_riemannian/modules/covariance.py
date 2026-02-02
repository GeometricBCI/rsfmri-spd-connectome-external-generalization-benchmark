"""
Covariance as PyTorch Layer following transformation style.
"""

from typing import Callable

import torch

from torch import nn

from ..functional.covariance import covariance


class CovLayer(nn.Module):
    """
    Covariance layer.

    This class computes the covariance of a batch of multivariate data.
    The input data is assumed to have shape (..., n_channels, n_times), where
    ... represents an arbitrary number of dimensions.
    """

    def __init__(self, method: Callable = covariance, device=None, dtype=None):
        super().__init__()
        self.method = method
        self._device = device
        self._dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : torch.Tensor
            EEG windows of shape (..., n_channels, n_times).

        Returns
        -------
        torch.Tensor
            Covariance matrices of shape (..., n_channels, n_channels).
        """
        return self.method(input).to(device=self._device, dtype=self._dtype)
