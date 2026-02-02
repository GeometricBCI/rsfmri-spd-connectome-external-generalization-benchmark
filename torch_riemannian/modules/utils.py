import torch
from torch import nn
from torch.nn import functional as F

from einops.layers.torch import Rearrange
from typing import Optional


class PatchEmbeddingLayer(nn.Module):
    """
    Extract patches from the input signal using an unfolding (conv-like) operation.

    This layer reshapes a 1D input tensor of shape (batch, channels, time)
    into patches of shape (batch, num_patches, channels, patch_size).

    Parameters
    ----------
    n_chans : int
        Number of input channels.
    patch_size : int
        The length of each patch.
    stride : int, optional
        The step size between patches. If not provided, defaults to patch_size (non-overlapping patches).
    device : default None, torch.device, optional
        The device on which to run the module (e.g., 'cuda' or 'cpu').
    dtype : default None, torch.dtype, optional
        The data type for tensors (e.g., torch.float32).
    """

    def __init__(
        self,
        n_chans: int,
        n_patches: int,
        stride: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.n_chans = n_chans
        self.n_patches = n_patches
        self.stride = stride
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # I dont like to define layers in the forward method
        # but I think this
        # is the best way to do it here
        # because the layer is dependent on the input shape
        x = x.to(device=self.device, dtype=self.dtype)

        time = x.shape[-1]
        patch_size = time // self.n_patches
        stride = (self.stride, 1) if self.stride is not None else patch_size

        x_unsqueezed = x.unsqueeze(-1)
        # shape: (batch, channels, time, 1)

        patches = F.unfold(
            input=x_unsqueezed, kernel_size=(patch_size, 1), stride=stride
        )
        # shape: (batch, channels*patch_size, patches)

        patches = Rearrange(
            "batch (chans time) patches -> batch patches chans time",
            time=patch_size,
            chans=self.n_chans,
        )(patches)
        # shape: (batch, patches, channels, patch_size)

        return patches
