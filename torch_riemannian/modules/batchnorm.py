import torch

from torch import nn
from geoopt import ManifoldParameter, SymmetricPositiveDefinite

from ..functional import (
    sym_invsqrtm2,
    sym_expm,
    sym_logm,
    sym_sqrtm,
    sym_invsqrtm,
    sym_powm,
)
from ..functional import geodesic_interpolation_spdairm


class BrooksBatchNorm(nn.Module):
    """SPD Batch Normalization from Brooks et al. (2019) [1]_.

    Riemannian Batch Normalization (RBN) layer for the Symmetric Positive Definite (SPD) manifold
    equipped with the affine invariance Riemannian metric (AIRM),
    The RBN layer was proposed by Brooks et al. (2019) [1]_.

    Following the idea of batch norm, this layer estimates the Fréchet mean of a batch and then
    transports the batch to vary around the identify matrix.
    If specified, the layer additionally transports the data to vary around a learnable bias
    parameter.


    **Mathematical Overview**:


    Given a batch of SPD matrices :math:`\\{ P_i \\}`, the layer performs the following steps:

    1. **Estimate the Fréchet mean**:

       The iterative Karcher flow algorithm (typically with `n_iter = 1`) is used to estimate
       the Fréchet mean :math:`\\mathcal{G}`, which is defined as the matrix that minimizes the sum
       of squared Riemannian distances to all matrices in the batch.

    2. **Centering via Parallel Transport**:

       Each matrix :math:`P_i` is centered by transporting it to vary around the identity matrix:

       .. math::

          \\tilde{P}_i = \\mathcal{G}^{-\\frac{1}{2}} P_i \\mathcal{G}^{-\\frac{1}{2}}

    3. **Scaling with Learnable Parameter**:

       A learnable SPD matrix :math:`G_{\\phi}` rebiases the centered data:

       .. math::

          \\hat{P}_i = G_{\\phi}^{\\frac{1}{2}} \\tilde{P}_i G_{\\phi}^{\\frac{1}{2}}

    4. **Update Running Mean**:

       The running mean is updated using an exponential moving average with the specified
       momentum.


    Parameters
    ----------
    num_features : int
        The size of the SPD matrices (number of features).
    momentum : float
        Momentum factor for updating the running mean.
    rebias : bool  = True
        Flag that indicates if the layer should rebias the data.
    n_iter : int  = 1
        Number of Karcher flow iterations to estimate the batch mean (default: `n_iter=1`).


    References
    ----------
    .. [1] Brooks, D., Schwander, O., Barbaresco, F., Schneider, J. Y., &
        Cord, M. (2019). Riemannian batch normalization for SPD neural networks.
        Advances in Neural Information Processing Systems, 32.

    """

    def __init__(
        self,
        num_features,
        momentum=0.1,
        rebias=True,
        n_iter=1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.rebias = rebias
        self.n_iter = n_iter

        self.register_buffer(
            "running_mean",
            torch.empty(1, num_features, num_features, device=device, dtype=dtype),
        )

        if self.rebias:
            self.bias = ManifoldParameter(
                torch.empty(1, num_features, num_features, device=device, dtype=dtype),
                manifold=SymmetricPositiveDefinite(),
            )
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_mean[0].fill_diagonal_(1)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.rebias:
            self.bias.zero_()
            self.bias[0].fill_diagonal_(1)

    def forward(self, input):
        """
        Forward pass of the Riemannian Batch Normalization layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, h, n, n), where each slice along
            the batch dimension is an SPD matrix.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as the input.
        """
        if self.training:
            # compute initial mean estimate (=Euclidean mean) for the batch
            mean = input.mean(dim=0, keepdim=True)
            if input.shape[0] > 1:
                # refine the running mean estimate with k steps of the Karcher flow method
                for _ in range(self.n_iter):
                    mean_sq, mean_invsq = sym_invsqrtm2.apply(mean.detach())
                    input_ts = sym_logm.apply(mean_invsq @ input @ mean_invsq)
                    mean_ts = input_ts.mean(dim=0, keepdim=True)
                    mean = mean_sq @ sym_expm.apply(mean_ts) @ mean_sq
            # update the running mean
            with torch.no_grad():
                self.running_mean = geodesic_interpolation_spdairm(
                    self.running_mean, mean, self.momentum
                )
        else:
            mean = self.running_mean

        # transport to identity matrix
        mean_invsq = sym_invsqrtm.apply(mean)
        input = mean_invsq @ input @ mean_invsq
        # optionally rebias the data
        if self.bias is not None:
            bias_sq = sym_sqrtm.apply(self.bias)
            input = bias_sq @ input @ bias_sq

        return input


class SPDBatchNorm(nn.Module):
    """SPD Batch Normalization from Kobler et al. (2022) [1]_.

    SPD Batch Normalization (SPDBN) layer for the Symmetric Positive Definite (SPD) manifold
    equipped with the affine invariance Riemannian metric (AIRM),
    The SPDBN layer was proposed by Kobler et al. (2022) [1]_.

    Parameters
    ----------
    num_features : int
        The size of the SPD matrices (number of features).
    momentum : float
        Momentum factor for updating the running mean.
    rebias : bool  = True
        Flag that indicates if the layer should rebias the data.
    n_iter : int  = 1
        Number of Karcher flow iterations to estimate the batch mean (default: `n_iter=1`).
    eps : float  = 1e-5
        Stabilizing factor that is added to prevent division by 0.

    References
    ----------
    .. [1] Kobler, R. J., Hirayama, J., Zhao, Q., & Kawanabe, M. (2022).
        SPD domain-specific batch normalization to crack interpretable unsupervised domain
        adaptation in EEG
        Advances in Neural Information Processing Systems, 35.

    """

    def __init__(
        self,
        num_features,
        momentum=0.1,
        affine=True,
        n_iter=1,
        bias_requires_grad=True,
        weight_requires_grad=True,
        eps=1e-5,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.affine = affine
        self.n_iter = n_iter
        self.bias_requires_grad = bias_requires_grad
        self.weight_requires_grad = weight_requires_grad
        self.eps = eps

        if device is None:
            device = torch.device("cpu")

        self.register_buffer(
            "running_mean",
            torch.empty(1, num_features, num_features, dtype=dtype, device=device),
        )
        self.register_buffer(
            "running_var", torch.empty(1, 1, dtype=dtype, device=device)
        )

        if self.affine:
            self.bias = ManifoldParameter(
                torch.empty(1, num_features, num_features, dtype=dtype, device=device),
                manifold=SymmetricPositiveDefinite(),
                requires_grad=bias_requires_grad,
            )
            self.weight = ManifoldParameter(
                torch.empty(1, 1, dtype=dtype, device=device),
                manifold=SymmetricPositiveDefinite(),
                requires_grad=weight_requires_grad,
            )
        else:
            self.register_buffer("bias", None)
            self.register_buffer("weight", None)
        self.reset_parameters()

    def reset_running_stats(self) -> None:
        self.running_mean.zero_()
        self.running_mean[0].fill_diagonal_(1)
        self.running_var.fill_(1.0)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.affine:
            self.bias.zero_()
            self.bias[0].fill_diagonal_(1)
            self.weight.fill_(1.0)

    def forward(self, input):
        """
        Forward pass of the Riemannian Batch Normalization layer.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, h, n, n), where each slice along
            the batch dimension is an SPD matrix.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as the input.
        """
        n_samples = input.shape[0]
        if self.training:
            # compute the initial running mean estimate
            batch_mean = input.mean(dim=0, keepdim=True)
            rm = geodesic_interpolation_spdairm(
                self.running_mean, batch_mean, self.momentum
            )

            if n_samples > 1:
                # iteratively refine the running mean estimate
                for _ in range(self.n_iter):
                    mean_sq, mean_invsq = sym_invsqrtm2.apply(rm.detach())
                    input_tangent = sym_logm.apply(mean_invsq @ input @ mean_invsq)
                    batch_mean_tangent = input_tangent.mean(dim=0, keepdim=True)
                    batch_mean = mean_sq @ sym_expm.apply(batch_mean_tangent) @ mean_sq
                    # update the running mean
                    rm = geodesic_interpolation_spdairm(
                        self.running_mean, batch_mean, self.momentum
                    )
                # approx. Frechet variance at the running mean
                batch_mean_tangent = sym_logm.apply(mean_invsq @ rm @ mean_invsq)
                batch_variance = (
                    torch.norm(
                        input_tangent - batch_mean_tangent,
                        p="fro",  # codespell:ignore fro
                        dim=(-2, -1),
                        keepdim=True,
                    )
                    .square()
                    .mean(dim=0, keepdim=True)
                    .squeeze(-1)
                )
            else:
                mean_invsq = sym_invsqrtm.apply(rm)
                batch_variance = (
                    sym_logm.apply(mean_invsq @ input @ mean_invsq)
                    .square()
                    .sum(dim=(-1, -2), keepdim=True)
                    .squeeze(-1)
                )

            rv = (
                1.0 - self.momentum
            ) * self.running_var + self.momentum * batch_variance
        else:
            rm = self.running_mean
            rv = self.running_var
        # transport to identity matrix
        mean_invsq = sym_invsqrtm.apply(rm)
        output = mean_invsq @ input @ mean_invsq
        # whiten the Frechet variance and
        # optionally rescale the data
        output = sym_powm.apply(
            output,
            (self.weight if self.weight is not None else 1.0) / (rv + self.eps).sqrt(),
        )
        # optionally rebias the data
        if self.bias is not None:
            bias_sq = sym_sqrtm.apply(self.bias)
            output = bias_sq @ output @ bias_sq

        if self.training:
            # store the updated running statistics
            self.running_mean = rm.detach()
            self.running_var = rv.detach()

        return output
