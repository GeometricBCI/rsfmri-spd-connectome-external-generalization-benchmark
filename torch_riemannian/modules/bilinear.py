import torch
import torch.nn as nn
from typing import Optional

from torch.nn.utils import parametrizations

from geoopt.manifolds import Stiefel
from geoopt.tensor import ManifoldParameter


class BiMap(nn.Module):
    r"""BiMap layer for Symmetric Positive Definite (SPD) matrices from [1]_.

    Applies a bilinear mapping to input symmetric matrices while constraining the mapping
    matrix to the Stiefel manifold.

    The layer transforms an input symmetric matrix :math:`X \in \mathbb{R}^{n \times n}` as follows:

    .. math::
        Y = W^T \, X \, W,

    where :math:`W \in \mathbb{R}^{n \times m}` is a learnable weight matrix whose columns are
    constrained to lie on the Stiefel manifold (i.e. :math:`W^T W = I`). This ensures that the
    transformation preserves important geometric properties of the input matrices.

    When :math:`m > n` (i.e., when :attr:`out_features` exceeds :attr:`in_features`), an additional
    transformation is applied via a :class:`BiMapIncreaseDim` module to increase the dimensionality
    of the input before applying the bilinear mapping.

    To maintain numerical stability during training, the weight parameters are initialized with a
    rectification procedure. This involves computing an eigen-decomposition of an intermediate matrix,
    clamping its eigenvalues to a minimum threshold :math:`\epsilon` (default 1e-4), and rescaling the
    weight matrix accordingly.

    Parameters
    ----------
    in_features : int
        The dimensionality of the input symmetric matrices (typically corresponding to the number of channels).
    out_features : int
        The target dimensionality for the projected symmetric matrices on the Stiefel manifold.
    threshold : float, optional
        The minimum threshold :math:`\epsilon` for eigenvalues during the rectification of the weight
        initialization. Default is 1e-4.
    device : torch.device or None, optional
        The device on which to allocate the model parameters.
    dtype : torch.dtype or None, optional
        The data type for the model parameters.

    Attributes
    ----------
    weight : ManifoldParameter
        The weight matrix :math:`W \in \mathbb{R}^{n \times m}`, constrained on the Stiefel manifold.
    increase_dim : BiMapIncreaseDim or None
        An optional module used to increase the input dimensionality when :attr:`out_features`
        is greater than :attr:`in_features`.

    References
    ----------
    .. [1] Huang, Z., & Van Gool, L. (2017). A Riemannian network for SPD matrix learning.
           In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 31, No. 1).

    Example
    -------
    >>> import torch
    >>> # Define a BiMap layer with input dimension 64 and output dimension 32.
    >>> layer = BiMap(in_features=64, out_features=32)
    >>> # Create a batch of 10 symmetric matrices (for example purposes, random matrices).
    >>> X = torch.randn(10, 64, 64)
    >>> # Compute the bilinear mapping.
    >>> Y = layer(X)
    """

    def __init__(
        self, in_features, out_features, threshold=1e-4, device=None, dtype=None
    ):
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.threshold_ = threshold
        self.manifold_ = Stiefel()

        self.increase_dim = None
        if out_features > in_features:
            self.increase_dim = BiMapIncreaseDim(
                in_features, out_features, device=device, dtype=dtype
            )
            self._in_features = out_features

        self.weight = ManifoldParameter(
            torch.empty(
                [1, self._in_features, self._out_features], device=device, dtype=dtype
            ),
            manifold=self.manifold_,
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        W = torch.rand(
            self.weight.shape, dtype=self.weight.dtype, device=self.weight.device
        )
        s, U = torch.linalg.eigh(W.mT @ W)
        smod = s.clamp(min=self.threshold_).rsqrt()
        A = U @ torch.diag_embed(smod) @ U.mT
        self.weight.data = W @ A

    def forward(self, X):
        output = X

        if self.increase_dim:
            output = self.increase_dim(output)

        return self.weight.mT @ output @ self.weight


class BiMapDepthWise(nn.Module):
    """
    BiMapDepthWise layer for Symmetric Positive Definite (SPD) matrices,
    proposed in [1]_.

    This layer performs a bilinear mapping of SPD matrices, transforming them
    while preserving the SPD structure. It applies a transformation of the form:

    .. math::
        P = \text{weight}^{\top} X \, \text{weight}

    where \\( \text{weight} \\) is a parameter matrix constrained to be on the
    Stiefel manifold.

    Parameters
    ----------
    depthwise : int
        The number of heads or channels (e.g., the 'h' dimension in the data).
    in_features : int
        The input dimension of the SPD matrices.
    out_features : int
        The output dimension of the SPD matrices after the bilinear mapping.

    Attributes
    ----------
    weight : geoopt.ManifoldParameter
        The transformation matrices on the Stiefel manifold of
        shape (h, in_features, out_features).
    increase_dim : Optional[nn.Module]
        A module to increase the dimension of input SPD matrices if needed.

    Notes
    -----
    - If `out_features` is greater than `in_features`, the input SPD matrices
    are first increased in dimension using `BiMapIncreaseDim`.
    - The layer ensures that the output matrices remain SPD.

    References
    ----------
    .. [1] Ju, C., & Guan, C. (2022). Tensor-CSPNet: A Novel Geometric
        Deep Learning Framework for Motor Imagery Classification. IEEE
        Transactions on Neural Networks and Learning Systems, 34(12), 10955-10969.
    """

    def __init__(
        self,
        depthwise: int,
        in_features: int,
        out_features: int,
        device=None,
        dtype=None,
    ):
        super(BiMapDepthWise, self).__init__()

        self.depthwise = depthwise
        self.increase_dim: Optional[nn.Module] = None

        self._in_features = in_features
        self._out_features = out_features

        if out_features > in_features:
            self.increase_dim = BiMapIncreaseDim(
                in_features, out_features, device=device, dtype=dtype
            )
            self._in_features = out_features

        # Here, we are using the `nn.Parameter` to create a parameter
        # that will be registered in the module. This allows us to
        # use the parameter in the forward pass and have it be
        # automatically updated during training.
        self.register_parameter(
            "weight",
            nn.Parameter(
                torch.empty(
                    self.depthwise,
                    self._in_features,
                    self._out_features,
                    device=device,
                    dtype=dtype,
                ),
                requires_grad=True,
            ),
        )

        self._init_bimap_parameter()

    def _init_bimap_parameter(self):
        v = torch.empty(
            self.depthwise,
            self._in_features,
            self._out_features,
            dtype=self.weight.dtype,
            device=self.weight.device,
        ).uniform_(0.0, 1.0)

        u, _, _ = torch.linalg.svd(v @ v.mT)
        self.weight.data = u[:, :, : self._out_features]

        parametrizations.orthogonal(self)

    def _bimap_multiplication(self, X: torch.Tensor) -> torch.Tensor:
        """
        Perform the bilinear mapping on the input tensor X.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, h, in_features, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor after bilinear mapping, of shape (batch_size, h, out_features, out_features).
        """
        weight = self.weight  # Shape: (h, in_features, out_features)

        weight_t = weight.mT  # Shape: (h, out_features, in_features)

        return weight_t @ (X @ weight)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BiMap layer.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (batch_size, h, n_in, n_in).

        Returns
        -------
        torch.Tensor
            Output tensor after bilinear mapping, of shape (batch_size, h, out_features, out_features).
        """
        if self.increase_dim:
            X = self.increase_dim(X)

        return self._bimap_multiplication(X)


class BiMapIncreaseDim(nn.Module):
    r"""Creates a bilinear mapping layer for SPD matrix dimensionality expansion, it is used in [1]_.


    Transforms input SPD matrices from size
        ``(..., in_features, in_features)``
    to
        ``(..., out_features, out_features)``
    using a semi-orthogonal projection and identity padding, preserving the
    symmetric positive definite (SPD) property.

    The transformation is defined as:

    .. math::
        Y = P + W X W^T

    where:

        - :math:`X \in \mathbb{R}^{k \times k}` is the input SPD matrix (k = in_features)
        - :math:`W \in \mathbb{R}^{d \times k}` is a semi-orthogonal projection matrix (d = out_features)
        - :math:`P \in \mathbb{R}^{d \times d}` is an identity padding matrix with:

            .. math::
                P_{ii} = \begin{cases}
                    0 & \text{if } i \leq k \\
                    1 & \text{otherwise}
                \end{cases}

    This operation maintains the SPD property while expanding the matrix dimensionality.


    Parameters
    ----------
    in_features : int
        Dimensionality of input SPD matrices (must be square).
    out_features : int
        Target dimensionality of output SPD matrices (must be ≥ in_features).
    device : torch.device
        Target device for layer parameters.
    dtype : torch.dtype
        Data type for layer parameters.


    Notes
    -----
    We are not sure about the original source of this layer and logic behind it, but it is used in [1]_.


    References
    ----------
    .. [1] Ju, C., & Guan, C. (2022). Tensor-CSPNet: A Novel Geometric
        Deep Learning Framework for Motor Imagery Classification. IEEE
        Transactions on Neural Networks and Learning Systems, 34(12), 10955-10969.


    Shape
    -----
    Input: :math:`(..., \text{in\_features}, \text{in\_features})`
    Output: :math:`(..., \text{out\_features}, \text{out\_features})`

    Examples
    --------
    >>> layer = BiMapIncreaseDim(16, 32)
    >>> x = torch.randn(2, 16, 16)  # Batch of 2 SPD matrices
    >>> output = layer(x)
    >>> output.shape
    torch.Size([2, 32, 32])

    # With channel dimension
    >>> x = torch.randn(4, 3, 16, 16)  # Batch of 4, 3-channel SPD matrices
    >>> output = layer(x)
    >>> output.shape
    torch.Size([4, 3, 32, 32])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """
        Parameters
        ----------
        in_features : int
            Dimensional of the input manifold space to be BiMapped. As the input
            is a symmetric matrix (same dimension for rows and columns),
            in_shape is n_channels from covariance layer.
        out_features : int
            Dimensional of the output manifold space to be BiMapped.
        device : torch.device
            Target device for layer parameters.
        dtype : torch.dtype
            Data type for layer parameters.
        """
        super(BiMapIncreaseDim, self).__init__()

        if out_features < in_features:
            raise ValueError("Output features must be ≥ input features")

        self.register_buffer(
            "projection_matrix",
            torch.eye(out_features, in_features, device=device, dtype=dtype),
        )
        self.register_buffer(
            "add",
            torch.diag(
                (torch.arange(out_features, device=device, dtype=dtype) >= in_features)
            ).to(dtype=dtype),
        )

    def forward(self, input):
        """

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (..., in_features, in_features).

        Returns
        -------
        torch.Tensor
            Output tensor after dimensionality expansion, of shape (..., out_features, out_features).

        """
        orig_ndim = input.ndim

        if orig_ndim == 3:
            input = input.unsqueeze(1)
        # Add a channel dimension to ensure compatibility with subsequent operations expecting a 4D tensor

        # Prepare buffers with broadcasting dimensions and convert to input dtype
        projection_matrix = self.projection_matrix.view(
            1, 1, *self.projection_matrix.shape
        ).to(input.dtype)  # (1, 1, out, in)
        add = self.add.view(1, 1, *self.add.shape).to(input.dtype)  # (1, 1, out, out)

        # Type-safe broadcasted operations
        output = add + (projection_matrix @ input @ projection_matrix.transpose(2, 3))

        # Maintain original dimensionality by removing the added channel dimension.
        # The squeeze operation ensures that the output tensor matches the input's
        # original dimensionality when the input is 3D. This is particularly important
        # for use cases where the input tensor does not have a channel dimension,
        # and the output needs to remain consistent with the input's shape.
        if orig_ndim == 3:
            output = output.squeeze(1)

        return output
