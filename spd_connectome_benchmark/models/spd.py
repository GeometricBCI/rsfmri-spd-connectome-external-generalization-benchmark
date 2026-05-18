"""SPD neural-network models used by the benchmark."""

from __future__ import annotations

import torch
import torch.nn as nn
from spd_learn.modules import BiMap, LogEig, ReEig

DEFAULT_SPDNET_REEIG_EPS = 1e-4


class SPDNetRegressor(nn.Module):
    """SPDNet regressor matching the paper's BiMap/ReEig/LogEig definition.

    Paper §2.5.2 and Supplementary Methods define BiMap/ReEig blocks followed
    by LogEig and the shared MLP head. ``dims`` stores one or more
    ``(input_dim, output_dim)`` BiMap pairs, for example ``(100, 25)`` for
    quarterdim and ``(100, 100, 100, 100)`` for the two-block variant.
    ReEig uses a fixed eigenvalue floor of ``1e-4`` by default. After LogEig,
    the symmetric matrix is vectorized with the unweighted upper triangle,
    including the diagonal.
    """

    def __init__(
        self,
        dims: tuple[int, ...] = (100, 100),
        out_dim: int = 1,
        fc_layer_no: int = 1,
        fc_hidden_dim: int = 100,
        fc_dropout: float = 0.5,
        reeig_epsilon: float = DEFAULT_SPDNET_REEIG_EPS,
        logeig_double_precision: bool = False,
    ):
        super().__init__()
        self.dims = tuple(dims)
        if len(self.dims) < 2 or len(self.dims) % 2 != 0:
            raise ValueError("dims must contain one or more (input_dim, output_dim) BiMap pairs")
        self.out_dim = out_dim
        self.fc_layer_no = fc_layer_no
        self.fc_hidden_dim = fc_hidden_dim
        self.fc_dropout = fc_dropout
        self.reeig_epsilon = reeig_epsilon
        self.logeig_double_precision = logeig_double_precision

        # Paper §2.5.2: SPDNet uses unweighted upper-triangle features.
        upper_idx = torch.triu_indices(self.dims[-1], self.dims[-1])
        self.register_buffer("triu_i", upper_idx[0])
        self.register_buffer("triu_j", upper_idx[1])

        # Keep these attribute names for compatibility with existing checkpoints.
        self.BiMap_Block = self._make_bimap_block(layer_num=len(self.dims) // 2)
        # Source behavior: apply one extra ReEig guard after the BiMap block.
        # This is a numerical safeguard beyond the layer list written in the paper.
        self.reeig_guard = ReEig(threshold=reeig_epsilon)
        # Keep LogEig as a matrix transform; vectorization is handled below so
        # the paper's unweighted upper-triangle convention is explicit here.
        self.logeig = LogEig(upper=False, flatten=False)
        self.fc = self._make_fc_block()

    def _make_bimap_block(self, layer_num: int) -> nn.Sequential:
        layers: list[nn.Module] = []

        for i in range(max(layer_num - 1, 0)):
            dim_in, dim_out = self.dims[2 * i], self.dims[2 * i + 1]
            layers.append(
                BiMap(
                    in_features=dim_in,
                    out_features=dim_out,
                )
            )
            layers.append(ReEig(threshold=self.reeig_epsilon))

        dim_in, dim_out = self.dims[-2], self.dims[-1]
        layers.append(
            BiMap(
                in_features=dim_in,
                out_features=dim_out,
            )
        )
        layers.append(ReEig(threshold=self.reeig_epsilon))
        return nn.Sequential(*layers)

    def _vectorize_upper_triangle(self, matrices: torch.Tensor) -> torch.Tensor:
        """Return unweighted upper-triangle entries, diagonal included."""
        return matrices[:, self.triu_i, self.triu_j]

    def _make_fc_block(self) -> nn.Sequential:
        feature_count = self.dims[-1] * (self.dims[-1] + 1) // 2
        layers: list[nn.Module] = [
            nn.Linear(feature_count, self.fc_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.fc_hidden_dim),
            nn.Dropout(self.fc_dropout),
        ]
        for _ in range(self.fc_layer_no - 1):
            layers += [
                nn.Linear(self.fc_hidden_dim, self.fc_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.fc_hidden_dim),
                nn.Dropout(self.fc_dropout),
            ]
        layers.append(nn.Linear(self.fc_hidden_dim, self.out_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.BiMap_Block(x)
        x = 0.5 * (x + x.transpose(-1, -2))
        x = self.reeig_guard(x)

        if self.logeig_double_precision:
            x = self.logeig(x.double()).float()
        else:
            x = self.logeig(x)

        x = self._vectorize_upper_triangle(x)
        return self.fc(x)
