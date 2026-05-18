"""Reusable code for the rs-fMRI SPD connectome benchmark."""

from spd_connectome_benchmark.config import PAPER_DATASETS
from spd_connectome_benchmark.connectomes import (
    estimate_connectome_matrices,
    vectorize_correlation_matrices,
    vectorize_correlation_upper,
)
from spd_connectome_benchmark.models import DEFAULT_SPDNET_REEIG_EPS, SPDNetRegressor

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_SPDNET_REEIG_EPS",
    "PAPER_DATASETS",
    "SPDNetRegressor",
    "__version__",
    "estimate_connectome_matrices",
    "vectorize_correlation_matrices",
    "vectorize_correlation_upper",
]
