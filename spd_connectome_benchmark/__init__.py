"""Reusable code for the rs-fMRI SPD connectome benchmark.

The package root stays deliberately lightweight so configuration resolution
and ``--dry-run`` do not import Torch, SPD layers, or participant-data loaders.
Legacy root-level exports are loaded lazily for backward compatibility.
"""

from importlib import import_module

from spd_connectome_benchmark.config import PAPER_DATASETS

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


def __getattr__(name: str):
    if name in {
        "estimate_connectome_matrices",
        "vectorize_correlation_matrices",
        "vectorize_correlation_upper",
    }:
        return getattr(import_module("spd_connectome_benchmark.connectomes"), name)
    if name in {"DEFAULT_SPDNET_REEIG_EPS", "SPDNetRegressor"}:
        return getattr(import_module("spd_connectome_benchmark.models"), name)
    raise AttributeError(name)
