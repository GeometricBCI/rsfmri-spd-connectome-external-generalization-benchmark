"""Typed validation helpers for synthetic and locally reconstructed inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidatedBenchmarkArrays:
    """Validated scan-level arrays used by a benchmark split."""

    connectomes: np.ndarray
    targets: np.ndarray
    dataset_labels: np.ndarray
    subject_groups: np.ndarray


def validate_timeseries(
    timeseries: Sequence[np.ndarray],
    *,
    expected_regions: int | None = None,
) -> tuple[np.ndarray, ...]:
    """Validate finite ``(timepoints, regions)`` scan arrays."""
    if not isinstance(timeseries, (list, tuple)) or not timeseries:
        raise ValueError("timeseries must be a non-empty list or tuple of 2D arrays")
    validated: list[np.ndarray] = []
    n_regions = expected_regions
    for index, scan in enumerate(timeseries):
        array = np.asarray(scan)
        if array.ndim != 2 or min(array.shape) < 2:
            raise ValueError(
                f"timeseries[{index}] must have shape (timepoints, regions)"
            )
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"timeseries[{index}] must have a floating-point dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"timeseries[{index}] contains NaN or infinity")
        if n_regions is None:
            n_regions = int(array.shape[1])
        if array.shape[1] != n_regions:
            raise ValueError("all scans must use the same atlas region count")
        validated.append(array)
    return tuple(validated)


def validate_spd_connectomes(
    connectomes: np.ndarray,
    *,
    expected_regions: int | None = None,
    symmetry_tolerance: float = 1e-7,
) -> np.ndarray:
    """Validate finite, symmetric positive-definite ``(N, P, P)`` matrices."""
    matrices = np.asarray(connectomes)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("connectomes must have shape (n_scans, n_regions, n_regions)")
    if matrices.shape[0] == 0 or matrices.shape[1] == 0:
        raise ValueError("connectomes must not be empty")
    if expected_regions is not None and matrices.shape[1] != expected_regions:
        raise ValueError(
            f"expected {expected_regions} atlas regions, got {matrices.shape[1]}"
        )
    if not np.issubdtype(matrices.dtype, np.floating):
        raise ValueError("connectomes must use a floating-point dtype")
    if not np.isfinite(matrices).all():
        raise ValueError("connectomes contain NaN or infinity")
    if not np.allclose(
        matrices,
        np.swapaxes(matrices, -1, -2),
        atol=symmetry_tolerance,
        rtol=0.0,
    ):
        raise ValueError("connectomes must be symmetric")
    minimum_eigenvalues = np.linalg.eigvalsh(matrices).min(axis=1)
    if np.any(minimum_eigenvalues <= 0.0):
        raise ValueError("connectomes must be positive definite")
    return matrices


def _validate_vector(
    values: np.ndarray,
    *,
    name: str,
    expected_length: int,
    numeric: bool,
    string_only: bool = False,
) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != expected_length:
        raise ValueError(f"{name} must be a 1D array with one value per scan")
    if numeric:
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must have a numeric dtype")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinity")
    else:
        for value in array:
            missing = pd.isna(value)
            if not isinstance(missing, (bool, np.bool_)) or bool(missing):
                raise ValueError(f"{name} must not contain missing labels")
            if string_only and not isinstance(value, (str, np.str_)):
                raise ValueError(
                    f"{name} must contain only non-empty string-like labels"
                )
            if isinstance(value, (str, np.str_)) and not str(value).strip():
                raise ValueError(f"{name} must not contain empty labels")
            try:
                hash(value)
            except TypeError as exc:
                raise ValueError(f"{name} labels must be hashable scalars") from exc
    return array


def validate_benchmark_arrays(
    connectomes: np.ndarray,
    targets: np.ndarray,
    dataset_labels: np.ndarray,
    subject_groups: np.ndarray,
    *,
    expected_regions: int | None = None,
) -> ValidatedBenchmarkArrays:
    """Validate the stable in-memory contract without participant metadata."""
    matrices = validate_spd_connectomes(
        connectomes,
        expected_regions=expected_regions,
    )
    n_scans = matrices.shape[0]
    return ValidatedBenchmarkArrays(
        connectomes=matrices,
        targets=_validate_vector(
            targets,
            name="targets",
            expected_length=n_scans,
            numeric=True,
        ),
        dataset_labels=_validate_vector(
            dataset_labels,
            name="dataset_labels",
            expected_length=n_scans,
            numeric=False,
            string_only=True,
        ),
        subject_groups=_validate_vector(
            subject_groups,
            name="subject_groups",
            expected_length=n_scans,
            numeric=False,
        ),
    )


def validate_split_disjointness(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_samples: int,
    subject_groups: np.ndarray | None = None,
) -> None:
    """Assert index and optional subject-group isolation for one outer split."""
    train = np.asarray(train_idx)
    test = np.asarray(test_idx)
    if train.ndim != 1 or test.ndim != 1 or not len(train) or not len(test):
        raise ValueError("train_idx and test_idx must be non-empty 1D arrays")
    if not np.issubdtype(train.dtype, np.integer) or not np.issubdtype(
        test.dtype, np.integer
    ):
        raise ValueError("split indices must be integers")
    if train.min() < 0 or test.min() < 0 or train.max() >= n_samples or test.max() >= n_samples:
        raise ValueError("split index is out of bounds")
    if np.unique(train).size != train.size or np.unique(test).size != test.size:
        raise ValueError("train_idx and test_idx must not contain duplicates")
    if np.intersect1d(train, test).size:
        raise ValueError("train and test indices overlap")
    if subject_groups is not None:
        groups = np.asarray(subject_groups)
        if groups.ndim != 1 or len(groups) != n_samples:
            raise ValueError("subject_groups must have one value per sample")
        if set(groups[train]).intersection(set(groups[test])):
            raise ValueError("subject groups overlap between train and test")
