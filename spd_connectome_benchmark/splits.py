"""Subject-safe cross-validation split construction."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from spd_connectome_benchmark.config import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_SIZE,
)


def make_groupkfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """Create subject-grouped K-fold splits with legacy ``R1..RK`` labels."""
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(X, y, groups=groups))
    fold_names = [f"R{i}" for i in range(1, len(splits) + 1)]
    return splits, fold_names


def make_lodo_splits(
    dataset_ids: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]:
    """Create leave-one-dataset-out splits in pooled scan order."""
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    fold_names: list[str] = []
    for dataset in pd.unique(dataset_ids):
        test_idx = np.where(dataset_ids == dataset)[0]
        train_idx = np.where(dataset_ids != dataset)[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
        fold_names.append(f"TEST_{dataset}")
    return splits, fold_names


def split_train_validation_by_group(
    train_idx: np.ndarray,
    groups: np.ndarray,
    val_size: float = DEFAULT_VALIDATION_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve grouped validation data inside one outer training fold."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    sub_train, sub_val = next(splitter.split(train_idx, groups=groups[train_idx]))
    return train_idx[sub_train], train_idx[sub_val]


def dispatch_cv_splits(
    cv: str,
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    dataset_ids: np.ndarray,
    n_splits: int,
) -> Mapping[str, tuple[list[tuple[np.ndarray, np.ndarray]], list[str]]]:
    """Dispatch the stable CLI's CV names to existing split implementations."""
    protocol = str(cv).strip().lower()
    if protocol == "gkf":
        protocol = "kfold"
    if protocol not in {"kfold", "lodo", "both"}:
        raise ValueError(f"Unsupported cross-validation protocol: {cv!r}")

    result = {}
    if protocol in {"kfold", "both"}:
        result["kfold"] = make_groupkfold_splits(X, y, groups, n_splits)
    if protocol in {"lodo", "both"}:
        result["lodo"] = make_lodo_splits(dataset_ids)
    return result
