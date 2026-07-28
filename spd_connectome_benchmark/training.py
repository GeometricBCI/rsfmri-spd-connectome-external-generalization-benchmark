"""Run-level randomness and conservative training-loop guards."""

from __future__ import annotations

import random

import numpy as np


def set_global_random_seed(seed: int) -> None:
    """Set the run-level Python, NumPy, and Torch random seeds."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_nonempty_training_batches(
    n_samples: int,
    batch_size: int,
    *,
    drop_last: bool,
) -> int:
    """Return batch count or fail before a zero-update training loop."""
    n_samples = int(n_samples)
    batch_size = int(batch_size)
    if n_samples < 1 or batch_size < 1:
        raise ValueError("n_samples and batch_size must be positive")
    n_batches = (
        n_samples // batch_size
        if drop_last
        else (n_samples + batch_size - 1) // batch_size
    )
    if n_batches == 0:
        raise ValueError(
            "Training would produce zero batches because drop_last=True and "
            f"n_samples ({n_samples}) is smaller than batch_size ({batch_size}). "
            "Choose a smaller batch size; the release does not silently change "
            "the established drop_last behavior."
        )
    return n_batches
