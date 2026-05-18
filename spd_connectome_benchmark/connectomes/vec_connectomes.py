"""Covariance, correlation, and vector features for rs-fMRI connectomes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import OAS

from spd_connectome_benchmark.config import DEFAULT_COVARIANCE_EPS


def estimate_connectome_matrices(
    timeseries: Sequence[np.ndarray] | np.ndarray,
    normalize: bool = True,
    n_jobs: int = -1,
    eps: float = DEFAULT_COVARIANCE_EPS,
) -> np.ndarray:
    """Build regularized SPD covariance/correlation connectomes.

    Paper §2.4 and Supplementary Methods "Connectome Construction" describe:
    per-scan OAS, covariance jitter ``eps * I``, correlation normalization with
    standard-deviation floor ``sqrt(eps)``, and final correlation jitter.
    The original code fits ``sklearn.covariance.OAS`` independently for each
    scan and falls back to ``np.cov`` only if OAS raises. The paper text states
    standard OAS, so this fallback is intentionally documented here.
    """
    if isinstance(timeseries, np.ndarray) and timeseries.ndim == 2:
        timeseries = [timeseries]
    elif not isinstance(timeseries, (list, tuple)):
        raise ValueError("timeseries must be a list of 2D arrays or a single 2D array")

    def estimate_single(scan_timeseries: np.ndarray) -> np.ndarray:
        # Original behavior: per-scan OAS, with np.cov as an exception fallback.
        try:
            covariance = OAS(store_precision=False).fit(scan_timeseries).covariance_
        except Exception:
            covariance = np.cov(scan_timeseries, rowvar=False)
        covariance = 0.5 * (covariance + covariance.T)
        covariance = covariance + np.eye(covariance.shape[0]) * eps
        return covariance

    matrices = Parallel(n_jobs=n_jobs)(delayed(estimate_single)(t) for t in timeseries)
    covariances = np.stack(matrices, axis=0)

    if not normalize:
        return covariances

    diagonal_scale = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
    # Matches the paper description: floor each standard deviation at sqrt(eps).
    diagonal_scale = np.maximum(diagonal_scale, np.sqrt(eps))
    denominator = diagonal_scale[:, :, None] * diagonal_scale[:, None, :]

    correlations = covariances / denominator
    correlations = 0.5 * (correlations + np.transpose(correlations, (0, 2, 1)))
    correlations = correlations + np.eye(correlations.shape[1]) * eps
    return correlations


def vectorize_correlation_matrices(X: np.ndarray, include_diagonal: bool = False) -> np.ndarray:
    """Vectorize correlation matrices for Euclidean baselines.

    Paper §2.5.3: CorrVec uses off-diagonal upper-triangular entries and does
    not apply the tangent-space ``sqrt(2)`` off-diagonal weighting.
    """
    offset = 0 if include_diagonal else 1
    tri_i, tri_j = np.triu_indices(X.shape[1], k=offset)
    return X[:, tri_i, tri_j].astype(np.float64, copy=False)


def vectorize_correlation_upper(X: np.ndarray) -> np.ndarray:
    """Vectorize off-diagonal upper-triangle correlations for CorrVec."""
    return vectorize_correlation_matrices(X, include_diagonal=False).astype(np.float32, copy=False)
