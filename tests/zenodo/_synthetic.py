from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import yaml


SYNTHETIC_DATASET = "abide"
SYNTHETIC_SAMPLE_UIDS = (
    "s00000000000000000000000000000001",
    "s00000000000000000000000000000002",
    "s00000000000000000000000000000003",
)
SYNTHETIC_SOURCE_BINDING_SHA256 = hashlib.sha256(
    b"runtime-generated synthetic source binding"
).hexdigest()


def correlation_matrices(
    *,
    seed: int = 17,
    n_samples: int = 3,
    n_regions: int = 4,
) -> np.ndarray:
    """Create deterministic, strictly positive-definite correlations."""

    rng = np.random.default_rng(seed)
    matrices = []
    for _ in range(n_samples):
        factors = rng.normal(size=(n_regions, n_regions + 2))
        covariance = factors @ factors.T + 0.5 * np.eye(n_regions)
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        matrices.append(correlation)
    return np.asarray(matrices, dtype=np.float64)


def write_tsv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False),
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
