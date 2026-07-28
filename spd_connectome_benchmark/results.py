"""Aggregate result serialization and path-redacted run metadata."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def timestamp_tag() -> str:
    """Return the cross-platform timestamp format used in output filenames."""
    return time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def portable_result_reference(
    output_path: str | Path,
    *,
    output_root: str | Path,
) -> str:
    """Return an output-root-relative reference without leaking a local root."""
    path = Path(output_path).expanduser().resolve(strict=False)
    root = Path(output_root).expanduser().resolve(strict=False)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def save_metrics_csv(
    metrics: Mapping[str, Sequence[float]],
    n_splits: int,
    out_csv: str,
    elapsed: float,
) -> pd.DataFrame:
    """Save single-dataset fold metrics using legacy ``R1..Rn`` columns."""
    columns = [f"R{i}" for i in range(1, n_splits + 1)] + ["Avg", "Time(sec)"]
    frame = pd.DataFrame(columns=columns)
    for metric_name, values in metrics.items():
        metric_values = list(values)
        if len(metric_values) != n_splits:
            raise ValueError(
                f"Metric {metric_name!r} has {len(metric_values)} values; "
                f"expected {n_splits}."
            )
        frame.loc[metric_name] = metric_values + [
            float(np.mean(metric_values)),
            elapsed,
        ]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=True)
    save_run_metadata_sidecar(out_csv, elapsed=elapsed)
    return frame


def save_protocol_metrics_csv(
    fold_metrics: Mapping[str, Sequence[float]],
    fold_names: Sequence[str],
    elapsed: float,
    csv_path: str | Path,
) -> pd.DataFrame:
    """Save pooled metrics with protocol-specific legacy fold labels."""

    def average_metric(values: Iterable[float]) -> float:
        array = np.asarray(list(values), dtype=np.float64)
        if array.size == 0 or np.all(np.isnan(array)):
            return np.nan
        return float(np.nanmean(array))

    columns = list(fold_names) + ["Avg", "Time(sec)"]
    frame = pd.DataFrame(columns=columns)
    for metric_name, metric_values in fold_metrics.items():
        values = list(metric_values)
        if len(values) != len(fold_names):
            raise ValueError(
                f"Metric {metric_name!r} has {len(values)} values; "
                f"expected {len(fold_names)}."
            )
        frame.loc[metric_name] = values + [average_metric(values), elapsed]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=True)
    save_run_metadata_sidecar(csv_path, elapsed=elapsed)
    print("Saved:", csv_path)
    return frame


def _dependency_versions() -> dict[str, str]:
    packages = [
        "spd-learn",
        "torch",
        "pyriemann",
        "scikit-learn",
        "nilearn",
        "nibabel",
        "neuroHarmonize",
        "neuroCombat",
        "statsmodels",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "seaborn",
    ]
    versions = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


_PATH_ARGUMENTS = {
    "--config",
    "--data_root",
    "--data-root",
    "--input-root",
    "--output-dir",
    "--output-root",
    "--results_folder",
    "--results_folder_root",
    "--weights_folder_path",
    "--weights_folder_root",
    "--pkl_dir",
    "--raw_data_dir",
    "--raw_abide_dir",
    "--adni_adnidod_raw_dir",
    "--oasis3_raw_dir",
    "--table_dir",
    "--single_results_dir",
    "--pooled_results_dir",
    "--ablation_results_dir",
    "--result_figure_dir",
    "--out_dir",
}

_MULTI_PATH_ARGUMENTS = {"--harmonized_spd_cache_dirs"}


def _redacted_command(argv: Sequence[str]) -> list[str]:
    if not argv:
        return []
    redacted = [Path(argv[0]).name]
    redact_next = False
    redact_until_option = False
    for argument in argv[1:]:
        if redact_until_option and not argument.startswith("-"):
            redacted.append("<redacted-path>")
            continue
        if redact_until_option:
            redact_until_option = False
        if redact_next:
            redacted.append("<redacted-path>")
            redact_next = False
            continue
        option, separator, _ = argument.partition("=")
        if option in _MULTI_PATH_ARGUMENTS:
            redacted.append(
                f"{option}=<redacted-path>" if separator else argument
            )
            redact_until_option = True
            continue
        if option in _PATH_ARGUMENTS:
            if separator:
                redacted.append(f"{option}=<redacted-path>")
            else:
                redacted.append(argument)
                redact_next = True
        else:
            redacted.append(argument)
    return redacted


def save_run_metadata_sidecar(
    csv_path: str | Path,
    *,
    elapsed: float | None = None,
) -> Path:
    """Write a path-redacted reproducibility sidecar next to a result CSV."""
    csv_path = Path(csv_path)
    sidecar_path = csv_path.with_suffix(csv_path.suffix + ".metadata.json")
    environment_names = (
        "RSFMRI_SPD_DATA_ROOT",
        "RSFMRI_SPD_RAW_DATA_DIR",
        "RSFMRI_SPD_ADNI_ADNIDOD_RAW_DIR",
        "RSFMRI_SPD_OASIS3_RAW_DIR",
        "RSFMRI_SPD_OUTPUT_ROOT",
        "RSFMRI_SPD_BENCHMARK_OUTPUT_ROOT",
    )
    metadata_payload = {
        "metadata_schema_version": 2,
        "created_at": timestamp_tag(),
        "elapsed_seconds": None if elapsed is None else float(elapsed),
        "command": _redacted_command(sys.argv),
        "python": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "environment_is_set": {
            name: bool(os.environ.get(name)) for name in environment_names
        },
        "dependencies": _dependency_versions(),
    }
    sidecar_path.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return sidecar_path
