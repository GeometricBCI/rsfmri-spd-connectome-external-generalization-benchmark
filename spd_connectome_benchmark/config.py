"""Project-wide constants for the SPD connectome benchmark."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def _discover_source_checkout() -> Path | None:
    """Return the source checkout root, if this package is imported from one."""
    source_candidate = PACKAGE_DIR.parent
    if (
        (source_candidate / "pyproject.toml").is_file()
        and (source_candidate / "spd_connectome_benchmark").is_dir()
    ):
        return source_candidate.resolve(strict=False)
    return None


SOURCE_CHECKOUT_ROOT = _discover_source_checkout()
PROJECT_ROOT = SOURCE_CHECKOUT_ROOT or Path.cwd().resolve(strict=False)

# Paper §2.1 and §2.7: the benchmark uses these six datasets.
PAPER_DATASETS = ("cobre", "adnidod", "camcan", "abide", "oasis3", "adni")
DEFAULT_POOLED_DATASETS = PAPER_DATASETS

# Prepared-data paths. The local default is a sibling of the checkout so that
# preparation cannot accidentally create participant-data directories in Git.
# Set RSFMRI_SPD_DATA_ROOT to another authorized external location as needed.
DEFAULT_DATA_ROOT = Path(
    os.environ.get("RSFMRI_SPD_DATA_ROOT", PROJECT_ROOT.parent / "rsfmri_spd_data")
).expanduser()
DEFAULT_ATLAS_NAME = "schaefer_100"
DEFAULT_ATLAS_DIR = DEFAULT_DATA_ROOT / f"atlas_{DEFAULT_ATLAS_NAME}"
DEFAULT_RAW_DATA_DIR = Path(
    os.environ.get("RSFMRI_SPD_RAW_DATA_DIR", DEFAULT_DATA_ROOT / "raw_data")
).expanduser()
DEFAULT_ADNI_ADNIDOD_RAW_DIR = Path(
    os.environ.get("RSFMRI_SPD_ADNI_ADNIDOD_RAW_DIR", DEFAULT_RAW_DATA_DIR)
).expanduser()
DEFAULT_OASIS3_RAW_DIR = Path(
    os.environ.get("RSFMRI_SPD_OASIS3_RAW_DIR", DEFAULT_RAW_DATA_DIR)
).expanduser()
DEFAULT_RESULTS_ROOT = Path(
    os.environ.get("RSFMRI_SPD_OUTPUT_ROOT", PROJECT_ROOT / "results")
).expanduser()
DEFAULT_TABLES_DIR = DEFAULT_RESULTS_ROOT / "tables"
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_ROOT / "figures"
DEFAULT_BENCHMARK_RESULTS_DIR = Path(
    os.environ.get(
        "RSFMRI_SPD_BENCHMARK_OUTPUT_ROOT",
        DEFAULT_RESULTS_ROOT / "benchmark_csv",
    )
).expanduser()
DEFAULT_SINGLE_RESULTS_DIR = DEFAULT_BENCHMARK_RESULTS_DIR / "single_dataset"
DEFAULT_POOLED_RESULTS_DIR = DEFAULT_BENCHMARK_RESULTS_DIR
DEFAULT_ABLATION_RESULTS_DIR = DEFAULT_BENCHMARK_RESULTS_DIR / "spdnet_ablation"
DEFAULT_MODEL_WEIGHTS_DIR = DEFAULT_RESULTS_ROOT / "model_weights"
DEFAULT_SINGLE_WEIGHTS_DIR = DEFAULT_MODEL_WEIGHTS_DIR / "single_dataset"
DEFAULT_POOLED_WEIGHTS_DIR = DEFAULT_MODEL_WEIGHTS_DIR / "pooled"
DEFAULT_ABLATION_WEIGHTS_DIR = DEFAULT_MODEL_WEIGHTS_DIR / "spdnet_ablation"

# Original experiment defaults used by the entry-point parsers.
# These constants mirror the latest paper where possible, but they are not
# silently changed when the original source code used a different default.
DEFAULT_COVARIANCE_EPS = 1e-5
DEFAULT_VALIDATION_SIZE = 0.1
DEFAULT_RANDOM_SEED = 1
DEFAULT_DEBUG_SAMPLE_COUNT = 50


def ensure_data_path_outside_project(
    path: Path | str,
    *,
    label: str = "data path",
) -> Path:
    """Resolve a data path and reject locations inside the Git checkout."""
    resolved = Path(path).expanduser().resolve(strict=False)
    project_root = PROJECT_ROOT.resolve(strict=False)
    try:
        resolved.relative_to(project_root)
    except ValueError:
        return resolved
    raise ValueError(
        f"{label} must be outside the Git repository: {resolved}. "
        "Choose an authorized external data directory."
    )
