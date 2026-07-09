"""Project-wide constants for the SPD connectome benchmark."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# Paper §2.1 and §2.7: the benchmark uses these six datasets.
PAPER_DATASETS = ("cobre", "adnidod", "camcan", "abide", "oasis3", "adni")
DEFAULT_POOLED_DATASETS = PAPER_DATASETS

# Prepared-data paths. Download the processed benchmark archive and extract it
# under ``data/`` by default, or set RSFMRI_SPD_DATA_ROOT to another location.
DEFAULT_DATA_ROOT = Path(os.environ.get("RSFMRI_SPD_DATA_ROOT", PROJECT_ROOT / "data")).expanduser()
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
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_TABLES_DIR = DEFAULT_RESULTS_ROOT / "tables"
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_ROOT / "figures"
DEFAULT_BENCHMARK_RESULTS_DIR = DEFAULT_RESULTS_ROOT / "benchmark_csv"
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
