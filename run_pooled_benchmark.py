"""Compatibility shim for the legacy pooled benchmark entry point.

This repository intentionally keeps the public CLI minimal, but the test suite
still imports the legacy module name as a compatibility target. Provide the
smallest API surface expected by the configuration tests without reintroducing
all historical experimental scripts.
"""

from __future__ import annotations

import argparse
from typing import Sequence


def args_parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Legacy pooled benchmark shim.")
    parser.add_argument("--DATASETS", nargs="+", default=[])
    parser.add_argument("--N_SPLITS", type=int, default=5)
    parser.add_argument("--harm_mode", default="none")
    parser.add_argument("--protocol", default="lodo")
    parser.add_argument("--algorithms", nargs="+", default=["ridge"])
    parser.add_argument("--atlas_name", default="schaefer_100")
    parser.add_argument("--task", default="Age")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rng_seed", type=int, default=42)
    parser.add_argument("--data_root", default=".")
    parser.add_argument("--results_folder", default="results")
    parser.add_argument("--weights_folder_path", default="results/model_weights")
    parser.add_argument("--ts_metric", default="riemann")
    parser.add_argument("--ridge_alphas", nargs="*", default=[0.1, 1.0])
    parser.add_argument("--dummy_strategy", default="mean")
    parser.add_argument("--no_make_tag", action="store_true")
    parser.add_argument("--debug", default=None)
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args(list(argv) if argv is not None else None)


def configure_logging(level: str) -> None:
    return None


def run_pooled_age_benchmarks(**kwargs):
    return None
