"""Shared command-line helpers for benchmark entry points."""

from __future__ import annotations

import argparse

import torch

from spd_connectome_benchmark.config import (
    DEFAULT_COVARIANCE_EPS,
    DEFAULT_DATA_ROOT,
    DEFAULT_POOLED_DATASETS,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VALIDATION_SIZE,
    PAPER_DATASETS,
)
from spd_connectome_benchmark.benchmark_tools.runtime import set_global_random_seed

RIDGE_ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def add_device_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-cuda", action="store_true", default=False)


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level.",
    )


def add_split_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--N_SPLITS",
        type=int,
        default=5,
        help="GroupKFold K. Paper §2.7 uses K=5.",
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=DEFAULT_VALIDATION_SIZE,
        help="Grouped validation fraction inside outer-train; paper uses 0.1.",
    )


def add_connectome_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cov_jobs", type=int, default=-1)
    parser.add_argument(
        "--cov_eps",
        type=float,
        default=DEFAULT_COVARIANCE_EPS,
        help="Connectome epsilon; paper uses 1e-5.",
    )


def add_spdnet_optimization_args(
    parser: argparse.ArgumentParser,
    *,
    train_batch_size: int,
    test_batch_size: int,
    train_batch_help: str | None = None,
    test_batch_help: str | None = None,
) -> None:
    parser.add_argument("--initial_lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=train_batch_size,
        help=train_batch_help,
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=test_batch_size,
        help=test_batch_help,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--clip_grad", type=float, default=1.0)


def add_spdnet_head_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fc_layer_no", type=int, default=1)
    parser.add_argument("--fc_hidden_dim", type=int, default=100)
    parser.add_argument("--fc_dropout", type=float, default=0.5)


def add_pooled_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--DATASETS",
        nargs="+",
        default=list(DEFAULT_POOLED_DATASETS),
        help=f"Paper datasets to pool. Default: {', '.join(PAPER_DATASETS)}.",
    )


def add_common_data_args(parser: argparse.ArgumentParser, *, atlas_arg: str) -> None:
    add_data_root_arg(parser)
    parser.add_argument(
        atlas_arg,
        type=str,
        default="schaefer_100",
        help="Atlas name. Paper uses Schaefer-100.",
    )
    parser.add_argument("--task", type=str, default="Age")
    parser.add_argument("--debug", type=int, default=None)
    parser.add_argument("--rng_seed", type=int, default=42)


def add_data_root_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DEFAULT_DATA_ROOT),
        help=(
            "Root containing atlas_<atlas_name>/*_X_y.pkl files. "
            "Defaults to RSFMRI_SPD_DATA_ROOT or ./data."
        ),
    )


def add_ridge_args(parser: argparse.ArgumentParser, *, include_n_jobs: bool) -> None:
    parser.add_argument("--ridge_inner_splits", type=int, default=5)
    if include_n_jobs:
        parser.add_argument("--ridge_n_jobs", type=int, default=1)
    parser.add_argument("--ridge_alphas", nargs="+", type=float, default=RIDGE_ALPHA_GRID)


def resolve_torch_device(no_cuda: bool) -> torch.device:
    use_cuda = (not no_cuda) and torch.cuda.is_available()
    return torch.device("cuda" if use_cuda else "cpu")


def finalize_single_dataset_runtime(args: argparse.Namespace) -> None:
    """Apply runtime side effects expected by the single-dataset script."""
    set_global_random_seed(args.seed)
    args.device = resolve_torch_device(args.no_cuda)
