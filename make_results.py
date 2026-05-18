"""Generate result tables and figures from prepared data and benchmark CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from spd_connectome_benchmark.config import (
    DEFAULT_ATLAS_DIR,
    DEFAULT_FIGURES_DIR,
    DEFAULT_POOLED_RESULTS_DIR,
    DEFAULT_RAW_DATA_DIR,
    DEFAULT_TABLES_DIR,
)
from spd_connectome_benchmark.analysis_outputs import (
    dataset_description,
    dataset_shift,
    lodo_tables,
)
from spd_connectome_benchmark.analysis_outputs.result_figures import (
    DEFAULT_ABLATION_RESULTS_DIR,
    DEFAULT_SINGLE_RESULTS_DIR,
    build_all_result_figures,
    configure_result_paths,
)

DEFAULT_RAW_ABIDE_DIR = DEFAULT_RAW_DATA_DIR / "ABIDE_pcp"


def _maybe_default(value: Path, include_defaults: bool) -> Path | str:
    return value if include_defaults else argparse.SUPPRESS


def _add_output_path_args(
    parser: argparse.ArgumentParser,
    *,
    include_defaults: bool = True,
) -> None:
    parser.add_argument(
        "--pkl_dir",
        type=Path,
        default=_maybe_default(DEFAULT_ATLAS_DIR, include_defaults),
        help="Directory with prepared atlas *_X_y.pkl files.",
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=_maybe_default(DEFAULT_TABLES_DIR, include_defaults),
        help="Directory for generated CSV tables.",
    )
    parser.add_argument(
        "--raw_abide_dir",
        type=Path,
        default=_maybe_default(DEFAULT_RAW_ABIDE_DIR, include_defaults),
        help="Optional ABIDE PCP cache used only for ABIDE sample accounting.",
    )
    parser.add_argument(
        "--single_results_dir",
        type=Path,
        default=_maybe_default(DEFAULT_SINGLE_RESULTS_DIR, include_defaults),
        help="Directory with within-dataset benchmark CSV results.",
    )
    parser.add_argument(
        "--pooled_results_dir",
        type=Path,
        default=_maybe_default(DEFAULT_POOLED_RESULTS_DIR, include_defaults),
        help="Directory with pooled benchmark CSV results.",
    )
    parser.add_argument(
        "--ablation_results_dir",
        type=Path,
        default=_maybe_default(DEFAULT_ABLATION_RESULTS_DIR, include_defaults),
        help="Directory with SPDNet ablation CSV results.",
    )
    parser.add_argument(
        "--result_figure_dir",
        type=Path,
        default=_maybe_default(DEFAULT_FIGURES_DIR, include_defaults),
        help="Directory for generated benchmark figures.",
    )


def _add_output_path_defaults(args: argparse.Namespace) -> None:
    for name, default in (
        ("pkl_dir", DEFAULT_ATLAS_DIR),
        ("table_dir", DEFAULT_TABLES_DIR),
        ("raw_abide_dir", DEFAULT_RAW_ABIDE_DIR),
        ("single_results_dir", DEFAULT_SINGLE_RESULTS_DIR),
        ("pooled_results_dir", DEFAULT_POOLED_RESULTS_DIR),
        ("ablation_results_dir", DEFAULT_ABLATION_RESULTS_DIR),
        ("result_figure_dir", DEFAULT_FIGURES_DIR),
    ):
        if not hasattr(args, name):
            setattr(args, name, default)


def _configure_result_output_paths(args: argparse.Namespace) -> None:
    configure_result_paths(
        single_results_dir=args.single_results_dir,
        pooled_results_dir=args.pooled_results_dir,
        ablation_results_dir=args.ablation_results_dir,
        result_figure_dir=args.result_figure_dir,
    )


def _run_dataset_description(args: argparse.Namespace) -> None:
    dataset_description.main(
        pkl_dir=args.pkl_dir,
        raw_abide_dir=args.raw_abide_dir,
        out_dir=args.table_dir,
    )


def _run_dataset_shift(args: argparse.Namespace) -> None:
    dataset_shift.main(
        pkl_dir=args.pkl_dir,
        table_dir=args.table_dir,
    )


def _run_lodo_tables(args: argparse.Namespace) -> None:
    lodo_tables.main(
        results_dir=args.pooled_results_dir,
        out_dir=args.table_dir,
    )


def _run_support_outputs(args: argparse.Namespace) -> None:
    # Keep Table 1, Figure 2, and Table 3 outputs together under results/.
    _run_dataset_description(args)
    _run_dataset_shift(args)
    _run_lodo_tables(args)


def _run_result_figures(args: argparse.Namespace) -> None:
    _configure_result_output_paths(args)
    build_all_result_figures()


def _run_all_outputs(args: argparse.Namespace) -> None:
    _run_support_outputs(args)
    _run_result_figures(args)


def _add_subcommand(
    subparsers: Any,
    name: str,
    help_text: str,
    runner: Callable[[argparse.Namespace], None],
) -> None:
    subparser = subparsers.add_parser(name, help=help_text)
    _add_output_path_args(subparser, include_defaults=False)
    subparser.set_defaults(run=runner)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    shared_parser = argparse.ArgumentParser(add_help=False)
    _add_output_path_args(shared_parser)
    parser = argparse.ArgumentParser(
        description=(
            "Generate result tables, descriptive figures, and benchmark "
            "result figures."
        ),
        parents=[shared_parser],
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_subcommand(
        subparsers,
        "support",
        "Generate dataset description tables, dataset-shift figures, and LODO summary tables.",
        _run_support_outputs,
    )
    _add_subcommand(
        subparsers,
        "dataset-description",
        "Generate dataset description and scan metadata tables.",
        _run_dataset_description,
    )
    _add_subcommand(
        subparsers,
        "dataset-shift",
        "Generate descriptive dataset-shift figures.",
        _run_dataset_shift,
    )
    _add_subcommand(
        subparsers,
        "lodo-tables",
        "Generate leave-one-dataset-out per-dataset summary tables.",
        _run_lodo_tables,
    )
    _add_subcommand(
        subparsers,
        "result-figures",
        "Generate benchmark result figures from existing result CSVs.",
        _run_result_figures,
    )
    _add_subcommand(
        subparsers,
        "all",
        "Generate support outputs and benchmark result figures.",
        _run_all_outputs,
    )

    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "support"
        args.run = _run_support_outputs
    _add_output_path_defaults(args)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
