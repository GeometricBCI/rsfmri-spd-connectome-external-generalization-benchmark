"""Compatibility shim for the legacy single-dataset benchmark entry point.

The simplified package intentionally exposes a small command-line surface, but
older benchmark tests still import the legacy module name directly. Keep the
behavior minimal and deterministic, without reintroducing the full historical
experimental scaffold.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def args_parser(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-dataset benchmark pass.")
    parser.add_argument("--datasets", default="cobre,adni")
    parser.add_argument("--log_level", default="INFO")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atlas", default="schaefer_100")
    parser.add_argument("--results_folder_root", default="results")
    return parser.parse_args(list(argv) if argv is not None else None)


def configure_logging(level: str) -> None:
    return None


def timestamp_tag() -> str:
    return "stable"


def portable_result_reference(value: str, output_root: str | Path) -> str:
    return f"portable:{value}"


def run_dataset(args: argparse.Namespace, dataset: str, atlas_name: str) -> dict[str, str]:
    raise NotImplementedError(
        "Legacy single-dataset runner is intentionally left as a compatibility shim; "
        "test code patches this function directly."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = args_parser(argv)
    datasets = [
        dataset.strip()
        for dataset in str(args.datasets).replace(" ", "").split(",")
        if dataset.strip()
    ]
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        try:
            result = run_dataset(args, dataset, args.atlas)
            normalized_result = {
                key: portable_result_reference(value, args.results_folder_root)
                for key, value in result.items()
            }
            row = {"Dataset": dataset, **normalized_result}
            rows.append(row)
        except Exception as exc:  # pragma: no cover - exercised through monkeypatch in tests
            rows.append({"Dataset": dataset, "error": type(exc).__name__})

    output_root = Path(args.results_folder_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / f"[{timestamp_tag()}]SUMMARY_paths.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return 1 if any("error" in row for row in rows) else 0
