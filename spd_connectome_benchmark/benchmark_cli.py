"""Unified, safe command-line entry point for benchmark execution."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from spd_connectome_benchmark import __version__
from spd_connectome_benchmark.configuration import (
    BenchmarkConfig,
    ConfigurationError,
    resolve_config,
)
from spd_connectome_benchmark.config import PROJECT_ROOT, SOURCE_CHECKOUT_ROOT
from spd_connectome_benchmark.protocols import harmonization_policy


@dataclass(frozen=True)
class ValidationReport:
    """Filename-level validation that never opens participant-level inputs."""

    missing_inputs: tuple[Path, ...]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_inputs and not self.errors


def validate_run_plan(config: BenchmarkConfig) -> ValidationReport:
    """Validate names and required prepared filenames without reading data."""
    missing = tuple(path for path in config.expected_input_paths() if not path.is_file())
    errors: list[str] = []
    if config.cv in {"lodo", "both"} and len(config.datasets) < 2:
        errors.append("LODO requires at least two datasets.")
    return ValidationReport(missing_inputs=missing, errors=tuple(errors))


def render_run_plan(
    config: BenchmarkConfig,
    report: ValidationReport,
) -> str:
    """Render a stable, machine-readable dry-run report."""
    logical_paths = {
        path: (Path(f"atlas_{config.atlas}") / path.name).as_posix()
        for path in config.expected_input_paths()
    }
    payload = {
        "mode": "dry-run" if config.dry_run else "validation",
        "valid": report.ok,
        "resolved_config": config.as_dict(include_local_paths=False),
        "selected_datasets": list(config.datasets),
        "required_inputs": [
            {
                "dataset": dataset,
                "path": logical_paths[path],
                "exists": path.is_file(),
            }
            for dataset, path in zip(config.datasets, config.expected_input_paths())
        ],
        "experiments": list(config.experiment_matrix()),
        "errors": list(report.errors),
        "missing_inputs": [
            logical_paths[path] for path in report.missing_inputs
        ],
        "side_effects": {
            "participant_files_opened": False,
            "output_directories_created": False,
            "models_run": False,
        },
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def _git_code_state() -> dict[str, str | bool | None]:
    """Return path-free source revision information when Git is available."""
    if SOURCE_CHECKOUT_ROOT is None:
        return {
            "package_version": __version__,
            "git_commit": None,
            "tracked_worktree_dirty": None,
            "untracked_worktree_dirty": None,
            "worktree_dirty": None,
        }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status_entries = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {
            "package_version": __version__,
            "git_commit": None,
            "tracked_worktree_dirty": None,
            "untracked_worktree_dirty": None,
            "worktree_dirty": None,
        }
    untracked_dirty = any(entry.startswith("?? ") for entry in status_entries)
    tracked_dirty = any(
        not entry.startswith("?? ") for entry in status_entries
    )
    return {
        "package_version": __version__,
        "git_commit": revision or None,
        "tracked_worktree_dirty": tracked_dirty,
        "untracked_worktree_dirty": untracked_dirty,
        "worktree_dirty": tracked_dirty or untracked_dirty,
    }


def _path_free_execution_parameters(args: object) -> dict[str, object]:
    """Serialize all resolved legacy runner arguments except local paths."""
    parameters: dict[str, object] = {}
    for name, value in sorted(vars(args).items()):
        normalized_name = name.lower()
        if any(
            token in normalized_name
            for token in ("path", "dir", "root", "folder")
        ):
            continue
        if isinstance(value, tuple):
            value = list(value)
        parameters[name] = value
    return parameters


def write_resolved_config_manifest(
    config: BenchmarkConfig,
    *,
    legacy_args: object,
) -> Path:
    """Persist a path-free selection and resolved-execution manifest."""
    protocols = ("kfold", "lodo") if config.cv == "both" else (config.cv,)
    logical_config = {
        "task": config.task,
        "target": config.target,
        "models": list(config.models),
        "cv": config.cv,
        "folds": config.folds,
        "harmonization": config.harmonization,
        "atlas": config.atlas,
        "datasets": list(config.datasets),
        "seed": config.seed,
        "data_shuffle_seed": config.data_shuffle_seed,
    }
    execution_parameters = _path_free_execution_parameters(legacy_args)
    code_state = _git_code_state()
    fingerprint_input = {
        "selection": logical_config,
        "execution_parameters": execution_parameters,
        "code_state": code_state,
    }
    canonical = json.dumps(
        fingerprint_input,
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    payload = {
        "run_config_schema_version": 1,
        "config_fingerprint_sha256": fingerprint,
        "selection": logical_config,
        "resolved_execution_parameters": execution_parameters,
        "code_state": code_state,
        "required_input_filenames": [
            path.name for path in config.expected_input_paths()
        ],
        "harmonization_policies": {
            protocol: {
                "fit_scope": harmonization_policy(protocol).fit_scope,
                "apply_to_test": harmonization_policy(protocol).apply_to_test,
                "test_target_used_by_preprocessor": (
                    harmonization_policy(protocol).test_target_used_by_preprocessor
                ),
                "label": harmonization_policy(protocol).label,
            }
            for protocol in protocols
        },
    }
    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config.output_root / f"run_config_{fingerprint[:12]}.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest_path


def dispatch_run(config: BenchmarkConfig) -> None:
    """Translate the stable config into the legacy pooled benchmark runner."""
    # Import only after filename validation. This keeps --dry-run independent
    # from Torch, Nilearn, PyRiemann, and pickle-based dataset loaders.
    import run_pooled_benchmark

    protocol = {"kfold": "gkf", "lodo": "lodo", "both": "both"}[config.cv]
    weight_dir = config.output_root / "model_weights"
    argv = [
        "--DATASETS",
        *config.datasets,
        "--N_SPLITS",
        str(config.folds),
        "--harm_mode",
        config.harmonization,
        "--protocol",
        protocol,
        "--algorithms",
        *config.models,
        "--atlas_name",
        config.atlas,
        "--task",
        config.target,
        "--seed",
        str(config.seed),
        "--rng_seed",
        str(config.data_shuffle_seed),
        "--data_root",
        str(config.input_root),
        "--results_folder",
        str(config.output_root),
        "--weights_folder_path",
        str(weight_dir),
    ]
    args = run_pooled_benchmark.args_parser(argv)
    write_resolved_config_manifest(config, legacy_args=args)
    run_pooled_benchmark.configure_logging(args.log_level)
    run_pooled_benchmark.run_pooled_age_benchmarks(
        args=args,
        datasets=tuple(args.DATASETS),
        atlas_name=args.atlas_name,
        task=args.task,
        debug=args.debug,
        rng_seed=args.rng_seed,
        ts_metric=args.ts_metric,
        ridge_alphas=tuple(args.ridge_alphas),
        dummy_strategy=args.dummy_strategy,
        make_tag=not args.no_make_tag,
        algorithms=tuple(args.algorithms),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve, validate, optionally display, and execute a benchmark run."""
    try:
        config = resolve_config(argv)
        report = validate_run_plan(config)
    except ConfigurationError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if config.dry_run:
        print(render_run_plan(config, report))
        return 0 if report.ok else 2

    if not report.ok:
        print(render_run_plan(config, report), file=sys.stderr)
        print(
            "Input validation failed. No participant files were opened and no "
            "output directories were created.",
            file=sys.stderr,
        )
        return 2

    dispatch_run(config)
    return 0
