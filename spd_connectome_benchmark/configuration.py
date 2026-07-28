"""Configuration loading for the unified benchmark command.

Values are resolved with the documented precedence:

1. explicit command-line values
2. YAML configuration
3. environment variables
4. documented local defaults
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from spd_connectome_benchmark.config import (
    DEFAULT_RANDOM_SEED,
    PAPER_DATASETS,
    PROJECT_ROOT,
)
from spd_connectome_benchmark.datasets import (
    canonical_atlas_name,
    canonical_dataset_names,
    prepared_dataset_path,
)

MODEL_CHOICES = ("ridge", "corr_ridge", "spdnet", "dummy")
CV_CHOICES = ("kfold", "lodo", "both")
HARMONIZATION_CHOICES = ("none", "harm", "both")
TASK_CHOICES = ("regression",)

_CONFIG_KEYS = {
    "task",
    "target",
    "model",
    "models",
    "cv",
    "folds",
    "harm",
    "harmonization",
    "atlas",
    "datasets",
    "seed",
    "data_shuffle_seed",
    "input_root",
    "data_root",
    "output_root",
    "output_dir",
    "dry_run",
}

_ENVIRONMENT_FIELDS = {
    "task": ("RSFMRI_SPD_TASK",),
    "target": ("RSFMRI_SPD_TARGET",),
    "models": ("RSFMRI_SPD_MODELS", "RSFMRI_SPD_MODEL"),
    "cv": ("RSFMRI_SPD_CV",),
    "folds": ("RSFMRI_SPD_FOLDS",),
    "harmonization": ("RSFMRI_SPD_HARMONIZATION", "RSFMRI_SPD_HARM"),
    "atlas": ("RSFMRI_SPD_ATLAS",),
    "datasets": ("RSFMRI_SPD_DATASETS",),
    "seed": ("RSFMRI_SPD_SEED",),
    "data_shuffle_seed": ("RSFMRI_SPD_DATA_SHUFFLE_SEED",),
    "input_root": ("RSFMRI_SPD_DATA_ROOT",),
    "output_root": ("RSFMRI_SPD_BENCHMARK_OUTPUT_ROOT",),
}


class ConfigurationError(ValueError):
    """Raised when a benchmark configuration is malformed or unsupported."""


@dataclass(frozen=True)
class BenchmarkConfig:
    """Fully resolved configuration for the supported age-regression benchmark."""

    task: str
    target: str
    models: tuple[str, ...]
    cv: str
    folds: int
    harmonization: str
    atlas: str
    datasets: tuple[str, ...]
    seed: int
    data_shuffle_seed: int
    input_root: Path
    output_root: Path
    dry_run: bool = False
    config_file: Path | None = None
    sources: Mapping[str, str] = field(default_factory=dict, repr=False, compare=False)

    def expected_input_paths(self) -> tuple[Path, ...]:
        """Return required prepared-file paths without opening them."""
        return tuple(
            prepared_dataset_path(self.input_root, self.atlas, dataset)
            for dataset in self.datasets
        )

    def experiment_matrix(self) -> tuple[dict[str, str], ...]:
        """List model/protocol/harmonization combinations to be executed."""
        protocols = ("kfold", "lodo") if self.cv == "both" else (self.cv,)
        harmonization_modes = (
            ("none", "harm") if self.harmonization == "both" else (self.harmonization,)
        )
        return tuple(
            {
                "task": self.task,
                "target": self.target,
                "model": model,
                "cv": protocol,
                "harmonization": harm,
            }
            for model in self.models
            for protocol in protocols
            for harm in harmonization_modes
        )

    def as_dict(
        self,
        *,
        include_sources: bool = True,
        include_local_paths: bool = True,
    ) -> dict[str, Any]:
        """Return a JSON-serializable resolved configuration."""
        payload: dict[str, Any] = {
            "task": self.task,
            "target": self.target,
            "models": list(self.models),
            "cv": self.cv,
            "folds": self.folds,
            "harmonization": self.harmonization,
            "atlas": self.atlas,
            "datasets": list(self.datasets),
            "seed": self.seed,
            "data_shuffle_seed": self.data_shuffle_seed,
            "input_root": (
                str(self.input_root)
                if include_local_paths
                else "<redacted-local-path>"
            ),
            "output_root": (
                str(self.output_root)
                if include_local_paths
                else "<redacted-local-path>"
            ),
            "dry_run": self.dry_run,
            "config_file": (
                None
                if self.config_file is None
                else (
                    str(self.config_file)
                    if include_local_paths
                    else "<redacted-local-path>"
                )
            ),
        }
        if include_sources:
            payload["sources"] = dict(self.sources)
        return payload


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the stable, configuration-aware benchmark CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the configuration-driven external-generalization benchmark. "
            "The current release supports chronological-age regression only."
        )
    )
    parser.add_argument("--config", type=Path, help="YAML configuration file.")
    parser.add_argument("--task", choices=TASK_CHOICES, default=argparse.SUPPRESS)
    parser.add_argument("--target", default=argparse.SUPPRESS)
    parser.add_argument(
        "--model",
        "--models",
        dest="models",
        nargs="+",
        default=argparse.SUPPRESS,
        help=f"One or more supported models: {', '.join(MODEL_CHOICES)}.",
    )
    parser.add_argument(
        "--cv",
        choices=CV_CHOICES,
        default=argparse.SUPPRESS,
        help="kfold means subject-grouped K-fold; lodo means leave-one-dataset-out.",
    )
    parser.add_argument("--folds", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--harm",
        "--harmonization",
        dest="harmonization",
        choices=HARMONIZATION_CHOICES,
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--atlas", default=argparse.SUPPRESS)
    parser.add_argument("--datasets", nargs="+", default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--data-shuffle-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--input-root",
        "--data-root",
        dest="input_root",
        type=Path,
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-dir",
        "--output-root",
        dest="output_root",
        type=Path,
        default=argparse.SUPPRESS,
    )
    dry_run_group = parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Resolve configuration and validate required filenames without "
            "opening participant-level files or running a model."
        ),
    )
    dry_run_group.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        default=argparse.SUPPRESS,
        help="Disable a dry-run value inherited from YAML or the environment.",
    )
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigurationError(
            "Configuration files must use a .yaml or .yml extension; "
            "non-YAML files are never opened as configuration."
        )
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - covered in dependency smoke checks
        raise ConfigurationError(
            "PyYAML is required to read --config files; install the project requirements."
        ) from exc

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError("The configuration root must be a YAML mapping.")
    unknown = sorted(set(payload) - _CONFIG_KEYS)
    if unknown:
        raise ConfigurationError(f"Unknown configuration keys: {', '.join(unknown)}")
    return dict(payload)


def _first_environment_value(
    environ: Mapping[str, str],
    names: Sequence[str],
) -> str | None:
    for name in names:
        value = environ.get(name)
        if value is not None and value != "":
            return value
    return None


def _environment_layer(environ: Mapping[str, str]) -> dict[str, Any]:
    layer: dict[str, Any] = {}
    for field_name, variable_names in _ENVIRONMENT_FIELDS.items():
        value = _first_environment_value(environ, variable_names)
        if value is not None:
            layer[field_name] = value
    if "output_root" not in layer:
        shared_results_root = _first_environment_value(
            environ,
            ("RSFMRI_SPD_OUTPUT_ROOT",),
        )
        if shared_results_root is not None:
            layer["output_root"] = Path(shared_results_root) / "benchmark_csv"
    return layer


def _split_values(value: Any, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        values = tuple(part for part in value.replace(",", " ").split() if part)
    elif isinstance(value, (list, tuple)):
        values = tuple(str(part) for part in value)
    else:
        raise ConfigurationError(f"{field_name} must be a string or list.")
    if not values:
        raise ConfigurationError(f"{field_name} must not be empty.")
    return values


def _normalize_model(value: str) -> str:
    model = str(value).strip().lower().replace("-", "_")
    if model not in MODEL_CHOICES:
        raise ConfigurationError(
            f"Unsupported model {value!r}; choose from {', '.join(MODEL_CHOICES)}."
        )
    return model


def _parse_boolean(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ConfigurationError(f"{field_name} must be a boolean.")


def _normalize_layer_aliases(layer: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(layer)
    for alias, canonical in (
        ("model", "models"),
        ("harm", "harmonization"),
        ("data_root", "input_root"),
        ("output_dir", "output_root"),
    ):
        if alias in normalized:
            if canonical in normalized:
                raise ConfigurationError(
                    f"Specify only one of {alias!r} and {canonical!r}."
                )
            normalized[canonical] = normalized.pop(alias)
    return normalized


def _resolve_path(value: Any, *, base_dir: Path) -> Path:
    if isinstance(value, bool) or not isinstance(value, (str, os.PathLike)):
        raise ConfigurationError("Path values must be non-empty strings.")
    if isinstance(value, str) and not value.strip():
        raise ConfigurationError("Path values must be non-empty strings.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve(strict=False)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _normalize_layer_paths(layer: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    normalized = _normalize_layer_aliases(layer)
    for name in ("input_root", "output_root"):
        if name in normalized:
            normalized[name] = _resolve_path(normalized[name], base_dir=base_dir)
    return normalized


def _apply_layer(
    values: dict[str, Any],
    sources: dict[str, str],
    layer: Mapping[str, Any],
    *,
    source: str,
) -> None:
    for key, value in layer.items():
        values[key] = value
        sources[key] = source


def _validated_config(
    values: Mapping[str, Any],
    *,
    config_file: Path | None,
    sources: Mapping[str, str],
) -> BenchmarkConfig:
    task = str(values["task"]).strip().lower()
    if task not in TASK_CHOICES:
        raise ConfigurationError(
            "This repository currently implements regression only; "
            "classification and combined tasks are not available."
        )
    target = str(values["target"]).strip()
    if target.lower() != "age":
        raise ConfigurationError(
            "This release implements chronological age (target: Age) only."
        )

    models = tuple(_normalize_model(value) for value in _split_values(values["models"], field_name="models"))
    if len(set(models)) != len(models):
        raise ConfigurationError("Model selections must not contain duplicates.")

    cv = str(values["cv"]).strip().lower()
    if cv == "gkf":
        cv = "kfold"
    if cv not in CV_CHOICES:
        raise ConfigurationError(f"Unsupported cv {cv!r}.")

    harmonization = str(values["harmonization"]).strip().lower()
    if harmonization not in HARMONIZATION_CHOICES:
        raise ConfigurationError(f"Unsupported harmonization mode {harmonization!r}.")

    try:
        folds = int(values["folds"])
        seed = int(values["seed"])
        data_shuffle_seed = int(values["data_shuffle_seed"])
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("folds and seeds must be integers.") from exc
    if folds < 2:
        raise ConfigurationError("folds must be at least 2.")
    if seed < 0 or data_shuffle_seed < 0:
        raise ConfigurationError("seeds must be non-negative.")

    try:
        datasets = canonical_dataset_names(
            _split_values(values["datasets"], field_name="datasets")
        )
        atlas = canonical_atlas_name(str(values["atlas"]))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    input_root = Path(values["input_root"]).resolve(strict=False)
    output_root = Path(values["output_root"]).resolve(strict=False)
    if _is_within(output_root, input_root) or _is_within(input_root, output_root):
        raise ConfigurationError(
            "input_root and output_root must not be equal or contain one "
            "another; benchmark artifacts must remain outside participant data."
        )
    project_root = PROJECT_ROOT.resolve(strict=False)
    if _is_within(input_root, project_root):
        raise ConfigurationError(
            "input_root must be outside the project directory so participant "
            "data cannot be placed in the Git checkout."
        )
    repository_results_root = (project_root / "results").resolve(strict=False)
    if _is_within(output_root, project_root) and not _is_within(
        output_root,
        repository_results_root,
    ):
        raise ConfigurationError(
            "An output_root inside the repository must be under results/ so "
            "generated artifacts remain excluded from Git."
        )

    return BenchmarkConfig(
        task=task,
        target="Age",
        models=models,
        cv=cv,
        folds=folds,
        harmonization=harmonization,
        atlas=atlas,
        datasets=datasets,
        seed=seed,
        data_shuffle_seed=data_shuffle_seed,
        input_root=input_root,
        output_root=output_root,
        dry_run=_parse_boolean(values.get("dry_run", False), field_name="dry_run"),
        config_file=config_file,
        sources=dict(sources),
    )


def resolve_config(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> BenchmarkConfig:
    """Parse and resolve CLI, YAML, environment, and default configuration."""
    environ = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else Path(cwd)
    parser = build_argument_parser()
    namespace = parser.parse_args(argv)
    cli_layer = vars(namespace).copy()
    config_value = cli_layer.pop("config", None)
    config_file = (
        _resolve_path(config_value, base_dir=cwd)
        if config_value is not None
        else None
    )

    defaults: dict[str, Any] = {
        "task": "regression",
        "target": "Age",
        "models": ("ridge",),
        "cv": "lodo",
        "folds": 5,
        "harmonization": "none",
        "atlas": "schaefer_100",
        "datasets": PAPER_DATASETS,
        "seed": DEFAULT_RANDOM_SEED,
        "data_shuffle_seed": 42,
        "input_root": (PROJECT_ROOT.parent / "rsfmri_spd_data").resolve(
            strict=False
        ),
        "output_root": (PROJECT_ROOT / "results" / "benchmark_csv").resolve(
            strict=False
        ),
        "dry_run": False,
    }
    values = dict(defaults)
    sources = {key: "default" for key in defaults}

    env_layer = _normalize_layer_paths(_environment_layer(environ), base_dir=cwd)
    _apply_layer(values, sources, env_layer, source="environment")

    if config_file is not None:
        yaml_layer = _normalize_layer_paths(
            _load_yaml(config_file),
            base_dir=config_file.parent,
        )
        _apply_layer(values, sources, yaml_layer, source="yaml")

    cli_layer = _normalize_layer_paths(cli_layer, base_dir=cwd)
    _apply_layer(values, sources, cli_layer, source="cli")

    return _validated_config(
        values,
        config_file=config_file,
        sources=sources,
    )


def config_json(
    config: BenchmarkConfig,
    *,
    include_local_paths: bool = False,
) -> str:
    """Serialize a resolved configuration deterministically and safely."""
    return json.dumps(
        config.as_dict(include_local_paths=include_local_paths),
        indent=2,
        sort_keys=True,
    )
