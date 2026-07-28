"""Validate a staged Zenodo release and emit redacted audit reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .checksums import (
    iter_regular_files,
    read_sha256sums,
    verify_manifest,
    verify_sha256sums,
    write_manifest,
    write_sha256sums,
)
from .metadata import (
    MANUSCRIPT_TITLE,
    build_citation_cff,
    build_zenodo_record_metadata,
    contains_placeholder,
    metadata_is_complete,
)
from .privacy_scan import redact, scan_release
from .schemas import (
    CAMCAN_CONNECTOME_RELATIVE_PATH,
    CAMCAN_DATA_DICTIONARY_RELATIVE_PATH,
    CAMCAN_LICENSE_RELATIVE_PATH,
    CAMCAN_METADATA_RELATIVE_PATH,
    CANONICAL_REPOSITORY_URL,
    KNOWN_DATASETS,
    RESTRICTED_DATASETS,
    SAFE_UID_RE,
    SCHEMA_VERSION,
    Finding,
    ValidationResult,
    canonical_dataset,
    dataset_policy,
    is_portable_relative_path,
    load_release_bundle,
    load_yaml,
    participant_inventory_decision,
    participant_release_decision,
    read_tsv,
    release_connectome_relative_path,
    release_metadata_relative_path,
    validate_connectome_npz,
    validate_metadata_columns,
    validate_release_bundle_documents,
    validate_release_numeric_settings,
)


FORBIDDEN_SUFFIXES = {
    ".pkl",
    ".pickle",
    ".npy",
    ".nii",
    ".dcm",
    ".dicom",
    ".ima",
    ".mnc",
    ".mgz",
    ".mgh",
    ".nrrd",
    ".hdr",
    ".img",
    ".pt",
    ".pth",
    ".ckpt",
}
FORBIDDEN_NAMES = {
    "participants.tsv",
    "participants.json",
}
REQUIRED_ROOT_ITEMS = {
    "README.md",
    "DATASET_CARD.md",
    "LICENSES.md",
    "VERSION",
    "metadata",
    "data",
    "splits",
    "restricted_reconstruction",
    "configs",
    "benchmark_results",
    "reproducibility",
    "manifests",
}
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-[\dX]{4}$")
RAW_T1_FILENAME_RE = re.compile(
    r"(?i)(?:^|[_ .-])(?:t1w|t1[-_ ]?weighted|mprage)(?:[_ .-]|$)"
)
ROOT_FILES = {"README.md", "DATASET_CARD.md", "LICENSES.md", "VERSION", "CITATION.cff"}
METADATA_FILES = {
    "zenodo_record_metadata.json",
    "zenodo_form_values.md",
    "dataset_inventory.tsv",
    "data_dictionary.tsv",
    "provenance.tsv",
    "release_policy_snapshot.yaml",
    "manual_approvals_snapshot.yaml",
    "release_config_snapshot.yaml",
    "metadata_allowlist_snapshot.yaml",
    "forbidden_patterns_snapshot.yaml",
}
REPRODUCIBILITY_FILES = {
    "commands.md",
    "environment.yml",
    "requirements-lock.txt",
    "git_commit.txt",
    "software_versions.json",
    "repository_snapshot.json",
}
MANIFEST_FILES = {
    "manifest.tsv",
    "SHA256SUMS.txt",
    "validation_report.json",
    "validation_report.md",
}
REQUIRED_STATIC_DIRECTORIES = {
    "metadata",
    "data",
    "splits",
    "restricted_reconstruction",
    "restricted_reconstruction/adni",
    "restricted_reconstruction/adnidod",
    "restricted_reconstruction/oasis3",
    "configs",
    "configs/datasets",
    "configs/experiments",
    "configs/preprocessing",
    "benchmark_results",
    "benchmark_results/aggregate_metrics",
    "benchmark_results/statistical_tests",
    "benchmark_results/figure_source_data",
    "reproducibility",
    "manifests",
}
REQUIRED_STATIC_FILES = {
    *(f"metadata/{name}" for name in METADATA_FILES),
    *(f"reproducibility/{name}" for name in REPRODUCIBILITY_FILES),
    *(
        f"configs/datasets/{dataset}.yaml"
        for dataset in (
            "abide",
            "adni",
            "oasis3",
            "camcan",
            "cobre",
            "adnidod",
            "1000brains",
        )
    ),
    "configs/experiments/README.md",
    "configs/preprocessing/connectome.yaml",
    "data/README.md",
    *(
        f"restricted_reconstruction/{dataset}/{name}"
        for dataset in ("adni", "adnidod", "oasis3")
        for name in ("README.md", "selection_config.yaml", "reconstruction_commands.md")
    ),
    "manifests/manifest.tsv",
    "manifests/SHA256SUMS.txt",
}
RELEASE_TABLE_COLUMNS = {
    "dataset_inventory": [
        "dataset",
        "content_category",
        "access_category",
        "file_count",
        "sample_count",
        "atlas",
        "matrix_type",
        "release_policy_decision",
    ],
    "data_dictionary": [
        "field",
        "description",
        "data_type",
        "units",
        "nullable",
        "content_category",
        "access_category",
    ],
    "provenance": [
        "relative_path",
        "sha256",
        "source_category",
        "content_category",
        "repository_commit",
        "config_fingerprint",
        "software_version",
        "transformation",
    ],
}


def _allowed_release_file(relative: Path) -> bool:
    parts = relative.parts
    if relative in {
        CAMCAN_CONNECTOME_RELATIVE_PATH,
        CAMCAN_METADATA_RELATIVE_PATH,
        CAMCAN_DATA_DICTIONARY_RELATIVE_PATH,
        CAMCAN_LICENSE_RELATIVE_PATH,
    }:
        return True
    if len(parts) == 1:
        return parts[0] in ROOT_FILES
    if len(parts) == 2 and parts[0] == "metadata":
        return parts[1] in METADATA_FILES
    if relative.as_posix() == "data/README.md":
        return True
    if (
        len(parts) == 4
        and parts[0:2] == ("data", "public_connectomes")
        and parts[2] in (KNOWN_DATASETS - {"camcan"})
        and parts[3] == "connectomes.npz"
    ):
        return True
    if (
        len(parts) == 4
        and parts[0:2] == ("data", "public_metadata")
        and parts[2] in (KNOWN_DATASETS - {"camcan"})
        and parts[3] == "metadata.tsv"
    ):
        return True
    if relative.as_posix() == "splits/README.md":
        return True
    if (
        len(parts) == 3
        and parts[0] == "splits"
        and parts[1] in KNOWN_DATASETS
        and parts[2] == "splits.tsv"
    ):
        return True
    if (
        len(parts) == 3
        and parts[0] == "restricted_reconstruction"
        and parts[1] in {"adni", "adnidod", "oasis3"}
        and parts[2]
        in {"README.md", "selection_config.yaml", "reconstruction_commands.md"}
    ):
        return True
    if (
        len(parts) == 3
        and parts[0:2] == ("configs", "datasets")
        and parts[2].endswith(".yaml")
        and parts[2][:-5]
        in {"abide", "adni", "oasis3", "camcan", "cobre", "adnidod", "1000brains"}
    ):
        return True
    if relative.as_posix() in {
        "configs/experiments/README.md",
        "configs/preprocessing/connectome.yaml",
    }:
        return True
    if len(parts) >= 3 and parts[0] == "benchmark_results":
        if parts[1] not in {
            "aggregate_metrics",
            "statistical_tests",
            "figure_source_data",
        }:
            return False
        if len(parts) == 3 and parts[2] == "README.md":
            return True
        return (
            len(parts) == 4
            and parts[2] in KNOWN_DATASETS
            and Path(parts[3]).suffix.lower() in {".csv", ".tsv", ".json", ".yaml", ".yml"}
        )
    if len(parts) == 2 and parts[0] == "reproducibility":
        return parts[1] in REPRODUCIBILITY_FILES
    if len(parts) == 2 and parts[0] == "manifests":
        return parts[1] in MANIFEST_FILES
    return False


def _allowed_release_directory(relative: Path) -> bool:
    parts = relative.parts
    if relative.as_posix() in {
        "data/camcan",
        "data/camcan/connectomes",
        "data/camcan/metadata",
    }:
        return True
    if len(parts) == 1:
        return parts[0] in (REQUIRED_ROOT_ITEMS - ROOT_FILES)
    if len(parts) == 2:
        parent, child = parts
        if parent == "data":
            return child in {"public_connectomes", "public_metadata", "camcan"}
        if parent == "restricted_reconstruction":
            return child in {"adni", "adnidod", "oasis3"}
        if parent == "configs":
            return child in {"datasets", "experiments", "preprocessing"}
        if parent == "benchmark_results":
            return child in {
                "aggregate_metrics",
                "statistical_tests",
                "figure_source_data",
            }
        if parent == "splits":
            return child in KNOWN_DATASETS
    if len(parts) == 3:
        if parts[0:2] in {
            ("data", "public_connectomes"),
            ("data", "public_metadata"),
        }:
            return parts[2] in (KNOWN_DATASETS - {"camcan"})
        if parts[0] == "benchmark_results" and parts[1] in {
            "aggregate_metrics",
            "statistical_tests",
            "figure_source_data",
        }:
            return parts[2] in KNOWN_DATASETS
    return False


def _snapshot_bundle(release_root: Path) -> dict[str, Any]:
    metadata_dir = release_root / "metadata"
    paths = {
        "config": metadata_dir / "release_config_snapshot.yaml",
        "dataset_policy": metadata_dir / "release_policy_snapshot.yaml",
        "metadata_allowlist": metadata_dir / "metadata_allowlist_snapshot.yaml",
        "forbidden_patterns": metadata_dir / "forbidden_patterns_snapshot.yaml",
        "manual_approvals": metadata_dir / "manual_approvals_snapshot.yaml",
    }
    missing = [key for key, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(
            "release is missing policy snapshot(s): " + ", ".join(sorted(missing))
        )
    bundle: dict[str, Any] = {}
    for key, path in paths.items():
        bundle[key] = load_yaml(path)
        bundle[f"{key}_path"] = path
    validate_release_bundle_documents(bundle, snapshot=True)
    return bundle


def _load_bundle(
    release_root: Path, config_path: str | Path | None
) -> dict[str, Any]:
    snapshot = _snapshot_bundle(release_root)
    if config_path is None:
        return snapshot
    external = load_release_bundle(config_path)
    for key in (
        "config",
        "dataset_policy",
        "metadata_allowlist",
        "forbidden_patterns",
        "manual_approvals",
    ):
        external_value = external[key]
        if key == "config":
            external_value = dict(external_value)
            external_value.pop("paths", None)
        if external_value != snapshot[key]:
            raise ValueError(
                "external release config does not exactly match frozen snapshots"
            )
    return snapshot


def _relative(root: Path, path: Path) -> str:
    value = path.relative_to(root).as_posix()
    for pattern in (
        re.compile(r"\b(?:sub|ses)-[A-Za-z0-9]{2,}\b", re.IGNORECASE),
        re.compile(r"\b\d{3}_S_\d{4}\b"),
        re.compile(r"\b[\w.-]+\.p(?:ickle|kl)\b", re.IGNORECASE),
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ):
        value = pattern.sub(lambda match: redact(match.group(0)), value)
    return value


def _check_filesystem(
    root: Path, result: ValidationResult, *, allow_missing_reports: bool
) -> None:
    actual_root = {path.name for path in root.iterdir()}
    missing = sorted(REQUIRED_ROOT_ITEMS - actual_root)
    for name in missing:
        result.add_error("MISSING_RELEASE_ITEM", f"required release item is missing: {name}")
    for relative in sorted(REQUIRED_STATIC_DIRECTORIES):
        if not (root / relative).is_dir():
            result.add_error(
                "MISSING_RELEASE_DIRECTORY",
                "a required release directory is missing",
                path=relative,
            )
    required_files = set(REQUIRED_STATIC_FILES)
    if not allow_missing_reports:
        required_files.update(
            {
                "manifests/validation_report.json",
                "manifests/validation_report.md",
            }
        )
    for relative in sorted(required_files):
        if not (root / relative).is_file():
            result.add_error(
                "MISSING_RELEASE_FILE",
                "a required release file is missing",
                path=relative,
            )
    for relative in (
        "benchmark_results/aggregate_metrics",
        "benchmark_results/statistical_tests",
        "benchmark_results/figure_source_data",
    ):
        category = root / relative
        if category.is_dir() and not any(category.iterdir()):
            result.add_error(
                "EMPTY_BENCHMARK_CATEGORY",
                "benchmark result categories must contain reviewed outputs or a README",
                path=relative,
            )
    portable_keys: set[str] = set()
    for path in root.rglob("*"):
        raw_relative = path.relative_to(root).as_posix()
        relative = _relative(root, path)
        portable_key = raw_relative.casefold()
        if (
            not is_portable_relative_path(raw_relative)
            or portable_key in portable_keys
        ):
            result.add_error(
                "NON_PORTABLE_RELEASE_PATH",
                "release contains a non-portable or case-colliding path",
                path=relative,
            )
        portable_keys.add(portable_key)
        if path.is_symlink():
            result.add_error(
                "SYMLINK",
                "symbolic links are not permitted in a frozen release",
                path=relative,
            )
            continue
        if path.is_dir():
            if not _allowed_release_directory(Path(relative)):
                result.add_error(
                    "UNAPPROVED_RELEASE_DIRECTORY",
                    "directory is outside the exact release-tree allowlist",
                    path=relative,
                )
            continue
        if not path.is_file():
            result.add_error(
                "UNSAFE_FILESYSTEM_OBJECT",
                "release contains a non-regular filesystem object",
                path=relative,
            )
            continue
        if not _allowed_release_file(Path(relative)):
            result.add_error(
                "UNAPPROVED_RELEASE_FILE",
                "file is outside the exact release-tree allowlist",
                path=relative,
            )
        lower_name = path.name.lower()
        suffix = path.suffix.lower()
        if lower_name.endswith(".nii.gz") or suffix in FORBIDDEN_SUFFIXES:
            result.add_error(
                "FORBIDDEN_FILE_TYPE",
                "release contains a forbidden raw or executable-serialized file",
                path=relative,
            )
        if RAW_T1_FILENAME_RE.search(lower_name):
            result.add_error(
                "RAW_T1_IMAGE",
                "release contains a T1-weighted or anatomical source-image filename",
                path=relative,
            )
        if (
            lower_name in FORBIDDEN_NAMES
            and relative != CAMCAN_METADATA_RELATIVE_PATH.as_posix()
        ):
            result.add_error(
                "RAW_PARTICIPANT_FILE",
                "release contains a raw participant metadata filename",
                path=relative,
            )
        if "timeseries" in lower_name or "time_series" in lower_name:
            result.add_error(
                "TIME_SERIES_FILE",
                "release contains a participant-level time-series filename",
                path=relative,
            )
    result.checks["filesystem"] = {
        "required_root_items": len(REQUIRED_ROOT_ITEMS),
        "file_count": len(iter_regular_files(root)),
    }


def _dataset_directories(root: Path) -> list[tuple[str, str, Path]]:
    locations = [
        ("participant_connectomes", root / "data" / "public_connectomes"),
        ("participant_metadata", root / "data" / "public_metadata"),
        ("exact_splits", root / "splits"),
        ("reconstruction_instructions", root / "restricted_reconstruction"),
    ]
    found: list[tuple[str, str, Path]] = []
    for scope, parent in locations:
        if not parent.is_dir():
            continue
        for child in parent.iterdir():
            if child.is_dir():
                found.append((scope, canonical_dataset(child.name), child))
    for scope, path in (
        (
            "participant_connectomes",
            root / CAMCAN_CONNECTOME_RELATIVE_PATH.parent,
        ),
        (
            "participant_metadata",
            root / CAMCAN_METADATA_RELATIVE_PATH.parent,
        ),
    ):
        if path.is_dir():
            found.append((scope, "camcan", path))
    return found


def _check_dataset_policy(
    root: Path, bundle: Mapping[str, Any], result: ValidationResult
) -> None:
    public: dict[str, set[str]] = {}
    for scope, dataset, path in _dataset_directories(root):
        relative = _relative(root, path)
        if dataset not in KNOWN_DATASETS:
            result.add_error(
                "UNKNOWN_DATASET",
                "release contains an unknown dataset directory",
                path=relative,
            )
            continue
        if scope == "reconstruction_instructions":
            if dataset not in RESTRICTED_DATASETS:
                result.add_warning(
                    "UNEXPECTED_RECONSTRUCTION_FOLDER",
                    "public dataset has a reconstruction-only folder",
                    path=relative,
                )
            continue
        public.setdefault(dataset, set()).add(scope)
        if dataset in RESTRICTED_DATASETS:
            result.add_error(
                "RESTRICTED_DATASET_LEAK",
                "restricted dataset contains participant-level release files",
                path=relative,
            )
        allowed, reason = participant_release_decision(bundle, dataset, scope)
        if not allowed:
            result.add_error(
                "POLICY_BLOCK",
                f"{dataset} {scope} is not explicitly approved ({reason})",
                path=relative,
            )
    release_schema = bundle["config"].get("release", {})
    approvals = bundle["manual_approvals"].get("approvals", {})
    if isinstance(approvals, Mapping) and isinstance(
        approvals.get("datasets"), Mapping
    ):
        approvals = approvals["datasets"]
    for dataset in public:
        config_path = root / "configs" / "datasets" / f"{dataset}.yaml"
        try:
            snapshot = load_yaml(config_path)
            schema = snapshot["public_connectome_schema"]
            if not isinstance(schema, Mapping):
                raise ValueError("public_connectome_schema is not a mapping")
        except (OSError, ValueError, KeyError, yaml.YAMLError):
            result.add_error(
                "DATASET_SCHEMA_SNAPSHOT",
                f"{dataset} public schema snapshot is missing or invalid",
                path=_relative(root, config_path),
            )
            continue
        if (
            schema.get("atlas") != release_schema.get("atlas")
            or schema.get("n_regions") != release_schema.get("expected_regions")
            or schema.get("matrix_type") != release_schema.get("matrix_type")
        ):
            result.add_error(
                "DATASET_SCHEMA_BINDING",
                f"{dataset} public schema does not match the release schema",
                path=_relative(root, config_path),
            )
        if release_schema.get("test_only") is not True:
            approval = (
                approvals.get(dataset, {}) if isinstance(approvals, Mapping) else {}
            )
            if (
                not isinstance(approval, Mapping)
                or approval.get("source_binding_sha256")
                != schema.get("source_binding_sha256")
                or canonical_dataset(approval.get("dataset")) != dataset
                or approval.get("atlas") != schema.get("atlas")
                or approval.get("n_regions") != schema.get("n_regions")
            ):
                result.add_error(
                    "SOURCE_IDENTITY_BINDING",
                    f"{dataset} approval is not bound to source, atlas, and regions",
                    path=_relative(root, config_path),
                )
    result.checks["public_datasets"] = {
        dataset: sorted(scopes) for dataset, scopes in sorted(public.items())
    }


def _check_config_and_reproducibility_snapshots(
    root: Path,
    bundle: Mapping[str, Any],
    result: ValidationResult,
    *,
    bind_to_local_head: bool,
) -> None:
    """Recompute deterministic snapshots instead of trusting their checksums."""

    release = bundle["config"].get("release", {})
    from .build_release import expected_licenses_text, release_document_banner

    banner = release_document_banner(bundle)
    if banner:
        for name in ("README.md", "DATASET_CARD.md", "LICENSES.md"):
            path = root / name
            try:
                if not path.read_text(encoding="utf-8").startswith(banner):
                    raise ValueError("missing release-status banner")
            except (OSError, UnicodeError, ValueError):
                result.add_error(
                    "RELEASE_STATUS_BANNER",
                    "draft or synthetic releases must carry the required status banner",
                    path=name,
                )

    licenses_path = root / "LICENSES.md"
    try:
        if licenses_path.read_text(encoding="utf-8") != expected_licenses_text(
            bundle
        ):
            raise ValueError("license inventory mismatch")
    except (OSError, UnicodeError, ValueError):
        result.add_error(
            "LICENSES_DOCUMENT",
            "LICENSES.md does not exactly match the frozen license policy",
            path="LICENSES.md",
        )
    connectome_root = root / "data" / "public_connectomes"
    public_datasets = {
        path.name
        for path in connectome_root.glob("*")
        if path.is_dir() and path.name in KNOWN_DATASETS
    }
    if (root / CAMCAN_CONNECTOME_RELATIVE_PATH).is_file():
        public_datasets.add("camcan")
    approvals = bundle["manual_approvals"].get("approvals", {})
    if isinstance(approvals, Mapping) and isinstance(
        approvals.get("datasets"), Mapping
    ):
        approvals = approvals["datasets"]
    atlas_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in (
        "abide",
        "adni",
        "oasis3",
        "camcan",
        "cobre",
        "adnidod",
        "1000brains",
    ):
        path = root / "configs" / "datasets" / f"{dataset}.yaml"
        try:
            value = load_yaml(path)
        except (OSError, ValueError, yaml.YAMLError):
            result.add_error(
                "DATASET_CONFIG_SNAPSHOT",
                "dataset config snapshot is missing or malformed",
                path=_relative(root, path),
            )
            continue
        expected_keys = {"schema_version", "dataset", "release_policy"}
        if dataset in public_datasets:
            expected_keys.add("public_connectome_schema")
        policies = bundle["dataset_policy"].get("datasets", {})
        expected_policy = (
            policies.get(dataset, {}) if isinstance(policies, Mapping) else {}
        )
        if (
            set(value) != expected_keys
            or value.get("schema_version") != SCHEMA_VERSION
            or value.get("dataset") != dataset
            or value.get("release_policy") != expected_policy
        ):
            result.add_error(
                "DATASET_CONFIG_SNAPSHOT",
                "dataset config snapshot does not match the frozen policy",
                path=_relative(root, path),
            )
            continue
        if dataset not in public_datasets:
            continue
        schema = value.get("public_connectome_schema")
        approval = (
            approvals.get(dataset, {}) if isinstance(approvals, Mapping) else {}
        )
        if (
            not isinstance(schema, Mapping)
            or set(schema)
            != {"atlas", "n_regions", "matrix_type", "source_binding_sha256"}
            or schema.get("atlas") != release.get("atlas")
            or schema.get("n_regions") != release.get("expected_regions")
            or schema.get("matrix_type") != release.get("matrix_type")
            or not re.fullmatch(
                r"[0-9a-f]{64}", str(schema.get("source_binding_sha256", ""))
            )
            or (
                isinstance(approval, Mapping)
                and approval.get("source_binding_sha256")
                and approval.get("source_binding_sha256")
                != schema.get("source_binding_sha256")
            )
        ):
            result.add_error(
                "DATASET_CONFIG_SCHEMA",
                "public dataset schema snapshot is incomplete or inconsistent",
                path=_relative(root, path),
            )
            continue
        atlas_by_dataset[dataset] = {
            "atlas": schema["atlas"],
            "n_regions": schema["n_regions"],
        }

    preprocessing_path = root / "configs" / "preprocessing" / "connectome.yaml"
    expected_preprocessing = {
        "schema_version": SCHEMA_VERSION,
        "atlas_by_dataset": atlas_by_dataset,
        "connectome": {
            "estimator": (
                "source-bound trusted export; OAS when reconstructed from time series"
            ),
            "matrix_type": release.get("matrix_type"),
            "allowed_dtypes": ["float32", "float64"],
            "spd_regularization": True,
        },
    }
    try:
        if load_yaml(preprocessing_path) != expected_preprocessing:
            raise ValueError("preprocessing snapshot mismatch")
    except (OSError, ValueError, yaml.YAMLError):
        result.add_error(
            "PREPROCESSING_CONFIG_SNAPSHOT",
            "preprocessing config does not match released dataset schemas",
            path=_relative(root, preprocessing_path),
        )

    experiments_path = root / "configs" / "experiments" / "README.md"
    expected_experiments = (
        "# Experiment configurations\n\n"
        "The frozen benchmark result configurations, when approved and present, "
        "are listed in the manifest.\n"
    )
    try:
        if experiments_path.read_text(encoding="utf-8") != expected_experiments:
            raise ValueError("experiment snapshot mismatch")
    except (OSError, UnicodeError, ValueError):
        result.add_error(
            "EXPERIMENT_CONFIG_SNAPSHOT",
            "experiment config README does not match the frozen workflow",
            path=_relative(root, experiments_path),
        )

    requirements_path = root / "reproducibility" / "requirements-lock.txt"
    try:
        requirement_lines = [
            line.strip()
            for line in requirements_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not requirement_lines or any(
            not re.fullmatch(r"[A-Za-z0-9_.-]+==[^\s]+", line)
            for line in requirement_lines
        ):
            raise ValueError("dependency lock is not exact")
    except (OSError, UnicodeError, ValueError):
        result.add_error(
            "REQUIREMENTS_LOCK",
            "reproducibility requirements must contain exact pinned dependencies",
            path=_relative(root, requirements_path),
        )

    environment_path = root / "reproducibility" / "environment.yml"
    from .build_release import (
        REPRODUCIBILITY_COMMANDS_TEXT,
        REPRODUCIBILITY_ENVIRONMENT_TEXT,
    )

    try:
        if (
            environment_path.read_text(encoding="utf-8")
            != REPRODUCIBILITY_ENVIRONMENT_TEXT
        ):
            raise ValueError("environment snapshot mismatch")
    except (OSError, UnicodeError, ValueError):
        result.add_error(
            "REPRODUCIBILITY_ENVIRONMENT",
            "environment.yml does not match the canonical frozen environment",
            path=_relative(root, environment_path),
        )

    commands_path = root / "reproducibility" / "commands.md"
    try:
        if (
            commands_path.read_text(encoding="utf-8")
            != REPRODUCIBILITY_COMMANDS_TEXT
        ):
            raise ValueError("commands mismatch")
    except (OSError, UnicodeError, ValueError):
        result.add_error(
            "REPRODUCIBILITY_COMMANDS",
            "reproducibility commands do not match the frozen workflow",
            path=_relative(root, commands_path),
        )

    snapshot_path = root / "reproducibility" / "repository_snapshot.json"
    git_commit_path = root / "reproducibility" / "git_commit.txt"
    try:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if not isinstance(snapshot, Mapping):
            raise ValueError("snapshot must be an object")
        configured_repository = str(
            bundle["config"].get("project", {}).get("repository_url", "")
        ).strip()
        if (
            set(snapshot)
            != {
                "schema_version",
                "git_commit",
                "tracked_worktree_dirty",
                "source_repository",
            }
            or snapshot.get("schema_version") != SCHEMA_VERSION
            or not re.fullmatch(
                r"[0-9a-f]{40}", str(snapshot.get("git_commit", ""))
            )
            or not isinstance(snapshot.get("tracked_worktree_dirty"), bool)
            or snapshot.get("source_repository")
            != CANONICAL_REPOSITORY_URL
            or configured_repository
            not in {"", CANONICAL_REPOSITORY_URL}
        ):
            raise ValueError("repository identity mismatch")
        repository_root = Path(__file__).resolve().parents[2]
        if (repository_root / ".git").exists():
            commit_exists = subprocess.run(
                [
                    "git",
                    "cat-file",
                    "-e",
                    f"{snapshot['git_commit']}^{{commit}}",
                ],
                cwd=repository_root,
                check=False,
                capture_output=True,
            )
            if commit_exists.returncode != 0:
                raise ValueError("release commit is absent from the repository")
        if bind_to_local_head:
            from .build_release import _git_snapshot

            local_commit, _ = _git_snapshot(repository_root)
            if snapshot["git_commit"] != local_commit:
                raise ValueError("release commit does not match local HEAD")
        expected_git_text = (
            f"commit: {snapshot['git_commit']}\n"
            f"worktree_dirty: {str(snapshot['tracked_worktree_dirty']).lower()}\n"
        )
        if git_commit_path.read_text(encoding="utf-8") != expected_git_text:
            raise ValueError("git snapshot mismatch")
    except (
        OSError,
        UnicodeError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        result.add_error(
            "GIT_SNAPSHOT",
            "git_commit.txt does not match repository_snapshot.json",
            path=_relative(root, git_commit_path),
        )

    versions_path = root / "reproducibility" / "software_versions.json"
    try:
        versions = json.loads(versions_path.read_text(encoding="utf-8"))
        if (
            not isinstance(versions, Mapping)
            or set(versions)
            != {
                "python",
                "implementation",
                "platform_system",
                "platform_machine",
                "packages",
            }
            or any(
                not isinstance(versions.get(field), str)
                or not versions[field].strip()
                for field in (
                    "python",
                    "implementation",
                    "platform_system",
                    "platform_machine",
                )
            )
            or not isinstance(versions.get("packages"), Mapping)
            or set(versions["packages"])
            != {"numpy", "pandas", "scikit-learn", "scipy", "PyYAML"}
            or any(
                not isinstance(value, str) or not value.strip()
                for value in versions["packages"].values()
            )
        ):
            raise ValueError("software versions schema mismatch")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        result.add_error(
            "SOFTWARE_VERSIONS",
            "software_versions.json has an invalid frozen schema",
            path=_relative(root, versions_path),
        )


def _check_connectomes_and_metadata(
    root: Path, bundle: Mapping[str, Any], result: ValidationResult
) -> None:
    try:
        settings = validate_release_numeric_settings(bundle["config"])
    except ValueError:
        result.add_error(
            "MATRIX_VALIDATION_CONFIG",
            "release matrix-validation settings are invalid",
        )
        return
    if settings["atlas"] is None or settings["expected_regions"] is None:
        result.add_error(
            "ATLAS_BINDING",
            "release config does not bind an atlas and expected region count",
        )
        return
    matrix_type = settings["matrix_type"]
    connectome_root = root / "data" / "public_connectomes"
    metadata_root = root / "data" / "public_metadata"
    counts: dict[str, dict[str, int]] = {}

    connectome_sources: list[tuple[str, Path]] = []
    if connectome_root.is_dir():
        for dataset_dir in sorted(connectome_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = canonical_dataset(dataset_dir.name)
            npz_files = sorted(dataset_dir.rglob("*.npz"))
            if len(npz_files) != 1 or npz_files[0].name != "connectomes.npz":
                result.add_error(
                    "CONNECTOME_FILE_LAYOUT",
                    "each public dataset must contain exactly connectomes.npz",
                    path=_relative(root, dataset_dir),
                )
                continue
            connectome_sources.append((dataset, npz_files[0]))
    camcan_connectome = root / CAMCAN_CONNECTOME_RELATIVE_PATH
    if camcan_connectome.is_file():
        connectome_sources.append(("camcan", camcan_connectome))

    expected_npz_paths: set[Path] = set()
    for dataset, path in connectome_sources:
        expected_npz_paths.add(path.resolve())
        count, findings = validate_connectome_npz(
            path,
            matrix_type=matrix_type,
            symmetry_tolerance=settings["symmetry_tolerance"],
            diagonal_tolerance=settings["diagonal_tolerance"],
            spd_eigenvalue_tolerance=settings["spd_eigenvalue_tolerance"],
            expected_regions=settings["expected_regions"],
        )
        for finding in findings:
            result.errors.append(
                Finding(
                    finding.code,
                    finding.severity,
                    finding.message,
                    _relative(root, path),
                    finding.line,
                    finding.redacted_value,
                )
            )
        if count is not None:
            counts.setdefault(dataset, {})["connectomes"] = count

    # Any NPZ outside the public connectome layout is still inspected safely.
    for path in root.rglob("*.npz"):
        if path.resolve() in expected_npz_paths:
            continue
        _, findings = validate_connectome_npz(
            path,
            matrix_type=matrix_type,
            symmetry_tolerance=settings["symmetry_tolerance"],
            diagonal_tolerance=settings["diagonal_tolerance"],
            spd_eigenvalue_tolerance=settings["spd_eigenvalue_tolerance"],
            expected_regions=settings["expected_regions"],
        )
        if not findings:
            result.add_error(
                "UNEXPECTED_NPZ",
                "NPZ exists outside the documented public-connectome layout",
                path=_relative(root, path),
            )
        else:
            for finding in findings:
                result.errors.append(
                    Finding(
                        finding.code,
                        finding.severity,
                        finding.message,
                        _relative(root, path),
                    )
                )

    metadata_sources: list[tuple[str, Path]] = []
    if metadata_root.is_dir():
        for dataset_dir in sorted(metadata_root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset = canonical_dataset(dataset_dir.name)
            metadata_files = sorted(dataset_dir.rglob("*.tsv"))
            if len(metadata_files) != 1 or metadata_files[0].name != "metadata.tsv":
                result.add_error(
                    "METADATA_FILE_LAYOUT",
                    "each public dataset must contain exactly metadata.tsv",
                    path=_relative(root, dataset_dir),
                )
                continue
            metadata_sources.append((dataset, metadata_files[0]))
    camcan_metadata = root / CAMCAN_METADATA_RELATIVE_PATH
    if camcan_metadata.is_file():
        metadata_sources.append(("camcan", camcan_metadata))

    for dataset, metadata_path in metadata_sources:
        try:
            columns, rows = read_tsv(metadata_path)
        except (OSError, UnicodeError, ValueError):
            result.add_error(
                "METADATA_TSV",
                "public metadata TSV is unreadable or malformed",
                path=_relative(root, metadata_path),
            )
            continue
        for finding in validate_metadata_columns(
            columns, bundle=bundle, dataset=dataset
        ):
            result.errors.append(
                Finding(
                    finding.code,
                    finding.severity,
                    finding.message,
                    _relative(root, metadata_path),
                    finding.line,
                    finding.redacted_value,
                )
            )
        if dataset == "camcan" and columns != ["sample_uid", "age", "sex"]:
            result.add_error(
                "CAMCAN_METADATA_SCHEMA",
                "CamCAN participants.tsv must use exactly sample_uid, age, sex "
                "in that order",
                path=_relative(root, metadata_path),
            )
        if "sample_uid" not in columns:
            result.add_error(
                "MISSING_SAMPLE_UID",
                "public metadata has no sample_uid column",
                path=_relative(root, metadata_path),
            )
            continue
        uids = [row["sample_uid"] for row in rows]
        if len(uids) != len(set(uids)):
            result.add_error(
                "DUPLICATE_SAMPLE_UID",
                "public metadata sample_uid values are not unique",
                path=_relative(root, metadata_path),
            )
        if any(not SAFE_UID_RE.fullmatch(uid) for uid in uids):
            result.add_error(
                "UNSAFE_SAMPLE_UID",
                "one or more sample_uid values violate the release schema",
                path=_relative(root, metadata_path),
            )
        if dataset == "camcan" and {"age", "sex"}.issubset(columns):
            invalid_age = False
            invalid_sex = False
            for row in rows:
                try:
                    age = float(row["age"])
                    invalid_age = invalid_age or not (
                        math.isfinite(age) and 0.0 <= age <= 130.0
                    )
                except (TypeError, ValueError):
                    invalid_age = True
                invalid_sex = invalid_sex or row["sex"] not in {"F", "M"}
            if invalid_age:
                result.add_error(
                    "CAMCAN_AGE_VALUE",
                    "CamCAN age values must be finite numbers between 0 and 130",
                    path=_relative(root, metadata_path),
                )
            if invalid_sex:
                result.add_error(
                    "CAMCAN_SEX_VALUE",
                    "CamCAN sex values must use the release enum F or M",
                    path=_relative(root, metadata_path),
                )
        counts.setdefault(dataset, {})["metadata"] = len(rows)
        counts[dataset]["uid_count"] = len(set(uids))

    for dataset, values in counts.items():
        if values.get("connectomes") != values.get("metadata"):
            result.add_error(
                "SAMPLE_COUNT_MISMATCH",
                f"{dataset} connectome count does not match metadata rows",
            )
    result.checks["sample_counts"] = counts


def _metadata_uids(root: Path, dataset: str) -> set[str]:
    path = root / release_metadata_relative_path(dataset)
    if not path.is_file():
        return set()
    try:
        columns, rows = read_tsv(path)
    except (OSError, UnicodeError, ValueError):
        return set()
    return {row["sample_uid"] for row in rows} if "sample_uid" in columns else set()


def _check_splits(root: Path, result: ValidationResult) -> None:
    split_root = root / "splits"
    split_counts: dict[str, int] = {}
    if not split_root.is_dir():
        return
    for dataset_dir in sorted(split_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = canonical_dataset(dataset_dir.name)
        files = sorted(dataset_dir.rglob("*.tsv"))
        if len(files) != 1 or files[0].name != "splits.tsv":
            result.add_error(
                "SPLIT_FILE_LAYOUT",
                "each split dataset must contain exactly splits.tsv",
                path=_relative(root, dataset_dir),
            )
            continue
        try:
            columns, rows = read_tsv(files[0])
        except (OSError, UnicodeError, ValueError):
            result.add_error(
                "SPLIT_TSV",
                "split TSV is unreadable or malformed",
                path=_relative(root, files[0]),
            )
            continue
        if columns != ["fold", "partition", "sample_uid"]:
            result.add_error(
                "SPLIT_SCHEMA",
                "split TSV must have fold, partition, sample_uid columns",
                path=_relative(root, files[0]),
            )
            continue
        if not rows:
            result.add_error(
                "SPLIT_EMPTY",
                "split TSV must contain at least one complete fold",
                path=_relative(root, files[0]),
            )
            continue
        known = _metadata_uids(root, dataset)
        seen: set[tuple[str, str]] = set()
        fold_members: dict[str, set[str]] = {}
        fold_partitions: dict[str, set[str]] = {}
        for row in rows:
            if not row["fold"] or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", row["fold"]
            ):
                result.add_error(
                    "SPLIT_FOLD",
                    "split fold label is unsafe",
                    path=_relative(root, files[0]),
                )
            membership = (row["fold"], row["sample_uid"])
            if membership in seen:
                result.add_error(
                    "SPLIT_OVERLAP",
                    "sample_uid occurs in more than one partition of a fold",
                    path=_relative(root, files[0]),
                )
            seen.add(membership)
            fold_members.setdefault(row["fold"], set()).add(row["sample_uid"])
            fold_partitions.setdefault(row["fold"], set()).add(row["partition"])
            if row["partition"] not in {"train", "validation", "test"}:
                result.add_error(
                    "SPLIT_PARTITION",
                    "split partition is not train/validation/test",
                    path=_relative(root, files[0]),
                )
            if row["sample_uid"] not in known:
                result.add_error(
                    "SPLIT_UNKNOWN_UID",
                    "split references an unknown sample_uid",
                    path=_relative(root, files[0]),
                )
        for fold in fold_members:
            if fold_members[fold] != known:
                result.add_error(
                    "SPLIT_INCOMPLETE",
                    "each fold must assign every public sample exactly once",
                    path=_relative(root, files[0]),
                )
            if not {"train", "test"}.issubset(fold_partitions[fold]):
                result.add_error(
                    "SPLIT_PARTITIONS_INCOMPLETE",
                    "each fold requires at least train and test partitions",
                    path=_relative(root, files[0]),
                )
        split_counts[dataset] = len(rows)
    result.checks["split_membership_rows"] = split_counts


def _check_aggregate_outputs(
    root: Path, bundle: Mapping[str, Any], result: ValidationResult
) -> None:
    from .build_release import (
        ReleaseBuildError,
        _aggregate_allowed_suffixes,
        _aggregate_review_is_complete,
        _validate_aggregate_file,
    )
    from .checksums import sha256_file

    tables = bundle["metadata_allowlist"].get("tables", {})
    folder_to_category = {
        "aggregate_metrics": "aggregate_metrics",
        "statistical_tests": "statistical_summaries",
        "figure_source_data": "figure_source_data",
    }
    checked = 0
    for folder, category in folder_to_category.items():
        parent = root / "benchmark_results" / folder
        if not parent.is_dir():
            continue
        entry = tables.get(category, {}) if isinstance(tables, Mapping) else {}
        columns = entry.get("allowed_columns", []) if isinstance(entry, Mapping) else []
        allowed = {str(value) for value in columns} if isinstance(columns, list) else set()
        for path in parent.rglob("*"):
            if not path.is_file() or path.name == "README.md":
                continue
            checked += 1
            if re.search(
                r"(?i)(?:individual|participant|subject|prediction|embedding|residual)",
                path.stem,
            ):
                result.add_error(
                    "INDIVIDUAL_OUTPUT_NAME",
                    "aggregate filename suggests blocked individual-level output",
                    path=_relative(root, path),
                )
            try:
                allowed_suffixes = _aggregate_allowed_suffixes(bundle, category)
            except ReleaseBuildError:
                result.add_error(
                    "AGGREGATE_EXTENSION_ALLOWLIST",
                    "aggregate extension allowlist is missing or unsafe",
                    path=_relative(root, path),
                )
                continue
            if path.suffix.lower() not in allowed_suffixes:
                result.add_error(
                    "AGGREGATE_EXTENSION",
                    "aggregate output extension is not explicitly allowed",
                    path=_relative(root, path),
                )
                continue
            if not allowed:
                result.add_error(
                    "AGGREGATE_ALLOWLIST",
                    f"no explicit column allowlist exists for {category}",
                    path=_relative(root, path),
                )
                continue
            dataset = canonical_dataset(path.parent.name)
            reviewed_relative = (
                Path("aggregate_outputs") / dataset / path.name
            ).as_posix()
            if not _aggregate_review_is_complete(
                bundle,
                dataset=dataset,
                category=category,
                relative_path=reviewed_relative,
                sha256=sha256_file(path),
            ):
                result.add_error(
                    "AGGREGATE_REVIEW_BINDING",
                    "aggregate output lacks a path-and-checksum-bound artifact review",
                    path=_relative(root, path),
                )
                continue
            try:
                _validate_aggregate_file(
                    path,
                    allowed,
                    expected_dataset=dataset,
                    content_category=category,
                )
            except (
                ReleaseBuildError,
                OSError,
                UnicodeError,
                ValueError,
                json.JSONDecodeError,
                yaml.YAMLError,
            ):
                result.add_error(
                    "AGGREGATE_SCHEMA",
                    "aggregate output failed its strict schema",
                    path=_relative(root, path),
                )
    result.checks["aggregate_files_checked"] = checked


def _check_camcan_companions(
    root: Path, bundle: Mapping[str, Any], result: ValidationResult
) -> None:
    """Require the complete, canonical CamCAN dataset-centric release subtree."""

    from .build_release import (
        _camcan_license_text,
        canonical_data_dictionary_row,
    )

    camcan_root = root / "data" / "camcan"
    if not camcan_root.exists():
        return

    required = {
        CAMCAN_CONNECTOME_RELATIVE_PATH,
        CAMCAN_METADATA_RELATIVE_PATH,
        CAMCAN_DATA_DICTIONARY_RELATIVE_PATH,
        CAMCAN_LICENSE_RELATIVE_PATH,
    }
    missing = sorted(
        relative.as_posix()
        for relative in required
        if not (root / relative).is_file()
    )
    if missing:
        result.add_error(
            "CAMCAN_RELEASE_LAYOUT",
            "CamCAN must include the complete canonical connectome, metadata, "
            "data-dictionary, and license file set",
            path="data/camcan",
        )

    license_path = root / CAMCAN_LICENSE_RELATIVE_PATH
    if license_path.is_file():
        try:
            if license_path.read_text(encoding="utf-8") != _camcan_license_text(
                bundle
            ):
                raise ValueError("license mismatch")
        except (OSError, UnicodeError, ValueError):
            result.add_error(
                "CAMCAN_LICENSE_BINDING",
                "CamCAN LICENSE.txt does not exactly match the frozen authorization",
                path=CAMCAN_LICENSE_RELATIVE_PATH.as_posix(),
            )

    global_dictionary_path = root / "metadata" / "data_dictionary.tsv"
    camcan_dictionary_path = root / CAMCAN_DATA_DICTIONARY_RELATIVE_PATH
    if global_dictionary_path.is_file() and camcan_dictionary_path.is_file():
        try:
            global_columns, _ = read_tsv(global_dictionary_path)
            camcan_columns, camcan_rows = read_tsv(camcan_dictionary_path)
            expected_fields = {
                ("connectomes", "participant_connectomes"),
                ("sample_uid", "participant_metadata"),
                ("age", "participant_metadata"),
                ("sex", "participant_metadata"),
            }
            expected_rows_as_lists = [
                canonical_data_dictionary_row(field, category)
                for field, category in sorted(
                    expected_fields, key=lambda value: (value[1], value[0])
                )
            ]
            expected_rows = [
                dict(zip(RELEASE_TABLE_COLUMNS["data_dictionary"], row))
                for row in expected_rows_as_lists
            ]
            if (
                global_columns != RELEASE_TABLE_COLUMNS["data_dictionary"]
                or camcan_columns != RELEASE_TABLE_COLUMNS["data_dictionary"]
                or camcan_rows != expected_rows
                or {
                    (row["field"], row["content_category"])
                    for row in camcan_rows
                }
                != expected_fields
            ):
                raise ValueError("dictionary mismatch")
        except (
            KeyError,
            OSError,
            UnicodeError,
            ValueError,
        ):
            result.add_error(
                "CAMCAN_DATA_DICTIONARY",
                "CamCAN data_dictionary.tsv is not the exact dataset-specific "
                "subset of the global dictionary",
                path=CAMCAN_DATA_DICTIONARY_RELATIVE_PATH.as_posix(),
            )


def _check_release_tables(
    root: Path, bundle: Mapping[str, Any], result: ValidationResult
) -> None:
    """Validate the three auditable release tables against their exact schemas."""

    configured_tables = bundle["metadata_allowlist"].get("tables", {})
    checked: dict[str, int] = {}
    parsed_rows: dict[str, list[dict[str, str]]] = {}
    for table, built_in_columns in RELEASE_TABLE_COLUMNS.items():
        path = root / "metadata" / f"{table}.tsv"
        try:
            columns, rows = read_tsv(path)
        except (OSError, UnicodeError, ValueError):
            result.add_error(
                "RELEASE_TABLE",
                f"{table}.tsv is unreadable or malformed",
                path=_relative(root, path),
            )
            continue
        configured = (
            configured_tables.get(table, {})
            if isinstance(configured_tables, Mapping)
            else {}
        )
        configured_columns = (
            configured.get("allowed_columns", [])
            if isinstance(configured, Mapping)
            else []
        )
        expected = (
            [str(value) for value in configured_columns]
            if isinstance(configured_columns, list) and configured_columns
            else built_in_columns
        )
        if columns != expected or columns != built_in_columns:
            result.add_error(
                "RELEASE_TABLE_SCHEMA",
                f"{table}.tsv does not use the exact reviewed column order",
                path=_relative(root, path),
            )
            checked[table] = len(rows)
            continue
        checked[table] = len(rows)
        parsed_rows[table] = rows

    for row in parsed_rows.get("dataset_inventory", []):
        if canonical_dataset(row["dataset"]) not in KNOWN_DATASETS:
            result.add_error(
                "INVENTORY_DATASET",
                "dataset inventory contains an unknown dataset",
                path="metadata/dataset_inventory.tsv",
            )
        for field in ("file_count", "sample_count"):
            try:
                if int(row[field]) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                result.add_error(
                    "INVENTORY_COUNT",
                    f"dataset inventory {field} must be a non-negative integer",
                    path="metadata/dataset_inventory.tsv",
                )

    inventory_columns = RELEASE_TABLE_COLUMNS["dataset_inventory"]
    inventory_expected: list[dict[str, str]] = []
    release_settings = bundle["config"].get("release", {})
    for dataset in sorted(KNOWN_DATASETS):
        metadata_path = root / release_metadata_relative_path(dataset)
        sample_count = 0
        if metadata_path.is_file():
            try:
                _, metadata_rows = read_tsv(metadata_path)
                sample_count = len(metadata_rows)
            except (OSError, UnicodeError, ValueError):
                sample_count = 0
        for category, path in (
            (
                "participant_connectomes",
                root / release_connectome_relative_path(dataset),
            ),
            ("participant_metadata", metadata_path),
            ("exact_splits", root / "splits" / dataset / "splits.tsv"),
        ):
            present = path.is_file()
            release_decision = participant_inventory_decision(
                bundle,
                dataset,
                category,
                present=present,
            )
            inventory_expected.append(
                {
                    "dataset": dataset,
                    "content_category": category,
                    "access_category": "public" if present else "not_included",
                    "file_count": "1" if present else "0",
                    "sample_count": str(sample_count if present else 0),
                    "atlas": str(release_settings.get("atlas", "")) if present else "",
                    "matrix_type": (
                        str(release_settings.get("matrix_type", ""))
                        if present
                        else ""
                    ),
                    "release_policy_decision": release_decision,
                }
            )
    aggregate_locations = {
        "aggregate_metrics": "aggregate_metrics",
        "statistical_summaries": "statistical_tests",
        "figure_source_data": "figure_source_data",
    }
    for category, folder in aggregate_locations.items():
        parent = root / "benchmark_results" / folder
        if not parent.is_dir():
            continue
        for dataset_dir in sorted(parent.iterdir()):
            if not dataset_dir.is_dir() or dataset_dir.name not in KNOWN_DATASETS:
                continue
            count = sum(
                1
                for path in dataset_dir.iterdir()
                if path.is_file() and path.name != "README.md"
            )
            if not count:
                continue
            policy = dataset_policy(bundle, dataset_dir.name)
            decisions = policy.get("decisions", {})
            decision = (
                decisions.get(category, "forbidden")
                if isinstance(decisions, Mapping)
                else "forbidden"
            )
            inventory_expected.append(
                {
                    "dataset": dataset_dir.name,
                    "content_category": category,
                    "access_category": "public",
                    "file_count": str(count),
                    "sample_count": "0",
                    "atlas": "",
                    "matrix_type": "",
                    "release_policy_decision": str(decision),
                }
            )
    actual_inventory = Counter(
        tuple(row[column] for column in inventory_columns)
        for row in parsed_rows.get("dataset_inventory", [])
    )
    expected_inventory = Counter(
        tuple(row[column] for column in inventory_columns)
        for row in inventory_expected
    )
    if actual_inventory != expected_inventory:
        result.add_error(
            "INVENTORY_MISMATCH",
            "dataset inventory does not exactly match release contents and policy",
            path="metadata/dataset_inventory.tsv",
        )

    expected_dictionary_fields: set[tuple[str, str]] = set()
    for dataset in KNOWN_DATASETS:
        path = root / release_connectome_relative_path(dataset)
        if path.is_file():
            expected_dictionary_fields.add(
                ("connectomes", "participant_connectomes")
            )
    for dataset in KNOWN_DATASETS:
        path = root / release_metadata_relative_path(dataset)
        if path.is_file():
            try:
                columns, _ = read_tsv(path)
                expected_dictionary_fields.update(
                    (column, "participant_metadata") for column in columns
                )
            except (OSError, UnicodeError, ValueError):
                pass
    for path in root.glob("splits/*/splits.tsv"):
        if path.is_file():
            try:
                columns, _ = read_tsv(path)
                expected_dictionary_fields.update(
                    (column, "exact_splits") for column in columns
                )
            except (OSError, UnicodeError, ValueError):
                pass
    for category, folder in aggregate_locations.items():
        for path in (root / "benchmark_results" / folder).glob("*/*"):
            if not path.is_file() or path.name == "README.md":
                continue
            try:
                if path.suffix.lower() in {".csv", ".tsv"}:
                    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
                    with path.open("r", encoding="utf-8", newline="") as source:
                        fields = next(
                            csv.reader(source, delimiter=delimiter, strict=True)
                        )
                else:
                    value = (
                        json.loads(path.read_text(encoding="utf-8"))
                        if path.suffix.lower() == ".json"
                        else yaml.safe_load(path.read_text(encoding="utf-8"))
                    )
                    if isinstance(value, Mapping) and set(value) == {"records"}:
                        value = value["records"]
                    fields = sorted(
                        {
                            str(field)
                            for record in value
                            if isinstance(record, Mapping)
                            for field in record
                        }
                    )
                expected_dictionary_fields.update(
                    (field, category) for field in fields
                )
            except (
                OSError,
                UnicodeError,
                ValueError,
                TypeError,
                csv.Error,
                json.JSONDecodeError,
                yaml.YAMLError,
            ):
                pass
    from .build_release import canonical_data_dictionary_row

    dictionary_rows = parsed_rows.get("data_dictionary", [])
    dictionary_columns = RELEASE_TABLE_COLUMNS["data_dictionary"]
    actual_dictionary_rows = Counter(
        tuple(row[column] for column in dictionary_columns)
        for row in dictionary_rows
    )
    expected_dictionary_rows = Counter(
        tuple(canonical_data_dictionary_row(field, category))
        for field, category in expected_dictionary_fields
    )
    if actual_dictionary_rows != expected_dictionary_rows:
        result.add_error(
            "DATA_DICTIONARY_MISMATCH",
            "data dictionary does not exactly match canonical field semantics",
            path="metadata/data_dictionary.tsv",
        )

    fingerprint_payload = {
        "config": bundle["config"],
        "dataset_policy": bundle["dataset_policy"],
        "metadata_allowlist": bundle["metadata_allowlist"],
        "forbidden_patterns": bundle["forbidden_patterns"],
        "manual_approvals": bundle["manual_approvals"],
    }
    expected_fingerprint = hashlib.sha256(
        yaml.safe_dump(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    expected_version = str(
        bundle["config"].get("project", {}).get("version", "")
    ).strip()
    expected_commit: str | None = None
    repository_snapshot = root / "reproducibility" / "repository_snapshot.json"
    try:
        snapshot_value = json.loads(repository_snapshot.read_text(encoding="utf-8"))
        if (
            not isinstance(snapshot_value, Mapping)
            or set(snapshot_value)
            != {
                "schema_version",
                "git_commit",
                "tracked_worktree_dirty",
                "source_repository",
            }
            or snapshot_value.get("schema_version") != SCHEMA_VERSION
            or not re.fullmatch(
                r"[0-9a-f]{40}", str(snapshot_value.get("git_commit", ""))
            )
            or not isinstance(snapshot_value.get("tracked_worktree_dirty"), bool)
            or snapshot_value.get("source_repository")
            != CANONICAL_REPOSITORY_URL
        ):
            raise ValueError("invalid repository snapshot")
        expected_commit = snapshot_value["git_commit"].strip()
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        result.add_error(
            "PROVENANCE_CONTEXT",
            "repository snapshot is missing or invalid for provenance verification",
            path="reproducibility/repository_snapshot.json",
        )

    provenance_bindings = {
        "public_connectomes": (
            "trusted_safe_export",
            "participant_connectomes",
            "allowlisted export with source identifiers and paths removed",
        ),
        "public_metadata": (
            "trusted_safe_export",
            "participant_metadata",
            "allowlisted export with source identifiers and paths removed",
        ),
        "splits": (
            "trusted_safe_export",
            "exact_splits",
            "allowlisted export with source identifiers and paths removed",
        ),
        "aggregate_metrics": (
            "reviewed_aggregate_output",
            "aggregate_metrics",
            "explicitly allowlisted aggregate artifact copy",
        ),
        "statistical_tests": (
            "reviewed_aggregate_output",
            "statistical_summaries",
            "explicitly allowlisted aggregate artifact copy",
        ),
        "figure_source_data": (
            "reviewed_aggregate_output",
            "figure_source_data",
            "explicitly allowlisted aggregate artifact copy",
        ),
    }

    def expected_binding(relative: str) -> tuple[str, str, str] | None:
        candidate = Path(relative)
        if candidate == CAMCAN_CONNECTOME_RELATIVE_PATH:
            return provenance_bindings["public_connectomes"]
        if candidate == CAMCAN_METADATA_RELATIVE_PATH:
            return provenance_bindings["public_metadata"]
        parts = candidate.parts
        if (
            len(parts) == 4
            and parts[0] == "data"
            and parts[1] in {"public_connectomes", "public_metadata"}
        ):
            return provenance_bindings[parts[1]]
        if len(parts) == 3 and parts[0] == "splits":
            return provenance_bindings["splits"]
        if (
            len(parts) == 4
            and parts[0] == "benchmark_results"
            and parts[1]
            in {"aggregate_metrics", "statistical_tests", "figure_source_data"}
        ):
            return provenance_bindings[parts[1]]
        return None

    seen_paths: set[str] = set()
    for row in parsed_rows.get("provenance", []):
        relative = row["relative_path"]
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or "\\" in relative
            or ".." in candidate.parts
            or relative in seen_paths
        ):
            result.add_error(
                "PROVENANCE_PATH",
                "provenance contains an unsafe or duplicate relative path",
                path="metadata/provenance.tsv",
            )
            continue
        seen_paths.add(relative)
        target = root / candidate
        if not target.is_file() or not target.resolve().is_relative_to(root):
            result.add_error(
                "PROVENANCE_TARGET",
                "provenance references a missing or unsafe release file",
                path="metadata/provenance.tsv",
            )
            continue
        from .checksums import sha256_file

        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) or sha256_file(
            target
        ) != row["sha256"]:
            result.add_error(
                "PROVENANCE_CHECKSUM",
                "provenance checksum does not match the release file",
                path="metadata/provenance.tsv",
            )
        binding = expected_binding(relative)
        if (
            binding is None
            or (
                row["source_category"],
                row["content_category"],
                row["transformation"],
            )
            != binding
            or row["config_fingerprint"] != expected_fingerprint
            or row["software_version"] != expected_version
            or expected_commit is None
            or row["repository_commit"] != expected_commit
        ):
            result.add_error(
                "PROVENANCE_BINDING",
                "provenance fields do not match the frozen release context",
                path="metadata/provenance.tsv",
            )
    expected_provenance = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.relative_to(root)
            in {
                CAMCAN_CONNECTOME_RELATIVE_PATH,
                CAMCAN_METADATA_RELATIVE_PATH,
            }
            or "public_connectomes" in path.parts
            or "public_metadata" in path.parts
            or (
                "splits" in path.parts
                and path.name == "splits.tsv"
            )
            or (
                "benchmark_results" in path.parts
                and path.name != "README.md"
            )
        )
    }
    if seen_paths != expected_provenance:
        result.add_error(
            "PROVENANCE_COVERAGE",
            "provenance does not exactly cover public data, splits, and "
            "aggregate outputs",
            path="metadata/provenance.tsv",
        )
    result.checks["config_fingerprint"] = expected_fingerprint
    result.checks["release_table_rows"] = checked


def _check_restricted_folders(
    root: Path,
    bundle: Mapping[str, Any] | None,
    result: ValidationResult,
    *,
    publication_ready: bool,
) -> None:
    restricted_root = root / "restricted_reconstruction"
    allowed_names = {"README.md", "selection_config.yaml", "reconstruction_commands.md"}
    for dataset in ("adni", "adnidod", "oasis3"):
        folder = restricted_root / dataset
        if not folder.is_dir():
            result.add_error(
                "MISSING_RECONSTRUCTION",
                f"restricted reconstruction folder is missing for {dataset}",
            )
            continue
        actual = {path.name for path in folder.iterdir() if path.is_file()}
        if actual != allowed_names:
            result.add_error(
                "RESTRICTED_FOLDER_CONTENT",
                f"{dataset} reconstruction folder has missing or unexpected files",
                path=_relative(root, folder),
            )
            continue
        selection_path = folder / "selection_config.yaml"
        try:
            selection = load_yaml(selection_path)
        except (OSError, ValueError, yaml.YAMLError):
            result.add_error(
                "RECONSTRUCTION_SCHEMA",
                "restricted reconstruction configuration is invalid",
                path=_relative(root, selection_path),
            )
            continue
        required_top = {
            "schema_version",
            "dataset",
            "participant_level_redistribution",
            "atlas",
            "matrix_construction",
            "task",
            "target",
            "sample_selection",
            "split_reconstruction",
        }
        sample_selection = selection.get("sample_selection")
        split = selection.get("split_reconstruction")
        structurally_valid = (
            set(selection) == required_top
            and selection.get("schema_version") == SCHEMA_VERSION
            and selection.get("dataset") == dataset
            and selection.get("participant_level_redistribution") == "forbidden"
            and isinstance(sample_selection, Mapping)
            and set(sample_selection)
            == {
                "implementation",
                "repository_commit",
                "authoritative_reference",
                "reviewed",
            }
            and isinstance(split, Mapping)
            and set(split)
            == {
                "protocols",
                "grouping_rule",
                "kfold_n_splits",
                "model_seed",
                "data_shuffle_seed",
                "release_exact_membership",
                "aggregate_fold_counts",
            }
            and split.get("release_exact_membership") is False
            and isinstance(split.get("protocols"), list)
            and split.get("protocols")
            == ["subject_grouped_kfold", "leave_one_dataset_out"]
            and isinstance(split.get("kfold_n_splits"), int)
            and not isinstance(split.get("kfold_n_splits"), bool)
            and split["kfold_n_splits"] >= 2
            and isinstance(split.get("model_seed"), int)
            and not isinstance(split.get("model_seed"), bool)
            and isinstance(split.get("data_shuffle_seed"), int)
            and not isinstance(split.get("data_shuffle_seed"), bool)
            and isinstance(split.get("aggregate_fold_counts"), list)
        )
        if bundle is not None:
            release = bundle["config"].get("release", {})
            reconstruction = bundle["config"].get("restricted_reconstruction", {})
            structurally_valid = structurally_valid and (
                selection.get("atlas") == release.get("atlas")
                and selection.get("task")
                == (
                    reconstruction.get("task", "regression")
                    if isinstance(reconstruction, Mapping)
                    else "regression"
                )
                and selection.get("target")
                == (
                    reconstruction.get("target", "Age")
                    if isinstance(reconstruction, Mapping)
                    else "Age"
                )
            )
        if not structurally_valid:
            result.add_error(
                "RECONSTRUCTION_SCHEMA",
                "restricted reconstruction configuration does not match the reviewed schema",
                path=_relative(root, selection_path),
            )
            continue
        counts = split["aggregate_fold_counts"]
        counts_valid = bool(counts) and all(
            isinstance(value, Mapping)
            and set(value) == {"fold", "n_samples"}
            and isinstance(value["fold"], str)
            and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value["fold"]))
            and isinstance(value["n_samples"], int)
            and not isinstance(value["n_samples"], bool)
            and value["n_samples"] > 0
            for value in counts
        )
        review_complete = (
            sample_selection.get("reviewed") is True
            and isinstance(sample_selection.get("authoritative_reference"), str)
            and bool(sample_selection["authoritative_reference"].strip())
            and counts_valid
        )
        if publication_ready and not review_complete:
            result.add_error(
                "RECONSTRUCTION_MANUAL_REVIEW",
                "restricted selection rules and aggregate fold counts require review",
                path=_relative(root, selection_path),
            )


def _check_catalogs(
    root: Path,
    result: ValidationResult,
    bundle: Mapping[str, Any] | None,
) -> bool:
    manifest = root / "manifests" / "manifest.tsv"
    sums = root / "manifests" / "SHA256SUMS.txt"
    catalog_ok = True
    if not manifest.is_file():
        result.add_error("MISSING_MANIFEST", "manifest.tsv is missing")
        catalog_ok = False
    else:
        errors = verify_manifest(
            root,
            manifest,
            bundle=bundle,
            excluded_relative={
                "manifests/manifest.tsv",
                "manifests/SHA256SUMS.txt",
            },
        )
        for error in errors:
            result.add_error("MANIFEST_MISMATCH", error, path="manifests/manifest.tsv")
        catalog_ok = catalog_ok and not errors
    if not sums.is_file():
        result.add_error("MISSING_CHECKSUMS", "SHA256SUMS.txt is missing")
        catalog_ok = False
    else:
        errors = verify_sha256sums(root, sums)
        for error in errors:
            result.add_error("CHECKSUM_MISMATCH", error, path="manifests/SHA256SUMS.txt")
        try:
            checksum_paths = {relative for _, relative in read_sha256sums(sums)}
        except (OSError, UnicodeError, ValueError):
            checksum_paths = set()
        expected = {
            path.relative_to(root).as_posix()
            for path in iter_regular_files(
                root, exclude_relative={"manifests/SHA256SUMS.txt"}
            )
        }
        if checksum_paths != expected:
            result.add_error(
                "CHECKSUM_COVERAGE",
                "SHA256SUMS.txt does not cover every release file except itself",
                path="manifests/SHA256SUMS.txt",
            )
            errors.append("coverage")
        catalog_ok = catalog_ok and not errors
    result.checks["catalogs_valid"] = catalog_ok
    return catalog_ok


def _check_metadata_readiness(
    root: Path,
    bundle: Mapping[str, Any],
    result: ValidationResult,
    *,
    publication_ready: bool,
) -> None:
    complete, problems = metadata_is_complete(bundle)
    for problem in problems:
        if publication_ready:
            result.add_error("MANUAL_ACTION_REQUIRED", problem)
        else:
            result.add_warning("MANUAL_ACTION_REQUIRED", problem)
    config = bundle["config"]
    project = config.get("project", {})
    release = config.get("release", {})
    if project.get("title") != MANUSCRIPT_TITLE:
        result.add_error("PROJECT_TITLE", "project title is not the exact manuscript title")
    if project.get("resource_type") != "dataset":
        result.add_error("RESOURCE_TYPE", "release resource type must be dataset")
    version = project.get("version")
    if not isinstance(version, str) or not re.fullmatch(
        r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
        r"(?:-[0-9A-Za-z.-]+)?",
        version,
    ):
        result.add_error("VERSION_SCHEMA", "release version is not valid semantic version")
    for key in ("test_only", "version_confirmed"):
        if key in release and not isinstance(release[key], bool):
            result.add_error(
                "CONFIG_TYPE",
                f"release.{key} must be a YAML boolean, not a string or number",
            )
    if (
        release.get("test_only") is not True
        and isinstance(version, str)
        and root.name != f"spd_connectome_benchmark_v{version}"
    ):
        result.add_error(
            "ARCHIVE_VERSION_NAME",
            "release directory name does not match the reviewed semantic version",
        )
    record_path = root / "metadata" / "zenodo_record_metadata.json"
    if not record_path.is_file():
        result.add_error("MISSING_ZENODO_METADATA", "Zenodo metadata JSON is missing")
        return
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        result.add_error(
            "INVALID_ZENODO_METADATA",
            "Zenodo metadata JSON is unreadable or malformed",
        )
        return
    if not isinstance(record, Mapping):
        result.add_error(
            "INVALID_ZENODO_METADATA",
            "Zenodo metadata JSON must contain an object",
        )
        return
    expected_record = build_zenodo_record_metadata(bundle)
    if record != expected_record:
        result.add_error(
            "ZENODO_METADATA_MISMATCH",
            "Zenodo metadata does not exactly match the reviewed release snapshot",
        )
    creators = record.get("creators", [])
    if isinstance(creators, list):
        for creator in creators:
            if isinstance(creator, Mapping) and creator.get("orcid"):
                orcid = str(creator["orcid"]).removeprefix("https://orcid.org/")
                if not ORCID_RE.fullmatch(orcid):
                    result.add_error("INVALID_ORCID", "creator ORCID has an invalid format")
    if contains_placeholder(record):
        result.add_error(
            "PLACEHOLDER_METADATA",
            "Zenodo metadata contains TODO or placeholder values",
        )
    if publication_ready:
        if release.get("test_only") is True:
            result.add_error(
                "SYNTHETIC_TEST_RELEASE",
                "a synthetic test release cannot be marked publication-ready",
            )
        snapshot = root / "reproducibility" / "repository_snapshot.json"
        try:
            revision = json.loads(snapshot.read_text(encoding="utf-8"))
            if not isinstance(revision, Mapping):
                raise ValueError("repository snapshot must be an object")
            if revision.get("tracked_worktree_dirty") is not False:
                result.add_error(
                    "DIRTY_SOURCE_SNAPSHOT",
                    "publication-ready release requires a clean source revision",
                )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            result.add_error(
                "INVALID_SOURCE_SNAPSHOT",
                "repository snapshot is missing or invalid",
            )
    else:
        snapshot = root / "reproducibility" / "repository_snapshot.json"
        try:
            revision = json.loads(snapshot.read_text(encoding="utf-8"))
            if (
                isinstance(revision, Mapping)
                and revision.get("tracked_worktree_dirty") is True
            ):
                result.add_warning(
                    "DIRTY_SOURCE_SNAPSHOT",
                    "the recorded commit is only a base revision; rebuild this "
                    "draft from clean, committed source before publication",
                )
        except (OSError, UnicodeError, json.JSONDecodeError):
            # Structural snapshot validation reports malformed or missing files.
            pass
    version_path = root / "VERSION"
    try:
        if version_path.read_text(encoding="utf-8").strip() != version:
            result.add_error(
                "VERSION_FILE_MISMATCH",
                "VERSION does not match the reviewed release snapshot",
            )
    except (OSError, UnicodeError):
        result.add_error("VERSION_FILE", "VERSION is missing or unreadable")
    citation_path = root / "CITATION.cff"
    expected_citation = build_citation_cff(bundle)
    expected_authors = expected_citation.get("authors", [])
    if citation_path.is_file():
        try:
            citation = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
            if not isinstance(citation, Mapping) or not citation.get("authors"):
                result.add_error(
                    "CITATION_AUTHORS",
                    "CITATION.cff must contain at least one reviewed creator",
                )
            elif citation != expected_citation:
                result.add_error(
                    "CITATION_MISMATCH",
                    "CITATION.cff does not exactly match reviewed creator metadata",
                )
        except (OSError, UnicodeError, yaml.YAMLError):
            result.add_error("CITATION_FILE", "CITATION.cff is invalid")
    elif expected_authors or publication_ready:
        result.add_error(
            "CITATION_FILE",
            "CITATION.cff is required after creator metadata is complete",
        )
    result.checks["metadata_complete"] = complete


def _check_unresolved_placeholders(root: Path, result: ValidationResult) -> None:
    patterns = (
        re.compile(r"(?i)\b(?:TODO|TBD|FIXME)\b"),
        re.compile(r"(?i)10\.x{4,}/[A-Za-z0-9./_-]*"),
        re.compile(r"\b0000-0000-0000-0000\b"),
    )
    text_suffixes = {
        ".cff",
        ".csv",
        ".json",
        ".md",
        ".tsv",
        ".txt",
        ".yaml",
        ".yml",
    }
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in patterns):
                result.add_error(
                    "UNRESOLVED_PLACEHOLDER",
                    "publication-ready release contains an unresolved placeholder",
                    path=_relative(root, path),
                    line=line_number,
                )


def _report_markdown(result: ValidationResult, *, publication_ready: bool) -> str:
    lines = [
        "# Release validation report",
        "",
        f"- Structural validation: `{'PASS' if result.ok else 'FAIL'}`",
        f"- Publication-ready mode: `{str(publication_ready).lower()}`",
        f"- Errors: `{len(result.errors)}`",
        f"- Warnings: `{len(result.warnings)}`",
        "",
        "## Errors",
        "",
    ]
    if result.errors:
        for finding in result.errors:
            location = f" ({finding.path}"
            if finding.line is not None:
                location += f":{finding.line}"
            if finding.path:
                location += ")"
            else:
                location = ""
            lines.append(f"- `{finding.code}`{location}: {finding.message}")
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    if result.warnings:
        for finding in result.warnings:
            location = f" ({finding.path})" if finding.path else ""
            lines.append(f"- `{finding.code}`{location}: {finding.message}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "Possible sensitive matches are redacted in reports. This validation",
            "does not replace legal, ethical, license, or data-governance review.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_reports(
    root: Path, result: ValidationResult, *, publication_ready: bool
) -> bool:
    target = root / "manifests"
    if (
        target.is_symlink()
        or (target.exists() and not target.is_dir())
        or not target.resolve().is_relative_to(root)
    ):
        result.add_error(
            "UNSAFE_REPORT_DESTINATION",
            "validation reports cannot be written through an unsafe manifests path",
        )
        return False
    filesystem = result.checks.get("filesystem")
    report_paths = tuple(
        target / name
        for name in ("validation_report.json", "validation_report.md")
    )
    if isinstance(filesystem, dict):
        filesystem["file_count"] = len(iter_regular_files(root)) + sum(
            not path.is_file() for path in report_paths
        )
    payload = result.as_dict()
    payload["publication_ready_requested"] = publication_ready
    payload["publication_status"] = (
        "ready_for_human_upload"
        if publication_ready and result.ok
        else "incomplete_or_not_requested"
    )
    try:
        target.mkdir(parents=True, exist_ok=True)
        (target / "validation_report.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (target / "validation_report.md").write_text(
            _report_markdown(result, publication_ready=publication_ready),
            encoding="utf-8",
        )
    except OSError:
        result.add_error(
            "REPORT_WRITE_FAILED",
            "validation reports could not be written safely",
        )
        return False
    return True


def validate_release(
    release_dir: str | Path,
    config_path: str | Path | None = None,
    *,
    publication_ready: bool = False,
    write_reports: bool = True,
) -> ValidationResult:
    """Run release checks without ever enabling pickle loading."""

    result = ValidationResult()
    supplied_root = Path(release_dir)
    if supplied_root.is_symlink():
        result.add_error(
            "SYMLINK",
            "release directory must not be a symbolic link",
        )
        return result
    root = supplied_root.resolve()
    if not root.is_dir():
        result.add_error("MISSING_RELEASE_DIR", "release directory does not exist")
        return result
    _check_filesystem(root, result, allow_missing_reports=write_reports)
    if any(finding.code == "SYMLINK" for finding in result.errors):
        # Fail before opening policy, table, or archive content through a
        # potentially escaping link.
        return result
    try:
        bundle = _load_bundle(root, config_path)
    except (OSError, ValueError, yaml.YAMLError):
        result.add_error(
            "POLICY_SNAPSHOT",
            "release policy snapshots are missing, malformed, or inconsistent",
        )
        bundle = None
    if bundle is not None:
        _check_dataset_policy(root, bundle, result)
        _check_config_and_reproducibility_snapshots(
            root,
            bundle,
            result,
            bind_to_local_head=(
                config_path is not None
                or root.is_relative_to(Path(__file__).resolve().parents[2])
            ),
        )
        _check_connectomes_and_metadata(root, bundle, result)
        _check_aggregate_outputs(root, bundle, result)
        _check_release_tables(root, bundle, result)
        _check_camcan_companions(root, bundle, result)
    _check_splits(root, result)
    _check_restricted_folders(
        root,
        bundle,
        result,
        publication_ready=publication_ready,
    )
    catalog_ok = _check_catalogs(root, result, bundle)
    privacy_findings = scan_release(
        root,
        bundle.get("forbidden_patterns_path") if bundle else None,
        release_config=bundle.get("config") if bundle else None,
    )
    result.extend(privacy_findings)
    result.checks["privacy_finding_count"] = len(privacy_findings)
    if bundle is not None:
        try:
            _check_metadata_readiness(
                root, bundle, result, publication_ready=publication_ready
            )
        except ValueError:
            result.add_error(
                "ZENODO_METADATA_SCHEMA",
                "reviewed Zenodo metadata schema is invalid",
            )
    if publication_ready:
        _check_unresolved_placeholders(root, result)
    if write_reports:
        reports_written = _write_reports(
            root, result, publication_ready=publication_ready
        )
        # Only refresh a previously valid catalog. Deliberately corrupt
        # manifests/checksums remain failures and are never silently repaired.
        if reports_written and catalog_ok and bundle is not None:
            excluded = {
                "manifests/manifest.tsv",
                "manifests/SHA256SUMS.txt",
            }
            write_manifest(
                root,
                root / "manifests" / "manifest.tsv",
                bundle=bundle,
                exclude_relative=excluded,
            )
            write_sha256sums(
                root,
                root / "manifests" / "SHA256SUMS.txt",
                exclude_relative={"manifests/SHA256SUMS.txt"},
            )
            post_write = ValidationResult()
            _check_catalogs(root, post_write, bundle)
            if not post_write.ok:
                result.errors.extend(post_write.errors)
                _write_reports(
                    root,
                    result,
                    publication_ready=publication_ready,
                )
                write_manifest(
                    root,
                    root / "manifests" / "manifest.tsv",
                    bundle=bundle,
                    exclude_relative=excluded,
                )
                write_sha256sums(
                    root,
                    root / "manifests" / "SHA256SUMS.txt",
                    exclude_relative={"manifests/SHA256SUMS.txt"},
                )
                final_catalog_check = ValidationResult()
                _check_catalogs(root, final_catalog_check, bundle)
                if not final_catalog_check.ok:
                    result.add_error(
                        "CATALOG_WRITE_RACE",
                        "catalogs changed or became inconsistent during validation",
                    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a staged Zenodo release.")
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--publication-ready", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-write-reports", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.dry_run:
        result = validate_release(
            args.release_dir,
            config_path=args.config,
            publication_ready=args.publication_ready,
            write_reports=False,
        )
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "would_write_reports": False,
                    "release_directory_exists": args.release_dir.is_dir(),
                    "publication_ready": args.publication_ready,
                    "validation": result.as_dict(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if result.ok else 2
    result = validate_release(
        args.release_dir,
        config_path=args.config,
        publication_ready=args.publication_ready,
        write_reports=not args.no_write_reports,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
