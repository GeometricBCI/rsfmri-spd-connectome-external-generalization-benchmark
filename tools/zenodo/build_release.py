"""Build a structured public release from a validated safe-export directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .checksums import sha256_file, write_manifest, write_sha256sums
from .metadata import (
    DATASET_DISPLAY_NAMES,
    MANUSCRIPT_TITLE,
    build_citation_cff,
    dataset_rights_display_value,
    manual_action_items,
    write_citation_cff,
    write_metadata_files,
)
from .privacy_scan import scan_release
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
    canonical_dataset,
    dataset_policy,
    is_portable_relative_path,
    load_json,
    load_release_bundle,
    participant_inventory_decision,
    participant_release_decision,
    read_tsv,
    release_connectome_relative_path,
    release_metadata_relative_path,
    validate_connectome_npz,
    validate_release_numeric_settings,
    validate_metadata_columns,
)


ALLOWED_SAFE_EXPORT_ROOT_FILES = {
    ".safe-export-root.json",
    "export_manifest.json",
}
ALLOWED_DATASET_FILE_KEYS = {"connectomes", "metadata", "splits"}
ALLOWED_AGGREGATE_CATEGORIES = {
    "aggregate_metrics",
    "statistical_summaries",
    "figure_source_data",
}
ALLOWED_AGGREGATE_SUFFIXES = {".csv", ".tsv", ".json", ".yaml", ".yml"}
UNRESOLVED_LICENSE_TEXT = "not yet selected"
SYNTHETIC_RELEASE_BANNER = (
    "> **SYNTHETIC TEST PACKAGE — DO NOT UPLOAD OR PUBLISH.**\n\n"
)
REAL_DATA_DRAFT_BANNER = (
    "> **REAL-DATA DRAFT — NOT APPROVED FOR UPLOAD OR PUBLICATION.**  \n"
    "> This staging tree contains real derived data, but publication-ready "
    "validation has not passed.\n\n"
)
REPRODUCIBILITY_ENVIRONMENT_TEXT = "\n".join(
    [
        "name: spd-connectome-benchmark-release",
        "channels:",
        "  - conda-forge",
        "dependencies:",
        "  - python=3.11",
        "  - pip",
        "  - pip:",
        "      - -r requirements-lock.txt",
        "",
    ]
)
REPRODUCIBILITY_COMMANDS_TEXT = "\n".join(
    [
        "# Reproducibility commands",
        "",
        "The validation and packaging programs are maintained in the GitHub",
        "repository; they are not duplicated inside this dataset archive. Check",
        "out the source revision recorded in `git_commit.txt`, install",
        "`requirements-lock.txt`, and run these commands from the repository root:",
        "",
        "If `git_commit.txt` reports `worktree_dirty: true`, the recorded commit is",
        "only the base revision and cannot reproduce uncommitted or untracked source",
        "changes. A publication candidate must be rebuilt from a clean, committed",
        "revision.",
        "",
        "```bash",
        'export RELEASE_DIR="<extracted-release-directory>"',
        'export UPLOAD_DIR="<new-upload-directory>"',
        'export RELEASE_CONFIG="<external-release-config-file>"',
        "",
        "python -m tools.zenodo.validate_release \\",
        '  --release-dir "$RELEASE_DIR" \\',
        '  --config "$RELEASE_CONFIG"',
        "python -m tools.zenodo.package_release \\",
        '  --release-dir "$RELEASE_DIR" \\',
        '  --upload-dir "$UPLOAD_DIR" \\',
        '  --config "$RELEASE_CONFIG"',
        "```",
        "",
        "Both commands compare the complete external release-policy bundle",
        "exactly with the frozen snapshots. Snapshot-only packaging is limited",
        "to explicit archive verification or synthetic testing and does not",
        "establish publication readiness:",
        "",
        "```bash",
        "python -m tools.zenodo.package_release \\",
        '  --release-dir "$RELEASE_DIR" \\',
        '  --upload-dir "$UPLOAD_DIR" \\',
        "  --archive-verification \\",
        "  --dry-run",
        "```",
        "",
    ]
)
DATA_DICTIONARY_FIELD_DEFINITIONS = {
    "connectomes": (
        "SPD connectome matrices with shape samples by regions by regions",
        "float32_or_float64_array",
        "unitless",
    ),
    "sample_uid": (
        "release-specific random namespace identifier not derived from a source ID",
        "string",
        "not_applicable",
    ),
    "fold": ("cross-validation fold label", "string", "not_applicable"),
    "partition": (
        "train, validation, or test membership",
        "enum",
        "not_applicable",
    ),
    "age": ("chronological age", "number", "years"),
    "diagnosis": (
        "dataset-approved diagnosis label",
        "string",
        "not_applicable",
    ),
    "sex": (
        "dataset-approved sex category",
        "string",
        "not_applicable",
    ),
    "site": (
        "dataset-approved acquisition-site category",
        "string",
        "not_applicable",
    ),
    "scanner": (
        "dataset-approved scanner category",
        "string",
        "not_applicable",
    ),
    "target": (
        "reviewed benchmark target label or value",
        "scalar",
        "task_defined",
    ),
    "dataset": ("canonical dataset key", "string", "not_applicable"),
}


class ReleaseBuildError(RuntimeError):
    """Safe-export or release-build failure."""


def canonical_data_dictionary_row(field: str, category: str) -> list[str]:
    """Return the one builder-owned semantic definition for a released field."""

    description, data_type, units = DATA_DICTIONARY_FIELD_DEFINITIONS.get(
        field,
        (
            "allowlisted aggregate field; semantics follow the frozen "
            "analysis configuration",
            "scalar",
            "metric_or_test_defined",
        ),
    )
    return [
        field,
        description,
        data_type,
        units,
        "false",
        category,
        "public_when_explicitly_approved",
    ]


def _copy_verified_file(source: Path, target: Path, expected_sha256: str) -> None:
    """Copy one already-attested file through one no-follow descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise ReleaseBuildError("safe-export source could not be opened safely") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ReleaseBuildError("safe-export source is not a regular file")
        target.parent.mkdir(parents=True, exist_ok=True)
        with os.fdopen(descriptor, "rb", closefd=False) as input_handle, target.open(
            "xb"
        ) as output_handle:
            while chunk := input_handle.read(1024 * 1024):
                digest.update(chunk)
                output_handle.write(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or digest.hexdigest() != expected_sha256
        ):
            target.unlink(missing_ok=True)
            raise ReleaseBuildError(
                "safe-export source changed or failed checksum during copy"
            )
    finally:
        os.close(descriptor)


def _aggregate_allowed_suffixes(
    bundle: Mapping[str, Any], category: str
) -> set[str]:
    content = bundle["config"].get("content_allowlist", {})
    categories = content.get("categories", {}) if isinstance(content, Mapping) else {}
    entry = categories.get(category, {}) if isinstance(categories, Mapping) else {}
    values = entry.get("allowed_extensions", []) if isinstance(entry, Mapping) else []
    if (
        not isinstance(values, list)
        or not values
        or any(
            not isinstance(value, str)
            or value.lower() not in ALLOWED_AGGREGATE_SUFFIXES
            for value in values
        )
    ):
        raise ReleaseBuildError(
            f"aggregate extension allowlist is missing or unsafe for {category}"
        )
    return {value.lower() for value in values}


def _safe_source(root: Path, relative_value: object) -> Path:
    raw = str(relative_value)
    relative = Path(raw)
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        not is_portable_relative_path(raw)
        or relative.is_absolute()
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or ".." in windows.parts
        or "\\" in raw
    ):
        raise ReleaseBuildError("safe-export manifest contains an unsafe path")
    candidate = root / relative
    if candidate.is_symlink():
        raise ReleaseBuildError("safe-export manifest references a symbolic link")
    path = candidate.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ReleaseBuildError("safe-export manifest references a missing or unsafe file")
    return path


def _verify_manifest_file(
    safe_root: Path, descriptor: Mapping[str, Any]
) -> Path:
    if set(descriptor) != {"path", "sha256"}:
        raise ReleaseBuildError("safe-export file descriptor has unknown fields")
    path = _safe_source(safe_root, descriptor["path"])
    expected = str(descriptor["sha256"])
    if len(expected) != 64 or sha256_file(path) != expected:
        raise ReleaseBuildError("safe-export file checksum mismatch")
    return path


def _validate_splits(
    path: Path, *, dataset: str, sample_uids: set[str]
) -> list[Finding]:
    findings: list[Finding] = []
    try:
        columns, rows = read_tsv(path)
    except (OSError, UnicodeError, ValueError):
        return [
            Finding(
                "SPLIT_TSV",
                "error",
                "split TSV is unreadable or malformed",
                path.name,
            )
        ]
    if columns != ["fold", "partition", "sample_uid"]:
        findings.append(
            Finding(
                "SPLIT_SCHEMA",
                "error",
                "split TSV must have fold, partition, sample_uid columns",
                path.name,
            )
        )
        return findings
    if not rows:
        return [
            Finding(
                "SPLIT_EMPTY",
                "error",
                "split TSV must contain at least one complete fold",
                path.name,
            )
        ]
    seen: set[tuple[str, str]] = set()
    fold_members: dict[str, set[str]] = {}
    fold_partitions: dict[str, set[str]] = {}
    allowed_partitions = {"train", "validation", "test"}
    for row in rows:
        fold = row["fold"]
        partition = row["partition"]
        uid = row["sample_uid"]
        if not fold or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", fold):
            findings.append(
                Finding("SPLIT_FOLD", "error", "split fold label is unsafe", path.name)
            )
        if partition not in allowed_partitions:
            findings.append(
                Finding(
                    "SPLIT_PARTITION",
                    "error",
                    "split partition is not train/validation/test",
                    path.name,
                )
            )
        if uid not in sample_uids:
            findings.append(
                Finding(
                    "SPLIT_UNKNOWN_UID",
                    "error",
                    "split references a sample_uid absent from metadata",
                    path.name,
                )
            )
        membership = (fold, uid)
        if membership in seen:
            findings.append(
                Finding(
                    "SPLIT_OVERLAP",
                    "error",
                    "sample_uid appears more than once in a fold",
                    path.name,
                )
            )
        seen.add(membership)
        fold_members.setdefault(fold, set()).add(uid)
        fold_partitions.setdefault(fold, set()).add(partition)
    for fold in fold_members:
        if fold_members[fold] != sample_uids:
            findings.append(
                Finding(
                    "SPLIT_INCOMPLETE",
                    "error",
                    "each fold must assign every public sample exactly once",
                    path.name,
                )
            )
        if not {"train", "test"}.issubset(fold_partitions[fold]):
            findings.append(
                Finding(
                    "SPLIT_PARTITIONS_INCOMPLETE",
                    "error",
                    "each fold requires at least train and test partitions",
                    path.name,
                )
            )
    return findings


def _validate_aggregate_file(
    path: Path,
    allowed_columns: set[str],
    *,
    expected_dataset: str,
    content_category: str,
) -> None:
    required_semantic_fields = {
        "aggregate_metrics": {
            "metric",
            "mean",
            "standard_deviation",
            "standard_error",
            "confidence_interval_lower",
            "confidence_interval_upper",
        },
        "statistical_summaries": {
            "comparison",
            "test_name",
            "statistic",
            "p_value",
            "effect_size",
        },
        "figure_source_data": {
            "metric",
            "value",
            "mean",
            "panel",
        },
    }
    semantic_fields = required_semantic_fields.get(content_category)
    if not semantic_fields:
        raise ReleaseBuildError("aggregate content category is unknown")

    def validate_record(record: Mapping[str, object]) -> None:
        keys = {str(key) for key in record}
        if "dataset" not in keys:
            raise ReleaseBuildError("aggregate output requires a dataset field")
        if canonical_dataset(record.get("dataset")) != expected_dataset:
            raise ReleaseBuildError(
                "aggregate dataset value does not match its manifest descriptor"
            )
        if not keys & semantic_fields:
            raise ReleaseBuildError(
                "aggregate output lacks a category-specific summary field"
            )

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter=delimiter, strict=True))
        except csv.Error as exc:
            raise ReleaseBuildError("aggregate table is malformed") from exc
        if not rows or not rows[0]:
            raise ReleaseBuildError("aggregate table is empty")
        header = rows[0]
        if len(header) != len(set(header)) or any(not column for column in header):
            raise ReleaseBuildError("aggregate table has duplicate or empty columns")
        if set(header) - allowed_columns:
            raise ReleaseBuildError("aggregate table has an unapproved column")
        if any(len(row) != len(header) for row in rows[1:]):
            raise ReleaseBuildError("aggregate table has a malformed row")
        if len(rows) == 1:
            raise ReleaseBuildError("aggregate table contains no records")
        for values in rows[1:]:
            validate_record(dict(zip(header, values)))
        return
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    elif suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle)
    else:
        raise ReleaseBuildError("aggregate output uses an unsafe format")
    if isinstance(value, Mapping) and set(value) == {"records"}:
        value = value["records"]
    if not isinstance(value, list) or not value:
        raise ReleaseBuildError(
            "structured aggregate output must be a non-empty list of records"
        )
    for record in value:
        if not isinstance(record, Mapping) or not record:
            raise ReleaseBuildError("aggregate record must be a non-empty mapping")
        if set(str(key) for key in record) - allowed_columns:
            raise ReleaseBuildError("aggregate record has an unapproved field")
        validate_record(record)
        for item in record.values():
            if isinstance(item, (Mapping, list, tuple)):
                raise ReleaseBuildError("aggregate record values must be scalar")
            if isinstance(item, (float, np.floating)) and not np.isfinite(item):
                raise ReleaseBuildError("aggregate record contains a non-finite value")


def _aggregate_review_is_complete(
    bundle: Mapping[str, Any],
    *,
    dataset: str,
    category: str,
    relative_path: str,
    sha256: str,
) -> bool:
    if bundle["config"].get("release", {}).get("test_only") is True:
        return True
    policy = dataset_policy(bundle, dataset)
    decisions = policy.get("decisions", {}) if isinstance(policy, Mapping) else {}
    decision = decisions.get(category) if isinstance(decisions, Mapping) else None
    if decision not in {
        "allowed_if_explicitly_allowlisted_and_reviewed",
        "requires_manual_approval_and_allowlist",
    }:
        return False
    approvals = bundle["manual_approvals"].get("approvals", {})
    artifacts = (
        approvals.get("release_artifacts", {})
        if isinstance(approvals, Mapping)
        else {}
    )
    record = (
        artifacts.get("aggregate_results_content_and_pdf_metadata_review", {})
        if isinstance(artifacts, Mapping)
        else {}
    )
    scopes = record.get("scope", []) if isinstance(record, Mapping) else []
    if not isinstance(scopes, list):
        return False
    required = ("approved_by", "approved_on", "evidence", "confirmation")
    protocol = bundle["manual_approvals"].get("approval_protocol", {})
    expected_confirmation = (
        str(protocol.get("required_confirmation_text", "")).strip()
        if isinstance(protocol, Mapping)
        and protocol.get("deliberate_confirmation_required") is True
        else str(record.get("confirmation", "")).strip()
    )
    reviewed_artifacts = (
        record.get("reviewed_artifacts", []) if isinstance(record, Mapping) else []
    )
    expected_artifact = {
        "dataset": dataset,
        "content_category": category,
        "relative_path": relative_path,
        "sha256": sha256,
    }
    return (
        isinstance(record, Mapping)
        and record.get("status") == "approved"
        and all(str(record.get(key, "")).strip() for key in required)
        and str(record.get("confirmation", "")).strip() == expected_confirmation
        and (
            category in scopes
            or "aggregate_outputs" in scopes
            or dataset in scopes
        )
        and isinstance(reviewed_artifacts, list)
        and reviewed_artifacts.count(expected_artifact) == 1
    )


def _validate_safe_export(
    safe_export_dir: str | Path,
    bundle: Mapping[str, Any],
    *,
    structural_only: bool = False,
) -> dict[str, Any]:
    supplied_root = Path(safe_export_dir)
    if supplied_root.is_symlink():
        raise ReleaseBuildError("safe-export directory must not be a symbolic link")
    root = supplied_root.resolve()
    if not root.is_dir():
        raise ReleaseBuildError("safe-export directory does not exist")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ReleaseBuildError("safe-export directory must not contain symbolic links")
    if any(
        not path.is_file() and not path.is_dir()
        for path in root.rglob("*")
    ):
        raise ReleaseBuildError(
            "safe-export directory contains a non-regular filesystem object"
        )
    marker_path = root / ".safe-export-root.json"
    manifest_path = root / "export_manifest.json"
    if (
        marker_path.is_symlink()
        or manifest_path.is_symlink()
        or not marker_path.is_file()
        or not manifest_path.is_file()
    ):
        raise ReleaseBuildError("safe-export marker or manifest is missing")
    marker = load_json(marker_path)
    if set(marker) != {
        "schema_version",
        "kind",
        "producer",
        "contains_pickle",
        "contains_original_identifiers",
    }:
        raise ReleaseBuildError("safe-export marker has unknown or missing fields")
    marker_version = marker.get("schema_version")
    if (
        isinstance(marker_version, bool)
        or not isinstance(marker_version, int)
        or marker_version != SCHEMA_VERSION
        or marker.get("kind") != "trusted_safe_export"
        or marker.get("producer") != "tools.zenodo.export_internal"
        or marker.get("contains_pickle") is not False
        or marker.get("contains_original_identifiers") is not False
    ):
        raise ReleaseBuildError("safe-export marker is invalid")
    manifest = load_json(manifest_path)
    allowed_manifest_keys = {
        "schema_version",
        "kind",
        "datasets",
        "aggregate_outputs",
        "producer",
    }
    if set(manifest) != allowed_manifest_keys:
        raise ReleaseBuildError(
            "safe-export manifest has unknown or missing top-level fields"
        )
    manifest_version = manifest.get("schema_version")
    if (
        isinstance(manifest_version, bool)
        or not isinstance(manifest_version, int)
        or manifest_version != SCHEMA_VERSION
        or manifest.get("kind") != "trusted_safe_export"
        or manifest.get("producer") != "tools.zenodo.export_internal"
    ):
        raise ReleaseBuildError("safe-export manifest schema is invalid")
    entries = manifest.get("datasets")
    if not isinstance(entries, list):
        raise ReleaseBuildError("safe-export datasets must be a list")

    referenced = set(ALLOWED_SAFE_EXPORT_ROOT_FILES)
    normalized_entries: list[dict[str, Any]] = []
    seen_datasets: set[str] = set()
    try:
        numeric_settings = validate_release_numeric_settings(bundle["config"])
    except ValueError as exc:
        raise ReleaseBuildError("invalid release matrix settings") from exc
    configured_atlas = numeric_settings["atlas"]
    configured_regions = numeric_settings["expected_regions"]
    configured_matrix_type = numeric_settings["matrix_type"]
    if configured_atlas is None or configured_regions is None:
        raise ReleaseBuildError(
            "release config must bind a supported atlas, region count, and matrix type"
        )
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ReleaseBuildError("safe-export dataset entry must be a mapping")
        if set(entry) != {
            "dataset",
            "sample_count",
            "matrix_type",
            "atlas",
            "n_regions",
            "source_binding_sha256",
            "files",
        }:
            raise ReleaseBuildError("safe-export dataset entry has unknown fields")
        dataset = canonical_dataset(entry["dataset"])
        if dataset not in KNOWN_DATASETS:
            raise ReleaseBuildError("safe-export contains an unknown dataset")
        if dataset in seen_datasets:
            raise ReleaseBuildError("safe-export contains a duplicate dataset")
        seen_datasets.add(dataset)
        if dataset in RESTRICTED_DATASETS:
            raise ReleaseBuildError(
                f"restricted dataset {dataset} cannot contain participant-level files"
            )
        files = entry["files"]
        if not isinstance(files, Mapping) or not files:
            raise ReleaseBuildError("safe-export dataset has no file descriptors")
        if set(files) - ALLOWED_DATASET_FILE_KEYS:
            raise ReleaseBuildError("safe-export dataset has an unknown file category")
        if not {"connectomes", "metadata"}.issubset(files):
            raise ReleaseBuildError("safe-export dataset requires connectomes and metadata")
        for scope, file_key in (
            ("participant_connectomes", "connectomes"),
            ("participant_metadata", "metadata"),
            ("exact_splits", "splits"),
        ):
            if file_key not in files:
                continue
            allowed, reason = participant_release_decision(bundle, dataset, scope)
            if not allowed:
                raise ReleaseBuildError(
                    f"{dataset} {scope} blocked by release policy: {reason}"
                )

        resolved: dict[str, Path] = {}
        verified_digests: dict[str, str] = {}
        for key, descriptor in files.items():
            if not isinstance(descriptor, Mapping):
                raise ReleaseBuildError("safe-export file descriptor must be a mapping")
            source = _verify_manifest_file(root, descriptor)
            expected_name = {
                "connectomes": "connectomes.npz",
                "metadata": "metadata.tsv",
                "splits": "splits.tsv",
            }[key]
            if source.name != expected_name:
                raise ReleaseBuildError("safe-export file uses an unexpected filename")
            expected_relative = Path("datasets") / dataset / expected_name
            if source.relative_to(root) != expected_relative:
                raise ReleaseBuildError(
                    "safe-export file path is not bound to its declared dataset"
                )
            referenced.add(source.relative_to(root).as_posix())
            resolved[key] = source
            verified_digests[key] = str(descriptor["sha256"])

        sample_count_value = entry["sample_count"]
        if (
            isinstance(sample_count_value, bool)
            or not isinstance(sample_count_value, int)
            or sample_count_value <= 0
        ):
            raise ReleaseBuildError("safe-export sample_count must be positive")
        sample_count = sample_count_value
        matrix_type = str(entry["matrix_type"])
        if matrix_type not in {"correlation", "covariance", "spd"}:
            raise ReleaseBuildError("safe-export matrix_type is unknown")
        atlas = str(entry["atlas"])
        n_regions_value = entry["n_regions"]
        if isinstance(n_regions_value, bool) or not isinstance(n_regions_value, int):
            raise ReleaseBuildError("safe-export n_regions must be an integer")
        n_regions = n_regions_value
        atlas_regions = {"schaefer_100": 100, "msdl_39": 39}
        if atlas not in atlas_regions or n_regions != atlas_regions[atlas]:
            raise ReleaseBuildError("safe-export atlas and region count are inconsistent")
        if (
            atlas != configured_atlas
            or n_regions != configured_regions
            or matrix_type != configured_matrix_type
        ):
            raise ReleaseBuildError(
                "safe-export matrix schema does not match the reviewed release config"
            )
        source_binding = str(entry["source_binding_sha256"])
        if len(source_binding) != 64 or any(
            char not in "0123456789abcdef" for char in source_binding
        ):
            raise ReleaseBuildError("safe-export source binding is invalid")
        if bundle["config"].get("release", {}).get("test_only") is not True:
            approvals = bundle["manual_approvals"].get("approvals", {})
            if isinstance(approvals, Mapping) and isinstance(
                approvals.get("datasets"), Mapping
            ):
                approvals = approvals["datasets"]
            approval = (
                approvals.get(dataset, {}) if isinstance(approvals, Mapping) else {}
            )
            if (
                not isinstance(approval, Mapping)
                or approval.get("source_binding_sha256") != source_binding
                or canonical_dataset(approval.get("dataset")) != dataset
                or approval.get("atlas") != atlas
                or approval.get("n_regions") != n_regions
            ):
                raise ReleaseBuildError(
                    f"{dataset} approval is not bound to source, atlas, and region count"
                )
        if not structural_only:
            count, findings = validate_connectome_npz(
                resolved["connectomes"],
                matrix_type=matrix_type,
                symmetry_tolerance=numeric_settings["symmetry_tolerance"],
                diagonal_tolerance=numeric_settings["diagonal_tolerance"],
                spd_eigenvalue_tolerance=numeric_settings[
                    "spd_eigenvalue_tolerance"
                ],
                expected_regions=configured_regions,
            )
            if findings:
                raise ReleaseBuildError(findings[0].message)
            columns, rows = read_tsv(resolved["metadata"])
            metadata_findings = validate_metadata_columns(
                columns, bundle=bundle, dataset=dataset
            )
            if metadata_findings:
                raise ReleaseBuildError(metadata_findings[0].message)
            if "sample_uid" not in columns:
                raise ReleaseBuildError("public metadata requires sample_uid")
            sample_uids = [row["sample_uid"] for row in rows]
            if len(sample_uids) != len(set(sample_uids)):
                raise ReleaseBuildError("sample_uid values must be unique")
            if any(not SAFE_UID_RE.fullmatch(value) for value in sample_uids):
                raise ReleaseBuildError("sample_uid does not match the release-safe schema")
            if count != sample_count or len(rows) != sample_count:
                raise ReleaseBuildError("safe-export sample counts do not match")
            if "splits" in resolved:
                split_findings = _validate_splits(
                    resolved["splits"],
                    dataset=dataset,
                    sample_uids=set(sample_uids),
                )
                if split_findings:
                    raise ReleaseBuildError(split_findings[0].message)
        normalized_entries.append(
            {
                "dataset": dataset,
                "sample_count": sample_count,
                "matrix_type": matrix_type,
                "atlas": atlas,
                "n_regions": n_regions,
                "source_binding_sha256": source_binding,
                "files": resolved,
                "file_sha256": verified_digests,
            }
        )

    aggregate_entries = manifest.get("aggregate_outputs", [])
    if not isinstance(aggregate_entries, list):
        raise ReleaseBuildError("aggregate_outputs must be a list")
    normalized_aggregates: list[dict[str, Any]] = []
    content_config = bundle["config"].get("content_allowlist", {})
    allowed_columns_by_category = (
        content_config.get("aggregate_columns", {})
        if isinstance(content_config, Mapping)
        else {}
    )
    metadata_tables = bundle["metadata_allowlist"].get("tables", {})
    for entry in aggregate_entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "dataset",
            "content_category",
            "file",
        }:
            raise ReleaseBuildError("aggregate output descriptor has unknown fields")
        dataset = canonical_dataset(entry["dataset"])
        category = str(entry["content_category"])
        if dataset not in KNOWN_DATASETS:
            raise ReleaseBuildError("aggregate output uses an unknown dataset")
        if category not in ALLOWED_AGGREGATE_CATEGORIES:
            raise ReleaseBuildError("aggregate output category is not allowed")
        descriptor = entry["file"]
        if not isinstance(descriptor, Mapping):
            raise ReleaseBuildError("aggregate file descriptor must be a mapping")
        source = _verify_manifest_file(root, descriptor)
        source_relative = source.relative_to(root)
        if (
            len(source_relative.parts) != 3
            or source_relative.parts[0] != "aggregate_outputs"
            or source_relative.parts[1] != dataset
        ):
            raise ReleaseBuildError(
                "aggregate file path is not bound to its declared dataset"
            )
        source_digest = sha256_file(source)
        if not _aggregate_review_is_complete(
            bundle,
            dataset=dataset,
            category=category,
            relative_path=source_relative.as_posix(),
            sha256=source_digest,
        ):
            raise ReleaseBuildError(
                "aggregate output lacks a path-and-checksum-bound artifact review"
            )
        if source.suffix.lower() not in _aggregate_allowed_suffixes(bundle, category):
            raise ReleaseBuildError("aggregate output uses an unsafe format")
        if re.search(
            r"(?i)(?:individual|participant|subject|prediction|embedding|residual)",
            source.stem,
        ):
            raise ReleaseBuildError(
                "aggregate filename suggests blocked individual-level output"
            )
        referenced.add(source.relative_to(root).as_posix())
        configured = (
            allowed_columns_by_category.get(category, [])
            if isinstance(allowed_columns_by_category, Mapping)
            else []
        )
        if (
            (not isinstance(configured, list) or not configured)
            and isinstance(metadata_tables, Mapping)
            and isinstance(metadata_tables.get(category), Mapping)
        ):
            configured = metadata_tables[category].get("allowed_columns", [])
        if not isinstance(configured, list) or not configured:
            raise ReleaseBuildError(
                f"no explicit aggregate column allowlist for {category}"
            )
        _validate_aggregate_file(
            source,
            {str(value) for value in configured},
            expected_dataset=dataset,
            content_category=category,
        )
        normalized_aggregates.append(
            {
                "dataset": dataset,
                "content_category": category,
                "source": source,
                "sha256": source_digest,
            }
        )

    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_directories: set[str] = set()
    for relative_name in referenced:
        parent = PurePosixPath(relative_name).parent
        while parent.as_posix() not in {".", ""}:
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    portable_entries = sorted(actual_files | actual_directories)
    portable_keys = [value.casefold() for value in portable_entries]
    if (
        any(not is_portable_relative_path(value) for value in portable_entries)
        or len(portable_keys) != len(set(portable_keys))
    ):
        raise ReleaseBuildError(
            "safe-export contains a non-portable or case-colliding path"
        )
    if actual_directories != expected_directories:
        raise ReleaseBuildError(
            "safe-export contains unmanifested or missing directories"
        )
    if actual_files != referenced:
        raise ReleaseBuildError("safe-export contains unmanifested or missing files")
    if any(path.suffix.lower() in {".pkl", ".pickle"} for path in root.rglob("*")):
        raise ReleaseBuildError("safe-export contains a forbidden pickle")
    if not structural_only:
        privacy_findings = scan_release(
            root,
            bundle.get("forbidden_patterns_path"),
            release_config=bundle["config"],
        )
        if privacy_findings:
            raise ReleaseBuildError(
                f"safe-export privacy scan failed ({len(privacy_findings)} finding(s))"
            )
    return {
        "root": root,
        "datasets": normalized_entries,
        "aggregate_outputs": normalized_aggregates,
    }


def _template_text(name: str) -> str:
    repository_root = Path(__file__).resolve().parents[2]
    path = repository_root / "release_templates" / name
    if not path.is_file():
        raise ReleaseBuildError(f"required release template is missing: {name}")
    return path.read_text(encoding="utf-8")


def _render_template(
    name: str,
    *,
    replacements: Mapping[str, str],
) -> str:
    text = _template_text(name)
    for key, replacement in replacements.items():
        text = text.replace("{{" + key + "}}", replacement)
    unresolved = sorted(
        set(re.findall(r"\{\{[A-Z0-9_]+\}\}", text))
    )
    if unresolved:
        raise ReleaseBuildError(
            f"template {name} has {len(unresolved)} unresolved variable(s)"
        )
    return text


def _release_license_value(bundle: Mapping[str, Any], key: str) -> str:
    licenses = bundle["config"].get("metadata", {}).get("licenses", {})
    value = licenses.get(key) if isinstance(licenses, Mapping) else None
    if isinstance(value, Mapping):
        return str(value.get("identifier") or UNRESOLVED_LICENSE_TEXT)
    return str(value or UNRESOLVED_LICENSE_TEXT)


def _release_license_status(bundle: Mapping[str, Any], key: str) -> str:
    """Return a human-readable review state for one release license."""

    licenses = bundle["config"].get("metadata", {}).get("licenses", {})
    value = licenses.get(key) if isinstance(licenses, Mapping) else None
    if isinstance(value, Mapping):
        status = str(value.get("status") or "").strip()
        return {
            "approved": "approved",
            "documented_in_repository": "documented in the repository",
            "dataset_specific_rights_confirmed_record_level_review_required": (
                "dataset-specific rights confirmed; record-level review pending"
            ),
            "manual_required": "pending final review",
        }.get(status, status.replace("_", " ") or "pending final review")
    if value is not None and str(value).strip():
        return "documented in the release configuration"
    return "pending final review"


def _dataset_rights_value(
    bundle: Mapping[str, Any], dataset: str, key: str
) -> str:
    metadata = bundle["config"].get("metadata", {})
    rights = (
        metadata.get("dataset_rights", {}).get(dataset, {})
        if isinstance(metadata, Mapping)
        and isinstance(metadata.get("dataset_rights", {}), Mapping)
        else {}
    )
    if not isinstance(rights, Mapping):
        return "manual review required"
    return dataset_rights_display_value(rights, key)


def release_document_banner(bundle: Mapping[str, Any]) -> str:
    """Return the mandatory status banner for generated public documents."""

    release = bundle["config"].get("release", {})
    if release.get("test_only") is True:
        return SYNTHETIC_RELEASE_BANNER
    if (
        release.get("version_confirmed") is not True
        or release.get("publication_ready") is not True
        or manual_action_items(bundle)
    ):
        return REAL_DATA_DRAFT_BANNER
    return ""


def expected_licenses_text(bundle: Mapping[str, Any]) -> str:
    """Render the canonical root license inventory from frozen policy."""

    derived = _release_license_value(bundle, "derived_data")
    documentation = _release_license_value(bundle, "documentation")
    rendered = _render_template(
        "LICENSES.md",
        replacements={
            "VERSION": str(
                bundle["config"].get("project", {}).get("version", "")
            ),
            "DERIVED_DATA_LICENSE_IDENTIFIER": derived,
            "DERIVED_DATA_LICENSE_APPROVAL_STATUS": _release_license_status(
                bundle, "derived_data"
            ),
            "DOCUMENTATION_LICENSE_IDENTIFIER": documentation,
            "DOCUMENTATION_LICENSE_APPROVAL_STATUS": _release_license_status(
                bundle, "documentation"
            ),
            **{
                f"{dataset.upper()}_{token}": _dataset_rights_value(
                    bundle, dataset, key
                )
                for dataset in ("camcan", "abide", "cobre")
                for token, key in (
                    ("RIGHTS_STATUS", "status"),
                    ("DATA_LICENSE", "license"),
                    ("CONDITIONS", "conditions"),
                    ("CITATIONS", "required_citations"),
                    ("RIGHTS_STATEMENT", "rights_statement"),
                )
            },
            "MANUAL_ACTIONS": "\n".join(
                f"- {item}" for item in manual_action_items(bundle)
            )
            or "- No unresolved actions recorded.",
        },
    )
    return release_document_banner(bundle) + rendered


def _write_restricted_reconstruction(
    release_root: Path, bundle: Mapping[str, Any]
) -> None:
    template = _template_text("RESTRICTED_DATA_RECONSTRUCTION.md")
    version = str(bundle["config"].get("project", {}).get("version", ""))
    settings = validate_release_numeric_settings(bundle["config"])
    atlas = str(settings["atlas"])
    matrix_type = str(settings["matrix_type"])
    reconstruction = bundle["config"].get("restricted_reconstruction", {})
    reconstruction = reconstruction if isinstance(reconstruction, Mapping) else {}
    reviews = reconstruction.get("dataset_reviews", {})
    reviews = reviews if isinstance(reviews, Mapping) else {}
    commit, _ = _git_snapshot(Path(__file__).resolve().parents[2])
    for dataset in ("adni", "adnidod", "oasis3"):
        target = release_root / "restricted_reconstruction" / dataset
        target.mkdir(parents=True, exist_ok=True)
        label = {"adni": "ADNI", "adnidod": "ADNI-DOD", "oasis3": "OASIS-3"}[
            dataset
        ]
        source_argument = (
            "--oasis3_raw_dir"
            if dataset == "oasis3"
            else "--adni_adnidod_raw_dir"
        )
        replacements = {
            "DATASET": label,
            "VERSION": version,
            "DATASET_DISPLAY_NAME": label,
            "PROJECT_VERSION": version,
            "DATASET_KEY": dataset,
            "DATASET_CLI_NAME": dataset,
            "ATLAS_NAME": atlas,
            "DATASET_SOURCE_ARGUMENT": source_argument,
            "AUTHORITATIVE_SELECTION_RULES": (
                "Apply the dataset-specific rules implemented by "
                "`prepare_fmri_datasets.py`, the frozen preprocessing snapshot, "
                "and the authorized source protocol. Before publication, a human "
                "reviewer must confirm that these sources agree; no participant "
                "membership or source identifier is distributed here."
            ),
            "MODEL_SEED": "1",
            "DATA_SHUFFLE_SEED": "42",
        }
        rendered = template
        for key, value in replacements.items():
            rendered = rendered.replace("{{" + key + "}}", value)
        if re.search(r"\{\{[A-Z0-9_]+\}\}", rendered):
            raise ReleaseBuildError(
                f"restricted reconstruction template is incomplete for {dataset}"
            )
        (target / "README.md").write_text(rendered, encoding="utf-8")
        selection = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "participant_level_redistribution": "forbidden",
            "atlas": atlas,
            "matrix_construction": (
                f"per-scan OAS {matrix_type} with SPD regularization"
            ),
            "task": reconstruction.get("task", "regression"),
            "target": reconstruction.get("target", "Age"),
            "sample_selection": {
                "implementation": "prepare_fmri_datasets.py",
                "repository_commit": commit,
                "authoritative_reference": (
                    reviews.get(dataset, {}).get("selection_rules_reference")
                    if isinstance(reviews.get(dataset), Mapping)
                    else None
                ),
                "reviewed": (
                    reviews.get(dataset, {}).get("selection_rules_reviewed") is True
                    if isinstance(reviews.get(dataset), Mapping)
                    else False
                ),
            },
            "split_reconstruction": {
                "protocols": reconstruction.get(
                    "outer_cv_protocols",
                    ["subject_grouped_kfold", "leave_one_dataset_out"],
                ),
                "grouping_rule": reconstruction.get(
                    "grouping_rule",
                    "keep every scan from one non-public subject group in one fold",
                ),
                "kfold_n_splits": reconstruction.get("kfold_n_splits", 5),
                "model_seed": reconstruction.get("model_seed", 1),
                "data_shuffle_seed": reconstruction.get("data_shuffle_seed", 42),
                "release_exact_membership": False,
                "aggregate_fold_counts": (
                    reviews.get(dataset, {}).get("aggregate_fold_counts", [])
                    if isinstance(reviews.get(dataset), Mapping)
                    else []
                ),
            },
        }
        (target / "selection_config.yaml").write_text(
            yaml.safe_dump(selection, sort_keys=False), encoding="utf-8"
        )
        (target / "reconstruction_commands.md").write_text(
            "\n".join(
                [
                    f"# Reconstructing {label} inputs",
                    "",
                    "Obtain authorized source data independently. Define local",
                    "directories outside the Git checkout, then run:",
                    "",
                    "```bash",
                    'export AUTHORIZED_SOURCE_ROOT="<authorized-source-directory>"',
                    'export PREPARED_OUTPUT_ROOT="<prepared-output-directory>"',
                    'export BENCHMARK_RESULTS_ROOT="<benchmark-results-directory>"',
                    "",
                    "python prepare_fmri_datasets.py \\",
                    f"  --dataset {dataset} \\",
                    f"  --atlas {atlas} \\",
                    '  --data_root "$PREPARED_OUTPUT_ROOT" \\',
                    f'  {source_argument} "$AUTHORIZED_SOURCE_ROOT"',
                    "",
                    "python run_benchmark.py \\",
                    "  --dry-run \\",
                    "  --task regression \\",
                    "  --target Age \\",
                    f"  --atlas {atlas} \\",
                    f"  --datasets {dataset} \\",
                    "  --cv kfold \\",
                    "  --seed 1 \\",
                    "  --data-shuffle-seed 42 \\",
                    '  --input-root "$PREPARED_OUTPUT_ROOT" \\',
                    '  --output-dir "$BENCHMARK_RESULTS_ROOT"',
                    "```",
                    "",
                    "Keep reconstructed participant-level files in the authorized",
                    "environment. Do not pass them to the trusted exporter or public",
                    "release builder; only separately reviewed aggregate outputs may",
                    "enter a future public release.",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def _git_snapshot(repository_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable", True
    return commit, dirty


def _software_versions() -> dict[str, Any]:
    packages = {}
    for distribution in (
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "PyYAML",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "packages": packages,
    }


def _write_reproducibility(release_root: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    target = release_root / "reproducibility"
    target.mkdir(parents=True, exist_ok=True)
    (target / "commands.md").write_text(
        REPRODUCIBILITY_COMMANDS_TEXT,
        encoding="utf-8",
    )
    requirements = repository_root / "requirements.txt"
    lock_text = (
        requirements.read_text(encoding="utf-8")
        if requirements.is_file()
        else "# Exact dependency lock must be supplied before publication.\n"
    )
    (target / "requirements-lock.txt").write_text(lock_text, encoding="utf-8")
    (target / "environment.yml").write_text(
        REPRODUCIBILITY_ENVIRONMENT_TEXT,
        encoding="utf-8",
    )
    commit, dirty = _git_snapshot(repository_root)
    (target / "git_commit.txt").write_text(
        f"commit: {commit}\nworktree_dirty: {str(dirty).lower()}\n",
        encoding="utf-8",
    )
    (target / "software_versions.json").write_text(
        json.dumps(_software_versions(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": commit,
        "tracked_worktree_dirty": dirty,
        "source_repository": CANONICAL_REPOSITORY_URL,
    }
    (target / "repository_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_release_metadata_tables(
    release_root: Path,
    safe_export: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> None:
    metadata_dir = release_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = metadata_dir / "dataset_inventory.tsv"
    included = {entry["dataset"]: entry for entry in safe_export["datasets"]}
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "dataset",
                "content_category",
                "access_category",
                "file_count",
                "sample_count",
                "atlas",
                "matrix_type",
                "release_policy_decision",
            ]
        )
        for dataset in (
            "abide",
            "adni",
            "oasis3",
            "camcan",
            "cobre",
            "adnidod",
            "1000brains",
        ):
            entry = included.get(dataset)
            for category, file_key in (
                ("participant_connectomes", "connectomes"),
                ("participant_metadata", "metadata"),
                ("exact_splits", "splits"),
            ):
                present = bool(entry and file_key in entry["files"])
                release_decision = participant_inventory_decision(
                    bundle,
                    dataset,
                    category,
                    present=present,
                )
                writer.writerow(
                    [
                        dataset,
                        category,
                        "public" if present else "not_included",
                        1 if present else 0,
                        entry["sample_count"] if present else 0,
                        entry["atlas"] if present else "",
                        entry["matrix_type"] if present else "",
                        release_decision,
                    ]
                )
        aggregate_counts: dict[tuple[str, str], int] = {}
        for artifact in safe_export["aggregate_outputs"]:
            key = (artifact["dataset"], artifact["content_category"])
            aggregate_counts[key] = aggregate_counts.get(key, 0) + 1
        for (dataset, category), file_count in sorted(aggregate_counts.items()):
            policy = dataset_policy(bundle, dataset)
            decisions = policy.get("decisions", {}) if isinstance(policy, Mapping) else {}
            decision = (
                decisions.get(category, "forbidden")
                if isinstance(decisions, Mapping)
                else "forbidden"
            )
            writer.writerow(
                [
                    dataset,
                    category,
                    "public",
                    file_count,
                    0,
                    "",
                    "",
                    decision,
                ]
            )
    dictionary_fields: set[tuple[str, str]] = set()
    for entry in safe_export["datasets"]:
        dataset = entry["dataset"]
        dictionary_fields.add(("connectomes", "participant_connectomes"))
        metadata_columns, _ = read_tsv(
            release_root / release_metadata_relative_path(dataset)
        )
        dictionary_fields.update(
            (column, "participant_metadata") for column in metadata_columns
        )
        if "splits" in entry["files"]:
            split_columns, _ = read_tsv(
                release_root / "splits" / dataset / "splits.tsv"
            )
            dictionary_fields.update(
                (column, "exact_splits") for column in split_columns
            )
    for artifact in safe_export["aggregate_outputs"]:
        folder = {
            "aggregate_metrics": "aggregate_metrics",
            "statistical_summaries": "statistical_tests",
            "figure_source_data": "figure_source_data",
        }[artifact["content_category"]]
        path = (
            release_root
            / "benchmark_results"
            / folder
            / artifact["dataset"]
            / artifact["source"].name
        )
        if path.suffix.lower() in {".csv", ".tsv"}:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            with path.open("r", encoding="utf-8", newline="") as source:
                fields = next(csv.reader(source, delimiter=delimiter, strict=True))
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
        dictionary_fields.update(
            (field, artifact["content_category"]) for field in fields
        )

    dictionary_rows = [
        canonical_data_dictionary_row(field, category)
        for field, category in sorted(
            dictionary_fields, key=lambda value: (value[1], value[0])
        )
    ]

    dictionary_path = metadata_dir / "data_dictionary.tsv"
    with dictionary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "field",
                "description",
                "data_type",
                "units",
                "nullable",
                "content_category",
                "access_category",
            ]
        )
        writer.writerows(dictionary_rows)
    if "camcan" in included:
        camcan_metadata_columns, _ = read_tsv(
            release_root / CAMCAN_METADATA_RELATIVE_PATH
        )
        camcan_fields = {
            ("connectomes", "participant_connectomes"),
            *(
                (column, "participant_metadata")
                for column in camcan_metadata_columns
            ),
        }
        camcan_dictionary_rows = [
            row
            for row in dictionary_rows
            if (row[0], row[5]) in camcan_fields
        ]
        camcan_dictionary_path = (
            release_root / CAMCAN_DATA_DICTIONARY_RELATIVE_PATH
        )
        camcan_dictionary_path.parent.mkdir(parents=True, exist_ok=True)
        with camcan_dictionary_path.open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "field",
                    "description",
                    "data_type",
                    "units",
                    "nullable",
                    "content_category",
                    "access_category",
                ]
            )
            writer.writerows(camcan_dictionary_rows)
    provenance_path = metadata_dir / "provenance.tsv"
    repository_root = Path(__file__).resolve().parents[2]
    commit, _ = _git_snapshot(repository_root)
    release_config_snapshot = dict(bundle["config"])
    release_config_snapshot.pop("paths", None)
    fingerprint_payload = {
        "config": release_config_snapshot,
        "dataset_policy": bundle["dataset_policy"],
        "metadata_allowlist": bundle["metadata_allowlist"],
        "forbidden_patterns": bundle["forbidden_patterns"],
        "manual_approvals": bundle["manual_approvals"],
    }
    config_fingerprint = hashlib.sha256(
        yaml.safe_dump(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    with provenance_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "relative_path",
                "sha256",
                "source_category",
                "content_category",
                "repository_commit",
                "config_fingerprint",
                "software_version",
                "transformation",
            ]
        )
        for dataset, entry in sorted(included.items()):
            for file_key, category, relative in (
                (
                    "connectomes",
                    "participant_connectomes",
                    release_connectome_relative_path(dataset),
                ),
                (
                    "metadata",
                    "participant_metadata",
                    release_metadata_relative_path(dataset),
                ),
                (
                    "splits",
                    "exact_splits",
                    Path("splits") / dataset / "splits.tsv",
                ),
            ):
                target = release_root / relative
                if file_key not in entry["files"] or not target.is_file():
                    continue
                writer.writerow(
                    [
                        relative.as_posix(),
                        sha256_file(target),
                        "trusted_safe_export",
                        category,
                        commit,
                        config_fingerprint,
                        bundle["config"].get("project", {}).get("version", ""),
                        "allowlisted export with source identifiers and paths removed",
                    ]
                )
        aggregate_folders = {
            "aggregate_metrics": "aggregate_metrics",
            "statistical_summaries": "statistical_tests",
            "figure_source_data": "figure_source_data",
        }
        for artifact in sorted(
            safe_export["aggregate_outputs"],
            key=lambda value: (
                value["dataset"],
                value["content_category"],
                value["source"].name,
            ),
        ):
            relative = (
                Path("benchmark_results")
                / aggregate_folders[artifact["content_category"]]
                / artifact["dataset"]
                / artifact["source"].name
            )
            output = release_root / relative
            writer.writerow(
                [
                    relative.as_posix(),
                    sha256_file(output),
                    "reviewed_aggregate_output",
                    artifact["content_category"],
                    commit,
                    config_fingerprint,
                    bundle["config"].get("project", {}).get("version", ""),
                    "explicitly allowlisted aggregate artifact copy",
                ]
            )
    (metadata_dir / "release_policy_snapshot.yaml").write_text(
        yaml.safe_dump(bundle["dataset_policy"], sort_keys=False),
        encoding="utf-8",
    )
    (metadata_dir / "manual_approvals_snapshot.yaml").write_text(
        yaml.safe_dump(bundle["manual_approvals"], sort_keys=False),
        encoding="utf-8",
    )
    (metadata_dir / "release_config_snapshot.yaml").write_text(
        yaml.safe_dump(release_config_snapshot, sort_keys=False),
        encoding="utf-8",
    )
    (metadata_dir / "metadata_allowlist_snapshot.yaml").write_text(
        yaml.safe_dump(bundle["metadata_allowlist"], sort_keys=False),
        encoding="utf-8",
    )
    (metadata_dir / "forbidden_patterns_snapshot.yaml").write_text(
        yaml.safe_dump(bundle["forbidden_patterns"], sort_keys=False),
        encoding="utf-8",
    )
    write_metadata_files(bundle, metadata_dir)


def _write_config_snapshots(
    release_root: Path,
    bundle: Mapping[str, Any],
    safe_export: Mapping[str, Any],
) -> None:
    target = release_root / "configs"
    settings = validate_release_numeric_settings(bundle["config"])
    (target / "datasets").mkdir(parents=True, exist_ok=True)
    included = {entry["dataset"]: entry for entry in safe_export["datasets"]}
    for dataset in (
        "abide",
        "adni",
        "oasis3",
        "camcan",
        "cobre",
        "adnidod",
        "1000brains",
    ):
        policy = bundle["dataset_policy"].get("datasets", {}).get(dataset, {})
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "release_policy": policy,
        }
        if dataset in included:
            snapshot["public_connectome_schema"] = {
                "atlas": included[dataset]["atlas"],
                "n_regions": included[dataset]["n_regions"],
                "matrix_type": included[dataset]["matrix_type"],
                "source_binding_sha256": included[dataset][
                    "source_binding_sha256"
                ],
            }
        (target / "datasets" / f"{dataset}.yaml").write_text(
            yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
        )
    (target / "experiments").mkdir(parents=True, exist_ok=True)
    (target / "experiments" / "README.md").write_text(
        "# Experiment configurations\n\n"
        "The frozen benchmark result configurations, when approved and present, "
        "are listed in the manifest.\n",
        encoding="utf-8",
    )
    (target / "preprocessing").mkdir(parents=True, exist_ok=True)
    preprocessing = {
        "schema_version": SCHEMA_VERSION,
        "atlas_by_dataset": {
            entry["dataset"]: {
                "atlas": entry["atlas"],
                "n_regions": entry["n_regions"],
            }
            for entry in safe_export["datasets"]
        },
        "connectome": {
            "estimator": "source-bound trusted export; OAS when reconstructed from time series",
            "matrix_type": settings["matrix_type"],
            "allowed_dtypes": ["float32", "float64"],
            "spd_regularization": True,
        },
    }
    (target / "preprocessing" / "connectome.yaml").write_text(
        yaml.safe_dump(preprocessing, sort_keys=False), encoding="utf-8"
    )


def _camcan_license_text(bundle: Mapping[str, Any]) -> str:
    policy = dataset_policy(bundle, "camcan")
    authorization = policy.get("participant_level_connectomes", {})
    if not isinstance(authorization, Mapping):
        raise ReleaseBuildError("CamCAN release authorization is missing")
    prohibited = authorization.get("prohibited", [])
    citations = authorization.get("required_citation", [])
    license_entry = authorization.get("license", {})
    confirmation = authorization.get("confirmation", {})
    if not isinstance(prohibited, list) or not isinstance(citations, list):
        raise ReleaseBuildError("CamCAN release authorization is malformed")
    if not isinstance(license_entry, Mapping) or not isinstance(
        confirmation, Mapping
    ):
        raise ReleaseBuildError("CamCAN license or confirmation is malformed")
    excluded_labels = {
        "raw_images": "raw MRI images",
        "raw_T1_images": "raw T1-weighted images",
        "T1w_images": "T1-weighted images",
        "identifiable_images": "identifiable images",
        "home_interview_variables": "Home Interview variables",
        "identifiable_behavioural_variables": (
            "identifiable behavioural variables"
        ),
        "CCID": "CCID and other source identifiers",
    }
    excluded = list(
        dict.fromkeys(
            excluded_labels.get(str(value), str(value).replace("_", " "))
            for value in prohibited
        )
    )
    release = bundle["config"].get("release", {})
    if release.get("test_only") is True:
        status = "synthetic test artifact; do not upload or publish"
    elif release.get("publication_ready") is not True:
        status = "real-data draft; not approved for upload or publication"
    else:
        status = "publication-ready release"
    review_sentence = (
        "The current artifact still requires final human review."
        if release.get("publication_ready") is not True
        else "The publication-ready validation report records the final review."
    )
    return "\n".join(
        [
            "CamCAN-derived connectome release notice",
            "",
            f"Release status: {status}.",
            (
                "License: Creative Commons Attribution 4.0 International "
                f"({license_entry.get('type', '')})"
            ),
            "License URI: https://creativecommons.org/licenses/by/4.0/",
            (
                "Permission basis: project-held written confirmation for the "
                "derived-data scope recorded in the frozen policy."
            ),
            (
                "Public scope: derived Schaefer-100 functional-connectivity "
                "matrices, release-generated sample_uid values, age, and sex."
            ),
            "Excluded material: " + "; ".join(excluded) + ".",
            "",
            "Required citation:",
            (
                "Shafto, M.A. et al. (2014). The Cambridge Centre for Ageing "
                "and Neuroscience (Cam-CAN) study protocol: a cross-sectional, "
                "lifespan, multidisciplinary examination of healthy cognitive "
                "ageing. BMC Neurology 14:204. "
                "https://doi.org/10.1186/s12883-014-0204-1"
            ),
            "",
            "This notice does not relicense CamCAN source data. The confirmed "
            "permission, frozen policy snapshot, completed artifact review, and "
            "source-binding SHA-256 must jointly define any published files. "
            + review_sentence,
            "",
        ]
    )


def _write_camcan_license(release_root: Path, bundle: Mapping[str, Any]) -> None:
    if not (release_root / CAMCAN_CONNECTOME_RELATIVE_PATH).is_file():
        return
    path = release_root / CAMCAN_LICENSE_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_camcan_license_text(bundle), encoding="utf-8")


def _refresh_catalogs(release_root: Path, bundle: Mapping[str, Any]) -> None:
    manifests = release_root / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    excluded = {
        "manifests/manifest.tsv",
        "manifests/SHA256SUMS.txt",
    }
    write_manifest(
        release_root,
        manifests / "manifest.tsv",
        bundle=bundle,
        exclude_relative=excluded,
    )
    write_sha256sums(
        release_root,
        manifests / "SHA256SUMS.txt",
        exclude_relative={"manifests/SHA256SUMS.txt"},
    )


def _build_release_in_workspace(
    config_path: str | Path,
    safe_export_dir: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
) -> Path:
    """Create a new immutable staging tree from safe public formats only."""

    bundle = load_release_bundle(config_path)
    config = bundle["config"]
    project = config.get("project", {})
    if project.get("title") != MANUSCRIPT_TITLE:
        raise ReleaseBuildError("release config does not use the exact manuscript title")
    version = str(project.get("version", "")).strip()
    if not version:
        raise ReleaseBuildError("release version is missing")
    release = config.get("release", {})
    directory_name = str(
        release.get(
            "directory_name",
            release.get("archive_basename", f"spd_connectome_benchmark_v{version}"),
        )
    )
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,79}", directory_name):
        raise ReleaseBuildError("release directory name is unsafe")
    if (
        release.get("test_only") is not True
        and directory_name != f"spd_connectome_benchmark_v{version}"
    ):
        raise ReleaseBuildError(
            "publication archive basename must match the reviewed semantic version"
        )
    safe_export = _validate_safe_export(
        safe_export_dir, bundle, structural_only=False
    )
    supplied_output = Path(output_dir)
    if supplied_output.is_symlink():
        raise ReleaseBuildError("output directory must not be a symbolic link")
    output_root = supplied_output.resolve()
    safe_root = Path(safe_export_dir).resolve()
    if (
        output_root == safe_root
        or output_root.is_relative_to(safe_root)
        or safe_root.is_relative_to(output_root)
    ):
        raise ReleaseBuildError("output directory must not overlap the safe export")
    destination = output_root / "staging" / directory_name
    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "would_write": False,
                    "release_directory_name": directory_name,
                    "datasets": [
                        entry["dataset"] for entry in safe_export["datasets"]
                    ],
                    "manual_action_count": len(manual_action_items(bundle)),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return destination
    if destination.exists():
        raise ReleaseBuildError(
            "staging destination already exists; choose a fresh output directory"
        )
    destination.mkdir(parents=True)
    inventory_lines = []
    if release.get("test_only") is True:
        sample_label = "synthetic samples"
    elif release_document_banner(bundle):
        sample_label = "draft release candidates"
    else:
        sample_label = "public samples"
    for entry in safe_export["datasets"]:
        dataset = entry["dataset"]
        display_name = DATASET_DISPLAY_NAMES.get(dataset, dataset)
        inventory_lines.append(
            f"- {display_name}: {entry['sample_count']} {sample_label}"
        )
        connectome_target = destination / release_connectome_relative_path(dataset)
        _copy_verified_file(
            entry["files"]["connectomes"],
            connectome_target,
            entry["file_sha256"]["connectomes"],
        )
        metadata_target = destination / release_metadata_relative_path(dataset)
        _copy_verified_file(
            entry["files"]["metadata"],
            metadata_target,
            entry["file_sha256"]["metadata"],
        )
        if "splits" in entry["files"]:
            split_target = destination / "splits" / dataset / "splits.tsv"
            _copy_verified_file(
                entry["files"]["splits"],
                split_target,
                entry["file_sha256"]["splits"],
            )
    data_root = destination / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    (data_root / "README.md").write_text(
        "# Public data\n\n"
        "Only dataset directories backed by explicit policy approvals are present. "
        "An absent directory means that participant-level files are not included.\n",
        encoding="utf-8",
    )
    split_root = destination / "splits"
    split_root.mkdir(parents=True, exist_ok=True)
    if not any(path.is_dir() for path in split_root.iterdir()):
        (split_root / "README.md").write_text(
            "# Exact split membership\n\n"
            "No exact participant-level split membership is included in this "
            "release. Consult the frozen dataset policy; absence never implies "
            "permission.\n",
            encoding="utf-8",
        )
    for entry in safe_export["aggregate_outputs"]:
        folder = {
            "aggregate_metrics": "aggregate_metrics",
            "statistical_summaries": "statistical_tests",
            "figure_source_data": "figure_source_data",
        }[entry["content_category"]]
        target = (
            destination
            / "benchmark_results"
            / folder
            / entry["dataset"]
            / entry["source"].name
        )
        _copy_verified_file(entry["source"], target, entry["sha256"])
    for folder in ("aggregate_metrics", "statistical_tests", "figure_source_data"):
        target = destination / "benchmark_results" / folder
        target.mkdir(parents=True, exist_ok=True)
        if not any(target.iterdir()):
            (target / "README.md").write_text(
                "# No approved outputs included\n\n"
                "This release contains no allowlisted files in this category.\n",
                encoding="utf-8",
            )

    actions = manual_action_items(bundle)
    included_names = {entry["dataset"] for entry in safe_export["datasets"]}
    policy_rows = [
        "| Dataset | Participant-level policy | Included |",
        "|---|---|---|",
    ]
    terms_rows = [
        "| Dataset | Original terms | Participant files in this release |",
        "|---|---|---|",
    ]
    for dataset in (
        "abide",
        "adni",
        "oasis3",
        "camcan",
        "cobre",
        "adnidod",
        "1000brains",
    ):
        policy = bundle["dataset_policy"].get("datasets", {}).get(dataset, {})
        policy_rows.append(
            f"| {dataset} | {policy.get('participant_level_release', 'forbidden')} "
            f"| {'yes' if dataset in included_names else 'no'} |"
        )
        terms_rows.append(
            f"| {dataset} | Governed by the original custodian; review required "
            f"| {'approved safe export only' if dataset in included_names else 'none'} |"
        )
    release_status = (
        "Synthetic test only"
        if release.get("test_only") is True
        else (
            "Publication-ready"
            if (
                not actions
                and release.get("version_confirmed") is True
                and release.get("publication_ready") is True
            )
            else "Real-data draft — pending review"
        )
    )
    commit, _ = _git_snapshot(Path(__file__).resolve().parents[2])
    replacements = {
        "PROJECT_TITLE": str(project["title"]),
        "PROJECT_VERSION": version,
        "VERSION": version,
        "ARCHIVE_BASENAME": directory_name,
        "VALIDATION_STATUS": release_status,
        "REPOSITORY_URL": str(project.get("repository_url", "")),
        "GIT_COMMIT": commit,
        "DATASET_RELEASE_POLICY_TABLE": "\n".join(policy_rows),
        "DATASET_INVENTORY": "\n".join(inventory_lines)
        or "- No participant-level datasets included.",
        "CONTENT_INVENTORY_SUMMARY": "\n".join(inventory_lines)
        or "- No participant-level datasets included.",
        "SOURCE_CODE_LICENSE_APPROVAL_STATUS": _release_license_value(
            bundle, "source_code"
        ),
        "DERIVED_DATA_LICENSE_IDENTIFIER": _release_license_value(
            bundle, "derived_data"
        ),
        "DERIVED_DATA_LICENSE_APPROVAL_STATUS": _release_license_status(
            bundle, "derived_data"
        ),
        "DOCUMENTATION_LICENSE_IDENTIFIER": _release_license_value(
            bundle, "documentation"
        ),
        "DOCUMENTATION_LICENSE_APPROVAL_STATUS": _release_license_status(
            bundle, "documentation"
        ),
        "DATASET_TERMS_TABLE": "\n".join(terms_rows),
        "ARTIFACT_LICENSE_MAPPING": "\n".join(
            [
                "| Artifact category | License/terms decision |",
                "|---|---|",
                "| Repository source code | "
                f"{_release_license_value(bundle, 'source_code')} |",
                "| Approved derived data | "
                f"{_release_license_value(bundle, 'derived_data')} |",
                "| Release documentation | "
                f"{_release_license_value(bundle, 'documentation')} |",
                "| Original source datasets | Original-custodian terms; not relicensed |",
            ]
        ),
        "MANUAL_ACTIONS": "\n".join(f"- {item}" for item in actions)
        or "- No unresolved actions recorded.",
    }
    for template_name, output_name in (
        ("README.md", "README.md"),
        ("DATASET_CARD.md", "DATASET_CARD.md"),
        ("LICENSES.md", "LICENSES.md"),
    ):
        rendered = (
            expected_licenses_text(bundle)
            if template_name == "LICENSES.md"
            else _render_template(template_name, replacements=replacements)
        )
        if template_name != "LICENSES.md":
            rendered = release_document_banner(bundle) + rendered
        (destination / output_name).write_text(
            rendered,
            encoding="utf-8",
        )
    (destination / "VERSION").write_text(version + "\n", encoding="utf-8")
    citation = build_citation_cff(bundle)
    if citation.get("authors"):
        write_citation_cff(bundle, destination / "CITATION.cff")
    _write_restricted_reconstruction(destination, bundle)
    _write_config_snapshots(destination, bundle, safe_export)
    _write_reproducibility(destination)
    _write_release_metadata_tables(destination, safe_export, bundle)
    _write_camcan_license(destination, bundle)
    _refresh_catalogs(destination, bundle)

    # Reports are written after the first catalog pass, then catalogs are
    # regenerated so the reports themselves are covered.
    from .validate_release import validate_release

    validation = validate_release(
        destination,
        config_path=config_path,
        publication_ready=False,
        write_reports=True,
    )
    _refresh_catalogs(destination, bundle)
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    for name in ("validation_report.json", "validation_report.md"):
        shutil.copyfile(
            destination / "manifests" / name,
            reports_dir / name,
        )
    if not validation.ok:
        raise ReleaseBuildError(
            f"generated staging failed validation ({len(validation.errors)} error(s))"
        )
    return destination


def build_release(
    config_path: str | Path,
    safe_export_dir: str | Path,
    output_dir: str | Path,
    *,
    dry_run: bool = False,
) -> Path:
    """Atomically create a release without exposing a partial staging tree."""

    requested_output = Path(output_dir)
    if dry_run:
        return _build_release_in_workspace(
            config_path,
            safe_export_dir,
            requested_output,
            dry_run=True,
        )
    if requested_output.is_symlink():
        raise ReleaseBuildError("output directory must not be a symbolic link")
    output_root = requested_output.resolve()
    safe_root = Path(safe_export_dir).resolve()
    if (
        output_root == safe_root
        or output_root.is_relative_to(safe_root)
        or safe_root.is_relative_to(output_root)
    ):
        raise ReleaseBuildError("output directory must not overlap the safe export")
    if output_root.exists():
        raise ReleaseBuildError(
            "output directory already exists; choose a fresh output directory"
        )

    parent = output_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    workspace = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.building-", dir=parent)
    )
    workspace_output = workspace / "output"
    try:
        workspace_release = _build_release_in_workspace(
            config_path,
            safe_export_dir,
            workspace_output,
            dry_run=False,
        )
        if output_root.exists():
            raise ReleaseBuildError(
                "output directory appeared during the build; refusing to overwrite it"
            )
        os.replace(workspace_output, output_root)
        return output_root / "staging" / workspace_release.name
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a Zenodo staging tree from a trusted safe export."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--safe-export-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        destination = build_release(
            args.config,
            args.safe_export_dir,
            args.output_dir,
            dry_run=args.dry_run,
        )
    except (ReleaseBuildError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"release build refused: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "staging_created": True,
                    "release_directory_name": destination.name,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
