"""Deterministic manifests and SHA-256 checksum helpers."""

from __future__ import annotations

import csv
import hashlib
import mimetypes
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

from .schemas import (
    AGGREGATE_CONTENT_CATEGORIES,
    CAMCAN_CONNECTOME_RELATIVE_PATH,
    CAMCAN_DATA_DICTIONARY_RELATIVE_PATH,
    CAMCAN_LICENSE_RELATIVE_PATH,
    CAMCAN_METADATA_RELATIVE_PATH,
    PARTICIPANT_CONTENT_CATEGORIES,
    canonical_dataset,
    dataset_policy,
    is_portable_relative_path,
    participant_release_decision,
)


MANIFEST_FIELDS = (
    "relative_path",
    "size_bytes",
    "sha256",
    "media_type",
    "dataset",
    "content_category",
    "access_category",
    "source_category",
    "release_policy_decision",
)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_regular_files(
    root: str | Path,
    *,
    exclude_relative: Iterable[str] = (),
) -> list[Path]:
    base = Path(root).resolve()
    excluded = {str(Path(value).as_posix()) for value in exclude_relative}
    files: list[Path] = []
    for path in base.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file() and path.relative_to(base).as_posix() not in excluded:
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(base).as_posix())


def _infer_dataset(relative: Path) -> str:
    for part in relative.parts:
        canonical = canonical_dataset(part)
        if canonical in {
            "abide",
            "adni",
            "oasis3",
            "camcan",
            "cobre",
            "adnidod",
            "1000brains",
        }:
            return canonical
    return "project"


def _infer_category(relative: Path) -> tuple[str, str, str]:
    parts = set(relative.parts)
    if relative == CAMCAN_CONNECTOME_RELATIVE_PATH:
        return "participant_connectomes", "public", "trusted_safe_export"
    if relative == CAMCAN_METADATA_RELATIVE_PATH:
        return "participant_metadata", "public", "trusted_safe_export"
    if relative in {
        CAMCAN_DATA_DICTIONARY_RELATIVE_PATH,
        CAMCAN_LICENSE_RELATIVE_PATH,
    }:
        return "documentation", "public", "release_builder"
    if (
        relative.as_posix() in {"data/README.md", "splits/README.md"}
        or (
            relative.name == "README.md"
            and "benchmark_results" in parts
        )
    ):
        return "documentation", "public", "release_builder"
    if "public_connectomes" in parts:
        return "participant_connectomes", "public", "trusted_safe_export"
    if "public_metadata" in parts:
        return "participant_metadata", "public", "trusted_safe_export"
    if "splits" in parts:
        return "exact_splits", "public", "trusted_safe_export"
    if "aggregate_metrics" in parts:
        return "aggregate_metrics", "public", "aggregate_output"
    if "statistical_tests" in parts:
        return "statistical_summaries", "public", "aggregate_output"
    if "figure_source_data" in parts:
        return "figure_source_data", "public", "aggregate_output"
    if "restricted_reconstruction" in parts:
        return "reconstruction_instructions", "documentation_only", "repository"
    if "configs" in parts:
        return "configuration", "public", "repository"
    if "reproducibility" in parts:
        return "configuration", "public", "repository"
    return "documentation", "public", "release_builder"


def _policy_decision(
    bundle: Mapping[str, Any] | None,
    dataset: str,
    category: str,
) -> str:
    if category in PARTICIPANT_CONTENT_CATEGORIES:
        if bundle is None:
            return "blocked:missing_policy_bundle"
        allowed, reason = participant_release_decision(bundle, dataset, category)
        return reason if allowed else f"blocked:{reason}"
    if category in AGGREGATE_CONTENT_CATEGORIES:
        if bundle is None:
            return "missing_policy_bundle"
        if dataset == "project":
            return "explicit_release_configuration_allowlist"
        policy = dataset_policy(bundle, dataset)
        decisions = policy.get("decisions", {}) if isinstance(policy, Mapping) else {}
        policy_key = {
            "configuration": "configuration_files",
            "processing_script": "processing_scripts",
        }.get(category, category)
        if not isinstance(decisions, Mapping) or policy_key not in decisions:
            return "blocked:missing_scope_policy"
        return str(decisions[policy_key])
    return "release_documentation"


def build_manifest_rows(
    root: str | Path,
    *,
    bundle: Mapping[str, Any] | None = None,
    exclude_relative: Iterable[str] = (),
) -> list[dict[str, str | int]]:
    base = Path(root).resolve()
    rows: list[dict[str, str | int]] = []
    for path in iter_regular_files(base, exclude_relative=exclude_relative):
        relative = path.relative_to(base)
        dataset = _infer_dataset(relative)
        category, access, source = _infer_category(relative)
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "media_type": media_type,
                "dataset": dataset,
                "content_category": category,
                "access_category": access,
                "source_category": source,
                "release_policy_decision": _policy_decision(
                    bundle, dataset, category
                ),
            }
        )
    return rows


def write_manifest(
    root: str | Path,
    output_path: str | Path,
    *,
    bundle: Mapping[str, Any] | None = None,
    exclude_relative: Iterable[str] = (),
) -> list[dict[str, str | int]]:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = build_manifest_rows(
        root, bundle=bundle, exclude_relative=exclude_relative
    )
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_sha256sums(
    root: str | Path,
    output_path: str | Path,
    *,
    exclude_relative: Iterable[str] = (),
) -> list[tuple[str, str]]:
    base = Path(root).resolve()
    destination = Path(output_path).resolve()
    excluded = {str(Path(value).as_posix()) for value in exclude_relative}
    if destination.is_relative_to(base):
        excluded.add(destination.relative_to(base).as_posix())
    rows: list[tuple[str, str]] = []
    for path in iter_regular_files(base, exclude_relative=excluded):
        relative = path.relative_to(base).as_posix()
        rows.append((sha256_file(path), relative))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for digest, relative in rows:
            handle.write(f"{digest}  {relative}\n")
    return rows


def read_sha256sums(path: str | Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            if "  " not in line:
                raise ValueError(f"invalid checksum line {line_number}")
            digest, relative = line.split("  ", 1)
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"invalid SHA-256 on line {line_number}")
            rows.append((digest, relative))
    return rows


def verify_sha256sums(
    root: str | Path, checksum_path: str | Path
) -> list[str]:
    base = Path(root).resolve()
    errors: list[str] = []
    try:
        rows = read_sha256sums(checksum_path)
    except (OSError, UnicodeError, ValueError):
        return ["checksum catalog is unreadable or malformed"]
    seen: set[str] = set()
    portable_seen: set[str] = set()
    for expected, relative in rows:
        normalized = Path(relative)
        posix = PurePosixPath(relative)
        windows = PureWindowsPath(relative)
        if (
            not is_portable_relative_path(relative)
            or normalized.is_absolute()
            or posix.is_absolute()
            or windows.is_absolute()
            or bool(windows.drive)
            or ".." in posix.parts
            or ".." in windows.parts
        ):
            errors.append("unsafe checksum path was rejected")
            continue
        if relative in seen:
            errors.append("duplicate checksum path was rejected")
            continue
        portable_key = relative.casefold()
        if portable_key in portable_seen:
            errors.append("checksum catalog contains a portable-path collision")
            continue
        seen.add(relative)
        portable_seen.add(portable_key)
        target = (base / normalized).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            errors.append("checksum catalog references a missing or unsafe target")
        elif sha256_file(target) != expected:
            errors.append("checksum mismatch detected")
    catalog = Path(checksum_path).resolve()
    excluded = (
        {catalog.relative_to(base).as_posix()}
        if catalog.is_relative_to(base)
        else set()
    )
    expected_paths = {
        path.relative_to(base).as_posix()
        for path in iter_regular_files(base, exclude_relative=excluded)
    }
    if seen != expected_paths:
        errors.append("checksum catalog does not exactly cover regular files")
    return errors


def verify_manifest(
    root: str | Path,
    manifest_path: str | Path,
    *,
    bundle: Mapping[str, Any] | None = None,
    excluded_relative: Iterable[str] = (),
) -> list[str]:
    base = Path(root).resolve()
    errors: list[str] = []
    try:
        with Path(manifest_path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t", strict=True)
            if tuple(reader.fieldnames or ()) != MANIFEST_FIELDS:
                return ["manifest header does not match the required schema"]
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return ["manifest is unreadable or malformed"]
    if any(None in row or any(row.get(field) is None for field in MANIFEST_FIELDS) for row in rows):
        return ["manifest contains a malformed row"]
    expected_paths = {
        path.relative_to(base).as_posix()
        for path in iter_regular_files(base, exclude_relative=excluded_relative)
    }
    row_paths = {row["relative_path"] for row in rows}
    if len(row_paths) != len(rows):
        errors.append("manifest contains duplicate paths")
    missing = sorted(expected_paths - row_paths)
    extra = sorted(row_paths - expected_paths)
    if missing:
        errors.append(f"manifest is missing {len(missing)} file(s)")
    if extra:
        errors.append(f"manifest references {len(extra)} unexpected file(s)")
    for row in rows:
        relative = Path(row["relative_path"])
        windows_relative = PureWindowsPath(row["relative_path"])
        target = (base / relative).resolve()
        if (
            not is_portable_relative_path(row["relative_path"])
            or relative.is_absolute()
            or windows_relative.is_absolute()
            or bool(windows_relative.drive)
            or ".." in relative.parts
            or ".." in windows_relative.parts
            or not target.is_relative_to(base)
        ):
            errors.append("manifest contains an unsafe path")
            continue
        if not target.is_file():
            continue
        if str(target.stat().st_size) != row["size_bytes"]:
            errors.append("manifest contains a file-size mismatch")
        if sha256_file(target) != row["sha256"]:
            errors.append("manifest checksum mismatch detected")
    expected_rows = build_manifest_rows(
        base,
        bundle=bundle,
        exclude_relative=excluded_relative,
    )
    expected_by_path = {
        str(row["relative_path"]): row for row in expected_rows
    }
    for row in rows:
        expected = expected_by_path.get(row["relative_path"])
        if expected is None:
            continue
        for field in MANIFEST_FIELDS:
            if str(row[field]) != str(expected[field]):
                errors.append(
                    f"manifest semantic field mismatch ({field})"
                )
    return errors
