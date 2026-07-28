"""Trusted-environment export from internal pickle inputs to safe formats.

This is the *only* release module allowed to deserialize pickle.  Pickle is an
executable format and must never be treated as untrusted input.  The command
requires an explicit flag, a trust attestation, a pinned SHA-256 digest, a
known input schema, and completed release-policy approvals.  It is intentionally
not called by the public release builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import uuid
import zipfile
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .checksums import sha256_file
from .privacy_scan import scan_release
from .schemas import (
    KNOWN_DATASETS,
    SCHEMA_VERSION,
    canonical_dataset,
    load_release_bundle,
    load_yaml,
    participant_release_decision,
    validate_connectome_npz,
    validate_metadata_columns,
    validate_release_numeric_settings,
)


ALLOWED_MANIFEST_KEYS = {
    "schema_version",
    "export_namespace",
    "trust_attestation",
    "datasets",
}
ALLOWED_DATASET_KEYS = {
    "dataset",
    "input_path",
    "sha256",
    "input_format",
    "schema",
    "atlas",
    "expected_regions",
    "source_identity_attestation",
}
ALLOWED_SCHEMA_KEYS = {
    "connectome_column",
    "timeseries_column",
    "metadata_columns",
    "fold_column",
    "partition_column",
    "matrix_type",
    "ignored_internal_columns",
}
ALLOWED_INPUT_FORMATS = {"pandas_dataframe"}
ALLOWED_PARTITIONS = {"train", "validation", "test"}
ATLAS_REGIONS = {"schaefer_100": 100, "msdl_39": 39}
ALLOWED_TRUST_ATTESTATION_KEYS = {
    "attested_by",
    "attested_on",
    "evidence",
    "source_controlled",
    "checksums_verified",
}
ALLOWED_SOURCE_IDENTITY_KEYS = {"dataset", "approved_by", "evidence"}


class TrustedExportError(RuntimeError):
    """A fail-closed trusted-export error with no sensitive value payload."""


def _strict_keys(
    mapping: Mapping[str, Any], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise TrustedExportError(
            f"{label} contains unrecognized fields ({len(unknown)} blocked)"
        )


def _validate_trust_attestation(manifest: Mapping[str, Any]) -> None:
    attestation = manifest.get("trust_attestation")
    if not isinstance(attestation, Mapping):
        raise TrustedExportError("input manifest requires a trust_attestation")
    _strict_keys(
        attestation,
        ALLOWED_TRUST_ATTESTATION_KEYS,
        "trust_attestation",
    )
    if set(attestation) != ALLOWED_TRUST_ATTESTATION_KEYS:
        raise TrustedExportError("trust_attestation has missing fields")
    required_text = ("attested_by", "attested_on", "evidence")
    if not all(
        isinstance(attestation.get(key), str)
        and attestation[key].strip()
        for key in required_text
    ):
        raise TrustedExportError("trust_attestation is incomplete")
    try:
        date.fromisoformat(attestation["attested_on"].strip())
    except ValueError as exc:
        raise TrustedExportError("trust_attestation date must use ISO format") from exc
    if attestation.get("source_controlled") is not True:
        raise TrustedExportError("source_controlled must be explicitly true")
    if attestation.get("checksums_verified") is not True:
        raise TrustedExportError("checksums_verified must be explicitly true")


def _load_internal_manifest(path: str | Path) -> dict[str, Any]:
    manifest = load_yaml(path)
    _strict_keys(manifest, ALLOWED_MANIFEST_KEYS, "input manifest")
    version = manifest.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SCHEMA_VERSION:
        raise TrustedExportError("unsupported internal-manifest schema_version")
    _validate_trust_attestation(manifest)
    try:
        uuid.UUID(str(manifest.get("export_namespace", "")))
    except ValueError as exc:
        raise TrustedExportError(
            "export_namespace must be a deliberately generated UUID"
        ) from exc
    datasets = manifest.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise TrustedExportError("input manifest requires a non-empty datasets list")
    seen: set[str] = set()
    for entry in datasets:
        if not isinstance(entry, Mapping):
            raise TrustedExportError("each dataset entry must be a mapping")
        _strict_keys(entry, ALLOWED_DATASET_KEYS, "dataset entry")
        if set(entry) != ALLOWED_DATASET_KEYS:
            raise TrustedExportError("dataset entry has missing required fields")
        dataset = canonical_dataset(entry.get("dataset"))
        if dataset not in KNOWN_DATASETS:
            raise TrustedExportError(
                "input manifest contains an unrecognized dataset label"
            )
        if dataset in seen:
            raise TrustedExportError("input manifest contains a duplicate dataset")
        seen.add(dataset)
        if entry.get("input_format") not in ALLOWED_INPUT_FORMATS:
            raise TrustedExportError("input manifest declares an unknown input schema")
        schema = entry.get("schema")
        if not isinstance(schema, Mapping):
            raise TrustedExportError("each dataset entry requires a schema mapping")
        _strict_keys(schema, ALLOWED_SCHEMA_KEYS, "dataset schema")
        for key in (
            "connectome_column",
            "timeseries_column",
            "fold_column",
            "partition_column",
        ):
            value = schema.get(key)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise TrustedExportError(
                    "operational source columns must be non-empty strings"
                )
        matrix_sources = [
            key
            for key in ("connectome_column", "timeseries_column")
            if isinstance(schema.get(key), str) and schema[key].strip()
        ]
        if len(matrix_sources) != 1:
            raise TrustedExportError(
                "schema must define exactly one connectome or time-series column"
            )
        has_fold = bool(schema.get("fold_column"))
        has_partition = bool(schema.get("partition_column"))
        if has_fold != has_partition:
            raise TrustedExportError(
                "fold_column and partition_column must be declared together"
            )
        if schema.get("matrix_type") not in {"correlation", "covariance", "spd"}:
            raise TrustedExportError(
                "matrix_type must be correlation, covariance, or spd"
            )
        if (
            schema.get("timeseries_column")
            and schema.get("matrix_type") != "correlation"
        ):
            raise TrustedExportError(
                "time-series export currently supports only correlation matrices"
            )
        metadata_mapping = schema.get("metadata_columns")
        if not isinstance(metadata_mapping, Mapping):
            raise TrustedExportError(
                "metadata_columns must map public names to source columns"
            )
        if any(
            not isinstance(public_name, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", public_name)
            or public_name == "sample_uid"
            or not isinstance(source_name, str)
            or not source_name.strip()
            for public_name, source_name in metadata_mapping.items()
        ):
            raise TrustedExportError(
                "metadata_columns contains an unsafe name, source, or reserved sample_uid"
            )
        operational_columns = [
            str(schema[matrix_sources[0]]),
            *[str(value) for value in metadata_mapping.values()],
            *[
                str(schema[key])
                for key in ("fold_column", "partition_column")
                if schema.get(key)
            ],
        ]
        if (
            any(not value.strip() for value in operational_columns)
            or len(operational_columns) != len(set(operational_columns))
        ):
            raise TrustedExportError(
                "exported and operational source columns must be non-empty and distinct"
            )
        ignored = schema.get("ignored_internal_columns", [])
        if (
            not isinstance(ignored, list)
            or any(not isinstance(value, str) or not value.strip() for value in ignored)
            or len(ignored) != len(set(ignored))
            or set(ignored) & set(operational_columns)
        ):
            raise TrustedExportError(
                "ignored_internal_columns must be unique and separate from used columns"
            )
        digest = str(entry.get("sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise TrustedExportError("every trusted input requires a lowercase SHA-256")
        atlas = str(entry.get("atlas", ""))
        expected_regions = entry.get("expected_regions")
        if atlas not in ATLAS_REGIONS or expected_regions != ATLAS_REGIONS[atlas]:
            raise TrustedExportError(
                "atlas and expected_regions must match a supported atlas schema"
            )
        identity = entry.get("source_identity_attestation")
        if (
            not isinstance(identity, Mapping)
            or set(identity) != ALLOWED_SOURCE_IDENTITY_KEYS
            or canonical_dataset(identity.get("dataset")) != dataset
            or not isinstance(identity.get("dataset"), str)
            or not isinstance(identity.get("approved_by"), str)
            or not identity["approved_by"].strip()
            or not isinstance(identity.get("evidence"), str)
            or not identity["evidence"].strip()
        ):
            raise TrustedExportError(
                "source_identity_attestation must bind the declared dataset"
            )
    return manifest


def _source_binding(
    entry: Mapping[str, Any], export_namespace: str | uuid.UUID
) -> str:
    payload = {
        "dataset": canonical_dataset(entry["dataset"]),
        "export_namespace": str(uuid.UUID(str(export_namespace))),
        "input_sha256": str(entry["sha256"]),
        "input_format": str(entry["input_format"]),
        "schema": entry["schema"],
        "atlas": str(entry["atlas"]),
        "expected_regions": int(entry["expected_regions"]),
        "source_identity_attestation": entry["source_identity_attestation"],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _approval_matches_source(
    bundle: Mapping[str, Any],
    entry: Mapping[str, Any],
    export_namespace: str | uuid.UUID,
) -> bool:
    dataset = canonical_dataset(entry["dataset"])
    approvals = bundle["manual_approvals"].get("approvals", {})
    if isinstance(approvals, Mapping) and isinstance(
        approvals.get("datasets"), Mapping
    ):
        approvals = approvals["datasets"]
    approval = approvals.get(dataset, {}) if isinstance(approvals, Mapping) else {}
    return (
        isinstance(approval, Mapping)
        and canonical_dataset(approval.get("dataset")) == dataset
        and approval.get("atlas") == entry["atlas"]
        and approval.get("n_regions") == entry["expected_regions"]
        and approval.get("source_binding_sha256")
        == _source_binding(entry, export_namespace)
    )


def _safe_uid(namespace: uuid.UUID, dataset: str, row_index: int) -> str:
    # Deliberately derived from export order, not from an original identifier.
    return "s" + uuid.uuid5(namespace, f"{dataset}:row:{row_index}").hex


def _serialize_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise TrustedExportError("approved metadata contains a non-finite value")
        return format(float(value), ".12g")
    if isinstance(value, (int, np.integer, bool, np.bool_)):
        return str(value)
    text = str(value).strip()
    if "\x00" in text or "\n" in text or "\r" in text:
        raise TrustedExportError("approved metadata contains unsafe control characters")
    return text


def _write_tsv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _serialize_cell(row.get(key)) for key in fieldnames})


def _write_deterministic_connectome_npz(path: Path, connectomes: np.ndarray) -> None:
    payload = io.BytesIO()
    np.lib.format.write_array(
        payload,
        np.asarray(connectomes),
        allow_pickle=False,
    )
    info = zipfile.ZipInfo("connectomes.npy", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with zipfile.ZipFile(
        path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        archive.writestr(info, payload.getvalue(), compresslevel=9)


def _load_trusted_dataframe(path: Path, expected_sha256: str):
    """Hash and deserialize one pinned regular file through the same descriptor."""

    import pandas as pd

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TrustedExportError("trusted pickle could not be opened safely") from exc
    with os.fdopen(descriptor, "rb") as handle, tempfile.TemporaryFile(
        mode="w+b"
    ) as frozen:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise TrustedExportError("trusted pickle is not a regular file")
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            frozen.write(chunk)
        after = os.fstat(handle.fileno())
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise TrustedExportError("trusted pickle changed while it was verified")
        if digest.hexdigest() != expected_sha256:
            raise TrustedExportError("trusted-input checksum mismatch")
        frozen.seek(0)
        value = pd.read_pickle(frozen)
    if not isinstance(value, pd.DataFrame):
        raise TrustedExportError("trusted pickle does not match pandas_dataframe schema")
    return value


def _preflight_trusted_input(entry: Mapping[str, Any]) -> None:
    """Verify path, type, stability, and checksum without deserializing pickle."""

    input_path = Path(str(entry["input_path"])).expanduser()
    if (
        not input_path.is_absolute()
        or input_path.suffix.lower() not in {".pkl", ".pickle"}
        or input_path.is_symlink()
        or not input_path.is_file()
    ):
        raise TrustedExportError(
            "trusted input must be an absolute regular pickle file"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(input_path, flags)
    except OSError as exc:
        raise TrustedExportError("trusted input could not be opened safely") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise TrustedExportError("trusted input is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise TrustedExportError("trusted input changed during dry-run verification")
    finally:
        os.close(descriptor)
    if digest.hexdigest() != str(entry["sha256"]):
        raise TrustedExportError("trusted-input checksum mismatch")


def _extract_connectomes(frame, schema: Mapping[str, Any]) -> np.ndarray:
    if schema.get("connectome_column"):
        column = str(schema["connectome_column"])
        if column not in frame.columns:
            raise TrustedExportError("declared connectome column is missing")
        matrices = [np.asarray(value) for value in frame[column].tolist()]
        try:
            connectomes = np.stack(matrices, axis=0)
        except ValueError as exc:
            raise TrustedExportError(
                "declared connectome column has inconsistent shapes"
            ) from exc
    else:
        column = str(schema["timeseries_column"])
        if column not in frame.columns:
            raise TrustedExportError("declared time-series column is missing")
        timeseries = [np.asarray(value) for value in frame[column].tolist()]
        for array in timeseries:
            if (
                array.ndim != 2
                or min(array.shape) < 2
                or not np.issubdtype(array.dtype, np.floating)
                or not np.isfinite(array).all()
            ):
                raise TrustedExportError(
                    "declared time-series column violates the numeric 2D schema"
                )
        from spd_connectome_benchmark.connectomes import estimate_connectome_matrices

        connectomes = estimate_connectome_matrices(
            timeseries, normalize=True, n_jobs=1
        )
    if not np.issubdtype(connectomes.dtype, np.floating):
        raise TrustedExportError("connectomes must have a floating-point dtype")
    return connectomes.astype(np.float32, copy=False)


def _export_dataset(
    entry: Mapping[str, Any],
    *,
    namespace: uuid.UUID,
    bundle: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    dataset = canonical_dataset(entry["dataset"])
    for scope in (
        "participant_connectomes",
        "participant_metadata",
    ):
        allowed, reason = participant_release_decision(bundle, dataset, scope)
        if not allowed:
            raise TrustedExportError(
                f"{dataset} {scope} export blocked by policy: {reason}"
            )
    schema = entry["schema"]
    has_splits = bool(schema.get("partition_column"))
    if has_splits:
        allowed, reason = participant_release_decision(
            bundle, dataset, "exact_splits"
        )
        if not allowed:
            raise TrustedExportError(
                f"{dataset} exact-split export blocked by policy: {reason}"
            )
    if not _approval_matches_source(bundle, entry, namespace):
        raise TrustedExportError(
            f"{dataset} approval is not bound to source, atlas, and region count"
        )

    input_path = Path(str(entry["input_path"])).expanduser()
    if (
        not input_path.is_absolute()
        or input_path.suffix.lower() not in {".pkl", ".pickle"}
        or input_path.is_symlink()
        or not input_path.is_file()
    ):
        raise TrustedExportError(
            f"{dataset} trusted input must be an absolute regular pickle file"
        )
    frame = _load_trusted_dataframe(input_path, str(entry["sha256"]))
    if frame.empty:
        raise TrustedExportError(f"{dataset} trusted input is empty")
    metadata_mapping = schema.get("metadata_columns", {})
    if not isinstance(metadata_mapping, Mapping):
        raise TrustedExportError("metadata_columns must map public names to source columns")
    used_columns = {
        str(schema.get("connectome_column") or schema.get("timeseries_column")),
        *[str(value) for value in metadata_mapping.values()],
        *[
            str(schema[key])
            for key in ("fold_column", "partition_column")
            if schema.get(key)
        ],
    }
    ignored_columns = {
        str(value) for value in schema.get("ignored_internal_columns", [])
    }
    if used_columns & ignored_columns:
        raise TrustedExportError(
            "ignored_internal_columns must not overlap exported or operational columns"
        )
    declared_columns = used_columns | ignored_columns
    actual_columns = {str(value) for value in frame.columns}
    if actual_columns != declared_columns:
        raise TrustedExportError(
            "trusted input has missing or unrecognized columns; "
            "declare every intentionally ignored internal column"
        )
    connectomes = _extract_connectomes(frame, schema)
    if len(frame) != len(connectomes):
        raise TrustedExportError(f"{dataset} sample-count mismatch")
    if connectomes.shape[1:] != (
        int(entry["expected_regions"]),
        int(entry["expected_regions"]),
    ):
        raise TrustedExportError(f"{dataset} atlas region count does not match attestation")

    public_columns = ["sample_uid", *[str(key) for key in metadata_mapping]]
    column_findings = validate_metadata_columns(
        public_columns, bundle=bundle, dataset=dataset
    )
    if column_findings:
        raise TrustedExportError(column_findings[0].message)
    source_columns = [str(value) for value in metadata_mapping.values()]
    if len(source_columns) != len(set(source_columns)):
        raise TrustedExportError("metadata source columns must be unique")
    if any(column not in frame.columns for column in source_columns):
        raise TrustedExportError("one or more declared metadata columns are missing")

    sample_uids = [
        _safe_uid(namespace, dataset, row_index) for row_index in range(len(frame))
    ]
    metadata_rows: list[dict[str, object]] = []
    for row_position, (_, row) in enumerate(frame.iterrows()):
        public_row: dict[str, object] = {"sample_uid": sample_uids[row_position]}
        for public_name, source_name in metadata_mapping.items():
            public_row[str(public_name)] = row[str(source_name)]
        metadata_rows.append(public_row)

    dataset_dir = output_dir / "datasets" / dataset
    dataset_dir.mkdir(parents=True, exist_ok=False)
    connectome_path = dataset_dir / "connectomes.npz"
    _write_deterministic_connectome_npz(connectome_path, connectomes)
    metadata_path = dataset_dir / "metadata.tsv"
    _write_tsv(metadata_path, public_columns, metadata_rows)
    files: dict[str, str] = {
        "connectomes": connectome_path.relative_to(output_dir).as_posix(),
        "metadata": metadata_path.relative_to(output_dir).as_posix(),
    }

    if has_splits:
        partition_column = str(schema["partition_column"])
        fold_column = str(schema.get("fold_column", "")).strip()
        required = {partition_column}
        if fold_column:
            required.add(fold_column)
        if any(column not in frame.columns for column in required):
            raise TrustedExportError("one or more declared split columns are missing")
        split_rows: list[dict[str, object]] = []
        for row_position, (_, row) in enumerate(frame.iterrows()):
            partition = str(row[partition_column]).strip().lower()
            if partition not in ALLOWED_PARTITIONS:
                raise TrustedExportError("split partition is not train/validation/test")
            fold = str(row[fold_column]).strip() if fold_column else "fold_1"
            if not fold or any(char in fold for char in "\r\n\t/\\"):
                raise TrustedExportError("split fold label is unsafe")
            split_rows.append(
                {
                    "fold": fold,
                    "partition": partition,
                    "sample_uid": sample_uids[row_position],
                }
            )
        split_path = dataset_dir / "splits.tsv"
        _write_tsv(
            split_path, ["fold", "partition", "sample_uid"], split_rows
        )
        files["splits"] = split_path.relative_to(output_dir).as_posix()

    numeric_settings = validate_release_numeric_settings(bundle["config"])
    count, findings = validate_connectome_npz(
        connectome_path,
        matrix_type=str(schema.get("matrix_type", "correlation")),
        symmetry_tolerance=numeric_settings["symmetry_tolerance"],
        diagonal_tolerance=numeric_settings["diagonal_tolerance"],
        spd_eigenvalue_tolerance=numeric_settings["spd_eigenvalue_tolerance"],
        expected_regions=int(entry["expected_regions"]),
    )
    if findings or count != len(metadata_rows):
        raise TrustedExportError(f"{dataset} safe-export validation failed")
    return {
        "dataset": dataset,
        "sample_count": len(metadata_rows),
        "matrix_type": str(schema.get("matrix_type", "correlation")),
        "atlas": str(entry["atlas"]),
        "n_regions": int(entry["expected_regions"]),
        "source_binding_sha256": _source_binding(entry, namespace),
        "files": {
            name: {"path": relative, "sha256": sha256_file(output_dir / relative)}
            for name, relative in files.items()
        },
    }


def export_internal(
    config_path: str | Path,
    input_manifest: str | Path,
    output_dir: str | Path,
    *,
    trusted_internal_input: bool,
    dry_run: bool = False,
) -> Path:
    """Execute the trusted export or return its dry-run plan."""

    if not trusted_internal_input:
        raise TrustedExportError(
            "refusing pickle input without --trusted-internal-input"
        )
    bundle = load_release_bundle(config_path)
    manifest = _load_internal_manifest(input_manifest)
    supplied_destination = Path(output_dir).expanduser()
    if supplied_destination.is_symlink():
        raise TrustedExportError("safe-export output must not be a symbolic link")
    destination = supplied_destination.resolve()
    input_manifest_path = Path(input_manifest).expanduser().resolve()
    input_paths = [
        Path(str(entry["input_path"])).expanduser().resolve()
        for entry in manifest["datasets"]
    ]
    repository_root = Path(__file__).resolve().parents[2]
    if (
        destination.is_relative_to(repository_root)
        or input_manifest_path.is_relative_to(repository_root)
        or any(path.is_relative_to(repository_root) for path in input_paths)
    ):
        raise TrustedExportError(
            "trusted inputs and safe-export output must remain outside the repository"
        )
    if any(
        destination == path
        or destination.is_relative_to(path)
        or path.is_relative_to(destination)
        for path in [input_manifest_path, *input_paths]
    ):
        raise TrustedExportError("safe-export directory overlaps trusted input")
    if destination.exists():
        raise TrustedExportError("safe-export directory must be absent")

    planned = []
    export_namespace = str(manifest["export_namespace"])
    for entry in manifest["datasets"]:
        dataset = canonical_dataset(entry["dataset"])
        schema = entry["schema"]
        scopes = ["participant_connectomes", "participant_metadata"]
        if schema.get("partition_column"):
            scopes.append("exact_splits")
        decisions = {
            scope: participant_release_decision(bundle, dataset, scope)
            for scope in scopes
        }
        _preflight_trusted_input(entry)
        source_binding_approved = _approval_matches_source(
            bundle, entry, export_namespace
        )
        planned.append(
            {
                "dataset": dataset,
                "source_binding_sha256": _source_binding(
                    entry, export_namespace
                ),
                "source_binding_approved": source_binding_approved,
                "scopes": {
                    scope: {"allowed": allowed, "reason": reason}
                    for scope, (allowed, reason) in decisions.items()
                },
            }
        )
    blocked = [
        item["dataset"]
        for item in planned
        if (
            not item["source_binding_approved"]
            or not all(decision["allowed"] for decision in item["scopes"].values())
        )
    ]
    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "would_deserialize_pickle": False,
                    "would_succeed": not blocked,
                    "dataset_count": len(planned),
                    "plan": planned,
                },
                indent=2,
                sort_keys=True,
            )
        )
        if blocked:
            raise TrustedExportError(
                "release policy or source binding blocks trusted export for: "
                + ", ".join(sorted(blocked))
            )
        return destination
    if blocked:
        raise TrustedExportError(
            "release policy or source binding blocks trusted export for: "
            + ", ".join(sorted(blocked))
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building-",
            dir=destination.parent,
        )
    )
    try:
        marker = staging / ".safe-export-root.json"
        marker.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "trusted_safe_export",
                    "producer": "tools.zenodo.export_internal",
                    "contains_pickle": False,
                    "contains_original_identifiers": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        namespace = uuid.UUID(export_namespace)
        exported = [
            _export_dataset(
                entry, namespace=namespace, bundle=bundle, output_dir=staging
            )
            for entry in manifest["datasets"]
        ]
        public_manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "trusted_safe_export",
            "producer": "tools.zenodo.export_internal",
            "datasets": exported,
            "aggregate_outputs": [],
        }
        manifest_path = staging / "export_manifest.json"
        manifest_path.write_text(
            json.dumps(public_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        findings = scan_release(
            staging,
            bundle.get("forbidden_patterns_path"),
            release_config=bundle["config"],
        )
        if findings:
            raise TrustedExportError(
                f"safe export blocked by privacy scan ({len(findings)} finding(s))"
            )
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert attested trusted pickle input into a safe export."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--trusted-internal-input",
        action="store_true",
        help="Acknowledge that pickle may execute code and the input is controlled.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        "WARNING: trusted internal pickle input may execute code. "
        "Use only in a controlled environment with an attested checksum.",
        file=sys.stderr,
    )
    try:
        destination = export_internal(
            args.config,
            args.input_manifest,
            args.output_dir,
            trusted_internal_input=args.trusted_internal_input,
            dry_run=args.dry_run,
        )
    except (TrustedExportError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"trusted export refused: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "safe_export_created": True,
                    "output_name": destination.name,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
