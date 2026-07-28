"""Shared schemas and fail-closed validation primitives.

The public build never imports or calls a pickle loader.  The only module that
may deserialize a trusted internal pickle is :mod:`export_internal`, which is a
separate, explicitly gated command.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import struct
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable, Mapping

import numpy as np
import yaml


SCHEMA_VERSION = 1
KNOWN_DATASETS = frozenset(
    {"abide", "adni", "oasis3", "camcan", "cobre", "adnidod", "1000brains"}
)
RESTRICTED_DATASETS = frozenset({"adni", "adnidod", "oasis3", "1000brains"})
CONFIRMED_PARTICIPANT_RELEASE_DATASETS = frozenset(
    {"abide", "camcan", "cobre"}
)
CONFIRMED_PARTICIPANT_AUTHORIZATIONS: dict[str, dict[str, Any]] = {
    "abide": {
        "status": "allowed",
        "license": {"type": "ABIDE_source_license"},
        "confirmation": {"available": True},
        "allowed_metadata": [],
        "prohibited": [
            "raw_images",
            "T1w_images",
            "original_source_paths",
            "original_identifiers",
            "unrestricted_clinical_text",
            "arbitrary_behavioural_variables",
            "questionnaire_responses",
        ],
        "required_conditions": [
            "attribution",
            "non_commercial_use",
            "share_alike_if_required",
        ],
        "required_citation": ["ABIDE_reference", "INDI_reference"],
    },
    "camcan": {
        "status": "allowed",
        "license": {"type": "CC-BY-4.0"},
        "confirmation": {"available": True},
        "allowed_metadata": ["age", "sex"],
        "prohibited": [
            "raw_images",
            "raw_T1_images",
            "T1w_images",
            "identifiable_images",
            "home_interview_variables",
            "identifiable_behavioural_variables",
            "CCID",
        ],
        "required_conditions": [
            "attribution",
            "exclude_identifiable_material",
            "limit_metadata_to_age_and_sex",
        ],
        "required_citation": ["Shafto_et_al_CamCAN_cohort_paper"],
    },
    "cobre": {
        "status": "allowed",
        "license": {"type": "COBRE_confirmed_license"},
        "confirmation": {"available": True},
        "allowed_metadata": [],
        "prohibited": [
            "raw_images",
            "T1w_images",
            "original_source_paths",
            "original_identifiers",
            "unrestricted_clinical_text",
            "arbitrary_behavioural_variables",
            "questionnaire_responses",
        ],
        "required_conditions": [
            "confirmed_cobre_fcp_redistribution_terms",
            "attribution",
            "non_commercial_use",
        ],
        "required_citation": ["COBRE_reference", "FCP_INDI_reference"],
    },
}
CONFIRMED_DATASET_RIGHTS_STATEMENTS = {
    "camcan": (
        "Approved derived-data release of participant-level "
        "functional-connectivity matrices under CC-BY-4.0; raw or "
        "identifiable images, T1-weighted images, Home Interview variables, "
        "and identifiable behavioural variables are excluded."
    ),
    "abide": (
        "Approved derived-data release of participant-level "
        "functional-connectivity matrices under the confirmed ABIDE "
        "source-data terms, retaining attribution, non-commercial-use, and "
        "applicable share-alike conditions."
    ),
    "cobre": (
        "Approved derived-data release of participant-level "
        "functional-connectivity matrices under the confirmed COBRE/FCP "
        "redistribution terms, including attribution and non-commercial-use "
        "conditions."
    ),
}
CONFIRMED_DATASET_RIGHTS = {
    dataset: {
        "status": "approved_derived_data_release",
        "license": authorization["license"]["type"],
        "conditions": list(authorization["required_conditions"]),
        "required_citations": list(authorization["required_citation"]),
        "rights_statement": CONFIRMED_DATASET_RIGHTS_STATEMENTS[dataset],
    }
    for dataset, authorization in CONFIRMED_PARTICIPANT_AUTHORIZATIONS.items()
}
PARTICIPANT_CONTENT_CATEGORIES = frozenset(
    {"participant_connectomes", "participant_metadata", "exact_splits"}
)
PARTICIPANT_POLICY_KEYS = {
    "participant_connectomes": "participant_level_connectomes",
    "participant_metadata": "participant_level_metadata",
    "exact_splits": "exact_split_membership",
}
AGGREGATE_CONTENT_CATEGORIES = frozenset(
    {
        "aggregate_metrics",
        "statistical_summaries",
        "figure_source_data",
        "configuration",
        "processing_script",
        "reconstruction_instructions",
    }
)
FORBIDDEN_METADATA_COLUMNS = frozenset(
    {
        "ccid",
        "participant_id",
        "subject_id",
        "scan_id",
        "session_id",
        "visit_id",
        "image_id",
        "original_filename",
        "file_path",
        "source_path",
        "acquisition_date",
        "birth_date",
        "date_of_birth",
        "clinical_notes",
        "clinical_text",
        "unrestricted_clinical_text",
        "free_text",
        "open_text",
        "comments",
        "home_interview",
        "home_interview_variables",
        "questionnaire",
        "questionnaire_response",
        "questionnaire_responses",
        "survey_item",
        "survey_response",
        "behavioral_measure",
        "behavioral_variables",
        "behavioural_measure",
        "behavioural_variables",
        "input_path",
        "local_path",
        "raw_filename",
        "source_filename",
    }
)
# Stage 1 emits UUIDv5 values as ``s`` plus 32 lower-case hexadecimal digits.
# Stage 2 requires that exact shape so a source identifier such as CamCAN CCID
# cannot be passed through merely by renaming its column to sample_uid.
SAFE_UID_RE = re.compile(r"^s[0-9a-f]{32}$")
MATRIX_TYPES = frozenset({"correlation", "covariance", "spd"})
ATLAS_REGION_COUNTS = {
    "schaefer_100": 100,
    "msdl_39": 39,
}
CAMCAN_CONNECTOME_RELATIVE_PATH = Path(
    "data/camcan/connectomes/camcan_schaefer100_fc.npz"
)
CAMCAN_METADATA_RELATIVE_PATH = Path(
    "data/camcan/metadata/participants.tsv"
)
CAMCAN_DATA_DICTIONARY_RELATIVE_PATH = Path(
    "data/camcan/data_dictionary.tsv"
)
CAMCAN_LICENSE_RELATIVE_PATH = Path("data/camcan/LICENSE.txt")
CANONICAL_REPOSITORY_URL = (
    "https://github.com/GeometricBCI/"
    "rsfmri-spd-connectome-external-generalization-benchmark"
)
_ATLAS_ALIASES = {
    "schaefer100": "schaefer_100",
    "schaefer-100": "schaefer_100",
    "msdl": "msdl_39",
    "msdl39": "msdl_39",
    "msdl-39": "msdl_39",
}
_TOLERANCE_LIMITS = {
    "symmetry_tolerance": (0.0, 1.0e-3),
    "diagonal_tolerance": (0.0, 1.0e-2),
    "spd_eigenvalue_tolerance": (0.0, 1.0e-3),
}
_CANONICAL_RELEASE_SAFETY_SETTINGS: dict[str, Any] = {
    "archive_format": "zip",
    "positive_definite_required": True,
    "numeric_dtype_allowlist": ["float32", "float64"],
    "diagonal_expected": 1.0,
    "normalized_archive_timestamp": "1980-01-01T00:00:00Z",
    "publication_gate": "manual_required",
}
_CANONICAL_SAFE_EXPORT_CONTRACT: dict[str, Any] = {
    "input_must_be_designated_directory": True,
    "input_manifest_required": True,
    "unknown_files": "reject",
    "unknown_fields": "reject",
    "symbolic_links": "reject",
    "copy_original_filenames": False,
    "copy_source_paths": False,
    "hashed_original_identifiers": "forbidden",
    "sample_uid_generation": (
        "only_when_confirmed_dataset_policy_and_artifact_binding_permit"
    ),
    "npz_allow_pickle": False,
    "finite_numeric_arrays_required": True,
}
_CANONICAL_PUBLICATION_REQUIREMENTS: dict[str, Any] = {
    "require_complete_manual_approvals": True,
    "require_complete_required_metadata": True,
    "require_dataset_specific_rights_statements": True,
    "require_mixed_record_license_review": True,
    "require_explicit_documentation_license": True,
    "require_clean_validation_report": True,
    "require_manifest_match": True,
    "require_checksum_match": True,
    "require_revalidated_archive": True,
    "allow_zenodo_api": False,
    "allow_token_access": False,
    "allow_doi_reservation": False,
}
FORBIDDEN_METADATA_COLUMN_RE = re.compile(
    r"^(?:"
    r"(?:participant|subject|patient|person|scan|session|visit|image)_?"
    r"(?:id|identifier|key|uid)"
    r"|(?:id|identifier|key|uid)_?"
    r"(?:participant|subject|patient|person|scan|session|visit|image)"
    r"|(?:record|research|medical)_?id"
    r"|(?:eid|rid|mrn)"
    r"|original_?file_?name"
    r"|file_?path"
    r"|source_?path"
    r"|acquisition_?date"
    r"|birth_?date"
    r"|date_?of_?birth"
    r"|clinical_?notes?"
    r"|(?:free|open|clinical)_?text"
    r"|comments?"
    r"|home_?interview(?:_.*)?"
    r"|(?:questionnaire|survey)(?:_.*)?"
    r"|behaviou?r(?:al)?(?:_.*)?"
    r"|(?:input|local|raw|source)_?(?:file_?name|path)(?:_.*)?"
    r")$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    """One redacted validation or privacy finding."""

    code: str
    severity: str
    message: str
    path: str | None = None
    line: int | None = None
    redacted_value: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class ValidationResult:
    """Machine- and human-readable validation result."""

    errors: list[Finding] = field(default_factory=list)
    warnings: list[Finding] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def add_error(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        redacted_value: str | None = None,
    ) -> None:
        self.errors.append(
            Finding(code, "error", message, path, line, redacted_value)
        )

    def add_warning(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        line: int | None = None,
        redacted_value: str | None = None,
    ) -> None:
        self.warnings.append(
            Finding(code, "warning", message, path, line, redacted_value)
        )

    def extend(self, findings: Iterable[Finding]) -> None:
        for finding in findings:
            if finding.severity == "error":
                self.errors.append(finding)
            else:
                self.warnings.append(finding)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "checks": self.checks,
            "errors": [finding.as_dict() for finding in self.errors],
            "warnings": [finding.as_dict() for finding in self.warnings],
        }


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject ambiguous top-level values."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError("refusing to load a YAML document through a symbolic link")
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {source}")
    return value


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {source}")
    return value


def resolve_reference(reference: str | Path, *, relative_to: Path) -> Path:
    """Resolve a policy reference without permitting filesystem traversal.

    Policy files may be named relative to the release config or by a committed
    repository-relative path such as ``configs/release/policy.yaml``. Absolute
    paths, parent traversal, home expansion, and Windows drive-relative paths
    are rejected before touching the filesystem.
    """

    raw = str(reference).strip()
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    if (
        not raw
        or raw in {".", ".."}
        or raw.startswith("~")
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise ValueError("policy references must be safe relative paths")

    config_root = Path(relative_to).resolve()
    primary_unresolved = config_root / Path(raw)
    current = config_root
    for part in Path(raw).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("policy reference traverses a symbolic link")
    primary = primary_unresolved.resolve()
    if not primary.is_relative_to(config_root):
        raise ValueError("policy reference escapes the release-config directory")
    if primary.is_file():
        return primary

    repository_root = Path(__file__).resolve().parents[2]
    repository_unresolved = repository_root / Path(raw)
    current = repository_root
    for part in Path(raw).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("policy reference traverses a symbolic link")
    repository_candidate = repository_unresolved.resolve()
    if (
        repository_candidate.is_relative_to(repository_root)
        and repository_candidate.is_file()
    ):
        return repository_candidate
    return primary


def is_portable_relative_path(value: object) -> bool:
    """Return whether a relative path is portable across POSIX and Windows."""

    if not isinstance(value, str):
        return False
    raw = value
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    parts = raw.split("/")
    windows_reserved = {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
    return not (
        not raw
        or len(raw) > 512
        or raw != unicodedata.normalize("NFC", raw)
        or "\\" in raw
        or posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(
            not part
            or len(part) > 128
            or part in {".", ".."}
            or part.endswith((" ", "."))
            or any(ord(character) < 32 or ord(character) == 127 for character in part)
            or any(character in '<>:"|?*' for character in part)
            or part.split(".", 1)[0].casefold() in windows_reserved
            for part in parts
        )
    )


def release_connectome_relative_path(dataset: str) -> Path:
    """Return the one canonical staged connectome path for a dataset."""

    canonical = canonical_dataset(dataset)
    if canonical == "camcan":
        return CAMCAN_CONNECTOME_RELATIVE_PATH
    return Path("data/public_connectomes") / canonical / "connectomes.npz"


def release_metadata_relative_path(dataset: str) -> Path:
    """Return the one canonical staged public-metadata path for a dataset."""

    canonical = canonical_dataset(dataset)
    if canonical == "camcan":
        return CAMCAN_METADATA_RELATIVE_PATH
    return Path("data/public_metadata") / canonical / "metadata.tsv"


def _require_policy_schema_version(name: str, document: Mapping[str, Any]) -> None:
    version = document.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ValueError(
            f"{name} has an unsupported or missing schema_version"
        )


def _require_canonical_safety_mapping(
    name: str,
    value: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    allow_additional_fields: bool = False,
) -> None:
    """Require a complete, type-exact fail-closed configuration mapping."""

    missing = sorted(set(expected) - set(value))
    if missing:
        raise ValueError(f"{name} is missing required safety fields: {missing}")
    if not allow_additional_fields:
        unknown = sorted(set(value) - set(expected))
        if unknown:
            raise ValueError(f"{name} contains unknown fields: {unknown}")
    for field_name, canonical_value in expected.items():
        actual_value = value[field_name]
        if (
            type(actual_value) is not type(canonical_value)
            or actual_value != canonical_value
        ):
            raise ValueError(
                f"{name}.{field_name} must use the canonical fail-closed value"
            )


def _validate_release_config_schema(
    config: Mapping[str, Any], *, snapshot: bool
) -> None:
    allowed_top = {
        "schema_version",
        "project",
        "release",
        "restricted_reconstruction",
        "metadata",
        "content_allowlist",
        "safe_export_contract",
        "publication_requirements",
    }
    if not snapshot:
        allowed_top.add("paths")
    if set(config) - allowed_top:
        raise ValueError("release config contains unknown top-level fields")
    if config.get("schema_version") != SCHEMA_VERSION or isinstance(
        config.get("schema_version"), bool
    ):
        raise ValueError("release config has an unsupported schema_version")

    mapping_fields = {
        "project",
        "release",
        "metadata",
        "restricted_reconstruction",
        "content_allowlist",
        "safe_export_contract",
        "publication_requirements",
    }
    if not snapshot:
        mapping_fields.add("paths")
    for key in mapping_fields:
        value = config.get(key, {})
        if not isinstance(value, Mapping):
            raise ValueError(f"release config {key} must be a mapping")

    project = config.get("project", {})
    if set(project) - {
        "title",
        "version",
        "resource_type",
        "repository_url",
        "version_source",
    }:
        raise ValueError("release project config contains unknown fields")
    for key in ("title", "version", "resource_type"):
        if not isinstance(project.get(key), str) or not project[key].strip():
            raise ValueError(f"release project.{key} must be non-empty text")
    repository_url = project.get("repository_url")
    if (
        repository_url is not None
        and repository_url != CANONICAL_REPOSITORY_URL
    ):
        raise ValueError("release project repository_url is not canonical")

    release = config.get("release", {})
    if set(release) - {
        "archive_basename",
        "directory_name",
        "archive_format",
        "atlas",
        "expected_regions",
        "matrix_type",
        "positive_definite_required",
        "numeric_dtype_allowlist",
        "symmetry_tolerance",
        "diagonal_expected",
        "diagonal_tolerance",
        "spd_eigenvalue_tolerance",
        "normalized_archive_timestamp",
        "test_only",
        "version_confirmed",
        "publication_ready",
        "publication_gate",
    }:
        raise ValueError("release settings contain unknown fields")
    _require_canonical_safety_mapping(
        "release",
        release,
        _CANONICAL_RELEASE_SAFETY_SETTINGS,
        allow_additional_fields=True,
    )

    if snapshot and "paths" in config:
        raise ValueError("frozen release config must not contain filesystem paths")
    paths = config.get("paths", {})
    if not snapshot:
        if set(paths) - {
            "dataset_policy",
            "metadata_allowlist",
            "forbidden_patterns",
            "manual_approvals",
            "release_templates",
            "safe_export_dir",
            "output_dir",
            "staging_subdir",
            "upload_subdir",
            "reports_subdir",
        }:
            raise ValueError("release paths config contains unknown fields")

    restricted = config.get("restricted_reconstruction", {})
    if set(restricted) - {
        "task",
        "target",
        "grouping_rule",
        "outer_cv_protocols",
        "kfold_n_splits",
        "model_seed",
        "data_shuffle_seed",
        "dataset_reviews",
    }:
        raise ValueError("restricted reconstruction config contains unknown fields")
    reviews = restricted.get("dataset_reviews", {})
    if not isinstance(reviews, Mapping):
        raise ValueError("restricted dataset reviews must be a mapping")
    seen_reviews: set[str] = set()
    for raw_name, review in reviews.items():
        dataset = canonical_dataset(raw_name)
        if (
            dataset not in {"adni", "adnidod", "oasis3"}
            or dataset in seen_reviews
            or not isinstance(review, Mapping)
            or set(review)
            - {
                "selection_rules_reference",
                "selection_rules_reviewed",
                "aggregate_fold_counts",
            }
        ):
            raise ValueError("restricted dataset review has unknown fields")
        seen_reviews.add(dataset)

    from .metadata import _validate_metadata_schema

    _validate_metadata_schema(config)

    content = config.get("content_allowlist", {})
    if set(content) - {
        "default_decision",
        "unknown_dataset_decision",
        "unknown_content_category_decision",
        "categories",
        "always_forbidden_extensions",
        "excluded_from_upload",
        "aggregate_columns",
    }:
        raise ValueError("content allowlist contains unknown fields")
    categories = content.get("categories", {})
    allowed_categories = {
        "participant_level_connectomes",
        "participant_level_metadata",
        "exact_split_membership",
        "aggregate_metrics",
        "statistical_summaries",
        "figure_source_data",
        "configuration_files",
        "processing_scripts",
        "reconstruction_instructions",
        "reproducibility_metadata",
        "release_documentation",
    }
    category_fields = {
        "allowed_extensions",
        "requires_dataset_approval",
        "requires_metadata_allowlist",
        "allow_pickle",
        "requires_release_safe_sample_uid",
        "requires_artifact_review",
    }
    if not isinstance(categories, Mapping) or set(categories) - allowed_categories:
        raise ValueError("content allowlist categories contain unknown entries")
    for category, value in categories.items():
        if not isinstance(value, Mapping) or set(value) - category_fields:
            raise ValueError(f"content category {category} contains unknown fields")
        extensions = value.get("allowed_extensions", [])
        if not isinstance(extensions, list) or any(
            not isinstance(extension, str)
            or not re.fullmatch(r"\.[a-z0-9.]+", extension)
            for extension in extensions
        ):
            raise ValueError("content category extensions must be explicit suffixes")
    aggregate_columns = content.get("aggregate_columns", {})
    if not isinstance(aggregate_columns, Mapping) or set(
        aggregate_columns
    ) - {"aggregate_metrics", "statistical_summaries", "figure_source_data"}:
        raise ValueError("aggregate column allowlist contains unknown entries")
    for columns in aggregate_columns.values():
        if not isinstance(columns, list) or any(
            not isinstance(value, str) or not value.strip() for value in columns
        ):
            raise ValueError("aggregate column allowlist must contain strings")

    safe_export = config.get("safe_export_contract", {})
    _require_canonical_safety_mapping(
        "safe-export contract",
        safe_export,
        _CANONICAL_SAFE_EXPORT_CONTRACT,
    )

    publication = config.get("publication_requirements", {})
    _require_canonical_safety_mapping(
        "publication requirements",
        publication,
        _CANONICAL_PUBLICATION_REQUIREMENTS,
    )


def _validate_dataset_policy_schema(document: Mapping[str, Any]) -> None:
    if set(document) - {
        "schema_version",
        "default",
        "unknown_dataset",
        "datasets",
        "approval_semantics",
    }:
        raise ValueError("dataset policy contains unknown top-level fields")
    entries = document.get("datasets")
    if not isinstance(entries, Mapping):
        raise ValueError("dataset policy must define a datasets mapping")

    canonical_entries: set[str] = set()
    decision_fields = {
        "participant_level_connectomes",
        "participant_level_metadata",
        "exact_split_membership",
        "aggregate_metrics",
        "statistical_summaries",
        "figure_source_data",
        "configuration_files",
        "processing_scripts",
        "reconstruction_instructions",
    }
    for raw_name, policy in entries.items():
        dataset = canonical_dataset(raw_name)
        if dataset not in KNOWN_DATASETS:
            raise ValueError("dataset policy contains an unknown dataset")
        if dataset in canonical_entries:
            raise ValueError("dataset policy contains duplicate dataset aliases")
        if not isinstance(policy, Mapping):
            raise ValueError("each dataset policy entry must be a mapping")
        if set(policy) - {
            "display_name",
            "aliases",
            "participant_level_release",
            "participant_level_connectomes",
            "final_approval_status",
            "approval_reference",
            "decisions",
            "restricted_reconstruction_required",
            "restricted_reconstruction_required_until_approved",
            "explicit_policy_change_required",
        }:
            raise ValueError("dataset policy entry contains unknown fields")
        decisions = policy.get("decisions", {})
        if decisions and (
            not isinstance(decisions, Mapping)
            or set(decisions) - decision_fields
        ):
            raise ValueError("dataset policy decisions contain unknown fields")
        aliases = policy.get("aliases", [])
        if not isinstance(aliases, list) or any(
            not isinstance(value, str) or not value.strip() for value in aliases
        ):
            raise ValueError("dataset policy aliases must be non-empty strings")
        authorization = policy.get("participant_level_connectomes")
        if dataset in CONFIRMED_PARTICIPANT_RELEASE_DATASETS:
            expected = CONFIRMED_PARTICIPANT_AUTHORIZATIONS[dataset]
            if (
                policy.get("participant_level_release") != "allowed"
                or authorization != expected
                or not isinstance(decisions, Mapping)
                or decisions.get("participant_level_connectomes")
                != "allowed_with_confirmed_terms_and_source_binding"
                or decisions.get("participant_level_metadata")
                != (
                    "allowed_with_confirmed_terms_source_binding_and_allowlist"
                )
                or decisions.get("exact_split_membership") != "forbidden"
            ):
                raise ValueError(
                    f"{dataset} confirmed participant-connectome authorization "
                    "is incomplete or inconsistent"
                )
        elif (
            policy.get("participant_level_release") == "allowed"
            or authorization is not None
        ):
            raise ValueError(
                "only explicitly confirmed datasets may define an allowed "
                "participant-connectome authorization"
            )
        canonical_entries.add(dataset)

    fallback = document.get("unknown_dataset", document.get("default"))
    if not isinstance(fallback, Mapping):
        raise ValueError(
            "dataset policy must define a fail-closed unknown/default policy"
        )
    fallback_decision = str(
        fallback.get("participant_level_release", "forbidden")
    )
    if fallback_decision not in {
        "forbidden",
        "forbidden_unless_explicitly_approved",
    }:
        raise ValueError(
            "missing known-dataset policies require a forbidden fallback"
        )
    for name in ("default", "unknown_dataset"):
        value = document.get(name)
        if value is None:
            continue
        if not isinstance(value, Mapping) or set(value) - {
            "participant_level_release",
            "final_approval_status",
            "unknown_fields",
            "unknown_content_categories",
            "decisions",
            "reason",
        }:
            raise ValueError(f"dataset policy {name} contains unknown fields")
        decisions = value.get("decisions", {})
        if decisions and (
            not isinstance(decisions, Mapping)
            or set(decisions) - decision_fields
        ):
            raise ValueError(f"dataset policy {name} decisions contain unknown fields")

    approval_semantics = document.get("approval_semantics", {})
    if not isinstance(approval_semantics, Mapping) or set(approval_semantics) - {
        "approval_file",
        "recognized_status",
        "required_fields",
        "empty_or_missing_fields_invalidate_approval",
        "approval_never_overrides_forbidden_policy",
        "policy_change_and_approval_both_required_for_forbidden_unless_explicitly_approved",
    }:
        raise ValueError("dataset policy approval_semantics contains unknown fields")
    required_fields = approval_semantics.get("required_fields", [])
    if not isinstance(required_fields, list) or any(
        not isinstance(value, str) or not value.strip() for value in required_fields
    ):
        raise ValueError("dataset policy approval required_fields must be strings")


def _validate_metadata_allowlist_schema(document: Mapping[str, Any]) -> None:
    allowed_top = {
        "schema_version",
        "default",
        "identifier_policy",
        "forbidden_columns",
        "forbidden_column_patterns",
        "tables",
        "arrays",
        "datasets",
        "approval_rule",
        "required_columns",
        "global_allowed_columns",
        "always_allowed",
    }
    if set(document) - allowed_top:
        raise ValueError("metadata allowlist contains unknown top-level fields")

    default = document.get("default", {})
    if not isinstance(default, Mapping) or set(default) - {
        "decision",
        "unknown_columns",
        "unknown_array_keys",
        "case_sensitive_column_matching",
    }:
        raise ValueError("metadata allowlist default contains unknown fields")

    identifiers = document.get("identifier_policy", {})
    if not isinstance(identifiers, Mapping) or set(identifiers) - {
        "original_identifiers",
        "hashed_original_identifiers",
        "release_safe_sample_uid",
    }:
        raise ValueError("metadata identifier policy contains unknown fields")
    uid_policy = identifiers.get("release_safe_sample_uid", {})
    if not isinstance(uid_policy, Mapping) or set(uid_policy) - {
        "decision",
        "must_not_encode_source_identifier",
        "must_be_unique",
        "permitted_locations",
    }:
        raise ValueError("sample_uid policy contains unknown fields")

    list_fields = (
        "forbidden_columns",
        "required_columns",
        "global_allowed_columns",
        "always_allowed",
    )
    for name in list_fields:
        values = document.get(name, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"metadata allowlist {name} must be a string list")
    global_participant_columns = {
        str(value)
        for name in ("required_columns", "global_allowed_columns", "always_allowed")
        for value in document.get(name, [])
    }
    if global_participant_columns - {"sample_uid"}:
        raise ValueError(
            "global participant metadata may allow only release-safe sample_uid"
        )

    patterns = document.get("forbidden_column_patterns", [])
    if not isinstance(patterns, list):
        raise ValueError("metadata allowlist forbidden_column_patterns must be a list")
    for expression in patterns:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("metadata forbidden-column regex must be non-empty")
        try:
            re.compile(expression, re.IGNORECASE)
        except re.error as exc:
            raise ValueError("metadata allowlist contains an invalid regex") from exc

    table_names = {
        "public_metadata",
        "exact_split_membership",
        "aggregate_metrics",
        "statistical_summaries",
        "figure_source_data",
        "dataset_inventory",
        "data_dictionary",
        "provenance",
    }
    tables = document.get("tables", {})
    table_fields = {
        "allowed_columns",
        "default_allowed_columns",
        "candidate_columns_requiring_dataset_specific_manual_approval",
        "allowed_partitions",
        "requires_dataset_approval",
        "reject_original_or_hashed_identifiers",
        "reject_unrecognized_columns",
    }
    if not isinstance(tables, Mapping) or set(tables) - table_names:
        raise ValueError("metadata allowlist tables contain unknown entries")
    for name, table in tables.items():
        if not isinstance(table, Mapping) or set(table) - table_fields:
            raise ValueError(f"metadata table policy contains unknown fields: {name}")
        for key in (
            "allowed_columns",
            "default_allowed_columns",
            "candidate_columns_requiring_dataset_specific_manual_approval",
            "allowed_partitions",
        ):
            values = table.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(
                    f"metadata table {name}.{key} must be a string list"
                )

    arrays = document.get("arrays", {})
    if not isinstance(arrays, Mapping) or set(arrays) - {"public_connectome_npz"}:
        raise ValueError("metadata allowlist arrays contain unknown entries")
    array_policy = arrays.get("public_connectome_npz", {})
    if not isinstance(array_policy, Mapping) or set(array_policy) - {
        "allowed_keys",
        "forbidden_keys",
        "connectomes",
        "allow_pickle",
        "allow_object_dtype",
    }:
        raise ValueError("public connectome array policy contains unknown fields")
    connectome_policy = array_policy.get("connectomes", {})
    if not isinstance(connectome_policy, Mapping) or set(connectome_policy) - {
        "ndim",
        "shape",
        "dtype",
        "finite",
        "symmetric",
        "square",
        "positive_definite",
    }:
        raise ValueError("connectome array policy contains unknown fields")

    datasets = document.get("datasets", {})
    dataset_fields = {
        "allowed_columns",
        "approved_columns",
        "public_metadata_columns",
        "exact_split_columns",
        "status",
    }
    if not isinstance(datasets, Mapping):
        raise ValueError("metadata allowlist datasets must be a mapping")
    seen_datasets: set[str] = set()
    for raw_name, entry in datasets.items():
        dataset = canonical_dataset(raw_name)
        if dataset not in KNOWN_DATASETS or dataset in seen_datasets:
            raise ValueError("metadata allowlist contains an unknown or duplicate dataset")
        if not isinstance(entry, Mapping) or set(entry) - dataset_fields:
            raise ValueError("metadata dataset allowlist contains unknown fields")
        for key in (
            "allowed_columns",
            "approved_columns",
            "public_metadata_columns",
            "exact_split_columns",
        ):
            values = entry.get(key, [])
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(
                    f"metadata dataset {dataset}.{key} must be a string list"
                )
        nonempty_column_fields = [
            key
            for key in (
                "allowed_columns",
                "approved_columns",
                "public_metadata_columns",
            )
            if entry.get(key)
        ]
        if len(nonempty_column_fields) > 1:
            raise ValueError(
                "metadata dataset must use one canonical participant-column list"
            )
        if dataset in CONFIRMED_PARTICIPANT_RELEASE_DATASETS:
            expected_columns = {
                "abide": ["sample_uid"],
                "camcan": ["sample_uid", "age", "sex"],
                "cobre": ["sample_uid"],
            }[dataset]
            if (
                entry.get("public_metadata_columns") != expected_columns
                or entry.get("exact_split_columns", []) != []
                or entry.get("status")
                not in {None, "permission_confirmed_artifact_binding_required"}
            ):
                raise ValueError(
                    f"{dataset} metadata allowlist is inconsistent with "
                    "the confirmed release scope"
                )
        seen_datasets.add(dataset)

    approval_rule = document.get("approval_rule", {})
    if not isinstance(approval_rule, Mapping) or set(approval_rule) - {
        "adding_a_column_or_array_key_requires",
        "deidentification_alone_is_not_permission",
    }:
        raise ValueError("metadata approval rule contains unknown fields")
    requirements = approval_rule.get("adding_a_column_or_array_key_requires", [])
    if not isinstance(requirements, list) or any(
        not isinstance(value, str) or not value.strip() for value in requirements
    ):
        raise ValueError("metadata approval requirements must be a string list")


def _validate_manual_approvals_schema(document: Mapping[str, Any]) -> None:
    if set(document) - {
        "schema_version",
        "approval_protocol",
        "publication_gate",
        "approvals",
        "manual_action_items",
        "manual_actions",
    }:
        raise ValueError("manual approvals contain unknown top-level fields")

    protocol = document.get("approval_protocol", {})
    if not isinstance(protocol, Mapping) or set(protocol) - {
        "default_status",
        "recognized_approved_status",
        "designated_reviewer",
        "designation_confirmed_on",
        "required_fields_for_approval",
        "null_or_empty_required_field_invalidates_approval",
        "approval_does_not_override_forbidden_policy",
        "deliberate_confirmation_required",
        "required_confirmation_text",
    }:
        raise ValueError("manual approval protocol contains unknown fields")
    required_fields = protocol.get("required_fields_for_approval", [])
    if not isinstance(required_fields, list) or any(
        not isinstance(value, str) or not value.strip() for value in required_fields
    ):
        raise ValueError("manual approval required fields must be a string list")
    immutable_minimum = {"approved_by", "approved_on", "scope", "evidence"}
    if protocol and (
        not immutable_minimum.issubset(set(required_fields))
        or len(required_fields) != len(set(required_fields))
        or set(required_fields)
        - {
            "approved_by",
            "approved_on",
            "scope",
            "evidence",
            "confirmation",
        }
    ):
        raise ValueError("manual approval required fields cannot weaken the minimum")
    designated_reviewer = protocol.get("designated_reviewer")
    designation_date = protocol.get("designation_confirmed_on")
    if (designated_reviewer is None) != (designation_date is None):
        raise ValueError(
            "designated reviewer and confirmation date must be recorded together"
        )
    if designated_reviewer is not None:
        if (
            not isinstance(designated_reviewer, str)
            or not designated_reviewer.strip()
            or not isinstance(designation_date, str)
        ):
            raise ValueError("designated reviewer record is invalid")
        try:
            date.fromisoformat(designation_date)
        except ValueError as exc:
            raise ValueError(
                "designated reviewer confirmation date must be ISO formatted"
            ) from exc

    publication_gate = document.get("publication_gate", {})
    record_fields = {
        "required",
        "status",
        "policy_ceiling",
        "dataset",
        "atlas",
        "n_regions",
        "source_binding_sha256",
        "documented_identifier",
        "evidence_candidate",
        "license_identifier",
        "approved_by",
        "approved_on",
        "scope",
        "evidence",
        "confirmation",
        "publish_ready",
        "reviewed_artifacts",
    }
    if not isinstance(publication_gate, Mapping) or set(publication_gate) - record_fields:
        raise ValueError("publication gate contains unknown fields")

    approvals = document.get("approvals", {})
    if not isinstance(approvals, Mapping):
        raise ValueError("manual approvals.approvals must be a mapping")
    group_keys = {"datasets", "licenses", "zenodo_metadata", "release_artifacts"}
    grouped = bool(set(approvals) & group_keys)
    if grouped and set(approvals) - group_keys:
        raise ValueError("manual approvals cannot mix grouped and direct records")
    groups: list[tuple[str, Mapping[str, Any], set[str]]] = []
    if grouped:
        allowed_group_entries = {
            "datasets": set(KNOWN_DATASETS),
            "licenses": {"source_code", "derived_data", "documentation"},
            "zenodo_metadata": {
                "creators_and_order",
                "affiliations_and_orcids",
                "funding",
                "related_identifiers_and_manuscript_doi",
            },
            "release_artifacts": {
                "aggregate_results_content_and_pdf_metadata_review",
                "privacy_scan_review",
                "manifest_and_checksum_review",
                "final_zenodo_form_review",
            },
        }
        for group_name, allowed_entries in allowed_group_entries.items():
            value = approvals.get(group_name, {})
            if not isinstance(value, Mapping):
                raise ValueError(f"manual approval group {group_name} must be a mapping")
            groups.append((group_name, value, allowed_entries))
    else:
        groups.append(("datasets", approvals, set(KNOWN_DATASETS)))

    for group_name, records, allowed_entries in groups:
        canonical_names = {
            canonical_dataset(name) if group_name == "datasets" else str(name)
            for name in records
        }
        if canonical_names - allowed_entries or len(canonical_names) != len(records):
            raise ValueError(
                f"manual approval group {group_name} has unknown or duplicate entries"
            )
        for record in records.values():
            if not isinstance(record, Mapping) or set(record) - record_fields:
                raise ValueError("manual approval record contains unknown fields")
            scope = record.get("scope")
            if scope is not None and (
                not isinstance(scope, list)
                or any(
                    not isinstance(value, str) or not value.strip() for value in scope
                )
                or len(scope) != len(set(scope))
            ):
                raise ValueError("manual approval scope must be a unique string list")
            reviewed_artifacts = record.get("reviewed_artifacts", [])
            if not isinstance(reviewed_artifacts, list):
                raise ValueError("reviewed_artifacts must be a list")
            seen_artifacts: set[tuple[str, str, str, str]] = set()
            for artifact in reviewed_artifacts:
                if not isinstance(artifact, Mapping) or set(artifact) != {
                    "dataset",
                    "content_category",
                    "relative_path",
                    "sha256",
                }:
                    raise ValueError("reviewed artifact has an invalid schema")
                dataset = canonical_dataset(artifact["dataset"])
                category = str(artifact["content_category"])
                relative = str(artifact["relative_path"])
                digest = str(artifact["sha256"])
                expected_prefix = f"aggregate_outputs/{dataset}/"
                key = (dataset, category, relative, digest)
                if (
                    dataset not in KNOWN_DATASETS
                    or category
                    not in {
                        "aggregate_metrics",
                        "statistical_summaries",
                        "figure_source_data",
                    }
                    or not relative.startswith(expected_prefix)
                    or not is_portable_relative_path(relative)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or key in seen_artifacts
                ):
                    raise ValueError("reviewed artifact binding is unsafe or duplicate")
                seen_artifacts.add(key)

    for name in ("manual_action_items", "manual_actions"):
        actions = document.get(name, [])
        if not isinstance(actions, list) or any(
            not isinstance(value, str) or not value.strip() for value in actions
        ):
            raise ValueError(f"{name} must be a string list")


def _validate_forbidden_patterns_schema(document: Mapping[str, Any]) -> None:
    allowed_top = {
        "schema_version",
        "scanner",
        "redaction",
        "forbidden_extensions",
        "forbidden_path_components",
        "forbidden_filename_patterns",
        "forbidden_text_patterns",
        "forbidden_tabular_columns",
        "approved_contexts",
        "reporting",
        "patterns",
    }
    if set(document) - allowed_top:
        raise ValueError("forbidden-pattern config contains unknown top-level fields")
    scanner = document.get("scanner", {})
    scanner_keys = {
        "default_decision_on_error",
        "scan_filenames",
        "scan_text_content",
        "report_line_numbers_when_available",
        "include_full_sensitive_match_in_report",
        "text_extensions",
    }
    if not isinstance(scanner, Mapping) or set(scanner) - scanner_keys:
        raise ValueError("forbidden-pattern scanner config has unknown fields")
    if scanner and (
        scanner.get("default_decision_on_error") != "reject"
        or scanner.get("scan_filenames") is not True
        or scanner.get("scan_text_content") is not True
        or scanner.get("include_full_sensitive_match_in_report") is not False
    ):
        raise ValueError("privacy scanner must retain fail-closed settings")
    mapping_keys = {
        "redaction": {"prefix_characters", "suffix_characters", "replacement"},
        "forbidden_tabular_columns": {"exact", "regex"},
        "approved_contexts": {"email_addresses", "local_paths", "credentials"},
        "reporting": {"redact_sensitive_matches", "report_fields"},
    }
    for name, allowed in mapping_keys.items():
        value = document.get(name, {})
        if not isinstance(value, Mapping) or set(value) - allowed:
            raise ValueError(f"forbidden-pattern {name} config has unknown fields")
    for name in (
        "forbidden_extensions",
        "forbidden_path_components",
        "forbidden_filename_patterns",
        "forbidden_text_patterns",
        "patterns",
    ):
        value = document.get(name, [])
        if not isinstance(value, list):
            raise ValueError(f"forbidden-pattern {name} must be a list")
    pattern_entries = [
        *document.get("forbidden_filename_patterns", []),
        *document.get("forbidden_text_patterns", []),
        *document.get("patterns", []),
    ]
    for entry in pattern_entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry) - {"id", "code", "category", "regex", "description"}
            or not str(entry.get("id", entry.get("code", ""))).strip()
            or not isinstance(entry.get("regex"), str)
            or not entry["regex"].strip()
            or "\n" in entry["regex"]
        ):
            raise ValueError("forbidden-pattern entry has an invalid schema")
        try:
            re.compile(entry["regex"])
        except re.error as exc:
            raise ValueError("forbidden-pattern config contains invalid regex") from exc


def validate_release_bundle_documents(
    bundle: Mapping[str, Any], *, snapshot: bool
) -> None:
    """Validate a primary or frozen release-policy bundle with one strict path."""

    config = bundle.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("release bundle config must be a mapping")
    _validate_release_config_schema(config, snapshot=snapshot)
    validators = {
        "dataset_policy": _validate_dataset_policy_schema,
        "metadata_allowlist": _validate_metadata_allowlist_schema,
        "forbidden_patterns": _validate_forbidden_patterns_schema,
        "manual_approvals": _validate_manual_approvals_schema,
    }
    for name, validator in validators.items():
        document = bundle.get(name)
        if not isinstance(document, Mapping):
            raise ValueError(f"release bundle {name} must be a mapping")
        _require_policy_schema_version(name, document)
        validator(document)


def load_release_bundle(config_path: str | Path) -> dict[str, Any]:
    """Load the release config and each referenced policy file.

    References are resolved relative to the primary config.  A missing policy
    is fatal so the caller cannot silently fall back to a permissive default.
    """

    supplied_path = Path(config_path)
    if supplied_path.is_symlink():
        raise ValueError("release config must not be a symbolic link")
    path = supplied_path.resolve()
    config = load_yaml(path)
    _validate_release_config_schema(config, snapshot=False)
    references = config.get("paths")
    if not isinstance(references, Mapping):
        raise ValueError("release config must define a paths mapping")
    required = {
        "dataset_policy",
        "metadata_allowlist",
        "forbidden_patterns",
        "manual_approvals",
    }
    missing = sorted(required - set(references))
    if missing:
        raise ValueError(f"release config is missing policy references: {missing}")
    bundle: dict[str, Any] = {
        "config": config,
        "config_path": path,
    }
    for key in sorted(required):
        reference = references[key]
        if not isinstance(reference, (str, Path)) or not str(reference).strip():
            raise ValueError(f"required policy reference is invalid: {key}")
        referenced = resolve_reference(reference, relative_to=path.parent)
        if not referenced.is_file():
            raise ValueError(f"required policy file does not exist: {key}")
        document = load_yaml(referenced)
        _require_policy_schema_version(key, document)
        if key == "dataset_policy":
            _validate_dataset_policy_schema(document)
        elif key == "metadata_allowlist":
            _validate_metadata_allowlist_schema(document)
        elif key == "forbidden_patterns":
            _validate_forbidden_patterns_schema(document)
        elif key == "manual_approvals":
            _validate_manual_approvals_schema(document)
        bundle[key] = document
        bundle[f"{key}_path"] = referenced
    validate_release_bundle_documents(bundle, snapshot=False)
    return bundle


def canonical_dataset(value: object) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "abide": "abide",
        "adni": "adni",
        "oasis3": "oasis3",
        "oasis-3": "oasis3",
        "oasis_3": "oasis3",
        "camcan": "camcan",
        "cam-can": "camcan",
        "cam_can": "camcan",
        "cobre": "cobre",
        "adnidod": "adnidod",
        "adni-dod": "adnidod",
        "adni_dod": "adnidod",
        "1000brains": "1000brains",
        "1000-brains": "1000brains",
        "1000_brains": "1000brains",
    }
    return aliases.get(normalized, normalized)


def dataset_policy(bundle: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    policies = bundle["dataset_policy"]
    entries = policies.get("datasets", {})
    canonical = canonical_dataset(dataset)
    policy = entries.get(canonical)
    if not isinstance(policy, Mapping):
        if canonical in KNOWN_DATASETS:
            return {"participant_level_release": "forbidden"}
        fallback = policies.get("unknown_dataset", policies.get("default", {}))
        if not isinstance(fallback, Mapping):
            fallback = {}
        return fallback
    return policy


def _approval_entry(bundle: Mapping[str, Any], dataset: str) -> Mapping[str, Any]:
    approvals = bundle["manual_approvals"].get("approvals", {})
    if isinstance(approvals, Mapping) and isinstance(
        approvals.get("datasets"), Mapping
    ):
        approvals = approvals["datasets"]
    entry = approvals.get(canonical_dataset(dataset), {})
    return entry if isinstance(entry, Mapping) else {}


def approval_is_explicit(
    bundle: Mapping[str, Any], dataset: str, scope: str
) -> bool:
    """Validate a complete approval for the requested dataset scope."""

    entry = _approval_entry(bundle, dataset)
    required = ("approved_by", "approved_on", "evidence")
    scopes = entry.get("scope", [])
    if not isinstance(scopes, list) or any(
        not isinstance(value, str) or not value.strip() for value in scopes
    ) or len(scopes) != len(set(scopes)) or any(
        value not in PARTICIPANT_CONTENT_CATEGORIES for value in scopes
    ):
        return False
    values = [str(entry.get(key, "")).strip() for key in required]
    if any(
        not value
        or re.search(
            r"(?i)\b(?:todo|tbd|fixme|placeholder|unknown|required[_ -]?value)\b",
            value,
        )
        for value in values
    ):
        return False
    try:
        date.fromisoformat(values[1])
    except ValueError:
        return False
    protocol = bundle["manual_approvals"].get("approval_protocol", {})
    if (
        isinstance(protocol, Mapping)
        and protocol.get("deliberate_confirmation_required") is True
    ):
        expected = str(protocol.get("required_confirmation_text", "")).strip()
        if not expected or str(entry.get("confirmation", "")).strip() != expected:
            return False
    return (
        entry.get("status") == "approved"
        and scope in scopes
    )


def participant_release_decision(
    bundle: Mapping[str, Any], dataset: str, scope: str
) -> tuple[bool, str]:
    """Return whether participant-level content is explicitly releasable."""

    canonical = canonical_dataset(dataset)
    if canonical not in KNOWN_DATASETS:
        return False, "unknown_dataset_forbidden"
    policy = dataset_policy(bundle, canonical)
    decision = str(policy.get("participant_level_release", "forbidden"))
    if decision in {"forbidden", "forbidden_unless_explicitly_approved"}:
        return False, decision
    if (
        decision != "allowed"
        or canonical not in CONFIRMED_PARTICIPANT_RELEASE_DATASETS
    ):
        return False, "unrecognized_policy_decision"
    decisions = policy.get("decisions", {})
    policy_key = PARTICIPANT_POLICY_KEYS.get(scope)
    expected_scope_decision = {
        "participant_connectomes": (
            "allowed_with_confirmed_terms_and_source_binding"
        ),
        "participant_metadata": (
            "allowed_with_confirmed_terms_source_binding_and_allowlist"
        ),
    }.get(scope, "")
    if (
        not isinstance(decisions, Mapping)
        or policy_key is None
        or decisions.get(policy_key) != expected_scope_decision
    ):
        return False, "content_scope_not_explicitly_releasable"
    authorization = policy.get("participant_level_connectomes", {})
    expected_authorization = CONFIRMED_PARTICIPANT_AUTHORIZATIONS[canonical]
    if authorization != expected_authorization:
        return False, "confirmed_authorization_policy_incomplete"
    configured_metadata = allowed_metadata_columns(bundle, canonical)
    expected_metadata = {
        "sample_uid",
        *(
            str(value)
            for value in expected_authorization.get("allowed_metadata", [])
        ),
    }
    if configured_metadata != expected_metadata:
        return False, "confirmed_metadata_allowlist_mismatch"
    license_config = expected_authorization["license"]
    license_type = (
        license_config.get("type")
        if isinstance(license_config, Mapping)
        else None
    )
    metadata = bundle["config"].get("metadata", {})
    rights = (
        metadata.get("dataset_rights", {}).get(canonical, {})
        if isinstance(metadata, Mapping)
        and isinstance(metadata.get("dataset_rights", {}), Mapping)
        else {}
    )
    if rights != CONFIRMED_DATASET_RIGHTS[canonical]:
        return False, "dataset_rights_metadata_missing_or_mismatched"
    approval = _approval_entry(bundle, canonical)
    if approval.get("license_identifier") != license_type:
        return False, "approval_license_mismatch"
    if scope == "exact_splits":
        tables = bundle["metadata_allowlist"].get("tables", {})
        split_table = (
            tables.get("exact_split_membership", {})
            if isinstance(tables, Mapping)
            else {}
        )
        global_columns = (
            split_table.get("allowed_columns", [])
            if isinstance(split_table, Mapping)
            else []
        )
        dataset_entries = bundle["metadata_allowlist"].get("datasets", {})
        metadata_entry = (
            dataset_entries.get(canonical, {})
            if isinstance(dataset_entries, Mapping)
            else {}
        )
        columns = (
            metadata_entry.get("exact_split_columns", [])
            if isinstance(metadata_entry, Mapping)
            else []
        )
        if (
            global_columns != ["fold", "partition", "sample_uid"]
            or
            not isinstance(columns, list)
            or columns != ["fold", "partition", "sample_uid"]
        ):
            return False, "exact_split_allowlist_incomplete"
    if not approval_is_explicit(bundle, canonical, scope):
        return False, "manual_approval_incomplete"
    return True, "explicit_confirmed_permission_and_source_bound_approval"


def participant_policy_decision(
    bundle: Mapping[str, Any], dataset: str, scope: str
) -> str:
    """Return the frozen policy label for one participant content category."""

    canonical = canonical_dataset(dataset)
    policy = dataset_policy(bundle, canonical)
    decisions = policy.get("decisions", {})
    policy_key = PARTICIPANT_POLICY_KEYS.get(scope)
    if isinstance(decisions, Mapping) and policy_key is not None:
        decision = decisions.get(policy_key)
        if decision is not None and str(decision).strip():
            return str(decision)
    return str(policy.get("participant_level_release", "forbidden"))


def participant_inventory_decision(
    bundle: Mapping[str, Any],
    dataset: str,
    scope: str,
    *,
    present: bool,
) -> str:
    """Return an inventory label without overstating a real draft's status."""

    if not present:
        return participant_policy_decision(bundle, dataset, scope)
    allowed, reason = participant_release_decision(bundle, dataset, scope)
    if not allowed:
        return reason
    release = bundle["config"].get("release", {})
    if (
        isinstance(release, Mapping)
        and release.get("test_only") is not True
        and release.get("publication_ready") is not True
    ):
        return "source_bound_draft_candidate_pending_final_review"
    return reason


def allowed_metadata_columns(
    bundle: Mapping[str, Any], dataset: str
) -> set[str]:
    """Return only columns explicitly allowed globally and for a dataset."""

    allowlist = bundle["metadata_allowlist"]
    globally_allowed: set[str] = set()
    for key in ("required_columns", "global_allowed_columns", "always_allowed"):
        values = allowlist.get(key, [])
        if isinstance(values, list):
            globally_allowed.update(
                str(value) for value in values if str(value) == "sample_uid"
            )
    dataset_entries = allowlist.get("datasets", {})
    entry = dataset_entries.get(canonical_dataset(dataset), {})
    if isinstance(entry, list):
        globally_allowed.update(str(value) for value in entry)
    elif isinstance(entry, Mapping):
        values = entry.get("public_metadata_columns", [])
        if isinstance(values, list):
            globally_allowed.update(str(value) for value in values)
    return globally_allowed


def normalize_metadata_column(value: object) -> str:
    """Normalize a public column name for deny-list and regex matching."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def metadata_column_is_forbidden(
    column: object,
    *,
    bundle: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether a metadata header matches a built-in/configured deny rule."""

    normalized = normalize_metadata_column(column)
    if (
        not normalized
        or normalized in FORBIDDEN_METADATA_COLUMNS
        or FORBIDDEN_METADATA_COLUMN_RE.search(normalized)
    ):
        return True
    if bundle is None:
        return False

    allowlist = bundle.get("metadata_allowlist", {})
    if not isinstance(allowlist, Mapping):
        raise ValueError("metadata_allowlist must be a mapping")
    configured_columns = allowlist.get("forbidden_columns", [])
    if not isinstance(configured_columns, list):
        raise ValueError("metadata allowlist forbidden_columns must be a list")
    if normalized in {
        normalize_metadata_column(value) for value in configured_columns
    }:
        return True

    expressions = allowlist.get("forbidden_column_patterns", [])
    if not isinstance(expressions, list):
        raise ValueError(
            "metadata allowlist forbidden_column_patterns must be a list"
        )
    for expression in expressions:
        try:
            pattern = re.compile(str(expression), re.IGNORECASE)
        except re.error as exc:
            raise ValueError("metadata allowlist contains an invalid regex") from exc
        if pattern.search(normalized):
            return True
    return False


def validate_metadata_columns(
    columns: Iterable[str],
    *,
    bundle: Mapping[str, Any],
    dataset: str,
) -> list[Finding]:
    allowed = allowed_metadata_columns(bundle, dataset)
    findings: list[Finding] = []
    for raw_column in columns:
        column = str(raw_column).strip()
        redacted_column = (
            column[:1] + "…" + column[-1:]
            if len(column) > 2
            else "[redacted]"
        )
        if metadata_column_is_forbidden(column, bundle=bundle):
            findings.append(
                Finding(
                    "FORBIDDEN_METADATA_COLUMN",
                    "error",
                    "tabular metadata contains a forbidden participant column",
                    redacted_value=redacted_column,
                )
            )
        elif column not in allowed:
            findings.append(
                Finding(
                    "UNAPPROVED_METADATA_COLUMN",
                    "error",
                    "tabular metadata contains a column not explicitly allowed",
                    redacted_value=redacted_column,
                )
            )
    return findings


def read_tsv(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            header = next(reader, None)
            if header is None:
                raise ValueError(f"TSV has no header: {source.name}")
            normalized_header = [column.strip() for column in header]
            if not header or any(not column for column in normalized_header):
                raise ValueError(f"TSV has an empty header column: {source.name}")
            if len(set(normalized_header)) != len(normalized_header):
                raise ValueError(f"TSV has duplicate header columns: {source.name}")

            rows: list[dict[str, str]] = []
            for line_number, values in enumerate(reader, start=2):
                if len(values) != len(header):
                    raise ValueError(
                        f"TSV row {line_number} has {len(values)} field(s); "
                        f"expected {len(header)}"
                    )
                rows.append(dict(zip(header, values)))
    except csv.Error as exc:
        raise ValueError(f"TSV is malformed: {source.name}") from exc
    return list(header), rows


def _validated_tolerance(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    lower, upper = _TOLERANCE_LIMITS[name]
    if not math.isfinite(numeric) or not lower <= numeric <= upper:
        raise ValueError(
            f"{name} must be finite and between {lower:g} and {upper:g}"
        )
    return numeric


def _canonical_atlas(value: object) -> str:
    atlas = str(value).strip().lower()
    atlas = _ATLAS_ALIASES.get(atlas, atlas)
    if atlas not in ATLAS_REGION_COUNTS:
        raise ValueError("release atlas is unknown")
    return atlas


def validate_release_numeric_settings(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize matrix/atlas settings used by release checks.

    The helper accepts either a full release config containing a ``release``
    mapping or that mapping directly. Atlas information remains optional for
    backward compatibility, but when supplied it is bound to its canonical
    expected region count.
    """

    if not isinstance(config, Mapping):
        raise ValueError("release numeric settings must be a mapping")
    if "release" in config:
        release = config["release"]
        if not isinstance(release, Mapping):
            raise ValueError("release config must define a release mapping")
    else:
        release = config

    matrix_type = str(release.get("matrix_type", "correlation")).strip().lower()
    if matrix_type not in MATRIX_TYPES:
        raise ValueError("release matrix_type is unknown")

    settings: dict[str, Any] = {
        "matrix_type": matrix_type,
        "symmetry_tolerance": _validated_tolerance(
            "symmetry_tolerance",
            release.get("symmetry_tolerance", 1.0e-6),
        ),
        "diagonal_tolerance": _validated_tolerance(
            "diagonal_tolerance",
            release.get("diagonal_tolerance", 1.0e-5),
        ),
        "spd_eigenvalue_tolerance": _validated_tolerance(
            "spd_eigenvalue_tolerance",
            release.get("spd_eigenvalue_tolerance", 0.0),
        ),
    }

    atlas_value = release.get("atlas")
    if atlas_value is None and "release" in config:
        atlas_value = config.get("atlas")
    atlas = _canonical_atlas(atlas_value) if atlas_value is not None else None

    explicit_regions = release.get(
        "expected_regions",
        release.get("n_regions"),
    )
    if explicit_regions is not None and (
        isinstance(explicit_regions, bool)
        or not isinstance(explicit_regions, int)
        or explicit_regions <= 0
    ):
        raise ValueError("expected_regions must be a positive integer")
    expected_regions = (
        ATLAS_REGION_COUNTS[atlas]
        if atlas is not None
        else explicit_regions
    )
    if (
        atlas is not None
        and explicit_regions is not None
        and explicit_regions != expected_regions
    ):
        raise ValueError("expected_regions does not match the configured atlas")
    settings["atlas"] = atlas
    settings["expected_regions"] = expected_regions
    return settings


def validate_connectome_npz(
    path: str | Path,
    *,
    matrix_type: str = "correlation",
    symmetry_tolerance: float = 1e-6,
    diagonal_tolerance: float = 1e-5,
    spd_eigenvalue_tolerance: float = 0.0,
    expected_regions: int | None = None,
) -> tuple[int | None, list[Finding]]:
    """Safely inspect a public NPZ without enabling pickle loading."""

    settings_input: dict[str, Any] = {
        "matrix_type": matrix_type,
        "symmetry_tolerance": symmetry_tolerance,
        "diagonal_tolerance": diagonal_tolerance,
        "spd_eigenvalue_tolerance": spd_eigenvalue_tolerance,
    }
    if expected_regions is not None:
        settings_input["expected_regions"] = expected_regions
    settings = validate_release_numeric_settings(settings_input)
    matrix_type = settings["matrix_type"]
    symmetry_tolerance = settings["symmetry_tolerance"]
    diagonal_tolerance = settings["diagonal_tolerance"]
    spd_eigenvalue_tolerance = settings["spd_eigenvalue_tolerance"]
    expected_regions = settings["expected_regions"]

    source = Path(path)
    findings: list[Finding] = []
    try:
        with zipfile.ZipFile(source, "r") as container:
            members = container.infolist()
            if (
                len(members) != 1
                or members[0].filename != "connectomes.npy"
                or members[0].is_dir()
                or members[0].flag_bits != 0
                or members[0].compress_type
                not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                or members[0].file_size <= 0
                or members[0].header_offset != 0
                or members[0].extra
                or members[0].comment
                or container.comment
            ):
                findings.append(
                    Finding(
                        "NPZ_CONTAINER_SCHEMA",
                        "error",
                        "connectome NPZ must contain exactly one unencrypted connectomes.npy member",
                        str(source),
                    )
                )
                return None, findings
            member = members[0]
            encoded_name = member.filename.encode("ascii")
            with source.open("rb") as raw_archive:
                local_header = raw_archive.read(30)
                if len(local_header) != 30:
                    raise zipfile.BadZipFile("truncated local header")
                (
                    signature,
                    _extract_version,
                    local_flags,
                    local_compression,
                    _mtime,
                    _mdate,
                    local_crc,
                    local_compressed_size,
                    local_file_size,
                    filename_length,
                    local_extra_length,
                ) = struct.unpack("<4s5H3L2H", local_header)
                local_name = raw_archive.read(filename_length)
                local_extra = raw_archive.read(local_extra_length)
                raw_archive.seek(0, 2)
                archive_size = raw_archive.tell()
            expected_start_dir = (
                30
                + len(encoded_name)
                + local_extra_length
                + member.compress_size
            )
            expected_archive_size = (
                expected_start_dir + 46 + len(encoded_name) + 22
            )
            standard_sizes = (
                not local_extra
                and local_compressed_size == member.compress_size
                and local_file_size == member.file_size
            )
            zip64_sizes = False
            if len(local_extra) == 20:
                extra_id, extra_size, file_size_64, compressed_size_64 = (
                    struct.unpack("<HHQQ", local_extra)
                )
                zip64_sizes = (
                    extra_id == 0x0001
                    and extra_size == 16
                    and local_compressed_size
                    in {0xFFFFFFFF, member.compress_size}
                    and local_file_size in {0xFFFFFFFF, member.file_size}
                    and compressed_size_64 == member.compress_size
                    and file_size_64 == member.file_size
                )
            if (
                signature != b"PK\x03\x04"
                or local_flags != 0
                or local_compression != member.compress_type
                or local_crc != member.CRC
                or filename_length != len(encoded_name)
                or local_name != encoded_name
                or not (standard_sizes or zip64_sizes)
                or container.start_dir != expected_start_dir
                or archive_size != expected_archive_size
            ):
                findings.append(
                    Finding(
                        "NPZ_CONTAINER_LAYOUT",
                        "error",
                        "connectome NPZ contains non-canonical or trailing ZIP data",
                        str(source),
                    )
                )
                return None, findings
            payload = container.read(member)
            payload_stream = io.BytesIO(payload)
            try:
                parsed_matrices = np.lib.format.read_array(
                    payload_stream, allow_pickle=False
                )
            except ValueError:
                findings.append(
                    Finding(
                        "OBJECT_ARRAY",
                        "error",
                        "NPZ contains an object array or requires pickle loading",
                        str(source),
                    )
                )
                return None, findings
            if payload_stream.tell() != len(payload):
                findings.append(
                    Finding(
                        "NPY_TRAILING_DATA",
                        "error",
                        "connectomes.npy contains trailing or concatenated payload bytes",
                        str(source),
                    )
                )
                return None, findings
        with np.load(source, allow_pickle=False) as archive:
            if archive.files != ["connectomes"]:
                findings.append(
                    Finding(
                        "NPZ_SCHEMA",
                        "error",
                        "connectome NPZ must contain only the 'connectomes' array",
                        str(source),
                    )
                )
            if "connectomes" not in archive.files:
                return None, findings
            try:
                matrices = archive["connectomes"]
            except ValueError:
                findings.append(
                    Finding(
                        "OBJECT_ARRAY",
                        "error",
                        "NPZ contains an object array or requires pickle loading",
                        str(source),
                    )
                )
                return None, findings
            if (
                matrices.shape != parsed_matrices.shape
                or matrices.dtype != parsed_matrices.dtype
            ):
                findings.append(
                    Finding(
                        "NPZ_ARRAY_MISMATCH",
                        "error",
                        "NPZ array view does not match its canonical NPY payload",
                        str(source),
                    )
                )
                return None, findings
    except (OSError, ValueError, zipfile.BadZipFile):
        findings.append(
            Finding(
                "INVALID_NPZ",
                "error",
                "NPZ is unreadable or malformed under allow_pickle=False",
                str(source),
            )
        )
        return None, findings

    dtype_allowed = matrices.dtype in {np.dtype("float32"), np.dtype("float64")}
    if not dtype_allowed:
        findings.append(
            Finding(
                "NPZ_DTYPE",
                "error",
                "connectomes must use exactly float32 or float64 without objects",
                str(source),
            )
        )
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        findings.append(
            Finding(
                "CONNECTOME_SHAPE",
                "error",
                "connectomes must have shape [n_samples, n_regions, n_regions]",
                str(source),
            )
        )
        return None, findings
    if not dtype_allowed:
        return int(matrices.shape[0]), findings
    if expected_regions is not None and matrices.shape[1] != expected_regions:
        findings.append(
            Finding(
                "ATLAS_REGION_COUNT",
                "error",
                (
                    f"connectomes have {matrices.shape[1]} regions; "
                    f"expected {expected_regions}"
                ),
                str(source),
            )
        )
    if matrices.shape[0] == 0 or matrices.shape[1] == 0:
        findings.append(
            Finding("EMPTY_CONNECTOMES", "error", "connectomes must not be empty", str(source))
        )
        return int(matrices.shape[0]), findings
    if not np.isfinite(matrices).all():
        findings.append(
            Finding("NONFINITE_ARRAY", "error", "connectomes contain non-finite values", str(source))
        )
    if not np.allclose(
        matrices,
        np.swapaxes(matrices, -1, -2),
        atol=symmetry_tolerance,
        rtol=0.0,
    ):
        findings.append(
            Finding("ASYMMETRIC_CONNECTOME", "error", "connectomes are not symmetric", str(source))
        )
    if matrix_type == "correlation":
        diagonal = np.diagonal(matrices, axis1=1, axis2=2)
        if not np.allclose(diagonal, 1.0, atol=diagonal_tolerance, rtol=0.0):
            findings.append(
                Finding(
                    "CORRELATION_DIAGONAL",
                    "error",
                    "correlation-matrix diagonal is not approximately one",
                    str(source),
                )
            )
    try:
        minimum = np.linalg.eigvalsh(matrices).min(axis=1)
        if np.any(minimum <= spd_eigenvalue_tolerance):
            findings.append(
                Finding(
                    "NON_SPD_CONNECTOME",
                    "error",
                    "connectomes are not positive definite",
                    str(source),
                )
            )
    except (np.linalg.LinAlgError, TypeError, ValueError):
        findings.append(
            Finding(
                "EIGENVALUE_FAILURE",
                "error",
                "SPD eigenvalue check failed",
                str(source),
            )
        )
    return int(matrices.shape[0]), findings
