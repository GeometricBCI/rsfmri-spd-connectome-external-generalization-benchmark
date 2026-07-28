"""Zenodo companion metadata generation without API calls or invented values."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml


MANUSCRIPT_TITLE = (
    "Benchmarking External Generalization of SPD Matrix Learning for "
    "Resting-State fMRI Connectome Prediction"
)
PLACEHOLDER_RE = re.compile(
    r"(?i)\b(?:todo|tbd|fixme|placeholder|unknown|required[_ -]?value)\b|"
    r"10\.x{4,}/|0000-0000-0000-0000"
)
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
METADATA_KEYS = {
    "creators",
    "description",
    "keywords",
    "licenses",
    "dataset_rights",
    "related_identifiers",
    "funding",
    "communities",
    "manuscript_doi",
    "required_manual_fields",
}
DATASET_RIGHTS_DATASETS = ("camcan", "abide", "cobre")
DATASET_RIGHT_KEYS = {
    "status",
    "license",
    "conditions",
    "required_citations",
    "rights_statement",
}
DATASET_DISPLAY_NAMES = {
    "camcan": "CamCAN",
    "abide": "ABIDE",
    "cobre": "COBRE",
}
RIGHTS_DISPLAY_VALUES = {
    "approved_derived_data_release": "approved derived-data release scope",
    "CC-BY-4.0": (
        "Creative Commons Attribution 4.0 International (CC BY 4.0)"
    ),
    "ABIDE_source_license": (
        "ABIDE/INDI source terms (exact title, version, and URI require final "
        "metadata review)"
    ),
    "COBRE_confirmed_license": (
        "COBRE/FCP source terms (exact title, version, and URI require final "
        "metadata review)"
    ),
    "attribution": "attribution required",
    "no_identifiable_images": "identifiable images excluded",
    "no_identifiable_behavioural_variables": (
        "identifiable behavioural variables excluded"
    ),
    "metadata_limited_to_age_and_sex": "public metadata limited to age and sex",
    "exclude_identifiable_material": "identifiable material excluded",
    "limit_metadata_to_age_and_sex": "public metadata limited to age and sex",
    "non_commercial_use": "non-commercial use only",
    "share_alike_if_required": "retain any applicable share-alike condition",
    "confirmed_cobre_fcp_redistribution_terms": (
        "retain the confirmed COBRE/FCP redistribution conditions"
    ),
    "Shafto_et_al_CamCAN_cohort_paper": (
        "Shafto et al. (2014), The Cambridge Centre for Ageing and "
        "Neuroscience (Cam-CAN) study protocol: a cross-sectional, lifespan, "
        "multidisciplinary examination of healthy cognitive ageing, "
        "BMC Neurology 14:204, "
        "https://doi.org/10.1186/s12883-014-0204-1"
    ),
    "ABIDE_reference": (
        "ABIDE consortium citation (complete wording requires final metadata "
        "review)"
    ),
    "INDI_reference": (
        "1000 Functional Connectomes Project/INDI citation and acknowledgements "
        "(complete wording requires final metadata review)"
    ),
    "COBRE_reference": (
        "COBRE dataset citation (complete wording requires final metadata review)"
    ),
    "FCP_INDI_reference": (
        "FCP/INDI citation and acknowledgement (complete wording requires final "
        "metadata review)"
    ),
}
MIXED_RIGHTS_ACTION = (
    "Complete and approve the Zenodo record-level mixed-rights presentation "
    "without replacing the dataset-specific terms with one global license."
)
CREATOR_KEYS = {"name", "affiliation", "orcid"}
RELATED_IDENTIFIER_KEYS = {"identifier", "relation", "resource_type"}
RELATED_IDENTIFIER_RELATIONS = {
    "isCitedBy",
    "cites",
    "isSupplementTo",
    "isSupplementedBy",
    "isContinuedBy",
    "continues",
    "isDescribedBy",
    "describes",
    "hasMetadata",
    "isMetadataFor",
    "hasVersion",
    "isVersionOf",
    "isNewVersionOf",
    "isPreviousVersionOf",
    "isPartOf",
    "hasPart",
    "isReferencedBy",
    "references",
    "isDocumentedBy",
    "documents",
    "isCompiledBy",
    "compiles",
    "isVariantFormOf",
    "isOriginalFormOf",
    "isIdenticalTo",
    "isAlternateIdentifier",
    "isSourceOf",
    "isDerivedFrom",
    "requires",
    "isRequiredBy",
    "isObsoletedBy",
    "obsoletes",
}


def contains_placeholder(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, Mapping):
        return any(contains_placeholder(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains_placeholder(item) for item in value)
    return bool(PLACEHOLDER_RE.search(str(value)))


def _metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def _validate_metadata_schema(config: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = _metadata(config)
    if set(metadata) - METADATA_KEYS:
        raise ValueError("Zenodo metadata contains unknown fields")
    if not isinstance(metadata.get("description", ""), str):
        raise ValueError("Zenodo description must be text")
    creators = metadata.get("creators", [])
    if not isinstance(creators, list):
        raise ValueError("Zenodo creators must be a list")
    for creator in creators:
        if (
            not isinstance(creator, Mapping)
            or set(creator) - CREATOR_KEYS
            or not isinstance(creator.get("name"), str)
            or not creator["name"].strip()
            or any(
                key in creator
                and (
                    not isinstance(creator[key], str)
                    or not creator[key].strip()
                )
                for key in CREATOR_KEYS - {"name"}
            )
        ):
            raise ValueError("Zenodo creator entry has an invalid or unknown field")
    for key in ("keywords", "required_manual_fields"):
        values = metadata.get(key, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() for value in values
        ):
            raise ValueError(f"Zenodo metadata {key} must be a string list")
    licenses = metadata.get("licenses", {})
    if not isinstance(licenses, Mapping) or set(licenses) - {
        "source_code",
        "derived_data",
        "documentation",
    }:
        raise ValueError("Zenodo licenses mapping has unknown fields")
    for value in licenses.values():
        if isinstance(value, str):
            if not value.strip():
                raise ValueError("Zenodo license identifier must not be empty")
        elif (
            not isinstance(value, Mapping)
            or set(value) - {"identifier", "status", "evidence"}
            or any(
                value.get(key) is not None and not isinstance(value.get(key), str)
                for key in ("identifier", "status", "evidence")
            )
        ):
            raise ValueError("Zenodo license entry has unknown fields")
    dataset_rights = metadata.get("dataset_rights", {})
    if not isinstance(dataset_rights, Mapping):
        raise ValueError("dataset_rights must be a mapping")
    if set(dataset_rights) - set(DATASET_RIGHTS_DATASETS):
        raise ValueError("dataset_rights contains an unknown dataset")
    if (
        config.get("release", {}).get("test_only") is not True
        and set(dataset_rights) != set(DATASET_RIGHTS_DATASETS)
    ):
        raise ValueError(
            "non-test metadata must define rights for CamCAN, ABIDE, and COBRE"
        )
    for dataset, value in dataset_rights.items():
        if (
            not isinstance(value, Mapping)
            or set(value) != DATASET_RIGHT_KEYS
            or value.get("status") != "approved_derived_data_release"
            or not isinstance(value.get("license"), str)
            or not value["license"].strip()
            or not isinstance(value.get("rights_statement"), str)
            or not value["rights_statement"].strip()
            or re.search(
                r"(?i)\bpending\s+approval\b",
                value["rights_statement"],
            )
        ):
            raise ValueError(
                f"{dataset} dataset rights are incomplete or contain unknown fields"
            )
        for key in ("conditions", "required_citations"):
            values = value.get(key)
            if (
                not isinstance(values, list)
                or not values
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in values
                )
                or len(values) != len(set(values))
            ):
                raise ValueError(
                    f"{dataset} dataset rights {key} must be a non-empty "
                    "unique string list"
                )
    if config.get("release", {}).get("test_only") is not True:
        from .schemas import CONFIRMED_DATASET_RIGHTS

        if dataset_rights != CONFIRMED_DATASET_RIGHTS:
            raise ValueError(
                "non-test dataset rights must exactly match the reviewed "
                "dataset-specific scope"
            )
    related = metadata.get("related_identifiers", [])
    if not isinstance(related, list):
        raise ValueError("related_identifiers must be a list")
    for value in related:
        if (
            not isinstance(value, Mapping)
            or set(value) - RELATED_IDENTIFIER_KEYS
            or set(value) != RELATED_IDENTIFIER_KEYS
            or not all(
                isinstance(value[key], str) and value[key].strip()
                for key in RELATED_IDENTIFIER_KEYS
            )
            or value["relation"] not in RELATED_IDENTIFIER_RELATIONS
        ):
            raise ValueError("related identifier has an invalid or unknown field")
    funding = metadata.get("funding", [])
    if not isinstance(funding, list) or any(
        not isinstance(value, Mapping)
        or set(value) != {"id"}
        or not isinstance(value["id"], str)
        or not value["id"].strip()
        for value in funding
    ):
        raise ValueError("funding entries must contain only an authoritative id")
    communities = metadata.get("communities", [])
    if not isinstance(communities, list) or any(
        not isinstance(value, str) or not value.strip() for value in communities
    ):
        raise ValueError("communities must be a list of reviewed identifiers")
    manuscript_doi = metadata.get("manuscript_doi")
    if manuscript_doi is not None and (
        not isinstance(manuscript_doi, str)
        or not DOI_RE.fullmatch(manuscript_doi.strip())
    ):
        raise ValueError("manuscript_doi must be null or a valid reviewed DOI")
    return metadata


def dataset_rights_entries(
    config: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Return reviewed dataset rights in stable display order."""

    metadata = _validate_metadata_schema(config)
    rights = metadata.get("dataset_rights", {})
    if not isinstance(rights, Mapping):
        return []
    return [
        (dataset, rights[dataset])
        for dataset in DATASET_RIGHTS_DATASETS
        if isinstance(rights.get(dataset), Mapping)
    ]


def dataset_rights_display_value(rights: Mapping[str, Any], key: str) -> str:
    """Return a release-facing description for one rights-policy value."""

    value = rights.get(key)
    values = value if isinstance(value, list) else [value]
    rendered = [
        RIGHTS_DISPLAY_VALUES.get(str(item), str(item).replace("_", " "))
        for item in values
        if item is not None and str(item).strip()
    ]
    return "; ".join(rendered) if rendered else "manual review required"


def dataset_rights_text(config: Mapping[str, Any]) -> str:
    """Render a plain-text rights block suitable for Zenodo description fields."""

    lines = [
        "Dataset-specific rights (permissions confirmed):",
        "",
    ]
    for dataset, rights in dataset_rights_entries(config):
        name = DATASET_DISPLAY_NAMES[dataset]
        lines.append(
            f"{name} — {dataset_rights_display_value(rights, 'status')}. "
            "Dataset-specific license or terms: "
            + dataset_rights_display_value(rights, "license")
            + ". Conditions: "
            + dataset_rights_display_value(rights, "conditions")
            + ". Required citations: "
            + dataset_rights_display_value(rights, "required_citations")
            + f". Rights statement: {rights['rights_statement']}"
        )
        lines.append("")
    lines.append(
        "No single record-level license supersedes these dataset-specific terms."
    )
    return "\n".join(lines)


def _dataset_rights_markdown(config: Mapping[str, Any]) -> str:
    """Render dataset-specific terms as a readable review worksheet."""

    lines: list[str] = []
    for dataset, rights in dataset_rights_entries(config):
        name = DATASET_DISPLAY_NAMES[dataset]
        lines.extend(
            [
                "### "
                f"{name} — {dataset_rights_display_value(rights, 'status')}",
                "",
                "- Dataset-specific license or terms: "
                f"{dataset_rights_display_value(rights, 'license')}.",
                "- Conditions: "
                f"{dataset_rights_display_value(rights, 'conditions')}.",
                "- Required citations: "
                f"{dataset_rights_display_value(rights, 'required_citations')}.",
                f"- Rights statement: {rights['rights_statement']}",
                "",
            ]
        )
    lines.extend(
        [
            "No single record-level license supersedes these dataset-specific "
            "terms.",
        ]
    )
    return "\n".join(lines)


def _review_record_is_complete(
    record: object, manual_approvals: Mapping[str, Any]
) -> bool:
    if not isinstance(record, Mapping) or record.get("status") != "approved":
        return False
    scope = record.get("scope")
    if (
        not isinstance(scope, list)
        or not scope
        or any(not isinstance(value, str) or not value.strip() for value in scope)
        or len(scope) != len(set(scope))
    ):
        return False
    protocol = manual_approvals.get("approval_protocol", {})
    minimum_required = {"approved_by", "approved_on", "scope", "evidence"}
    required = (
        protocol.get(
            "required_fields_for_approval",
            sorted(minimum_required),
        )
        if isinstance(protocol, Mapping)
        else sorted(minimum_required)
    )
    if (
        not isinstance(required, list)
        or not minimum_required.issubset({str(key) for key in required})
        or any(
        not str(record.get(str(key), "")).strip() for key in required
        )
    ):
        return False
    try:
        date.fromisoformat(str(record.get("approved_on", "")).strip())
    except ValueError:
        return False
    if (
        isinstance(protocol, Mapping)
        and protocol.get("deliberate_confirmation_required") is True
    ):
        expected = str(protocol.get("required_confirmation_text", "")).strip()
        if not expected or str(record.get("confirmation", "")).strip() != expected:
            return False
    return True


def _derived_data_rights_review_is_complete(
    bundle: Mapping[str, Any],
) -> bool:
    """Check the explicit review used when one global data license is unsuitable."""

    metadata = _metadata(bundle["config"])
    rights = metadata.get("dataset_rights", {})
    licenses = metadata.get("licenses", {})
    derived = licenses.get("derived_data") if isinstance(licenses, Mapping) else None
    if (
        not isinstance(rights, Mapping)
        or not rights
        or not isinstance(derived, Mapping)
        or derived.get("status") != "approved"
        or not str(derived.get("evidence", "")).strip()
    ):
        return False
    approvals = bundle.get("manual_approvals", {}).get("approvals", {})
    license_group = (
        approvals.get("licenses", {}) if isinstance(approvals, Mapping) else {}
    )
    review = (
        license_group.get("derived_data", {})
        if isinstance(license_group, Mapping)
        else {}
    )
    if (
        not _review_record_is_complete(review, bundle["manual_approvals"])
        or review.get("scope") != ["approved_derived_data_artifacts"]
    ):
        return False
    identifier = derived.get("identifier")
    reviewed_identifier = (
        review.get("license_identifier") if isinstance(review, Mapping) else None
    )
    if identifier is None:
        return reviewed_identifier in {None, ""}
    return reviewed_identifier == identifier


def manual_action_items(bundle: Mapping[str, Any]) -> list[str]:
    """Return unresolved metadata and approval actions."""

    config = bundle["config"]
    metadata = _validate_metadata_schema(config)
    project = config.get("project", {})
    release = config.get("release", {})
    items: list[str] = []
    if project.get("title") != MANUSCRIPT_TITLE:
        items.append("Confirm the exact manuscript title.")
    if not str(project.get("version", "")).strip():
        items.append("Confirm the semantic release version.")
    creators = metadata.get("creators", [])
    if not isinstance(creators, list) or not creators:
        items.append("Provide and approve the ordered Zenodo creator list.")
    licenses = metadata.get("licenses", {})
    for key, label in (
        ("source_code", "repository source-code"),
        ("derived_data", "derived-data"),
        ("documentation", "documentation"),
    ):
        value = licenses.get(key) if isinstance(licenses, Mapping) else None
        if key == "derived_data" and metadata.get("dataset_rights"):
            if release.get("test_only") is not True:
                if not _derived_data_rights_review_is_complete(bundle):
                    items.append(MIXED_RIGHTS_ACTION)
            continue
        if isinstance(value, Mapping):
            value = value.get("identifier")
        if not str(value or "").strip():
            items.append(f"Confirm the {label} license.")
    if not str(metadata.get("description", "")).strip():
        items.append("Approve the Zenodo dataset description.")
    if release.get("version_confirmed") is not True:
        items.append("Confirm that the configured release version is final.")
    manual_approval_config = bundle["manual_approvals"]
    approvals = manual_approval_config.get("approvals", {})
    if isinstance(approvals, Mapping) and isinstance(
        approvals.get("datasets"), Mapping
    ):
        approvals = approvals["datasets"]
    if isinstance(approvals, Mapping):
        for dataset, approval in sorted(approvals.items()):
            if (
                isinstance(approval, Mapping)
                and approval.get("required")
                and not _review_record_is_complete(
                    approval, manual_approval_config
                )
            ):
                policy = bundle.get("dataset_policy", {}).get("datasets", {}).get(
                    dataset, {}
                )
                if (
                    isinstance(policy, Mapping)
                    and policy.get("participant_level_release") == "allowed"
                ):
                    items.append(
                        f"Bind the confirmed {dataset} permission to the reviewed "
                        "safe export and complete its artifact-level attestation."
                    )
                else:
                    items.append(
                        f"Confirm the public exclusion and reconstruction scope "
                        f"for {dataset} in its human review record."
                    )
    all_approvals = manual_approval_config.get("approvals", {})
    if isinstance(all_approvals, Mapping):
        for group_name in ("licenses", "zenodo_metadata", "release_artifacts"):
            group = all_approvals.get(group_name)
            if not isinstance(group, Mapping):
                continue
            for record_name, record in sorted(group.items()):
                if not _review_record_is_complete(record, manual_approval_config):
                    if (
                        group_name == "licenses"
                        and record_name == "derived_data"
                        and metadata.get("dataset_rights")
                    ):
                        items.append(MIXED_RIGHTS_ACTION)
                    else:
                        items.append(
                            f"Complete the {group_name}.{record_name} "
                            "approval record."
                        )
    publication_gate = manual_approval_config.get("publication_gate")
    if isinstance(publication_gate, Mapping) and (
        not _review_record_is_complete(publication_gate, manual_approval_config)
        or publication_gate.get("publish_ready") is not True
    ):
        items.append("Complete the explicit final publication gate approval.")
    if release.get("test_only") is not True:
        protocol = manual_approval_config.get("approval_protocol")
        if (
            not isinstance(protocol, Mapping)
            or protocol.get("deliberate_confirmation_required") is not True
            or not str(protocol.get("required_confirmation_text", "")).strip()
        ):
            items.append("Restore the required deliberate manual-approval protocol.")
        required_groups = {
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
        approval_groups = (
            all_approvals if isinstance(all_approvals, Mapping) else {}
        )
        dataset_group = approval_groups.get("datasets")
        required_datasets = {
            "abide",
            "adni",
            "oasis3",
            "camcan",
            "cobre",
            "adnidod",
            "1000brains",
        }
        if not isinstance(dataset_group, Mapping):
            items.append("Restore the required dataset approval group.")
        else:
            for dataset in sorted(required_datasets):
                record = dataset_group.get(dataset)
                if (
                    not isinstance(record, Mapping)
                    or record.get("required") is not True
                    or not _review_record_is_complete(record, manual_approval_config)
                ):
                    policy = (
                        bundle.get("dataset_policy", {})
                        .get("datasets", {})
                        .get(dataset, {})
                    )
                    if (
                        isinstance(policy, Mapping)
                        and policy.get("participant_level_release") == "allowed"
                    ):
                        items.append(
                            f"Bind the confirmed {dataset} permission to the "
                            "reviewed safe export and complete its artifact-level "
                            "attestation."
                        )
                    else:
                        items.append(
                            f"Confirm the public exclusion and reconstruction "
                            f"scope for {dataset} in its human review record."
                        )
        for group_name, required_records in required_groups.items():
            group = approval_groups.get(group_name)
            if not isinstance(group, Mapping):
                items.append(f"Restore the required {group_name} approval group.")
                continue
            for record_name in sorted(required_records - set(group)):
                items.append(
                    f"Restore the required {group_name}.{record_name} approval record."
                )
        exact_review_scopes = {
            ("zenodo_metadata", "creators_and_order"): "zenodo_creators_and_order",
            (
                "zenodo_metadata",
                "affiliations_and_orcids",
            ): "zenodo_affiliations_and_orcids",
            ("zenodo_metadata", "funding"): "zenodo_funding",
            (
                "zenodo_metadata",
                "related_identifiers_and_manuscript_doi",
            ): "zenodo_related_identifiers_and_manuscript_doi",
            (
                "release_artifacts",
                "aggregate_results_content_and_pdf_metadata_review",
            ): "aggregate_outputs",
            (
                "release_artifacts",
                "privacy_scan_review",
            ): "release_privacy_scan",
            (
                "release_artifacts",
                "manifest_and_checksum_review",
            ): "release_manifest_and_checksums",
            (
                "release_artifacts",
                "final_zenodo_form_review",
            ): "zenodo_final_form",
        }
        for (group_name, record_name), expected_scope in exact_review_scopes.items():
            group = approval_groups.get(group_name, {})
            record = (
                group.get(record_name, {}) if isinstance(group, Mapping) else {}
            )
            if (
                isinstance(record, Mapping)
                and _review_record_is_complete(record, manual_approval_config)
                and record.get("scope") != [expected_scope]
            ):
                items.append(
                    f"Bind {group_name}.{record_name} to its exact review scope."
                )
        dataset_scope_names = {
            "participant_connectomes",
            "participant_metadata",
            "exact_splits",
            "aggregate_metrics",
            "statistical_summaries",
            "figure_source_data",
            "configuration",
            "processing_script",
            "reconstruction_instructions",
        }
        participant_scopes = {
            "participant_connectomes",
            "participant_metadata",
            "exact_splits",
        }
        if isinstance(dataset_group, Mapping):
            for dataset, record in dataset_group.items():
                scopes = record.get("scope", []) if isinstance(record, Mapping) else []
                if (
                    not isinstance(record, Mapping)
                    or str(record.get("dataset", "")).strip().lower() != dataset
                    or not isinstance(scopes, list)
                    or not set(scopes).issubset(dataset_scope_names)
                    or (
                        dataset in {"adni", "adnidod", "oasis3", "1000brains"}
                        and bool(set(scopes) & participant_scopes)
                    )
                ):
                    items.append(
                        f"Bind the {dataset} approval to its dataset and permitted scope."
                    )
        license_group = approval_groups.get("licenses", {})
        configured_licenses = (
            metadata.get("licenses", {})
            if isinstance(metadata.get("licenses", {}), Mapping)
            else {}
        )
        expected_license_scopes = {
            "source_code": "repository_source_code",
            "derived_data": "approved_derived_data_artifacts",
            "documentation": "release_documentation",
        }
        for license_name, expected_scope in expected_license_scopes.items():
            if license_name == "derived_data" and metadata.get("dataset_rights"):
                if not _derived_data_rights_review_is_complete(bundle):
                    items.append(MIXED_RIGHTS_ACTION)
                continue
            configured = configured_licenses.get(license_name)
            record = (
                license_group.get(license_name, {})
                if isinstance(license_group, Mapping)
                else {}
            )
            identifier = (
                configured.get("identifier")
                if isinstance(configured, Mapping)
                else None
            )
            record_identifier_key = (
                "documented_identifier"
                if license_name == "source_code"
                else "license_identifier"
            )
            scopes = record.get("scope", []) if isinstance(record, Mapping) else []
            if (
                not str(identifier or "").strip()
                or not isinstance(record, Mapping)
                or record.get(record_identifier_key) != identifier
                or not isinstance(scopes, list)
                or scopes != [expected_scope]
            ):
                items.append(
                    f"Bind the {license_name} approval to its exact identifier and artifact scope."
                )
        if not isinstance(publication_gate, Mapping):
            items.append("Restore the explicit final publication gate.")
        elif (
            _review_record_is_complete(
                publication_gate, manual_approval_config
            )
            and publication_gate.get("scope") != ["zenodo_publication"]
        ):
            items.append(
                "Bind the final publication gate to the exact zenodo_publication scope."
            )
    explicit_actions = bundle["manual_approvals"].get(
        "manual_actions",
        bundle["manual_approvals"].get("manual_action_items", []),
    )
    if isinstance(explicit_actions, list):
        items.extend(str(item) for item in explicit_actions if str(item).strip())
    return list(dict.fromkeys(items))


def metadata_is_complete(bundle: Mapping[str, Any]) -> tuple[bool, list[str]]:
    config = bundle["config"]
    metadata = _validate_metadata_schema(config)
    problems = manual_action_items(bundle)
    if contains_placeholder(
        {
            "project": config.get("project", {}),
            "release": config.get("release", {}),
            "metadata": metadata,
            "manual_approvals": bundle.get("manual_approvals", {}),
        }
    ):
        problems.append(
            "Remove placeholder values from project, approval, and Zenodo metadata."
        )
    licenses = metadata.get("licenses", {})
    if isinstance(licenses, Mapping):
        for key, value in licenses.items():
            if key == "derived_data" and metadata.get("dataset_rights"):
                if config.get("release", {}).get("test_only") is not True:
                    if not _derived_data_rights_review_is_complete(bundle):
                        problems.append(MIXED_RIGHTS_ACTION)
                continue
            if isinstance(value, Mapping) and value.get("status") not in {
                "approved",
                "documented_in_repository",
            }:
                problems.append(f"Complete the {key} license approval status.")
            if (
                config.get("release", {}).get("test_only") is not True
                and not isinstance(value, Mapping)
            ):
                problems.append(
                    f"Use an explicitly reviewed identifier/status mapping for {key}."
                )
    creators = metadata.get("creators", [])
    if isinstance(creators, list):
        for index, creator in enumerate(creators, start=1):
            if not isinstance(creator, Mapping) or not str(
                creator.get("name", "")
            ).strip():
                problems.append(f"Creator {index} is missing a name.")
    return not problems, list(dict.fromkeys(problems))


def build_citation_cff(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact CFF structure written into the release."""

    config = bundle["config"]
    metadata = _validate_metadata_schema(config)
    project = config.get("project", {})
    creators = metadata.get("creators", [])
    authors = []
    if isinstance(creators, list):
        for creator in creators:
            if not isinstance(creator, Mapping) or not creator.get("name"):
                continue
            name = str(creator["name"])
            parts = name.rsplit(",", maxsplit=1)
            if len(parts) == 2:
                family, given = (part.strip() for part in parts)
            else:
                name_parts = name.rsplit(" ", maxsplit=1)
                given, family = (
                    (name_parts[0], name_parts[1])
                    if len(name_parts) == 2
                    else ("", name_parts[0])
                )
            entry: dict[str, str] = {"family-names": family}
            if given:
                entry["given-names"] = given
            for source, target in (("orcid", "orcid"), ("affiliation", "affiliation")):
                if creator.get(source):
                    entry[target] = str(creator[source])
            authors.append(entry)
    return {
        "cff-version": "1.2.0",
        "message": "If you use this dataset release, cite the accompanying record.",
        "title": str(project.get("title", "")),
        "type": "dataset",
        "version": str(project.get("version", "")),
        "authors": authors,
    }


def build_zenodo_record_metadata(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Create conservative Zenodo record metadata from explicit configuration."""

    config = bundle["config"]
    project = config.get("project", {})
    metadata = _validate_metadata_schema(config)
    base_description = str(metadata.get("description", "")).strip()
    rights_description = dataset_rights_text(config)
    record: dict[str, Any] = {
        "upload_type": "dataset",
        "title": str(project.get("title", "")),
        "version": str(project.get("version", "")),
        "description": "\n\n".join(
            value for value in (base_description, rights_description) if value
        ),
        "creators": [],
    }
    creators = metadata.get("creators", [])
    if isinstance(creators, list):
        for creator in creators:
            if not isinstance(creator, Mapping):
                continue
            clean = {
                key: str(creator[key])
                for key in ("name", "affiliation", "orcid")
                if str(creator.get(key, "")).strip()
            }
            if clean:
                record["creators"].append(clean)
    keywords = metadata.get("keywords", [])
    if isinstance(keywords, list) and keywords:
        record["keywords"] = [str(value) for value in keywords]
    licenses = metadata.get("licenses", {})
    if isinstance(licenses, Mapping):
        derived_license = licenses.get("derived_data")
        if isinstance(derived_license, Mapping):
            derived_license = derived_license.get("identifier")
        if str(derived_license or "").strip():
            record["license"] = str(derived_license)
    related = metadata.get("related_identifiers", [])
    if isinstance(related, list) and related:
        record["related_identifiers"] = [dict(value) for value in related]
    manuscript_doi = metadata.get("manuscript_doi")
    if manuscript_doi:
        manuscript_record = {
            "identifier": str(manuscript_doi),
            "relation": "isSupplementTo",
            "resource_type": "publication",
        }
        existing = record.setdefault("related_identifiers", [])
        if not any(
            value.get("identifier") == manuscript_record["identifier"]
            for value in existing
        ):
            existing.append(manuscript_record)
    grants = metadata.get("funding", [])
    if isinstance(grants, list) and grants:
        record["grants"] = [dict(value) for value in grants]
    communities = metadata.get("communities", [])
    if isinstance(communities, list) and communities:
        record["communities"] = [
            {"identifier": str(value)}
            for value in communities
        ]
    return record


def write_metadata_files(
    bundle: Mapping[str, Any], metadata_dir: str | Path
) -> tuple[Path, Path]:
    destination = Path(metadata_dir)
    destination.mkdir(parents=True, exist_ok=True)
    record = build_zenodo_record_metadata(bundle)
    json_path = destination / "zenodo_record_metadata.json"
    json_path.write_text(
        json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    form_path = destination / "zenodo_form_values.md"
    creators = record.get("creators", [])
    creator_lines = (
        "\n".join(
            "- "
            + "; ".join(
                f"{key}: {creator[key]}"
                for key in ("name", "affiliation", "orcid")
                if creator.get(key)
            )
            for creator in creators
        )
        if creators
        else "- Manual entry required before publication"
    )
    keywords = record.get("keywords", [])
    keyword_lines = (
        "\n".join(f"- {value}" for value in keywords)
        if keywords
        else "- None explicitly reviewed."
    )
    related = record.get("related_identifiers", [])
    related_lines = (
        "\n".join(
            f"- {value['identifier']} ({value['relation']}, {value['resource_type']})"
            for value in related
        )
        if related
        else "- None explicitly reviewed."
    )
    grants = record.get("grants", [])
    grant_lines = (
        "\n".join(f"- {value['id']}" for value in grants)
        if grants
        else "- None explicitly reviewed."
    )
    communities = record.get("communities", [])
    community_lines = (
        "\n".join(f"- {value['identifier']}" for value in communities)
        if communities
        else "- None explicitly reviewed."
    )
    config = bundle["config"]
    release = config.get("release", {})
    if release.get("test_only") is True:
        document_banner = (
            "> **SYNTHETIC TEST METADATA — DO NOT UPLOAD OR PUBLISH.**"
        )
    elif (
        release.get("publication_ready") is not True
        or manual_action_items(bundle)
    ):
        document_banner = (
            "> **REAL-DATA DRAFT — NOT APPROVED FOR UPLOAD OR PUBLICATION.**"
        )
    else:
        document_banner = ""
    base_description = str(_metadata(config).get("description", "")).strip()
    rights_markdown = _dataset_rights_markdown(config)
    form_path.write_text(
        "\n".join(
            [
                "# Zenodo form values",
                "",
                *([document_banner, ""] if document_banner else []),
                "This file is a review aid. Uploading the JSON companion does not",
                "guarantee that the Zenodo web form will populate every field.",
                "",
                f"- Resource type: `{record['upload_type']}`",
                f"- Title: {record['title']}",
                f"- Version: `{record['version']}`",
                "",
                "## Creators",
                "",
                creator_lines,
                "",
                "## Description",
                "",
                base_description or "Manual entry required before publication.",
                "",
                "## Dataset-specific rights",
                "",
                rights_markdown,
                "",
                "## License",
                "",
                str(
                    record.get("license")
                    or (
                        "Record-level mixed-rights selection requires final "
                        "manual review; it must not override the dataset-specific "
                        "terms below."
                    )
                ),
                "",
                "## Keywords",
                "",
                keyword_lines,
                "",
                "## Related identifiers",
                "",
                related_lines,
                "",
                "## Funding",
                "",
                grant_lines,
                "",
                "## Communities",
                "",
                community_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, form_path


def write_citation_cff(bundle: Mapping[str, Any], path: str | Path) -> Path:
    citation = build_citation_cff(bundle)
    destination = Path(path)
    destination.write_text(
        yaml.safe_dump(citation, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return destination
