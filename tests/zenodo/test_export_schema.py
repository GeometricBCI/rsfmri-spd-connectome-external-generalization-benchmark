from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from tools.zenodo.export_internal import (
    TrustedExportError,
    _load_internal_manifest,
    _source_binding,
    export_internal,
)
from tools.zenodo.schemas import (
    SAFE_UID_RE,
    allowed_metadata_columns,
    load_release_bundle,
    metadata_column_is_forbidden,
    participant_release_decision,
    validate_connectome_npz,
    validate_metadata_columns,
    validate_release_bundle_documents,
)

from ._synthetic import SYNTHETIC_SAMPLE_UIDS, correlation_matrices


def _finding_codes(findings) -> set[str]:
    return {finding.code for finding in findings}


def test_release_bundle_resolves_all_policy_files(release_config_factory):
    config_path = release_config_factory()

    bundle = load_release_bundle(config_path)

    assert bundle["config_path"] == config_path.resolve()
    for name in (
        "dataset_policy",
        "metadata_allowlist",
        "forbidden_patterns",
        "manual_approvals",
    ):
        assert bundle[name]["schema_version"] == 1
        assert bundle[f"{name}_path"].is_file()


def test_formal_reviewer_designation_does_not_open_an_approval_gate():
    repository_root = Path(__file__).resolve().parents[2]
    bundle = load_release_bundle(
        repository_root / "configs" / "release" / "zenodo_release.yaml"
    )

    protocol = bundle["manual_approvals"]["approval_protocol"]
    assert protocol["designated_reviewer"] == "Ce Ju"
    assert protocol["designation_confirmed_on"] == "2026-07-27"
    assert participant_release_decision(
        bundle, "camcan", "participant_connectomes"
    ) == (False, "manual_approval_incomplete")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("designated_reviewer", ""),
        ("designation_confirmed_on", "27-07-2026"),
    ],
)
def test_reviewer_designation_must_be_complete_and_valid(
    field,
    value,
):
    repository_root = Path(__file__).resolve().parents[2]
    bundle = load_release_bundle(
        repository_root / "configs" / "release" / "zenodo_release.yaml"
    )
    protocol = bundle["manual_approvals"]["approval_protocol"]
    protocol["designated_reviewer"] = "Synthetic Reviewer"
    protocol["designation_confirmed_on"] = "2026-07-27"
    protocol[field] = value

    with pytest.raises(ValueError, match="designated reviewer"):
        validate_release_bundle_documents(bundle, snapshot=False)


@pytest.mark.parametrize(
    ("section", "field", "unsafe_value"),
    [
        ("release", "archive_format", "tar"),
        ("release", "positive_definite_required", False),
        ("release", "positive_definite_required", 1),
        ("release", "numeric_dtype_allowlist", ["float64"]),
        ("release", "diagonal_expected", 0.0),
        (
            "release",
            "normalized_archive_timestamp",
            "2025-01-01T00:00:00Z",
        ),
        ("release", "publication_gate", "automatic"),
        (
            "safe_export_contract",
            "input_must_be_designated_directory",
            False,
        ),
        ("safe_export_contract", "input_manifest_required", False),
        ("safe_export_contract", "unknown_files", "allow"),
        ("safe_export_contract", "unknown_fields", "allow"),
        ("safe_export_contract", "symbolic_links", "allow"),
        ("safe_export_contract", "copy_original_filenames", True),
        ("safe_export_contract", "copy_source_paths", True),
        (
            "safe_export_contract",
            "hashed_original_identifiers",
            "allowed",
        ),
        (
            "safe_export_contract",
            "sample_uid_generation",
            "copy_source_identifier",
        ),
        ("safe_export_contract", "npz_allow_pickle", True),
        (
            "safe_export_contract",
            "finite_numeric_arrays_required",
            False,
        ),
        (
            "publication_requirements",
            "require_complete_manual_approvals",
            False,
        ),
        (
            "publication_requirements",
            "require_complete_required_metadata",
            False,
        ),
        (
            "publication_requirements",
            "require_dataset_specific_rights_statements",
            False,
        ),
        (
            "publication_requirements",
            "require_mixed_record_license_review",
            False,
        ),
        (
            "publication_requirements",
            "require_explicit_documentation_license",
            False,
        ),
        (
            "publication_requirements",
            "require_clean_validation_report",
            False,
        ),
        ("publication_requirements", "require_manifest_match", False),
        ("publication_requirements", "require_checksum_match", False),
        (
            "publication_requirements",
            "require_revalidated_archive",
            False,
        ),
        ("publication_requirements", "allow_zenodo_api", True),
        ("publication_requirements", "allow_token_access", True),
        ("publication_requirements", "allow_doi_reservation", True),
    ],
)
def test_release_bundle_rejects_weakened_safety_contracts(
    release_config_factory,
    section,
    field,
    unsafe_value,
):
    config_path = release_config_factory()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config[section][field] = unsafe_value
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical fail-closed value"):
        load_release_bundle(config_path)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("release", "archive_format"),
        ("safe_export_contract", "unknown_files"),
        ("publication_requirements", "require_checksum_match"),
    ],
)
def test_release_bundle_requires_complete_safety_contracts(
    release_config_factory,
    section,
    field,
):
    config_path = release_config_factory()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    del config[section][field]
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required safety fields"):
        load_release_bundle(config_path)


def test_release_bundle_fails_closed_when_policy_is_missing(
    release_config_factory,
):
    config_path = release_config_factory()
    config = config_path.read_text(encoding="utf-8")
    config_path.write_text(
        config.replace("  metadata_allowlist: metadata_allowlist.yaml\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing policy references"):
        load_release_bundle(config_path)


def test_participant_release_requires_complete_scoped_approval(
    release_config_factory,
):
    approved = load_release_bundle(release_config_factory(approved=True))
    pending = load_release_bundle(release_config_factory(approved=False))

    assert participant_release_decision(
        approved, "ABIDE", "participant_connectomes"
    ) == (
        True,
        "explicit_confirmed_permission_and_source_bound_approval",
    )
    assert participant_release_decision(
        pending, "abide", "participant_connectomes"
    ) == (False, "manual_approval_incomplete")
    assert participant_release_decision(
        approved, "abide", "not_an_approved_scope"
    ) == (False, "content_scope_not_explicitly_releasable")


def test_unknown_dataset_is_forbidden_even_with_an_approval_entry(
    release_config_factory,
):
    config_path = release_config_factory()
    bundle = load_release_bundle(config_path)
    bundle["manual_approvals"]["approvals"]["mystery_cohort"] = {
        "status": "approved",
        "approved_by": "Synthetic Test Authority",
        "approved_on": "2000-01-01",
        "scope": ["participant_connectomes"],
        "evidence": "synthetic-test-policy",
    }

    assert participant_release_decision(
        bundle, "mystery_cohort", "participant_connectomes"
    ) == (False, "unknown_dataset_forbidden")


def test_confirmed_policy_forbids_exact_split_membership(
    release_config_factory,
):
    bundle = load_release_bundle(release_config_factory())

    assert participant_release_decision(
        bundle, "abide", "participant_connectomes"
    ) == (
        True,
        "explicit_confirmed_permission_and_source_bound_approval",
    )
    assert participant_release_decision(
        bundle, "abide", "participant_metadata"
    ) == (
        True,
        "explicit_confirmed_permission_and_source_bound_approval",
    )
    assert participant_release_decision(
        bundle, "abide", "exact_splits"
    ) == (False, "content_scope_not_explicitly_releasable")


def test_formal_camcan_policy_forbids_exact_splits_and_ccid():
    repository_root = Path(__file__).resolve().parents[2]
    bundle = load_release_bundle(
        repository_root / "configs" / "release" / "zenodo_release.yaml"
    )

    policy = bundle["dataset_policy"]["datasets"]["camcan"]
    authorization = policy["participant_level_connectomes"]

    assert policy["participant_level_release"] == "allowed"
    assert authorization == {
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
    }
    assert policy["decisions"]["exact_split_membership"] == "forbidden"
    assert participant_release_decision(
        bundle, "camcan", "participant_connectomes"
    ) == (False, "manual_approval_incomplete")
    assert participant_release_decision(
        bundle, "camcan", "participant_metadata"
    ) == (False, "manual_approval_incomplete")
    assert participant_release_decision(
        bundle, "camcan", "exact_splits"
    ) == (False, "content_scope_not_explicitly_releasable")
    assert metadata_column_is_forbidden("CCID", bundle=bundle)


def test_source_bound_camcan_confirmation_opens_only_connectomes_and_metadata(
    release_config_factory,
):
    bundle = load_release_bundle(
        release_config_factory(dataset="camcan", approved=True)
    )

    assert allowed_metadata_columns(bundle, "camcan") == {
        "sample_uid",
        "age",
        "sex",
    }
    for scope in ("participant_connectomes", "participant_metadata"):
        assert participant_release_decision(bundle, "camcan", scope) == (
            True,
            "explicit_confirmed_permission_and_source_bound_approval",
        )
    assert participant_release_decision(
        bundle, "camcan", "exact_splits"
    ) == (False, "content_scope_not_explicitly_releasable")


def test_camcan_approval_license_must_match_written_authorization(
    release_config_factory,
):
    bundle = load_release_bundle(
        release_config_factory(dataset="camcan", approved=True)
    )
    bundle["manual_approvals"]["approvals"]["camcan"][
        "license_identifier"
    ] = "CC0-1.0"

    for scope in ("participant_connectomes", "participant_metadata"):
        assert participant_release_decision(bundle, "camcan", scope) == (
            False,
            "approval_license_mismatch",
        )


def test_camcan_allowed_policy_cannot_omit_written_authorization(
    release_config_factory,
):
    bundle = load_release_bundle(
        release_config_factory(dataset="camcan", approved=True)
    )
    del bundle["dataset_policy"]["datasets"]["camcan"][
        "participant_level_connectomes"
    ]

    with pytest.raises(ValueError, match="confirmed.*authorization"):
        validate_release_bundle_documents(bundle, snapshot=False)


def test_release_bundle_rejects_a_fabricated_repository_url(
    release_config_factory,
):
    bundle = load_release_bundle(release_config_factory())
    bundle["config"]["project"][
        "repository_url"
    ] = "https://example.invalid/fabricated"

    with pytest.raises(ValueError, match="repository_url is not canonical"):
        validate_release_bundle_documents(bundle, snapshot=False)


def test_manual_approval_with_empty_scope_never_opens_a_gate(
    release_config_factory,
):
    bundle = load_release_bundle(release_config_factory())
    bundle["manual_approvals"]["approvals"]["abide"]["scope"] = []

    for scope in ("participant_connectomes", "participant_metadata"):
        assert participant_release_decision(bundle, "abide", scope) == (
            False,
            "manual_approval_incomplete",
        )
    assert participant_release_decision(
        bundle, "abide", "exact_splits"
    ) == (False, "content_scope_not_explicitly_releasable")


def test_metadata_allowlist_never_overrides_forbidden_identity_columns(
    release_config_factory,
):
    bundle = load_release_bundle(release_config_factory())
    bundle["metadata_allowlist"]["datasets"]["abide"][
        "public_metadata_columns"
    ].append("subject_id")

    findings = validate_metadata_columns(
        ["sample_uid", "age", "subject_id", "unreviewed_measure"],
        bundle=bundle,
        dataset="abide",
    )

    assert allowed_metadata_columns(bundle, "abide") == {
        "sample_uid",
        "subject_id",
    }
    assert _finding_codes(findings) == {
        "FORBIDDEN_METADATA_COLUMN",
        "UNAPPROVED_METADATA_COLUMN",
    }


def test_synthetic_sample_uids_match_the_public_uid_contract():
    assert all(SAFE_UID_RE.fullmatch(value) for value in SYNTHETIC_SAMPLE_UIDS)
    assert SAFE_UID_RE.fullmatch("cc110033") is None


def test_connectome_npz_accepts_only_safe_float_spd_correlations(tmp_path):
    path = tmp_path / "connectomes.npz"
    matrices = correlation_matrices()
    np.savez(path, connectomes=matrices)

    count, findings = validate_connectome_npz(
        path,
        matrix_type="correlation",
        symmetry_tolerance=1e-8,
        diagonal_tolerance=1e-8,
    )

    assert count == matrices.shape[0]
    assert findings == []


def test_connectome_npz_rejects_object_arrays_without_pickle_loading(tmp_path):
    path = tmp_path / "connectomes.npz"
    values = np.empty((1,), dtype=object)
    values[0] = {"synthetic": True}
    np.savez(path, connectomes=values)

    count, findings = validate_connectome_npz(path)

    assert count is None
    assert "OBJECT_ARRAY" in _finding_codes(findings)


def test_connectome_npz_rejects_duplicate_member_hiding_object_payload(tmp_path):
    path = tmp_path / "connectomes.npz"
    object_payload = io.BytesIO()
    np.lib.format.write_array(
        object_payload,
        np.asarray([{"synthetic": True}], dtype=object),
        allow_pickle=True,
    )
    safe_payload = io.BytesIO()
    np.lib.format.write_array(
        safe_payload,
        correlation_matrices(n_samples=1),
        allow_pickle=False,
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("connectomes.npy", object_payload.getvalue())
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("connectomes.npy", safe_payload.getvalue())

    count, findings = validate_connectome_npz(path)

    assert count is None
    assert "NPZ_CONTAINER_SCHEMA" in _finding_codes(findings)


def test_connectome_npz_rejects_trailing_concatenated_object_payload(tmp_path):
    path = tmp_path / "connectomes.npz"
    safe_payload = io.BytesIO()
    np.lib.format.write_array(
        safe_payload,
        correlation_matrices(n_samples=1),
        allow_pickle=False,
    )
    object_payload = io.BytesIO()
    np.lib.format.write_array(
        object_payload,
        np.asarray([{"synthetic": True}], dtype=object),
        allow_pickle=True,
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "connectomes.npy",
            safe_payload.getvalue() + object_payload.getvalue(),
        )

    count, findings = validate_connectome_npz(path)

    assert count is None
    assert "NPY_TRAILING_DATA" in _finding_codes(findings)


def test_connectome_npz_rejects_float16_with_a_finding_not_an_exception(tmp_path):
    path = tmp_path / "connectomes.npz"
    np.savez(
        path,
        connectomes=correlation_matrices(n_samples=1).astype(np.float16),
    )

    count, findings = validate_connectome_npz(path)

    assert count == 1
    assert "NPZ_DTYPE" in _finding_codes(findings)


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (
            lambda matrices: matrices.__setitem__((0, 0, 1), 0.91),
            "ASYMMETRIC_CONNECTOME",
        ),
        (
            lambda matrices: matrices.__setitem__((0, 0, 0), 0.5),
            "CORRELATION_DIAGONAL",
        ),
        (
            lambda matrices: matrices.__setitem__((0, 0, 0), -1.0),
            "NON_SPD_CONNECTOME",
        ),
    ],
)
def test_connectome_npz_rejects_invalid_matrix_contract(
    tmp_path,
    mutator,
    expected_code,
):
    path = tmp_path / "connectomes.npz"
    matrices = correlation_matrices()
    mutator(matrices)
    np.savez(path, connectomes=matrices)

    _, findings = validate_connectome_npz(path)

    assert expected_code in _finding_codes(findings)


def test_connectome_npz_rejects_unexpected_members(tmp_path):
    path = tmp_path / "connectomes.npz"
    np.savez(
        path,
        connectomes=correlation_matrices(),
        participant_ids=np.asarray(["synthetic-placeholder"]),
    )

    _, findings = validate_connectome_npz(path)

    assert "NPZ_CONTAINER_SCHEMA" in _finding_codes(findings)


def test_release_bundle_rejects_non_mapping_yaml(release_config_factory):
    config_path = release_config_factory()
    config_path.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_release_bundle(config_path)


def test_internal_manifest_rejects_reserved_sample_uid_mapping(
    internal_manifest_factory,
):
    manifest_path = internal_manifest_factory(
        schema_overrides={
            "metadata_columns": {"sample_uid": "SubjectID"},
        }
    )

    with pytest.raises(TrustedExportError, match="reserved sample_uid"):
        _load_internal_manifest(manifest_path)


@pytest.mark.parametrize("matrix_type", ["covariance", "spd"])
def test_time_series_export_rejects_mislabeled_matrix_types(
    internal_manifest_factory,
    matrix_type,
):
    manifest_path = internal_manifest_factory(
        schema_overrides={
            "connectome_column": None,
            "timeseries_column": "TimeSeries",
            "matrix_type": matrix_type,
        }
    )

    with pytest.raises(
        TrustedExportError,
        match="time-series export currently supports only correlation",
    ):
        _load_internal_manifest(manifest_path)


def test_time_series_export_accepts_the_implemented_correlation_contract(
    internal_manifest_factory,
):
    manifest_path = internal_manifest_factory(
        schema_overrides={
            "connectome_column": None,
            "timeseries_column": "TimeSeries",
            "matrix_type": "correlation",
        }
    )

    manifest = _load_internal_manifest(manifest_path)

    assert manifest["datasets"][0]["schema"]["matrix_type"] == "correlation"


def test_export_dry_run_rejects_reserved_uid_before_deserialization(
    tmp_path,
    release_config_factory,
    internal_manifest_factory,
    monkeypatch,
):
    config_path = release_config_factory()
    manifest_path = internal_manifest_factory(
        schema_overrides={
            "metadata_columns": {"sample_uid": "SubjectID"},
        }
    )
    output_dir = tmp_path / "safe-export"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("invalid manifest must fail before pickle loading")

    monkeypatch.setattr(
        "tools.zenodo.export_internal._load_trusted_dataframe",
        fail_if_called,
    )

    with pytest.raises(TrustedExportError, match="reserved sample_uid"):
        export_internal(
            config_path,
            manifest_path,
            output_dir,
            trusted_internal_input=True,
            dry_run=True,
        )

    assert not output_dir.exists()


@pytest.mark.parametrize("conflicting_source", ["Connectome", "Fold", "Partition"])
def test_internal_manifest_rejects_metadata_source_operational_overlap(
    internal_manifest_factory,
    conflicting_source,
):
    manifest_path = internal_manifest_factory(
        schema_overrides={
            "metadata_columns": {"age": conflicting_source},
            "fold_column": "Fold",
            "partition_column": "Partition",
        }
    )

    with pytest.raises(TrustedExportError, match="source columns.*distinct"):
        _load_internal_manifest(manifest_path)


def test_stage1_requires_explicit_trusted_input_gate_before_manifest_access(
    tmp_path,
    release_config_factory,
    internal_manifest_factory,
    monkeypatch,
):
    config_path = release_config_factory()
    manifest_path = internal_manifest_factory()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("manifest must not be loaded before the trust gate")

    monkeypatch.setattr(
        "tools.zenodo.export_internal._load_internal_manifest",
        fail_if_called,
    )

    with pytest.raises(TrustedExportError, match="--trusted-internal-input"):
        export_internal(
            config_path,
            manifest_path,
            tmp_path / "safe-export",
            trusted_internal_input=False,
            dry_run=True,
        )


def test_stage1_dry_run_never_deserializes_or_writes(
    tmp_path,
    release_config_factory,
    internal_manifest_factory,
    monkeypatch,
    capsys,
):
    manifest_path = internal_manifest_factory()
    manifest = _load_internal_manifest(manifest_path)
    config_path = release_config_factory(
        source_binding_sha256=_source_binding(
            manifest["datasets"][0], manifest["export_namespace"]
        )
    )
    output_dir = tmp_path / "safe-export"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run must never deserialize pickle")

    monkeypatch.setattr(
        "tools.zenodo.export_internal._load_trusted_dataframe",
        fail_if_called,
    )

    destination = export_internal(
        config_path,
        manifest_path,
        output_dir,
        trusted_internal_input=True,
        dry_run=True,
    )
    report = json.loads(capsys.readouterr().out)

    assert destination == output_dir.resolve()
    assert report["dry_run"] is True
    assert report["would_deserialize_pickle"] is False
    assert report["would_succeed"] is True
    assert report["dataset_count"] == 1
    assert not output_dir.exists()


def test_stage1_uses_reviewed_release_tolerances_for_float32_connectomes(
    tmp_path,
    release_config_factory,
    internal_manifest_factory,
    monkeypatch,
):
    import pandas as pd

    manifest_path = internal_manifest_factory()
    manifest = _load_internal_manifest(manifest_path)
    config_path = release_config_factory(
        source_binding_sha256=_source_binding(
            manifest["datasets"][0], manifest["export_namespace"]
        )
    )
    config_document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_document["release"]["diagonal_tolerance"] = 1.0e-4
    config_path.write_text(
        yaml.safe_dump(config_document, sort_keys=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "safe-export"
    matrix = np.eye(100, dtype=np.float32) * np.float32(1.00001)
    frame = pd.DataFrame(
        {
            "SubjectID": ["internal-only-id"],
            "Connectome": [matrix],
        }
    )

    monkeypatch.setattr(
        "tools.zenodo.export_internal._load_trusted_dataframe",
        lambda *_args, **_kwargs: frame,
    )

    destination = export_internal(
        config_path,
        manifest_path,
        output_dir,
        trusted_internal_input=True,
    )

    assert destination == output_dir.resolve()
    count, findings = validate_connectome_npz(
        destination / "datasets/abide/connectomes.npz",
        diagonal_tolerance=1.0e-4,
        expected_regions=100,
    )
    assert count == 1
    assert findings == []
