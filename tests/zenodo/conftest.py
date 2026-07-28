from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tools.zenodo.schemas import (
    CONFIRMED_DATASET_RIGHTS,
    CONFIRMED_PARTICIPANT_AUTHORIZATIONS,
)

from ._synthetic import (
    SYNTHETIC_DATASET,
    SYNTHETIC_SAMPLE_UIDS,
    SYNTHETIC_SOURCE_BINDING_SHA256,
    correlation_matrices,
    sha256_file,
    write_tsv,
    write_yaml,
)


SYNTHETIC_DATASET_RIGHTS = copy.deepcopy(CONFIRMED_DATASET_RIGHTS)


@pytest.fixture
def release_config_factory(tmp_path):
    """Create a complete fail-closed release-policy bundle."""

    def factory(
        *,
        approved: bool = True,
        dataset: str = SYNTHETIC_DATASET,
        test_only: bool = True,
        source_binding_sha256: str = SYNTHETIC_SOURCE_BINDING_SHA256,
        config_overrides: dict[str, object] | None = None,
    ) -> Path:
        root = tmp_path / f"release-config-{dataset}-{approved}-{test_only}"
        root.mkdir(parents=True, exist_ok=True)

        if dataset not in CONFIRMED_PARTICIPANT_AUTHORIZATIONS:
            raise ValueError("synthetic release fixtures support confirmed datasets")
        authorization = copy.deepcopy(
            CONFIRMED_PARTICIPANT_AUTHORIZATIONS[dataset]
        )
        policy_entry: dict[str, object] = {
            "participant_level_release": "allowed",
            "participant_level_connectomes": authorization,
            "decisions": {
                "participant_level_connectomes": (
                    "allowed_with_confirmed_terms_and_source_binding"
                ),
                "participant_level_metadata": (
                    "allowed_with_confirmed_terms_source_binding_and_allowlist"
                ),
                "exact_split_membership": "forbidden",
            },
        }
        metadata_columns = {
            "abide": ["sample_uid"],
            "camcan": ["sample_uid", "age", "sex"],
            "cobre": ["sample_uid"],
        }[dataset]
        exact_split_columns: list[str] = []
        approval_scope = [
            "participant_connectomes",
            "participant_metadata",
        ]

        write_yaml(
            root / "dataset_policy.yaml",
            {
                "schema_version": 1,
                "datasets": {dataset: policy_entry},
                "unknown_dataset": {
                    "participant_level_release": "forbidden"
                },
            },
        )
        write_yaml(
            root / "metadata_allowlist.yaml",
            {
                "schema_version": 1,
                "required_columns": ["sample_uid"],
                "tables": {
                    "exact_split_membership": {
                        "allowed_columns": [
                            "fold",
                            "partition",
                            "sample_uid",
                        ]
                    },
                    "aggregate_metrics": {
                        "allowed_columns": [
                            "dataset",
                            "metric",
                            "value",
                        ]
                    }
                },
                "datasets": {
                    dataset: {
                        "public_metadata_columns": metadata_columns,
                        "exact_split_columns": exact_split_columns,
                    }
                },
            },
        )
        write_yaml(
            root / "forbidden_patterns.yaml",
            {"schema_version": 1, "patterns": []},
        )
        approval = {
            "dataset": dataset,
            "atlas": "schaefer_100",
            "n_regions": 100,
            "status": (
                "approved"
                if approved
                else "permission_confirmed_artifact_binding_required"
            ),
            "approved_by": "Synthetic Test Authority" if approved else "",
            "approved_on": "2000-01-01" if approved else "",
            "scope": approval_scope,
            "evidence": "synthetic-test-policy" if approved else "",
            "source_binding_sha256": (
                source_binding_sha256 if approved else ""
            ),
        }
        approval["license_identifier"] = authorization["license"]["type"]
        write_yaml(
            root / "manual_approvals.yaml",
            {
                "schema_version": 1,
                "approvals": {dataset: approval},
            },
        )

        config: dict[str, object] = {
            "schema_version": 1,
            "project": {
                "title": (
                    "Benchmarking External Generalization of SPD Matrix "
                    "Learning for Resting-State fMRI Connectome Prediction"
                ),
                "version": "0.0.0-test",
                "resource_type": "dataset",
            },
            "release": {
                "archive_basename": (
                    "synthetic-rsfmri-spd-release"
                    if test_only
                    else "spd_connectome_benchmark_v0.0.0-test"
                ),
                "archive_format": "zip",
                "matrix_type": "correlation",
                "atlas": "schaefer_100",
                "expected_regions": 100,
                "positive_definite_required": True,
                "numeric_dtype_allowlist": ["float32", "float64"],
                "symmetry_tolerance": 1e-8,
                "diagonal_expected": 1.0,
                "diagonal_tolerance": 1e-8,
                "spd_eigenvalue_tolerance": 0.0,
                "normalized_archive_timestamp": "1980-01-01T00:00:00Z",
                "test_only": test_only,
                "version_confirmed": True,
                "publication_gate": "manual_required",
            },
            "paths": {
                "dataset_policy": "dataset_policy.yaml",
                "metadata_allowlist": "metadata_allowlist.yaml",
                "forbidden_patterns": "forbidden_patterns.yaml",
                "manual_approvals": "manual_approvals.yaml",
            },
            "metadata": {
                "creators": [
                    {
                        "name": "Synthetic Test Author",
                        "affiliation": "Synthetic Test Institution",
                    }
                ],
                "description": "Runtime-generated synthetic test material only.",
                "keywords": ["synthetic", "test"],
                "licenses": {
                    "source_code": "BSD-3-Clause",
                    "derived_data": {
                        "identifier": None,
                        "status": (
                            "dataset_specific_rights_confirmed_"
                            "record_level_review_required"
                        ),
                        "evidence": "synthetic-test-dataset-rights",
                    },
                    "documentation": "CC-BY-4.0",
                },
                "dataset_rights": copy.deepcopy(SYNTHETIC_DATASET_RIGHTS),
                "related_identifiers": [],
                "funding": [],
            },
            "safe_export_contract": {
                "input_must_be_designated_directory": True,
                "input_manifest_required": True,
                "unknown_files": "reject",
                "unknown_fields": "reject",
                "symbolic_links": "reject",
                "copy_original_filenames": False,
                "copy_source_paths": False,
                "hashed_original_identifiers": "forbidden",
                "sample_uid_generation": (
                    "only_when_confirmed_dataset_policy_and_"
                    "artifact_binding_permit"
                ),
                "npz_allow_pickle": False,
                "finite_numeric_arrays_required": True,
            },
            "publication_requirements": {
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
            },
        }
        if config_overrides:
            config.update(config_overrides)
        config_path = root / "release.yaml"
        write_yaml(config_path, config)
        return config_path

    return factory


@pytest.fixture
def safe_export_factory(tmp_path):
    """Create an in-memory-derived export without touching real data."""

    def factory(
        *,
        dataset: str = SYNTHETIC_DATASET,
        metadata_columns: tuple[str, ...] = ("sample_uid",),
        include_splits: bool = False,
        manifest_overrides: dict[str, object] | None = None,
    ) -> Path:
        root = tmp_path / f"safe-export-{dataset}"
        dataset_dir = root / "datasets" / dataset
        dataset_dir.mkdir(parents=True, exist_ok=True)

        matrices = correlation_matrices(n_regions=100)
        connectomes_path = dataset_dir / "connectomes.npz"
        metadata_path = dataset_dir / "metadata.tsv"
        splits_path = dataset_dir / "splits.tsv"
        np.savez(connectomes_path, connectomes=matrices)

        sample_uids = SYNTHETIC_SAMPLE_UIDS
        metadata_rows = []
        for index, sample_uid in enumerate(sample_uids):
            values: dict[str, object] = {
                "sample_uid": sample_uid,
                "age": 20 + index,
                "sex": "F" if index % 2 == 0 else "M",
                "subject_id": f"forbidden-{index}",
            }
            metadata_rows.append(
                {
                    column: values.get(column, f"synthetic-{index}")
                    for column in metadata_columns
                }
            )
        write_tsv(
            metadata_path,
            fieldnames=list(metadata_columns),
            rows=metadata_rows,
        )
        if include_splits:
            write_tsv(
                splits_path,
                fieldnames=["fold", "partition", "sample_uid"],
                rows=[
                    {
                        "fold": 0,
                        "partition": "test" if index == 0 else "train",
                        "sample_uid": sample_uid,
                    }
                    for index, sample_uid in enumerate(sample_uids)
                ],
            )

        files: dict[str, object] = {
            "connectomes": {
                "path": f"datasets/{dataset}/connectomes.npz",
                "sha256": sha256_file(connectomes_path),
            },
            "metadata": {
                "path": f"datasets/{dataset}/metadata.tsv",
                "sha256": sha256_file(metadata_path),
            },
        }
        if include_splits:
            files["splits"] = {
                "path": f"datasets/{dataset}/splits.tsv",
                "sha256": sha256_file(splits_path),
            }

        manifest: dict[str, object] = {
            "schema_version": 1,
            "kind": "trusted_safe_export",
            "datasets": [
                {
                    "dataset": dataset,
                    "sample_count": len(sample_uids),
                    "matrix_type": "correlation",
                    "atlas": "schaefer_100",
                    "n_regions": int(matrices.shape[-1]),
                    "source_binding_sha256": SYNTHETIC_SOURCE_BINDING_SHA256,
                    "files": files,
                }
            ],
            "aggregate_outputs": [],
            "producer": "tools.zenodo.export_internal",
        }
        if manifest_overrides:
            manifest.update(manifest_overrides)
        (root / "export_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / ".safe-export-root.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
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
        return root

    return factory


@pytest.fixture
def internal_manifest_factory(tmp_path):
    """Create an inert, non-deserializable filename sentinel for Stage 1 dry-run."""

    def factory(
        *,
        dataset: str = SYNTHETIC_DATASET,
        schema_overrides: dict[str, object] | None = None,
    ) -> Path:
        schema: dict[str, object] = {
            "connectome_column": "Connectome",
            "metadata_columns": {},
            "matrix_type": "correlation",
            "ignored_internal_columns": ["SubjectID"],
        }
        if schema_overrides:
            schema.update(schema_overrides)
        input_path = (tmp_path / f"{dataset}-synthetic-input.pkl").resolve()
        input_path.write_bytes(b"not-a-pickle")
        manifest = {
            "schema_version": 1,
            "export_namespace": "12345678-1234-5678-9234-567812345678",
            "trust_attestation": {
                "attested_by": "Synthetic Test Authority",
                "attested_on": "2000-01-01",
                "evidence": "runtime-synthetic-test",
                "source_controlled": True,
                "checksums_verified": True,
            },
            "datasets": [
                {
                    "dataset": dataset,
                    "input_path": str(input_path),
                    "sha256": sha256_file(input_path),
                    "input_format": "pandas_dataframe",
                    "schema": schema,
                    "atlas": "schaefer_100",
                    "expected_regions": 100,
                    "source_identity_attestation": {
                        "dataset": dataset,
                        "approved_by": "Synthetic Test Authority",
                        "evidence": "runtime-synthetic-test",
                    },
                }
            ],
        }
        path = tmp_path / f"internal-manifest-{dataset}.yaml"
        write_yaml(path, manifest)
        return path

    return factory


@pytest.fixture
def built_release(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    """Build one complete release entirely from the synthetic safe export."""

    from tools.zenodo.build_release import build_release

    config_path = release_config_factory()
    safe_export = safe_export_factory()
    release_dir = build_release(
        config_path,
        safe_export,
        tmp_path / "release-build-output",
    )
    return {
        "config_path": config_path,
        "safe_export": safe_export,
        "release_dir": release_dir,
    }


@pytest.fixture
def built_camcan_release(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    """Build the canonical CamCAN subtree from synthetic matrices and metadata."""

    from tools.zenodo.build_release import build_release

    config_path = release_config_factory(dataset="camcan")
    safe_export = safe_export_factory(
        dataset="camcan",
        metadata_columns=("sample_uid", "age", "sex"),
        include_splits=False,
    )
    release_dir = build_release(
        config_path,
        safe_export,
        tmp_path / "camcan-release-build-output",
    )
    return {
        "config_path": config_path,
        "safe_export": safe_export,
        "release_dir": release_dir,
    }
