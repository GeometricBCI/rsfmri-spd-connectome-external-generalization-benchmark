from __future__ import annotations

import json
import pickle

import pytest

from tools.zenodo.build_release import _refresh_catalogs
from tools.zenodo.checksums import sha256_file
from tools.zenodo.schemas import load_release_bundle
from tools.zenodo.validate_release import validate_release


def _error_codes(result) -> set[str]:
    return {finding.code for finding in result.errors}


def _warning_codes(result) -> set[str]:
    return {finding.code for finding in result.warnings}


def _file_snapshot(root):
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in root.rglob("*")
        if path.is_file()
    }


def test_built_synthetic_release_passes_structural_validation(built_release):
    release_dir = built_release["release_dir"]
    result = validate_release(
        release_dir,
        config_path=built_release["config_path"],
        write_reports=False,
    )

    assert result.ok, result.as_dict()
    assert result.errors == []
    assert result.checks["privacy_finding_count"] == 0
    assert result.checks["catalogs_valid"] is True
    assert result.checks["sample_counts"]["abide"] == {
        "connectomes": 3,
        "metadata": 3,
        "uid_count": 3,
    }
    stored_report = json.loads(
        (
            release_dir / "manifests/validation_report.json"
        ).read_text(encoding="utf-8")
    )
    assert stored_report["checks"]["filesystem"]["file_count"] == len(
        _file_snapshot(release_dir)
    )


def test_validator_can_use_frozen_policy_snapshots(built_release):
    result = validate_release(
        built_release["release_dir"],
        config_path=None,
        write_reports=False,
    )

    assert result.ok, result.as_dict()


def test_camcan_release_passes_with_its_dataset_specific_companions(
    built_camcan_release,
):
    result = validate_release(
        built_camcan_release["release_dir"],
        config_path=built_camcan_release["config_path"],
        write_reports=False,
    )

    assert result.ok, result.as_dict()
    assert result.checks["public_datasets"]["camcan"] == [
        "participant_connectomes",
        "participant_metadata",
    ]


@pytest.mark.parametrize(
    ("original", "replacement", "expected_code"),
    [
        ("\t20\tF", "\tnan\tF", "CAMCAN_AGE_VALUE"),
        ("\t20\tF", "\t20\tunknown", "CAMCAN_SEX_VALUE"),
    ],
)
def test_validator_rejects_invalid_camcan_age_or_sex_values(
    built_camcan_release,
    original,
    replacement,
    expected_code,
):
    release_dir = built_camcan_release["release_dir"]
    participants = release_dir / "data/camcan/metadata/participants.tsv"
    participants.write_text(
        participants.read_text(encoding="utf-8").replace(
            original, replacement, 1
        ),
        encoding="utf-8",
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert expected_code in _error_codes(result)


def test_validator_rejects_tampered_camcan_license_even_after_recataloging(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    license_path = release_dir / "data/camcan/LICENSE.txt"
    license_path.write_text(
        license_path.read_text(encoding="utf-8").replace(
            "CC-BY-4.0", "CC0-1.0", 1
        ),
        encoding="utf-8",
    )
    _refresh_catalogs(
        release_dir,
        load_release_bundle(built_camcan_release["config_path"]),
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "CAMCAN_LICENSE_BINDING" in _error_codes(result)


def test_validator_rejects_ccids_disguised_as_camcan_sample_uids(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    participants = release_dir / "data/camcan/metadata/participants.tsv"
    old_digest = sha256_file(participants)
    participants.write_text(
        "sample_uid\tage\tsex\n"
        "cc110033\t20\tF\n"
        "cc220044\t21\tM\n"
        "cc330055\t22\tF\n",
        encoding="utf-8",
    )
    provenance = release_dir / "metadata/provenance.tsv"
    provenance.write_text(
        provenance.read_text(encoding="utf-8").replace(
            old_digest, sha256_file(participants), 1
        ),
        encoding="utf-8",
    )
    _refresh_catalogs(
        release_dir,
        load_release_bundle(built_camcan_release["config_path"]),
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "UNSAFE_SAMPLE_UID" in _error_codes(result)


def test_validator_rejects_camcan_home_interview_metadata(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    participants = release_dir / "data/camcan/metadata/participants.tsv"
    participants.write_text(
        "sample_uid\tage\tsex\thome_interview_score\n"
        "s00000000000000000000000000000001\t20\tF\t1\n"
        "s00000000000000000000000000000002\t21\tM\t2\n"
        "s00000000000000000000000000000003\t22\tF\t3\n",
        encoding="utf-8",
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert {
        "FORBIDDEN_METADATA_COLUMN",
        "IDENTIFIER_COLUMN",
    } & _error_codes(result)


def test_validator_rejects_camcan_t1_image(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    unsafe = release_dir / "data/camcan/T1w_source.nii.gz"
    unsafe.write_bytes(b"synthetic sentinel; not an image")

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert {"FORBIDDEN_FILE_TYPE", "RAW_T1_IMAGE"} <= _error_codes(result)


def test_validator_rejects_mutated_camcan_dictionary_semantics(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    for relative in (
        "metadata/data_dictionary.tsv",
        "data/camcan/data_dictionary.tsv",
    ):
        path = release_dir / relative
        path.write_text(
            path.read_text(encoding="utf-8").replace(
                "chronological age", "unreviewed age semantics"
            ),
            encoding="utf-8",
        )
    _refresh_catalogs(
        release_dir,
        load_release_bundle(built_camcan_release["config_path"]),
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert {
        "DATA_DICTIONARY_MISMATCH",
        "CAMCAN_DATA_DICTIONARY",
    } <= _error_codes(result)


def test_validator_rejects_contradictory_root_license_after_recataloging(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    root_license = release_dir / "LICENSES.md"
    root_license.write_text(
        root_license.read_text(encoding="utf-8")
        + "\nContradictory CamCAN license: CC0-1.0\n",
        encoding="utf-8",
    )
    _refresh_catalogs(
        release_dir,
        load_release_bundle(built_camcan_release["config_path"]),
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "LICENSES_DOCUMENT" in _error_codes(result)


def test_validator_binds_repository_snapshot_to_local_head(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    snapshot_path = release_dir / "reproducibility/repository_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    original_commit = snapshot["git_commit"]
    fabricated_commit = "0" * 40
    assert fabricated_commit != original_commit
    snapshot["git_commit"] = fabricated_commit
    snapshot["source_repository"] = "https://example.invalid/fabricated"
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / "reproducibility/git_commit.txt").write_text(
        "commit: " + fabricated_commit + "\n"
        f"worktree_dirty: {str(snapshot['tracked_worktree_dirty']).lower()}\n",
        encoding="utf-8",
    )
    provenance = release_dir / "metadata/provenance.tsv"
    provenance.write_text(
        provenance.read_text(encoding="utf-8").replace(
            original_commit, fabricated_commit
        ),
        encoding="utf-8",
    )
    _refresh_catalogs(
        release_dir,
        load_release_bundle(built_camcan_release["config_path"]),
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "GIT_SNAPSHOT" in _error_codes(result)


def test_validator_requires_the_complete_camcan_file_set(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    (release_dir / "data/camcan/LICENSE.txt").unlink()

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "CAMCAN_RELEASE_LAYOUT" in _error_codes(result)


def test_participants_filename_exception_is_limited_to_camcan_path(
    built_release,
):
    release_dir = built_release["release_dir"]
    (release_dir / "data/participants.tsv").write_text(
        "sample_uid\tage\nabide0000001\t20\n",
        encoding="utf-8",
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "RAW_PARTICIPANT_FILE" in _error_codes(result)


def test_validator_rejects_legacy_camcan_public_layout(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    legacy = release_dir / "data/public_metadata/camcan"
    legacy.mkdir(parents=True)
    (legacy / "metadata.tsv").write_text(
        "sample_uid\tage\tsex\ncamcan_999999\t40\tF\n",
        encoding="utf-8",
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "UNAPPROVED_RELEASE_DIRECTORY" in _error_codes(result)


def test_read_only_validation_does_not_mutate_release(built_release):
    release_dir = built_release["release_dir"]
    before = _file_snapshot(release_dir)

    result = validate_release(release_dir, write_reports=False)

    assert result.ok, result.as_dict()
    assert _file_snapshot(release_dir) == before


def test_draft_validation_warns_when_source_snapshot_is_dirty(built_release):
    release_dir = built_release["release_dir"]
    snapshot_path = release_dir / "reproducibility/repository_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["tracked_worktree_dirty"] = True
    snapshot_path.write_text(
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (release_dir / "reproducibility/git_commit.txt").write_text(
        f"commit: {snapshot['git_commit']}\nworktree_dirty: true\n",
        encoding="utf-8",
    )
    _refresh_catalogs(
        release_dir,
        load_release_bundle(built_release["config_path"]),
    )

    result = validate_release(release_dir, write_reports=False)

    assert result.ok, result.as_dict()
    assert "DIRTY_SOURCE_SNAPSHOT" in _warning_codes(result)


def test_publication_ready_mode_refuses_synthetic_release(built_release):
    result = validate_release(
        built_release["release_dir"],
        publication_ready=True,
        write_reports=False,
    )

    assert not result.ok
    assert "SYNTHETIC_TEST_RELEASE" in _error_codes(result)


def test_validator_rejects_pickle_by_name_without_deserializing(
    built_release,
    monkeypatch,
):
    release_dir = built_release["release_dir"]
    unsafe = release_dir / "data" / "synthetic_payload.pkl"
    unsafe.write_bytes(b"not-a-pickle")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("release validation must never deserialize pickle")

    monkeypatch.setattr(pickle, "load", fail_if_called)

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert {"FORBIDDEN_FILE_TYPE", "SOURCE_PICKLE_NAME"} <= _error_codes(result)


def test_validator_detects_restricted_participant_directory(built_release):
    release_dir = built_release["release_dir"]
    (release_dir / "data" / "public_metadata" / "adni").mkdir()

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "RESTRICTED_DATASET_LEAK" in _error_codes(result)


def test_validator_detects_unknown_dataset_directory(built_release):
    release_dir = built_release["release_dir"]
    (release_dir / "data" / "public_metadata" / "mystery_cohort").mkdir()

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "UNKNOWN_DATASET" in _error_codes(result)


def test_validator_detects_split_overlap(built_release):
    release_dir = built_release["release_dir"]
    split_path = release_dir / "splits" / "abide" / "splits.tsv"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(
        "fold\tpartition\tsample_uid\n"
        "0\ttest\ts00000000000000000000000000000001\n"
        "0\ttrain\ts00000000000000000000000000000001\n",
        encoding="utf-8",
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "SPLIT_OVERLAP" in _error_codes(result)


def test_validator_rejects_a_recataloged_noncanonical_environment(built_release):
    release_dir = built_release["release_dir"]
    environment = release_dir / "reproducibility" / "environment.yml"
    environment.write_text("name: unrelated-environment\n", encoding="utf-8")
    _refresh_catalogs(
        release_dir,
        load_release_bundle(built_release["config_path"]),
    )

    result = validate_release(release_dir, write_reports=False)

    assert not result.ok
    assert "REPRODUCIBILITY_ENVIRONMENT" in _error_codes(result)


def test_validator_does_not_repair_a_corrupt_manifest(built_release):
    release_dir = built_release["release_dir"]
    manifest = release_dir / "manifests" / "manifest.tsv"
    corrupt = "not\tthe\trequired\tschema\n"
    manifest.write_text(corrupt, encoding="utf-8")

    result = validate_release(release_dir, write_reports=True)

    assert not result.ok
    assert "MANIFEST_MISMATCH" in _error_codes(result)
    assert manifest.read_text(encoding="utf-8") == corrupt


def test_validation_reports_do_not_echo_sensitive_matches(built_release):
    release_dir = built_release["release_dir"]
    sensitive = "/Users/private-reviewer/source/sub-secret01/file.txt"
    (release_dir / "synthetic-notes.md").write_text(
        f"Never publish {sensitive}\n",
        encoding="utf-8",
    )

    result = validate_release(release_dir, write_reports=True)

    assert not result.ok
    assert "ABSOLUTE_POSIX_PATH" in _error_codes(result)
    reports = (
        release_dir / "manifests" / "validation_report.json"
    ).read_text(encoding="utf-8")
    reports += "\n" + (
        release_dir / "manifests" / "validation_report.md"
    ).read_text(encoding="utf-8")
    assert sensitive not in reports


def test_missing_release_directory_returns_a_result_instead_of_raising(tmp_path):
    missing = tmp_path / "does-not-exist"

    result = validate_release(missing, write_reports=False)

    assert not result.ok
    assert _error_codes(result) == {"MISSING_RELEASE_DIR"}
    assert not missing.exists()
