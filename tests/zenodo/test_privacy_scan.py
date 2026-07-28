from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tools.zenodo.privacy_scan import scan_release


def _codes(findings) -> set[str]:
    return {finding.code for finding in findings}


def test_clean_synthetic_safe_export_has_no_privacy_findings(safe_export_factory):
    root = safe_export_factory()

    assert scan_release(root) == []


def test_binary_pickle_name_is_flagged_without_deserialization(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "release"
    root.mkdir()
    # Deliberately invalid bytes ensure this test cannot accidentally become a
    # participant-data fixture or rely on pickle deserialization.
    (root / "synthetic_payload.pkl").write_bytes(b"not-a-pickle")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("privacy scanning must never deserialize pickle")

    monkeypatch.setattr("pickle.load", fail_if_called)

    findings = scan_release(root)

    assert "SOURCE_PICKLE_NAME" in _codes(findings)


def test_tabular_identifier_columns_are_rejected(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "metadata.tsv").write_text(
        "sample_uid\tsubject_id\nabide0000001\tsynthetic-only\n",
        encoding="utf-8",
    )

    findings = scan_release(root)

    assert "IDENTIFIER_COLUMN" in _codes(findings)


def test_local_paths_and_identifiers_are_redacted_in_findings(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    sensitive = "/Users/private-reviewer/source/sub-secret01/file.txt"
    (root / "notes.md").write_text(
        f"Do not publish {sensitive}\n",
        encoding="utf-8",
    )

    findings = scan_release(root)

    assert {"ABSOLUTE_POSIX_PATH", "BIDS_PARTICIPANT_IDENTIFIER"} <= _codes(
        findings
    )
    assert all(sensitive not in finding.message for finding in findings)
    assert all(finding.redacted_value != sensitive for finding in findings)


def test_markdown_details_tags_are_not_mistaken_for_local_paths(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "notes.md").write_text(
        "<details>\n<summary>Review notes</summary>\nNone.\n</details>\n",
        encoding="utf-8",
    )

    assert scan_release(root) == []


def test_approved_creator_email_is_not_reported(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    email = "synthetic.author@example.org"
    (root / "CITATION.cff").write_text(f"email: {email}\n", encoding="utf-8")
    release_config = {
        "metadata": {
            "creators": [
                {
                    "name": "Synthetic Test Author",
                    "email": email,
                }
            ]
        }
    }

    assert scan_release(root, release_config=release_config) == []


def test_configured_forbidden_pattern_is_enforced(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "notes.txt").write_text(
        "synthetic-internal-marker-123\n",
        encoding="utf-8",
    )
    patterns = tmp_path / "patterns.yaml"
    patterns.write_text(
        yaml.safe_dump(
            {
                "patterns": [
                    {
                        "code": "INTERNAL_MARKER",
                        "regex": r"synthetic-internal-marker-\d+",
                        "description": "internal test marker",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    findings = scan_release(root, patterns)

    assert "INTERNAL_MARKER" in _codes(findings)


def _formal_forbidden_patterns() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "release"
        / "forbidden_patterns.yaml"
    )


def test_release_safe_uid_age_and_sex_are_privacy_safe(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "metadata.tsv").write_text(
        "sample_uid\tage\tsex\n"
        "s00000000000000000000000000000001\t40\tF\n",
        encoding="utf-8",
    )

    assert scan_release(root, _formal_forbidden_patterns()) == []


@pytest.mark.parametrize(
    "column",
    [
        "home_interview_score",
        "questionnaire_response",
        "survey_item_1",
        "behavioral_measure",
        "behavioural_variables",
        "comments",
        "unrestricted_clinical_text",
        "raw_filename",
    ],
)
def test_identifiable_or_unrestricted_metadata_columns_are_blocked(
    tmp_path,
    column,
):
    root = tmp_path / "release"
    root.mkdir()
    (root / "metadata.tsv").write_text(
        f"sample_uid\t{column}\n"
        "s00000000000000000000000000000001\tsynthetic-value\n",
        encoding="utf-8",
    )

    findings = scan_release(root, _formal_forbidden_patterns())

    assert "IDENTIFIER_COLUMN" in _codes(findings)


def test_relative_source_path_is_blocked(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "notes.txt").write_text(
        "sourcedata/internal_subject.csv\n",
        encoding="utf-8",
    )

    findings = scan_release(root, _formal_forbidden_patterns())

    assert "RELATIVE_SOURCE_PATH" in _codes(findings)


def test_configured_raw_t1_extension_and_filename_are_blocked(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "participant_T1w.dicom").write_bytes(b"synthetic sentinel")

    findings = scan_release(root, _formal_forbidden_patterns())

    assert {
        "FORBIDDEN_FILE_EXTENSION",
        "RAW_T1_IMAGE",
    } <= _codes(findings)
