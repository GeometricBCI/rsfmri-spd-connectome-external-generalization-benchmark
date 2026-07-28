from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tools.zenodo.metadata import (
    MIXED_RIGHTS_ACTION,
    _validate_metadata_schema,
    build_zenodo_record_metadata,
    manual_action_items,
)
from tools.zenodo.schemas import (
    load_release_bundle,
    participant_release_decision,
)


def _formal_config() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "release"
        / "zenodo_release.yaml"
    )


def test_formal_zenodo_metadata_contains_three_confirmed_rights_statements():
    bundle = load_release_bundle(_formal_config())

    record = build_zenodo_record_metadata(bundle)
    description = record["description"]

    assert "CamCAN — approved derived-data release scope" in description
    assert "ABIDE — approved derived-data release scope" in description
    assert "COBRE — approved derived-data release scope" in description
    assert "CC BY 4.0" in description
    assert "ABIDE_source_license" not in description
    assert "COBRE_confirmed_license" not in description
    assert "ABIDE/INDI source terms" in description
    assert "COBRE/FCP source terms" in description
    assert "pending approval" not in description.lower()
    assert "license" not in record


def test_completed_mixed_rights_review_removes_its_publication_blocker():
    bundle = load_release_bundle(_formal_config())
    protocol = bundle["manual_approvals"]["approval_protocol"]
    derived = bundle["config"]["metadata"]["licenses"]["derived_data"]
    derived.update(
        {
            "status": "approved",
            "evidence": "human-reviewed-mixed-rights-record",
        }
    )
    review = bundle["manual_approvals"]["approvals"]["licenses"]["derived_data"]
    review.update(
        {
            "status": "approved",
            "approved_by": "Designated human reviewer",
            "approved_on": "2026-07-27",
            "scope": ["approved_derived_data_artifacts"],
            "evidence": "human-reviewed-mixed-rights-record",
            "confirmation": protocol["required_confirmation_text"],
        }
    )
    bundle["manual_approvals"]["manual_action_items"] = [
        item
        for item in bundle["manual_approvals"]["manual_action_items"]
        if "mixed dataset-specific rights" not in item
    ]

    assert MIXED_RIGHTS_ACTION not in manual_action_items(bundle)


def test_generated_form_includes_dataset_rights_once_in_the_description(
    built_release,
):
    form = (
        built_release["release_dir"]
        / "metadata"
        / "zenodo_form_values.md"
    ).read_text(encoding="utf-8")

    assert "## Description" in form
    assert "## Dataset-specific rights" in form
    assert "SYNTHETIC TEST METADATA — DO NOT UPLOAD OR PUBLISH" in form
    assert form.count("CamCAN — approved derived-data release scope") == 1
    assert form.count("ABIDE — approved derived-data release scope") == 1
    assert form.count("COBRE — approved derived-data release scope") == 1
    assert "pending approval" not in form.lower()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rights: rights.update(
            {
                "mystery": copy.deepcopy(rights["abide"]),
            }
        ),
        lambda rights: rights["abide"].update({"license": ""}),
        lambda rights: rights["cobre"].update({"conditions": []}),
        lambda rights: rights["camcan"].update({"required_citations": []}),
        lambda rights: rights["abide"].update({"status": "manual_required"}),
        lambda rights: rights["cobre"].update({"unexpected": "value"}),
    ],
)
def test_dataset_rights_schema_fails_closed(
    release_config_factory,
    mutation,
):
    bundle = load_release_bundle(release_config_factory())
    config = copy.deepcopy(bundle["config"])
    rights = config["metadata"]["dataset_rights"]
    mutation(rights)

    with pytest.raises(ValueError, match="dataset.rights|rights"):
        _validate_metadata_schema(config)


@pytest.mark.parametrize("dataset", ["abide", "camcan", "cobre"])
def test_participant_release_requires_matching_dataset_license_metadata(
    release_config_factory,
    dataset,
):
    bundle = load_release_bundle(
        release_config_factory(dataset=dataset, approved=True)
    )
    bundle["config"]["metadata"]["dataset_rights"][dataset]["license"] = (
        "mismatched-license"
    )

    assert participant_release_decision(
        bundle, dataset, "participant_connectomes"
    ) == (False, "dataset_rights_metadata_missing_or_mismatched")


@pytest.mark.parametrize("dataset", ["abide", "camcan", "cobre"])
@pytest.mark.parametrize(
    "mutation",
    [
        lambda rights: rights.__setitem__(
            "conditions", rights["conditions"][:-1]
        ),
        lambda rights: rights.__setitem__(
            "conditions", list(reversed(rights["conditions"]))
        ),
        lambda rights: rights.__setitem__(
            "rights_statement", "Attribution only."
        ),
    ],
)
def test_participant_release_requires_exact_reviewed_rights_scope(
    release_config_factory,
    dataset,
    mutation,
):
    bundle = load_release_bundle(
        release_config_factory(dataset=dataset, approved=True)
    )
    mutation(bundle["config"]["metadata"]["dataset_rights"][dataset])

    assert participant_release_decision(
        bundle, dataset, "participant_connectomes"
    ) == (False, "dataset_rights_metadata_missing_or_mismatched")
