from __future__ import annotations

import json

import pytest
import yaml

import tools.zenodo.build_release as release_builder
from tools.zenodo.build_release import ReleaseBuildError, build_release
from tools.zenodo.checksums import sha256_file
from tools.zenodo.validate_release import validate_release


def _load_export_manifest(safe_export):
    path = safe_export / "export_manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _write_export_manifest(path, manifest):
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _refresh_dataset_file_checksum(safe_export, file_key):
    manifest_path, manifest = _load_export_manifest(safe_export)
    descriptor = manifest["datasets"][0]["files"][file_key]
    descriptor["sha256"] = sha256_file(safe_export / descriptor["path"])
    _write_export_manifest(manifest_path, manifest)


def test_builder_creates_a_complete_valid_release(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()

    release_dir = build_release(
        config_path,
        safe_export,
        tmp_path / "output",
    )

    assert release_dir.is_dir()
    assert (release_dir / "data/public_connectomes/abide/connectomes.npz").is_file()
    assert (release_dir / "data/public_metadata/abide/metadata.tsv").is_file()
    assert not (release_dir / "splits/abide").exists()
    assert (release_dir / "manifests/manifest.tsv").is_file()
    assert (release_dir / "manifests/SHA256SUMS.txt").is_file()
    assert (release_dir / "CITATION.cff").is_file()
    dataset_card = (release_dir / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert "- ABIDE: 3 synthetic samples" in dataset_card
    assert "synthetic/public samples" not in dataset_card
    inventory = (
        release_dir / "metadata/dataset_inventory.tsv"
    ).read_text(encoding="utf-8")
    exact_split_row = next(
        line
        for line in inventory.splitlines()
        if line.startswith("abide\texact_splits\t")
    )
    assert exact_split_row.endswith("\tforbidden")
    assert not any(
        path.name.lower().endswith((".pkl", ".pickle", ".nii", ".nii.gz"))
        for path in release_dir.rglob("*")
    )

    validation = validate_release(
        release_dir,
        config_path=config_path,
        write_reports=False,
    )
    assert validation.ok, validation.as_dict()


def test_real_data_draft_has_a_prominent_do_not_upload_banner(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory(test_only=False)
    release_dir = build_release(
        config_path,
        safe_export_factory(),
        tmp_path / "output",
    )
    banner = (
        "> **REAL-DATA DRAFT — NOT APPROVED FOR UPLOAD OR PUBLICATION.**"
    )

    for name in ("README.md", "DATASET_CARD.md", "LICENSES.md"):
        assert (release_dir / name).read_text(encoding="utf-8").startswith(
            banner
        )
    assert "- ABIDE: 3 draft release candidates" in (
        release_dir / "README.md"
    ).read_text(encoding="utf-8")
    inventory = (
        release_dir / "metadata/dataset_inventory.tsv"
    ).read_text(encoding="utf-8")
    connectome_row = next(
        line
        for line in inventory.splitlines()
        if line.startswith("abide\tparticipant_connectomes\t")
    )
    assert connectome_row.endswith(
        "\tsource_bound_draft_candidate_pending_final_review"
    )


def test_restricted_reconstruction_keeps_participant_files_private(
    built_release,
):
    release_dir = built_release["release_dir"]
    for dataset in ("adni", "adnidod", "oasis3"):
        commands = (
            release_dir
            / "restricted_reconstruction"
            / dataset
            / "reconstruction_commands.md"
        ).read_text(encoding="utf-8")
        assert "Do not pass them to the trusted exporter" in commands
        assert "Run the trusted exporter" not in commands


def test_draft_without_creators_omits_invalid_citation_file(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["metadata"]["creators"] = []
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    release_dir = build_release(
        config_path,
        safe_export_factory(),
        tmp_path / "output-without-creators",
    )

    assert not (release_dir / "CITATION.cff").exists()
    validation = validate_release(
        release_dir,
        config_path=config_path,
        write_reports=False,
    )
    assert validation.ok, validation.as_dict()


def test_builder_creates_only_the_canonical_camcan_zenodo_subtree(
    built_camcan_release,
):
    release_dir = built_camcan_release["release_dir"]
    expected = {
        "data/camcan/connectomes/camcan_schaefer100_fc.npz",
        "data/camcan/metadata/participants.tsv",
        "data/camcan/data_dictionary.tsv",
        "data/camcan/LICENSE.txt",
    }

    assert all((release_dir / relative).is_file() for relative in expected)
    assert not (release_dir / "data/public_connectomes/camcan").exists()
    assert not (release_dir / "data/public_metadata/camcan").exists()
    assert not (release_dir / "splits/camcan").exists()
    assert (
        release_dir / "data/camcan/metadata/participants.tsv"
    ).read_text(encoding="utf-8").splitlines()[0] == "sample_uid\tage\tsex"
    license_text = (release_dir / "data/camcan/LICENSE.txt").read_text(
        encoding="utf-8"
    )
    assert "CC-BY-4.0" in license_text
    assert "https://doi.org/10.1186/s12883-014-0204-1" in license_text
    assert "a cross-sectional, lifespan" in license_text
    assert "Shafto_et_al_CamCAN_cohort_paper" not in license_text
    assert "home_interview_variables" not in license_text

    validation = validate_release(
        release_dir,
        config_path=built_camcan_release["config_path"],
        write_reports=False,
    )
    assert validation.ok, validation.as_dict()
    assert validation.checks["sample_counts"]["camcan"] == {
        "connectomes": 3,
        "metadata": 3,
        "uid_count": 3,
    }


def test_builder_creates_a_valid_confirmed_cobre_release(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory(dataset="cobre")
    safe_export = safe_export_factory(
        dataset="cobre",
        metadata_columns=("sample_uid",),
        include_splits=False,
    )

    release_dir = build_release(
        config_path,
        safe_export,
        tmp_path / "cobre-output",
    )
    validation = validate_release(
        release_dir,
        config_path=config_path,
        write_reports=False,
    )

    assert validation.ok, validation.as_dict()
    assert (
        release_dir / "data/public_connectomes/cobre/connectomes.npz"
    ).is_file()
    assert (
        release_dir / "data/public_metadata/cobre/metadata.tsv"
    ).read_text(encoding="utf-8").splitlines()[0] == "sample_uid"
    metadata = (
        release_dir / "metadata/zenodo_record_metadata.json"
    ).read_text(encoding="utf-8")
    assert "COBRE_confirmed_license" not in metadata
    assert "COBRE/FCP source terms" in metadata
    assert "approved derived-data release" in metadata


def test_builder_rejects_camcan_ccid_even_with_written_confirmation(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory(dataset="camcan")
    safe_export = safe_export_factory(
        dataset="camcan",
        metadata_columns=("sample_uid", "age", "sex", "CCID"),
        include_splits=False,
    )

    with pytest.raises(
        ReleaseBuildError,
        match="forbidden participant (metadata|column)",
    ):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_camcan_ccids_disguised_as_sample_uids(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory(dataset="camcan")
    safe_export = safe_export_factory(
        dataset="camcan",
        metadata_columns=("sample_uid", "age", "sex"),
        include_splits=False,
    )
    metadata_path = safe_export / "datasets/camcan/metadata.tsv"
    metadata_path.write_text(
        "sample_uid\tage\tsex\n"
        "cc110033\t20\tF\n"
        "cc220044\t21\tM\n"
        "cc330055\t22\tF\n",
        encoding="utf-8",
    )
    _refresh_dataset_file_checksum(safe_export, "metadata")

    with pytest.raises(
        ReleaseBuildError,
        match="sample_uid does not match the release-safe schema",
    ):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_camcan_exact_split_membership(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory(dataset="camcan")
    safe_export = safe_export_factory(
        dataset="camcan",
        metadata_columns=("sample_uid", "age", "sex"),
        include_splits=True,
    )

    with pytest.raises(
        ReleaseBuildError,
        match="content_scope_not_explicitly_releasable",
    ):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_dry_run_is_non_writing_and_deeply_validates_safe_export(
    tmp_path,
    release_config_factory,
    safe_export_factory,
    capsys,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()
    output_dir = tmp_path / "not-created"

    destination = build_release(
        config_path,
        safe_export,
        output_dir,
        dry_run=True,
    )
    report = json.loads(capsys.readouterr().out)

    assert report == {
        "datasets": ["abide"],
        "dry_run": True,
        "manual_action_count": 0,
        "release_directory_name": destination.name,
        "would_write": False,
    }
    assert str(tmp_path) not in json.dumps(report)
    assert not output_dir.exists()
    assert not destination.exists()


def test_builder_requires_explicit_dataset_approval(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory(approved=False, test_only=False)
    safe_export = safe_export_factory()

    with pytest.raises(ReleaseBuildError, match="manual_approval_incomplete"):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_unknown_datasets_fail_closed(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory(dataset="mystery_cohort")

    with pytest.raises(ReleaseBuildError, match="unknown dataset"):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_never_exports_restricted_participant_content(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory(dataset="adni")

    with pytest.raises(ReleaseBuildError, match="restricted dataset adni"):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_unmanifested_files(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()
    (safe_export / "unmanifested.txt").write_text(
        "synthetic sentinel\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseBuildError, match="unmanifested or missing"):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_manifest_path_traversal_before_copying(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()
    manifest_path, manifest = _load_export_manifest(safe_export)
    manifest["datasets"][0]["files"]["connectomes"]["path"] = "../outside.npz"
    _write_export_manifest(manifest_path, manifest)

    with pytest.raises(ReleaseBuildError, match="unsafe path"):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_safe_export_checksum_mismatch(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()
    (safe_export / "datasets/abide/metadata.tsv").write_text(
        "sample_uid\tage\nabide0000001\t999\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseBuildError, match="checksum mismatch"):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_forbidden_metadata_even_when_synthetic(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory(
        metadata_columns=("sample_uid", "subject_id")
    )

    with pytest.raises(
        ReleaseBuildError,
        match="forbidden participant (metadata|column)",
    ):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_abide_exact_split_membership(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory(include_splits=True)
    split_path = safe_export / "datasets/abide/splits.tsv"
    split_path.write_text("fold\tpartition\tsample_uid\n", encoding="utf-8")
    _refresh_dataset_file_checksum(safe_export, "splits")

    with pytest.raises(
        ReleaseBuildError,
        match="content_scope_not_explicitly_releasable",
    ):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_rejects_unconfirmed_abide_split_before_copying(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory(include_splits=True)
    split_path = safe_export / "datasets/abide/splits.tsv"
    split_path.write_text(
        "fold\tpartition\tsample_uid\n"
        "0\ttest\ts00000000000000000000000000000001\n"
        "0\ttrain\ts00000000000000000000000000000001\n"
        "0\ttrain\ts00000000000000000000000000000002\n"
        "0\ttrain\ts00000000000000000000000000000003\n",
        encoding="utf-8",
    )
    _refresh_dataset_file_checksum(safe_export, "splits")

    with pytest.raises(
        ReleaseBuildError,
        match="content_scope_not_explicitly_releasable",
    ):
        build_release(config_path, safe_export, tmp_path / "output")


@pytest.mark.parametrize(
    ("descriptor_dataset", "path_dataset", "table_dataset"),
    [
        ("abide", "cobre", "abide"),
        ("abide", "abide", "cobre"),
    ],
)
def test_builder_binds_aggregate_dataset_to_descriptor_path_and_table(
    tmp_path,
    release_config_factory,
    safe_export_factory,
    descriptor_dataset,
    path_dataset,
    table_dataset,
):
    config_path = release_config_factory(
        config_overrides={
            "content_allowlist": {
                "aggregate_columns": {
                    "aggregate_metrics": ["dataset", "metric", "value"],
                },
                "categories": {
                    "aggregate_metrics": {
                        "allowed_extensions": [".csv"],
                    }
                },
            }
        }
    )
    safe_export = safe_export_factory()
    aggregate_path = (
        safe_export / "aggregate_outputs" / path_dataset / "metrics.csv"
    )
    aggregate_path.parent.mkdir(parents=True)
    aggregate_path.write_text(
        "dataset,metric,value\n"
        f"{table_dataset},mae,0.1\n",
        encoding="utf-8",
    )
    manifest_path, manifest = _load_export_manifest(safe_export)
    manifest["aggregate_outputs"] = [
        {
            "dataset": descriptor_dataset,
            "content_category": "aggregate_metrics",
            "file": {
                "path": aggregate_path.relative_to(safe_export).as_posix(),
                "sha256": sha256_file(aggregate_path),
            },
        }
    ]
    _write_export_manifest(manifest_path, manifest)

    with pytest.raises(ReleaseBuildError, match="dataset"):
        build_release(config_path, safe_export, tmp_path / "output")


def test_builder_refuses_to_overwrite_an_existing_staging_tree(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()
    output_dir = tmp_path / "output"
    first = build_release(config_path, safe_export, output_dir)

    with pytest.raises(ReleaseBuildError, match="already exists"):
        build_release(config_path, safe_export, output_dir)

    assert first.is_dir()


def test_builder_derives_metadata_from_the_frozen_staging_copy(
    tmp_path,
    release_config_factory,
    safe_export_factory,
    monkeypatch,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()
    metadata_source = (safe_export / "datasets/abide/metadata.tsv").resolve()
    original_copy = release_builder._copy_verified_file

    def copy_then_replace_source(source, target, expected_sha256):
        original_copy(source, target, expected_sha256)
        if source == metadata_source:
            source.write_text("null\n", encoding="utf-8")

    monkeypatch.setattr(
        release_builder,
        "_copy_verified_file",
        copy_then_replace_source,
    )
    release_dir = build_release(config_path, safe_export, tmp_path / "output")

    assert (
        release_dir / "data/public_metadata/abide/metadata.tsv"
    ).read_text(encoding="utf-8").startswith("sample_uid\n")
    validation = validate_release(release_dir, write_reports=False)
    assert validation.ok, validation.as_dict()


def test_failed_build_removes_its_private_workspace(
    tmp_path,
    release_config_factory,
    safe_export_factory,
    monkeypatch,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()
    output = tmp_path / "output"

    def fail_after_public_files_are_copied(*args, **kwargs):
        raise RuntimeError("synthetic injected failure")

    monkeypatch.setattr(
        release_builder,
        "_write_release_metadata_tables",
        fail_after_public_files_are_copied,
    )

    with pytest.raises(RuntimeError, match="synthetic injected failure"):
        build_release(config_path, safe_export, output)

    assert not output.exists()
    assert list(tmp_path.glob(".output.building-*")) == []
