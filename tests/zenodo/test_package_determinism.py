from __future__ import annotations

import json
import os
import zipfile

import pytest
import yaml

from tools.zenodo.checksums import sha256_file, verify_sha256sums
from tools.zenodo.package_release import ReleasePackageError, package_release


def test_synthetic_package_is_deterministic_and_self_checking(
    tmp_path,
    built_release,
):
    release_dir = built_release["release_dir"]
    first_upload = tmp_path / "upload-first"
    second_upload = tmp_path / "upload-second"

    first_archive = package_release(
        release_dir,
        first_upload,
        config_path=built_release["config_path"],
        allow_synthetic=True,
    )
    # ZIP metadata must be independent of filesystem timestamps.
    for path in release_dir.rglob("*"):
        if path.is_file():
            os.utime(path, (1_700_000_000, 1_700_000_000))
    second_archive = package_release(
        release_dir,
        second_upload,
        config_path=built_release["config_path"],
        allow_synthetic=True,
    )

    assert sha256_file(first_archive) == sha256_file(second_archive)
    assert first_archive.read_bytes() == second_archive.read_bytes()
    assert verify_sha256sums(
        first_upload,
        first_upload / "SHA256SUMS.txt",
    ) == []
    with zipfile.ZipFile(first_archive) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        assert names == sorted(names)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)
        assert all(not name.startswith("/") and ".." not in name.split("/") for name in names)
        assert {name.split("/", 1)[0] for name in names} == {release_dir.name}

    assert "DO NOT UPLOAD OR PUBLISH" in (
        first_upload / "README.md"
    ).read_text(encoding="utf-8")
    assert "DO NOT UPLOAD OR PUBLISH" in (
        first_upload / "ZENODO_UPLOAD_CHECKLIST.md"
    ).read_text(encoding="utf-8")
    checklist = (
        first_upload / "ZENODO_UPLOAD_CHECKLIST.md"
    ).read_text(encoding="utf-8")
    assert "[x] CamCAN derived-data sharing permission confirmed" in checklist
    assert "[x] ABIDE derived-data sharing conditions confirmed" in checklist
    assert "[x] COBRE derived-data sharing conditions confirmed" in checklist
    assert "[ ] final metadata verification" in checklist
    assert "[ ] final author verification" in checklist
    assert "[ ] final Zenodo upload approval" in checklist
    assert "Dataset permission pending" not in checklist


def test_package_refuses_synthetic_release_without_explicit_test_gate(
    tmp_path,
    built_release,
):
    with pytest.raises(ReleasePackageError, match="validation failed"):
        package_release(
            built_release["release_dir"],
            tmp_path / "upload",
            config_path=built_release["config_path"],
            allow_synthetic=False,
        )


def test_publication_package_requires_external_config(
    tmp_path,
    built_release,
):
    with pytest.raises(ReleasePackageError, match="--config is required"):
        package_release(
            built_release["release_dir"],
            tmp_path / "upload",
            dry_run=True,
        )


def test_package_rejects_external_config_that_differs_from_frozen_snapshots(
    tmp_path,
    built_release,
):
    config_path = built_release["config_path"]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["metadata"]["description"] = "Changed after the release was frozen."
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ReleasePackageError, match="validation failed"):
        package_release(
            built_release["release_dir"],
            tmp_path / "upload",
            config_path=config_path,
            dry_run=True,
            allow_synthetic=True,
        )


def test_package_dry_run_is_non_writing_for_a_valid_release(
    tmp_path,
    built_release,
    capsys,
):
    upload_dir = tmp_path / "upload"

    archive = package_release(
        built_release["release_dir"],
        upload_dir,
        dry_run=True,
        allow_synthetic=True,
    )
    report = json.loads(capsys.readouterr().out)

    assert report == {
        "allow_synthetic": True,
        "archive_verification": False,
        "archive_name": archive.name,
        "config_bound": False,
        "dry_run": True,
        "would_write": False,
        "would_succeed": True,
    }
    assert not upload_dir.exists()
    assert not archive.exists()


def test_archive_verification_mode_is_snapshot_only_and_nonpublishable(
    tmp_path,
    built_release,
):
    upload_dir = tmp_path / "archive-verification"

    archive = package_release(
        built_release["release_dir"],
        upload_dir,
        archive_verification=True,
    )

    assert archive.is_file()
    for name in ("README.md", "ZENODO_UPLOAD_CHECKLIST.md"):
        text = (upload_dir / name).read_text(encoding="utf-8")
        assert "ARCHIVE-VERIFICATION PACKAGE" in text
        assert "DO NOT UPLOAD OR PUBLISH" in text


def test_package_dry_run_rejects_missing_release_directory(tmp_path):
    missing = tmp_path / "missing-release"
    upload = tmp_path / "upload"

    with pytest.raises(ReleasePackageError, match="does not exist"):
        package_release(
            missing,
            upload,
            dry_run=True,
            allow_synthetic=True,
        )

    assert not upload.exists()


def test_package_refuses_nonempty_upload_directory(
    tmp_path,
    built_release,
):
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "keep.txt").write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(ReleasePackageError, match="must be absent"):
        package_release(
            built_release["release_dir"],
            upload,
            allow_synthetic=True,
        )

    assert (upload / "keep.txt").read_text(encoding="utf-8") == "do not overwrite\n"


def test_allow_synthetic_flag_cannot_weaken_a_non_test_release(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    from tools.zenodo.build_release import build_release

    config_path = release_config_factory(test_only=False)
    release_dir = build_release(
        config_path,
        safe_export_factory(),
        tmp_path / "release-output",
    )

    with pytest.raises(ReleasePackageError, match="test_only=true"):
        package_release(
            release_dir,
            tmp_path / "upload",
            allow_synthetic=True,
        )
