from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from tools.zenodo.export_internal import _load_internal_manifest, _source_binding


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _run_module(module, *arguments):
    return subprocess.run(
        [sys.executable, "-B", "-m", module, *map(str, arguments)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_release_builder_cli_dry_run_is_path_redacted_and_non_writing(
    tmp_path,
    release_config_factory,
    safe_export_factory,
):
    config_path = release_config_factory()
    safe_export = safe_export_factory()
    output_dir = tmp_path / "output"

    completed = _run_module(
        "tools.zenodo.build_release",
        "--config",
        config_path,
        "--safe-export-dir",
        safe_export,
        "--output-dir",
        output_dir,
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["dry_run"] is True
    assert report["would_write"] is False
    assert str(tmp_path) not in completed.stdout
    assert not output_dir.exists()


def test_release_validator_cli_dry_run_is_non_writing(tmp_path):
    release_dir = tmp_path / "existing-release"
    release_dir.mkdir()

    completed = _run_module(
        "tools.zenodo.validate_release",
        "--release-dir",
        release_dir,
        "--dry-run",
    )

    assert completed.returncode == 2, completed.stderr
    report = json.loads(completed.stdout)
    assert report["dry_run"] is True
    assert report["publication_ready"] is False
    assert report["release_directory_exists"] is True
    assert report["would_write_reports"] is False
    assert report["validation"]["ok"] is False
    assert list(release_dir.iterdir()) == []


def test_release_package_cli_dry_run_is_non_writing(
    tmp_path,
    built_release,
):
    upload_dir = tmp_path / "upload"

    completed = _run_module(
        "tools.zenodo.package_release",
        "--release-dir",
        built_release["release_dir"],
        "--upload-dir",
        upload_dir,
        "--config",
        built_release["config_path"],
        "--allow-synthetic-test-package",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["dry_run"] is True
    assert report["would_write"] is False
    assert report["archive_verification"] is False
    assert report["config_bound"] is True
    assert str(tmp_path) not in completed.stdout
    assert not upload_dir.exists()


def test_release_package_cli_requires_config_for_publication_mode(
    tmp_path,
    built_release,
):
    upload_dir = tmp_path / "upload"

    completed = _run_module(
        "tools.zenodo.package_release",
        "--release-dir",
        built_release["release_dir"],
        "--upload-dir",
        upload_dir,
        "--dry-run",
    )

    assert completed.returncode == 2
    assert "--config is required for publication packaging" in completed.stderr
    assert not upload_dir.exists()


def test_release_package_cli_allows_explicit_archive_verification_mode(
    tmp_path,
    built_release,
):
    upload_dir = tmp_path / "upload"

    completed = _run_module(
        "tools.zenodo.package_release",
        "--release-dir",
        built_release["release_dir"],
        "--upload-dir",
        upload_dir,
        "--archive-verification",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["archive_verification"] is True
    assert report["config_bound"] is False
    assert report["would_write"] is False
    assert not upload_dir.exists()


def test_release_package_cli_dry_run_rejects_missing_release(tmp_path):
    completed = _run_module(
        "tools.zenodo.package_release",
        "--release-dir",
        tmp_path / "missing-release",
        "--upload-dir",
        tmp_path / "upload",
        "--allow-synthetic-test-package",
        "--dry-run",
    )

    assert completed.returncode == 2
    assert "does not exist" in completed.stderr
    assert not (tmp_path / "upload").exists()


def test_stage1_cli_requires_explicit_trust_flag_before_any_input_access(
    tmp_path,
    release_config_factory,
    internal_manifest_factory,
):
    output_dir = tmp_path / "safe-export"

    completed = _run_module(
        "tools.zenodo.export_internal",
        "--config",
        release_config_factory(),
        "--input-manifest",
        internal_manifest_factory(),
        "--output-dir",
        output_dir,
        "--dry-run",
    )

    assert completed.returncode == 2
    assert "--trusted-internal-input" in completed.stderr
    assert not output_dir.exists()


def test_stage1_cli_dry_run_never_deserializes_or_writes(
    tmp_path,
    release_config_factory,
    internal_manifest_factory,
):
    output_dir = tmp_path / "safe-export"
    manifest_path = internal_manifest_factory()
    manifest = _load_internal_manifest(manifest_path)
    config_path = release_config_factory(
        source_binding_sha256=_source_binding(
            manifest["datasets"][0], manifest["export_namespace"]
        )
    )

    completed = _run_module(
        "tools.zenodo.export_internal",
        "--config",
        config_path,
        "--input-manifest",
        manifest_path,
        "--output-dir",
        output_dir,
        "--trusted-internal-input",
        "--dry-run",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["dry_run"] is True
    assert report["would_deserialize_pickle"] is False
    assert report["would_succeed"] is True
    assert str(tmp_path) not in completed.stdout
    assert not output_dir.exists()
