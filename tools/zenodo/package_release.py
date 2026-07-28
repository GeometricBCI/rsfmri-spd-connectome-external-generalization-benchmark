"""Create a deterministic, upload-ready archive and companion files."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping, Sequence

import yaml

from .checksums import (
    read_sha256sums,
    sha256_file,
    verify_sha256sums,
    write_manifest,
    write_sha256sums,
)
from .metadata import dataset_rights_entries, metadata_is_complete
from .schemas import (
    is_portable_relative_path,
    load_yaml,
    validate_release_bundle_documents,
)
from .validate_release import validate_release


class ReleasePackageError(RuntimeError):
    """Packaging failed closed."""


def _snapshot_bundle(release_root: Path) -> dict[str, Any]:
    metadata_dir = release_root / "metadata"
    paths = {
        "config": metadata_dir / "release_config_snapshot.yaml",
        "dataset_policy": metadata_dir / "release_policy_snapshot.yaml",
        "metadata_allowlist": metadata_dir / "metadata_allowlist_snapshot.yaml",
        "forbidden_patterns": metadata_dir / "forbidden_patterns_snapshot.yaml",
        "manual_approvals": metadata_dir / "manual_approvals_snapshot.yaml",
    }
    if any(not path.is_file() for path in paths.values()):
        raise ReleasePackageError("release policy snapshots are incomplete")
    bundle = {key: load_yaml(path) for key, path in paths.items()}
    try:
        validate_release_bundle_documents(bundle, snapshot=True)
    except ValueError as exc:
        raise ReleasePackageError("release policy snapshots are invalid") from exc
    return bundle


def _refresh_release_catalogs(
    release_root: Path, bundle: Mapping[str, Any]
) -> None:
    excluded = {"manifests/manifest.tsv", "manifests/SHA256SUMS.txt"}
    write_manifest(
        release_root,
        release_root / "manifests" / "manifest.tsv",
        bundle=bundle,
        exclude_relative=excluded,
    )
    write_sha256sums(
        release_root,
        release_root / "manifests" / "SHA256SUMS.txt",
        exclude_relative={"manifests/SHA256SUMS.txt"},
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800  # UTF-8 filenames
    return info


def _write_deterministic_zip(release_root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        normalized_names: set[str] = set()
        for path in sorted(
            (item for item in release_root.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(release_root).as_posix(),
        ):
            if path.is_symlink():
                raise ReleasePackageError(
                    "release changed during packaging or contains a symbolic link"
                )
            name = (
                PurePosixPath(release_root.name)
                / PurePosixPath(path.relative_to(release_root).as_posix())
            ).as_posix()
            normalized = name.casefold()
            if (
                not is_portable_relative_path(name)
                or normalized in normalized_names
            ):
                raise ReleasePackageError(
                    "release contains a non-portable or case-colliding path"
                )
            normalized_names.add(normalized)
            with path.open("rb") as source, archive.open(
                _zip_info(name),
                "w",
                force_zip64=True,
            ) as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)


def _safe_extract(archive_path: Path, destination: Path) -> Path:
    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        normalized_names: set[str] = set()
        for name in names:
            pure = PurePosixPath(name)
            windows = PureWindowsPath(name)
            normalized = pure.as_posix()
            if (
                not is_portable_relative_path(name)
                or not name
                or "\x00" in name
                or "\\" in name
                or pure.is_absolute()
                or windows.is_absolute()
                or bool(windows.drive)
                or ".." in pure.parts
                or "." in pure.parts
                or any(
                    not part
                    or part.endswith((" ", "."))
                    or part.split(".", 1)[0].casefold()
                    in {
                        "con",
                        "prn",
                        "aux",
                        "nul",
                        "com1",
                        "com2",
                        "com3",
                        "com4",
                        "com5",
                        "com6",
                        "com7",
                        "com8",
                        "com9",
                        "lpt1",
                        "lpt2",
                        "lpt3",
                        "lpt4",
                        "lpt5",
                        "lpt6",
                        "lpt7",
                        "lpt8",
                        "lpt9",
                    }
                    for part in pure.parts
                )
                or normalized.casefold() in normalized_names
            ):
                raise ReleasePackageError("archive contains an unsafe member path")
            normalized_names.add(normalized.casefold())
        archive.extractall(destination)
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise ReleasePackageError("archive must extract to exactly one release root")
    return roots[0]


def _write_upload_readme(
    upload_dir: Path,
    archive_name: str,
    *,
    synthetic_test: bool,
    archive_verification: bool,
) -> None:
    if synthetic_test:
        banner = "> **SYNTHETIC TEST PACKAGE — DO NOT UPLOAD OR PUBLISH.**\n\n"
    elif archive_verification:
        banner = (
            "> **ARCHIVE-VERIFICATION PACKAGE — SNAPSHOT-ONLY; "
            "DO NOT UPLOAD OR PUBLISH.**\n\n"
        )
    else:
        banner = ""
    (upload_dir / "README.md").write_text(
        banner
        + "\n".join(
            [
                "# Zenodo upload files",
                "",
                "Upload only the files in this directory after the checklist is",
                "completed by an authorized human reviewer.",
                "",
                f"- `{archive_name}`: the structured, frozen dataset release.",
                "- `README.md`: this upload-file guide.",
                "- `LICENSES.md`: code, derived-data, documentation, and source-data terms.",
                "- `SHA256SUMS.txt`: checksums for every companion file except itself.",
                "- `zenodo_record_metadata.json`: reviewed metadata companion/API input.",
                "- `ZENODO_UPLOAD_CHECKLIST.md`: required manual web-form and policy checks.",
                "",
                "The metadata JSON is a companion file. Uploading it does not guarantee",
                "that every Zenodo web-interface field will be populated automatically.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_upload_checklist(
    upload_dir: Path,
    archive_name: str,
    *,
    synthetic_test: bool,
    archive_verification: bool,
    bundle: Mapping[str, Any],
) -> None:
    if synthetic_test:
        banner = "> **SYNTHETIC TEST PACKAGE — DO NOT UPLOAD OR PUBLISH.**\n\n"
    elif archive_verification:
        banner = (
            "> **ARCHIVE-VERIFICATION PACKAGE — SNAPSHOT-ONLY; "
            "DO NOT UPLOAD OR PUBLISH.**\n\n"
        )
    else:
        banner = ""
    rights = dict(dataset_rights_entries(bundle["config"]))
    if set(rights) != {"camcan", "abide", "cobre"} or any(
        value.get("status") != "approved_derived_data_release"
        for value in rights.values()
    ):
        raise ReleasePackageError(
            "upload checklist requires all three confirmed dataset-rights records"
        )
    template_path = (
        Path(__file__).resolve().parents[2]
        / "release_templates"
        / "ZENODO_UPLOAD_CHECKLIST.md"
    )
    try:
        rendered = template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReleasePackageError("upload checklist template is unavailable") from exc
    rendered = rendered.replace("{{SYNTHETIC_BANNER}}", banner).replace(
        "{{ARCHIVE_NAME}}", archive_name
    )
    if re.search(r"\{\{[A-Z0-9_]+\}\}", rendered):
        raise ReleasePackageError("upload checklist template is unresolved")
    (upload_dir / "ZENODO_UPLOAD_CHECKLIST.md").write_text(
        rendered,
        encoding="utf-8",
    )


def package_release(
    release_dir: str | Path,
    upload_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    dry_run: bool = False,
    allow_synthetic: bool = False,
    archive_verification: bool = False,
) -> Path:
    """Validate, archive twice, compare, extract, and revalidate.

    Publication packaging requires ``config_path`` so every validation pass
    can compare the external release-policy bundle with the frozen snapshots.
    Snapshot-only operation is limited to explicit archive-verification and
    synthetic-test modes.
    """

    supplied_release = Path(release_dir)
    if supplied_release.is_symlink():
        raise ReleasePackageError("release directory must not be a symbolic link")
    release_root = supplied_release.resolve()
    supplied_destination = Path(upload_dir)
    if supplied_destination.is_symlink():
        raise ReleasePackageError("upload directory must not be a symbolic link")
    destination = supplied_destination.resolve()
    archive_name = f"{release_root.name}.zip"
    final_archive = destination / archive_name
    if not release_root.is_dir():
        raise ReleasePackageError("release directory does not exist")
    if any(path.is_symlink() for path in release_root.rglob("*")):
        raise ReleasePackageError("release directory must not contain symbolic links")
    if (
        destination == release_root
        or destination.is_relative_to(release_root)
        or release_root.is_relative_to(destination)
    ):
        raise ReleasePackageError(
            "upload directory and release directory must not overlap"
        )
    if destination.exists():
        raise ReleasePackageError(
            "upload directory must be absent; refusing to overwrite"
        )
    if config_path is None and not (archive_verification or allow_synthetic):
        raise ReleasePackageError(
            "--config is required for publication packaging; "
            "snapshot-only operation requires --archive-verification or "
            "--allow-synthetic-test-package"
        )
    bundle = _snapshot_bundle(release_root)
    release_config = bundle["config"].get("release", {})
    if allow_synthetic and release_config.get("test_only") is not True:
        raise ReleasePackageError(
            "--allow-synthetic-test-package requires release.test_only=true"
        )
    publication_ready = not (allow_synthetic or archive_verification)
    complete, problems = metadata_is_complete(bundle)
    if not complete:
        raise ReleasePackageError(
            f"required approvals or metadata are incomplete ({len(problems)} item(s))"
        )
    preview = validate_release(
        release_root,
        config_path=config_path,
        publication_ready=publication_ready,
        write_reports=False,
    )
    if not preview.ok:
        raise ReleasePackageError(
            f"release validation failed ({len(preview.errors)} error(s))"
        )
    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "would_write": False,
                    "would_succeed": True,
                    "archive_name": archive_name,
                    "allow_synthetic": allow_synthetic,
                    "archive_verification": archive_verification,
                    "config_bound": config_path is not None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return final_archive
    validation = validate_release(
        release_root,
        config_path=config_path,
        publication_ready=publication_ready,
        write_reports=True,
    )
    if not validation.ok:
        raise ReleasePackageError(
            f"release validation failed ({len(validation.errors)} error(s))"
        )
    _refresh_release_catalogs(release_root, bundle)
    confirmation = validate_release(
        release_root,
        config_path=config_path,
        publication_ready=publication_ready,
        write_reports=False,
    )
    if not confirmation.ok:
        raise ReleasePackageError(
            f"post-checksum validation failed ({len(confirmation.errors)} error(s))"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="zenodo-package-",
        dir=destination.parent,
    ) as temp_name:
        temp = Path(temp_name)
        frozen_root = temp / "frozen" / release_root.name
        frozen_root.parent.mkdir()
        shutil.copytree(release_root, frozen_root, symlinks=True)
        frozen_validation = validate_release(
            frozen_root,
            config_path=config_path,
            publication_ready=publication_ready,
            write_reports=False,
        )
        if not frozen_validation.ok:
            raise ReleasePackageError(
                "frozen release snapshot did not pass revalidation"
            )
        first = temp / "first.zip"
        second = temp / "second.zip"
        _write_deterministic_zip(frozen_root, first)
        _write_deterministic_zip(frozen_root, second)
        if sha256_file(first) != sha256_file(second):
            raise ReleasePackageError("archive generation is not deterministic")
        extracted_root = _safe_extract(first, temp / "extracted")
        extracted_validation = validate_release(
            extracted_root,
            config_path=config_path,
            publication_ready=publication_ready,
            write_reports=False,
        )
        if not extracted_validation.ok:
            raise ReleasePackageError(
                "extracted archive did not pass release revalidation"
            )
        staged_upload = temp / "upload_files"
        staged_upload.mkdir()
        shutil.copyfile(first, staged_upload / archive_name)
        shutil.copyfile(frozen_root / "LICENSES.md", staged_upload / "LICENSES.md")
        shutil.copyfile(
            frozen_root / "metadata" / "zenodo_record_metadata.json",
            staged_upload / "zenodo_record_metadata.json",
        )
        _write_upload_readme(
            staged_upload,
            archive_name,
            synthetic_test=allow_synthetic,
            archive_verification=archive_verification,
        )
        _write_upload_checklist(
            staged_upload,
            archive_name,
            synthetic_test=allow_synthetic,
            archive_verification=archive_verification,
            bundle=bundle,
        )
        sums = staged_upload / "SHA256SUMS.txt"
        write_sha256sums(
            staged_upload,
            sums,
            exclude_relative={"SHA256SUMS.txt"},
        )
        checksum_errors = verify_sha256sums(staged_upload, sums)
        checksum_paths = {relative for _, relative in read_sha256sums(sums)}
        expected_paths = {
            path.relative_to(staged_upload).as_posix()
            for path in staged_upload.iterdir()
            if path.is_file() and path.name != "SHA256SUMS.txt"
        }
        if checksum_errors or checksum_paths != expected_paths:
            raise ReleasePackageError(
                "upload companion checksum verification failed"
            )
        if (
            (staged_upload / "LICENSES.md").read_bytes()
            != (extracted_root / "LICENSES.md").read_bytes()
            or (staged_upload / "zenodo_record_metadata.json").read_bytes()
            != (
                extracted_root
                / "metadata"
                / "zenodo_record_metadata.json"
            ).read_bytes()
        ):
            raise ReleasePackageError(
                "upload companions diverge from the frozen archive"
            )
        os.replace(staged_upload, destination)
    return final_archive


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic upload-ready Zenodo archive."
    )
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--upload-dir", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        help=(
            "External release config whose complete policy bundle must exactly "
            "match the frozen release snapshots; required for publication "
            "packaging."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--archive-verification",
        action="store_true",
        help=(
            "Allow snapshot-only verification of a preserved archive. This "
            "mode does not establish source-config binding for publication."
        ),
    )
    parser.add_argument(
        "--allow-synthetic-test-package",
        action="store_true",
        help="Package an explicitly test_only release; never upload this artifact.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        archive = package_release(
            args.release_dir,
            args.upload_dir,
            config_path=args.config,
            dry_run=args.dry_run,
            allow_synthetic=args.allow_synthetic_test_package,
            archive_verification=args.archive_verification,
        )
    except (ReleasePackageError, ValueError, OSError, yaml.YAMLError) as exc:
        print(f"release packaging refused: {exc}", file=sys.stderr)
        return 2
    if not args.dry_run:
        print(
            json.dumps(
                {
                    "ok": True,
                    "archive": archive.name,
                    "sha256": sha256_file(archive),
                    "archive_verification": args.archive_verification,
                    "config_bound": args.config is not None,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
