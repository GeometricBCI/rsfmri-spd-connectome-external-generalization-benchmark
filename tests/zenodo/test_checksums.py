from __future__ import annotations

import hashlib
import os

import pytest

from tools.zenodo.checksums import (
    build_manifest_rows,
    iter_regular_files,
    read_sha256sums,
    sha256_file,
    verify_manifest,
    verify_sha256sums,
    write_manifest,
    write_sha256sums,
)


def test_sha256_file_matches_standard_library_for_chunked_binary_input(tmp_path):
    path = tmp_path / "payload.bin"
    payload = bytes(range(256)) * 8193
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=127) == hashlib.sha256(payload).hexdigest()


def test_checksum_catalog_is_sorted_complete_and_self_excluding(tmp_path):
    root = tmp_path / "release"
    (root / "z").mkdir(parents=True)
    (root / "a.txt").write_text("alpha\n", encoding="utf-8")
    (root / "z" / "b.txt").write_text("beta\n", encoding="utf-8")
    checksum_path = root / "SHA256SUMS.txt"

    written = write_sha256sums(root, checksum_path)

    assert [relative for _, relative in written] == ["a.txt", "z/b.txt"]
    assert read_sha256sums(checksum_path) == written
    assert verify_sha256sums(root, checksum_path) == []


def test_checksum_verification_detects_tampering(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    payload = root / "artifact.txt"
    payload.write_text("before\n", encoding="utf-8")
    checksum_path = root / "SHA256SUMS.txt"
    write_sha256sums(root, checksum_path)

    payload.write_text("after\n", encoding="utf-8")

    assert any(
        "checksum mismatch" in error
        for error in verify_sha256sums(root, checksum_path)
    )


def test_checksum_verification_rejects_traversal_and_duplicates(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "safe.txt").write_text("safe\n", encoding="utf-8")
    digest = sha256_file(root / "safe.txt")
    checksum_path = root / "SHA256SUMS.txt"
    checksum_path.write_text(
        f"{digest}  ../outside.txt\n"
        f"{digest}  safe.txt\n"
        f"{digest}  safe.txt\n",
        encoding="utf-8",
    )

    errors = verify_sha256sums(root, checksum_path)

    assert any("unsafe checksum path" in error for error in errors)
    assert any("duplicate checksum path" in error for error in errors)


def test_manifest_is_relative_sorted_and_detects_content_changes(tmp_path):
    root = tmp_path / "release"
    (root / "data" / "public_metadata" / "abide").mkdir(parents=True)
    payload = root / "data" / "public_metadata" / "abide" / "metadata.tsv"
    payload.write_text("sample_uid\tage\nabide0000001\t20\n", encoding="utf-8")
    manifest_path = root / "manifest.tsv"

    rows = write_manifest(
        root,
        manifest_path,
        exclude_relative={"manifest.tsv"},
    )

    assert rows == build_manifest_rows(root, exclude_relative={"manifest.tsv"})
    assert [row["relative_path"] for row in rows] == [
        "data/public_metadata/abide/metadata.tsv"
    ]
    assert rows[0]["dataset"] == "abide"
    assert rows[0]["content_category"] == "participant_metadata"
    assert verify_manifest(
        root,
        manifest_path,
        excluded_relative={"manifest.tsv"},
    ) == []

    payload.write_text("sample_uid\tage\nabide0000001\t21\n", encoding="utf-8")
    assert any(
        "manifest checksum mismatch" in error
        for error in verify_manifest(
            root,
            manifest_path,
            excluded_relative={"manifest.tsv"},
        )
    )


def test_regular_file_iteration_never_follows_symbolic_links(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    (root / "kept.txt").write_text("safe\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("synthetic external sentinel\n", encoding="utf-8")
    link = root / "linked.txt"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symbolic links are unavailable on this platform")

    assert [path.name for path in iter_regular_files(root)] == ["kept.txt"]
