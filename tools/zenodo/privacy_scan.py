"""Configurable, redacting privacy scan for release filenames and text."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Mapping

from .schemas import (
    CAMCAN_METADATA_RELATIVE_PATH,
    SAFE_UID_RE,
    Finding,
    load_yaml,
    metadata_column_is_forbidden,
)


TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
BUILTIN_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        "ABSOLUTE_POSIX_PATH",
        r"(?<![:/\w])/(?!/|summary>|details>)[^\s<>'\"`]+",
        "absolute local POSIX path",
    ),
    (
        "ABSOLUTE_WINDOWS_PATH",
        r"\b[A-Za-z]:[\\/][^\s<>'\"`]+",
        "absolute local Windows path",
    ),
    (
        "FILE_URI",
        r"\bfile://[^\s<>'\"`]+",
        "local file URI",
    ),
    (
        "HOME_RELATIVE_PATH",
        r"(?<![\w])~[\\/][^\s<>'\"`]+",
        "home-relative local path",
    ),
    (
        "UNC_PATH",
        r"(?<!\\)\\\\[^\\/\s<>'\"`]+[\\/][^\s<>'\"`]+",
        "UNC or network-share path",
    ),
    (
        "EMAIL_ADDRESS",
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "email address outside approved creator metadata",
    ),
    (
        "ADNI_STYLE_IDENTIFIER",
        r"\b\d{3}_S_\d{4}\b",
        "ADNI-style participant identifier",
    ),
    (
        "BIDS_PARTICIPANT_IDENTIFIER",
        r"\b(?:sub|ses)-[A-Za-z0-9]{2,}\b",
        "BIDS participant or session identifier",
    ),
    (
        "SOURCE_PICKLE_NAME",
        r"\b[\w.-]+\.p(?:ickle|kl)\b",
        "source pickle filename",
    ),
    (
        "RELATIVE_SOURCE_PATH",
        r"(?i)\b(?:sourcedata|raw_data|local_data|derivatives)[\\/][^\s<>'\"`]+",
        "relative source-data path",
    ),
    (
        "POSSIBLE_SECRET",
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*[\"']?([A-Za-z0-9_./+=-]{8,})",
        "possible token, credential, or API key",
    ),
)


def redact(value: str) -> str:
    if len(value) <= 6:
        return value[:1] + "…" + value[-1:]
    visible = max(2, min(6, len(value) // 5))
    return value[:visible] + "…" + value[-visible:]


def _parse_patterns(
    raw_patterns: object,
) -> list[tuple[str, str, str]]:
    patterns: list[tuple[str, str, str]] = []
    if not isinstance(raw_patterns, list):
        raise ValueError("forbidden-pattern config must contain a patterns list")
    for entry in raw_patterns:
        if not isinstance(entry, Mapping):
            raise ValueError("each forbidden pattern must be a mapping")
        code = str(entry.get("code", entry.get("id", ""))).strip().upper()
        expression = str(entry.get("regex", "")).strip()
        description = str(entry.get("description", code)).strip()
        if not code or not expression:
            raise ValueError("configured patterns require code and regex")
        re.compile(expression)
        patterns.append((code, expression, description))
    return patterns


def _configured_rules(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {
            "generic_patterns": [],
            "filename_patterns": [],
            "text_patterns": [],
            "forbidden_extensions": set(),
            "forbidden_path_components": set(),
            "tabular_exact": set(),
            "tabular_regex": [],
            "text_extensions": set(TEXT_SUFFIXES),
        }
    config = load_yaml(config_path)
    tabular = config.get("forbidden_tabular_columns", {})
    if not isinstance(tabular, Mapping):
        raise ValueError("forbidden_tabular_columns must be a mapping")
    scanner = config.get("scanner", {})
    if not isinstance(scanner, Mapping):
        raise ValueError("privacy scanner config must be a mapping")
    expressions = tabular.get("regex", [])
    if not isinstance(expressions, list):
        raise ValueError("forbidden tabular regex rules must be a list")
    return {
        "generic_patterns": _parse_patterns(config.get("patterns", [])),
        "filename_patterns": _parse_patterns(
            config.get("forbidden_filename_patterns", [])
        ),
        "text_patterns": _parse_patterns(
            config.get("forbidden_text_patterns", [])
        ),
        "forbidden_extensions": {
            str(value).lower()
            for value in config.get("forbidden_extensions", [])
        },
        "forbidden_path_components": {
            str(value).casefold()
            for value in config.get("forbidden_path_components", [])
        },
        "tabular_exact": {
            str(value).strip().lower() for value in tabular.get("exact", [])
        },
        "tabular_regex": [
            re.compile(str(value), re.IGNORECASE) for value in expressions
        ],
        "text_extensions": {
            str(value).lower()
            for value in scanner.get("text_extensions", TEXT_SUFFIXES)
        },
    }


def _approved_creator_emails(config: Mapping[str, Any] | None) -> set[str]:
    if not config:
        return set()
    metadata = config.get("metadata", {})
    creators = metadata.get("creators", []) if isinstance(metadata, Mapping) else []
    approved = set()
    if isinstance(creators, list):
        for creator in creators:
            if isinstance(creator, Mapping) and creator.get("email"):
                approved.add(str(creator["email"]).lower())
    return approved


def _scan_tabular_header(
    path: Path,
    relative: str,
    *,
    configured_exact: set[str],
    configured_regex: list[re.Pattern[str]],
) -> list[Finding]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(
                csv.reader(handle, delimiter=delimiter, strict=True),
                [],
            )
    except (OSError, UnicodeError, csv.Error):
        return [
            Finding(
                "TABULAR_HEADER_SCAN_FAILED",
                "error",
                "tabular header could not be safely decoded or read",
                relative,
            )
        ]
    findings: list[Finding] = []
    for column in header:
        normalized = re.sub(
            r"[^a-z0-9]+", "_", str(column).strip().lower()
        ).strip("_")
        if (
            metadata_column_is_forbidden(column)
            or normalized in configured_exact
            or any(pattern.search(normalized) for pattern in configured_regex)
        ):
            findings.append(
                Finding(
                    "IDENTIFIER_COLUMN",
                    "error",
                    "tabular header contains a forbidden identifier column",
                    relative,
                    1,
                    redact(column),
                )
            )
    if "sample_uid" in header:
        uid_index = header.index("sample_uid")
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = csv.reader(handle, delimiter=delimiter, strict=True)
                next(rows, None)
                for line_number, row in enumerate(rows, start=2):
                    if (
                        uid_index >= len(row)
                        or SAFE_UID_RE.fullmatch(row[uid_index].strip()) is None
                    ):
                        value = row[uid_index] if uid_index < len(row) else ""
                        findings.append(
                            Finding(
                                "UNSAFE_SAMPLE_UID_VALUE",
                                "error",
                                "sample_uid does not match the release-safe schema",
                                relative,
                                line_number,
                                redact(value),
                            )
                        )
        except (OSError, UnicodeError, csv.Error):
            findings.append(
                Finding(
                    "TABULAR_VALUE_SCAN_FAILED",
                    "error",
                    "tabular values could not be safely decoded or read",
                    relative,
                )
            )
    return findings


def scan_release(
    root: str | Path,
    config_path: str | Path | None = None,
    *,
    release_config: Mapping[str, Any] | None = None,
) -> list[Finding]:
    """Scan release-visible paths and text without echoing full matches."""

    base = Path(root).resolve()
    findings: list[Finding] = []
    rules = _configured_rules(config_path)
    generic_patterns = [
        (code, re.compile(expression, re.IGNORECASE), description)
        for code, expression, description in (
            *BUILTIN_PATTERNS,
            *rules["generic_patterns"],
        )
    ]
    filename_patterns = [
        *generic_patterns,
        *[
            (code, re.compile(expression, re.IGNORECASE), description)
            for code, expression, description in rules["filename_patterns"]
        ],
    ]
    text_patterns = [
        *generic_patterns,
        *[
            (code, re.compile(expression, re.IGNORECASE), description)
            for code, expression, description in rules["text_patterns"]
        ],
    ]
    approved_emails = _approved_creator_emails(release_config)
    for path in sorted(base.rglob("*")):
        relative = path.relative_to(base).as_posix()
        reported_relative = relative
        for _, pattern, _ in filename_patterns:
            reported_relative = pattern.sub(
                lambda match: redact(match.group(0)),
                reported_relative,
            )
        for code, pattern, description in filename_patterns:
            if (
                code == "PARTICIPANT_TABLE"
                and relative == CAMCAN_METADATA_RELATIVE_PATH.as_posix()
            ):
                continue
            match = pattern.search(relative)
            if match:
                findings.append(
                    Finding(
                        code,
                        "error",
                        f"filename contains a likely {description}",
                        reported_relative,
                        redacted_value=redact(match.group(0)),
                    )
                )
        path_components = {part.casefold() for part in Path(relative).parts}
        for component in sorted(
            path_components & rules["forbidden_path_components"]
        ):
            findings.append(
                Finding(
                    "FORBIDDEN_PATH_COMPONENT",
                    "error",
                    "path contains a forbidden source or working-directory component",
                    reported_relative,
                    redacted_value=redact(component),
                )
            )
        if path.is_symlink() or not path.is_file():
            continue
        lower_name = path.name.lower()
        if any(
            lower_name.endswith(str(extension))
            for extension in rules["forbidden_extensions"]
        ):
            findings.append(
                Finding(
                    "FORBIDDEN_FILE_EXTENSION",
                    "error",
                    "file uses a forbidden raw-data or executable-serialized extension",
                    reported_relative,
                    redacted_value=redact(path.suffix.lower()),
                )
            )
        if path.suffix.lower() in {".csv", ".tsv"}:
            findings.extend(
                _scan_tabular_header(
                    path,
                    reported_relative,
                    configured_exact=rules["tabular_exact"],
                    configured_regex=rules["tabular_regex"],
                )
            )
        if path.suffix.lower() not in rules["text_extensions"]:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            findings.append(
                Finding(
                    "TEXT_SCAN_FAILED",
                    "error",
                    "text file could not be safely decoded or read",
                    reported_relative,
                )
            )
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            # Regex source in this generated policy snapshot intentionally
            # contains examples of forbidden syntax. Only that exact YAML
            # field is exempt; descriptions and every other value are scanned.
            if (
                relative == "metadata/forbidden_patterns_snapshot.yaml"
                and line.lstrip().startswith("regex:")
            ):
                continue
            for code, pattern, description in text_patterns:
                for match in pattern.finditer(line):
                    value = match.group(0)
                    if code == "EMAIL_ADDRESS" and value.lower() in approved_emails:
                        continue
                    findings.append(
                        Finding(
                            code,
                            "error",
                            f"text contains a likely {description}",
                            reported_relative,
                            line_number,
                            redact(value),
                        )
                    )
    return findings
