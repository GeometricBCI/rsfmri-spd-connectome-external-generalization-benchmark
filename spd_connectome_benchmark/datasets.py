"""Dataset and atlas registry for the benchmark's public code interface."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable


@dataclass(frozen=True)
class DatasetSpec:
    """Logical description of one dataset supported by the benchmark code."""

    name: str
    display_name: str
    access_class: str


@dataclass(frozen=True)
class AtlasSpec:
    """Logical atlas name and the expected number of regions."""

    name: str
    n_regions: int
    representation: str


_DATASETS = {
    "abide": DatasetSpec("abide", "ABIDE", "approval-required"),
    "adni": DatasetSpec("adni", "ADNI", "restricted"),
    "oasis3": DatasetSpec("oasis3", "OASIS-3", "restricted"),
    "camcan": DatasetSpec("camcan", "Cam-CAN", "approval-required"),
    "cobre": DatasetSpec("cobre", "COBRE", "approval-required"),
    "adnidod": DatasetSpec("adnidod", "ADNI-DOD", "restricted"),
}

DATASET_REGISTRY = MappingProxyType(_DATASETS)

_DATASET_ALIASES = {
    "oasis-3": "oasis3",
    "cam-can": "camcan",
    "adni-dod": "adnidod",
}

_ATLASES = {
    "schaefer_100": AtlasSpec("schaefer_100", 100, "labels"),
    "msdl_39": AtlasSpec("msdl_39", 39, "maps"),
}

ATLAS_REGISTRY = MappingProxyType(_ATLASES)

_ATLAS_ALIASES = {
    "schaefer100": "schaefer_100",
    "schaefer-100": "schaefer_100",
    "msdl": "msdl_39",
    "msdl39": "msdl_39",
    "msdl-39": "msdl_39",
}

_SAFE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _normalized_name(value: str, *, kind: str) -> str:
    name = str(value).strip().lower()
    if not name or not _SAFE_NAME.fullmatch(name):
        raise ValueError(f"Invalid {kind} name: {value!r}")
    return name


def canonical_dataset_name(value: str) -> str:
    """Return a registered dataset name, rejecting unknown/non-public inputs."""
    name = _normalized_name(value, kind="dataset")
    name = _DATASET_ALIASES.get(name, name)
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unsupported dataset {value!r}. Unknown datasets are non-public "
            "by default and must be reviewed before registration."
        )
    return name


def canonical_dataset_names(values: Iterable[str]) -> tuple[str, ...]:
    """Normalize dataset names while preserving order and rejecting duplicates."""
    names = tuple(canonical_dataset_name(value) for value in values)
    if not names:
        raise ValueError("At least one dataset is required.")
    if len(set(names)) != len(names):
        raise ValueError("Dataset selections must not contain duplicates.")
    return names


def canonical_atlas_name(value: str) -> str:
    """Return a supported atlas name."""
    name = _normalized_name(value, kind="atlas")
    name = _ATLAS_ALIASES.get(name, name)
    if name not in ATLAS_REGISTRY:
        raise ValueError(
            f"Unsupported atlas {value!r}. This release implements only "
            f"{', '.join(ATLAS_REGISTRY)}."
        )
    return name


def prepared_dataset_path(input_root: Path, atlas: str, dataset: str) -> Path:
    """Resolve the prepared-file path without opening or deserializing it."""
    atlas_name = canonical_atlas_name(atlas)
    dataset_name = canonical_dataset_name(dataset)
    return Path(input_root) / f"atlas_{atlas_name}" / f"{dataset_name}_X_y.pkl"
