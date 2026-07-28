import pytest

from spd_connectome_benchmark.datasets import (
    ATLAS_REGISTRY,
    DATASET_REGISTRY,
    canonical_atlas_name,
    canonical_dataset_name,
    canonical_dataset_names,
)


def test_release_registry_has_the_six_benchmark_datasets():
    assert set(DATASET_REGISTRY) == {
        "abide",
        "adni",
        "oasis3",
        "camcan",
        "cobre",
        "adnidod",
    }


def test_registry_aliases_are_explicit():
    assert canonical_dataset_name("Cam-CAN") == "camcan"
    assert canonical_dataset_name("OASIS-3") == "oasis3"
    assert canonical_atlas_name("msdl") == "msdl_39"
    assert set(ATLAS_REGISTRY) == {"schaefer_100", "msdl_39"}


def test_unknown_dataset_is_non_public_by_default():
    with pytest.raises(ValueError, match="non-public"):
        canonical_dataset_name("1000BRAINS")


def test_duplicate_dataset_selection_is_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        canonical_dataset_names(["adni", "ADNI"])
