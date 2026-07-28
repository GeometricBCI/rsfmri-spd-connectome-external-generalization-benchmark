import numpy as np
import pandas as pd
import pytest

from spd_connectome_benchmark.data_contract import (
    validate_benchmark_arrays,
    validate_spd_connectomes,
    validate_split_disjointness,
    validate_timeseries,
)


def synthetic_spd(n_scans=6, n_regions=4, seed=0):
    rng = np.random.RandomState(seed)
    factors = rng.normal(size=(n_scans, n_regions, n_regions))
    return factors @ np.swapaxes(factors, -1, -2) + 1e-4 * np.eye(n_regions)


def test_valid_synthetic_contract():
    matrices = synthetic_spd()
    validated = validate_benchmark_arrays(
        matrices,
        targets=np.linspace(20, 70, len(matrices)),
        dataset_labels=np.array(["a", "a", "b", "b", "c", "c"]),
        subject_groups=np.array(["s1", "s2", "s3", "s4", "s5", "s6"]),
        expected_regions=4,
    )

    assert validated.connectomes.shape == (6, 4, 4)


@pytest.mark.parametrize("mutation", ["nonsquare", "nan", "asymmetric", "nonspd", "integer"])
def test_malformed_connectomes_are_rejected(mutation):
    matrices = synthetic_spd()
    if mutation == "nonsquare":
        matrices = matrices[:, :, :-1]
    elif mutation == "nan":
        matrices[0, 0, 0] = np.nan
    elif mutation == "asymmetric":
        matrices[0, 0, 1] += 1
    elif mutation == "nonspd":
        matrices[0, 0, 0] = -100
    elif mutation == "integer":
        matrices = matrices.astype(int)

    with pytest.raises(ValueError):
        validate_spd_connectomes(matrices)


def test_timeseries_requires_consistent_regions_and_finite_values():
    valid = [np.ones((20, 4)), np.ones((18, 4))]
    assert len(validate_timeseries(valid, expected_regions=4)) == 2

    invalid = [np.ones((20, 4)), np.ones((18, 5))]
    with pytest.raises(ValueError, match="same atlas region count"):
        validate_timeseries(invalid)


def test_split_validation_rejects_subject_overlap():
    groups = np.array(["a", "a", "b", "c"])
    with pytest.raises(ValueError, match="subject groups overlap"):
        validate_split_disjointness(
            np.array([0, 2]),
            np.array([1, 3]),
            n_samples=4,
            subject_groups=groups,
        )


@pytest.mark.parametrize("bad_label", [None, np.nan, pd.NA, 7, "  "])
def test_dataset_labels_reject_missing_non_string_or_empty_values(bad_label):
    matrices = synthetic_spd()
    labels = np.array(["a"] * len(matrices), dtype=object)
    labels[0] = bad_label

    with pytest.raises(ValueError):
        validate_benchmark_arrays(
            matrices,
            targets=np.arange(len(matrices), dtype=float),
            dataset_labels=labels,
            subject_groups=np.arange(len(matrices)),
        )


def test_subject_groups_allow_numeric_scalars_but_reject_missing():
    matrices = synthetic_spd()
    common = {
        "connectomes": matrices,
        "targets": np.arange(len(matrices), dtype=float),
        "dataset_labels": np.array(["a"] * len(matrices)),
    }

    validated = validate_benchmark_arrays(
        **common,
        subject_groups=np.arange(len(matrices)),
    )
    assert np.array_equal(validated.subject_groups, np.arange(len(matrices)))

    missing_groups = np.arange(len(matrices), dtype=object)
    missing_groups[0] = None
    with pytest.raises(ValueError, match="missing"):
        validate_benchmark_arrays(**common, subject_groups=missing_groups)


@pytest.mark.parametrize(
    ("train_idx", "test_idx"),
    [
        (np.array([0, 0, 1]), np.array([2, 3])),
        (np.array([0, 1]), np.array([2, 2, 3])),
    ],
)
def test_split_validation_rejects_duplicate_indices(train_idx, test_idx):
    with pytest.raises(ValueError, match="duplicates"):
        validate_split_disjointness(
            train_idx,
            test_idx,
            n_samples=4,
        )
