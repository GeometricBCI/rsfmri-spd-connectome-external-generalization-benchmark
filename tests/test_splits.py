import numpy as np

from spd_connectome_benchmark.data_contract import validate_split_disjointness
from spd_connectome_benchmark.protocols import harmonization_policy
from spd_connectome_benchmark.splits import (
    dispatch_cv_splits,
    make_groupkfold_splits,
    make_lodo_splits,
    split_train_validation_by_group,
)


def test_groupkfold_is_subject_disjoint():
    groups = np.repeat(np.arange(6), 2)
    X = np.zeros((len(groups), 1))
    y = np.arange(len(groups))

    splits, names = make_groupkfold_splits(X, y, groups, n_splits=3)

    assert names == ["R1", "R2", "R3"]
    for train_idx, test_idx in splits:
        validate_split_disjointness(
            train_idx,
            test_idx,
            n_samples=len(groups),
            subject_groups=groups,
        )


def test_lodo_holds_out_one_complete_dataset_once():
    dataset_ids = np.array(["a", "a", "b", "b", "c", "c"])
    splits, names = make_lodo_splits(dataset_ids)

    assert names == ["TEST_a", "TEST_b", "TEST_c"]
    observed_test = []
    for train_idx, test_idx in splits:
        assert len(set(dataset_ids[test_idx])) == 1
        assert set(dataset_ids[train_idx]).isdisjoint(set(dataset_ids[test_idx]))
        observed_test.extend(test_idx.tolist())
    assert sorted(observed_test) == list(range(len(dataset_ids)))


def test_cv_dispatch_and_validation_seed_are_deterministic():
    groups = np.repeat(np.arange(8), 2)
    X = np.zeros((len(groups), 1))
    y = np.arange(len(groups))
    dataset_ids = np.repeat(np.array(["a", "b", "c", "d"]), 4)

    dispatched = dispatch_cv_splits(
        "both",
        X=X,
        y=y,
        groups=groups,
        dataset_ids=dataset_ids,
        n_splits=4,
    )
    assert set(dispatched) == {"kfold", "lodo"}

    train_idx = np.arange(len(groups))
    first = split_train_validation_by_group(train_idx, groups, val_size=0.25, seed=7)
    second = split_train_validation_by_group(train_idx, groups, val_size=0.25, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))


def test_harmonization_protocol_boundaries_are_explicit():
    kfold = harmonization_policy("kfold")
    lodo = harmonization_policy("lodo")

    assert kfold.apply_to_test is True
    assert kfold.test_target_used_by_preprocessor is True
    assert lodo.apply_to_test is False
    assert lodo.test_target_used_by_preprocessor is False
    assert lodo.label == "source_only"
