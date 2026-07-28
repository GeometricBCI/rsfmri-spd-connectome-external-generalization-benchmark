import numpy as np

from spd_connectome_benchmark.benchmark_tools import harmonization


class SpyHarmonizer:
    events = []

    def __init__(self, **kwargs):
        self.feature_names = kwargs["feature_names"]

    def fit(self, frame, y=None):
        self.events.append(("fit", len(frame)))
        return self

    def transform(self, frame, y=None):
        self.events.append(("transform", len(frame)))
        return frame[self.feature_names].to_numpy() + 10.0


def test_harmonizer_fits_train_only_and_lodo_leaves_test_unchanged(monkeypatch):
    monkeypatch.setattr(harmonization, "RIEHarmonizer", SpyHarmonizer)
    SpyHarmonizer.events = []
    train = np.arange(12, dtype=float).reshape(4, 3)
    test = np.arange(6, dtype=float).reshape(2, 3)
    train_covars = np.array([["a", 20], ["a", 30], ["b", 40], ["b", 50]])
    test_covars = np.array([["c", 60], ["c", 70]])

    train_h, test_h = harmonization.harmonize_tangent_features(
        train,
        test,
        train_covars,
        test_covars,
        apply_harm_to_test=False,
    )

    assert SpyHarmonizer.events == [("fit", 4), ("transform", 4)]
    assert np.array_equal(train_h, train + 10)
    assert np.array_equal(test_h, test)


def test_groupkfold_policy_transforms_test_after_train_fit(monkeypatch):
    monkeypatch.setattr(harmonization, "RIEHarmonizer", SpyHarmonizer)
    SpyHarmonizer.events = []
    train = np.ones((3, 2))
    test = np.ones((2, 2))
    train_covars = np.array([["a", 20], ["a", 30], ["b", 40]])
    test_covars = np.array([["b", 50], ["a", 60]])

    _, test_h = harmonization.harmonize_tangent_features(
        train,
        test,
        train_covars,
        test_covars,
        apply_harm_to_test=True,
    )

    assert SpyHarmonizer.events == [
        ("fit", 3),
        ("transform", 3),
        ("transform", 2),
    ]
    assert np.array_equal(test_h, test + 10)


def test_legacy_feature_cache_is_rejected_and_covariates_are_not_persisted(
    tmp_path,
    monkeypatch,
):
    save_path = tmp_path / "fold_harmonized_features.npz"
    train = np.ones((3, 2))
    test = np.ones((2, 2))
    train_idx = np.array([0, 1, 2])
    test_idx = np.array([3, 4])
    train_covars = np.array([["a", 20], ["a", 30], ["b", 40]])
    test_covars = np.array([["c", 50], ["c", 60]])
    np.savez_compressed(
        save_path,
        Z_tr_h=np.full_like(train, -1),
        Z_te_h=np.full_like(test, -1),
        train_idx=train_idx,
        test_idx=test_idx,
    )
    calls = []

    def recompute(*args, **kwargs):
        calls.append(True)
        return train + 1, test + 1

    monkeypatch.setattr(harmonization, "harmonize_tangent_features", recompute)
    train_h, test_h = harmonization.load_or_harmonize_features(
        train,
        test,
        train_covars,
        test_covars,
        apply_harm_to_test=False,
        save_dir=tmp_path,
        fold_tag="fold",
        train_idx=train_idx,
        test_idx=test_idx,
        feature_kind="tangent",
        feature_metric="riemann",
    )

    assert calls == [True]
    assert np.array_equal(train_h, train + 1)
    assert np.array_equal(test_h, test + 1)
    with np.load(save_path, allow_pickle=False) as cache:
        assert "cache_schema_version" in cache.files
        assert "input_signature" in cache.files
        assert "cov_train" not in cache.files
        assert "cov_test" not in cache.files
