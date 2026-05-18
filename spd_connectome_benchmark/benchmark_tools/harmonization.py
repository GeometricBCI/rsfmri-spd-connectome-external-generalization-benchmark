"""Fold-local tangent-space harmonization helpers.

The pooled benchmark and SPDNet ablation both use the same paper protocol:
fit the tangent reference and ComBat model on the current outer-training split,
then optionally apply the learned harmonization to the held-out split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from neuroHarmonize import harmonizationApply, harmonizationLearn
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin


class RIEHarmonizer(BaseEstimator, TransformerMixin):
    """Sklearn-style wrapper around neuroHarmonize for split-local features."""

    def __init__(
        self,
        feature_names,
        covariates_names,
        eb=True,
        smooth_terms=None,
        smooth_term_bounds=(None, None),
    ):
        self.feature_names = feature_names
        self.covariates_names = list(covariates_names)
        self.eb = eb
        self.smooth_terms = smooth_terms or []
        self.smooth_term_bounds = smooth_term_bounds
        self.model = None

    def fit(self, X, y=None):
        data = X[self.feature_names].to_numpy()
        covars = X[self.covariates_names]
        self.model, _ = harmonizationLearn(
            data,
            covars,
            eb=self.eb,
            smooth_terms=self.smooth_terms,
            smooth_term_bounds=self.smooth_term_bounds,
        )
        return self

    def transform(self, X, y=None):
        data = X[self.feature_names].to_numpy()
        covars = X[self.covariates_names]
        return harmonizationApply(data, covars, self.model)


def harmonize_tangent_features(
    Z_tr,
    Z_te,
    cov_train,
    cov_test,
    apply_harm_to_test: bool,
    covar_cols=("SITE", "age"),
):
    """Fit ComBat harmonization on training features and transform features.

    Paper Methods 2.6/2.7.1: empirical Bayes with SITE as batch and age as a
    smooth biological covariate in the range (0, 120), fitted only on the
    current outer-training split.
    """
    feat_cols = [f"cov_{i}" for i in range(Z_tr.shape[1])]
    df_train = pd.concat(
        [pd.DataFrame(cov_train, columns=covar_cols), pd.DataFrame(Z_tr, columns=feat_cols)],
        axis=1,
    )
    df_test = pd.concat(
        [pd.DataFrame(cov_test, columns=covar_cols), pd.DataFrame(Z_te, columns=feat_cols)],
        axis=1,
    )

    smooth_terms = ["age"] if "age" in covar_cols else []
    harmonizer = RIEHarmonizer(
        feature_names=feat_cols,
        covariates_names=covar_cols,
        eb=True,
        smooth_terms=smooth_terms,
        smooth_term_bounds=(0, 120),
    )
    harmonizer.fit(df_train)
    Z_tr_h = harmonizer.transform(df_train)
    Z_te_h = harmonizer.transform(df_test) if apply_harm_to_test else Z_te

    Z_tr_h = np.nan_to_num(Z_tr_h, nan=0.0, posinf=0.0, neginf=0.0)
    Z_te_h = np.nan_to_num(Z_te_h, nan=0.0, posinf=0.0, neginf=0.0)
    return Z_tr_h, Z_te_h


def load_or_harmonize_features(
    Z_tr,
    Z_te,
    cov_train,
    cov_test,
    apply_harm_to_test: bool,
    save_dir: str | Path,
    fold_tag: str,
    train_idx,
    test_idx,
    feature_kind: str,
    feature_metric: str,
):
    """Return cached or newly harmonized tangent/vector features for one fold."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{fold_tag}_harmonized_features.npz"

    if save_path.exists():
        try:
            with np.load(save_path, allow_pickle=False) as cache:
                required = {
                    "Z_tr_h",
                    "Z_te_h",
                    "train_idx",
                    "test_idx",
                    "apply_harm_to_test",
                    "feature_kind",
                    "feature_metric",
                    "cov_train",
                    "cov_test",
                }
                if required.issubset(cache.files):
                    cache_ok = (
                        cache["Z_tr_h"].shape == Z_tr.shape
                        and cache["Z_te_h"].shape == Z_te.shape
                        and bool(cache["apply_harm_to_test"].item()) == bool(apply_harm_to_test)
                        and str(cache["feature_kind"].item()) == str(feature_kind)
                        and str(cache["feature_metric"].item()) == str(feature_metric)
                        and np.array_equal(cache["train_idx"], train_idx)
                        and np.array_equal(cache["test_idx"], test_idx)
                        and np.array_equal(cache["cov_train"], cov_train.astype(str))
                        and np.array_equal(cache["cov_test"], cov_test.astype(str))
                    )
                    if cache_ok:
                        print("Loaded harmonized features:", save_path)
                        return np.array(cache["Z_tr_h"]), np.array(cache["Z_te_h"])
            print("Ignoring incompatible harmonized feature cache:", save_path)
        except Exception as exc:
            print(f"Ignoring unreadable harmonized feature cache {save_path}: {exc}")

    Z_tr_h, Z_te_h = harmonize_tangent_features(
        Z_tr,
        Z_te,
        cov_train,
        cov_test,
        apply_harm_to_test=apply_harm_to_test,
    )
    np.savez_compressed(
        save_path,
        Z_tr_h=Z_tr_h,
        Z_te_h=Z_te_h,
        train_idx=train_idx,
        test_idx=test_idx,
        apply_harm_to_test=bool(apply_harm_to_test),
        feature_kind=str(feature_kind),
        feature_metric=str(feature_metric),
        cov_train=cov_train.astype(str),
        cov_test=cov_test.astype(str),
    )
    print("Saved harmonized features:", save_path)
    return Z_tr_h, Z_te_h


def load_or_harmonize_spd_matrices(
    X,
    y,
    dataset_ids,
    train_idx,
    test_idx,
    apply_harm_to_test: bool,
    ts_metric: str,
    save_dir: str | Path,
    fold_tag: str,
):
    """Return cached or newly harmonized SPD matrices for one fold."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{fold_tag}_harmonized_spd.npz"

    if save_path.exists():
        try:
            with np.load(save_path, allow_pickle=False) as cache:
                required = {"X_tr", "X_te", "train_idx", "test_idx"}
                if required.issubset(cache.files):
                    X_tr_h = cache["X_tr"]
                    X_te_h = cache["X_te"]
                    cache_ok = (
                        X_tr_h.shape == X[train_idx].shape
                        and X_te_h.shape == X[test_idx].shape
                        and np.array_equal(cache["train_idx"], train_idx)
                        and np.array_equal(cache["test_idx"], test_idx)
                    )
                    if "apply_harm_to_test" in cache.files:
                        cache_ok = (
                            cache_ok
                            and bool(cache["apply_harm_to_test"].item()) == bool(apply_harm_to_test)
                        )
                    if "ts_metric" in cache.files:
                        cache_ok = cache_ok and str(cache["ts_metric"].item()) == str(ts_metric)
                    if cache_ok:
                        X_h = X.copy()
                        X_h[train_idx] = X_tr_h
                        X_h[test_idx] = X_te_h
                        print("Loaded harmonized SPD matrices:", save_path)
                        return X_h
            print("Ignoring incompatible harmonized SPD cache:", save_path)
        except Exception as exc:
            print(f"Ignoring unreadable harmonized SPD cache {save_path}: {exc}")

    X_tr = X[train_idx]
    X_te = X[test_idx]
    y_tr = y[train_idx]
    y_te = y[test_idx]

    ts = TangentSpace(metric=ts_metric, tsupdate=False)
    Z_tr = ts.fit_transform(X_tr)
    Z_te = ts.transform(X_te)

    cov_train = np.stack([dataset_ids[train_idx], y_tr], axis=1)
    cov_test = np.stack([dataset_ids[test_idx], y_te], axis=1)
    Z_tr_h, Z_te_h = harmonize_tangent_features(
        Z_tr,
        Z_te,
        cov_train,
        cov_test,
        apply_harm_to_test=apply_harm_to_test,
    )

    X_tr_h = ts.inverse_transform(Z_tr_h)
    X_te_h = ts.inverse_transform(Z_te_h) if apply_harm_to_test else X_te

    X_h = X.copy()
    X_h[train_idx] = X_tr_h
    X_h[test_idx] = X_te_h

    np.savez_compressed(
        save_path,
        X_tr=X_tr_h,
        X_te=X_te_h,
        train_idx=train_idx,
        test_idx=test_idx,
        reference=ts.reference_,
        apply_harm_to_test=bool(apply_harm_to_test),
        ts_metric=str(ts_metric),
    )
    print("Saved harmonized SPD matrices:", save_path)
    return X_h
