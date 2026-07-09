import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from spd_connectome_benchmark.config import DEFAULT_COVARIANCE_EPS, PAPER_DATASETS
from spd_connectome_benchmark.connectomes import (
    estimate_connectome_matrices,
    vectorize_correlation_upper,
)
from spd_connectome_benchmark.models.spd import SPDNetRegressor
from spd_connectome_benchmark.benchmark_tools.runtime import (
    load_age_timeseries,
    load_pooled_age_dataset,
    save_metrics_csv,
    split_train_validation_by_group,
)
from spd_learn.modules import BiMap, LogEig, ReEig


def test_paper_dataset_list_has_only_reported_datasets():
    assert PAPER_DATASETS == ("cobre", "adnidod", "camcan", "abide", "oasis3", "adni")


def test_spd_learn_layers_are_available():
    assert BiMap is not None
    assert ReEig is not None
    assert LogEig is not None


def test_connectome_construction_returns_spd_correlations():
    rng = np.random.RandomState(0)
    timeseries = [rng.normal(size=(130, 6)), rng.normal(size=(125, 6))]

    matrices = estimate_connectome_matrices(timeseries, normalize=True, n_jobs=1)

    assert matrices.shape == (2, 6, 6)
    assert np.allclose(matrices, np.swapaxes(matrices, -1, -2))
    assert np.all(np.linalg.eigvalsh(matrices) > 0)
    assert np.allclose(
        np.diagonal(matrices, axis1=1, axis2=2),
        1.0 + DEFAULT_COVARIANCE_EPS,
    )


def test_grouped_validation_split_keeps_subjects_disjoint():
    train_idx = np.arange(20)
    groups = np.repeat(np.arange(10), 2)

    sub_train, sub_val = split_train_validation_by_group(train_idx, groups, val_size=0.2, seed=1)

    assert set(groups[sub_train]).isdisjoint(set(groups[sub_val]))
    assert len(np.unique(groups[sub_val])) == 2


def test_corrvec_uses_isometric_off_diagonal_upper_triangle():
    matrices = np.arange(2 * 4 * 4, dtype=float).reshape(2, 4, 4)

    features = vectorize_correlation_upper(matrices)

    assert features.shape == (2, 6)
    assert np.allclose(features[0], np.sqrt(2.0) * np.array([1, 2, 3, 6, 7, 11], dtype=float))


def test_spdnet_forward_shape_for_small_spd_batch():
    model = SPDNetRegressor(dims=(6, 6), out_dim=1)
    x = torch.eye(6).repeat(3, 1, 1)

    y = model(x)

    assert y.shape == (3, 1)


def test_pooled_loader_prefixes_subject_ids(tmp_path):
    atlas_dir = tmp_path / "atlas_schaefer_100"
    atlas_dir.mkdir()
    for dataset in ("cobre", "adni"):
        df = pd.DataFrame(
            {
                "SubjectID": ["001", "002"],
                "TimeSeries": [np.eye(6), np.eye(6) * 2],
                "Age": [20.0, 30.0],
            }
        )
        with open(atlas_dir / f"{dataset}_X_y.pkl", "wb") as f:
            pickle.dump(df, f)

    loaded = load_pooled_age_dataset(
        datasets=("cobre", "adni"),
        atlas_name="schaefer_100",
        task="Age",
        debug=None,
        rng_seed=0,
        data_root=tmp_path,
        verbose=False,
    )

    assert set(loaded.dataset_ids) == {"cobre", "adni"}
    assert all(str(subject_id).startswith(("cobre_", "adni_")) for subject_id in loaded.subject_ids)


def test_metric_csv_writes_reproducibility_metadata(tmp_path):
    csv_path = tmp_path / "metrics.csv"

    save_metrics_csv({"MAE": [1.0, 2.0]}, n_splits=2, out_csv=str(csv_path), elapsed=0.5)

    metadata_path = csv_path.with_suffix(".csv.metadata.json")
    payload = json.loads(metadata_path.read_text())
    assert payload["elapsed_seconds"] == 0.5
    assert "python" in payload
    assert "dependencies" in payload


def test_real_prepared_data_smoke_when_configured():
    data_root = os.environ.get("RSFMRI_SPD_TEST_DATA_ROOT")
    if not data_root:
        pytest.skip("Set RSFMRI_SPD_TEST_DATA_ROOT to run the real-data smoke test.")

    subject_ids, timeseries, y, y_type = load_age_timeseries(
        dataset="cobre",
        atlas_name="schaefer_100",
        task="Age",
        debug=3,
        rng=np.random.RandomState(0),
        data_root=data_root,
    )

    matrices = estimate_connectome_matrices(timeseries, normalize=True, n_jobs=1)
    assert y_type == "continuous"
    assert len(subject_ids) == len(y) == matrices.shape[0]
