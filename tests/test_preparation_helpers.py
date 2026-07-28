import numpy as np
import pandas as pd
import pytest

import prepare_fmri_datasets


def test_global_signal_regression_rejects_all_missing_confounds(monkeypatch):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("fMRI images must not be loaded after invalid confounds")

    monkeypatch.setattr(prepare_fmri_datasets.nib, "load", forbidden_load)

    with pytest.raises(ValueError, match="all confound entries are missing"):
        prepare_fmri_datasets.global_signal_regression(
            ["first.nii.gz", "second.nii.gz"],
            [None, None],
        )


def test_global_signal_regression_rejects_mixed_confound_types(monkeypatch):
    def forbidden_load(*args, **kwargs):
        raise AssertionError("fMRI images must not be loaded after invalid confounds")

    monkeypatch.setattr(prepare_fmri_datasets.nib, "load", forbidden_load)

    with pytest.raises(TypeError, match="must have one type"):
        prepare_fmri_datasets.global_signal_regression(
            ["first.nii.gz", "second.nii.gz"],
            [np.zeros((5, 1)), pd.DataFrame({"motion": np.zeros(5)})],
        )


def test_global_signal_regression_rejects_row_count_mismatch(monkeypatch):
    class SyntheticImage:
        def get_fdata(self):
            return np.zeros((2, 2, 2, 5))

    monkeypatch.setattr(
        prepare_fmri_datasets.nib,
        "load",
        lambda path: SyntheticImage(),
    )

    with pytest.raises(ValueError, match="4 rows.*5 time points"):
        prepare_fmri_datasets.global_signal_regression(
            ["synthetic.nii.gz"],
            [np.zeros((4, 2))],
        )


@pytest.mark.parametrize(
    ("mask", "message"),
    [
        (np.array([], dtype=int), "is empty"),
        (np.array([0, 1, 4], dtype=int), "retains no time points"),
    ],
)
def test_postprocess_sample_mask_rejects_masks_with_no_retained_points(
    mask,
    message,
):
    with pytest.raises(ValueError, match=message):
        prepare_fmri_datasets.postprocess_sample_mask(
            [mask],
            ["synthetic.nii.gz"],
            start_idx=5,
        )


def test_postprocess_sample_mask_rejects_implicit_empty_result(monkeypatch):
    class ShortImage:
        def get_fdata(self):
            return np.zeros((2, 2, 2, 5))

    monkeypatch.setattr(
        prepare_fmri_datasets.nib,
        "load",
        lambda path: ShortImage(),
    )

    with pytest.raises(ValueError, match="retains no time points"):
        prepare_fmri_datasets.postprocess_sample_mask(
            None,
            ["synthetic.nii.gz"],
            start_idx=5,
        )


def test_postprocess_sample_mask_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same number of entries"):
        prepare_fmri_datasets.postprocess_sample_mask(
            [np.array([5]), np.array([5])],
            ["synthetic.nii.gz"],
            start_idx=5,
        )
