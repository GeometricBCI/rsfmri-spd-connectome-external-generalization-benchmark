from types import SimpleNamespace

import pandas as pd
import pytest

import run_single_dataset_benchmark


@pytest.mark.parametrize(
    ("failed_datasets", "expected_status"),
    [
        (set(), 0),
        ({"adni"}, 1),
        ({"cobre", "adni"}, 1),
    ],
)
def test_single_dataset_main_returns_failure_after_writing_summary(
    tmp_path,
    monkeypatch,
    failed_datasets,
    expected_status,
):
    args = SimpleNamespace(
        datasets="cobre,adni",
        log_level="INFO",
        device="cpu",
        atlas="schaefer_100",
        results_folder_root=str(tmp_path),
    )
    monkeypatch.setattr(
        run_single_dataset_benchmark,
        "args_parser",
        lambda argv=None: args,
    )
    monkeypatch.setattr(
        run_single_dataset_benchmark,
        "configure_logging",
        lambda level: None,
    )
    monkeypatch.setattr(
        run_single_dataset_benchmark,
        "timestamp_tag",
        lambda: "stable",
    )
    monkeypatch.setattr(
        run_single_dataset_benchmark,
        "portable_result_reference",
        lambda value, output_root: f"portable:{value}",
    )

    def fake_run_dataset(args, dataset, atlas_name):
        if dataset in failed_datasets:
            raise RuntimeError(f"{dataset} failed")
        return {"Ridge": f"{dataset}/metrics.csv"}

    monkeypatch.setattr(
        run_single_dataset_benchmark,
        "run_dataset",
        fake_run_dataset,
    )

    status = run_single_dataset_benchmark.main([])

    summary_path = tmp_path / "[stable]SUMMARY_paths.csv"
    assert status == expected_status
    assert summary_path.is_file()
    summary = pd.read_csv(summary_path).set_index("Dataset")
    assert set(summary.index) == {"cobre", "adni"}
    for dataset in failed_datasets:
        assert summary.loc[dataset, "error"] == "RuntimeError"
    for dataset in {"cobre", "adni"} - failed_datasets:
        assert summary.loc[dataset, "Ridge"] == (
            f"portable:{dataset}/metrics.csv"
        )
